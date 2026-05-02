import csv
from pathlib import Path


def _load_sklearn_dependencies():
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            roc_auc_score,
            confusion_matrix,
        )
        from sklearn.model_selection import train_test_split
        return {
            "ok": True,
            "RandomForestClassifier": RandomForestClassifier,
            "accuracy_score": accuracy_score,
            "precision_score": precision_score,
            "recall_score": recall_score,
            "f1_score": f1_score,
            "roc_auc_score": roc_auc_score,
            "confusion_matrix": confusion_matrix,
            "train_test_split": train_test_split,
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _stress_score_from_row(row):
    # DASS-21 stress items (1-indexed): 1,6,8,11,12,14,18
    stress_items = [1, 6, 8, 11, 12, 14, 18]
    raw_sum = 0
    for idx in stress_items:
        raw_sum += int(float(row[f"q{idx}"]))
    return raw_sum * 2


def _stress_severity_label(stress_score):
    if stress_score <= 14:
        return "normal"
    if stress_score <= 18:
        return "mild"
    if stress_score <= 25:
        return "moderate"
    if stress_score <= 33:
        return "severe"
    return "extremely_severe"


def _load_dass21_dataset(csv_path):
    if not csv_path.exists():
        return [], []

    rows = []
    labels = []

    def normalize_key(key):
        # Handles BOM and inconsistent header casing/spacing.
        return (key or "").replace("\ufeff", "").strip().lower()

    with csv_path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            normalized_row = {normalize_key(k): v for k, v in row.items()}

            try:
                features = [int(float(normalized_row[f"q{i}"])) for i in range(1, 22)]
                score = _stress_score_from_row(normalized_row)
            except (KeyError, TypeError, ValueError):
                # Skip malformed rows so endpoint never crashes due to one bad record.
                continue

            rows.append(features)
            labels.append(_stress_severity_label(score))

    return rows, labels


def get_fingerprint_model_metrics():
    deps = _load_sklearn_dependencies()
    if not deps["ok"]:
        return {
            "status": "error",
            "message": "scikit-learn is not available in backend environment",
            "details": deps["error"],
        }

    csv_path = Path(__file__).resolve().parents[1] / "model" / "dass21_dataset.csv"
    X, y_multiclass = _load_dass21_dataset(csv_path)

    if len(X) < 50:
        return {
            "status": "error",
            "message": "Not enough rows in dataset for reliable metrics",
            "rows": len(X),
        }

    # Binary task: moderate+ stress risk vs low risk
    high_risk_classes = {"moderate", "severe", "extremely_severe"}
    y_binary = [1 if label in high_risk_classes else 0 for label in y_multiclass]

    def class_counts(values):
        counts = {}
        for v in values:
            counts[v] = counts.get(v, 0) + 1
        return counts

    binary_counts = class_counts(y_binary)
    multiclass_counts = class_counts(y_multiclass)

    binary_stratify = y_binary if len(binary_counts) > 1 and min(binary_counts.values()) >= 2 else None
    multiclass_stratify = y_multiclass if len(multiclass_counts) > 1 and min(multiclass_counts.values()) >= 2 else None

    train_test_split = deps["train_test_split"]
    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
        X, y_binary, test_size=0.2, random_state=42, stratify=binary_stratify
    )

    RandomForestClassifier = deps["RandomForestClassifier"]
    rf_binary = RandomForestClassifier(n_estimators=250, random_state=42)
    rf_binary.fit(X_train_b, y_train_b)

    y_pred_b = rf_binary.predict(X_test_b)
    y_prob_b = rf_binary.predict_proba(X_test_b)[:, 1]

    precision_score = deps["precision_score"]
    recall_score = deps["recall_score"]
    f1_score = deps["f1_score"]
    accuracy_score = deps["accuracy_score"]
    roc_auc_score = deps["roc_auc_score"]
    confusion_matrix = deps["confusion_matrix"]

    binary_metrics = {
        "accuracy": round(float(accuracy_score(y_test_b, y_pred_b)), 4),
        "precision": round(float(precision_score(y_test_b, y_pred_b, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test_b, y_pred_b, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test_b, y_pred_b, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test_b, y_prob_b)), 4),
        "confusion_matrix": confusion_matrix(y_test_b, y_pred_b).tolist(),
        "positive_class": "moderate_or_higher_stress",
    }

    # Multiclass task: full stress severity
    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
        X, y_multiclass, test_size=0.2, random_state=42, stratify=multiclass_stratify
    )

    rf_multi = RandomForestClassifier(n_estimators=300, random_state=42)
    rf_multi.fit(X_train_m, y_train_m)

    y_pred_m = rf_multi.predict(X_test_m)
    y_prob_m = rf_multi.predict_proba(X_test_m)
    class_order = list(rf_multi.classes_)

    multiclass_metrics = {
        "precision_macro": round(float(precision_score(y_test_m, y_pred_m, average="macro", zero_division=0)), 4),
        "recall_macro": round(float(recall_score(y_test_m, y_pred_m, average="macro", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y_test_m, y_pred_m, average="macro", zero_division=0)), 4),
        "classes": class_order,
    }

    try:
        multiclass_metrics["roc_auc_ovr_macro"] = round(
            float(roc_auc_score(y_test_m, y_prob_m, multi_class="ovr", average="macro")), 4
        )
    except ValueError:
        multiclass_metrics["roc_auc_ovr_macro"] = None
        multiclass_metrics["roc_auc_note"] = "Unavailable due to insufficient class support in test split"

    feature_importance = {
        f"q{i+1}": round(float(v), 4)
        for i, v in enumerate(rf_multi.feature_importances_)
    }
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]

    return {
        "status": "success",
        "dataset": {
            "path": str(csv_path),
            "rows": len(X),
            "feature_count": 21,
            "class_distribution": multiclass_counts,
            "split_strategy": {
                "binary_stratified": binary_stratify is not None,
                "multiclass_stratified": multiclass_stratify is not None,
            },
        },
        "binary_high_risk_metrics": binary_metrics,
        "multiclass_severity_metrics": multiclass_metrics,
        "top_question_importance": top_features,
        "notes": [
            "Binary target: moderate/severe/extremely_severe vs normal/mild",
            "Scores are computed on held-out 20% test split with random_state=42",
            "Use this endpoint output directly in panel slides for Precision/Recall/F1/ROC-AUC",
        ],
    }
