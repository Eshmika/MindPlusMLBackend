import numpy as np

# Advanced ML model integration
try:
    from sklearn.ensemble import RandomForestRegressor
    import joblib
except ImportError:
    RandomForestRegressor = None
    joblib = None

MODEL_PATH = "model/stress_rf_model.pkl"

FEATURE_NAMES = [
    "stress_today",
    "energy_level",
    "sleep_hours",
    "workload_intensity",
    "event_load",
    "emotional_load",
    "deadline_pressure",
]


def clamp(value, low, high):
    return max(low, min(high, value))


def safe_float(value, default):
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_log_schema(log):
    # Canonical schema for dynamic questionnaire support.
    stress_today = safe_float(log.get("stress_today"), 5.0)
    energy_level = safe_float(log.get("energy_level"), 5.0)
    sleep_hours = safe_float(log.get("sleep_hours"), 6.5)
    workload_intensity = safe_float(log.get("workload_intensity"), 5.0)

    emotional_load = safe_float(log.get("emotional_load"), stress_today)
    deadline_pressure = safe_float(log.get("deadline_pressure"), workload_intensity)

    return {
        "stress_today": clamp(stress_today, 0, 10),
        "energy_level": clamp(energy_level, 0, 10),
        "sleep_hours": clamp(sleep_hours, 0, 12),
        "workload_intensity": clamp(workload_intensity, 0, 10),
        "emotional_load": clamp(emotional_load, 0, 10),
        "deadline_pressure": clamp(deadline_pressure, 0, 10),
    }


def extract_features(log, event_load=0):
    normalized = normalize_log_schema(log)
    return [
        normalized["stress_today"],
        normalized["energy_level"],
        normalized["sleep_hours"],
        normalized["workload_intensity"],
        safe_float(event_load, 0),
        normalized["emotional_load"],
        normalized["deadline_pressure"],
    ]


def event_pressure(events, sensitivity=1.0):
    total = 0
    for e in events:
        importance = clamp(safe_float(e.get("importance"), 1), 1, 10)
        days = clamp(safe_float(e.get("days_ahead"), 1), 1, 365)
        pressure = importance * np.exp(-days / 3)
        total += pressure
    return total * 0.4 * sensitivity


def train_stress_model(logs, events):
    if RandomForestRegressor is None:
        return None

    if not logs or len(logs) < 12:
        return None

    X = []
    y = []
    shared_event_load = event_pressure(events, sensitivity=1.0)

    normalized_logs = [normalize_log_schema(log) for log in logs]

    for i in range(1, len(normalized_logs)):
        prev = normalized_logs[i - 1]
        curr = normalized_logs[i]
        X.append(extract_features(prev, shared_event_load))
        y.append(curr.get("stress_today", 5))

    if len(X) < 10:
        return None

    model = RandomForestRegressor(n_estimators=120, random_state=42)
    model.fit(X, y)

    if joblib:
        joblib.dump(model, MODEL_PATH)

    return model


def predict_stress_ml(model, log, event_load=0):
    X = [extract_features(log, event_load)]
    pred = model.predict(X)[0]
    return clamp(round(float(pred), 2), 0, 10)


def calculate_stress_momentum(logs):
    if len(logs) < 2:
        return 0

    diffs = []
    for i in range(1, len(logs)):
        prev = logs[i - 1].get("stress_today")
        curr = logs[i].get("stress_today")
        if prev is not None and curr is not None:
            diffs.append(curr - prev)
    return float(np.mean(diffs)) if diffs else 0


def emotional_variability(logs):
    values = [l.get("stress_today") for l in logs if l.get("stress_today") is not None]
    if len(values) < 2:
        return 0
    return float(np.std(values))


def sleep_recovery(logs):
    sleep_values = [l.get("sleep_hours") for l in logs if l.get("sleep_hours") is not None]
    if not sleep_values:
        return 0
    avg_sleep = float(np.mean(sleep_values))
    recovery = (avg_sleep - 6) * 0.5
    return clamp(recovery, -1, 2)


def energy_buffer(logs):
    energy = [l.get("energy_level") for l in logs if l.get("energy_level") is not None]
    if not energy:
        return 0
    avg_energy = float(np.mean(energy))
    return (avg_energy - 5) * -0.2


def baseline_stress(baseline):
    if not baseline:
        return 4.5

    # Accept either 0-10 or DASS-like 0-42 baseline scales.
    score = baseline.get("stress_score")
    if score is None:
        score = baseline.get("stress")
    if score is None:
        score = baseline.get("stressLevel")

    value = safe_float(score, 4.5)
    if value > 10:
        value = (value / 42.0) * 10.0
    return clamp(value, 0, 10)


def get_feature_importance(model):
    if not model or not hasattr(model, "feature_importances_"):
        uniform = 1 / len(FEATURE_NAMES)
        return {name: round(uniform, 4) for name in FEATURE_NAMES}

    values = model.feature_importances_
    return {
        name: round(float(values[idx]), 4)
        for idx, name in enumerate(FEATURE_NAMES)
    }


def assign_fingerprint(logs, event_load, feature_importance):
    if not logs:
        return {
            "label": "Workload-overwhelmed",
            "scores": {
                "Deadline-sensitive": 0.4,
                "Sleep-reactive": 0.3,
                "Workload-overwhelmed": 0.6,
                "Emotionally-driven stress": 0.4,
            },
            "dominant_drivers": ["workload_intensity", "event_load"],
        }

    mean_log = {
        "stress_today": float(np.mean([l["stress_today"] for l in logs])),
        "energy_level": float(np.mean([l["energy_level"] for l in logs])),
        "sleep_hours": float(np.mean([l["sleep_hours"] for l in logs])),
        "workload_intensity": float(np.mean([l["workload_intensity"] for l in logs])),
        "emotional_load": float(np.mean([l["emotional_load"] for l in logs])),
        "deadline_pressure": float(np.mean([l["deadline_pressure"] for l in logs])),
    }

    sleep_deficit = clamp((8 - mean_log["sleep_hours"]) / 4, 0, 1)
    low_energy = clamp((6 - mean_log["energy_level"]) / 6, 0, 1)
    workload_high = clamp(mean_log["workload_intensity"] / 10, 0, 1)
    stress_level = clamp(mean_log["stress_today"] / 10, 0, 1)
    emotion_high = clamp(mean_log["emotional_load"] / 10, 0, 1)
    deadline_high = clamp(mean_log["deadline_pressure"] / 10, 0, 1)
    event_high = clamp(event_load / 4, 0, 1)

    fi = feature_importance
    scores = {
        "Deadline-sensitive": (
            0.45 * deadline_high
            + 0.35 * event_high
            + 0.20 * workload_high
            + 0.30 * fi.get("deadline_pressure", 0)
            + 0.20 * fi.get("event_load", 0)
        ),
        "Sleep-reactive": (
            0.55 * sleep_deficit
            + 0.25 * low_energy
            + 0.20 * stress_level
            + 0.35 * fi.get("sleep_hours", 0)
            + 0.15 * fi.get("energy_level", 0)
        ),
        "Workload-overwhelmed": (
            0.50 * workload_high
            + 0.25 * event_high
            + 0.25 * stress_level
            + 0.40 * fi.get("workload_intensity", 0)
        ),
        "Emotionally-driven stress": (
            0.50 * emotion_high
            + 0.30 * stress_level
            + 0.20 * low_energy
            + 0.40 * fi.get("emotional_load", 0)
            + 0.20 * fi.get("stress_today", 0)
        ),
    }

    normalized_scores = {k: round(clamp(v, 0, 1.5), 3) for k, v in scores.items()}
    label = max(normalized_scores, key=normalized_scores.get)

    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
    dominant_drivers = [name for name, _ in top_features]

    return {
        "label": label,
        "scores": normalized_scores,
        "dominant_drivers": dominant_drivers,
    }


def predict_future_stress(payload):
    logs_raw = payload.get("recent_logs") or []
    events = payload.get("upcoming_events") or []
    baseline = payload.get("baseline")

    logs = [normalize_log_schema(log) for log in logs_raw if isinstance(log, dict)]
    if not logs:
        logs = [normalize_log_schema({})]

    base = baseline_stress(baseline)
    momentum = calculate_stress_momentum(logs)
    variability = emotional_variability(logs)
    recovery = sleep_recovery(logs)
    energy = energy_buffer(logs)
    event_load = event_pressure(events)

    model = None
    if RandomForestRegressor is not None and len(logs) > 10:
        model = train_stress_model(logs, events)

    predictions = []
    current = base
    for day in range(1, 6):
        if model:
            log_idx = min(day, len(logs) - 1)
            log = logs[log_idx]
            pred = predict_stress_ml(model, log, event_load * (1 / day))
            predictions.append(pred)
        else:
            drift = (
                momentum * 0.6
                + variability * 0.2
                + event_load * (1 / day)
                - recovery
                + energy
            )
            drift = clamp(drift, -2, 2)
            noise = 0.1 * np.sin(day)
            current = current * 0.92 + drift + noise
            current = clamp(current, 0, 10)
            predictions.append(round(float(current), 2))

    confidence = clamp(1 - variability / 10, 0.5, 1)
    feature_importance = get_feature_importance(model)
    fingerprint = assign_fingerprint(logs, event_load, feature_importance)

    return {
        "schema_version": "v2",
        "future_5_days": predictions,
        "confidence": round(float(confidence), 2),
        "model_used": bool(model),
        "feature_importance": feature_importance,
        "fingerprint": fingerprint,
    }
