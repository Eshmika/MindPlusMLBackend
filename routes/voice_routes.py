from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import joblib
import re
from datetime import datetime
import uuid
import numpy as np

router = APIRouter(prefix="/voice", tags=["Voice Stress"])

# --------------------------------------------------
# Load ML assets
# --------------------------------------------------
try:
    model = joblib.load("models/stress_type_model.pkl")
    label_encoder = joblib.load("models/stress_label_encoder.pkl")
    print("✅ Multi-stress model loaded successfully")
    print(f"📋 Model type: {type(model)}")
    print(f"📋 Classes: {label_encoder.classes_}")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    model = None
    label_encoder = None

# --------------------------------------------------
# Schemas
# --------------------------------------------------
class StressRequest(BaseModel):
    user_id: str
    text: str
    audio_url: str | None = None

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text

def calculate_stress_level(score: float) -> str:
    """Convert score to level: Low, Moderate, High"""
    if score < 0.33:
        return "Low"
    elif score < 0.66:
        return "Moderate"
    else:
        return "High"

# reusable keyword lists used by several helpers
ACADEMIC_KEYWORDS = ['exam', 'test', 'study', 'homework', 'grade', 'school', 'college',
                     'university', 'assignment', 'academic', 'class', 'course', 'learning']
FINANCIAL_KEYWORDS = ['money', 'debt', 'bill', 'pay', 'payment', 'payments', 'cost',
                      'expensive', 'afford', 'budget', 'financial', 'financially',
                      'income', 'rent', 'loan', 'credit', 'broke', 'poor']
SOCIAL_KEYWORDS = ['lonely', 'alone', 'friend', 'friends', 'relationship',
                   'relationships', 'social', 'isolated', 'isolation', 'people',
                   'family', 'talk', 'connect', 'rejected']
EMOTIONAL_KEYWORDS = ['sad', 'anxious', 'anxiety', 'worried', 'worry',
                      'depressed', 'depression', 'overwhelm', 'overwhelmed',
                      'cry', 'crying', 'feel', 'feeling', 'emotion', 'emotional',
                      'upset', 'hurt', 'pain', 'stress', 'stressed']


def count_keyword_matches(text: str) -> dict:
    """Return raw counts for each stress category based on keyword lists."""
    t = text.lower()
    return {
        "Academic": sum(1 for w in ACADEMIC_KEYWORDS if w in t),
        "Financial": sum(1 for w in FINANCIAL_KEYWORDS if w in t),
        "Social": sum(1 for w in SOCIAL_KEYWORDS if w in t),
        "Emotional": sum(1 for w in EMOTIONAL_KEYWORDS if w in t),
    }


def scores_from_counts(counts: dict, predicted: str | None = None) -> dict:
    """Convert raw keyword counts to 0-1 scores.

    If `predicted` is provided, guarantee that category has at least
    a minimum score and zero-out others (model-guided scoring).
    """
    total = sum(counts.values())
    if total == 0:
        # fallback moderate values
        return {"Academic": 0.4, "Financial": 0.3, "Social": 0.3, "Emotional": 0.5}

    # normalize by max count so that highest category becomes 1.0
    max_count = max(counts.values())
    scores = {k: counts[k] / max_count if max_count > 0 else 0.0 for k in counts}

    if predicted and predicted in scores:
        # enforce model's choice
        for k in scores:
            scores[k] = 0.0
        scores[predicted] = 1.0
    return scores


def get_multi_stress_scores(text: str):
    """
    Legacy keyword-based scoring used when the model cannot be applied.
    See `count_keyword_matches` and `scores_from_counts` for the newer
    model-guided approach.
    """
    text_lower = text.lower()
    
    print(f"🔍 Analyzing text: '{text_lower}'")
    
    # Keyword-based scoring (0-1 scale)
    scores = {
        "Academic": 0.0,
        "Financial": 0.0,
        "Social": 0.0,
        "Emotional": 0.0
    }
    
    # Academic keywords (trimmed to reduce overlap with other domains)
    academic_count = sum(1 for word in ACADEMIC_KEYWORDS if word in text_lower)
    scores["Academic"] = min(1.0, academic_count * 0.25)
    print(f"📚 Academic keywords found: {academic_count}, score: {scores['Academic']}")
    
    # Financial keywords
    financial_count = sum(1 for word in FINANCIAL_KEYWORDS if word in text_lower)
    scores["Financial"] = min(1.0, financial_count * 0.25)
    print(f"💰 Financial keywords found: {financial_count}, score: {scores['Financial']}")
    
    # Social keywords
    social_count = sum(1 for word in SOCIAL_KEYWORDS if word in text_lower)
    scores["Social"] = min(1.0, social_count * 0.25)
    print(f"👥 Social keywords found: {social_count}, score: {scores['Social']}")
    
    # Emotional keywords
    emotional_count = sum(1 for word in EMOTIONAL_KEYWORDS if word in text_lower)
    scores["Emotional"] = min(1.0, emotional_count * 0.25)
    print(f"💭 Emotional keywords found: {emotional_count}, score: {scores['Emotional']}")
    
    # If no keywords found at all, set moderate stress across board
    total_score = sum(scores.values())
    if total_score == 0:
        print("⚠️ No keywords found, setting default moderate scores")
        scores = {
            "Academic": 0.4,
            "Financial": 0.3,
            "Social": 0.3,
            "Emotional": 0.5
        }
    else:
        # Boost the highest scoring category
        max_key = max(scores, key=scores.get)
        if scores[max_key] > 0:
            scores[max_key] = min(1.0, scores[max_key] + 0.15)
            print(f"✨ Boosted {max_key} to {scores[max_key]}")

        # enforce exclusivity: zero out the others so one dominates
        dominant = max_key
        for key in scores:
            if key != dominant:
                scores[key] = 0.0
        print(f"🔒 Enforced dominance: {dominant}, zeroed others")
    
    # Ensure minimum variance (if all are same, add some variation)
    unique_values = len(set(scores.values()))
    if unique_values == 1 and scores["Academic"] > 0:
        scores["Emotional"] = min(1.0, scores["Emotional"] + 0.1)
        scores["Social"] = max(0.0, scores["Social"] - 0.1)
    
    print(f"📊 Final scores: {scores}")
    return scores

# --------------------------------------------------
# Routes
# --------------------------------------------------
@router.post("/analyze-stress")
def analyze_stress(data: StressRequest):
    """
    Analyze ALL stress types simultaneously with individual scores.
    
    Returns raw 0-1 scores for each stress category based on model prediction
    and keyword matching. No high/medium/low categorization is applied to avoid
    variability issues - the raw scores reflect the relative strength of each
    stress type in the text. Keyword counts are included for transparency.
    """
    
    if not model or not label_encoder:
        raise HTTPException(status_code=500, detail="ML model not loaded")

    if not data.text or len(data.text.strip()) < 10:
        raise HTTPException(
            status_code=400,
            detail="Text too short for stress analysis"
        )

    try:
        cleaned = clean_text(data.text)
        
        print(f"=" * 60)
        print(f"🎤 ANALYZING TEXT")
        print(f"=" * 60)
        print(f"Original: {data.text}")
        print(f"Cleaned: {cleaned}")
        print(f"-" * 60)

        # Use model to determine dominant category, then score via keywords
        stress_scores = None
        if model and label_encoder:
            try:
                pred = model.predict([cleaned])[0]
                predicted_type = label_encoder.inverse_transform([pred])[0]
                print(f"🎯 Model predicted: {predicted_type}")

                counts = count_keyword_matches(data.text)
                print(f"🔢 Keyword counts: {counts}")

                stress_scores = scores_from_counts(counts, predicted=predicted_type)
                print(f"📊 Scores from counts (model-guided): {stress_scores}")
            except Exception as e:
                print(f"⚠️ Model prediction failed: {e}")
                stress_scores = None

        # if we didn't obtain scores via the new method, fall back
        if stress_scores is None:
            if model and hasattr(model, "predict_proba"):
                try:
                    proba = model.predict_proba([cleaned])[0]
                    stress_scores = {label_encoder.classes_[i]: round(float(p), 3)
                                     for i, p in enumerate(proba)}
                    print(f"✅ Using model probabilities")
                    print(f"Probabilities: {stress_scores}")
                except Exception as e:
                    print(f"⚠️ predict_proba failed: {e}")
                    stress_scores = get_multi_stress_scores(data.text)
            else:
                print(f"⚠️ Using legacy keyword fallback")
                stress_scores = get_multi_stress_scores(data.text)

        # Ensure all 4 types exist
        for stress_type in ["Academic", "Financial", "Social", "Emotional"]:
            if stress_type not in stress_scores:
                stress_scores[stress_type] = 0.0

        print(f"-" * 60)
        print(f"📊 FINAL STRESS SCORES:")
        for stype, score in stress_scores.items():
            print(f"  {stype}: {score:.3f} ({score*100:.1f}%)")
        print(f"-" * 60)

        # Find dominant stress type
        dominant_type = max(stress_scores, key=stress_scores.get)
        dominant_score = stress_scores[dominant_type]
        
        # Calculate total stress score
        total_stress = sum(stress_scores.values())
        
        # Remove level categorization - just provide raw scores
        # stress_levels = {}
        # for stress_type, score in stress_scores.items():
        #     stress_levels[stress_type] = calculate_stress_level(score)

        # Add keyword counts for transparency/proof
        keyword_counts = count_keyword_matches(data.text)

        result = {
            "success": True,
            "analysis_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": data.user_id,
            "text": data.text,
            "audio_url": data.audio_url,
            
            # Multi-stress analysis with raw scores only
            "stress_scores": stress_scores,
            # "stress_levels": stress_levels,  # removed
            "keyword_counts": keyword_counts,  # added for evidence
            "dominant_type": dominant_type,
            "dominant_score": dominant_score,
            "total_stress_score": round(total_stress, 3),
            
            # Overall assessment (no level)
            "overall_score": round(total_stress / 4, 3),
            "confidence": round(dominant_score, 3)
        }
        
        print(f"✅ Analysis complete!")
        print(f"Dominant: {dominant_type} ({dominant_score:.3f})")
        print(f"Total: {total_stress:.3f}")
        print(f"Overall: {result['overall_score']:.3f}")
        print(f"=" * 60)
        
        return result

    except Exception as e:
        print(f"=" * 60)
        print(f"❌ ANALYSIS ERROR: {e}")
        print(f"=" * 60)
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Stress analysis failed: {str(e)}")


@router.get("/stress-history/{user_id}")
def stress_history(user_id: str):
    return {
        "success": True,
        "user_id": user_id,
        "analyses": [],
        "message": "History feature (local storage)"
    }


@router.get("/health")
def voice_health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "label_encoder_loaded": label_encoder is not None,
        "has_predict_proba": hasattr(model, "predict_proba") if model else False
    }


@router.post("/test-model")
def test_model(data: StressRequest):
    """Test endpoint to debug model behavior"""
    
    if not model or not label_encoder:
        return {"error": "Models not loaded"}
    
    try:
        cleaned = clean_text(data.text)
        
        result = {
            "original_text": data.text,
            "cleaned_text": cleaned,
            "model_type": str(type(model)),
            "has_predict_proba": hasattr(model, "predict_proba"),
            "classes": list(label_encoder.classes_)
        }
        
        # Try prediction
        try:
            pred = model.predict([cleaned])[0]
            predicted_type = label_encoder.inverse_transform([pred])[0]
            result["prediction"] = predicted_type
        except Exception as e:
            result["prediction_error"] = str(e)
        
        # Try probabilities
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba([cleaned])[0]
                result["probabilities"] = {
                    label: float(prob) 
                    for label, prob in zip(label_encoder.classes_, proba)
                }
            except Exception as e:
                result["proba_error"] = str(e)
        
        # Keyword counts & derived scores
        result["keyword_counts"] = count_keyword_matches(data.text)
        result["keyword_scores"] = scores_from_counts(result["keyword_counts"],
                                                      predicted=result.get("prediction"))
        
        return result
        
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }