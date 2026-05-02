# Import required libraries for ML model, data validation, and file operations
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from transformers import pipeline
import os
import re
import json
from pathlib import Path


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    emotion: str
    confidence: float
    model: Optional[str] = None
    keywords: List[str]


class CopingStrategyRequest(BaseModel):
    emotion: str
    confidence: float


class CopingStrategyResponse(BaseModel):
    emotion: str
    confidence: float
    severity: str
    strategy: Optional[str]
    coping_strategy: Optional[str] = None
    technique: Optional[str] = None
    duration_seconds: Optional[int] = None


# Common words to filter out when extracting keywords from text
STOPWORDS = {
    "the","a","an","and","or","but","if","then","so","to","for","of","on","in","at","is","am","are","was","were","be","been","being","i","you","he","she","it","they","them","we","me","my","your","our","with","this","that","those","these","about","just","very","really","feel","feeling"
}

# Load the emotion classification model from Hugging Face
MODEL_NAME = os.getenv("EMOTION_MODEL_NAME", "j-hartmann/emotion-english-distilroberta-base")

classifier = pipeline("text-classification", model=MODEL_NAME)

# Path to the coping strategies JSON file
COPING_STRATEGY_PATH = Path(__file__).parent.parent / "CopingStrategy.json"


def _normalize_strategy_entry(entry: Any) -> Optional[Dict[str, Any]]:
    # Backward compatible: allow both string and object strategy values.
    if isinstance(entry, str):
        return {
            "coping_strategy": entry,
            "technique": None,
            "duration_seconds": None,
        }

    if not isinstance(entry, dict):
        return None

    duration = entry.get("duration_seconds")
    if duration is not None:
        try:
            duration = int(duration)
        except (TypeError, ValueError):
            duration = None

    return {
        "coping_strategy": entry.get("coping_strategy"),
        "technique": entry.get("technique"),
        "duration_seconds": duration,
    }


def load_coping_strategies(path: Path) -> Dict[str, Dict[str, Dict[str, Any]]]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
            normalized: Dict[str, Dict[str, Dict[str, Any]]] = {}
            for emotion, strategies in payload.items():
                if not isinstance(strategies, dict):
                    continue
                per_severity: Dict[str, Dict[str, Any]] = {}
                for severity, strategy_value in strategies.items():
                    entry = _normalize_strategy_entry(strategy_value)
                    if entry is not None:
                        per_severity[str(severity).lower()] = entry
                if per_severity:
                    normalized[str(emotion).lower()] = per_severity
            return normalized
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid coping strategy JSON: {exc}")


COPING_STRATEGIES = load_coping_strategies(COPING_STRATEGY_PATH)


# Health check endpoint to verify service is running
async def health():
    return {"status": "ok", "service": "emotion", "model": MODEL_NAME}


# Extract important keywords from text by filtering out stopwords
def extract_keywords(text: str, max_keywords: int = 5) -> List[str]:

    tokens = re.findall(r"[A-Za-z']+", text.lower())
    filtered = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    
    seen = set()
    ordered_unique = []
    for t in filtered:
        if t not in seen:
            seen.add(t)
            ordered_unique.append(t)
        if len(ordered_unique) >= max_keywords:
            break
    return ordered_unique


# Determine severity level based on confidence score
def pick_severity(confidence: float) -> str:
    if confidence >= 0.75:
        return "high"
    if confidence >= 0.4:
        return "medium"
    return "low"


# Get appropriate coping strategy based on emotion and confidence level
def get_coping_strategy(emotion: str, confidence: float) -> Optional[Dict[str, Any]]:
    severity = pick_severity(confidence)

    strategies = COPING_STRATEGIES.get(emotion.lower()) or COPING_STRATEGIES.get("neutral")
    if not strategies:
        return None
    return strategies.get(severity)


# Predict emotion from input text
async def predict(payload: PredictRequest) -> PredictResponse:
    text = payload.text.strip()
    if not text:
        return PredictResponse(emotion="neutral", confidence=0.0, model=MODEL_NAME, keywords=[])
    raw = classifier(text)   
    first = raw[0]    
    if isinstance(first, list): 
        best = max(first, key=lambda x: x.get("score", 0.0))
    else:
        best = first
    emotion = best["label"].lower()
    confidence = float(best["score"])
    keywords = extract_keywords(text)
    return PredictResponse(emotion=emotion, confidence=confidence, model=MODEL_NAME, keywords=keywords)


# Get coping strategy recommendation for a detected emotion
async def coping_strategy(payload: CopingStrategyRequest):    
    emotion = payload.emotion.strip().lower() or "neutral"
    confidence = max(0.0, min(1.0, payload.confidence))
    severity = pick_severity(confidence)
    strategy_entry = get_coping_strategy(emotion, confidence) or {}
    strategy_text = strategy_entry.get("coping_strategy")
    return CopingStrategyResponse(
        emotion=emotion,
        confidence=confidence,
        severity=severity,
        strategy=strategy_text,
        coping_strategy=strategy_text,
        technique=strategy_entry.get("technique"),
        duration_seconds=strategy_entry.get("duration_seconds"),
    )
