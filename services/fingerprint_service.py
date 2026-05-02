import math
from datetime import datetime
import statistics

def calculate_weeks_since(timestamp_str):
    onboarding_date = datetime.fromisoformat(timestamp_str.replace("Z", ""))
    now = datetime.utcnow()
    delta = now - onboarding_date
    return delta.days / 7


def compute_baseline_weight(weeks):
    return math.exp(-0.1 * weeks)


def normalize(value):
    return value / 42


def compute_behavior_metrics(logs):

    stress_values = [log["stress_today"] for log in logs]
    sleep_values = [log["sleep_hours"] for log in logs]

    rolling_stress = sum(stress_values) / len(stress_values)
    sleep_avg = sum(sleep_values) / len(sleep_values)

    volatility = statistics.pstdev(stress_values) if len(stress_values) > 1 else 0

    return rolling_stress, sleep_avg, volatility


def determine_identity(adaptive_vec, volatility, sleep_avg):

    stress = adaptive_vec["stress"]
    anxiety = adaptive_vec["anxiety"]
    depression = adaptive_vec["depression"]

    if sleep_avg < 6:
        return "Sleep Vulnerable"

    if anxiety > stress and volatility > 1.5:
        return "Emotionally Reactive"

    if stress > anxiety and stress > depression:
        return "Academically Sensitive"

    if volatility < 1 and stress > 0.4:
        return "Quiet Accumulator"

    if stress >= 0.5:
        return "Balanced but Overloaded"

    return "Resilient Stabilizer"


def evolve_fingerprint(baseline, logs):

    weeks = calculate_weeks_since(baseline["timestamp"])
    weight = compute_baseline_weight(weeks)

    rolling_stress, sleep_avg, volatility = compute_behavior_metrics(logs)

    base_vec = {
        "stress": normalize(baseline["stress"]),
        "anxiety": normalize(baseline["anxiety"]),
        "depression": normalize(baseline["depression"])
    }

    beh_vec = {
        "stress": normalize(rolling_stress),
        "anxiety": normalize(baseline["anxiety"]),
        "depression": normalize(baseline["depression"])
    }

    adaptive_vec = {
        key: weight * base_vec[key] + (1 - weight) * beh_vec[key]
        for key in base_vec
    }

    identity = determine_identity(adaptive_vec, volatility, sleep_avg)

    return {
        "identity": identity,
        "baselineWeight": round(weight, 3),
        "adaptiveVector": adaptive_vec,
        "volatilityIndex": round(volatility, 2),
        "sleepAverage": round(sleep_avg, 2)
    }
