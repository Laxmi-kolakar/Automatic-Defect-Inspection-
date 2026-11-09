def make_decision(predicted_class=None, confidence=None, model_available=False, threshold=0.80):
    """Conservative disposition: PASS/FAIL only when a model returns a confident known class."""
    if not model_available or predicted_class is None or confidence is None:
        return "REVIEW REQUIRED"
    label = str(predicted_class).strip().lower()
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        return "REVIEW REQUIRED"
    if not 0 <= confidence <= 1 or confidence < threshold:
        return "REVIEW REQUIRED"
    if label in {"normal", "good", "ok", "pass", "healthy", "non-defective", "non_defective"}:
        return "PASS"
    if label in {"defect", "defective", "bad", "fail", "crack", "scratch", "damaged", "unhealthy"}:
        return "FAIL"
    return "REVIEW REQUIRED"
