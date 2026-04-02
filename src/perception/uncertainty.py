from typing import Optional


def flag_uncertainty(confidence: Optional[float], threshold: float = 0.5) -> bool:
    if confidence is None:
        return True
    return confidence < threshold
