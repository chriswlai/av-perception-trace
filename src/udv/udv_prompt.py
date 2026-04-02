import json
from typing import Dict


def build_udv_prompt(perception_frame: Dict[str, object]) -> str:
    frame_json = json.dumps(perception_frame, indent=2)
    return (
        "You are an Understand–Decide–Verify reasoning model. "
        "Return ONLY JSON that matches the UDV schema.\n\n"
        f"PerceptionFrame:\n{frame_json}\n\n"
        "Required JSON keys: understand, decide, verify."
    )
