"""Access to the recorded BoTorch call results in ``golden.json``."""

from __future__ import annotations

import json
from pathlib import Path


GOLDEN_PATH = Path(__file__).parent / "golden.json"


def load_golden() -> dict[str, dict]:
    """Recorded results keyed by ``spec_key`` (see ``golden.json``'s ``_meta``)."""
    raw = json.loads(GOLDEN_PATH.read_text())
    return {key: value for key, value in raw.items() if key != "_meta"}


GOLDEN = load_golden()
