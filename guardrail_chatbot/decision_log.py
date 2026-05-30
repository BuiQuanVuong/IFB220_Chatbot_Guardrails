"""
Structured decision logging.

Every guardrail decision -- allow or deny, and why -- is appended to a
JSONL file (one JSON object per line) and optionally echoed to the console.
JSONL is chosen over a free-text log because it is trivially parseable for
later analysis (e.g. "how many injections were blocked this session?").
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from . import config


class DecisionLogger:
    def __init__(self, path: str = config.LOG_FILE,
                to_console: bool = config.LOG_TO_CONSOLE):
        self.path = path
        self.to_console = to_console

    def log(self, *, stage: str, verdict: str, reason: str,
            scores: dict[str, Any] | None = None,
            user_input: str | None = None) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "stage": stage,          # which guardrail layer made the call
            "verdict": verdict,      # "allow" | "deny"
            "reason": reason,        # human-readable justification
            "scores": scores or {},  # numeric evidence (similarities, tokens)
        }
        if user_input is not None:
            # Truncate to keep the log readable; full text is the user's anyway.
            record["input"] = user_input[:200]
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        if self.to_console and verdict == "deny":
            s = scores or {}
            detail = " ".join(f"{k}={v}" for k, v in s.items())
            print(f"  [guardrail] DENY @ {stage}: {reason} {detail}".rstrip())
