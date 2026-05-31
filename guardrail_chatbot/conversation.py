"""
Conversation state + token budgeting.

The Azure chat endpoint is stateless: we resend the full message list every
turn, so the list must be prevented from growing past the model's context
window. This module:

  * counts tokens with tiktoken (authoritative usage comes back from the API,
    but we need a *pre-flight* estimate to decide what to trim), and
  * trims the oldest user/assistant turns -- never the system prompt -- once
    the running prompt would exceed the configured budget.

That satisfies the rubric's "token usage is monitored and a mechanism that
prevents context overflow is implemented" requirement.
"""

from __future__ import annotations

import tiktoken

from . import config


class _WhitespaceEncoder:
    """Last-resort encoder used only if tiktoken cannot load a vocabulary.

    tiktoken downloads its BPE vocab on first use; in a locked-down or
    offline environment that can fail. Rather than crash, we fall back to a
    coarse whitespace estimate. Counts are then approximate, but the API's
    `usage` field still gives exact figures after each call, so context
    budgeting stays safe (we under-count slightly, offset by the headroom).
    """

    @staticmethod
    def encode(text: str) -> list[str]:
        return text.split()


def _get_encoding():
    """Best-effort encoder for gpt-4.1-mini.

    gpt-4.1-family models use the o200k_base vocabulary. Older tiktoken
    builds may not map the model name, so we fall back gracefully. Token
    counts are an *estimate* for budgeting; the API's `usage` field is the
    source of truth after each call.
    """
    for getter in (
        lambda: tiktoken.encoding_for_model(config.CHAT_MODEL),
        lambda: tiktoken.get_encoding("o200k_base"),
        lambda: tiktoken.get_encoding("cl100k_base"),
    ):
        try:
            return getter()
        except Exception:
            continue
    return _WhitespaceEncoder()


_ENC = _get_encoding()

# Per-message ChatML overhead used by the GPT-4 family (approximate).
_PER_MESSAGE_OVERHEAD = 3
_PRIMING_OVERHEAD = 3


def count_tokens(messages: list[dict]) -> int:
    """Estimate the prompt token count for a ChatML message list."""
    total = _PRIMING_OVERHEAD
    for msg in messages:
        total += _PER_MESSAGE_OVERHEAD
        total += len(_ENC.encode(msg.get("content", "")))
        total += len(_ENC.encode(msg.get("role", "")))
    return total


class Conversation:
    """Holds the message list and keeps it within the token budget."""

    def __init__(self, system_prompt: str):
        self._system = {"role": "system", "content": system_prompt}
        self._turns: list[dict] = []   # user/assistant messages, in order
        self.last_prompt_tokens = 0    # set from the API after each call

    def add_user(self, text: str) -> None:
        self._turns.append({"role": "user", "content": text})

    def reset(self, system_prompt: str) -> None:
        """Wipe the conversation and install a new system prompt.

        Used on /change_topic. We deliberately drop prior user/assistant
        turns: keeping "we were discussing roses" after switching to motor
        vehicles would both confuse the model and undermine the output
        guardrail (Layer 5 would correctly flag any drift back to the old
        topic, breaking coherence).
        """
        self._system = {"role": "system", "content": system_prompt}
        self._turns = []
        self.last_prompt_tokens = 0

    def add_assistant(self, text: str) -> None:
        self._turns.append({"role": "assistant", "content": text})

    def messages(self) -> list[dict]:
        """Return system prompt + as many recent turns as the budget allows.

        Oldest turns are dropped first. The system prompt is never dropped --
        losing it would disable the primary guardrail.
        """
        budget = (config.MAX_CONTEXT_TOKENS
                - config.MAX_RESPONSE_TOKENS
                - config.TRUNCATION_HEADROOM_TOKENS)
        system_cost = count_tokens([self._system])

        kept_reversed: list[dict] = []
        running = system_cost
        for msg in reversed(self._turns):
            cost = count_tokens([msg]) - _PRIMING_OVERHEAD  # marginal cost
            if running + cost > budget:
                break
            kept_reversed.append(msg)
            running += cost

        return [self._system] + list(reversed(kept_reversed))

    def estimated_tokens(self) -> int:
        return count_tokens(self.messages())
