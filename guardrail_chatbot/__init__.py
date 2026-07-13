"""Topic-constrained chatbot with layered AI guardrails (IFB220 A2)."""

__all__ = ["run"]


def __getattr__(name):
    # Lazy import so that importing lightweight submodules (e.g. ``config`` in
    # the offline tests) does not pull in the network stack (requests/dotenv)
    # via ``chatbot``. ``from guardrail_chatbot import run`` still works.
    if name == "run":
        from .chatbot import run
        return run
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
