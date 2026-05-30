"""
Application entry point: wires the layers together and runs the chat loop.

Flow per user turn:
    input -> guardrail.check_input
        -> (deny) polite refusal, log, continue
        -> (allow) chat completion -> guardrail.check_output
                                    -> (deny) safe fallback
                                    -> (allow) show reply
        -> token budgeting / context truncation
"""

from __future__ import annotations

from . import api_client, config
from .conversation import Conversation
from .decision_log import DecisionLogger
from .embeddings import Embedder
from .guardrails import GuardrailEngine, build_system_prompt

# Denials from these stages must NOT have the offending user text written into
# the conversation history -- we don't feed attack strings back to the model.
_NON_PERSISTED_DENIALS = {
    "regex_injection", "semantic_injection", "unsafe_input", "output_leak",
}


def make_judge(key: str):
    """Build the Layer 4b LLM judge: an *independent* on-topic classifier.

    It is a separate, single-shot call with its own minimal prompt and no
    conversation history, so a poisoned chat context cannot influence its
    verdict. It returns a strict YES/NO, which we map to a boolean.
    """
    def judge(text: str, topic_name: str) -> bool:
        messages = [
            {"role": "system",
            "content": (
                f"You are a binary classifier. Decide whether the user's "
                f"message is genuinely about the topic '{topic_name}'. "
                "Answer with exactly one word: YES or NO. Do not explain.")},
            {"role": "user", "content": text},
        ]
        try:
            result = api_client.chat(messages, key, max_tokens=2, temperature=0)
            return result.content.strip().upper().startswith("Y")
        except api_client.APIError:
            # Fail closed: if the judge is unavailable, treat as off-topic.
            return False
    return judge


def run() -> None:
    key = api_client.get_api_key()

    # Inversion of control: the Embedder receives a raw embed callable.
    embedder = Embedder(lambda texts: api_client.embed(texts, key))
    logger = DecisionLogger()
    judge = make_judge(key)
    engine = GuardrailEngine(embedder, config.TOPIC, logger, judge_fn=judge)

    system_prompt = build_system_prompt(config.TOPIC)
    conv = Conversation(system_prompt)

    topic = config.TOPIC.name
    print(f"=== {topic.capitalize()} Assistant ===")
    print(f"Ask me anything about {topic}. Type /help for commands.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("/quit", "/exit"):
            print("Goodbye!")
            break
        if user_input.lower() == "/help":
            print("  /tokens  show current context-token estimate")
            print("  /quit    exit\n")
            continue
        if user_input.lower() == "/tokens":
            print(f"  ~{conv.estimated_tokens()} prompt tokens "
                f"(budget {config.MAX_CONTEXT_TOKENS}); "
                f"last API prompt_tokens={conv.last_prompt_tokens}\n")
            continue

        # ---- Input guardrails --------------------------------------------
        v_in = engine.check_input(user_input)
        if not v_in.allowed:
            print(f"Bot: {v_in.refusal}\n")
            if v_in.stage not in _NON_PERSISTED_DENIALS:
                # Off-topic: keep the exchange coherent for later turns.
                conv.add_user(user_input)
                conv.add_assistant(v_in.refusal)
            continue

        # ---- Model call ---------------------------------------------------
        conv.add_user(user_input)
        try:
            result = api_client.chat(conv.messages(), key)
        except api_client.APIError as exc:
            print(f"Bot: Sorry, I hit a problem reaching the service ({exc}).\n")
            continue
        conv.last_prompt_tokens = result.prompt_tokens

        # ---- Output guardrails -------------------------------------------
        v_out = engine.check_output(result.content, system_prompt)
        shown = result.content if v_out.allowed else v_out.refusal
        conv.add_assistant(shown)
        print(f"Bot: {shown}")
        print(f"  (tokens: prompt={result.prompt_tokens}, "
            f"completion={result.completion_tokens}, "
            f"total={result.total_tokens})\n")


if __name__ == "__main__":
    run()
