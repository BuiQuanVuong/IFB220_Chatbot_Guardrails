"""
Application entry point: wires the layers together and runs the chat loop.

Flow per user turn:
    input -> slash-command handler (incl. /change_topic)
        -> guardrail.check_input
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
from .guardrails import GuardrailEngine

# Not feeding attack strings back to the model.
_NON_PERSISTED_DENIALS = {
    "regex_injection", "semantic_injection", "unsafe_input", "output_leak",
}


def make_judge(key: str):
    """Build the Layer 4b LLM judge: an *independent* on-topic classifier.

    A separate, single-shot call with its own minimal prompt and no
    conversation history, so a poisoned chat context cannot influence its
    verdict. Returns a strict YES/NO, mapped to a boolean.
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
            return False
    return judge


def _print_topics_menu(current_slug: str) -> None:
    print(f"  Available topics (current: {current_slug}):")
    width = max(len(s) for s in config.TOPICS) + 2
    for slug, topic in config.TOPICS.items():
        marker = "*" if slug == current_slug else " "
        print(f"   {marker} /{slug:<{width}} {topic.description[:60]}"
            + ("..." if len(topic.description) > 60 else ""))
    print("  Type the topic command (e.g. /sport) to switch, or "
        "/change_topic <name>.\n")


def _switch_topic(slug: str, engine: GuardrailEngine,
                conv: Conversation) -> None:
    new_topic = config.TOPICS[slug]
    engine.set_topic(new_topic)
    conv.reset(engine.system_prompt)
    print(f"  Switched to: {new_topic.name}. (Conversation history cleared.)")
    print(f"  Ask me anything about {new_topic.name}.\n")


def _handle_command(user_input: str, engine: GuardrailEngine,
                    conv: Conversation, current_slug: list) -> str | None:
    """Handle slash commands.

    Returns:
        "quit"     -- loop should exit
        "handled"  -- command consumed, continue loop
        None       -- not a recognised command, treat as user message

    ``current_slug`` is a one-element list used as a mutable reference so
    /change_topic can update it without restructuring the loop.
    """
    text = user_input.lower()

    if text in ("/quit", "/exit"):
        print("Goodbye!")
        return "quit"

    if text == "/help":
        print("  /help                       show this help")
        print("  /tokens                     show current context-token estimate")
        print("  /change_topic [<name>]      list topics, or switch directly")
        print("  /<topic_name>               switch to that topic (e.g. /sport)")
        print("  /quit                       exit\n")
        return "handled"

    if text == "/tokens":
        print(f"  ~{conv.estimated_tokens()} prompt tokens "
            f"(budget {config.MAX_CONTEXT_TOKENS}); "
            f"last API prompt_tokens={conv.last_prompt_tokens}\n")
        return "handled"

    # /change_topic
    parts = text.split()
    if parts[0] == "/change_topic":
        if len(parts) == 1:
            _print_topics_menu(current_slug[0])
            return "handled"
        target = parts[1].lstrip("/")
        if target not in config.TOPICS:
            print(f"  Unknown topic '{target}'. "
                f"Choices: {', '.join(config.TOPICS)}\n")
            return "handled"
        _switch_topic(target, engine, conv)
        current_slug[0] = target
        return "handled"

    # Per-topic shortcut: /gardening, /motor_vehicles, /sport, /cinematography
    if text.startswith("/"):
        candidate = text.lstrip("/")
        if candidate in config.TOPICS:
            if candidate == current_slug[0]:
                print(f"  Already on '{candidate}'.\n")
                return "handled"
            _switch_topic(candidate, engine, conv)
            current_slug[0] = candidate
            return "handled"
        # Unknown slash command -- explain rather than send to the model.
        print(f"  Unknown command '{user_input}'. Type /help.\n")
        return "handled"

    return None


def run() -> None:
    key = api_client.get_api_key()

    # The Embedder receives a raw embed callable.
    embedder = Embedder(lambda texts: api_client.embed(texts, key))
    logger = DecisionLogger()
    judge = make_judge(key)

    current_slug = [config.DEFAULT_TOPIC_SLUG]
    engine = GuardrailEngine(embedder, config.TOPICS[current_slug[0]],
                            logger, judge_fn=judge)
    conv = Conversation(engine.system_prompt)

    print(f"=== {engine.topic.name.capitalize()} Assistant ===")
    print(f"Ask me anything about {engine.topic.name}. Type /help for commands.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        cmd = _handle_command(user_input, engine, conv, current_slug)
        if cmd == "quit":
            break
        if cmd == "handled":
            continue

        # Input guardrails
        v_in = engine.check_input(user_input)
        if not v_in.allowed:
            print(f"Bot: {v_in.refusal}\n")
            if v_in.stage not in _NON_PERSISTED_DENIALS:
                # Keep the exchange coherent for later turns.
                conv.add_user(user_input)
                conv.add_assistant(v_in.refusal)
            continue

        # Model call
        conv.add_user(user_input)
        try:
            result = api_client.chat(conv.messages(), key)
        except api_client.APIError as exc:
            print(f"Bot: Sorry, I hit a problem reaching the service ({exc}).\n")
            continue
        conv.last_prompt_tokens = result.prompt_tokens

        # Output guardrails
        v_out = engine.check_output(result.content)
        shown = result.content if v_out.allowed else v_out.refusal
        conv.add_assistant(shown)
        print(f"Bot: {shown}")
        print(f"  (tokens: prompt={result.prompt_tokens}, "
            f"completion={result.completion_tokens}, "
            f"total={result.total_tokens})\n")


if __name__ == "__main__":
    run()
