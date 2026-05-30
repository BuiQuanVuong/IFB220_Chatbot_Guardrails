"""
Threshold calibration tool.

The default thresholds in config.py are *starting points*. ada-002 embeddings
are anisotropic -- unrelated texts still score fairly high cosine similarity
(see Ethayarajh, 2019, "How Contextual are Contextualized Word
Representations?") -- so good thresholds must be measured, not guessed.

Run this against the live API:

    python calibrate.py

It prints, for a labelled probe set, the distribution of positive/negative
similarity and the contrastive margin, then suggests a `floor` and `margin`
that separate on-topic from off-topic probes. Paste the suggestions into
config.TOPIC and re-run to confirm.
"""

from __future__ import annotations

from guardrail_chatbot import api_client, config
from guardrail_chatbot.embeddings import Embedder
from guardrail_chatbot.guardrails import AnchorSet

# Held-out probes NOT used as anchors, so this is an honest test.
ON_TOPIC_PROBES = [
    "Why are the leaves on my basil turning yellow?",
    "Can I grow strawberries in a hanging basket?",
    "What's the best mulch to retain moisture in summer?",
    "How do I prune lavender after it flowers?",
    "Is coffee grounds good for my compost?",
]
OFF_TOPIC_PROBES = [
    "What's the weather forecast for tomorrow?",
    "Help me debug this JavaScript error.",
    "Who won the football match last night?",
    "What's a good investment for retirement?",
    "Translate this sentence into Spanish.",
]


def main() -> None:
    key = api_client.get_api_key()
    embedder = Embedder(lambda texts: api_client.embed(texts, key))
    pos = AnchorSet("pos", config.TOPIC.positive_anchors, embedder)
    neg = AnchorSet("neg", config.TOPIC.negative_anchors, embedder)

    def stats(probes, label):
        rows = []
        for p in probes:
            v = embedder.embed_one(p)
            sp, sn = pos.max_similarity(v), neg.max_similarity(v)
            rows.append((sp, sn, sp - sn))
            print(f"  [{label}] pos={sp:.4f} neg={sn:.4f} margin={sp-sn:+.4f}  {p}")
        return rows

    print("\nON-TOPIC probes:")
    on = stats(ON_TOPIC_PROBES, "ON ")
    print("\nOFF-TOPIC probes:")
    off = stats(OFF_TOPIC_PROBES, "OFF")

    on_margins = sorted(m for _, _, m in on)
    off_margins = sorted(m for _, _, m in off)
    on_pos = sorted(sp for sp, _, _ in on)

    print("\n--- Suggestions ---")
    # A margin between the worst on-topic and best off-topic separates them.
    suggested_margin = (on_margins[0] + off_margins[-1]) / 2
    suggested_floor = on_pos[0] - 0.01
    print(f"  worst on-topic margin : {on_margins[0]:+.4f}")
    print(f"  best off-topic margin : {off_margins[-1]:+.4f}")
    print(f"  -> suggested margin   : {suggested_margin:+.4f}")
    print(f"  -> suggested floor    : {suggested_floor:.4f}")
    if on_margins[0] <= off_margins[-1]:
        print("  WARNING: on/off-topic margins overlap. Add more/better "
            "anchors, or rely more on the LLM judge for borderline cases.")


if __name__ == "__main__":
    main()
