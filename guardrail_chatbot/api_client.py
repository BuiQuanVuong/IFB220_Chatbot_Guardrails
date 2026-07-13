"""
Client for the two Azure AI endpoints.

Responsibilities:
  * attach the api-key header,
  * POST the request, retrying on transient errors,
  * surface token usage so the conversation manager can budget context,
  * raise an exception on failure rather than crashing the loop.

Keeping this in a separate module isolates every network detail behind two
simple functions (`chat` and `embed`). The guardrail logic never touches
`requests`, which keeps the layers decoupled and easy to unit test offline.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass

import requests
from dotenv import load_dotenv

from . import config


class APIError(RuntimeError):
    """Raised when the Azure API cannot satisfy a request."""


@dataclass
class ChatResult:
    """Return type for a chat completion: the text plus token accounting."""
    content: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


def get_api_key() -> str:
    """Load the API key from the environment / .env, prompting once if absent."""
    load_dotenv()
    key = os.getenv("AI_API_KEY")
    if key:
        return key.strip()

    print("--- First-time setup: no AI_API_KEY found ---")
    key = input("Enter your AI API key: ").strip()
    if not key:
        print("Error: key cannot be empty.")
        sys.exit(1)
    with open(".env", "a", encoding="utf-8") as f:
        f.write(f"\nAI_API_KEY={key}\n")
    load_dotenv()
    print("Saved key to .env\n")
    return key


def _post(url: str, key: str, payload: dict) -> dict:
    """POST with a small exponential-backoff retry for transient failures."""
    headers = {"Content-Type": "application/json", "api-key": key}
    last_err: Exception | None = None
    for attempt in range(config.MAX_RETRIES + 1):
        try:
            resp = requests.post(
                url, headers=headers, json=payload,
                timeout=config.REQUEST_TIMEOUT_S,
            )
            # 429 / 5xx are worth retrying; 4xx (except 429) are not.
            if resp.status_code == 429 or resp.status_code >= 500:
                raise APIError(f"transient HTTP {resp.status_code}")
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, APIError) as exc:
            last_err = exc
            if attempt < config.MAX_RETRIES:
                time.sleep(1.5 * (attempt + 1))  # 1.5s, 3.0s, ...
            else:
                break
    raise APIError(f"request to {url.split('/deployments/')[-1][:30]}... "
                    f"failed after retries: {last_err}")


def chat(messages: list[dict], key: str,
        max_tokens: int = config.MAX_RESPONSE_TOKENS,
        temperature: float = 0.4) -> ChatResult:
    """Call gpt-4.1-mini and return the reply plus usage figures."""
    payload = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    data = _post(config.CHAT_ENDPOINT, key, payload)
    try:
        content = data["choices"][0]["message"]["content"].strip()
        usage = data.get("usage", {})
        return ChatResult(
            content=content,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
        )
    except (KeyError, IndexError, TypeError) as exc:
        raise APIError(f"unexpected chat response shape: {exc}") from exc


def embed(texts: list[str], key: str) -> list[list[float]]:
    """Embed one or more strings with ada-002. Returns a list of vectors.

    The endpoint accepts a single string or an array (see the embedding
    schema). We always pass an array so a single call can embed several
    anchors at once -- far fewer round-trips than one call per string.
    """
    payload = {"input": texts}
    data = _post(config.EMBED_ENDPOINT, key, payload)
    try:
        # The API may return items out of order; sort by `index` to be safe.
        items = sorted(data["data"], key=lambda d: d["index"])
        return [item["embedding"] for item in items]
    except (KeyError, TypeError) as exc:
        raise APIError(f"unexpected embedding response shape: {exc}") from exc
