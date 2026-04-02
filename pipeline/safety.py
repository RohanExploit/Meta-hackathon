"""NeMo Guardrails equivalent — Llama Guard 3 input/output middleware.

Uses Groq's hosted `llama-guard-3-8b` to classify text as safe/unsafe.
Both the user's inbound query and the model's outbound response are screened.
If either is classified `unsafe`, the pipeline halts immediately and returns
a predefined security-violation response.

This module exposes two async functions:
    - guard_input(text)  → (is_safe: bool, category: str | None)
    - guard_output(text) → (is_safe: bool, category: str | None)

And a convenience exception:
    - SafetyViolation
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Tuple

from groq import AsyncGroq

from .config import get_config

logger = logging.getLogger(__name__)

# ── Predefined violation responses ──────────────────────────────────
INPUT_VIOLATION_RESPONSE = (
    "⚠️ Your query has been flagged as potentially unsafe and cannot be processed. "
    "Please rephrase your request or contact support if you believe this is an error."
)

OUTPUT_VIOLATION_RESPONSE = (
    "⚠️ The generated response was flagged by our safety system and has been withheld. "
    "This is a precautionary measure to prevent potentially harmful content."
)


class SafetyViolation(Exception):
    """Raised when Llama Guard flags content as unsafe."""

    def __init__(self, direction: str, category: str | None = None):
        self.direction = direction  # "input" or "output"
        self.category = category
        msg = f"Safety violation on {direction}"
        if category:
            msg += f" (category: {category})"
        super().__init__(msg)


@dataclass
class GuardResult:
    """Structured result from a Llama Guard check."""

    is_safe: bool
    raw_response: str
    category: str | None = None


# ── Internal helper ──────────────────────────────────────────────────

_client: AsyncGroq | None = None


def _get_client() -> AsyncGroq:
    global _client
    if _client is None:
        cfg = get_config()
        _client = AsyncGroq(api_key=cfg.groq.api_key)
    return _client


async def _classify(
    text: str,
    role: str = "User",
) -> GuardResult:
    """Send text through Llama Guard 3 for safety classification.

    Llama Guard 3 returns either:
        "safe"
        "unsafe\\nS<category_number>"

    Args:
        text: The text to classify.
        role: "User" for input guarding, "Agent" for output guarding.

    Returns:
        GuardResult with is_safe flag and optional violation category.
    """
    cfg = get_config()
    client = _get_client()

    # Llama Guard 3 expects a specific prompt format
    guard_prompt = (
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"Task: Check if there is unsafe content in '{role}' messages "
        f"in conversations according to our safety policy.\n\n"
        f"<BEGIN CONVERSATION>\n\n"
        f"{role}: {text}\n\n"
        f"<END CONVERSATION>\n\n"
        f"Provide your safety assessment for {role} in the above conversation:\n"
        f"- First line must read 'safe' or 'unsafe'.\n"
        f"- If unsafe, a second line must include a comma-separated list of "
        f"violated categories.<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    try:
        response = await client.chat.completions.create(
            model=cfg.groq.guard_model,
            messages=[{"role": "user", "content": guard_prompt}],
            temperature=0.0,
            max_tokens=50,
        )
        raw = (response.choices[0].message.content or "").strip().lower()
    except Exception as exc:
        # If the guard itself fails, we log and allow through (fail-open)
        # In production you may want fail-closed instead.
        logger.warning("Llama Guard call failed: %s — defaulting to safe", exc)
        return GuardResult(is_safe=True, raw_response=f"ERROR: {exc}")

    if raw.startswith("unsafe"):
        lines = raw.split("\n", 1)
        category = lines[1].strip() if len(lines) > 1 else None
        return GuardResult(is_safe=False, raw_response=raw, category=category)

    return GuardResult(is_safe=True, raw_response=raw)


# ── Public API ───────────────────────────────────────────────────────

async def guard_input(text: str) -> GuardResult:
    """Screen a user query before it enters the pipeline.

    Raises SafetyViolation if flagged unsafe.
    """
    result = await _classify(text, role="User")
    if not result.is_safe:
        logger.warning("INPUT blocked: %s", result.raw_response)
        raise SafetyViolation("input", result.category)
    return result


async def guard_output(text: str) -> GuardResult:
    """Screen the final generated response before returning to the user.

    Raises SafetyViolation if flagged unsafe.
    """
    result = await _classify(text, role="Agent")
    if not result.is_safe:
        logger.warning("OUTPUT blocked: %s", result.raw_response)
        raise SafetyViolation("output", result.category)
    return result


async def guard_both(query: str, response: str) -> Tuple[GuardResult, GuardResult]:
    """Convenience: run input and output guards concurrently.

    This is useful when you want to re-validate the query alongside
    the response in a single await.
    """
    return await asyncio.gather(
        guard_input(query),
        guard_output(response),
    )
