"""
BriefMap — hybrid LLM client (Anthropic Claude + Google Gemini).

Provider assignment per stage:
  Anthropic  → Stages 2, 3, 5, 9  (core reasoning, quality-critical)
  Gemini     → Stages 4, 6        (validation fallback, supplementary — cost saving)
  Gemini Pro → Stage 4.5          (large-context competitor site audit)

Both providers share the same call_structured() interface so stages
don't need to know which model they're using.

Reliability + cost features:
  - Anthropic calls use FORCED TOOL-USE so the model must emit JSON that
    matches the pydantic schema (no fence-stripping, far fewer retries).
  - Anthropic system prompts are sent with cache_control so repeated
    per-pillar / per-brief calls hit the prompt cache (~90% input discount).
  - Gemini calls request JSON output via response_mime_type + response_schema.
  - Low temperature (0.3) by default for structured stages.
  - Text-extraction fallback is kept for resilience if a provider ignores
    the structured-output request.
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Literal, TypeVar

from google import genai as google_genai
from anthropic import Anthropic
from pydantic import BaseModel, ValidationError


T = TypeVar("T", bound=BaseModel)

# ── Model constants ──────────────────────────────────────────────────────────
ANTHROPIC_SONNET   = "claude-sonnet-4-6"           # main reasoning model
ANTHROPIC_HAIKU    = "claude-haiku-4-5"            # cheap structured work
GEMINI_FLASH       = "gemini-2.5-flash"            # fast + cheap (1.5 family is retired)
GEMINI_PRO         = "gemini-2.5-pro"              # large context window

DEFAULT_MODEL       = ANTHROPIC_SONNET
DEFAULT_MAX_TOKENS  = 8000
DEFAULT_TEMPERATURE = 0.3   # structured JSON stages want low variance

Provider = Literal["anthropic", "gemini"]


# ── Client builders ──────────────────────────────────────────────────────────

def get_anthropic_client() -> Anthropic:
    """Build Anthropic client from env / .env / Streamlit secrets."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. "
            "Set it in your environment, .env file, or Streamlit secrets."
        )
    return Anthropic(api_key=api_key)


def _get_gemini_client():
    """Build Gemini client from env / .env / Streamlit secrets (google-genai SDK)."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not set. "
            "Set it in your environment, .env file, or Streamlit secrets."
        )
    return google_genai.Client(api_key=api_key)


def _detect_provider(model: str) -> Provider:
    """Infer provider from model string."""
    if model.startswith("gemini"):
        return "gemini"
    return "anthropic"


# ── JSON extraction (fallback path) ──────────────────────────────────────────

def _extract_json(text: str) -> str:
    """
    Strip markdown fences and preamble/postamble from any JSON object.
    Only used when structured output was not returned by the provider.
    """
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1)
    first_brace = text.find("{")
    last_brace  = text.rfind("}")
    if first_brace == -1 or last_brace == -1 or last_brace < first_brace:
        raise ValueError(f"No JSON object found in response. Got:\n{text[:500]}")
    return text[first_brace : last_brace + 1]


# ── Prompt loader ─────────────────────────────────────────────────────────────

def load_prompt(name: str) -> str:
    """Load a system prompt from the prompts/ directory (always UTF-8)."""
    path = Path(__file__).parent.parent / "prompts" / f"{name}.txt"
    return path.read_text(encoding="utf-8")


# ── Unified call_structured ───────────────────────────────────────────────────

def call_structured(
    system_prompt: str,
    user_message: str,
    response_model: type[T],
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_retries: int = 2,
    temperature: float = DEFAULT_TEMPERATURE,
    stage: str = "unknown",
) -> T:
    """
    Call either Anthropic or Gemini based on the model string,
    then parse the response into a pydantic model.

    Retries on transient API errors and malformed/invalid JSON.
    """
    provider = _detect_provider(model)
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            if provider == "anthropic":
                result = _call_anthropic(
                    system_prompt, user_message, model, max_tokens,
                    response_model=response_model,
                    temperature=temperature, stage=stage,
                )
            else:
                result = _call_gemini(
                    system_prompt, user_message, model,
                    response_model=response_model,
                    temperature=temperature, stage=stage,
                )

            if isinstance(result, dict):
                data = result                      # structured output path
            else:
                data = json.loads(_extract_json(result))
            return response_model.model_validate(data)

        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            last_error = e
            if attempt < max_retries:
                user_message += (
                    f"\n\nIMPORTANT: Your previous response failed JSON validation. "
                    f"Error: {type(e).__name__}: {str(e)[:200]}. "
                    f"Return ONLY valid JSON matching the schema."
                )
                continue
            raise RuntimeError(
                f"Failed after {max_retries + 1} attempts. Last error: {e}"
            ) from e

        except Exception as e:
            last_error = e
            if attempt < max_retries:
                time.sleep(2 ** attempt)
                continue
            raise

    raise RuntimeError(f"Unreachable. Last error: {last_error}")


# ── Provider-specific call helpers ────────────────────────────────────────────

def _log_usage(stage: str, model: str, input_tokens: int, output_tokens: int) -> None:
    from stages.cost_tracker import current_tracker
    current_tracker().log_llm_call(
        stage=stage, model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def _call_anthropic(
    system_prompt: str,
    user_message: str,
    model: str,
    max_tokens: int,
    *,
    response_model: type[BaseModel] | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    stage: str = "unknown",
) -> dict | str:
    """
    Anthropic call. When response_model is given, force a tool call whose
    input_schema is the pydantic JSON schema — the API then guarantees a
    schema-shaped dict. The system prompt is cached (5-min ephemeral cache)
    so repeated per-pillar/per-brief calls pay ~10% of base input price.
    Falls back to plain text if no tool_use block comes back.
    """
    client = get_anthropic_client()

    kwargs: dict[str, Any] = {
        "model":       model,
        "max_tokens":  max_tokens,
        "temperature": temperature,
        "system": [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages": [{"role": "user", "content": user_message}],
    }

    if response_model is not None:
        kwargs["tools"] = [{
            "name":         "emit_result",
            "description":  "Emit the final structured result as JSON matching the schema.",
            "input_schema": response_model.model_json_schema(),
        }]
        kwargs["tool_choice"] = {"type": "tool", "name": "emit_result"}

    try:
        response = client.messages.create(**kwargs)
    except Exception:
        if response_model is None:
            raise
        # Some proxies / older API versions reject tool_choice or cache_control.
        # Retry once as a plain text call.
        kwargs.pop("tools", None)
        kwargs.pop("tool_choice", None)
        kwargs["system"] = system_prompt
        response = client.messages.create(**kwargs)

    usage = response.usage
    _log_usage(stage, model, usage.input_tokens, usage.output_tokens)

    for block in response.content:
        if getattr(block, "type", "") == "tool_use":
            return dict(block.input)
    for block in response.content:
        if getattr(block, "type", "") == "text":
            return block.text
    raise ValueError("Anthropic response contained no usable content block.")


def _call_gemini(
    system_prompt: str,
    user_message: str,
    model: str,
    *,
    response_model: type[BaseModel] | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    stage: str = "unknown",
) -> dict | str:
    """
    Gemini call. Requests native JSON output (response_mime_type +
    response_schema) so fence-stripping is rarely needed. Falls back to a
    plain call if the SDK/model rejects the schema config.
    """
    client = _get_gemini_client()

    base_config = dict(
        system_instruction=system_prompt,
        temperature=temperature,
    )

    response = None
    if response_model is not None:
        try:
            response = client.models.generate_content(
                model=model,
                contents=user_message,
                config=google_genai.types.GenerateContentConfig(
                    **base_config,
                    response_mime_type="application/json",
                    response_schema=response_model,
                ),
            )
        except Exception:
            response = None  # fall through to plain call

    if response is None:
        response = client.models.generate_content(
            model=model,
            contents=user_message,
            config=google_genai.types.GenerateContentConfig(**base_config),
        )

    try:
        meta = response.usage_metadata
        _log_usage(
            stage, model,
            meta.prompt_token_count or 0,
            meta.candidates_token_count or 0,
        )
    except Exception:
        pass
    return response.text


# ── Convenience wrappers for stages ──────────────────────────────────────────

def call_anthropic_structured(
    system_prompt: str,
    user_message: str,
    response_model: type[T],
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    stage: str = "unknown",
) -> T:
    """Always uses Anthropic Sonnet. For quality-critical stages."""
    return call_structured(
        system_prompt, user_message, response_model,
        model=ANTHROPIC_SONNET, max_tokens=max_tokens,
        temperature=temperature, stage=stage,
    )


def call_gemini_flash_structured(
    system_prompt: str,
    user_message: str,
    response_model: type[T],
    *,
    temperature: float = DEFAULT_TEMPERATURE,
    stage: str = "unknown",
) -> T:
    """Always uses Gemini Flash. For cost-saving stages (4, 6)."""
    return call_structured(
        system_prompt, user_message, response_model,
        model=GEMINI_FLASH, temperature=temperature, stage=stage,
    )


def call_gemini_pro_structured(
    system_prompt: str,
    user_message: str,
    response_model: type[T],
    *,
    temperature: float = DEFAULT_TEMPERATURE,
    stage: str = "unknown",
) -> T:
    """Always uses Gemini Pro (large context). For competitor site audit (Stage 4.5)."""
    return call_structured(
        system_prompt, user_message, response_model,
        model=GEMINI_PRO, temperature=temperature, stage=stage,
    )
