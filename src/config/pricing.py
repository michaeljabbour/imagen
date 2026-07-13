"""Approximate image-generation pricing for cost estimation.

These figures are published examples intended to help callers compare
provider/model/quality combinations before generation. They are not a billing
calculator: undocumented combinations return unavailable rather than being
extrapolated, and live pricing or account-specific terms may differ.

Sources (as of 2026-07): OpenAI gpt-image pricing tiers and Google
Gemini image pricing pages. Update ``PRICING`` when providers change rates.
"""

from __future__ import annotations

from dataclasses import dataclass

from .constants import GEMINI_MODEL_ALIASES, GEMINI_MODELS

# Published per-image examples. Do not extrapolate arbitrary-resolution costs:
# provider billing is not documented as a linear pixel multiplier.
_OPENAI_GPT_IMAGE_2_USD: dict[str, dict[str, float]] = {
    "1024x1024": {"low": 0.006, "medium": 0.053, "high": 0.211},
    "1024x1536": {"low": 0.005, "medium": 0.041, "high": 0.165},
    "1536x1024": {"low": 0.005, "medium": 0.041, "high": 0.165},
}

_GEMINI_FLASH_USD: dict[str, float] = {
    "0.5K": 0.045,
    "1K": 0.067,
    "2K": 0.101,
    "4K": 0.151,
}
_GEMINI_PRO_USD: dict[str, float] = {"1K": 0.134, "2K": 0.134, "4K": 0.240}
_GEMINI_LITE_USD: dict[str, float] = {"1K": 0.0336}


@dataclass
class CostEstimate:
    """Result of a cost estimation."""

    provider: str
    model: str | None
    quality: str | None
    size: str | None
    n: int
    per_image_usd: float | None
    total_usd: float | None
    approximate: bool = True
    note: str | None = None


def _estimate_openai(
    model: str | None, quality: str | None, size: str | None
) -> tuple[float | None, str | None]:
    resolved_model = model or "gpt-image-2"
    if resolved_model != "gpt-image-2":
        return None, f"No documented local pricing table for OpenAI model '{resolved_model}'"

    q = (quality or "auto").lower()
    s = (size or "1024x1024").lower().replace("X", "x")
    prices = _OPENAI_GPT_IMAGE_2_USD.get(s)
    if prices is None:
        return None, (
            f"No documented sample price for gpt-image-2 size '{size or s}'; "
            "cost is not extrapolated from pixel count"
        )
    price = prices.get(q)
    if price is None:
        return None, f"No documented sample price for gpt-image-2 quality '{quality or q}'"
    return price, None


def _estimate_gemini(model: str | None, size: str | None) -> tuple[float | None, str | None]:
    s = (size or "1K").upper()
    requested_model = model or "gemini-3.1-flash-image"
    resolved_model = GEMINI_MODEL_ALIASES.get(requested_model, requested_model)
    if resolved_model not in GEMINI_MODELS:
        return None, f"Unsupported Gemini image model '{requested_model}'"
    if resolved_model == "gemini-3.1-flash-lite-image":
        prices = _GEMINI_LITE_USD
    elif resolved_model == "gemini-3-pro-image":
        prices = _GEMINI_PRO_USD
    else:
        prices = _GEMINI_FLASH_USD
    price = prices.get(s)
    if price is None:
        return None, f"No documented price for Gemini model '{requested_model}' at size '{s}'"
    return price, None


def estimate_generation_cost(
    provider: str,
    *,
    model: str | None = None,
    quality: str | None = None,
    size: str | None = None,
    n: int = 1,
) -> CostEstimate:
    """Estimate the cost of generating ``n`` images.

    Returns a :class:`CostEstimate`; ``per_image_usd``/``total_usd`` are
    ``None`` when the combination isn't in the pricing table (with the
    reason in ``note``).
    """
    provider = provider.lower()
    n = max(1, int(n))

    if provider == "openai":
        per_image, note = _estimate_openai(model, quality, size)
    elif provider == "gemini":
        per_image, note = _estimate_gemini(model, size)
    else:
        per_image, note = None, f"No pricing data for provider '{provider}'"

    total = round(per_image * n, 4) if per_image is not None else None
    return CostEstimate(
        provider=provider,
        model=model,
        quality=quality,
        size=size,
        n=n,
        per_image_usd=per_image,
        total_usd=total,
        note=note,
    )


def format_cost_estimate(est: CostEstimate) -> str:
    """Render a cost estimate as markdown."""
    lines = ["## 💵 Cost Estimate", ""]
    lines.append(f"**Provider:** {est.provider.title()}")
    if est.model:
        lines.append(f"**Model:** {est.model}")
    if est.quality:
        lines.append(f"**Quality:** {est.quality}")
    if est.size:
        lines.append(f"**Size:** {est.size}")
    lines.append(f"**Images:** {est.n}")
    lines.append("")

    if est.total_usd is not None:
        lines.append(f"**Estimated cost:** ~${est.total_usd:.4f} (${est.per_image_usd:.4f}/image)")
    else:
        lines.append(f"**Estimated cost:** unavailable — {est.note or 'no pricing data'}")

    lines.extend(
        [
            "",
            "> ⚠️ Approximate. Real cost depends on live provider pricing and "
            "(for OpenAI) actual image output tokens.",
        ]
    )
    return "\n".join(lines)
