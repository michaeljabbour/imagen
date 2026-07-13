"""Tests for cost estimation pricing (src/config/pricing.py)."""

from __future__ import annotations

from src.config.pricing import (
    CostEstimate,
    estimate_generation_cost,
    format_cost_estimate,
)


class TestEstimateGenerationCost:
    def test_openai_basic(self):
        est = estimate_generation_cost("openai", quality="medium", size="1024x1024", n=1)
        assert est.provider == "openai"
        assert est.per_image_usd is not None
        assert est.total_usd == est.per_image_usd

    def test_openai_uses_published_sample_prices(self):
        square = estimate_generation_cost("openai", quality="high", size="1024x1024")
        landscape = estimate_generation_cost("openai", quality="high", size="1536x1024")
        assert square.per_image_usd == 0.211
        assert landscape.per_image_usd == 0.165

    def test_openai_quality_tiers_ordered(self):
        low = estimate_generation_cost("openai", quality="low", size="1024x1024")
        high = estimate_generation_cost("openai", quality="high", size="1024x1024")
        assert high.per_image_usd > low.per_image_usd

    def test_n_multiplies_total(self):
        est = estimate_generation_cost("openai", quality="medium", size="1024x1024", n=4)
        assert est.total_usd == round(est.per_image_usd * 4, 4)

    def test_openai_unknown_quality_returns_none(self):
        est = estimate_generation_cost("openai", quality="ultra", size="1024x1024")
        assert est.per_image_usd is None
        assert est.total_usd is None
        assert "quality 'ultra'" in est.note

    def test_openai_custom_size_is_not_extrapolated(self):
        est = estimate_generation_cost("openai", quality="high", size="2560x1440")
        assert est.per_image_usd is None
        assert "not extrapolated" in (est.note or "")

    def test_gemini_resolution_pricing(self):
        one_k = estimate_generation_cost("gemini", size="1K")
        four_k = estimate_generation_cost("gemini", size="4K")
        assert four_k.per_image_usd > one_k.per_image_usd
        assert one_k.per_image_usd == 0.067
        assert four_k.per_image_usd == 0.151

    def test_gemini_pro_model_costs_more(self):
        flash = estimate_generation_cost("gemini", model="gemini-3.1-flash-image", size="2K")
        pro = estimate_generation_cost("gemini", model="gemini-3-pro-image", size="2K")
        assert pro.per_image_usd > flash.per_image_usd
        assert flash.per_image_usd == 0.101
        assert pro.per_image_usd == 0.134

    def test_gemini_lite_published_price(self):
        lite = estimate_generation_cost("gemini", model="gemini-3.1-flash-lite-image", size="1K")
        assert lite.per_image_usd == 0.0336

    def test_gemini_unknown_size_returns_none(self):
        est = estimate_generation_cost("gemini", size="8K")
        assert est.per_image_usd is None
        assert "No documented price" in est.note

    def test_unknown_provider_returns_none(self):
        est = estimate_generation_cost("midjourney")
        assert est.per_image_usd is None
        assert "No pricing data" in est.note

    def test_n_floored_to_one(self):
        est = estimate_generation_cost("openai", quality="low", size="1024x1024", n=0)
        assert est.n == 1


class TestFormatCostEstimate:
    def test_format_with_price(self):
        est = estimate_generation_cost("openai", quality="medium", size="1024x1024", n=2)
        out = format_cost_estimate(est)
        assert "Cost Estimate" in out
        assert "$" in out
        assert "Approximate" in out

    def test_format_without_price(self):
        est = CostEstimate(
            provider="openai",
            model=None,
            quality="ultra",
            size="1024x1024",
            n=1,
            per_image_usd=None,
            total_usd=None,
            note="Unknown quality 'ultra' for OpenAI",
        )
        out = format_cost_estimate(est)
        assert "unavailable" in out
