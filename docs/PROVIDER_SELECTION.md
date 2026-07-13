# Provider Selection — Quick Reference

This is the focused decision card for callers using `imagen-mcp` over the standard MCP protocol.

Generation auto-selects a provider based on prompt content. Override via the
`provider` parameter when you have a specific reason. When Gemini is selected,
the runtime default is `gemini-3.1-flash-image`; Pro is used only when requested
with `gemini_model`.

---

## Decision card

```
┌──────────────────────────────────────────────────────┐
│  Need TEXT readable in the image?                    │
│      → provider="openai",                            │
│        openai_model="gpt-image-2"                    │
│      Examples: menus, posters, UI mockups,           │
│                infographics, brand wordmarks         │
│                                                      │
│  Need REFERENCE images for continuity?               │
│      → provider="gemini",                            │
│        gemini_model="nano-banana-pro"                │
│        with reference_images=[base64_1, base64_2…]   │
│      Examples: same character across shots,          │
│                campaign continuity, brand identity   │
│                                                      │
│  Need PHOTOREAL hero, product, portrait, no text?    │
│      → provider="gemini",                            │
│        gemini_model="nano-banana-pro"                │
│      Examples: macro product beauty, editorial       │
│                portrait, lifestyle commercial        │
│                                                      │
│  Need TARGETED edit of a prior image?                │
│      → use edit_image (OpenAI gpt-image-2)           │
│      Examples: "change the sky to sunset",           │
│                "remove the power lines"              │
│                                                      │
│  Need 4K resolution?                                 │
│      → OpenAI: size="3840x2160" (experimental)       │
│        OR Gemini: size="4K"                          │
│                                                      │
│  Need transparent background (alpha)?                │
│      → gpt-image-2 cannot output alpha; generate      │
│        opaque, then remove the background downstream │
└──────────────────────────────────────────────────────┘
```

---

## Reference-image discipline (when continuity matters)

For multi-shot or character-driven work:

1. **First shot** — generate text-only OR with the operator's source-IP image as a reference.
2. **Subsequent shots** — pass the prior approved shot via `reference_images` (Gemini provider, base64-encoded).
3. **Persistent anchor** — keep the first approved shot in the reference set across the whole sequence; chain the immediate predecessor as the secondary ref.
4. **Setting-match ranking** — when multiple refs are available, prefer setting-match over face-clarity. Lighting > setting > composition > face quality.
5. **Validate before sending** — keep each reference within the selected
   model's documented input requirements and category limits.

This protocol is the difference between "three unrelated generations" and "a campaign that holds together." Skip it on multi-shot work and you'll burn iterations chasing identity drift.

---

## Quick provider notes

| Provider | Model | Strengths | Limits |
|---|---|---|---|
| OpenAI | `gpt-image-2` | Strong text rendering, constrained custom sizes, sequential `edit_image` | No transparent output or Google Search; >2K experimental |
| Google | `nano-banana-2` (alias for `gemini-3.1-flash-image`) | 0.5K/1K/2K/4K, reference inputs, Search grounding | Category limits: 10 objects/4 characters |
| Google | `nano-banana-pro` (alias for `gemini-3-pro-image`) | 1K/2K/4K, Thinking mode, up to 14 refs | Category limits: 6 objects/5 characters/3 styles |
| Google | `nano-banana-lite` (alias for `gemini-3.1-flash-lite-image`) | Cost-optimized 1K, up to 14 object refs, SynthID+C2PA | No Google Search; 1K only |

For deeper capability details, model constraints, and published pricing, see this repository's README and provider implementation documentation.

An unavailable explicit provider pin fails closed. In auto mode, hard Gemini
requirements (reference images, Search grounding, and recognized real-time-data
requests) likewise fail when Gemini is unavailable. Soft quality preferences
may fall back to the configured provider with a notice.

---

## Cost awareness (rough, not binding)

| Operation | Approximate cost |
|---|---|
| gpt-image-2, 1024×1024, high quality | $0.211 |
| gpt-image-2, 1024×1536 or 1536×1024, high quality | $0.165 |
| Gemini 3.1 Flash Image, 1K / 2K / 4K | $0.067 / $0.101 / $0.151 |
| Gemini 3 Pro Image, 1K or 2K / 4K | $0.134 / $0.240 |
| Gemini 3.1 Flash Lite Image, 1K | $0.0336 |

These are published examples, not a formula. `estimate_cost` returns
unavailable for undocumented OpenAI custom-size combinations instead of
inventing a pixel multiplier.

For projects with > 30 generations, surface an estimate to the operator before kicking off.

---

## Override the auto-selector

The MCP's auto-selection is heuristic. To pin explicitly:

```python
# Force OpenAI (e.g., for text-heavy content the heuristic missed)
generate_image(prompt="…", provider="openai")

# Force Gemini Pro for highest-fidelity portrait
generate_image(prompt="…", provider="gemini", gemini_model="nano-banana-pro")

# Force a specific model
generate_image(prompt="…", provider="openai", openai_model="gpt-image-1.5")
```

When the operator has expressed a preference, treat it as authoritative — don't override even if your heuristic would choose differently.

Prompt enhancement is opt-in (`ENABLE_PROMPT_ENHANCEMENT=false` by default).
Enabling it adds a separate assistant-model API request, with additional cost
and latency, before the image-generation request.
