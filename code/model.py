"""model.py

Thin API wrappers for calling LLMs as *code conformance judges*.

This repo intentionally does NOT ship any API keys.
All credentials are read from environment variables (or a local .env file).

Supported providers
-------------------
- OpenAI via `openai` (OpenAI-compatible)
- Google Gemini via `google-generativeai`
- Anthropic Claude via `anthropic`
- OpenRouter via OpenAI-compatible endpoint

Environment variables
---------------------
Required (depending on the model you run):
- OPENAI_API_KEY
- GOOGLE_API_KEY   (or GEMINI_API_KEY)
- ANTHROPIC_API_KEY
- OPENROUTER_API_KEY

Optional:
- OPENROUTER_BASE_URL (default: https://openrouter.ai/api/v1)
- OPENROUTER_HTTP_REFERER
- OPENROUTER_X_TITLE

Security note
-------------
These wrappers send prompts and code to third-party APIs.
Do not include private source code or sensitive data unless you have approval.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()


# ----------------------------
# OpenAI (and OpenAI-compatible)
# ----------------------------


@dataclass
class ModelCallConfig:
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout_s: float = 120.0


class OpenAIModel:
    """Wrapper for OpenAI chat completion models (e.g., gpt-4o)."""

    def __init__(self, model_name: str = "gpt-4o", api_key: Optional[str] = None, config: Optional[ModelCallConfig] = None):
        from openai import OpenAI

        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("Missing OPENAI_API_KEY")

        self.client = OpenAI(api_key=api_key, timeout=(config.timeout_s if config else 120.0))
        self.model_name = model_name
        self.config = config or ModelCallConfig()

    def check_requirement(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()


# ----------------------------
# Google Gemini
# ----------------------------


class GoogleModel:
    """Wrapper for Google Gemini via google-generativeai."""

    def __init__(self, model_name: str = "gemini-2.0-flash", api_key: Optional[str] = None, temperature: float = 0.0):
        import google.generativeai as genai

        api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("Missing GOOGLE_API_KEY (or GEMINI_API_KEY)")

        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(model_name)
        self.model_name = model_name
        self.temperature = temperature

    def check_requirement(self, prompt: str) -> str:
        import google.generativeai as genai

        resp = self.client.generate_content(
            contents=prompt,
            generation_config=genai.GenerationConfig(temperature=self.temperature, top_p=1),
        )
        txt = getattr(resp, "text", None)
        return (txt if txt is not None else str(resp)).strip()


# ----------------------------
# Anthropic Claude
# ----------------------------


class AnthropicModel:
    """Wrapper for Anthropic Claude via anthropic SDK."""

    def __init__(self, model_name: str = "claude-3-5-sonnet-20240620", api_key: Optional[str] = None, max_tokens: int = 4096, temperature: float = 0.0):
        import anthropic

        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("Missing ANTHROPIC_API_KEY")

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature

    def check_requirement(self, prompt: str) -> str:
        resp = self.client.messages.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        # anthropic returns a list of content blocks
        return (resp.content[0].text or "").strip()


# ----------------------------
# OpenRouter (OpenAI-compatible)
# ----------------------------


class OpenRouterModel:
    """OpenRouter wrapper via OpenAI-compatible chat completions API."""

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        timeout_s: float = 120.0,
        max_retries: int = 3,
    ):
        from openai import OpenAI

        api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise EnvironmentError("Missing OPENROUTER_API_KEY")

        base_url = base_url or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

        http_referer = os.getenv("OPENROUTER_HTTP_REFERER")
        x_title = os.getenv("OPENROUTER_X_TITLE")

        default_headers: Dict[str, str] = {}
        if http_referer:
            default_headers["HTTP-Referer"] = http_referer
        if x_title:
            default_headers["X-Title"] = x_title

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers=default_headers if default_headers else None,
            timeout=timeout_s,
            max_retries=max_retries,
        )
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature

    def check_requirement(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()

    def validate_model_endpoints(self) -> List[Dict[str, Any]]:
        """(Optional) Query OpenRouter endpoints for the specified model."""
        import requests
        from urllib.parse import quote

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise EnvironmentError("Missing OPENROUTER_API_KEY")
        if "/" not in self.model_name:
            raise ValueError(f"Invalid OpenRouter model id: {self.model_name}")

        author, slug = self.model_name.split("/", 1)
        url = f"https://openrouter.ai/api/v1/models/{quote(author)}/{quote(slug, safe='')}/endpoints"
        headers = {"Authorization": f"Bearer {api_key}"}

        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        return (r.json().get("data", {}) or {}).get("endpoints", []) or []
