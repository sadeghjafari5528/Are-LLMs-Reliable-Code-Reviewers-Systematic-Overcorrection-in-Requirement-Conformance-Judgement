"""factory.py

Small helpers to build model wrappers by name.

This file contains no credentials. Configure API keys via environment variables.
"""

from __future__ import annotations

from typing import List

from model import AnthropicModel, GoogleModel, OpenAIModel, OpenRouterModel


DEFAULT_MODELS: List[str] = [
    "gpt-4o",
    "gemini-2.0-flash",
    "claude-haiku-4-5",
    "meta-llama/llama-3.1-8b-instruct",
    "mistralai/mistral-small-3.2-24b-instruct",
]


def build_model(model_name: str):
    """Return the appropriate wrapper for the given model name."""
    if model_name == "gpt-4o" or model_name.startswith("gpt-"):
        return OpenAIModel(model_name=model_name)
    if model_name.startswith("gemini"):
        return GoogleModel(model_name=model_name)
    if model_name.startswith("claude"):
        return AnthropicModel(model_name=model_name)
    if "/" in model_name:
        return OpenRouterModel(model_name=model_name)
    raise ValueError(f"Unknown model: {model_name}")
