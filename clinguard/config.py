"""
Central model configuration + lazy chat-model factory.

Resolves (provider, model) per agent role from env with defaults, and builds
the LangChain chat client on demand — never at import — so every module can be
imported and unit-tested without any API key present. This also unblocks
per-role cross-model routing (pipeline on Claude, judge/verifier on OpenAI).
"""

import os

from dotenv import load_dotenv

load_dotenv()

# Central model IDs. Keys are read from the environment only (never hardcoded,
# never Claude Code's own ANTHROPIC key) — fail clearly if one is missing.
_OPENAI_MODEL = "gpt-4o-mini"
_CLAUDE_HAIKU = "claude-haiku-4-5"

# Role -> (provider, model, max_tokens). Cross-model by design to avoid
# self-evaluation bias: pipeline (extractor, classifier/grader) on Haiku-tier
# Claude; verifier and judge on OpenAI gpt-4o-mini. max_tokens is capped tight
# on every call for spend control (~$2/provider budget).
_DEFAULTS = {
    "extractor":      ("anthropic", _CLAUDE_HAIKU, 512),
    "classifier":     ("anthropic", _CLAUDE_HAIKU, 512),
    # Cross-model verifier: OpenAI on purpose — a different model from the Claude
    # grader, so agreement is a genuine independent check, not self-confirmation.
    "verifier":       ("openai",    _OPENAI_MODEL, 384),
    "judge":          ("openai",    _OPENAI_MODEL, 512),
    # Synthetic test-set generation — OpenAI on purpose: labels stay trustworthy
    # only if the grader (Claude) never sees its own phrasing.
    "generator":      ("openai",    _OPENAI_MODEL, 256),
}

_ENV_KEY = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}

# Cache one client per (provider, model, temperature, max_tokens).
_CACHE: dict = {}


def _resolve(role: str) -> tuple[str, str, int]:
    d_provider, d_model, d_max = _DEFAULTS.get(role, ("openai", _OPENAI_MODEL, 512))
    provider = os.getenv(f"CLINGUARD_{role.upper()}_PROVIDER", d_provider)
    model = os.getenv(f"CLINGUARD_{role.upper()}_MODEL", d_model)
    max_tokens = int(os.getenv(f"CLINGUARD_{role.upper()}_MAX_TOKENS", d_max))
    return provider, model, max_tokens


def _require_key(provider: str) -> None:
    env_key = _ENV_KEY.get(provider)
    if env_key and not os.getenv(env_key):
        raise RuntimeError(
            f"{env_key} is not set but is required for provider '{provider}'. "
            f"Add it to your .env before running the pipeline."
        )


def get_chat_model(role: str, temperature: float = 0.0):
    """
    Return a lazily-constructed LangChain chat client for the given role.

    Reads CLINGUARD_<ROLE>_PROVIDER / _MODEL / _MAX_TOKENS env overrides,
    falling back to _DEFAULTS. Every call is capped with max_tokens for spend
    control. Raises RuntimeError with a clear message if the needed API key is
    missing. Clients are cached per (provider, model, temperature, max_tokens).
    """
    provider, model, max_tokens = _resolve(role)
    key = (provider, model, temperature, max_tokens)
    if key in _CACHE:
        return _CACHE[key]

    _require_key(provider)

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        client = ChatOpenAI(model=model, temperature=temperature, max_tokens=max_tokens)
    elif provider == "anthropic":
        # Claude 4.x adaptive-thinking models reject non-default sampling
        # params, so temperature is intentionally not forwarded here.
        from langchain_anthropic import ChatAnthropic
        client = ChatAnthropic(model=model, max_tokens=max_tokens)
    else:
        raise ValueError(f"Unknown provider '{provider}' for role '{role}'")

    _CACHE[key] = client
    return client
