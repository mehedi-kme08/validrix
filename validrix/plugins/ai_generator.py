"""
ai_generator.py — AI-powered test case generator.

Design decision: Prompt engineering over fine-tuning.
  WHY: Fine-tuning requires labelled datasets, retraining pipelines, and
       ongoing maintenance as pytest APIs evolve.  A well-crafted system
       prompt with few-shot examples produces high-quality, up-to-date code
       using a frozen foundation model — zero retraining required.

  Alternatives considered:
    - Template-based generation (string substitution): fast but brittle;
      cannot infer test structure from natural language.
    - LangChain / LlamaIndex: adds ~200 MB of transitive deps for features
      we don't need. We call the Claude API directly to keep deps minimal.

  Tradeoffs:
    - API call adds latency (~3-8 s per generation). Acceptable for a
      developer-time tool; unacceptable in a hot test path.
    - Generated code quality depends on prompt quality. We version-control
      the prompt so improvements are reviewable and auditable.
    - We add a code-fence extractor because LLMs occasionally wrap output
      in markdown — this makes generation robust to model verbosity changes.

  Provider abstraction:
    The _LLMClient ABC lets us swap Anthropic ↔ OpenAI without touching
    the generation logic.  New providers (Gemini, local Ollama) need only
    implement _LLMClient.complete().
"""

from __future__ import annotations

import logging
import re
import textwrap
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import anthropic
import openai

from validrix.core.config_manager import AIConfig, ConfigManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt — versioned here so changes are tracked in git history
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT: Final[str] = textwrap.dedent("""
    You are a senior QA automation engineer specialising in Python and pytest.
    Your task is to generate complete, runnable pytest test functions from a
    plain-English description of the feature under test.

    Rules:
    1. Output ONLY valid Python code — no explanation, no markdown prose.
    2. Wrap all code in a single ```python ... ``` code fence.
    3. Use pytest conventions: test functions start with `test_`, fixtures
       are injected via function parameters.
    4. Include type hints on all function signatures.
    5. Add a one-line docstring per test explaining what it verifies.
    6. Use `pytest.mark` decorators where appropriate
       (@pytest.mark.smoke, @pytest.mark.regression, etc.).
    7. For UI tests, use Playwright's `page` fixture; import it from
       `playwright.sync_api import Page`.
    8. For API tests, use `httpx.Client`.
    9. Group related tests in a plain class (no inheritance required).
    10. Include at least one positive test, one negative test, one edge case.
    11. Use `pytest.mark.parametrize` where it reduces duplication.
    12. Add `# Arrange / # Act / # Assert` comments to structure each test.
""").strip()


# ---------------------------------------------------------------------------
# LLM client abstraction
# ---------------------------------------------------------------------------

class _LLMClient(ABC):
    """Abstract LLM client — swap providers without changing generation logic."""

    @abstractmethod
    def complete(self, system: str, user: str, config: AIConfig) -> str:
        """
        Send a completion request.

        Args:
            system: System prompt setting the model's role.
            user:   User-facing prompt describing the test to generate.
            config: AI configuration (model, tokens, temperature).

        Returns:
            Raw text response from the model.
        """


class _AnthropicClient(_LLMClient):
    """Thin wrapper around the Anthropic Messages API."""

    def complete(self, system: str, user: str, config: AIConfig) -> str:
        client = anthropic.Anthropic(
            api_key=config.anthropic_api_key,
            timeout=config.timeout_seconds,
        )
        message = client.messages.create(
            model=config.model,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return message.content[0].text  # type: ignore[union-attr]


class _OpenAIClient(_LLMClient):
    """Thin wrapper around the OpenAI Chat Completions API."""

    def complete(self, system: str, user: str, config: AIConfig) -> str:
        client = openai.OpenAI(
            api_key=config.openai_api_key,
            timeout=config.timeout_seconds,
        )
        response = client.chat.completions.create(
            model=config.model,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content or ""


def _make_client(config: AIConfig) -> _LLMClient:
    """Factory: return the correct LLM client for the configured provider."""
    if config.provider == "anthropic":
        return _AnthropicClient()
    if config.provider == "openai":
        return _OpenAIClient()
    raise ValueError(f"Unknown AI provider: {config.provider!r}")


# ---------------------------------------------------------------------------
# Generation result
# ---------------------------------------------------------------------------

@dataclass
class GenerationResult:
    """
    Output of a single AI test generation request.

    Attributes:
        description: The plain-English input that was provided.
        code:        Valid Python pytest code, ready to write to a file.
        saved_path:  Path where the code was written, or None if not saved.
        token_usage: Rough token counts for cost tracking.
    """

    description: str
    code: str
    saved_path: Path | None = None
    token_usage: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Standalone generator (used by CLI and programmatic API)
# ---------------------------------------------------------------------------

class AITestGenerator:
    """
    Generates pytest test functions from plain-English descriptions.

    This is the primary Validrix AI feature. It exposes both a programmatic
    API (``generate()``) and integrates with the CLI (``validrix generate``).

    Example::

        gen = AITestGenerator()
        result = gen.generate(
            description="Login page with email and password fields",
            output_path=Path("tests/test_login.py"),
        )
        print(result.code)
    """

    def __init__(self, ai_config: AIConfig | None = None) -> None:
        cfg = ConfigManager.load()
        self._ai_config = ai_config or cfg.ai
        self._client = _make_client(self._ai_config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        description: str,
        output_path: Path | None = None,
        extra_context: str = "",
    ) -> GenerationResult:
        """
        Generate pytest test functions for the given feature description.

        Args:
            description:   Plain English description of the feature/page/API.
            output_path:   If provided, write the generated code to this file.
            extra_context: Additional instructions appended to the user prompt.

        Returns:
            GenerationResult with the generated code and metadata.

        Raises:
            anthropic.APIError | openai.APIError: On provider failures.
            ValueError: If the response contains no extractable Python code.
        """
        logger.info("Generating tests for: %r", description)

        user_prompt = self._build_user_prompt(description, extra_context)
        raw_response = self._client.complete(
            system=_SYSTEM_PROMPT,
            user=user_prompt,
            config=self._ai_config,
        )

        code = self._extract_code(raw_response)
        code = self._add_file_header(code, description)

        result = GenerationResult(description=description, code=code)

        if output_path:
            result.saved_path = self._write_to_file(code, output_path)

        logger.info("Generation complete. Lines: %d", code.count("\n"))
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_user_prompt(description: str, extra_context: str) -> str:
        prompt = (
            f"Generate comprehensive pytest tests for the following feature:\n\n"
            f"FEATURE: {description}\n\n"
            "Requirements:\n"
            "- At minimum: 1 happy-path test, 1 negative test, 1 edge-case test\n"
            "- Use pytest.mark.parametrize for data-driven scenarios\n"
            "- Include fixtures for any shared setup\n"
            "- All assertions must have descriptive messages\n"
        )
        if extra_context:
            prompt += f"\nAdditional context:\n{extra_context}\n"
        return prompt

    @staticmethod
    def _extract_code(raw: str) -> str:
        """Extract Python source from a markdown-fenced code block."""
        fence_match = re.search(
            r"```(?:python)?\s*\n(.*?)```",
            raw,
            re.DOTALL,
        )
        if fence_match:
            return fence_match.group(1).strip()

        stripped = raw.strip()
        if stripped:
            return stripped

        raise ValueError(
            "AI response contained no extractable Python code. "
            f"Raw response (first 200 chars): {raw[:200]!r}"
        )

    @staticmethod
    def _add_file_header(code: str, description: str) -> str:
        header = textwrap.dedent(f"""\
            # =============================================================================
            # AUTO-GENERATED by Validrix AI Generator
            # Feature: {description}
            # Review and adjust before committing to your test suite.
            # =============================================================================

        """)
        return header + code

    @staticmethod
    def _write_to_file(code: str, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(code, encoding="utf-8")
        logger.info("Generated tests written to: %s", path)
        return path


# ---------------------------------------------------------------------------
# pytest plugin wrapper — consumed by pytest11 entry point
# ---------------------------------------------------------------------------

class AIGeneratorPlugin:
    """
    Validrix AI Generator — pytest plugin wrapper.

    The generator itself is invoked via the CLI (``validrix generate``).
    This class satisfies the pytest11 entry-point contract and exposes
    the underlying AITestGenerator for programmatic access.
    """

    #: Expose the generator for programmatic use:  AIGeneratorPlugin.create_generator()
    create_generator = AITestGenerator

    def pytest_configure(self, config: object) -> None:  # noqa: ANN001
        """Register ini options consumed by the generator CLI."""
