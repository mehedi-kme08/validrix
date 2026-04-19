# =============================================================================
# Validrix — multi-stage Dockerfile
#
# Design decision: Multi-stage build.
#   WHY: The builder stage installs system-level tools (gcc, build headers)
#        needed to compile some Python deps (e.g., pydantic's Rust core).
#        The final stage copies only the installed packages, producing a lean
#        image with no build tools — smaller attack surface, faster pulls.
#
#   Alternatives considered:
#     - Single stage with apt-get cleanup: still includes build tools in the
#       image layers even after rm; multi-stage is cleaner.
#     - Alpine base: smaller image but glibc/musl mismatch causes issues with
#       Playwright's pre-compiled Chromium binaries.
#
#   Base image rationale:
#     python:3.11-slim-bookworm — Debian Bookworm is LTS, "slim" excludes
#     docs and optional locale data while retaining glibc, which Playwright
#     requires for its bundled browser binaries.
# =============================================================================

# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim-bookworm AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy project manifest first — Docker cache layer stays valid until toml changes
COPY pyproject.toml ./
COPY README.md LICENSE ./
COPY validrix/ ./validrix/

# Install into a prefix so we can copy it to the final stage
RUN pip install --no-cache-dir --prefix=/install ".[dev]"


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim-bookworm AS runtime

LABEL org.opencontainers.image.title="Validrix"
LABEL org.opencontainers.image.description="AI-Powered PyTest Plugin Framework"
LABEL org.opencontainers.image.source="https://github.com/mehedi-kme08/validrix"

# Playwright's Chromium needs these system libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Chromium runtime deps
    libnss3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libgtk-3-0 \
    # Convenience
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy source and tests
COPY validrix/ ./validrix/
COPY tests/ ./tests/
COPY pyproject.toml ./

# Install Playwright browsers — cached in its own layer so code changes
# don't re-download ~300 MB of browser binaries
RUN playwright install chromium --with-deps

# Non-root user for security
RUN useradd --create-home --shell /bin/bash validrix
USER validrix

# Health check — verify pytest is importable
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import pytest; import validrix" || exit 1

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VALIDRIX_ENVIRONMENT=dev

# Default: run the full test suite
CMD ["pytest", "--tb=short", "-ra"]
