#!/usr/bin/env bash
# Prepare dependencies only; never build or run the lecture.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT/.cache/uv}"
uv sync --project "$ROOT" --frozen
uv sync --project "$ROOT/hippo_backend" --frozen
