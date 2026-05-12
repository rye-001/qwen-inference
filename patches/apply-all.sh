#!/usr/bin/env bash
#
# apply-all.sh — apply every patches/*.patch in lexical order to the ggml
# source tree. Invoked by CMake's FetchContent PATCH_COMMAND from the
# fetched ggml source directory (cwd).
#
# Idempotent: each patch is git-apply'd only if the dry-run check passes.
# A patch that's already applied (or whose pre-image is missing) is skipped
# with a notice rather than failing the build.
#
# Args:
#   $1  absolute path to the qwen-inference patches/ directory
#
# Exit codes:
#   0  success (one or more patches applied, or none needed)
#   1  hard failure (apply check passed but apply itself failed — corrupted state)

set -euo pipefail

PATCH_DIR="${1:?usage: apply-all.sh <patches-dir>}"
shopt -s nullglob

applied=0
skipped=0

for p in "$PATCH_DIR"/*.patch; do
    name="$(basename "$p")"
    if git apply --check "$p" 2>/dev/null; then
        if git apply "$p"; then
            echo "qinf: applied $name"
            applied=$((applied + 1))
        else
            echo "qinf: ERROR — $name passed --check but failed to apply" >&2
            exit 1
        fi
    else
        echo "qinf: skipping $name (already applied or context mismatch)"
        skipped=$((skipped + 1))
    fi
done

echo "qinf: patch summary — applied=$applied skipped=$skipped"
exit 0
