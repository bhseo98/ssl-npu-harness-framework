#!/usr/bin/env bash
# Pre-commit cross-check via OpenAI Codex CLI.
#
# Behavior:
#   - reads `git diff --cached` (staged changes) + a project-aware prompt
#   - pipes them to `codex exec --sandbox read-only --ephemeral`
#   - first line of the verdict determines exit code:
#       🟢 LGTM    → exit 0 (commit proceeds)
#       🟡 NIT     → exit 0 (commit proceeds, with note)
#       🔴 BLOCK   → exit 1 (commit aborted; override with --no-verify)
#   - verdict saved to `.codex-reviews/<UTC-timestamp>.md` (gitignored)
#
# Fail-open: if `codex` is missing, auth is broken, or the network fails,
# we print a warning and exit 0 — offline development must not be blocked.

set -u
set -o pipefail

# Ensure user-local bin (where ~/.local/bin/codex lives) is visible to the hook.
export PATH="$HOME/.local/bin:$PATH"

REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null) || {
    echo "ℹ️  codex-review.sh: not inside a git repo, skipping." >&2
    exit 0
}
cd "$REPO_ROOT"

# Skip when nothing is staged.
if git diff --cached --quiet; then
    echo "ℹ️  codex-review.sh: no staged changes — skipping." >&2
    exit 0
fi

# Fail-open if codex CLI is unavailable.
if ! command -v codex >/dev/null 2>&1; then
    echo "⚠  codex CLI not on PATH — skipping cross-check (fail-open)." >&2
    echo "   Install: curl-download from https://github.com/openai/codex/releases" >&2
    exit 0
fi

# Hard block on accidental secret staging — independent of Codex.
SECRET_FILES=$(git diff --cached --name-only | grep -E 'auth\.json|^\.env(\.|$)|\.codex/(auth|config)|^id_rsa|\.pem$' || true)
if [[ -n "$SECRET_FILES" ]]; then
    echo "🔴 BLOCK: secret-looking files in staged set:" >&2
    echo "$SECRET_FILES" >&2
    echo "Override (NOT recommended): git commit --no-verify" >&2
    exit 1
fi

mkdir -p "$REPO_ROOT/.codex-reviews"
TS=$(date -u +%Y%m%dT%H%M%SZ)
VERDICT_FILE="$REPO_ROOT/.codex-reviews/${TS}.md"

PROMPT_HEADER='You are reviewing a git diff in `target-app` — a voice AI evaluation
framework targeting a 2 GB embedded edge device.

Project constraints (verbatim):
- DRAM budget: 2 GB total for STT + LLM + TTS combined
- Python / PyTorch first; will eventually lower to MLIR (so PyTorch
  constructs that are hostile to MLIR / torch-mlir lowering are concerning)
- Model swap via dict-driven registry must stay trivial
  (see `src/target_app/registry.py` and `interfaces.py`)
- Must still run CPU-only on a virtual platform

Review goals, in order:
  1. Functional regressions introduced by the staged diff
  2. Anything that breaks the registry-based model-swap contract
  3. Modified code that would not survive PyTorch → MLIR lowering
  4. Anything pushing pipeline RAM above 2 GB at runtime
  5. Style nits (only after the above)

OUTPUT FORMAT — these rules are strict; the wrapper script parses line 1:
  • First line MUST start with exactly one of:
        🟢 LGTM
        🟡 NIT
        🔴 BLOCK
  • Then up to 3 short bullets, each citing file:line in the diff.
  • Do not output anything before the verdict line.

The staged diff follows.'

DIFF=$(git diff --cached)

echo "🔎  codex-review: $(git diff --cached --stat | tail -1)" >&2

set +e
printf '%s\n\n=== STAGED DIFF ===\n%s\n' "$PROMPT_HEADER" "$DIFF" \
    | codex exec \
        --sandbox read-only \
        --ephemeral \
        --color never \
        --output-last-message "$VERDICT_FILE" \
        --cd "$REPO_ROOT" \
        - >/dev/null 2>&1
CODEX_EXIT=$?
set -e

if [[ $CODEX_EXIT -ne 0 || ! -s "$VERDICT_FILE" ]]; then
    echo "⚠  codex exec failed (exit=$CODEX_EXIT) — fail-open." >&2
    echo "   (Common causes: not authenticated → run 'codex login', or no network.)" >&2
    exit 0
fi

# Show the verdict to the developer.
echo "" >&2
echo "──────── Codex cross-check verdict ────────" >&2
cat "$VERDICT_FILE" >&2
echo "───────────────────────────────────────────" >&2
echo "💾  full verdict: $VERDICT_FILE" >&2
echo "" >&2

FIRST_LINE=$(head -n1 "$VERDICT_FILE")
# Anchor at start of line so "BLOCKED" inside a sentence does NOT match.
# An emoji prefix is permitted ("🔴 BLOCK", "🟢 LGTM", "🟡 NIT") as well as
# the bare label without the emoji.
case "$FIRST_LINE" in
    "🔴 BLOCK"*|"🔴BLOCK"*|"BLOCK"*)
        echo "🛑  commit BLOCKED by Codex. Override with: git commit --no-verify" >&2
        exit 1
        ;;
    "🟢 LGTM"*|"🟢LGTM"*|"LGTM"*|"🟡 NIT"*|"🟡NIT"*|"NIT"*)
        exit 0
        ;;
    *)
        echo "⚠  Codex returned an unrecognized verdict format:" >&2
        echo "    first line: \"$FIRST_LINE\"" >&2
        echo "   Treating as fail-open (commit allowed) — set CODEX_REVIEW_STRICT=1" >&2
        echo "   in the environment to flip this to fail-closed (block on bad parse)." >&2
        if [[ "${CODEX_REVIEW_STRICT:-0}" == "1" ]]; then
            exit 1
        fi
        exit 0
        ;;
esac
