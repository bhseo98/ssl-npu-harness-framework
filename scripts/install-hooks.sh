#!/usr/bin/env bash
# Install repo-level git hooks. Run once after cloning:
#     bash scripts/install-hooks.sh
#
# Installs:
#   .git/hooks/pre-commit  →  shim that calls scripts/codex-review.sh
#
# Idempotent. Re-running is safe.

set -euo pipefail

REPO_ROOT=$(git rev-parse --show-toplevel)
HOOK="$REPO_ROOT/.git/hooks/pre-commit"
REVIEW_SCRIPT="$REPO_ROOT/scripts/codex-review.sh"

if [[ ! -f "$REVIEW_SCRIPT" ]]; then
    echo "❌ Expected $REVIEW_SCRIPT to exist. Have you pulled the latest?" >&2
    exit 1
fi

chmod +x "$REVIEW_SCRIPT"

# Pre-existing hook handling.
# We must not silently disable any pre-existing pre-commit check (e.g.
# `pre-commit` framework, lint hooks, secret scanners). If we find one,
# move it aside *and* chain it from the new shim so it still runs on
# every commit, ahead of the Codex review.
SHIM_MARKER='scripts/codex-review.sh'
BACKUP=""
if [[ -e "$HOOK" ]]; then
    if grep -q "$SHIM_MARKER" "$HOOK" 2>/dev/null; then
        echo "ℹ  Existing pre-commit hook already points at $SHIM_MARKER — leaving untouched."
        echo "   (Re-run after editing the shim if you actually want to refresh it.)"
        exit 0
    fi
    BACKUP="${HOOK}.bak.$(date -u +%Y%m%dT%H%M%SZ)"
    mv "$HOOK" "$BACKUP"
    chmod +x "$BACKUP"
    echo "⚠  Pre-existing pre-commit hook moved to:"
    echo "      $BACKUP"
    echo "   The new shim will run BOTH (backup → codex review). The"
    echo "   backup must keep its execute bit and its current path; if you"
    echo "   delete it, only Codex review will run."
fi

if [[ -n "$BACKUP" ]]; then
    # Chained shim: existing-hook first, codex review second.
    # Heredoc *without* quotes around 'EOF' so $BACKUP expands now, while
    # the runtime substitutions ($@, $(git rev-parse ...)) are escaped to
    # expand later.
    cat > "$HOOK" <<EOF
#!/usr/bin/env bash
# Installed by scripts/install-hooks.sh.
# Chains the previously-existing pre-commit hook with the Codex review.
# Either check failing aborts the commit; both must pass.
set -e
if [[ -x "$BACKUP" ]]; then
    "$BACKUP" "\$@"
fi
exec "\$(git rev-parse --show-toplevel)/scripts/codex-review.sh" "\$@"
EOF
else
    # Fresh install: codex review only.
    cat > "$HOOK" <<'EOF'
#!/usr/bin/env bash
# Installed by scripts/install-hooks.sh — DO NOT EDIT here; edit the
# script under version control: scripts/codex-review.sh.
exec "$(git rev-parse --show-toplevel)/scripts/codex-review.sh" "$@"
EOF
fi
chmod +x "$HOOK"

echo "✅ Installed pre-commit hook → $HOOK"
echo "   Reviews are written to .codex-reviews/ (gitignored)."
echo "   Bypass once: git commit --no-verify"
echo ""
echo "Sanity check:"
echo "  command -v codex     →  $(command -v codex 2>/dev/null || echo '(not on PATH — install from https://github.com/openai/codex/releases)')"
if command -v codex >/dev/null 2>&1; then
    echo "  codex --version       →  $(codex --version 2>/dev/null || echo '?')"
    if [[ -f "$HOME/.codex/auth.json" ]]; then
        echo "  ~/.codex/auth.json    →  present"
    else
        echo "  ~/.codex/auth.json    →  MISSING (run 'codex login' once)"
    fi
fi
