#!/usr/bin/env bash
# Export every Mermaid block in docs/architecture.md to docs/diagrams/*.svg.
#
# The Mermaid source lives inline in the Markdown: GitHub renders it natively, it diffs as text,
# and there is exactly one copy of the truth. This script is how that one copy also becomes an SVG
# for readers outside a Markdown renderer, so the diagrams are never hand-drawn or hand-exported.
#
# Requires mermaid-cli. Set MMDC_BIN to an installed executable to avoid a download:
#   ./scripts/export_diagrams.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$REPO_ROOT/docs/architecture.md"
OUT="$REPO_ROOT/docs/diagrams"
mkdir -p "$OUT"

# Block order in docs/architecture.md -> output filename. Keep in sync with the headings there.
NAMES=(pipeline_dag attribution_sequence)

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT INT TERM

echo "==> Extracting mermaid blocks from ${SRC#"$REPO_ROOT"/}"
awk -v dir="$WORK" '
  /^```mermaid$/ { inblock=1; n++; next }
  /^```$/        { if (inblock) { inblock=0; close(dir "/block_" n ".mmd") } ; next }
  inblock        { print > (dir "/block_" n ".mmd") }
' "$SRC"

count=0
if [[ -n "${MMDC_BIN:-}" ]]; then
  MMDC_CMD=("$MMDC_BIN")
else
  MMDC_CMD=(npx -y @mermaid-js/mermaid-cli@11)
fi

for i in "${!NAMES[@]}"; do
  block="$WORK/block_$((i + 1)).mmd"
  [ -f "$block" ] || { echo "FAIL: docs/architecture.md has no block $((i + 1)) (${NAMES[$i]})"; exit 1; }
  echo "    ${NAMES[$i]}.svg"
  "${MMDC_CMD[@]}" \
    --input "$block" \
    --output "$OUT/${NAMES[$i]}.svg" \
    --backgroundColor white \
    --quiet
  count=$((count + 1))
done

echo "==> wrote $count SVG(s) to ${OUT#"$REPO_ROOT"/}"
