#!/usr/bin/env bash
# Clone THIS repo's committed HEAD into a throwaway directory and run the documented
# quickstart exactly as a stranger would. Cleans up after itself, always.
#
# The point is that ONLY COMMITTED FILES EXIST in the clone. Anything untracked that the
# quickstart needs is a bug, and this script is how that bug is found before a reader does.
#
# It must never be made to pass by weakening a check. If the README is wrong, fix the README
# and re-run the fixed version.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-committed}"
WORK="$(mktemp -d)"
cleanup() { cd /; rm -rf "$WORK"; }
trap cleanup EXIT INT TERM

if [ "$MODE" = "committed" ]; then
  echo "==> Cloning committed HEAD into $WORK"
  git clone --quiet "$REPO_ROOT" "$WORK/clone"
elif [ "$MODE" = "--working-tree" ]; then
  echo "==> Copying the prospective public working tree into $WORK"
  uv run --project "$REPO_ROOT" --no-sync python - "$REPO_ROOT" "$WORK/clone" <<'PY'
import os
import shutil
import subprocess
import sys
from pathlib import Path

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
destination.mkdir()
result = subprocess.run(
    ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
    cwd=source,
    check=True,
    capture_output=True,
)
paths = sorted(
    Path(os.fsdecode(value))
    for value in result.stdout.split(b"\0")
    if value and (source / os.fsdecode(value)).exists()
)
for relative in paths:
    source_path = source / relative
    target_path = destination / relative
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if source_path.is_symlink():
        target_path.symlink_to(os.readlink(source_path))
    else:
        shutil.copy2(source_path, target_path)
(destination / ".public-files").write_text(
    "".join(f"{path.as_posix()}\n" for path in paths),
    encoding="utf-8",
)
PY
  export PUBLIC_FILE_MANIFEST="$WORK/clone/.public-files"
else
  echo "usage: $0 [--working-tree]"
  exit 2
fi
cd "$WORK/clone"
if [ "$MODE" = "committed" ]; then
  echo "    HEAD $(git rev-parse --short HEAD)  $(git log -1 --pretty=%s)"
else
  echo "    prospective tree copied without .git"
fi

echo "==> Asserting the clone carries no unwanted weight"
if [ "$MODE" = "committed" ]; then
  tracked_kb="$(git ls-files -z | xargs -0 du -ck 2>/dev/null | tail -1 | cut -f1)"
  scaffolding="$(git ls-files)"
else
  tracked_kb="$(tr '\n' '\0' < .public-files | xargs -0 du -ck 2>/dev/null | tail -1 | cut -f1)"
  scaffolding="$(cat .public-files)"
fi
echo "    tracked size: ${tracked_kb} KB"
if [ "$tracked_kb" -gt 5120 ]; then
  echo "FAIL: tracked files exceed 5 MB, something large got committed"; exit 1
fi
if printf '%s\n' "$scaffolding" | grep -iqE 'AGENT-BRIEF|CLAUDE\.md|AGENTS\.md|^PLAN\.md|PROGRESS\.md|\.claude/'; then
  echo "FAIL: agent scaffolding is public"
  printf '%s\n' "$scaffolding" | grep -iE 'AGENT-BRIEF|CLAUDE\.md|AGENTS\.md|^PLAN\.md|PROGRESS\.md|\.claude/'
  exit 1
fi

echo "==> Installing the exact locked environment"
make setup

echo "==> Asserting tracked data contains no contact details"
uv run python scripts/check_committed_data.py

# This check reads only committed state, before any generator can change the clone.
echo "==> Asserting every README structure-tree path exists"
structure_paths="$WORK/structure-paths.txt"
# NOT a bracket expression over the box-drawing characters. `[├└]` is matched byte-wise in the C
# locale, so the previous pattern found 15 paths under zsh and zero under bash, and because the
# result was only checked for existence, the zero case had been silently passing as "all paths
# exist". Anchoring on the literal "── " is byte-safe in either locale.
sed -n -E 's/^[^A-Za-z0-9]*── ([A-Za-z0-9_./-]+).*/\1/p' README.md > "$structure_paths" || true
[ -s "$structure_paths" ] || {
  echo "FAIL: README structure-tree extraction found zero paths"; exit 1
}
missing=0
while read -r p; do
  [ -e "$p" ] || { echo "    MISSING: $p"; missing=1; }
done < "$structure_paths"
[ "$missing" -eq 0 ] || { echo "FAIL: README structure tree does not match reality"; exit 1; }

echo "==> Asserting local Markdown links and image sources resolve"
uv run python scripts/check_markdown_links.py

echo "==> Running the documented quickstart, verbatim from the README"
make smoke
cp reports/evidence/quality.json "$WORK/quality.json"
make quality-evidence
# Statement coverage is platform dependent, so a byte comparison was too strict:
# utils/device.py branches on whether MPS exists. scripts/check_quality_drift.py
# holds the displayed percent, the statement total and the suite counts exactly,
# and allows only the raw line counts to move.
uv run python scripts/check_quality_drift.py \
  "$WORK/quality.json" reports/evidence/quality.json || {
  echo "FAIL: regenerated quality evidence differs from the committed artifact"
  diff -u "$WORK/quality.json" reports/evidence/quality.json || true
  uv run coverage report -m || true
  exit 1
}
# The comparison above is the point of regenerating; the published artifact is the
# committed one. Restore its exact bytes so the evidence digests in SHA256SUMS,
# which cover every file in reports/evidence, still describe what is committed.
cp "$WORK/quality.json" reports/evidence/quality.json

echo "==> Recomputing every published headline number from committed source arrays"
uv run python scripts/check_published_numbers.py

echo "==> Asserting the smoke run produced real artifacts"
test -f runs/latest/metrics.json || { echo "FAIL: no metrics.json"; exit 1; }
uv run python -c "
import json
from pathlib import Path
m=json.loads(Path('runs/latest/metrics.json').read_text(encoding='utf-8'))
a=m.get('accuracy')
assert isinstance(a,(int,float)) and 0.0 < a < 1.0, f'bad accuracy: {a!r}'
assert m['models']['roberta']['random_weights'] is True, 'smoke must not fetch pretrained weights'
for name, block in m['models'].items():
    ci = block['accuracy_ci']
    assert ci['method'] == 'wilson' and ci['low'] <= block['accuracy'] <= ci['high'], name
print(f'    smoke accuracy = {a:.4f}  (random weights, a plumbing check, not a result)')"

echo "==> Asserting the committed figures are present"
expected_figures=(
  attention_entropy_atlas.png
  attention_from_token.png
  attention_heatmap.png
  baseline_ablation.png
  confusion_matrix_baseline.png
  confusion_matrix_roberta.png
  embedding_space_3d.png
  layer_probe_accuracy.png
  saliency_negative.png
  saliency_positive.png
  training_curves.png
)
for name in "${expected_figures[@]}"; do
  test -f "docs/images/$name" || { echo "FAIL: missing docs/images/$name"; exit 1; }
done
figure_count="$(find docs/images -maxdepth 1 -type f -name '*.png' | wc -l | tr -d ' ')"
[ "$figure_count" -eq "${#expected_figures[@]}" ] || {
  echo "FAIL: docs/images contains $figure_count PNGs; expected exactly ${#expected_figures[@]}"
  exit 1
}
echo "    exact eleven-file publication set present"

echo '==> Regenerating published figure data from evidence and checking all tracked figures'
uv run python scripts/check_published_figures.py

echo '==> Regenerating the published report from evidence and asserting byte identity'
uv run python evaluate.py -i reports/evidence/run_2 -a reports/evidence/run_3 \
  -s reports/evidence/run_5 -o "$WORK/RESULTS.md"
if ! cmp -s "$WORK/RESULTS.md" reports/RESULTS.md; then
  echo "FAIL: regenerated reports/RESULTS.md differs from the committed file"
  diff -u reports/RESULTS.md "$WORK/RESULTS.md" || true
  exit 1
fi
echo "    reports/RESULTS.md is byte-identical"

echo '==> Asserting no blocking plt.show() outside an explicit --show gate'
# One implementation of the rule, shared with CI and with the test suite. A grep here cannot
# tell a call from a docstring, and both this repo and scipy discuss plt.show() in prose.
uv run python scripts/check_no_blocking_show.py

echo "==> Lint + types as documented"
uv run ruff check .
uv run ruff format --check .
uv run mypy .

echo
echo "PASS: fresh clone reproduces the documented quickstart."
