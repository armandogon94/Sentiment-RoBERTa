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
WORK="$(mktemp -d)"
cleanup() { cd /; rm -rf "$WORK"; }
trap cleanup EXIT INT TERM

echo "==> Cloning committed HEAD into $WORK"
git clone --quiet "$REPO_ROOT" "$WORK/clone"
cd "$WORK/clone"
echo "    HEAD $(git rev-parse --short HEAD)  $(git log -1 --pretty=%s)"

echo "==> Asserting the clone carries no unwanted weight"
tracked_kb="$(git ls-files -z | xargs -0 du -ck 2>/dev/null | tail -1 | cut -f1)"
echo "    tracked size: ${tracked_kb} KB"
if [ "$tracked_kb" -gt 5120 ]; then
  echo "FAIL: tracked files exceed 5 MB, something large got committed"; exit 1
fi
if git ls-files | grep -iqE 'AGENT-BRIEF|CLAUDE\.md|AGENTS\.md|^PLAN\.md|PROGRESS\.md|\.claude/'; then
  echo "FAIL: agent scaffolding is tracked"; git ls-files | grep -iE 'AGENT-BRIEF|CLAUDE\.md|AGENTS\.md|^PLAN\.md|PROGRESS\.md|\.claude/'; exit 1
fi

echo "==> Asserting tracked data contains no contact details"
python3 scripts/check_committed_data.py

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

echo "==> Running the documented quickstart, verbatim from the README"
make setup
make smoke
make test

echo "==> Recomputing every published headline number from committed prediction vectors"
uv run python scripts/check_published_numbers.py

echo "==> Asserting the smoke run produced real artifacts"
test -f runs/latest/metrics.json || { echo "FAIL: no metrics.json"; exit 1; }
uv run python -c "
import json,sys
m=json.load(open('runs/latest/metrics.json'))
a=m.get('accuracy')
assert isinstance(a,(int,float)) and 0.0 < a < 1.0, f'bad accuracy: {a!r}'
assert m['models']['roberta']['random_weights'] is True, 'smoke must not fetch pretrained weights'
for name, block in m['models'].items():
    ci = block['accuracy_ci']
    assert ci['method'] == 'wilson' and ci['low'] <= block['accuracy'] <= ci['high'], name
print(f'    smoke accuracy = {a:.4f}  (random weights, a plumbing check, not a result)')"

echo "==> Asserting the committed figures are present"
expected_figures=(
  attention_from_token.png
  attention_heatmap.png
  baseline_ablation.png
  confusion_matrix_baseline.png
  confusion_matrix_roberta.png
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
echo "    exact eight-file publication set present"

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
python3 scripts/check_no_blocking_show.py

echo "==> Lint + types as documented"
uv run ruff check .
uv run ruff format --check .
uv run mypy .

echo
echo "PASS: fresh clone reproduces the documented quickstart."
