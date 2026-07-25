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
  echo "FAIL: tracked files exceed 5 MB — something large got committed"; exit 1
fi
if git ls-files | grep -iqE 'AGENT-BRIEF|CLAUDE\.md|AGENTS\.md|^PLAN\.md|\.claude/'; then
  echo "FAIL: agent scaffolding is tracked"; git ls-files | grep -iE 'AGENT-BRIEF|CLAUDE\.md|AGENTS\.md|^PLAN\.md|\.claude/'; exit 1
fi

# These two checks read ONLY committed state, so they run before anything is generated: a
# generator that overwrites a committed artifact must not be able to mask an orphan number.
echo "==> Asserting every README structure-tree path exists"
missing=0
while read -r p; do
  [ -e "$p" ] || { echo "    MISSING: $p"; missing=1; }
done < <(grep -oE '^(│|├|└|  )*[├└]── [A-Za-z0-9_./-]+' README.md | sed -E 's/.*── //')
[ "$missing" -eq 0 ] || { echo "FAIL: README structure tree does not match reality"; exit 1; }

echo "==> Asserting no README number is orphaned from a measured run"
# Every 0.xxxx in the README must appear in a committed reports/ artifact. The run dirs are
# gitignored, so reports/RESULTS.md is the committed evidence chain.
orphans=0
while read -r n; do
  grep -qr -- "$n" reports/ docs/PROGRESS.md 2>/dev/null || { echo "    ORPHAN: $n"; orphans=1; }
done < <(grep -oE '\b0\.[0-9]{3,4}\b' README.md | sort -u)
[ "$orphans" -eq 0 ] || { echo "FAIL: README contains a number with no measured source"; exit 1; }

echo "==> Running the documented quickstart, verbatim from the README"
uv sync
uv run pytest -q tests/test_smoke.py          # runs on the COMMITTED data/sample/

echo "==> Asserting the smoke run produced real artifacts"
# test_smoke.py runs into a pytest tmp dir by design (so it never clobbers a real run),
# so reproduce the documented `make smoke` invocation here to get runs/latest in the tree.
uv run python train.py -c cfg/smoke.yaml --force >/dev/null
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
print(f'    smoke accuracy = {a:.4f}  (random weights — a plumbing check, not a result)')"

echo "==> Asserting the committed figures are present"
ls docs/images/*.png >/dev/null || { echo "FAIL: no committed figures"; exit 1; }
echo "    $(ls docs/images/*.png | wc -l | tr -d ' ') PNGs committed"

# NOTE: no backticks in these echo strings. Inside double quotes bash treats them as
# command substitution — an earlier version of this line silently ran `make figures` and
# `make report`, which overwrote the committed reports/RESULTS.md inside the clone with the
# smoke run's output and made every real number in the README look orphaned.
echo '==> Asserting the figure and report generators run against a fresh run'
uv run python scripts/export_figures.py -i runs/latest -o "$WORK/figs" --skip-model-figures >/dev/null
uv run python evaluate.py -i runs/latest -o "$WORK/RESULTS.md" >/dev/null
grep -q "McNemar" "$WORK/RESULTS.md" || { echo "FAIL: generated report has no significance test"; exit 1; }

echo "==> Asserting no blocking plt.show() outside an explicit --show gate"
# `git ls-files` rather than `grep -r`: the clone has a .venv by this point (uv sync ran above)
# and scipy/pandas docstrings are full of `>>> plt.show()`. Only OUR tracked sources are in scope.
if git ls-files '*.py' | xargs grep -n 'plt\.show()' | grep -v 'args.show\|if show'; then
  echo "FAIL: unguarded plt.show()"; exit 1
fi

echo "==> Lint + types as documented"
uv run ruff check .
uv run ruff format --check .
uv run mypy .

echo
echo "PASS: fresh clone reproduces the documented quickstart."
