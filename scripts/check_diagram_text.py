#!/usr/bin/env python3
"""Fail if any glyph in an exported diagram falls outside the shape that is supposed to hold it.

Renders every diagram source in the repo (docs/diagrams/*.mmd plus every ```mermaid block in
README.md and docs/**/*.md) to SVG with mermaid-cli, then measures the result geometrically:

  * every label must be a real <text> element. A <foreignObject> label is frozen at the width
    the exporter's headless Chrome measured, and clips the last characters for any reader whose
    font is even slightly wider, which is exactly how ":3070" becomes ":307".
  * every text row is measured with the real font files mermaid asks for (Trebuchet MS, and
    Verdana as the widest common fallback) and must fit inside its container shape.

Usage:
    uv run python scripts/check_diagram_text.py            # render + check the whole repo
    uv run python scripts/check_diagram_text.py FILE.svg   # check already-exported SVGs

Exit code 0 means every character of every label is inside its box in both fonts.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from PIL import ImageFont

SVG = "{http://www.w3.org/2000/svg}"
# MMDC_BIN lets a caller point at an already-installed mermaid-cli instead of npx.
MMDC = shlex.split(os.environ.get("MMDC_BIN", "")) or ["npx", "-y", "@mermaid-js/mermaid-cli"]
FONT_PATHS = {
    "Trebuchet MS": "/System/Library/Fonts/Supplemental/Trebuchet MS.ttf",
    "Verdana": "/System/Library/Fonts/Supplemental/Verdana.ttf",
    "DejaVuSans": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
}
FONTS = {n: p for n, p in FONT_PATHS.items() if Path(p).exists()}
MARGIN = 0.0  # a glyph may touch its box edge, never cross it
_cache: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}


def _font(name: str, size: int) -> ImageFont.FreeTypeFont:
    key = (name, size)
    if key not in _cache:
        _cache[key] = ImageFont.truetype(FONTS[name], size)
    return _cache[key]


def text_width(s: str, size: float) -> float:
    px = max(1, round(size))
    if not FONTS:  # no metric-compatible font on this machine: fall back to a wide estimate
        return len(s) * px * 0.62
    return max(_font(n, px).getlength(s) for n in FONTS)


@dataclass
class Shape:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def area(self) -> float:
        return (self.x1 - self.x0) * (self.y1 - self.y0)


@dataclass
class Row:
    s: str
    cx: float
    y: float
    size: float
    anchor: str

    def span(self, w: float) -> tuple[float, float]:
        if self.anchor == "middle":
            return self.cx - w / 2, self.cx + w / 2
        if self.anchor == "end":
            return self.cx - w, self.cx
        return self.cx, self.cx + w


_TRANSLATE = re.compile(r"translate\(\s*(-?[\d.eE+]+)[,\s]+(-?[\d.eE+]+)?\s*\)")
_FONTSIZE = re.compile(r"font-size:\s*([\d.]+)px")
_RULE = re.compile(r"([^{}]+)\{([^{}]*)\}")


def anchor_rules(svg_text: str) -> list[tuple[list[str], str]]:
    """CSS `text-anchor` rules, as (descendant tokens, value).

    mermaid centres flowchart node labels through a stylesheet rule rather than an attribute,
    while an entity-relationship label is left-aligned by having no rule at all. Reading the
    stylesheet is the only way to know which way a given label is anchored.
    """
    css = svg_text[: svg_text.find("</style>")]
    out: list[tuple[list[str], str]] = []
    for m in _RULE.finditer(css):
        v = re.search(r"text-anchor:\s*(\w+)", m.group(2))
        if not v:
            continue
        for sel in m.group(1).split(","):
            toks = [t for t in re.split(r"\s+|>", sel.strip()) if t]
            toks = [t.split(":")[0] for t in toks if not t.startswith("#")]
            if toks:
                out.append((toks, v.group(1)))
    return out


def css_anchor(rules: list[tuple[list[str], str]], tag: str, classes: list[str]) -> str | None:
    """Resolve the anchor for an element whose ancestor classes are `classes` (self last)."""
    hit = None
    own = set(classes[-1].split()) if classes else set()
    for toks, value in rules:
        last = toks[-1]
        if last.startswith("."):
            if last[1:] not in own:
                continue
        elif last != tag:
            continue
        pool = set()
        for c in classes:
            pool.update(c.split())
        if all(t[1:] in pool for t in toks[:-1] if t.startswith(".")):
            hit = value
    return hit


def _offset(el: ET.Element) -> tuple[float, float]:
    m = _TRANSLATE.search(el.get("transform") or "")
    if not m:
        return 0.0, 0.0
    return float(m.group(1)), float(m.group(2) or 0.0)


def _num(el: ET.Element, name: str, default: float = 0.0) -> float:
    v = el.get(name)
    try:
        return float(v) if v is not None else default
    except ValueError:
        return default


def _anchor(el: ET.Element, inherited: str) -> str:
    style = el.get("style") or ""
    m = re.search(r"text-anchor:\s*(\w+)", style)
    if m:
        return m.group(1)
    return el.get("text-anchor") or inherited


def _size(el: ET.Element, inherited: float) -> float:
    m = _FONTSIZE.search(el.get("style") or "")
    if m:
        return float(m.group(1))
    v = el.get("font-size")
    if v and v.endswith("px"):
        return float(v[:-2])
    return inherited


def collect(
    el: ET.Element,
    dx: float,
    dy: float,
    shapes: list[Shape],
    rows: list[Row],
    size: float,
    anchor: str,
    rules: tuple[tuple[list[str], str], ...] = (),
    classes: tuple[str, ...] = (),
) -> None:
    ox, oy = _offset(el)
    dx, dy = dx + ox, dy + oy
    size = _size(el, size)
    anchor = _anchor(el, anchor)
    classes = (*classes, el.get("class") or "")
    tag = el.tag

    if tag == f"{SVG}rect":
        w, h = _num(el, "width"), _num(el, "height")
        if w > 1 and h > 1:
            x, y = _num(el, "x") + dx, _num(el, "y") + dy
            shapes.append(Shape(x, y, x + w, y + h))
    elif tag == f"{SVG}polygon":
        pts = [float(v) for v in re.findall(r"-?[\d.]+", el.get("points", ""))]
        if len(pts) >= 4:
            xs, ys = pts[0::2], pts[1::2]
            shapes.append(Shape(min(xs) + dx, min(ys) + dy, max(xs) + dx, max(ys) + dy))
    elif tag in (f"{SVG}ellipse", f"{SVG}circle"):
        rx = _num(el, "rx") or _num(el, "r")
        ry = _num(el, "ry") or _num(el, "r")
        cx, cy = _num(el, "cx") + dx, _num(el, "cy") + dy
        if rx > 1:
            shapes.append(Shape(cx - rx, cy - ry, cx + rx, cy + ry))
    elif tag == f"{SVG}text":
        tx, ty = _num(el, "x") + dx, _num(el, "y") + dy
        short = tag.split("}")[-1]
        if not re.search(r"text-anchor", (el.get("style") or "")) and el.get("text-anchor") is None:
            anchor = css_anchor(list(rules), short, list(classes)) or "start"
        outer = [t for t in el.findall(f"{SVG}tspan") if t.get("x") is not None]
        if outer:
            for t in outer:
                s = "".join(t.itertext()).strip()
                if s:
                    rows.append(Row(s, _num(t, "x") + dx, ty, _size(t, size), _anchor(t, anchor)))
        else:
            s = "".join(el.itertext()).strip()
            if s:
                rows.append(Row(s, tx, ty, size, anchor))
        return  # never descend into a <text>: its tspans are already accounted for

    for child in el:
        collect(child, dx, dy, shapes, rows, size, anchor, rules, classes)


def check_svg(path: Path) -> list[str]:
    raw = path.read_text()
    problems: list[str] = []
    if "<foreignObject" in raw:
        problems.append(
            "labels are <foreignObject>, which clips the last characters of a long label; "
            "render with htmlLabels: false"
        )
    root = ET.fromstring(raw)
    default_size = 16.0
    m = re.search(r"#\w[\w-]*\s+\.nodeLabel[^{}]*{[^}]*?font-size:\s*([\d.]+)px", raw, re.S)
    if m:
        default_size = float(m.group(1))

    shapes: list[Shape] = []
    rows: list[Row] = []
    collect(root, 0.0, 0.0, shapes, rows, default_size, "start", tuple(anchor_rules(raw)), ())

    for row in rows:
        w = text_width(row.s, row.size)
        x0, x1 = row.span(w)
        # the tightest shape that vertically brackets this row and horizontally overlaps it
        holders = [
            s for s in shapes if s.y0 - 2 <= row.y <= s.y1 + 2 and s.x0 < (x0 + x1) / 2 < s.x1
        ]
        if not holders:
            continue  # free-floating label (edge label, message text): nothing to clip against
        box = min(holders, key=lambda s: s.area)
        if x0 < box.x0 + MARGIN or x1 > box.x1 - MARGIN:
            over = max(box.x0 + MARGIN - x0, x1 - (box.x1 - MARGIN))
            problems.append(
                f"{row.s!r} overflows its box by {over:.1f}px "
                f"(row {w:.1f}px wide in a {box.x1 - box.x0:.1f}px shape)"
            )
    return problems


def sources(repo: Path) -> list[tuple[str, str]]:
    """(label, mermaid source) for every diagram the repo publishes."""
    out: list[tuple[str, str]] = []
    for mmd in sorted(repo.glob("docs/diagrams/*.mmd")):
        out.append((str(mmd.relative_to(repo)), mmd.read_text()))
    md_files = [repo / "README.md", *sorted(repo.glob("docs/**/*.md"))]
    for md in md_files:
        if not md.exists():
            continue
        for i, block in enumerate(re.findall(r"```mermaid\n(.*?)```", md.read_text(), re.S)):
            out.append((f"{md.relative_to(repo)}#mermaid-{i + 1}", block))
    return out


def main(argv: list[str]) -> int:
    if argv:
        failed = 0
        for a in argv:
            problems = check_svg(Path(a))
            print(("FAIL  " if problems else "PASS  ") + a)
            for p in problems:
                print("        " + p)
            failed += bool(problems)
        return 1 if failed else 0

    repo = Path(__file__).resolve().parent.parent
    failed = 0
    with tempfile.TemporaryDirectory() as tmp:
        for label, src in sources(repo):
            stem = re.sub(r"[^\w.-]", "_", label)
            mmd = Path(tmp) / f"{stem}.mmd"
            svg = Path(tmp) / f"{stem}.svg"
            mmd.write_text(src)
            r = subprocess.run(
                [*MMDC, "-i", str(mmd), "-o", str(svg), "-b", "white"],
                capture_output=True,
                text=True,
            )
            if r.returncode != 0 or not svg.exists():
                print(
                    f"FAIL  {label}\n        mermaid-cli could not render it:\n"
                    f"        {r.stderr.strip()[:400]}"
                )
                failed += 1
                continue
            problems = check_svg(svg)
            print(("FAIL  " if problems else "PASS  ") + label)
            for p in problems:
                print("        " + p)
            failed += bool(problems)
    print(f"\n{'FAILED' if failed else 'OK'}: {failed} diagram(s) with text outside their box")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
