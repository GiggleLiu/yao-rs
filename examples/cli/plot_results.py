#!/usr/bin/env python3
"""Render simple SVG plots for generated CLI result JSON files."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from xml.sax.saxutils import escape


def usage() -> int:
    print("usage: python3 examples/cli/plot_results.py <results-dir> <plots-dir>", file=sys.stderr)
    return 2


def fmt4(value: float) -> str:
    return f"{value:.4f}"


def infer_num_qubits(data: dict, count: int) -> int:
    raw = data.get("num_qubits")
    if isinstance(raw, int) and raw >= 0:
        return raw
    if count <= 1:
        return 0
    return int(math.ceil(math.log2(count)))


def write_probability_svg(stem: str, data: dict, out_path: Path) -> None:
    probabilities = [float(value) for value in data["probabilities"]]
    count = len(probabilities)
    num_qubits = infer_num_qubits(data, count)

    margin_left = 64
    margin_right = 28
    margin_top = 64
    margin_bottom = 88
    chart_h = 220
    bar_w = 24 if count <= 16 else 10
    gap = 8 if count <= 16 else 4
    width = max(560, margin_left + margin_right + count * bar_w + max(0, count - 1) * gap)
    height = margin_top + chart_h + margin_bottom
    baseline = margin_top + chart_h
    scale_max = max(1.0, max(probabilities, default=0.0))

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f"  <title id=\"title\">{escape(stem)} probabilities</title>",
        f"  <desc id=\"desc\">Probability bar chart generated from {escape(stem)}.json.</desc>",
        "  <rect width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>",
        f"  <text x=\"24\" y=\"32\" font-family=\"Arial, sans-serif\" font-size=\"20\" font-weight=\"700\" fill=\"#172033\">{escape(stem)}</text>",
        "  <text x=\"24\" y=\"54\" font-family=\"Arial, sans-serif\" font-size=\"12\" fill=\"#5b6472\">probability distribution</text>",
        f"  <line x1=\"{margin_left}\" y1=\"{baseline}\" x2=\"{width - margin_right}\" y2=\"{baseline}\" stroke=\"#8792a2\" stroke-width=\"1\"/>",
        f"  <line x1=\"{margin_left}\" y1=\"{margin_top}\" x2=\"{margin_left}\" y2=\"{baseline}\" stroke=\"#8792a2\" stroke-width=\"1\"/>",
        f"  <text x=\"{margin_left - 12}\" y=\"{margin_top + 4}\" text-anchor=\"end\" font-family=\"Arial, sans-serif\" font-size=\"11\" fill=\"#5b6472\">1.0000</text>",
        f"  <text x=\"{margin_left - 12}\" y=\"{baseline + 4}\" text-anchor=\"end\" font-family=\"Arial, sans-serif\" font-size=\"11\" fill=\"#5b6472\">0.0000</text>",
    ]

    for index, probability in enumerate(probabilities):
        x = margin_left + index * (bar_w + gap)
        bar_h = 0 if probability <= 0 else max(1.0, probability / scale_max * chart_h)
        y = baseline - bar_h
        label = format(index, f"0{num_qubits}b") if num_qubits > 0 else "0"
        fill = "#2764d8" if probability == max(probabilities) and probability > 0 else "#68a0ff"
        elements.append(
            f"  <rect x=\"{x:.1f}\" y=\"{y:.1f}\" width=\"{bar_w}\" height=\"{bar_h:.1f}\" rx=\"3\" fill=\"{fill}\"/>"
        )
        if probability > 0.001 or count <= 16:
            elements.append(
                f"  <text x=\"{x + bar_w / 2:.1f}\" y=\"{max(14.0, y - 6):.1f}\" text-anchor=\"middle\" font-family=\"Arial, sans-serif\" font-size=\"10\" fill=\"#172033\">{fmt4(probability)}</text>"
            )
        if count <= 16:
            elements.append(
                f"  <text x=\"{x + bar_w / 2:.1f}\" y=\"{baseline + 18}\" text-anchor=\"middle\" font-family=\"Arial, sans-serif\" font-size=\"10\" fill=\"#394150\">{escape(label)}</text>"
            )
        else:
            elements.append(
                f"  <text x=\"{x + bar_w / 2:.1f}\" y=\"{baseline + 18}\" text-anchor=\"middle\" transform=\"rotate(90 {x + bar_w / 2:.1f} {baseline + 18})\" font-family=\"Arial, sans-serif\" font-size=\"9\" fill=\"#394150\">{escape(label)}</text>"
            )

    elements.extend(
        [
            f"  <text x=\"{(margin_left + width - margin_right) / 2:.1f}\" y=\"{height - 18}\" text-anchor=\"middle\" font-family=\"Arial, sans-serif\" font-size=\"12\" fill=\"#5b6472\">basis state</text>",
            "</svg>",
        ]
    )
    out_path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def write_expectation_svg(stem: str, data: dict, out_path: Path) -> None:
    operator = str(data.get("operator", "operator"))
    value = data.get("expectation_value", {})
    re_value = float(value.get("re", 0.0))
    im_value = float(value.get("im", 0.0))
    width = 560
    height = 260
    axis_left = 84
    axis_right = 500
    axis_y = 170
    zero_x = (axis_left + axis_right) / 2
    half_w = (axis_right - axis_left) / 2
    marker_x = zero_x + max(-1.0, min(1.0, re_value)) * half_w

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f"  <title id=\"title\">{escape(stem)} expectation value</title>",
        f"  <desc id=\"desc\">Expectation metric chart generated from {escape(stem)}.json.</desc>",
        "  <rect width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>",
        f"  <text x=\"24\" y=\"34\" font-family=\"Arial, sans-serif\" font-size=\"20\" font-weight=\"700\" fill=\"#172033\">{escape(stem)}</text>",
        f"  <text x=\"24\" y=\"62\" font-family=\"Arial, sans-serif\" font-size=\"15\" fill=\"#394150\">operator: {escape(operator)}</text>",
        f"  <text x=\"24\" y=\"98\" font-family=\"Arial, sans-serif\" font-size=\"32\" font-weight=\"700\" fill=\"#2764d8\">{fmt4(re_value)}</text>",
        f"  <text x=\"134\" y=\"98\" font-family=\"Arial, sans-serif\" font-size=\"13\" fill=\"#5b6472\">real expectation</text>",
        f"  <text x=\"24\" y=\"122\" font-family=\"Arial, sans-serif\" font-size=\"12\" fill=\"#5b6472\">imaginary part {fmt4(im_value)}</text>",
        f"  <line x1=\"{axis_left}\" y1=\"{axis_y}\" x2=\"{axis_right}\" y2=\"{axis_y}\" stroke=\"#8792a2\" stroke-width=\"2\"/>",
        f"  <line x1=\"{axis_left}\" y1=\"{axis_y - 8}\" x2=\"{axis_left}\" y2=\"{axis_y + 8}\" stroke=\"#8792a2\" stroke-width=\"2\"/>",
        f"  <line x1=\"{zero_x:.1f}\" y1=\"{axis_y - 10}\" x2=\"{zero_x:.1f}\" y2=\"{axis_y + 10}\" stroke=\"#394150\" stroke-width=\"2\"/>",
        f"  <line x1=\"{axis_right}\" y1=\"{axis_y - 8}\" x2=\"{axis_right}\" y2=\"{axis_y + 8}\" stroke=\"#8792a2\" stroke-width=\"2\"/>",
        f"  <circle cx=\"{marker_x:.1f}\" cy=\"{axis_y}\" r=\"10\" fill=\"#2764d8\"/>",
        f"  <text x=\"{axis_left}\" y=\"{axis_y + 32}\" text-anchor=\"middle\" font-family=\"Arial, sans-serif\" font-size=\"11\" fill=\"#5b6472\">-1</text>",
        f"  <text x=\"{zero_x:.1f}\" y=\"{axis_y + 32}\" text-anchor=\"middle\" font-family=\"Arial, sans-serif\" font-size=\"11\" fill=\"#5b6472\">0</text>",
        f"  <text x=\"{axis_right}\" y=\"{axis_y + 32}\" text-anchor=\"middle\" font-family=\"Arial, sans-serif\" font-size=\"11\" fill=\"#5b6472\">1</text>",
        "</svg>",
    ]
    out_path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        return usage()

    results_dir = Path(argv[1])
    plots_dir = Path(argv[2])
    if not results_dir.is_dir():
        print(f"results directory does not exist: {results_dir}", file=sys.stderr)
        return 1

    plots_dir.mkdir(parents=True, exist_ok=True)
    for result_path in sorted(results_dir.glob("*.json")):
        data = json.loads(result_path.read_text(encoding="utf-8"))
        out_path = plots_dir / f"{result_path.stem}.svg"
        if "probabilities" in data:
            write_probability_svg(result_path.stem, data, out_path)
        elif "expectation_value" in data:
            write_expectation_svg(result_path.stem, data, out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
