#!/usr/bin/env python3
"""Convert qflex-style circuit .txt files to yao-rs native circuit JSON."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ComplexPair = list[float]


def convert_text(text: str) -> dict[str, Any]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("empty circuit file")

    declared_qubits = int(lines[0])
    qubit_map: dict[int, int] = {}
    gates: list[tuple[int, int, dict[str, Any]]] = []

    for seq, line in enumerate(lines[1:]):
        layer, element = parse_gate_line(line, qubit_map)
        gates.append((layer, seq, element))

    gates.sort(key=lambda item: (item[0], item[1]))
    num_qubits = len(qubit_map) if qubit_map else declared_qubits
    return {
        "num_qubits": num_qubits,
        "elements": [element for _, _, element in gates],
    }


def convert_file(input_path: Path | str, output_path: Path | str) -> None:
    input_path = Path(input_path)
    output_path = Path(output_path)
    circuit = convert_text(input_path.read_text(encoding="utf-8"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(circuit, indent=2) + "\n", encoding="utf-8")


def parse_gate_line(line: str, qubit_map: dict[int, int]) -> tuple[int, dict[str, Any]]:
    words = [word for word in re.split(r"[(),\s]+", line.strip()) if word]
    if len(words) < 3:
        raise ValueError(f"invalid gate line: {line!r}")

    layer = int(words[0])
    gate_name = words[1]
    if gate_name == "rz":
        if len(words) != 4:
            raise ValueError(f"rz line must be 'layer rz(theta) qubit': {line!r}")
        theta = float(words[2])
        targets = remap_targets([int(words[3])], qubit_map)
        element = gate("Rz", targets, params=[theta])
    elif gate_name == "fsim":
        if len(words) != 6:
            raise ValueError(f"fsim line must be 'layer fsim(theta, phi) q0 q1': {line!r}")
        theta = float(words[2])
        phi = float(words[3])
        targets = remap_targets([int(words[4]), int(words[5])], qubit_map)
        element = gate("FSim", targets, params=[theta, phi])
    else:
        targets = remap_targets([int(word) for word in words[2:]], qubit_map)
        element = fixed_gate(gate_name, targets)

    return layer, element


def remap_targets(raw_targets: list[int], qubit_map: dict[int, int]) -> list[int]:
    mapped = []
    for raw in raw_targets:
        if raw not in qubit_map:
            qubit_map[raw] = len(qubit_map)
        mapped.append(qubit_map[raw])
    return mapped


def fixed_gate(name: str, targets: list[int]) -> dict[str, Any]:
    match name:
        case "h":
            require_targets(name, targets, 1)
            return gate("H", targets)
        case "x":
            require_targets(name, targets, 1)
            return gate("X", targets)
        case "y":
            require_targets(name, targets, 1)
            return gate("Y", targets)
        case "z":
            require_targets(name, targets, 1)
            return gate("Z", targets)
        case "s":
            require_targets(name, targets, 1)
            return gate("S", targets)
        case "t":
            require_targets(name, targets, 1)
            return gate("T", targets)
        case "x_1_2":
            require_targets(name, targets, 1)
            return gate("SqrtX", targets)
        case "y_1_2":
            require_targets(name, targets, 1)
            return gate("SqrtY", targets)
        case "hz_1_2":
            require_targets(name, targets, 1)
            return gate(
                "Custom",
                targets,
                label="HZ12",
                is_diagonal=False,
                matrix=[
                    [c(0.5, 0.0), c(0.5, 0.5)],
                    [c(0.5, -0.5), c(0.5, 0.0)],
                ],
            )
        case "cx":
            require_targets(name, targets, 2)
            return gate("X", [targets[1]], controls=[targets[0]])
        case "cz":
            require_targets(name, targets, 2)
            return gate("Z", [targets[1]], controls=[targets[0]])
        case "swap":
            require_targets(name, targets, 2)
            return gate("SWAP", targets)
        case "i":
            require_targets(name, targets, 1)
            return gate(
                "Custom",
                targets,
                label="I",
                is_diagonal=True,
                matrix=[
                    [c(1.0, 0.0), c(0.0, 0.0)],
                    [c(0.0, 0.0), c(1.0, 0.0)],
                ],
            )
        case _:
            raise ValueError(f"unknown qflex gate: {name}")


def gate(
    name: str,
    targets: list[int],
    *,
    controls: list[int] | None = None,
    params: list[float] | None = None,
    label: str | None = None,
    is_diagonal: bool | None = None,
    matrix: list[list[ComplexPair]] | None = None,
) -> dict[str, Any]:
    element: dict[str, Any] = {
        "type": "gate",
        "gate": name,
        "targets": targets,
    }
    if controls is not None:
        element["controls"] = controls
    if params is not None:
        element["params"] = params
    if label is not None:
        element["label"] = label
    if is_diagonal is not None:
        element["is_diagonal"] = is_diagonal
    if matrix is not None:
        element["matrix"] = matrix
    return element


def c(re: float, im: float) -> ComplexPair:
    return [re, im]


def require_targets(name: str, targets: list[int], expected: int) -> None:
    if len(targets) != expected:
        raise ValueError(f"gate {name} requires {expected} targets, got {len(targets)}")


def convert_directory(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for input_path in sorted(input_dir.glob("*.txt")):
        output_path = output_dir / f"{input_path.stem}.json"
        convert_file(input_path, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="qflex .txt file or directory")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output JSON file, or output directory when input is a directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.input.is_dir():
        output_dir = args.output or args.input / "yao-json"
        convert_directory(args.input, output_dir)
    else:
        output_path = args.output or args.input.with_suffix(".json")
        convert_file(args.input, output_path)


if __name__ == "__main__":
    main()
