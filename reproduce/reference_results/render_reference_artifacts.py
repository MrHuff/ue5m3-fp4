#!/usr/bin/env python3
# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render report-ready figures and compact tables from sanitized results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as error:  # pragma: no cover - exercised by the documented environment.
    raise SystemExit("rendering reference artifacts requires matplotlib") from error

STEPS = tuple(range(2_500, 30_001, 2_500))
ORDER = (
    "bf16",
    "ue5m3_te_recipe",
    "ue5m3_proposed_b16",
    "ue5m3_proposed_b32",
    "ue5m3_proposed_torch",
    "nvfp4_te_recipe",
    "nvfp4_proposed_settings",
)
LABELS = {
    "bf16": "BF16",
    "ue5m3_te_recipe": "UE5M3 with TE settings, D=1",
    "ue5m3_proposed_b16": "Proposed UE5M3, B=16, D=50",
    "ue5m3_proposed_b32": "Proposed UE5M3, B=32, D=50",
    "ue5m3_proposed_torch": "UE5M3 Torch control, B=16, D=50",
    "nvfp4_te_recipe": "Transformer Engine NVFP4, D=1",
    "nvfp4_proposed_settings": "Native NVFP4, no RHT/all linears, D=1",
}
COLORS = {
    "bf16": "#222222",
    "ue5m3_te_recipe": "#7A5195",
    "ue5m3_proposed_b16": "#007C83",
    "ue5m3_proposed_b32": "#6A994E",
    "ue5m3_proposed_torch": "#4C78A8",
    "nvfp4_te_recipe": "#C23B4A",
    "nvfp4_proposed_settings": "#D58A00",
}
BENCHMARKS = ("core_9mcqa_olmes", "mmlu_olmes", "mmlu_pro_mc")
BENCHMARK_LABELS = {
    "core_9mcqa_olmes": "Core 9 MCQA",
    "mmlu_olmes": "OLMES MMLU",
    "mmlu_pro_mc": "MMLU-Pro MC",
}
PDF_METADATA = {"CreationDate": None, "ModDate": None}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _save_figure(figure: Any, output: Path, stem: str) -> list[Path]:
    paths = [output / f"{stem}.pdf", output / f"{stem}.png"]
    figure.savefig(paths[0], bbox_inches="tight", metadata=PDF_METADATA)
    figure.savefig(paths[1], bbox_inches="tight", dpi=220)
    plt.close(figure)
    return paths


def _validation_matrix(rows: list[dict[str, str]]) -> dict[str, dict[int, float]]:
    matrix: dict[str, dict[int, float]] = defaultdict(dict)
    for row in rows:
        key = row["trajectory_key"]
        step = int(row["step"])
        if key not in ORDER or step not in STEPS:
            raise ValueError(f"unexpected validation cell {(key, step)}")
        if step in matrix[key]:
            raise ValueError(f"duplicate validation cell {(key, step)}")
        matrix[key][step] = float(row["nll"])
    expected = {(key, step) for key in ORDER for step in STEPS}
    actual = {(key, step) for key, values in matrix.items() for step in values}
    if actual != expected:
        raise ValueError("validation table is not a complete seven-by-twelve matrix")
    return matrix


def _render_validation(
    rows: list[dict[str, str]],
    figure_dir: Path,
    table_dir: Path,
) -> list[Path]:
    matrix = _validation_matrix(rows)
    reference = matrix["bf16"]
    relative = {
        key: {
            step: 100.0 * (reference[step] - values[step]) / reference[step] for step in STEPS
        }
        for key, values in matrix.items()
    }

    figure, (loss_axis, delta_axis) = plt.subplots(
        2,
        1,
        figsize=(7.25, 5.6),
        sharex=True,
        gridspec_kw={"height_ratios": [1.55, 1.0], "hspace": 0.10},
    )
    for key in ORDER:
        emphasized = key in {"ue5m3_proposed_b16", "ue5m3_te_recipe", "nvfp4_te_recipe"}
        style = "--" if key == "bf16" else "-"
        nll_values = [matrix[key][step] for step in STEPS]
        relative_values = [relative[key][step] for step in STEPS]
        loss_axis.plot(
            STEPS,
            [value if value <= 3.05 else float("nan") for value in nll_values],
            color=COLORS[key],
            label=LABELS[key],
            linestyle=style,
            linewidth=2.2 if emphasized else 1.45,
            marker="o",
            markersize=3.0 if emphasized else 2.3,
            alpha=1.0 if emphasized else 0.82,
        )
        for step, value in zip(STEPS, nll_values, strict=True):
            if value <= 3.05:
                continue
            loss_axis.scatter(  # place and label an explicit clipped spike
                step,
                2.84,
                marker="^",
                s=30,
                color=COLORS[key],
                zorder=4,
            )
            loss_axis.text(
                step,
                2.80,
                f"{value:.2f}",
                color=COLORS[key],
                fontsize=7.0,
                ha="center",
                va="top",
                fontweight="bold",
            )
        delta_axis.plot(
            STEPS,
            [value if value >= -12.0 else float("nan") for value in relative_values],
            color=COLORS[key],
            linestyle=style,
            linewidth=2.2 if emphasized else 1.45,
            marker="o",
            markersize=3.0 if emphasized else 2.3,
            alpha=1.0 if emphasized else 0.82,
        )
        for step, value in zip(STEPS, relative_values, strict=True):
            if value >= -12.0:
                continue
            delta_axis.scatter(
                step,
                -11.75,
                marker="v",
                s=30,
                color=COLORS[key],
                zorder=4,
            )
            delta_axis.text(
                step,
                -11.35,
                f"{value:+.2f}%",
                color=COLORS[key],
                fontsize=7.0,
                ha="center",
                va="bottom",
                fontweight="bold",
            )
    loss_axis.set_ylabel("Held-out validation NLL")
    loss_axis.set_title("A. Validation loss under each inference recipe", loc="left")
    loss_axis.set_ylim(2.2, 3.08)
    loss_axis.grid(alpha=0.18)
    loss_axis.legend(frameon=False, ncol=2, fontsize=7.0, loc="upper right")
    delta_axis.axhline(0.0, color=COLORS["bf16"], linewidth=1.0)
    delta_axis.set_xlabel("Optimizer step")
    delta_axis.set_ylabel("Relative NLL vs BF16 (%)\n$100(L_{BF16}-L)/L_{BF16}$")
    delta_axis.set_title(
        "B. BF16-relative validation loss (positive indicates lower NLL)",
        loc="left",
    )
    delta_axis.set_ylim(-12.25, 0.6)
    delta_axis.grid(alpha=0.18)
    delta_axis.set_xlim(STEPS[0], STEPS[-1])
    delta_axis.set_xticks(STEPS[1::2])
    figure.subplots_adjust(left=0.12, right=0.985, bottom=0.10, top=0.95)
    outputs = _save_figure(figure, figure_dir, "validation_loss_and_percent")

    table_path = table_dir / "final_validation.csv"
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ("trajectory_key", "label", "step", "nll", "percent_nll_reduction_vs_bf16")
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for key in ORDER:
            writer.writerow(
                {
                    "trajectory_key": key,
                    "label": LABELS[key],
                    "step": STEPS[-1],
                    "nll": f"{matrix[key][STEPS[-1]]:.12g}",
                    "percent_nll_reduction_vs_bf16": f"{relative[key][STEPS[-1]]:.12g}",
                }
            )
    return [*outputs, table_path]


def _render_olmes(
    rows: list[dict[str, str]],
    figure_dir: Path,
    table_dir: Path,
) -> list[Path]:
    scores: dict[tuple[str, str], float] = {}
    for row in rows:
        cell = (row["task_key"], row["benchmark"])
        if cell in scores:
            raise ValueError(f"duplicate OLMES cell {cell}")
        scores[cell] = float(row["score_percent"])
    expected = {(key, benchmark) for key in ORDER for benchmark in BENCHMARKS}
    if set(scores) != expected:
        raise ValueError("OLMES aggregate scores are not a complete seven-by-three matrix")
    differences = {cell: score - scores[("bf16", cell[1])] for cell, score in scores.items()}
    common_limit = max(1.0, float(int(max(abs(value) for value in differences.values())) + 1))
    common_ticks = [common_limit * value / 2.0 for value in (-2, -1, 0, 1, 2)]
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(7.25, 4.25),
        sharey=True,
        gridspec_kw={"wspace": 0.12},
    )
    positions = list(range(len(ORDER)))
    for axis, benchmark in zip(axes, BENCHMARKS, strict=True):
        axis.axvline(0.0, color=COLORS["bf16"], linewidth=1.1, linestyle="--")
        for position, key in enumerate(ORDER):
            value = differences[(key, benchmark)]
            marker = "D" if key == "bf16" else "s" if key == "nvfp4_te_recipe" else "o"
            axis.scatter(
                value,
                position,
                s=42 if key in {"bf16", "nvfp4_te_recipe"} else 34,
                marker=marker,
                color=COLORS[key],
                edgecolor="white",
                linewidth=0.7,
                zorder=3,
            )
            axis.annotate(
                f"{value:+.2f}",
                (value, position),
                xytext=(4 if value <= 0 else -4, 0),
                textcoords="offset points",
                ha="left" if value <= 0 else "right",
                va="center",
                fontsize=6.7,
                color=COLORS[key],
            )
        axis.set_xlim(-common_limit, common_limit)
        axis.set_xticks(common_ticks)
        axis.set_title(BENCHMARK_LABELS[benchmark])
        axis.grid(axis="x", alpha=0.22)
    axes[0].set_yticks(positions, [LABELS[key] for key in ORDER], fontsize=7.2)
    axes[0].invert_yaxis()
    figure.suptitle("Step-30,000 OLMES accuracy relative to BF16", fontsize=10.5)
    figure.supxlabel("Accuracy difference from BF16 (percentage points)", fontsize=8.0)
    figure.subplots_adjust(left=0.34, right=0.985, bottom=0.15, top=0.90)
    outputs = _save_figure(figure, figure_dir, "olmes_bf16_differences")

    table_path = table_dir / "olmes_scores_and_deltas.csv"
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ("task_key", "label", "benchmark", "score_percent", "delta_vs_bf16_pp")
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for key in ORDER:
            for benchmark in BENCHMARKS:
                writer.writerow(
                    {
                        "task_key": key,
                        "label": LABELS[key],
                        "benchmark": benchmark,
                        "score_percent": f"{scores[(key, benchmark)]:.12g}",
                        "delta_vs_bf16_pp": f"{differences[(key, benchmark)]:.12g}",
                    }
                )
    return [*outputs, table_path]


def render(data_directory: Path, output_directory: Path) -> dict[str, Any]:
    figure_dir = output_directory / "figures"
    table_dir = output_directory / "tables"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    validation_source = data_directory / "validation_metrics.csv"
    olmes_source = data_directory / "olmes_aggregate_scores.csv"
    outputs = [
        *_render_validation(_read_csv(validation_source), figure_dir, table_dir),
        *_render_olmes(_read_csv(olmes_source), figure_dir, table_dir),
    ]
    manifest = {
        "schema": "ue5m3_fp4_rendered_reference_artifacts_v1",
        "inputs": [
            {"file": path.name, "sha256": _sha256(path)}
            for path in (validation_source, olmes_source)
        ],
        "outputs": [
            {
                "file": path.relative_to(output_directory).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in outputs
        ],
        "formula": {
            "validation_percent": "100 * (BF16_NLL - candidate_NLL) / BF16_NLL",
            "olmes_percentage_points": "candidate_score_percent - BF16_score_percent",
        },
        "runtime": {
            "python": sys.version.split()[0],
            "matplotlib": matplotlib.__version__,
        },
    }
    manifest_path = output_directory / "artifact_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_data = Path(__file__).resolve().parent
    parser.add_argument("--data-directory", type=Path, default=default_data)
    parser.add_argument("--output-directory", type=Path, default=default_data / "generated")
    arguments = parser.parse_args()
    manifest = render(
        arguments.data_directory.resolve(),
        arguments.output_directory.resolve(),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
