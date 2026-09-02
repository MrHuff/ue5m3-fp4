#!/usr/bin/env python3
# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render the report's scale-target figure from the public archived counts."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ARCHIVE_SCHEMA = "ue5m3_fp4_public_archived_scale_target_histograms_v1"
SNAPSHOT_SPECS = {
    "bf16": {"label": "BF16", "color": "#222222"},
    "ue5m3_proposed_b16": {"label": "Proposed B=16", "color": "#007C83"},
}
SCALE_TARGETS = (448.0, 2048.0)


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 220,
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "legend.fontsize": 8.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
            "lines.linewidth": 1.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def validate_histogram(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a JSON object")
    counts = value.get("counts")
    edges = value.get("log10_bin_edges")
    if not isinstance(counts, list) or not counts:
        raise ValueError(f"{label} must contain non-empty counts")
    if not isinstance(edges, list) or len(edges) != len(counts) + 1:
        raise ValueError(f"{label} edges/counts are inconsistent")
    if any(type(count) is not int or count < 0 for count in counts):
        raise ValueError(f"{label} counts must be nonnegative integers")
    numeric_edges = np.asarray(edges, dtype=np.float64)
    if not bool(np.isfinite(numeric_edges).all()) or not bool(
        (np.diff(numeric_edges) > 0).all()
    ):
        raise ValueError(f"{label} edges must be finite and strictly increasing")
    for field in (
        "total_count",
        "finite_count",
        "nonfinite_count",
        "zero_count",
        "positive_count",
        "underflow_count",
        "in_range_count",
        "overflow_count",
    ):
        if type(value.get(field)) is not int or value[field] < 0:
            raise ValueError(f"{label}.{field} must be a nonnegative integer")
    if value["total_count"] != value["finite_count"] + value["nonfinite_count"]:
        raise ValueError(f"{label} total accounting failed")
    if value["finite_count"] != value["zero_count"] + value["positive_count"]:
        raise ValueError(f"{label} finite accounting failed")
    if value["in_range_count"] != sum(counts):
        raise ValueError(f"{label} binned accounting failed")
    if value["positive_count"] != (
        value["underflow_count"] + value["in_range_count"] + value["overflow_count"]
    ):
        raise ValueError(f"{label} range accounting failed")
    expected_zero_fraction = value["zero_count"] / max(1, value["finite_count"])
    if not math.isclose(
        float(value["zero_fraction_of_finite"]),
        expected_zero_fraction,
        rel_tol=1e-12,
        abs_tol=1e-15,
    ):
        raise ValueError(f"{label} zero fraction failed")
    return value


def _load(path: Path, expected_key: str) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid snapshot JSON: {path.name}") from error
    if not isinstance(document, dict) or document.get("schema") != ARCHIVE_SCHEMA:
        raise ValueError(f"unexpected snapshot schema: {path.name}")
    if document.get("archive_key") != expected_key:
        raise ValueError(f"snapshot key differs from filename: {path.name}")
    pooled = document.get("capture", {}).get("pooled", {})
    for name in (
        "x_histogram",
        "weight_histogram",
        "dy_histogram",
        "wgrad_dy_block_amax_histogram",
    ):
        validate_histogram(pooled.get(name), label=f"{expected_key}.{name}")
    scales = pooled.get("raw_scale_codes")
    if not isinstance(scales, Mapping) or set(scales) != {"448", "2048"}:
        raise ValueError(f"{expected_key} raw scale targets changed")
    for target in SCALE_TARGETS:
        record = scales[f"{target:g}"]
        if float(record.get("target")) != target:
            raise ValueError(f"{expected_key} scale target changed")
        validate_histogram(
            record.get("raw_code_histogram"),
            label=f"{expected_key}.raw_scale_codes.{target:g}",
        )
    return document


def _density(histogram: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    edges = np.asarray(histogram["log10_bin_edges"], dtype=np.float64)
    counts = np.asarray(histogram["counts"], dtype=np.float64)
    positive = int(histogram["positive_count"])
    if positive <= 0:
        raise ValueError("cannot normalize a histogram without positive values")
    return edges, counts / (positive * np.diff(edges))


def _zero_percent(fraction: float) -> str:
    percent = 100.0 * fraction
    return f"{percent:.2f}%" if percent >= 1.0 else f"{percent:.3f}%"


def render(snapshots: Mapping[str, Mapping[str, Any]], output: Path) -> None:
    _configure_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.25))
    panels = (
        (axes[0, 0], "x_histogram", "A. Down-projection input $|X|$"),
        (axes[0, 1], "weight_histogram", "B. Down-projection weight $|W|$"),
        (axes[1, 0], "dy_histogram", "C. Upstream gradient $|dY|$"),
    )
    for axis, histogram_name, title in panels:
        active_left: list[float] = []
        active_right: list[float] = []
        for key, spec in SNAPSHOT_SPECS.items():
            histogram = snapshots[key]["capture"]["pooled"][histogram_name]
            edges, density = _density(histogram)
            active = np.flatnonzero(density > 0.0)
            active_left.append(float(edges[active[0]]))
            active_right.append(float(edges[active[-1] + 1]))
            axis.stairs(
                np.where(density > 0.0, density, np.nan),
                edges,
                baseline=None,
                color=spec["color"],
                label=spec["label"],
                linewidth=1.55,
            )
        axis.set_yscale("log")
        axis.set_xlim(min(active_left) - 0.25, max(active_right) + 0.25)
        axis.set_title(title)
        axis.set_xlabel(r"$\log_{10}$ magnitude")
        axis.set_ylabel("Nonzero density per decade")
        axis.grid(which="major", alpha=0.22)
        if histogram_name in {"x_histogram", "dy_histogram"}:
            zero_lines = []
            for key, spec in SNAPSHOT_SPECS.items():
                fraction = snapshots[key]["capture"]["pooled"][histogram_name][
                    "zero_fraction_of_finite"
                ]
                zero_lines.append(f"{spec['label']}: {_zero_percent(fraction)}")
            axis.text(
                0.025,
                0.035,
                "Exact zero mass (not plotted)\n" + "; ".join(zero_lines),
                transform=axis.transAxes,
                fontsize=6.5,
                va="bottom",
                bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.88},
            )
    axes[0, 0].legend(frameon=False, loc="upper left", fontsize=7.2)

    scale_axis = axes[1, 1]
    line_styles = {448.0: "-", 2048.0: "--"}
    active_left = []
    active_right = []
    for key, spec in SNAPSHOT_SPECS.items():
        raw_scales = snapshots[key]["capture"]["pooled"]["raw_scale_codes"]
        for target in SCALE_TARGETS:
            histogram = raw_scales[f"{target:g}"]["raw_code_histogram"]
            edges, density = _density(histogram)
            active = np.flatnonzero(density > 0.0)
            active_left.append(float(edges[active[0]]))
            active_right.append(float(edges[active[-1] + 1]))
            scale_axis.stairs(
                np.where(density > 0.0, density, np.nan),
                edges,
                baseline=None,
                color=spec["color"],
                linestyle=line_styles[target],
                linewidth=1.55,
                label="_nolegend_",
            )
    scale_axis.set_yscale("log")
    scale_axis.set_xlim(min(active_left) - 0.15, max(active_right) + 0.15)
    scale_axis.set_title(r"D. $dY^{\mathsf{T}}$ block-scale code (current amax)")
    scale_axis.set_xlabel(r"$\log_{10}$ raw UE5M3 scale code")
    scale_axis.set_ylabel("Density per decade")
    checkpoint_handles = [
        scale_axis.plot([], [], color=spec["color"], label=spec["label"], linewidth=1.55)[0]
        for spec in SNAPSHOT_SPECS.values()
    ]
    target_handles = [
        scale_axis.plot(
            [],
            [],
            color="#555555",
            linestyle=line_styles[target],
            label=f"$T={target:g}$",
            linewidth=1.55,
        )[0]
        for target in SCALE_TARGETS
    ]
    checkpoint_legend = scale_axis.legend(
        handles=checkpoint_handles,
        frameon=False,
        fontsize=6.5,
        loc="upper left",
    )
    scale_axis.add_artist(checkpoint_legend)
    scale_axis.legend(
        handles=target_handles,
        frameon=False,
        fontsize=6.5,
        loc="upper right",
    )
    scale_axis.grid(which="major", alpha=0.22)

    fig.suptitle(
        "Step-30,000 late-layer value and scale-target distributions",
        fontsize=10.5,
        y=0.985,
    )
    fig.text(
        0.5,
        0.012,
        "One held-out 8,192-token sequence; BF16 execution of loaded master weights. "
        "Scale codes use each module's current dY amax.",
        ha="center",
        fontsize=6.8,
        color="#555555",
    )
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.96), h_pad=1.15, w_pad=1.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output.with_suffix(".pdf"),
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(output.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=Path(__file__).with_name("archived"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    snapshots = {key: _load(args.archive_dir / f"{key}.json", key) for key in SNAPSHOT_SPECS}
    reference = snapshots["bf16"]
    proposed = snapshots["ue5m3_proposed_b16"]
    if reference["validation_identity"] != proposed["validation_identity"]:
        raise ValueError("snapshots do not share a validation identity")
    if reference["sequence_identity"] != proposed["sequence_identity"]:
        raise ValueError("snapshots do not share a token-row identity")
    render(snapshots, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
