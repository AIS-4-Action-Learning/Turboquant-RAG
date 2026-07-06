#!/usr/bin/env python3
"""Generate BF16 vs TurboQuant telemetry comparison tables and plots.

This script reads JSONL benchmark outputs from a telemetry directory and emits:
1) Memory usage comparison table
2) Latency comparison table
3) Comparison plots for memory and latency ratios

It is tailored to files named like:
- BF16_<context>.jsonl
- TURBOQUANT_<context>.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


@dataclass
class FileMetrics:
    context_length: int
    model: str
    rows_in_file: int
    trials_used: int
    peak_vram_mb: float
    ttft_ms: float
    tpot_ms: float
    total_per_trial_ms: float
    total_three_trials_ms: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Directory containing telemetry JSONL files. Defaults to telemetry_results/ or telemtry_results/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for generated tables and plots. Defaults to <input-dir>/analysis.",
    )
    return parser.parse_args()


def resolve_input_dir(user_input: Path | None) -> Path:
    if user_input is not None:
        return user_input

    candidates = [
        Path("telemetry_results"),
        Path("telemtry_results"),
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not find telemetry directory. Expected one of: telemetry_results/, telemtry_results/."
    )


def parse_filename(path: Path) -> tuple[str, int] | None:
    stem = path.stem
    m = re.match(r"^(BF16|TURBOQUANT)_(\d+)$", stem, flags=re.IGNORECASE)
    if not m:
        return None

    raw_model, raw_ctx = m.groups()
    model = "BF16" if raw_model.upper() == "BF16" else "TQ"
    context_length = int(raw_ctx)
    return model, context_length


def parse_timestamp(value: str | None) -> datetime:
    if not value:
        return datetime.min
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return datetime.min


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["_timestamp"] = parse_timestamp(row.get("timestamp"))
            rows.append(row)
    return rows


def select_trials(rows: list[dict[str, Any]], max_trials: int = 3) -> list[dict[str, Any]]:
    if not rows:
        return []

    rows_sorted = sorted(rows, key=lambda r: r.get("_timestamp", datetime.min))

    # Prefer "latest per question_id" when question IDs are present (handles reruns appended in same file).
    has_qid = any(r.get("question_id") is not None for r in rows_sorted)
    if has_qid:
        latest_by_qid: dict[str, dict[str, Any]] = {}
        for row in rows_sorted:
            qid = row.get("question_id")
            if qid is None:
                continue
            latest_by_qid[str(qid)] = row

        if latest_by_qid:
            selected = list(latest_by_qid.values())
            selected = sorted(selected, key=lambda r: str(r.get("question_id")))
            if len(selected) >= max_trials:
                return selected[:max_trials]

    # Fallback: use most recent N rows.
    return rows_sorted[-max_trials:]


def mean_metric(rows: list[dict[str, Any]], key: str) -> float:
    values = [r[key] for r in rows if r.get(key) is not None]
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def sum_metric(rows: list[dict[str, Any]], key: str) -> float:
    values = [r[key] for r in rows if r.get(key) is not None]
    if not values:
        return float("nan")
    return float(sum(values))


def extract_file_metrics(path: Path) -> FileMetrics | None:
    parsed = parse_filename(path)
    if parsed is None:
        return None

    model, context_length = parsed
    rows = read_jsonl(path)
    if not rows:
        return None

    selected = select_trials(rows, max_trials=3)
    if not selected:
        return None

    # Prefer SMI VRAM if present; fallback to torch value.
    peak_vram = mean_metric(selected, "peak_vram_smi_mb")
    if pd.isna(peak_vram):
        peak_vram = mean_metric(selected, "peak_vram_torch_mb")

    return FileMetrics(
        context_length=context_length,
        model=model,
        rows_in_file=len(rows),
        trials_used=len(selected),
        peak_vram_mb=peak_vram,
        ttft_ms=mean_metric(selected, "ttft_ms"),
        tpot_ms=mean_metric(selected, "tpot_ms"),
        total_per_trial_ms=mean_metric(selected, "total_ms"),
        total_three_trials_ms=sum_metric(selected, "total_ms"),
    )


def build_tables(file_metrics: list[FileMetrics]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_df = pd.DataFrame([m.__dict__ for m in file_metrics])
    if raw_df.empty:
        raise RuntimeError("No telemetry data could be parsed from JSONL files.")

    bf16 = raw_df[raw_df["model"] == "BF16"].set_index("context_length")
    tq = raw_df[raw_df["model"] == "TQ"].set_index("context_length")

    common_contexts = sorted(set(bf16.index).intersection(set(tq.index)))
    if not common_contexts:
        raise RuntimeError("No overlapping context lengths between BF16 and TurboQuant files.")

    memory_df = pd.DataFrame(
        {
            "Context_length": common_contexts,
            "Turboquant_VRAM_MB": [tq.loc[c, "peak_vram_mb"] for c in common_contexts],
            "Turboquant_TTFT_ms": [tq.loc[c, "ttft_ms"] for c in common_contexts],
            "Turboquant_TPOT_ms": [tq.loc[c, "tpot_ms"] for c in common_contexts],
            "BF16_VRAM_MB": [bf16.loc[c, "peak_vram_mb"] for c in common_contexts],
            "BF16_TTFT_ms": [bf16.loc[c, "ttft_ms"] for c in common_contexts],
            "BF16_TPOT_ms": [bf16.loc[c, "tpot_ms"] for c in common_contexts],
        }
    )
    memory_df["Memory_Usage_Ratio_TQ_over_BF16"] = (
        memory_df["Turboquant_VRAM_MB"] / memory_df["BF16_VRAM_MB"]
    )

    latency_df = pd.DataFrame(
        {
            "Context_length": common_contexts,
            "BF16_per_trial_ms": [bf16.loc[c, "total_per_trial_ms"] for c in common_contexts],
            "BF16_3_trials_ms": [bf16.loc[c, "total_three_trials_ms"] for c in common_contexts],
            "TQ_per_trial_ms": [tq.loc[c, "total_per_trial_ms"] for c in common_contexts],
            "TQ_3_trials_ms": [tq.loc[c, "total_three_trials_ms"] for c in common_contexts],
            "TTFT_TQ_over_BF16": [tq.loc[c, "ttft_ms"] / bf16.loc[c, "ttft_ms"] for c in common_contexts],
            "TPOT_TQ_over_BF16": [tq.loc[c, "tpot_ms"] / bf16.loc[c, "tpot_ms"] for c in common_contexts],
            "Total_Time_TQ_over_BF16": [
                tq.loc[c, "total_per_trial_ms"] / bf16.loc[c, "total_per_trial_ms"] for c in common_contexts
            ],
        }
    )

    selection_df = raw_df.sort_values(["context_length", "model"]).reset_index(drop=True)
    return memory_df, latency_df, selection_df


def save_tables(memory_df: pd.DataFrame, latency_df: pd.DataFrame, selection_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    memory_df.to_csv(output_dir / "memory_usage_comparison.csv", index=False)
    latency_df.to_csv(output_dir / "latency_comparison.csv", index=False)
    selection_df.to_csv(output_dir / "trial_selection_summary.csv", index=False)


def plot_memory(memory_df: pd.DataFrame, output_dir: Path) -> None:
    x = memory_df["Context_length"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # VRAM comparison + ratio.
    ax = axes[0]
    ax.plot(x, memory_df["BF16_VRAM_MB"], marker="o", label="BF16 VRAM (MB)")
    ax.plot(x, memory_df["Turboquant_VRAM_MB"], marker="o", label="TurboQuant VRAM (MB)")
    ax.set_title("VRAM Usage vs Context Length")
    ax.set_xlabel("Context Length")
    ax.set_ylabel("Peak VRAM (MB)")
    ax.grid(True, alpha=0.25)

    ax2 = ax.twinx()
    ax2.plot(
        x,
        memory_df["Memory_Usage_Ratio_TQ_over_BF16"],
        color="purple",
        linestyle="--",
        marker="s",
        label="Memory ratio (TQ/BF16)",
    )
    ax2.set_ylabel("Ratio")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="best")

    # TTFT / TPOT comparison.
    ax = axes[1]
    ax.plot(x, memory_df["BF16_TTFT_ms"], marker="o", label="BF16 TTFT (ms)")
    ax.plot(x, memory_df["Turboquant_TTFT_ms"], marker="o", label="TurboQuant TTFT (ms)")
    ax.plot(x, memory_df["BF16_TPOT_ms"], marker="^", linestyle="--", label="BF16 TPOT (ms)")
    ax.plot(x, memory_df["Turboquant_TPOT_ms"], marker="^", linestyle="--", label="TurboQuant TPOT (ms)")
    ax.set_title("TTFT and TPOT vs Context Length")
    ax.set_xlabel("Context Length")
    ax.set_ylabel("Milliseconds")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_dir / "memory_comparison.png", dpi=200)
    plt.close(fig)


def plot_latency(latency_df: pd.DataFrame, output_dir: Path) -> None:
    x = latency_df["Context_length"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    ax.plot(x, latency_df["BF16_per_trial_ms"], marker="o", label="BF16 per trial")
    ax.plot(x, latency_df["TQ_per_trial_ms"], marker="o", label="TQ per trial")
    ax.plot(x, latency_df["BF16_3_trials_ms"], marker="^", linestyle="--", label="BF16 3 trials")
    ax.plot(x, latency_df["TQ_3_trials_ms"], marker="^", linestyle="--", label="TQ 3 trials")
    ax.set_title("Latency Comparison")
    ax.set_xlabel("Context Length")
    ax.set_ylabel("Milliseconds")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    ax = axes[1]
    ax.plot(x, latency_df["TTFT_TQ_over_BF16"], marker="o", label="TTFT ratio (TQ/BF16)")
    ax.plot(x, latency_df["TPOT_TQ_over_BF16"], marker="o", label="TPOT ratio (TQ/BF16)")
    ax.plot(x, latency_df["Total_Time_TQ_over_BF16"], marker="o", label="Total ratio (TQ/BF16)")
    ax.axhline(1.0, color="black", linestyle=":", linewidth=1)
    ax.set_title("Latency Ratios (TQ / BF16)")
    ax.set_xlabel("Context Length")
    ax.set_ylabel("Ratio")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_dir / "latency_comparison.png", dpi=200)
    plt.close(fig)


def rounded(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy().round(4)


def main() -> None:
    args = parse_args()
    input_dir = resolve_input_dir(args.input_dir)
    output_dir = args.output_dir or (input_dir / "analysis")

    files = sorted(input_dir.glob("*.jsonl"))
    metrics: list[FileMetrics] = []
    for path in files:
        m = extract_file_metrics(path)
        if m is not None:
            metrics.append(m)

    memory_df, latency_df, selection_df = build_tables(metrics)
    save_tables(memory_df, latency_df, selection_df, output_dir)
    plot_memory(memory_df, output_dir)
    plot_latency(latency_df, output_dir)

    print("Input directory:", input_dir)
    print("Output directory:", output_dir)
    print()
    print("Memory usage comparison table:")
    print(rounded(memory_df).to_string(index=False))
    print()
    print("Latency comparison table:")
    print(rounded(latency_df).to_string(index=False))
    print()
    print("Generated files:")
    for p in sorted(output_dir.iterdir()):
        print("-", p.name)


if __name__ == "__main__":
    main()
