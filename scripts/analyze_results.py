#!/usr/bin/env python3
"""
Quick analysis script to compare baseline LoRA vs SLoRA results.
Reads W&B logs and generates comparison report.
"""
import argparse
from pathlib import Path


def analyze_runs(baseline_dir: str, slora_dir: str, output_path: str):
    """Compare baseline and SLoRA training runs."""
    import json

    print("=== SLoRA Results Analysis ===\n")

    baseline_path = Path(baseline_dir)
    slora_path = Path(slora_dir)

    print(f"Baseline: {baseline_path}")
    print(f"SLoRA:    {slora_path}")
    print()

    trainer_state_baseline = baseline_path / "trainer_state.json"
    trainer_state_slora = slora_path / "trainer_state.json"

    if not trainer_state_baseline.exists() or not trainer_state_slora.exists():
        print("ERROR: trainer_state.json not found in one or both output directories")
        print("Make sure training has completed and saved checkpoints.")
        return

    with open(trainer_state_baseline) as f:
        baseline_state = json.load(f)

    with open(trainer_state_slora) as f:
        slora_state = json.load(f)

    baseline_loss = (
        baseline_state["best_metric"] if "best_metric" in baseline_state else "N/A"
    )
    slora_loss = slora_state["best_metric"] if "best_metric" in slora_state else "N/A"

    baseline_steps = baseline_state["global_step"]
    slora_steps = slora_state["global_step"]

    print("Final Results:")
    print(f"  Baseline - Steps: {baseline_steps}, Best Loss: {baseline_loss}")
    print(f"  SLoRA    - Steps: {slora_steps}, Best Loss: {slora_loss}")
    print()

    if slora_loss != "N/A" and baseline_loss != "N/A":
        loss_diff = (slora_loss - baseline_loss) / baseline_loss * 100
        print(f"Loss difference: {loss_diff:+.2f}%")

    print()
    print(
        "✅ Pass criteria: SLoRA loss ≤ +0.5% of baseline with ≥30% fewer accepted steps"
    )
    print()
    print("Check W&B for detailed metrics:")
    print("  - gate/acceptance_rate")
    print("  - gate/novelty over time")
    print("  - train_loss vs gate/accepted_steps")

    if output_path:
        with open(output_path, "w") as f:
            f.write(f"# SLoRA Analysis Report\n\n")
            f.write(f"## Results\n\n")
            f.write(f"- Baseline: {baseline_steps} steps, loss={baseline_loss}\n")
            f.write(f"- SLoRA: {slora_steps} steps, loss={slora_loss}\n")
            if slora_loss != "N/A" and baseline_loss != "N/A":
                loss_diff = (slora_loss - baseline_loss) / baseline_loss * 100
                f.write(f"- Loss difference: {loss_diff:+.2f}%\n")
        print(f"\nReport saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze SLoRA vs baseline results")
    parser.add_argument(
        "--baseline", type=str, required=True, help="Path to baseline output dir"
    )
    parser.add_argument(
        "--slora", type=str, required=True, help="Path to SLoRA output dir"
    )
    parser.add_argument(
        "--output", type=str, default="reports/analysis.md", help="Output report path"
    )
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    analyze_runs(args.baseline, args.slora, args.output)


if __name__ == "__main__":
    main()
