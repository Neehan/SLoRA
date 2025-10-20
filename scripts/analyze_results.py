#!/usr/bin/env python3
"""
Compare baseline LoRA vs FLoRA training results.
"""
import argparse
import json
from pathlib import Path


def analyze_runs(baseline_dir: str, flora_dir: str, output_path: str):
    """Compare baseline and FLoRA training runs."""
    baseline_path = Path(baseline_dir)
    flora_path = Path(flora_dir)

    trainer_state_baseline = baseline_path / "trainer_state.json"
    trainer_state_flora = flora_path / "trainer_state.json"

    if not trainer_state_baseline.exists() or not trainer_state_flora.exists():
        print("ERROR: trainer_state.json not found")
        return

    with open(trainer_state_baseline) as f:
        baseline_state = json.load(f)
    with open(trainer_state_flora) as f:
        flora_state = json.load(f)

    print("=" * 80)
    print("FLoRA RESULTS COMPARISON")
    print("=" * 80)

    baseline_final = baseline_state["log_history"][-1]
    flora_final = flora_state["log_history"][-1]

    baseline_loss = baseline_final["loss"]
    flora_loss = flora_final["loss"]

    baseline_runtime = None
    flora_runtime = None
    for entry in reversed(baseline_state["log_history"]):
        if "train_runtime" in entry:
            baseline_runtime = entry["train_runtime"]
            break
    for entry in reversed(flora_state["log_history"]):
        if "train_runtime" in entry:
            flora_runtime = entry["train_runtime"]
            break

    print("\n📊 TRAINING LOSS")
    print("-" * 80)
    print(f"Baseline: {baseline_loss:.4f}")
    print(f"FLoRA:    {flora_loss:.4f}")
    print(
        f"Diff:     {flora_loss - baseline_loss:+.4f} ({(flora_loss - baseline_loss) / baseline_loss * 100:+.2f}%)"
    )

    print("\n⏱️  RUNTIME")
    print("-" * 80)
    if baseline_runtime and flora_runtime:
        print(f"Baseline: {baseline_runtime:.1f}s")
        print(f"FLoRA:    {flora_runtime:.1f}s")
        print(f"Diff:     {flora_runtime - baseline_runtime:+.1f}s")
    else:
        print("Runtime not found in logs")

    print("\n⚙️  GATE METRICS (FLoRA)")
    print("-" * 80)

    gate_file = flora_path / "gate_metrics.json"
    gate_entries = [
        e for e in flora_state["log_history"] if "gate/acceptance_rate_overall" in e
    ]

    acc_rate = None
    if gate_file.exists():
        with open(gate_file) as f:
            gate_data = json.load(f)
        acc_rate = gate_data["acceptance_rate"]
        accepted = gate_data["accepted_steps"]
        total = gate_data["total_steps"]
        rejected = gate_data["rejected_steps"]

        print(f"Acceptance Rate:     {acc_rate:.2%}")
        print(f"Accepted Steps:      {accepted}")
        print(f"Rejected Steps:      {rejected}")
        print(f"Total Steps:         {total}")
        print(f"Efficiency Gain:     {(1 - acc_rate) * 100:.1f}% fewer updates")
    elif gate_entries:
        final_gate = gate_entries[-1]
        acc_rate = final_gate["gate/acceptance_rate_overall"]
        accepted = final_gate.get("gate/accepted_steps_total", 0)
        total = flora_state["global_step"]

        print(f"Acceptance Rate:     {acc_rate:.2%}")
        print(f"Accepted Steps:      {accepted}")
        print(f"Rejected Steps:      {total - accepted}")
        print(f"Total Steps:         {total}")
        print(f"Efficiency Gain:     {(1 - acc_rate) * 100:.1f}% fewer updates")
    else:
        print("⚠️  No gate metrics found - check W&B")

    print("\n📈 EVALUATION METRICS")
    print("-" * 80)

    baseline_eval = [e for e in baseline_state["log_history"] if "eval_loss" in e]
    flora_eval = [e for e in flora_state["log_history"] if "eval_loss" in e]

    if baseline_eval and flora_eval:
        print(f"Baseline Eval Loss: {baseline_eval[-1]['eval_loss']:.4f}")
        print(f"FLoRA Eval Loss:    {flora_eval[-1]['eval_loss']:.4f}")
    else:
        print("No eval metrics found (eval_steps may be > max_steps)")

    print("\n" + "=" * 80)
    print("✅ SUCCESS CRITERIA")
    print("=" * 80)

    loss_diff_pct = abs((flora_loss - baseline_loss) / baseline_loss * 100)
    print(f"✓ Loss difference: {loss_diff_pct:.2f}% (target: <0.5%)")

    if acc_rate is not None:
        rejection_rate = (1 - acc_rate) * 100
        print(f"✓ Update reduction: {rejection_rate:.1f}% (target: ≥30%)")

        if loss_diff_pct < 0.5 and rejection_rate >= 30:
            print("\n🎉 SUCCESS!")
        elif loss_diff_pct < 0.5:
            print(
                f"\n⚠️  Loss good but rejection {rejection_rate:.1f}% < 30%. Decrease target_novelty (target rate)"
            )
        elif rejection_rate >= 30:
            print(
                f"\n⚠️  Rejection good but loss degraded {loss_diff_pct:.2f}%. Increase target_novelty (target rate)"
            )
        else:
            print("\n❌ Needs tuning")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=str, required=True)
    parser.add_argument("--flora", type=str, required=True)
    parser.add_argument("--output", type=str, default="reports/analysis.md")
    args = parser.parse_args()

    analyze_runs(args.baseline, args.flora, args.output)


if __name__ == "__main__":
    main()
