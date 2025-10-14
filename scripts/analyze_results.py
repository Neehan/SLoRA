#!/usr/bin/env python3
"""
Compare baseline LoRA vs SLoRA training results.
"""
import argparse
import json
from pathlib import Path


def analyze_runs(baseline_dir: str, slora_dir: str, output_path: str):
    """Compare baseline and SLoRA training runs."""
    baseline_path = Path(baseline_dir)
    slora_path = Path(slora_dir)

    trainer_state_baseline = baseline_path / "trainer_state.json"
    trainer_state_slora = slora_path / "trainer_state.json"

    if not trainer_state_baseline.exists() or not trainer_state_slora.exists():
        print("ERROR: trainer_state.json not found")
        return

    with open(trainer_state_baseline) as f:
        baseline_state = json.load(f)
    with open(trainer_state_slora) as f:
        slora_state = json.load(f)

    print("=" * 80)
    print("SLoRA RESULTS COMPARISON")
    print("=" * 80)

    baseline_final = baseline_state["log_history"][-1]
    slora_final = slora_state["log_history"][-1]

    baseline_loss = baseline_final["loss"]
    slora_loss = slora_final["loss"]

    baseline_runtime = None
    slora_runtime = None
    for entry in reversed(baseline_state["log_history"]):
        if "train_runtime" in entry:
            baseline_runtime = entry["train_runtime"]
            break
    for entry in reversed(slora_state["log_history"]):
        if "train_runtime" in entry:
            slora_runtime = entry["train_runtime"]
            break

    print("\n📊 TRAINING LOSS")
    print("-" * 80)
    print(f"Baseline: {baseline_loss:.4f}")
    print(f"SLoRA:    {slora_loss:.4f}")
    print(f"Diff:     {slora_loss - baseline_loss:+.4f} ({(slora_loss - baseline_loss) / baseline_loss * 100:+.2f}%)")

    print("\n⏱️  RUNTIME")
    print("-" * 80)
    if baseline_runtime and slora_runtime:
        print(f"Baseline: {baseline_runtime:.1f}s")
        print(f"SLoRA:    {slora_runtime:.1f}s")
        print(f"Diff:     {slora_runtime - baseline_runtime:+.1f}s")
    else:
        print("Runtime not found in logs")

    print("\n⚙️  GATE METRICS (SLoRA)")
    print("-" * 80)

    gate_entries = [e for e in slora_state["log_history"] if "gate/acceptance_rate_overall" in e]
    if gate_entries:
        final_gate = gate_entries[-1]
        acc_rate = final_gate["gate/acceptance_rate_overall"]
        accepted = final_gate.get("gate/accepted_steps_total", 0)
        total = slora_state["global_step"]

        print(f"Acceptance Rate:     {acc_rate:.2%}")
        print(f"Accepted Steps:      {accepted}")
        print(f"Rejected Steps:      {total - accepted}")
        print(f"Total Steps:         {total}")
        print(f"Efficiency Gain:     {(1 - acc_rate) * 100:.1f}% fewer updates")
    else:
        print("⚠️  No gate metrics found")

    print("\n📈 EVALUATION METRICS")
    print("-" * 80)

    baseline_eval = [e for e in baseline_state["log_history"] if "eval_loss" in e]
    slora_eval = [e for e in slora_state["log_history"] if "eval_loss" in e]

    if baseline_eval and slora_eval:
        print(f"Baseline Eval Loss: {baseline_eval[-1]['eval_loss']:.4f}")
        print(f"SLoRA Eval Loss:    {slora_eval[-1]['eval_loss']:.4f}")
    else:
        print("No eval metrics found (eval_steps may be > max_steps)")

    print("\n" + "=" * 80)
    print("✅ SUCCESS CRITERIA")
    print("=" * 80)

    loss_diff_pct = abs((slora_loss - baseline_loss) / baseline_loss * 100)
    print(f"✓ Loss difference: {loss_diff_pct:.2f}% (target: <0.5%)")

    if gate_entries:
        rejection_rate = (1 - acc_rate) * 100
        print(f"✓ Update reduction: {rejection_rate:.1f}% (target: ≥30%)")

        if loss_diff_pct < 0.5 and rejection_rate >= 30:
            print("\n🎉 SUCCESS!")
        elif loss_diff_pct < 0.5:
            print(f"\n⚠️  Loss good but rejection {rejection_rate:.1f}% < 30%. Increase tau_n")
        elif rejection_rate >= 30:
            print(f"\n⚠️  Rejection good but loss degraded {loss_diff_pct:.2f}%. Decrease tau_n")
        else:
            print("\n❌ Needs tuning")

    print("=" * 80)

    if output_path:
        with open(output_path, "w") as f:
            f.write("# SLoRA Analysis Report\n\n")
            f.write(f"## Training Loss\n\n")
            f.write(f"- Baseline: {baseline_loss:.4f}\n")
            f.write(f"- SLoRA: {slora_loss:.4f}\n")
            f.write(f"- Difference: {slora_loss - baseline_loss:+.4f} ({(slora_loss - baseline_loss) / baseline_loss * 100:+.2f}%)\n")
            if gate_entries:
                f.write(f"\n## Gate Metrics\n\n")
                f.write(f"- Acceptance Rate: {acc_rate:.2%}\n")
                f.write(f"- Accepted: {accepted} / {total}\n")
                f.write(f"- Efficiency Gain: {(1 - acc_rate) * 100:.1f}%\n")
        print(f"\nReport saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=str, required=True)
    parser.add_argument("--slora", type=str, required=True)
    parser.add_argument("--output", type=str, default="reports/analysis.md")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    analyze_runs(args.baseline, args.slora, args.output)


if __name__ == "__main__":
    main()
