#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON:-$repo_root/.venv/bin/python}"
gflownet_root="${GFLOWNET_ROOT:-$repo_root/../gflownet}"

usage() {
  cat <<'EOF'
Usage:
  scripts/run_compare.sh [quick|full] [hydra overrides...]

Modes:
  quick  Runs a small multi-seed comparison across all methods.
  full   Runs the default comparisons config.

Examples:
  scripts/run_compare.sh quick
  scripts/run_compare.sh full comparison.methods=[active,gflownet,hybrid]
EOF
}

mode="${1:-quick}"
if [[ "$mode" == "-h" || "$mode" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "$mode" == "quick" || "$mode" == "full" ]]; then
  shift || true
else
  printf 'Unknown mode: %s\n\n' "$mode" >&2
  usage >&2
  exit 1
fi

if [[ ! -x "$python_bin" ]]; then
  printf 'Missing Python executable: %s\n' "$python_bin" >&2
  exit 1
fi

if [[ ! -d "$gflownet_root" ]]; then
  printf 'Missing sibling gflownet checkout: %s\n' "$gflownet_root" >&2
  exit 1
fi

cmd=(
  "$python_bin"
  "$repo_root/experiments/run_comparisons.py"
  "gflownet_root=$gflownet_root"
)

if [[ "$mode" == "quick" ]]; then
  cmd+=(
    "seeds=[0,1,2]"
    "oracle.budget=20"
    "oracle.vocabulary_check=false"
    "dataset.num_queries=20"
    "baseline.epochs=3"
    "baseline.hidden_dim=64"
    "baseline.n_layers=1"
    "baseline.batch_size=16"
    "active.initial_size=8"
    "active.batch_size=4"
    "active.max_rounds=3"
    "active.candidate_pool_size=16"
    "active.surrogate.type=ensemble"
    "active.surrogate.hidden_dim=32"
    "active.surrogate.n_layers=1"
    "active.surrogate.ensemble_size=2"
    "active.surrogate.epochs=2"
    "active.surrogate.batch_size=16"
    "gflownet.n_train_steps=5"
    "gflownet.batch_size_forward=4"
    "gflownet.evaluation_samples=4"
    "gflownet.policy.hidden_dim=32"
    "gflownet.policy.n_layers=1"
    "gflownet.evaluator.period=5"
    "gflownet.evaluator.n=4"
    "gflownet.evaluator.checkpoints_period=5"
    "hybrid.initial_size=8"
    "hybrid.batch_size=4"
    "hybrid.max_rounds=2"
    "hybrid.candidate_pool_size=8"
    "hybrid.fallback_random_pool_size=8"
    "hybrid.surrogate.type=ensemble"
    "hybrid.surrogate.hidden_dim=32"
    "hybrid.surrogate.n_layers=1"
    "hybrid.surrogate.ensemble_size=2"
    "hybrid.surrogate.epochs=2"
    "hybrid.surrogate.batch_size=16"
    "hybrid.gflownet.n_train_steps=5"
    "hybrid.gflownet.batch_size_forward=4"
    "hybrid.gflownet.sample_size=8"
    "hybrid.gflownet.policy.hidden_dim=32"
    "hybrid.gflownet.policy.n_layers=1"
    "hybrid.gflownet.evaluator.period=5"
    "hybrid.gflownet.evaluator.n=4"
    "hybrid.gflownet.evaluator.checkpoints_period=5"
  )
fi

cmd+=("$@")

tmp_output="$(mktemp)"
cleanup() {
  rm -f "$tmp_output"
}
trap cleanup EXIT

printf 'Running command:\n  %s\n' "${cmd[*]}"
"${cmd[@]}" | tee "$tmp_output"

run_dir="$("$python_bin" - "$tmp_output" <<'PY'
import json
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(encoding="utf-8")
marker = text.rfind('{\n  "run_dir"')
if marker == -1:
    marker = text.rfind('{"run_dir"')
if marker == -1:
    raise SystemExit("Could not find run_dir in runner output.")
payload = json.loads(text[marker:])
print(payload["run_dir"])
PY
)"

"$python_bin" - "$run_dir" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
metrics = json.loads((run_dir / "aggregate_metrics.json").read_text(encoding="utf-8"))

print()
print(f"Results saved under: {run_dir}")
print("Key artifacts:")
for artifact in (
    "aggregate_metrics.json",
    "pairwise_tests.json",
    "curve_best.json",
    "curve_best.png",
    "curve_top10.json",
    "curve_top10.png",
):
    path = run_dir / artifact
    if path.exists():
        print(f"  - {path}")

rows = []
for method, summary in metrics.items():
    rows.append(
        (
            float(summary["best_score"]["mean"]),
            method,
            float(summary["top10_score"]["mean"]),
            float(summary["queries_to_90pct_best"]["mean"]),
        )
    )

rows.sort(reverse=True)
print()
print("Best-score comparison:")
for best_score, method, top10_score, q90 in rows:
    print(
        f"  - {method:10s} best_score={best_score:6.2f} "
        f"top10_score={top10_score:6.2f} queries_to_90pct_best={q90:6.2f}"
    )
PY
