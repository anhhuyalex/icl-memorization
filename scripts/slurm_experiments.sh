#!/usr/bin/env bash
# Run one saved experiment from the array submitted by submit_experiments.py.
set -euo pipefail

manifest="${1:?Missing run list}"
python_executable="${2:?Missing Python executable}"
output_root="${3:?Missing output directory}"
task_index="${SLURM_ARRAY_TASK_ID:?This script must run as a Slurm array task}"
project_dir="${4:?Missing project directory}"
cd "$project_dir"

exec "$python_executable" "$project_dir/scripts/submit_experiments.py" \
  --run-task "$manifest" --task-index "$task_index" \
  --python "$python_executable" --savedir "$output_root"
