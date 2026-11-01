# Meta-memorization and memorization scaling laws in transformers

Code for Meta-memorization and memorization scaling laws in transformers


## Environment
Create the environment from
`environment.yml`. It includes both Conda and pip packages, with
package versions pinned.  

Install Miniconda or Anaconda, then run:

```bash
conda env create --prefix ./.conda -f environment.yml
conda activate ./.conda
```
Ghostscript (`gs`) must
also be available for Figure 6, and figure generation requires a working LaTeX
installation and fonts. 

If a copied TeX installation reports that `latex.fmt` was made by a different
executable version, rebuild it within the active environment with
`fmtutil-sys --byfmt latex`.

## Generate experiment data

### Run sweeps in parallel with Slurm

Activate the Conda environment, then submit all experiments:

```bash
python scripts/submit_experiments.py --savedir ./results
```

By default, this submits one Slurm array, with at most 20 runs active at once. Each run
requests one GPU, one CPU, 30 GB of memory, and 24 hours. To change the
concurrency, memory, time limit, output directory, or cluster partition:

```bash
python scripts/submit_experiments.py --concurrency 10 --memory 40G \
  --time 36:00:00 --savedir /path/to/results \
  --sbatch-option=--partition=gpu
```

Use `--sbatch-option` more than once to add other cluster-specific options.
Use `--dry-run` to prepare the run list and print the submission command
without submitting jobs. The submitter prints the path to `runs.json`, which
maps each array index to an experiment and its original configuration index.
It also saves copies of the configurations next to the run list. Logs are in
the neighboring `logs` directory, named `<job-id>_<array-index>.out` and
`.err`. Experiment records are written under the specified `--savedir`.

Check the status of the submitted job (replace `JOB_ID` with the printed ID):

```bash
squeue -j JOB_ID
sacct -j JOB_ID --format=JobID,State,ExitCode
```

To retry failed array indices 12 and 45 using the same run list:

```bash
python scripts/submit_experiments.py \
  --retry /path/to/results/slurm-submissions/sweep-XXXXXX/runs.json \
  --indices 12,45
```

Retries use the original output directory unless `--savedir` is given. Other
resource options can also be supplied again. The 12-layer no-resampling preset
may exit successfully without training when a model falls outside its
parameter-count bounds; this is a deliberate filter, not a failed Slurm task.

### Run sweeps sequentially

After activating the environment, run commands from the repository director to generate experiment data.
Each JSON file in `configs/experiments/` describes one experiment sweep. To run
all configured sweeps and write their records under `./results`:

```bash
python scripts/run_experiment.py --all --savedir ./results
```

This runs all experimental configurations sequentially. 
To see the commands without running experiments, add `--dry-run`. 

If a run fails, the launcher continues with the remaining runs,
then reports the failures and exits with a nonzero status.

To inspect or run one index of a sweep:

```bash
python scripts/run_experiment.py configs/experiments/figure2_12layer_noresample.json \
  --index 0 --savedir ./results --dry-run

# Remove --dry-run to start this run.
```

To use a Slurm array, first print the number of indices, then invoke the
launcher from an array job with `SLURM_ARRAY_TASK_ID`:

```bash
python scripts/run_experiment.py configs/experiments/figure2_12layer_noresample.json \
  --print-array-size

python scripts/run_experiment.py configs/experiments/figure2_12layer_noresample.json \
  --index "${SLURM_ARRAY_TASK_ID}" --savedir /path/to/results
```

The first command prints a count; use array indices from 0 through count minus
1. The launcher does not submit the array job.

## Generate figures

Run commands from the repository directory after the experiments have finished.
To generate all main and supplemental figures from records in `./results`:

```bash
python analysis.py --figures all
```

To rebuild the figure caches from those records before rendering, run:

```bash
python analysis.py --figures all --force_regenerate_cache
```

The default input directory is `./results`, matching the experiment commands
above. If training used another directory, set `ICL_RESULTS_DIR` to that
directory. You can also generate individual figures:

```bash
export ICL_RESULTS_DIR=/path/to/results
python analysis.py --figures S2
python analysis.py --figures 2 --force_regenerate_cache
```