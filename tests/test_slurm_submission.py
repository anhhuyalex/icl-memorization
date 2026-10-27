"""Exercise Slurm submission and worker routing without running training."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(os.environ.get("ICL_TEST_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "scripts"))
import run_experiment as launcher


class SlurmSubmissionTests(unittest.TestCase):
    def test_manifest_and_worker_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            bindir = temp / "bin"
            bindir.mkdir()
            sbatch = bindir / "sbatch"
            sbatch.write_text("#!/bin/sh\nprintf '12345\\n'\n")
            sbatch.chmod(0o755)
            output_root = temp / "results"
            env = os.environ.copy()
            env["CONDA_PREFIX"] = str(Path(sys.executable).resolve().parent.parent)
            env["PATH"] = str(bindir) + os.pathsep + env.get("PATH", "")
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "submit_experiments.py"),
                 "--savedir", str(output_root)], cwd=ROOT, env=env,
                capture_output=True, text=True, check=True,
            )
            self.assertIn("Submitted Slurm array: 12345", result.stdout)
            manifests = list((output_root / "slurm-submissions").glob("sweep-*/runs.json"))
            self.assertEqual(len(manifests), 1)
            manifest = manifests[0]
            saved = json.loads(manifest.read_text())
            self.assertEqual(saved["savedir"], str(output_root))

            expected = []
            for path in sorted((ROOT / "configs" / "experiments").glob("*.json")):
                config = launcher.load_config(path)
                expected.extend((path.name, config["name"], index)
                                for index in launcher.eligible_indices(config))
            actual = [(Path(task["config"]).name, task["name"], task["index"])
                      for task in saved["tasks"]]
            self.assertEqual(actual, expected)
            self.assertEqual(len(actual), len(set(actual)))

            # Invoke the real worker for the first and last run of each
            # configuration. The Python wrapper executes the real submitter
            # and launcher, but records the final command instead of training.
            wrapper = temp / "python-wrapper"
            wrapper.write_text(
                f"#!{sys.executable}\n"
                "import json, os, sys\n"
                "from pathlib import Path\n"
                "i = 2 if len(sys.argv) > 1 and sys.argv[1] == '-u' else 1\n"
                "if len(sys.argv) > i and Path(sys.argv[i]).name.startswith('main'):\n"
                "    Path(os.environ['TEST_CAPTURE']).write_text(json.dumps({'argv': sys.argv[1:], 'wandb_mode': os.environ.get('WANDB_MODE')}))\n"
                "else:\n"
                f"    os.execv({sys.executable!r}, [{sys.executable!r}] + sys.argv[1:])\n"
            )
            wrapper.chmod(0o755)
            sample = {}
            for array_index, task in enumerate(saved["tasks"]):
                config_name = Path(task["config"]).name
                sample.setdefault(config_name, []).append(array_index)
            for indices in sample.values():
                for array_index in {indices[0], indices[-1]}:
                    task = saved["tasks"][array_index]
                    config = launcher.load_config(Path(task["config"]))
                    capture = temp / "capture.json"
                    worker_env = env | {"SLURM_ARRAY_TASK_ID": str(array_index),
                                        "TEST_CAPTURE": str(capture)}
                    subprocess.run(
                        ["bash", str(ROOT / "scripts" / "slurm_experiments.sh"),
                         str(manifest), str(wrapper), str(output_root), str(ROOT)],
                        cwd=ROOT, env=worker_env, capture_output=True, text=True, check=True,
                    )
                    captured = json.loads(capture.read_text())
                    command = launcher.build_command(config, task["index"], str(wrapper), str(output_root))
                    self.assertEqual(captured["argv"], command[1:])
                    if "WANDB_MODE" in config.get("environment", {}):
                        self.assertEqual(captured["wandb_mode"], config["environment"]["WANDB_MODE"])


if __name__ == "__main__":
    unittest.main()
