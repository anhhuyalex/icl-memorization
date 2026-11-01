"""Check Figure 6's running average without importing the plotting module.

``analysis.py`` needs torch and a LaTeX installation at import time, so the
smoothing function is extracted via its syntax tree, like the other tests.
"""
import ast
import os
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("ICL_TEST_ROOT", Path(__file__).resolve().parents[1]))
SWITCH = 10_000
WINDOW = 100


def load_smoother():
    tree = ast.parse((ROOT / "analysis.py").read_text())
    node = next(n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "_smooth_figure6_phase")
    namespace = {"pd": pd}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "analysis.py", "exec"), namespace)
    return namespace["_smooth_figure6_phase"]


def frame(accuracy, phase, start):
    return pd.DataFrame({
        "natural_step": np.arange(start, start + len(accuracy)),
        "accuracy": accuracy,
        "phase": phase,
    })


class Figure6SmoothingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.smooth = staticmethod(load_smoother())
        cls.rng = np.random.RandomState(0)

    def test_value_is_trailing_mean_of_last_window_steps(self):
        accuracy = self.rng.rand(SWITCH)
        out = self.smooth(frame(accuracy, "after", SWITCH), WINDOW).set_index("natural_step")["accuracy"]
        for step in (SWITCH + WINDOW - 1, SWITCH + 1234, SWITCH + SWITCH - 1):
            expected = accuracy[step - SWITCH - WINDOW + 1: step - SWITCH + 1].mean()
            self.assertAlmostEqual(out[step], expected)

    def test_first_point_is_a_full_window_and_nothing_straddles_the_switch(self):
        pre = self.smooth(frame(self.rng.rand(SWITCH), "before", 0), WINDOW)
        post = self.smooth(frame(self.rng.rand(SWITCH), "after", SWITCH), WINDOW)
        self.assertEqual(pre["natural_step"].min(), WINDOW - 1)
        self.assertEqual(pre["natural_step"].max(), SWITCH - 1)
        self.assertEqual(post["natural_step"].min(), SWITCH + WINDOW - 1)

    def test_runs_sharing_a_hidden_size_are_averaged_per_step(self):
        first, second = self.rng.rand(SWITCH), self.rng.rand(SWITCH)
        both = pd.concat([frame(first, "after", SWITCH), frame(second, "after", SWITCH)])
        out = self.smooth(both, WINDOW).set_index("natural_step")["accuracy"]
        step = SWITCH + 500
        expected = ((first + second) / 2)[step - SWITCH - WINDOW + 1: step - SWITCH + 1].mean()
        self.assertAlmostEqual(out[step], expected)
        self.assertEqual(out.index.min(), SWITCH + WINDOW - 1)

    def test_empty_phase_returns_empty_frame(self):
        self.assertTrue(self.smooth(frame([], "after", SWITCH), WINDOW).empty)


if __name__ == "__main__":
    unittest.main()
