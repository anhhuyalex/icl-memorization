"""Check launcher arguments and training presets without running GPU experiments.

We execute each script's actual argparse definitions and preset functions via
their syntax tree. GPU model construction and its size filter are deliberately
outside this test.
"""
import argparse
import ast
import contextlib
import functools
import io
import os
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(os.environ.get("ICL_TEST_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "scripts"))
import run_experiment as launcher


def parser_for(path):
    tree = ast.parse(path.read_text())
    nodes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "parser" for t in node.targets):
            nodes.append(node)
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and isinstance(node.value.func.value, ast.Name)
                and node.value.func.value.id == "parser" and node.value.func.attr == "add_argument"):
            nodes.append(node)
    nodes.sort(key=lambda node: node.lineno)
    namespace = {"argparse": argparse}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), "exec"), namespace)
    return namespace["parser"]


class RemoveGpuChecks(ast.NodeTransformer):
    def visit_If(self, node):
        # The 12-layer no-resampling branch constructs a model solely to
        # enforce its size filter. Test its parameters separately below.
        if (isinstance(node.test, ast.Compare)
                and "memo_apr19_zipf_gpt2_vary_num_hidden_features_num_heads_noresample" in ast.unparse(node.test)):
            for i, child in enumerate(node.body):
                if isinstance(child, ast.Assert) and "cuda" in ast.unparse(child.test):
                    node.body = node.body[:i]
                    break
        node.body = [child for child in node.body
                     if not (isinstance(child, ast.Assert) and "cuda" in ast.unparse(child.test))]
        return self.generic_visit(node)


@functools.lru_cache(maxsize=None)
def preset_code(path):
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name.startswith("set_")]
    nodes += [node for node in tree.body if isinstance(node, ast.If)
              and ast.unparse(node.test).startswith("args.wandb_group_name ")]
    nodes += [node for node in tree.body if isinstance(node, ast.If)
              and ast.unparse(node.test) == "args.arch == 'OneLayerAttention' and args.num_hidden_features_mlp is None"]
    module = RemoveGpuChecks().visit(ast.Module(body=nodes, type_ignores=[]))
    return compile(ast.fix_missing_locations(module), str(path), "exec")


def effective_args(path, args):
    if path.name in {"main_dam.py", "main_infonce.py"}:
        return args
    namespace = {"args": args, "np": np}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(preset_code(path), namespace)
    return args


class ExperimentConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.configs = [(path, launcher.load_config(path))
                       for path in sorted((ROOT / "configs" / "experiments").glob("*.json"))]
        cls.parsers = {c["entrypoint"]: parser_for(ROOT / c["entrypoint"]) for _, c in cls.configs}

    def test_every_eligible_run_has_valid_cli_and_preserves_configured_parameters(self):
        """Every eligible launcher command is accepted and its values survive presets."""
        count = 0
        for path, config in self.configs:
            parser = self.parsers[config["entrypoint"]]
            for index in launcher.eligible_indices(config):
                with self.subTest(config=path.name, index=index):
                    command = launcher.build_command(config, index, sys.executable, "/tmp/test-results")
                    args = parser.parse_args(command[3:])
                    before = vars(args).copy()
                    effective_args(ROOT / config["entrypoint"], args)
                    supplied = set(config["arguments"]) | set(launcher.configurations(config)[index]) | {"savedir"}
                    destinations = {parser._option_string_actions["--" + name].dest for name in supplied}
                    for name in destinations:
                        value = before[name]
                        if name in {"num_heads", "num_hidden_features", "num_hidden_features_mlp"}:
                            # Preset sweep axes are checked independently below.
                            continue
                        self.assertEqual(getattr(args, name), value, f"{path.name} overrides {name}")
                    count += 1
        self.assertGreater(count, 800)

    def test_fork_sweep_decodes_feature_fastest(self):
        for name, heads, features in [
            ("figure4_fork_progress.json", [1, 2, 6, 10, 14, 18, 22], [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]),
            ("s8_zero_attention_probe.json", [1, 2, 10, 18], [1, 4, 22, 107, 512]),
            ("figure4_swap_mlp.json", [1, 2, 6, 10, 14, 18], [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]),
        ]:
            config = launcher.load_config(ROOT / "configs" / "experiments" / name)
            parser = self.parsers[config["entrypoint"]]
            for index in launcher.eligible_indices(config):
                with self.subTest(config=name, index=index):
                    command = launcher.build_command(config, index, sys.executable, "/tmp/test-results")
                    args = effective_args(ROOT / config["entrypoint"], parser.parse_args(command[3:]))
                    self.assertEqual(args.num_heads, heads[index // len(features)])
                    self.assertEqual(args.num_hidden_features, features[index % len(features)])

    def test_head_and_width_sweeps_decode_every_index(self):
        specifications = {
            "figure2_12layer_noresample.json": (list(range(2, 15)), [2, 4, 8, 16, 32, 64, 128, 256]),
            "figure2_12layer_doresample.json": (list(range(2, 15, 2)), [2, 4, 8, 16, 32, 64, 128, 256]),
            "figure2_1layer_noresample.json": ([1] + list(range(2, 17, 2)), [2 ** i for i in range(10)]),
            "figure2_1layer_doresample.json": ([1] + list(range(2, 17, 2)), [2 ** i for i in range(10)]),
            "figure4_linear_probe.json": ([1, 2, 6, 10, 14], [1, 4, 16, 64, 512]),
            "s2_uniform_model_size.json": (list(range(1, 13)), [2 ** i for i in range(1, 13)]),
        }
        for name, (heads, widths) in specifications.items():
            config = launcher.load_config(ROOT / "configs" / "experiments" / name)
            parser = self.parsers[config["entrypoint"]]
            for index in launcher.eligible_indices(config):
                with self.subTest(config=name, index=index):
                    command = launcher.build_command(config, index, sys.executable, "/tmp/test-results")
                    args = effective_args(ROOT / config["entrypoint"], parser.parse_args(command[3:]))
                    self.assertEqual(args.num_heads, heads[index % len(heads)])
                    self.assertEqual(args.num_hidden_features, widths[index // len(heads)])

    def test_explicit_figure7_architectures_reach_training_unchanged(self):
        config = launcher.load_config(ROOT / "configs" / "experiments" / "figure7_full_mlp_legacy.json")
        parser = self.parsers[config["entrypoint"]]
        for index, run in enumerate(launcher.configurations(config)):
            with self.subTest(index=index):
                command = launcher.build_command(config, index, sys.executable, "/tmp/test-results")
                args = effective_args(ROOT / config["entrypoint"], parser.parse_args(command[3:]))
                for key, value in run.items():
                    self.assertEqual(getattr(args, key), value)

    def test_figure7_filter_matches_actual_head_width_product(self):
        config = launcher.load_config(ROOT / "configs" / "experiments" / "figure7_variable_mlp.json")
        parser = self.parsers[config["entrypoint"]]
        selected = set(launcher.eligible_indices(config))
        products = set(config["sweep"]["allowed_head_feature_products"]["products"])
        for index in range(len(launcher.configurations(config))):
            with self.subTest(index=index):
                command = launcher.build_command(config, index, sys.executable, "/tmp/test-results")
                try:
                    args = effective_args(ROOT / config["entrypoint"], parser.parse_args(command[3:]))
                except (SystemExit, ValueError):
                    self.assertNotIn(index, selected)
                    continue
                self.assertEqual(index in selected, args.num_heads * args.num_hidden_features in products)
                heads = [9] + list(range(2, 17, 2))
                self.assertEqual(args.num_heads, heads[index % len(heads)])
                self.assertEqual(args.num_hidden_features, 2 ** ((index // len(heads)) % 10))
                fraction = np.linspace(0.1, 5.0, 10)[index // (len(heads) * 10)]
                self.assertEqual(args.num_hidden_features_mlp,
                                 int(args.num_heads * args.num_hidden_features * fraction))


if __name__ == "__main__":
    unittest.main()
