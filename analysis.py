import os
import sys
import shutil
import glob
import argparse
import math
import pickle
import re
import gc
from multiprocessing import Pool
from collections import defaultdict
import copy
import time
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.special import logit
# The experiment launcher writes transformer checkpoints under results/cache
# and DAM checkpoints under results. Override the root for another cluster.
RESULTS_DIR = os.path.abspath(os.environ.get("ICL_RESULTS_DIR", "./results"))

# Conda TeX executables may need libraries and font maps from their own prefix.
# Respect settings supplied by the user and do not assume a particular cluster.
prefix_bin = os.path.join(sys.prefix, "bin")
prefix_lib = os.path.join(sys.prefix, "lib")
prefix_share = os.path.join(sys.prefix, "share")
if os.path.isdir(prefix_bin):
    os.environ["PATH"] = prefix_bin + os.pathsep + os.environ.get("PATH", "")
if os.path.isdir(prefix_lib):
    os.environ["LD_LIBRARY_PATH"] = prefix_lib + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
if os.path.isfile(os.path.join(prefix_share, "texmf.cnf")):
    for variable, directory in (("TEXMFVAR", "texmf-var"), ("TEXMFCONFIG", "texmf-config"),
                                ("TEXMFDIST", "texmf-dist"), ("TEXMFLOCAL", "texmf-local")):
        os.environ.setdefault(variable, os.path.join(prefix_share, directory))
    os.environ.setdefault("TEXMFCNF", prefix_share + os.pathsep)

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = f"/tmp/matplotlib_{os.getuid()}" 

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm
import matplotlib.ticker as mticker
import seaborn as sns
import torch

import gpt
import utils


# Use one perceptually uniform scale whenever color represents model size.
NUM_PARAMETERS_CMAP = "viridis"
# Keep Figure 4 accuracy heatmaps visually distinct from the model-size scale.
FIGURE4_ACCURACY_CMAP = sns.color_palette("mako", as_cmap=True)

# Figure 5 is the typography reference: each of its eight attention panels is
# 1.26 inches square and uses the following font sizes.  Other figures scale
# these values with their physical panel size immediately before saving.
FIGURE5_REFERENCE_PANEL_INCHES = 1.26
MAX_TYPOGRAPHY_SCALE = 2.0
PANEL_TITLE_FONT_SIZE = 10.0
AXIS_LABEL_FONT_SIZE = 9.5
TICK_LABEL_FONT_SIZE = 7.5
LEGEND_FONT_SIZE = 9.5
LEGEND_TITLE_FONT_SIZE = 9.5


def apply_figure_style():
    """Apply consistent publication-quality styling with LaTeX for all figures."""
    utils.set_plot_style()
    mpl.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage{times}\usepackage{amsmath}\usepackage{amssymb}",
        "font.size": AXIS_LABEL_FONT_SIZE,
        "axes.labelsize": AXIS_LABEL_FONT_SIZE,
        "axes.titlesize": PANEL_TITLE_FONT_SIZE,
        "xtick.labelsize": TICK_LABEL_FONT_SIZE,
        "ytick.labelsize": TICK_LABEL_FONT_SIZE,
        "legend.fontsize": LEGEND_FONT_SIZE,
        "legend.title_fontsize": LEGEND_TITLE_FONT_SIZE,
        "figure.titlesize": PANEL_TITLE_FONT_SIZE,
    })


apply_figure_style()


def _figure5_relative_font_scale(axis, fig):
    """Return the Figure-5-relative font scale for one non-colorbar panel."""
    position = axis.get_position()
    width_inches = position.width * fig.get_figwidth()
    height_inches = position.height * fig.get_figheight()

    # Sequence strips have equal-aspect image axes that are only one cell high;
    # their descriptive titles instead scale with the strip width.
    if not axis.get_xticks().size and not axis.get_yticks().size:
        characteristic_size = width_inches
    else:
        characteristic_size = np.sqrt(width_inches * height_inches)
    return float(np.clip(
        characteristic_size / FIGURE5_REFERENCE_PANEL_INCHES,
        1.0,
        MAX_TYPOGRAPHY_SCALE,
    ))


def standardize_figure_typography(fig):
    """Scale every figure's typography relative to Figure 5 panel geometry."""
    primary_axes = [
        axis for axis in fig.axes
        if axis.get_visible() and not hasattr(axis, "_colorbar")
    ]
    primary_scales = {
        axis: getattr(
            axis,
            "_typography_scale_override",
            _figure5_relative_font_scale(axis, fig),
        )
        for axis in primary_axes
    }
    colorbar_scale = float(np.median(list(primary_scales.values()))) if primary_scales else 1.0

    for axis in fig.axes:
        if not axis.get_visible():
            continue
        scale = colorbar_scale if hasattr(axis, "_colorbar") else primary_scales[axis]

        for tick_label in (*axis.get_xticklabels(), *axis.get_yticklabels()):
            tick_label.set_fontsize(TICK_LABEL_FONT_SIZE * scale)

        title = axis.title
        if title.get_text():
            title.set_fontsize(PANEL_TITLE_FONT_SIZE * scale)
        for label in (axis.xaxis.label, axis.yaxis.label):
            if label.get_text():
                label.set_fontsize(AXIS_LABEL_FONT_SIZE * scale)

        legend = axis.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(LEGEND_FONT_SIZE * scale)
            legend.get_title().set_fontsize(LEGEND_TITLE_FONT_SIZE * scale)


# =============================================================================
# FIGURE 1: Experimental Setup (Sequences + Zipf Sampling)
# =============================================================================
def create_figure1():
    """Create publication-quality Figure 1 for NeurIPS camera-ready submission.

    Combines:
      (a) Sequence visualizations (4 example binary sequences of length 20)
      (b) Zipfian sampling distribution (Empirical vs Expected counts on log-log scale)
    Aligned side-by-side with bold subpanel labels 'a' and 'b'.
    """
    apply_figure_style()

    fig = plt.figure(figsize=(10.5, 3.6))

    outer_gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.12, 1.0],
        wspace=0.30,
        left=0.06,
        right=0.97,
        top=0.80,
        bottom=0.16,
    )

    # Panel (a): Sequences visualization
    left_gs = outer_gs[0].subgridspec(4, 1, hspace=0.95)

    seqs = np.array([
        [0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0],
        [0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1],
    ])

    axes_a = []
    for i in range(4):
        ax = fig.add_subplot(left_gs[i])
        ax.imshow(seqs[i : i + 1], aspect="equal", cmap="rocket", extent=[0, 20, 0, 1])
        ax.set_title(f"Sequence {i}", fontsize=PANEL_TITLE_FONT_SIZE, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        axes_a.append(ax)

    # Panel (b): Zipf distribution
    ax_b = fig.add_subplot(outer_gs[1])

    K = 1000
    Pk = 1.0 / np.arange(1, K + 1)
    Pk = Pk / Pk.sum()

    n = 100000
    np.random.seed(42)
    samples = np.random.choice(np.arange(1, K + 1), size=n, p=Pk)
    counts = np.bincount(samples, minlength=K + 1)[1:]

    ranks = np.arange(1, K + 1)
    expected = n * Pk
    mask = counts > 0

    ax_b.loglog(ranks[mask], counts[mask], ".", ms=3.0, color="C0", alpha=0.85, label="Empirical")
    ax_b.loglog(ranks, expected, "-", lw=1.8, color="C1", label="Expected")

    ax_b.set_xlabel("Rank", fontsize=AXIS_LABEL_FONT_SIZE)
    ax_b.set_ylabel("Count", fontsize=AXIS_LABEL_FONT_SIZE)
    ax_b.set_title("Rank Frequency Distribution", fontsize=PANEL_TITLE_FONT_SIZE, pad=6)

    ax_b.set_xlim(0.8, 1500)
    ax_b.set_ylim(4, 25000)

    ax_b.tick_params(
        which="both", direction="in", top=False, right=False,
        labelsize=TICK_LABEL_FONT_SIZE,
    )
    ax_b.legend(
        loc="upper right",
        frameon=True,
        edgecolor="0.6",
        facecolor="white",
        framealpha=0.9,
        fontsize=LEGEND_FONT_SIZE,
        handletextpad=0.5,
        borderpad=0.4,
    )

    fig.canvas.draw()
    pos_a = axes_a[0].get_position()
    pos_b = ax_b.get_position()

    t_b = ax_b.title.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
    t_a = axes_a[0].title.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())

    # Consistent panel label options matching Figure 4:
    dx_a = -0.025
    dx_b = -0.070
    y_label = max(t_a.y1, t_b.y1) + 0.040

    fig.text(pos_a.x0 + dx_a, y_label, r"\textbf{a}", fontsize=22, fontweight="bold", va="bottom", ha="left")
    fig.text(pos_b.x0 + dx_b, y_label, r"\textbf{b}", fontsize=22, fontweight="bold", va="bottom", ha="left")

    return fig


# =============================================================================
# FIGURE 2 HELPERS: Data Extraction, Power Law Fitting & Subpanels
# =============================================================================
class DummyParam(int):
    """Integer subclass providing .numel() compatibility with PyTorch tensors."""
    def numel(self):
        return int(self)


class DummyArray(np.ndarray):
    """NumPy array subclass that drops massive attention weight buffers (~160MB/epoch)
    to prevent memory exhaustion when reading checkpoints saved with --is_save_attn_reps."""
    def __setstate__(self, state):
        shape = state[1]
        # Any array with > 100,000 elements (e.g. attn_reps 256*16*100*100 = 40.9M floats)
        # is discarded without keeping its ~164MB memory buffer.
        if math.prod(shape) > 100_000:
            super().__setstate__((state[0], (0,), state[2], state[3], b""))
        else:
            super().__setstate__(state)


def _dummy_numpy_reconstruct(subtype, shape, dtype):
    return DummyArray(shape, dtype=dtype)


class FastParamUnpickler(pickle.Unpickler):
    """Bypasses full tensor buffer allocation while preserving logs, args,
    and exact parameter counts from tensor shape metadata (~50x speedup).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.param_count = 0

    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: None
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            def dummy_tensor(storage, storage_offset, size, stride, requires_grad, backward_hooks, metadata=None):
                numel = math.prod(size)
                self.param_count += numel
                return DummyParam(numel)
            return dummy_tensor
        if module in ("numpy.core.multiarray", "numpy._core.multiarray") and name == "_reconstruct":
            return _dummy_numpy_reconstruct
        return super().find_class(module, name)


def fit_power_law(x, y):
    """Fits power law: y = C * x^gamma via linear regression in log-log space."""
    mask = (x > 0) & (y > 0)
    x_val = x[mask]
    y_val = y[mask]

    if len(x_val) < 2:
        return np.nan, np.nan, None, None

    log_x = np.log10(x_val)
    log_y = np.log10(y_val)

    gamma, log_c = np.polyfit(log_x, log_y, 1)

    x_fit = np.logspace(log_x.min(), log_x.max(), 100)
    y_fit = (10 ** log_c) * (x_fit ** gamma)

    return gamma, 10 ** log_c, x_fit, y_fit


def plot_train_loss_panel(ax, df_data, title="Train Loss vs. Number of Parameters", d_min=None, d_max=None, cbar_ax=None):
    ax.set_title(title, fontsize=15, pad=8)
    """Plots Train Loss vs Number of Parameters with markers for H and colormap for D."""
    H_values = sorted(df_data["num_heads"].unique())
    markers = ["o", "X", "s", "P", "D", "^", "v", "<", ">", "*", "h", "8"]
    marker_map = {h: markers[i % len(markers)] for i, h in enumerate(H_values)}

    D_values = sorted(df_data["num_hidden_features"].unique())
    # Panel (a) encodes hidden-feature count D by colour in both rows.
    # Use the project-wide perceptually uniform parameter-style palette.
    cmap = plt.get_cmap("viridis")
    dmin = min(D_values) if d_min is None else d_min
    dmax = max(D_values) if d_max is None else d_max
    norm = mpl.colors.LogNorm(vmin=dmin, vmax=dmax)

    for h in H_values:
        dh = df_data[df_data["num_heads"] == h]
        if len(dh) == 0:
            continue
        ax.scatter(
            dh["num_parameters"],
            dh["train_loss"],
            c=dh["num_hidden_features"],
            cmap=cmap,
            norm=norm,
            marker=marker_map[h],
            s=55,
            alpha=0.9,
            edgecolors="none",
        )

    ax.set_xlabel("Number of Parameters", fontsize=14)
    ax.set_ylabel("Train Loss", fontsize=14)
    ax.set_xscale("log")
    ax.tick_params(which="both", direction="in", top=False, right=False, labelsize=11)

    # Colorbar for D
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    if cbar_ax is not None:
        cbar = ax.figure.colorbar(sm, cax=cbar_ax)
    else:
        cbar = ax.figure.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(r"$D$", fontsize=14)
    cbar.ax.tick_params(labelsize=11)
    cbar.ax.yaxis.set_major_formatter(mpl.ticker.LogFormatterMathtext(base=10))

    # Legend for H
    style_handles = [
        Line2D([0], [0], marker=marker_map[h], linestyle="None", color="black",
               markerfacecolor="black", markersize=6.5, label=str(h))
        for h in H_values
    ]
    ax.legend(
        handles=style_handles,
        title=r"$H$",
        loc="upper right",
        ncol=2,
        framealpha=0.9,
        fontsize=9,
        title_fontsize=10.5,
        handletextpad=0.2,
        borderpad=0.25,
        columnspacing=0.5,
    )


def plot_power_law_panel(ax, capacity_df, iters_list, title, is_12layer=True):
    """Plots Maximum rank memorized vs Number of parameters with power law fits."""
    ax.set_title(title, fontsize=15, pad=8)
    colors = ["C0", "C1", "C2"]

    for i, iters in enumerate(iters_list):
        iter_col = "iters" if "iters" in capacity_df.columns else "iters_checkpoint"
        rank_col = "max_rank" if "max_rank" in capacity_df.columns else "max_rank_at_saturation"
        sub = capacity_df[capacity_df[iter_col] == iters].dropna(
            subset=["num_parameters", rank_col]
        )
        sub = sub[(sub["num_parameters"] > 0) & (sub[rank_col] > 0)]

        # Panel (b) strictly uses the exact same failure filters as Panel (a)

        if not is_12layer:
            sub = sub[(sub["num_parameters"] >= 1e4) & (sub["num_parameters"] <= 1e7)]

        p = sub["num_parameters"].values
        r = sub[rank_col].values
        c = colors[i % len(colors)]

        gamma, C, x_fit, y_fit = fit_power_law(p, r)

        ax.plot(p, r, "o", color=c, ms=6, alpha=0.9)
        if x_fit is not None:
            ax.plot(x_fit, y_fit, "--", color=c, lw=1.8, label=rf"$\gamma = {gamma:.2f}$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e4, 1e7)
    ax.set_ylim(1e1, 1.2e5)
    ax.set_xlabel("Number of parameters", fontsize=14)
    ax.set_ylabel("Maximum rank memorized", fontsize=14)
    ax.tick_params(which="both", direction="in", top=False, right=False, labelsize=11)
    ax.legend(loc="upper left", frameon=True, fontsize=11)


def plot_broken_axis_panel(ax1, ax2, avg_df, norm, title=None, palette=NUM_PARAMETERS_CMAP):
    """Plots broken axis accuracy vs iteration with data switch marker and break slashes."""
    left_df = avg_df[(avg_df["iteration"] >= 0) & (avg_df["iteration"] <= 10000)].copy()
    right_df = avg_df[(avg_df["iteration"] >= 40000) & (avg_df["iteration"] <= 50000)].copy()

    common_kwargs = dict(
        y="accuracy",
        hue="num_parameters",
        palette=palette,
        hue_norm=norm,
        legend=False,
        errorbar=None,
    )

    sns.lineplot(
        data=left_df.sort_values(["num_parameters", "iteration"]),
        x="iteration",
        ax=ax1,
        **common_kwargs,
    )
    sns.lineplot(
        data=right_df.sort_values(["num_parameters", "iteration"]),
        x="iteration",
        ax=ax2,
        **common_kwargs,
    )

    # Red dashed line for data switch
    ax2.axvline(x=40000, color="red", linestyle="--", linewidth=1.5)
    # Match the panel-title size after Figure 2's Figure-5-relative typography
    # standardization (the large Figure 2 panels receive the 2x scale).
    ax2.text(
        40250,
        0.93,
        "Data Switch",
        color="red",
        fontsize=PANEL_TITLE_FONT_SIZE * MAX_TYPOGRAPHY_SCALE,
    )

    ax1.set_xlim(0, 10000)
    ax2.set_xlim(40000, 50000)
    ax1.set_ylim(0.45, 1.0)

    ax1.set_ylabel("Accuracy", fontsize=14)
    ax2.set_ylabel("")
    ax1.set_xlabel("")
    ax2.set_xlabel("")

    # Broken axis tick formatting: [0.0, 0.5, 1.0] and [4.0, 4.5, 5.0]
    ax1.set_xticks([0, 5000, 10000])
    ax2.set_xticks([40000, 45000, 50000])
    ax1.set_xticklabels(["0.0", "0.5", "1.0"])
    ax2.set_xticklabels(["4.0", "4.5", "5.0"])

    left_labels = ax1.get_xticklabels()
    right_labels = ax2.get_xticklabels()
    left_labels[-1].set_horizontalalignment("right")
    right_labels[0].set_horizontalalignment("left")

    ax1.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax1.tick_params(which="both", direction="in", top=False, right=False, labelsize=11)
    ax2.tick_params(which="both", direction="in", top=False, right=False, labelleft=False, left=False, labelsize=11)

    # Draw broken axis diagonal slash marks
    d = 0.015
    kwargs = dict(transform=ax1.transAxes, color="k", clip_on=False, linewidth=1.0)
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (-d, +d), **kwargs)
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)

    # Shared common x-axis label between the two subplots
    ax1.text(
        1.04,
        -0.10,
        r"Iteration $\times 10^4$",
        transform=ax1.transAxes,
        ha="center",
        va="top",
        # This shared label is a free text artist, so it does not pass through
        # ``standardize_figure_typography`` like the panel-a/b axis labels.
        fontsize=AXIS_LABEL_FONT_SIZE * MAX_TYPOGRAPHY_SCALE,
    )

    if title:
        ax1.set_title(title, x=1.04, y=1.0, pad=8, fontsize=15)


def prepare_figure2_panel_c_data(_df, max_rank=3000):
    """Apply the panel-c rank/iteration filters and average sequence accuracy."""
    plot_df = pd.DataFrame(_df).copy()
    plot_df["num_parameters"] = pd.to_numeric(plot_df["num_parameters"], errors="coerce")
    plot_df["epoch"] = pd.to_numeric(plot_df["epoch"], errors="coerce")
    plot_df["sequence_rank"] = pd.to_numeric(plot_df["sequence_rank"], errors="coerce")
    plot_df["accuracy"] = pd.to_numeric(plot_df["accuracy"], errors="coerce")

    plot_df["iteration"] = plot_df["epoch"] * 1000
    plot_df = plot_df[
        (plot_df["sequence_rank"] >= 0)
        & (plot_df["sequence_rank"] <= max_rank)
    ].copy()

    return (
        plot_df
        .groupby(["num_parameters", "iteration"], as_index=False)["accuracy"]
        .mean()
    )


# Figure 2 top-row and bottom-row data provenance.
# Panels (a) and (b) use the no-resampling experiments;
# panel (c) uses the resampling experiments used for data-switch evaluation.
FIGURE2_TOP_AB_EXPERIMENTS = (
    "memo_apr19_zipf_gpt2_vary_num_hidden_features_num_heads_noresample",
)
FIGURE2_TOP_C_EXPERIMENTS = (
    "memo_may1_zipf_gpt2_vary_num_hidden_features_num_heads_doresample",
)
FIGURE2_TOP_AB_RECORD_GLOBS = [
    f"{RESULTS_DIR}/cache/{experiment}/*"
    for experiment in (
        [FIGURE2_TOP_AB_EXPERIMENTS]
        if isinstance(FIGURE2_TOP_AB_EXPERIMENTS, str)
        else FIGURE2_TOP_AB_EXPERIMENTS
    )
]
FIGURE2_TOP_C_RECORD_GLOBS = [
    f"{RESULTS_DIR}/cache/{experiment}/*"
    for experiment in (
        [FIGURE2_TOP_C_EXPERIMENTS]
        if isinstance(FIGURE2_TOP_C_EXPERIMENTS, str)
        else FIGURE2_TOP_C_EXPERIMENTS
    )
]
# Retain singular aliases for backward compatibility
FIGURE2_TOP_EXPERIMENT = (
    FIGURE2_TOP_AB_EXPERIMENTS[0]
    if isinstance(FIGURE2_TOP_AB_EXPERIMENTS, (tuple, list))
    else FIGURE2_TOP_AB_EXPERIMENTS
)
FIGURE2_TOP_RECORD_GLOB = FIGURE2_TOP_AB_RECORD_GLOBS[0]
FIGURE2_TOP_C_EXPERIMENT = (
    FIGURE2_TOP_C_EXPERIMENTS[0]
    if isinstance(FIGURE2_TOP_C_EXPERIMENTS, (tuple, list))
    else FIGURE2_TOP_C_EXPERIMENTS
)
FIGURE2_TOP_C_RECORD_GLOB = FIGURE2_TOP_C_RECORD_GLOBS[0]
FIGURE2_TOP_A_CACHE = "./cache/df_train_loss_12layer.csv"
FIGURE2_TOP_B_CACHE = "./cache/df_capacity_12layer.csv"
FIGURE2_TOP_C_CACHE = "./cache/df_dataswitch_12layer.csv"
FIGURE2_TOP_C_MAX_PARAMETERS = 2e6
FIGURE2_BOTTOM_A_EXPERIMENTS = (
    "memo_jan20_zipf_onelayerattention_vary_num_hidden_features_num_heads_noresample",
)
FIGURE2_BOTTOM_B_EXPERIMENT = (
    "memo_jan20_zipf_onelayerattention_vary_num_hidden_features_num_heads_noresample"
)

FIGURE2_BOTTOM_C_EXPERIMENT = (
    "memo_jan20_zipf_onelayerattention_vary_num_hidden_features_num_heads_doresample"
)
FIGURE2_BOTTOM_MAX_PARAMETERS = 7e7
FIGURE2_BOTTOM_A_RECORD_GLOBS = [
    f"{RESULTS_DIR}/cache/{experiment}/*"
    for experiment in FIGURE2_BOTTOM_A_EXPERIMENTS
]
FIGURE2_BOTTOM_B_RECORD_GLOB = (
    f"{RESULTS_DIR}/cache/"
    f"{FIGURE2_BOTTOM_B_EXPERIMENT}/*"
)
FIGURE2_BOTTOM_C_RECORD_GLOB = (
    f"{RESULTS_DIR}/cache/"
    f"{FIGURE2_BOTTOM_C_EXPERIMENT}/*"
)
# The source experiment is encoded in the filename so a legacy Jan. 24--26
# cache cannot be reused for panel (a).
FIGURE2_BOTTOM_A_CACHE = "./cache/df_train_loss_1layer_jan20_noresample.csv"
FIGURE2_BOTTOM_B_CACHE = "./cache/df_capacity_1layer_jan20_noresample.csv"
# This evaluation cache is rebuilt from the Jan 20 one-layer resampling logs.
FIGURE2_BOTTOM_C_CACHE = "./cache/df_dataswitch_1layer_jan20_doresample.csv"
FIGURE2_BOTTOM_C_CACHE_COLUMNS = {
    "num_parameters",
    "epoch",
    "sequence_rank",
    "accuracy",
}


class FastFigure2TopUnpickler(pickle.Unpickler):
    """Load Figure 2 logs while retaining only tensor shape information."""

    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda value: None
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            def dummy_tensor(storage, storage_offset, size, stride,
                             requires_grad, backward_hooks, metadata=None):
                class DummyTensor:
                    def numel(self):
                        return math.prod(size)
                return DummyTensor()
            return dummy_tensor
        return super().find_class(module, name)


def _extract_figure2_top_run(folder):
    """Extract the three top-row Figure 2 data tables from one raw run."""
    record_paths = [
        path for path in glob.glob(os.path.join(folder, "*.pkl"))
        if not path.endswith(("_all_sequences.pkl", "_model_final.pkl")) and "_model_iter_" not in os.path.basename(path)
    ]
    if len(record_paths) != 1:
        return None
    try:
        with open(record_paths[0], "rb") as file:
            data = FastFigure2TopUnpickler(file).load()
    except Exception:
        return None

    logs = data.get("logs", [])
    if len(logs) < 60:
        return None
    args = data.get("args", {})
    num_hidden_features = args.get("num_hidden_features", 1)
    num_heads = args.get("num_heads", 1)
    width = num_hidden_features * num_heads
    model_parameters = sum(
        parameter.numel() for parameter in data.get("model", {}).values()
        if hasattr(parameter, "numel")
    )
    if model_parameters == 0:
        num_layers = args.get("num_layers", 12)
        context_length = args.get("len_context", 100)
        vocab_size = args.get("vocab_size", 2)
        # GPT has token and position embeddings, ``num_layers`` transformer
        # blocks, a final layer norm, and a vocabulary-sized output head.
        model_parameters = (
            num_layers * (12 * width ** 2 + 13 * width)
            + (context_length + 2 * vocab_size + 2) * width
            + vocab_size
        )

    if model_parameters >= FIGURE2_TOP_C_MAX_PARAMETERS:
        return []

    final_loss = logs[-1].get("train_loss")
    if final_loss is None:
        return None
    failed = (
        (model_parameters > 3e6 and final_loss > 0.21)
        or (model_parameters > 1e6 and final_loss > 0.4)
    )
    capacity_rows = []
    data_switch_rows = []
    final_test_accuracy = None
    for log in logs:
        epoch = log.get("epoch")
        test_metrics = log.get("test_metrics", {})
        accuracy = test_metrics.get("accuracy")
        sequence_rank = test_metrics.get("sequence_rank")
        if accuracy is None or sequence_rank is None:
            continue
        accuracy = np.asarray(accuracy)
        sequence_rank = np.asarray(sequence_rank) * 10
        if len(accuracy) >= 1000:
            final_test_accuracy = float(accuracy[:1000].mean())

        if epoch in [60, 120, 190]:
            learned_ranks = sequence_rank[accuracy > 0.90]
            if len(learned_ranks):
                capacity_rows.append({
                    "num_parameters": model_parameters,
                    "epoch": epoch,
                    "iters": epoch * 1000,
                    "max_rank": int(learned_ranks.max()),
                    "failed": failed,
                })

        if epoch is not None and (0 <= epoch <= 10 or 40 <= epoch <= 50):
            first_3000 = sequence_rank <= 3000
            if np.any(first_3000):
                data_switch_rows.append({
                    "num_parameters": model_parameters,
                    "iteration": epoch * 1000,
                    "accuracy": float(accuracy[first_3000].mean()),
                    "failed": failed,
                })

    return {
        "loss": {
            "folder": os.path.basename(folder),
            "num_parameters": model_parameters,
            "num_heads": num_heads,
            "num_hidden_features": num_hidden_features * num_heads,
            "train_loss": final_loss,
            "epoch": logs[-1].get("epoch", len(logs) - 1),
            "failed": failed,
            "test_accuracy": final_test_accuracy,
        },
        "capacity": capacity_rows,
        "data_switch": data_switch_rows,
    }


def _extract_figure2_top_c_run(folder):
    """Extract panel-c data-switch accuracy from one GPT-2 resampling run."""
    record_paths = [
        path for path in glob.glob(os.path.join(folder, "*.pkl"))
        if not path.endswith(("_all_sequences.pkl", "_model_final.pkl")) and "_model_iter_" not in os.path.basename(path)
    ]
    if len(record_paths) != 1:
        return []
    try:
        with open(record_paths[0], "rb") as file:
            data = FastFigure2TopUnpickler(file).load()
    except Exception:
        return []

    if "logs" not in data or len(data["logs"]) < 10:
        return []
    logs = data["logs"]
    args = data.get("args", {})
    num_hidden_features = args.get("num_hidden_features", 1)
    num_heads = args.get("num_heads", 1)
    width = num_hidden_features * num_heads
    model_parameters = sum(
        parameter.numel() for parameter in data.get("model", {}).values()
        if hasattr(parameter, "numel")
    )
    if model_parameters == 0:
        num_layers = args.get("num_layers", 12)
        context_length = args.get("len_context", 100)
        vocab_size = args.get("vocab_size", 2)
        model_parameters = (
            num_layers * (12 * width ** 2 + 13 * width)
            + (context_length + 2 * vocab_size + 2) * width
            + vocab_size
        )

    if model_parameters >= FIGURE2_TOP_C_MAX_PARAMETERS:
        return []

    final_loss = logs[-1].get("train_loss")
    failed = False
    if final_loss is not None:
        failed = (
            (model_parameters > 3e6 and final_loss > 0.21)
            or (model_parameters > 1e6 and final_loss > 0.4)
        )

    data_switch_rows = []
    for log in logs:
        epoch = log.get("epoch")
        if epoch is None or not (0 <= epoch <= 10 or 40 <= epoch <= 50):
            continue
        test_metrics = log.get("test_metrics", {})
        accuracy = test_metrics.get("accuracy")
        sequence_rank = test_metrics.get("sequence_rank")
        if accuracy is None or sequence_rank is None:
            continue
        accuracy = np.asarray(accuracy)
        sequence_rank = np.asarray(sequence_rank) * 10
        first_3000 = sequence_rank <= 3000
        if np.any(first_3000):
            data_switch_rows.append({
                "num_parameters": model_parameters,
                "iteration": epoch * 1000,
                "accuracy": float(accuracy[first_3000].mean()),
                "failed": failed,
            })
    return data_switch_rows


def get_default_workers(requested=None, max_cap=None):
    """Determine effective worker count respecting Slurm allocations."""
    if requested is not None and requested > 0:
        return min(requested, max_cap) if max_cap else requested
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        try:
            count = int(slurm_cpus)
            return min(count, max_cap) if max_cap else count
        except ValueError:
            pass
    available = os.cpu_count() or 8
    return min(available, max_cap) if max_cap else available


def regenerate_figure2_top_ab_cache(workers=None):
    """Rebuild the top-row panels (a, b) Figure 2 caches from raw GPT-2 records."""
    all_folders = sorted({
        folder
        for record_glob in FIGURE2_TOP_AB_RECORD_GLOBS
        for folder in glob.glob(record_glob)
    })
    if not all_folders:
        raise FileNotFoundError(
            f"No Figure 2 top-row (a, b) records match any of: {FIGURE2_TOP_AB_RECORD_GLOBS}."
        )
    print(
        "Regenerating Figure 2 top-row (a, b) caches from "
        f"{', '.join(FIGURE2_TOP_AB_EXPERIMENTS)} ({len(all_folders)} runs)"
    )
    num_workers = min(get_default_workers(workers), len(all_folders))
    with Pool(processes=num_workers) as pool:
        results = [
            result for result in pool.imap_unordered(_extract_figure2_top_run, all_folders)
            if result is not None
        ]
    if not results:
        raise RuntimeError("No usable Figure 2 top-row (a, b) records were found.")

    loss_frame = pd.DataFrame([result["loss"] for result in results])
    capacity_frame = pd.DataFrame(
        row for result in results for row in result["capacity"]
    )
    if loss_frame.empty or capacity_frame.empty:
        raise RuntimeError("The extracted Figure 2 top-row (a, b) data is incomplete.")
    os.makedirs("./cache", exist_ok=True)
    loss_frame.to_csv(FIGURE2_TOP_A_CACHE, index=False)
    capacity_frame.to_csv(FIGURE2_TOP_B_CACHE, index=False)
    print(f"Wrote {len(loss_frame)} rows to {FIGURE2_TOP_A_CACHE}")
    print(f"Wrote {len(capacity_frame)} rows to {FIGURE2_TOP_B_CACHE}")


def regenerate_figure2_top_c_cache(workers=None):
    """Rebuild the top-row panel (c) Figure 2 cache from raw GPT-2 resampling records."""
    all_folders = sorted({
        folder
        for record_glob in FIGURE2_TOP_C_RECORD_GLOBS
        for folder in glob.glob(record_glob)
    })
    if not all_folders:
        raise FileNotFoundError(
            f"No Figure 2 top-row panel (c) records match any of: {FIGURE2_TOP_C_RECORD_GLOBS}."
        )
    print(
        "Regenerating Figure 2 top-row panel (c) cache from "
        f"{', '.join(FIGURE2_TOP_C_EXPERIMENTS)} ({len(all_folders)} runs)"
    )
    data_switch_rows = []
    num_workers = min(get_default_workers(workers), len(all_folders))
    with Pool(processes=num_workers) as pool:
        for run_rows in pool.imap_unordered(_extract_figure2_top_c_run, all_folders):
            data_switch_rows.extend(run_rows)
    data_switch_frame = pd.DataFrame(data_switch_rows)
    if data_switch_frame.empty:
        raise RuntimeError("The extracted Figure 2 top-row panel (c) data is empty.")
    data_switch_frame = data_switch_frame[data_switch_frame["num_parameters"] < FIGURE2_TOP_C_MAX_PARAMETERS].copy()
    os.makedirs("./cache", exist_ok=True)
    data_switch_frame.to_csv(FIGURE2_TOP_C_CACHE, index=False)
    print(f"Wrote {len(data_switch_frame)} rows to {FIGURE2_TOP_C_CACHE}")


def regenerate_figure2_top_cache(workers=None):
    """Rebuild all top-row Figure 2 caches from raw GPT-2 records."""
    regenerate_figure2_top_ab_cache(workers=workers)
    regenerate_figure2_top_c_cache(workers=workers)


def filter_figure2_bottom_row_models(dataframe):
    """Exclude bottom-row models larger than the camera-ready cutoff
    and keep only runs where num_hidden_features_mlp_frac == 1
    (i.e. num_hidden_features_mlp == num_heads * num_hidden_features).
    """
    df = dataframe[dataframe["num_parameters"] <= FIGURE2_BOTTOM_MAX_PARAMETERS].copy()
    if "num_hidden_features_mlp" in df.columns:
        df = df[df["num_hidden_features_mlp"] == df["num_heads"] * df["num_hidden_features"]].copy()
    elif "num_heads" in df.columns and "num_hidden_features" in df.columns:
        w = df["num_heads"] * df["num_hidden_features"]
        expected_p = 6 * w ** 2 + 114 * w + 2
        df = df[df["num_parameters"] == expected_p].copy()
    return df


def filter_figure2_bottom_a_mlp_widths(dataframe):
    """Keep 1-layer panel-a models with MLP width in ``[W, 3 W]``.

    Here ``W = num_heads * num_hidden_features`` is the hidden dimension of
    the attention heads.  Older panel-a caches do not store
    ``num_hidden_features_mlp``.  For those caches, recover it exactly from
    the OneLayerAttention parameter count (with its three MLP layers) so the
    plotting criterion is independent of cache age.
    """
    required_columns = {"num_parameters", "num_heads", "num_hidden_features"}
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        raise ValueError(
            "Figure 2 bottom panel (a) is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    df = dataframe[dataframe["num_parameters"] <= FIGURE2_BOTTOM_MAX_PARAMETERS].copy()
    attention_width = df["num_heads"] * df["num_hidden_features"]
    if "num_hidden_features_mlp" in df.columns:
        mlp_width = df["num_hidden_features_mlp"]
    else:
        # P = 3 W^2 + 109 W + W M + 2 M^2 + 5 M + 2, where M is the
        # MLP width.  Solve the positive quadratic root for M.
        linear_coefficient = attention_width + 5
        constant_coefficient = (
            3 * attention_width ** 2 + 109 * attention_width + 2 - df["num_parameters"]
        )
        discriminant = linear_coefficient ** 2 - 8 * constant_coefficient
        mlp_width = (-linear_coefficient + np.sqrt(discriminant)) / 4

    return df[mlp_width.between(attention_width, 3 * attention_width)].copy()


def _extract_figure2_bottom_a_run(folder):
    """Extract one raw one-layer run without materializing its model weights."""
    record_paths = [
        path for path in glob.glob(os.path.join(folder, "*.pkl"))
        if not path.endswith(("_all_sequences.pkl", "_model_final.pkl")) and "_model_iter_" not in os.path.basename(path)
    ]
    if len(record_paths) != 1:
        return [], [], None
    try:
        with open(record_paths[0], "rb") as file:
            record = FastParamUnpickler(file)
            data = record.load()
    except Exception as error:
        return [], [], f"{record_paths[0]}: {error}"

    loss_rows = []
    capacity_rows = []
    args = data.get("args", {})
    try:
        # FastParamUnpickler deliberately omits tensor payloads.  Derive the
        # exact OneLayerAttention parameter count from the saved architecture
        # instead of summing its empty replacement ``model`` dictionary.
        model_parameters = _one_layer_parameter_count_from_args(args)
    except (KeyError, TypeError, ValueError) as error:
        return [], [], f"{record_paths[0]}: cannot count model parameters: {error}"
    if model_parameters > FIGURE2_BOTTOM_MAX_PARAMETERS:
        return [], [], None

    num_heads = args.get("num_heads")
    num_hidden_features = args.get("num_hidden_features")
    num_hidden_features_mlp = args.get("num_hidden_features_mlp")
    for log in data.get("logs", []):
        epoch = log.get("epoch")
        if epoch is None or "train_loss" not in log:
            continue
        test_metrics = log.get("test_metrics", {})
        accuracy = test_metrics.get("accuracy")
        # Evaluations are ordered by rank and retain every tenth sequence:
        # entries 0:1000 therefore cover ranks 0 through 9,990.  Match the
        # 12-layer S1 panel, which reports this same top-10,000-rank mean.
        accuracy_array = np.asarray(accuracy).reshape(-1) if accuracy is not None else None
        test_accuracy = (
            float(accuracy_array[:1000].mean())
            if accuracy_array is not None and len(accuracy_array) >= 1000
            else np.nan
        )
        loss_rows.append({
            "train_loss": log["train_loss"],
            "test_accuracy": test_accuracy,
            "test_accuracy_rank_limit": 10_000,
            "num_parameters": model_parameters,
            "num_hidden_features": num_hidden_features,
            "num_heads": num_heads,
            "num_hidden_features_mlp": num_hidden_features_mlp,
            "epoch": epoch,
        })

        if epoch not in [20, 50, 79]:
            continue
        sequence_ranks = test_metrics.get("sequence_rank")
        if accuracy is None or sequence_ranks is None:
            continue
        accuracy_array = np.asarray(accuracy)
        rank_array = np.asarray(sequence_ranks) * 10
        learned_ranks = rank_array[accuracy_array > 0.90]
        capacity_rows.append({
            "num_parameters": model_parameters,
            "num_heads": num_heads,
            "num_hidden_features": num_hidden_features,
            "num_hidden_features_mlp": num_hidden_features_mlp,
            "epoch": epoch,
            "iters": epoch * 1000,
            "max_rank": int(learned_ranks.max()) if len(learned_ranks) else 0,
        })
    return loss_rows, capacity_rows, None


def _one_layer_parameter_count_from_folder(folder):
    """Read the one-layer width from its run name without opening its pickle."""
    match = re.search(r"_hidden_(\d+)_nheads_(\d+)_", os.path.basename(folder))
    if match is None:
        return None
    hidden_features, num_heads = map(int, match.groups())
    width = hidden_features * num_heads
    # OneLayerAttention with context length 100, vocabulary size 2, and three
    # MLP layers: 6 * width^2 + 114 * width + 2 parameters.
    return 6 * width ** 2 + 114 * width + 2


def regenerate_figure2_bottom_a_cache(workers=None):
    """Rebuild Figure 2 bottom-row panel (a) from its raw records.

    The fast unpickler reads model tensor shapes without allocating model
    weights, then extracts epoch-wise training loss and 90%-accuracy capacity
    from ``FIGURE2_BOTTOM_A_EXPERIMENTS``.  Models above the camera-ready
    parameter cutoff are excluded before the cache is written.
    """
    loss_rows = []
    capacity_rows = []
    all_folders = sorted({
        folder
        for record_glob in FIGURE2_BOTTOM_A_RECORD_GLOBS
        for folder in glob.glob(record_glob)
    })
    if not all_folders:
        raise FileNotFoundError(
            "No Figure 2 bottom-row records match any of: "
            f"{FIGURE2_BOTTOM_A_RECORD_GLOBS}."
        )
    folders = []
    for folder in all_folders:
        parameter_count = _one_layer_parameter_count_from_folder(folder)
        if parameter_count is not None and parameter_count > FIGURE2_BOTTOM_MAX_PARAMETERS:
            continue
        folders.append(folder)
    print(
        "Regenerating Figure 2 bottom-row panel (a) cache from "
        f"{', '.join(FIGURE2_BOTTOM_A_EXPERIMENTS)} "
        f"({len(folders)} of {len(all_folders)} runs)"
    )

    if not folders:
        raise RuntimeError(
            "All Figure 2 bottom-row records exceed the "
            f"{FIGURE2_BOTTOM_MAX_PARAMETERS:g}-parameter limit."
        )
    num_workers = min(get_default_workers(workers, max_cap=4), len(folders))
    with Pool(processes=num_workers, maxtasksperchild=5) as pool:
        for run_loss_rows, run_capacity_rows, error in pool.imap_unordered(
            _extract_figure2_bottom_a_run, folders
        ):
            if error is not None:
                print(f"Skipping unreadable record {error}")
            loss_rows.extend(run_loss_rows)
            capacity_rows.extend(run_capacity_rows)

    loss_frame = pd.DataFrame(loss_rows)
    capacity_frame = pd.DataFrame(capacity_rows)
    if loss_frame.empty or capacity_frame.empty:
        raise RuntimeError(
            "No usable Figure 2 bottom-row panel (a) records were found in "
            f"{FIGURE2_BOTTOM_A_RECORD_GLOBS}."
        )

    loss_frame.to_csv(FIGURE2_BOTTOM_A_CACHE, index=False)
    print(f"Wrote {len(loss_frame)} rows to {FIGURE2_BOTTOM_A_CACHE}")


def _extract_figure2_bottom_b_run(folder):
    """Extract panel-b capacity values from one Jan 20 no-resampling run."""
    record_paths = [
        path for path in glob.glob(os.path.join(folder, "*.pkl"))
        if not path.endswith(("_all_sequences.pkl", "_model_final.pkl")) and "_model_iter_" not in os.path.basename(path)
    ]
    if len(record_paths) != 1:
        return [], None
    try:
        with open(record_paths[0], "rb") as file:
            data = FastParamUnpickler(file).load()
    except Exception as error:
        return [], f"{record_paths[0]}: {error}"

    if "logs" not in data or len(data["logs"]) < 10:
        return [], None

    args = data.get("args", {})
    try:
        num_parameters = _one_layer_parameter_count_from_args(args)
    except (KeyError, TypeError, ValueError) as error:
        return [], f"{record_paths[0]}: cannot count model parameters: {error}"
    if num_parameters > 7e7:
        return [], None
    if num_parameters > 1e6 and data["logs"][-1].get("train_loss", 0) > 0.3:
        return [], None

    logs = data["logs"]
    num_heads = args.get("num_heads")
    num_hidden_features = args.get("num_hidden_features")
    num_hidden_features_mlp = args.get("num_hidden_features_mlp")

    rows = []
    for log in logs:
        epoch = log.get("epoch")
        if epoch not in [20, 50, 79]:
            continue
        test_metrics = log.get("test_metrics", {})
        accuracy = test_metrics.get("accuracy")
        sequence_ranks = test_metrics.get("sequence_rank")
        if accuracy is None or sequence_ranks is None:
            continue
        accuracy_array = np.asarray(accuracy)
        # Jan. 20 one-layer records already store sequence ranks on the
        # original 1..K scale.  Rescaling again would create impossible ranks
        # above K=100,000 in panel (b).
        rank_array = np.asarray(sequence_ranks)
        learned_ranks = rank_array[accuracy_array > 0.90]
        rows.append({
            "num_parameters": num_parameters,
            "num_heads": num_heads,
            "num_hidden_features": num_hidden_features,
            "epoch": epoch,
            "iters": epoch * 1000,
            "max_rank": int(learned_ranks.max()) if len(learned_ranks) else 0,
        })
    return rows, None


def regenerate_figure2_bottom_b_cache(workers=None):
    """Rebuild Figure 2 bottom panel (b) from the Jan 20 no-resampling records."""
    all_folders = sorted(glob.glob(FIGURE2_BOTTOM_B_RECORD_GLOB))
    folders = []
    for folder in all_folders:
        parameter_count = _one_layer_parameter_count_from_folder(folder)
        if parameter_count is not None and parameter_count > FIGURE2_BOTTOM_MAX_PARAMETERS:
            continue
        folders.append(folder)
    if not folders:
        raise FileNotFoundError(
            "No Figure 2 bottom panel (b) records match "
            f"{FIGURE2_BOTTOM_B_RECORD_GLOB}."
        )
    print(
        "Regenerating Figure 2 bottom panel (b) cache from "
        f"{FIGURE2_BOTTOM_B_EXPERIMENT} ({len(folders)} of {len(all_folders)} runs)"
    )

    rows = []
    num_workers = min(get_default_workers(workers, max_cap=4), len(folders))
    with Pool(processes=num_workers) as pool:
        for run_rows, error in pool.imap_unordered(_extract_figure2_bottom_b_run, folders):
            if error is not None:
                print(f"Skipping unreadable record {error}")
            rows.extend(run_rows)
    cache_frame = pd.DataFrame(rows)
    if cache_frame.empty:
        raise RuntimeError("No usable Figure 2 bottom panel (b) records were found.")
    cache_frame.to_csv(FIGURE2_BOTTOM_B_CACHE, index=False)
    print(f"Wrote {len(cache_frame)} rows to {FIGURE2_BOTTOM_B_CACHE}")


def _extract_figure2_bottom_c_run(folder):
    """Extract per-sequence panel-c values from one Jan 20 resampling run."""
    record_paths = [
        path for path in glob.glob(os.path.join(folder, "*.pkl"))
        if not path.endswith(("_all_sequences.pkl", "_model_final.pkl")) and "_model_iter_" not in os.path.basename(path)
    ]
    if len(record_paths) != 1:
        return [], None
    try:
        with open(record_paths[0], "rb") as file:
            data = FastParamUnpickler(file).load()
    except Exception as error:
        return [], f"{record_paths[0]}: {error}"

    if "logs" not in data or "model" not in data or not data["logs"]:
        return [], None

    num_parameters = sum(p.numel() for k, p in data["model"].items())
    if num_parameters > 1e6 and data["logs"][-1]["train_loss"] > 0.3:
        return [], None
    if num_parameters > 7e7:
        return [], None
    if len(data["logs"]) < 10:
        return [], None

    rows = []
    for log in data["logs"]:
        epoch = log.get("epoch")
        if epoch is None or not (0 <= epoch <= 10 or 40 <= epoch <= 50):
            continue
        test_metrics = log.get("test_metrics", {})
        accuracy = test_metrics.get("accuracy")
        sequence_ranks = test_metrics.get("sequence_rank")
        if accuracy is None or sequence_ranks is None:
            continue
        accuracy = np.asarray(accuracy)
        sequence_ranks = np.asarray(sequence_ranks) * 10
        rows.extend(
            {
                "num_parameters": num_parameters,
                "epoch": epoch,
                "sequence_rank": sequence_rank,
                "accuracy": sequence_accuracy,
            }
            for sequence_rank, sequence_accuracy in zip(sequence_ranks, accuracy)
        )
    return rows, None


def regenerate_figure2_bottom_c_cache(workers=None):
    """Rebuild Figure 2 bottom panel (c) from the Jan 20 resampling records."""
    all_folders = sorted(glob.glob(FIGURE2_BOTTOM_C_RECORD_GLOB))
    folders = []
    for folder in all_folders:
        parameter_count = _one_layer_parameter_count_from_folder(folder)
        if parameter_count is not None and parameter_count > FIGURE2_BOTTOM_MAX_PARAMETERS:
            continue
        folders.append(folder)
    if not folders:
        raise FileNotFoundError(
            "No Figure 2 bottom panel (c) records match "
            f"{FIGURE2_BOTTOM_C_RECORD_GLOB}."
        )
    print(
        "Regenerating Figure 2 bottom panel (c) cache from "
        f"{FIGURE2_BOTTOM_C_EXPERIMENT} ({len(folders)} of {len(all_folders)} runs)"
    )

    rows = []
    num_workers = min(get_default_workers(workers), len(folders))
    with Pool(processes=num_workers) as pool:
        for run_rows, error in pool.imap_unordered(_extract_figure2_bottom_c_run, folders):
            if error is not None:
                print(f"Skipping unreadable record {error}")
            rows.extend(run_rows)
    cache_frame = pd.DataFrame(rows)
    if cache_frame.empty:
        raise RuntimeError("No usable Figure 2 bottom panel (c) records were found.")
    cache_frame.to_csv(FIGURE2_BOTTOM_C_CACHE, index=False)
    print(f"Wrote {len(cache_frame)} rows to {FIGURE2_BOTTOM_C_CACHE}")


def ensure_figure2_caches(force_regenerate=False, parallel=False, workers=None):
    """Create missing Figure 2 caches, or replace them when forced."""
    top_ab_caches = [FIGURE2_TOP_A_CACHE, FIGURE2_TOP_B_CACHE]
    tasks_to_run = []
    if force_regenerate or any(not os.path.isfile(path) for path in top_ab_caches):
        tasks_to_run.append(("top_ab", regenerate_figure2_top_ab_cache))
    if force_regenerate or not os.path.isfile(FIGURE2_TOP_C_CACHE):
        tasks_to_run.append(("top_c", regenerate_figure2_top_c_cache))
    bottom_a_cache_is_stale = False
    if os.path.isfile(FIGURE2_BOTTOM_A_CACHE):
        try:
            bottom_a_header = pd.read_csv(FIGURE2_BOTTOM_A_CACHE, nrows=100)
            required_columns = {
                "train_loss", "num_parameters", "num_hidden_features",
                "num_heads", "num_hidden_features_mlp", "epoch",
                "test_accuracy_rank_limit",
            }
            parameter_counts = pd.to_numeric(
                bottom_a_header.get("num_parameters"), errors="coerce"
            )
            bottom_a_cache_is_stale = (
                not required_columns.issubset(bottom_a_header.columns)
                or parameter_counts.isna().any()
                or (parameter_counts <= 0).any()
            )
        except (OSError, pd.errors.EmptyDataError):
            bottom_a_cache_is_stale = True
    if force_regenerate or not os.path.isfile(FIGURE2_BOTTOM_A_CACHE) or bottom_a_cache_is_stale:
        tasks_to_run.append(("bottom_a", regenerate_figure2_bottom_a_cache))
    if force_regenerate or not os.path.isfile(FIGURE2_BOTTOM_B_CACHE):
        tasks_to_run.append(("bottom_b", regenerate_figure2_bottom_b_cache))
    bottom_c_cache_is_stale = False
    if os.path.isfile(FIGURE2_BOTTOM_C_CACHE):
        try:
            bottom_c_cache_is_stale = not FIGURE2_BOTTOM_C_CACHE_COLUMNS.issubset(
                pd.read_csv(FIGURE2_BOTTOM_C_CACHE, nrows=0).columns
            )
        except (OSError, pd.errors.EmptyDataError):
            bottom_c_cache_is_stale = True
    if force_regenerate or not os.path.isfile(FIGURE2_BOTTOM_C_CACHE) or bottom_c_cache_is_stale:
        tasks_to_run.append(("bottom_c", regenerate_figure2_bottom_c_cache))

    if not tasks_to_run:
        return

    total_workers = get_default_workers(workers)
    if parallel and len(tasks_to_run) > 1:
        print(
            f"Regenerating {len(tasks_to_run)} Figure 2 cache sets concurrently using {total_workers} total CPUs...",
            flush=True,
        )
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _exec_cache_task(name, func):
            if len(tasks_to_run) == 1:
                task_workers = total_workers
            elif name == "bottom_a":
                task_workers = max(2, total_workers * 2 // 3)
            else:
                remaining = max(1, total_workers - (total_workers * 2 // 3))
                task_workers = max(2, remaining // max(1, len(tasks_to_run) - 1))
            func(workers=task_workers)
            return name

        with ThreadPoolExecutor(max_workers=len(tasks_to_run)) as executor:
            futures = [
                executor.submit(_exec_cache_task, name, func)
                for name, func in tasks_to_run
            ]
            for future in as_completed(futures):
                future.result()
    else:
        for _, func in tasks_to_run:
            func(workers=total_workers)


# =============================================================================
# FIGURE 4: Cache Extraction
# =============================================================================
FIGURE4_LINEAR_PROBE_EXPERIMENT = "memo_mar9_zipf_onelayerattention_linearprobe"
FIGURE4_FORK_PROGRESS_EXPERIMENT = (
    "memo_apr3_zipf_onelayerattention_lr_1e3_swapmlp_eval_fork_progress"
)
FIGURE4_LINEAR_PROBE_RECORD_GLOB = (
    f"{RESULTS_DIR}/cache/"
    f"{FIGURE4_LINEAR_PROBE_EXPERIMENT}/*"
)
FIGURE4_FORK_PROGRESS_RECORD_GLOB = (
    f"{RESULTS_DIR}/cache/"
    f"{FIGURE4_FORK_PROGRESS_EXPERIMENT}/*"
)
FIGURE4_LINEAR_PROBE_CACHE = "./cache/df_fig4_linearprobe.csv"
FIGURE4_FORK_PROGRESS_CACHE = "./cache/df_fig4_fork_progress.csv"


def _figure4_record_path(folder):
    """Return the checkpoint path used by the original Figure 4 extractor."""
    subdir = os.path.basename(folder)
    path = os.path.join(folder, f"{subdir}.pkl")
    return path if os.path.isfile(path) else None


def _one_layer_parameter_count_from_args(args):
    """Count parameters without allocating a OneLayerAttention model."""
    hidden_width = args["num_heads"] * args["num_hidden_features"]
    mlp_width = args.get("num_hidden_features_mlp") or hidden_width
    num_mlp_layers = args["num_mlp_layers"]
    vocab_size = args["vocab_size"]
    context_length = args["len_context"]

    # Token/position embeddings, two layer norms, QKV projection, MLP, and
    # language-model head.  This is the same count as state_dict().values()
    # for gpt.OneLayerAttention, but avoids allocating its weight tensors.
    return (
        (vocab_size + context_length) * hidden_width
        + 4 * hidden_width
        + 3 * hidden_width ** 2 + 3 * hidden_width
        + hidden_width * mlp_width + mlp_width
        + (num_mlp_layers - 1) * (mlp_width ** 2 + mlp_width)
        + mlp_width * vocab_size + vocab_size
    )


def _extract_figure4_linear_probe_run(task):
    """Extract Figure 4 linear-probe rows from one checkpoint."""
    irun, folder = task
    record_path = _figure4_record_path(folder)
    if record_path is None:
        return [], None
    try:
        with open(record_path, "rb") as file:
            data = FastParamUnpickler(file).load()
    except Exception as error:
        return [], f"{record_path}: {error}"

    args = data.get("args", {})
    model_parameters = sum(
        value for value in data.get("model", {}).values()
        if isinstance(value, (int, np.integer))
    )
    if not model_parameters:
        return [], f"{record_path}: missing model parameters"

    rows = []
    for log in data.get("logs", []):
        metrics = log.get("test_metrics", {})
        probe_value = metrics.get("probe_novel_sequences_0")
        if not isinstance(probe_value, (float, np.floating)):
            continue
        try:
            rows.append({
                "num_heads": args["num_heads"],
                "num_hidden_features": args["num_hidden_features"],
                "num_parameters": model_parameters,
                "probe_novel_sequences_0": probe_value,
                "probe_novel_sequences_1": metrics["probe_novel_sequences_1"],
                "probe_novel_sequences_2": metrics["probe_novel_sequences_2"],
                "probe_novel_sequences_3": metrics["probe_novel_sequences_3"],
                "epoch": log["epoch"],
                "irun": irun,
                "iteration": log["epoch"] * args["num_iters_per_epoch"],
            })
        except (KeyError, TypeError) as error:
            return rows, f"{record_path}: incomplete linear-probe log ({error})"
    return rows, None


def _extract_figure4_fork_progress_run(task):
    """Extract Figure 4 fork-progress rows from one checkpoint."""
    irun, folder = task
    record_path = _figure4_record_path(folder)
    if record_path is None:
        return [], None
    try:
        with open(record_path, "rb") as file:
            data = FastParamUnpickler(file).load()
        args = data["args"]
        model_parameters = _one_layer_parameter_count_from_args(args)
    except Exception as error:
        return [], f"{record_path}: {error}"

    rows = []
    for log in data.get("logs", []):
        probe_metrics = log.get("probe_metrics", {})
        test_metrics = log.get("test_metrics", {})
        if "accuracy" not in probe_metrics or "mlp_probe_metrics" not in test_metrics:
            continue
        try:
            rows.append({
                "num_heads": args["num_heads"],
                "num_hidden_features": args["num_hidden_features"],
                "probe_accuracy": float(
                    np.stack(probe_metrics["accuracy"]).flatten()[-10:].mean()
                ),
                "num_parameters": model_parameters,
                "probe_novel_sequences_0": test_metrics["mlp_probe_metrics"]["accuracy"][0],
                "probe_novel_sequences_1": test_metrics["mlp_probe_metrics"]["accuracy"][1],
                "probe_novel_sequences_2": test_metrics["mlp_probe_metrics"]["accuracy"][2],
                "probe_novel_sequences_3": test_metrics["mlp_probe_metrics"]["accuracy"][3],
                "epoch": log["epoch"],
                "irun": irun,
            })
        except (KeyError, IndexError, TypeError, ValueError) as error:
            return rows, f"{record_path}: incomplete fork-progress log ({error})"
    return rows, None


def _regenerate_figure4_cache(record_glob, experiment, extractor, cache_path, workers=None):
    """Run a Figure 4 extractor and save its complete table."""
    folders = sorted(glob.glob(record_glob))
    if not folders:
        raise FileNotFoundError(f"No Figure 4 records match {record_glob}.")
    num_workers = min(get_default_workers(workers), len(folders))
    print(f"Regenerating Figure 4 cache from {experiment} ({len(folders)} runs) with {num_workers} workers")

    rows = []
    tasks = list(enumerate(folders))
    with Pool(processes=num_workers) as pool:
        for run_rows, error in pool.imap_unordered(extractor, tasks):
            if error is not None:
                print(f"Skipping unreadable record {error}")
            rows.extend(run_rows)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(f"No usable Figure 4 records were found in {record_glob}.")
    os.makedirs("./cache", exist_ok=True)
    frame.to_csv(cache_path, index=False)
    print(f"Wrote {len(frame)} rows to {cache_path}")


def regenerate_figure4_caches(workers=None):
    """Rebuild both data tables required by generate_figure4.py."""
    _regenerate_figure4_cache(
        FIGURE4_LINEAR_PROBE_RECORD_GLOB,
        FIGURE4_LINEAR_PROBE_EXPERIMENT,
        _extract_figure4_linear_probe_run,
        FIGURE4_LINEAR_PROBE_CACHE,
        workers=workers,
    )
    _regenerate_figure4_cache(
        FIGURE4_FORK_PROGRESS_RECORD_GLOB,
        FIGURE4_FORK_PROGRESS_EXPERIMENT,
        _extract_figure4_fork_progress_run,
        FIGURE4_FORK_PROGRESS_CACHE,
        workers=workers,
    )


def ensure_figure4_caches(force_regenerate=False, workers=None):
    """Create missing Figure 4 caches, or replace both when forced."""
    cache_paths = [FIGURE4_LINEAR_PROBE_CACHE, FIGURE4_FORK_PROGRESS_CACHE]
    if force_regenerate or any(not os.path.isfile(path) for path in cache_paths):
        regenerate_figure4_caches(workers=workers)


FIGURE4_POST_SWAP_CACHE = "./figs/Figure6/df_first_sequences_loader_post_swap.csv"
FIGURE4_FIRST_SEQUENCES_CACHE = "./figs/Figure6/df_first_sequences_loader.csv"
FIGURE4_SECOND_SEQUENCES_CACHE = "./figs/Figure6/df_second_sequences_loader.csv"
FIGURE4_SWAP_RECORD_GLOB = os.path.join(
    RESULTS_DIR, "cache", "memo_apr3_zipf_onelayerattention_lr_1e-3_swapmlp_eval", "*"
)


def ensure_figure4_heatmap_caches(force_regenerate=False):
    """Evaluate the final and swapped MLP against each set of saved sequences."""
    paths = [FIGURE4_FIRST_SEQUENCES_CACHE, FIGURE4_SECOND_SEQUENCES_CACHE,
             FIGURE4_POST_SWAP_CACHE]
    if not force_regenerate and all(os.path.isfile(path) for path in paths):
        return
    folders = sorted(glob.glob(FIGURE4_SWAP_RECORD_GLOB))
    rows = [[], [], []]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for folder in folders:
        name = os.path.basename(folder)
        try:
            with open(os.path.join(folder, name + ".pkl"), "rb") as stream:
                record = utils.CPU_Unpickler(stream).load()
            with open(os.path.join(folder, name + "_all_sequences.pkl"), "rb") as stream:
                sequence_sets = utils.CPU_Unpickler(stream).load()["sequences"]
            args = record["args"]
            model = gpt.OneLayerAttention(args["len_context"], args["num_heads"],
                                          args["num_hidden_features"], args["vocab_size"],
                                          args["num_mlp_layers"]).to(device)
            model.load_state_dict(record["model_state_dict"])
            parameters = sum(tensor.numel() for tensor in record["model_state_dict"].values())
            model.eval()
            with torch.no_grad():
                for group, sequences in enumerate((sequence_sets[0], sequence_sets[1], sequence_sets[0])):
                    if group == 2:
                        model.linear_modules.load_state_dict(record["mlp_state_dict"]["pre_switch"]["linear_modules"])
                        model.lm_head.load_state_dict(record["mlp_state_dict"]["pre_switch"]["lm_head"])
                    sequences = torch.as_tensor(sequences[::10])
                    for start in range(0, len(sequences), 100):
                        batch = sequences[start:start + 100].to(device)
                        output = model(batch, start // 100)
                        accuracy = (output[:, :-1].argmax(dim=-1) == batch[:, 1:]).float().mean(dim=-1)
                        rows[group].extend({"rank": (start + i) * 10, "num_parameters": parameters,
                                            "is_correct": float(value)}
                                           for i, value in enumerate(accuracy.cpu()))
        except (OSError, KeyError, IndexError, ValueError, RuntimeError, pickle.UnpicklingError) as error:
            print(f"Skipping Figure 4 heatmap record {folder}: {error}", flush=True)
    if any(not group for group in rows):
        raise FileNotFoundError(f"No complete Figure 4 swap-MLP records found in {FIGURE4_SWAP_RECORD_GLOB}")
    os.makedirs(os.path.dirname(paths[0]), exist_ok=True)
    for path, group in zip(paths, rows):
        pd.DataFrame(group).to_csv(path, index=False)
        print(f"Wrote {len(group)} rows to {path}", flush=True)


def _add_figure4_spines(ax, linewidth=1.2, color="black"):
    for side in ["left", "right", "top", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(linewidth)
        ax.spines[side].set_edgecolor(color)


def _match_figure4_marker_colors(ax):
    """Make every Seaborn line marker use its resolved line color."""
    for line in ax.get_lines():
        color = line.get_color()
        line.set_markeredgecolor(color)
        line.set_markerfacecolor(color)


def _set_linear_rescaled_ticks(
    ax,
    x_values,
    y_values,
    x_scale=1e4,
    y_scale=1e7,
    x_tick_step=1.5,
    y_tick_step=1.0,
    show_yticklabels=True,
):
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)

    ncols = len(x_values)
    nrows = len(y_values)

    x_min = x_values.min() / x_scale
    x_max = x_values.max() / x_scale
    y_min = y_values.min() / y_scale
    y_max = y_values.max() / y_scale

    x_ticks_lab = np.arange(
        0,
        np.floor(x_max / x_tick_step) * x_tick_step + 0.5 * x_tick_step,
        x_tick_step,
    )
    y_ticks_lab = np.arange(
        np.ceil(y_min / y_tick_step) * y_tick_step,
        y_max + 0.5 * y_tick_step,
        y_tick_step,
    )

    if len(x_ticks_lab) == 0 or not np.isclose(x_ticks_lab[0], 0):
        x_ticks_lab = np.insert(x_ticks_lab, 0, 0.0)
    if len(y_ticks_lab) == 0:
        y_ticks_lab = np.array([y_min, y_max])

    x_ticks_lab = x_ticks_lab[(x_ticks_lab >= x_min) & (x_ticks_lab <= x_max)]
    y_ticks_lab = y_ticks_lab[(y_ticks_lab >= y_min) & (y_ticks_lab <= y_max)]

    x_pos = np.interp(
        x_ticks_lab * x_scale,
        [x_values.min(), x_values.max()],
        [0.5, ncols - 0.5],
    )
    y_pos = np.interp(
        y_ticks_lab * y_scale,
        [y_values.min(), y_values.max()],
        [0.5, nrows - 0.5],
    )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        [f"{v:.1f}".rstrip("0").rstrip(".") for v in x_ticks_lab],
        rotation=90,
    )

    ax.set_yticks(y_pos)
    if show_yticklabels:
        ax.set_yticklabels(
            [f"{v:.1f}".rstrip("0").rstrip(".") for v in y_ticks_lab],
            rotation=0,
        )
    else:
        ax.set_yticklabels([])


def create_figure4(force_regenerate_cache=False):
    """Create publication-quality Figure 4 for NeurIPS camera-ready submission.

    Combines:
      Top Row: Attention accuracy heatmaps (Second sequences, First sequences, Post-swap)
      Bottom Row: Linear decodability vs iteration (Attention, MLP) and logit probe accuracy
    """
    ensure_figure4_caches(force_regenerate=force_regenerate_cache)
    ensure_figure4_heatmap_caches(force_regenerate=force_regenerate_cache)
    apply_figure_style()

    # Top row data
    df_post = pd.read_csv(FIGURE4_POST_SWAP_CACHE)
    df_first = pd.read_csv(FIGURE4_FIRST_SEQUENCES_CACHE)
    df_second = pd.read_csv(FIGURE4_SECOND_SEQUENCES_CACHE)

    # Bottom row data
    df_de = pd.read_csv(FIGURE4_LINEAR_PROBE_CACHE)
    final_it = df_de["iteration"].max()
    df_de["iteration"] = pd.to_numeric(df_de["iteration"], errors="coerce")
    df_de["probe_novel_sequences_0"] = pd.to_numeric(df_de["probe_novel_sequences_0"], errors="coerce")
    df_de["probe_novel_sequences_3"] = pd.to_numeric(df_de["probe_novel_sequences_3"], errors="coerce")
    df_de["num_parameters"] = pd.to_numeric(df_de["num_parameters"], errors="coerce")

    df_final = df_de[df_de["iteration"] == final_it]
    best_idx = df_final.groupby("num_parameters")["probe_novel_sequences_0"].idxmax()
    best_runs = df_final.loc[best_idx, ["num_parameters", "irun"]]

    df_best = (
        df_de.merge(best_runs, on=["num_parameters", "irun"], how="inner")
        .sort_values(["num_parameters", "iteration"])
        .reset_index(drop=True)
    )
    params_de = np.sort(df_best["num_parameters"].unique())

    df_f = pd.read_csv(FIGURE4_FORK_PROGRESS_CACHE)
    eps = np.finfo(float).eps
    p = df_f["probe_accuracy"].clip(eps, 1 - eps)
    df_f["logit_probe_accuracy"] = logit(p)

    fig = plt.figure(figsize=(15.5, 10.5))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.035], wspace=0.30, hspace=0.48)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    cbar_top_ax = fig.add_subplot(gs[0, 3])

    ax_d = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1])
    ax_f = fig.add_subplot(gs[1, 2])
    cbar_bottom_ax = fig.add_subplot(gs[1, 3])

    # ----------------- TOP ROW -----------------
    top_datasets = [
        (ax_a, df_second, "Second Sequences Accuracy", "a", True),
        (ax_b, df_first, "First Sequences Accuracy", "b", False),
        (ax_c, df_post, "First Sequences Post-Swap Accuracy", "c", False),
    ]

    for ax, raw_df, title, panel_lbl, is_first in top_datasets:
        df_sub = pd.DataFrame(
            raw_df[(raw_df["num_parameters"] > 2e6) & (raw_df["num_parameters"] < 6e7)]
        ).copy()
        df_sub["rank"] = pd.to_numeric(df_sub["rank"], errors="coerce")
        df_sub["num_parameters"] = pd.to_numeric(df_sub["num_parameters"], errors="coerce")
        df_sub["is_correct"] = pd.to_numeric(df_sub["is_correct"], errors="coerce")

        pivot_df = (
            df_sub.pivot_table(
                columns="rank",
                index="num_parameters",
                values="is_correct",
                aggfunc="mean",
            )
            .sort_index()
            .sort_index(axis=1)
        )

        sns.heatmap(
            pivot_df,
            cmap=FIGURE4_ACCURACY_CMAP,
            vmin=0.5,
            vmax=0.8,
            ax=ax,
            cbar=(panel_lbl == "c"),
            cbar_ax=cbar_top_ax if (panel_lbl == "c") else None,
            cbar_kws={"label": "Accuracy"} if (panel_lbl == "c") else None,
        )

        ax.set_title(title, fontsize=14, pad=10)
        ax.set_xlabel(r"Rank ($\times 10^4$)", fontsize=13)
        ax.set_ylabel(r"Number of Parameters ($\times 10^7$)" if is_first else "", fontsize=13)

        _set_linear_rescaled_ticks(
            ax,
            x_values=pivot_df.columns.to_numpy(),
            y_values=pivot_df.index.to_numpy(),
            x_scale=1e4,
            y_scale=1e7,
            x_tick_step=1.5,
            y_tick_step=1.0,
            show_yticklabels=is_first,
        )

        _add_figure4_spines(ax, linewidth=1.2)

    _add_figure4_spines(cbar_top_ax, linewidth=1.2)
    cbar_top_ax.set_ylabel("Accuracy", fontsize=13)

    # ----------------- BOTTOM ROW -----------------
    cmap = plt.get_cmap(NUM_PARAMETERS_CMAP)
    norm_de = mpl.colors.LogNorm(vmin=params_de.min(), vmax=params_de.max())
    param_to_color_de = {p: cmap(norm_de(p)) for p in params_de}
    norm_f = mpl.colors.LogNorm(vmin=df_f["num_parameters"].min(), vmax=df_f["num_parameters"].max())

    # Panel d
    sns.lineplot(
        data=df_best,
        x="iteration",
        y="probe_novel_sequences_0",
        hue="num_parameters",
        hue_order=params_de,
        palette=param_to_color_de,
        marker="+",
        markersize=4,
        linewidth=2.2,
        estimator=None,
        sort=False,
        ax=ax_d,
    )
    _match_figure4_marker_colors(ax_d)
    if ax_d.get_legend():
        ax_d.get_legend().remove()
    ax_d.set_xlabel(r"Iteration ($\times 10^4$)", fontsize=13)
    ax_d.set_ylabel("Linear Decodability", fontsize=13)
    ax_d.set_title("Attention Heads", fontsize=14, pad=10)
    ax_d.xaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(lambda value, _: f"{value / 1e4:g}")
    )
    ax_d.set_ylim(-0.05, 0.9)
    _add_figure4_spines(ax_d, linewidth=1.2)

    # Panel e
    sns.lineplot(
        data=df_best,
        x="iteration",
        y="probe_novel_sequences_3",
        hue="num_parameters",
        hue_order=params_de,
        palette=param_to_color_de,
        marker="+",
        markersize=4,
        linewidth=2.2,
        estimator=None,
        sort=False,
        ax=ax_e,
    )
    _match_figure4_marker_colors(ax_e)
    if ax_e.get_legend():
        ax_e.get_legend().remove()
    ax_e.set_xlabel(r"Iteration ($\times 10^4$)", fontsize=13)
    ax_e.set_ylabel("Linear Decodability", fontsize=13)
    ax_e.set_title("MLP", fontsize=14, pad=10)
    ax_e.xaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(lambda value, _: f"{value / 1e4:g}")
    )
    ax_e.set_ylim(-0.05, 0.9)
    _add_figure4_spines(ax_e, linewidth=1.2)

    # Panel f
    sns.scatterplot(
        data=df_f,
        x="probe_novel_sequences_0",
        y="logit_probe_accuracy",
        hue="num_parameters",
        palette=NUM_PARAMETERS_CMAP,
        hue_norm=norm_f,
        legend=False,
        marker="+",
        s=110,
        linewidth=2.2,
        ax=ax_f,
    )
    ax_f.set_xlabel("Linear Decodability", fontsize=13)
    ax_f.set_ylabel("Logit of Probe Accuracy", fontsize=13)
    ax_f.set_xlim(-0.05, 1.0)
    ax_f.set_ylim(-0.1, 2.55)
    _add_figure4_spines(ax_f, linewidth=1.2)

    # Bottom Colorbar
    sm_bottom = mpl.cm.ScalarMappable(norm=norm_f, cmap=NUM_PARAMETERS_CMAP)
    sm_bottom.set_array([])
    cbar_bottom = fig.colorbar(sm_bottom, cax=cbar_bottom_ax)
    cbar_bottom.set_label("Number of Parameters", fontsize=13)
    cbar_bottom.formatter = mpl.ticker.LogFormatterSciNotation()
    cbar_bottom.update_ticks()
    _add_figure4_spines(cbar_bottom_ax, linewidth=1.2)

    # Subpanel labels
    fig.canvas.draw()
    pos_a = ax_a.get_position()
    pos_b = ax_b.get_position()
    pos_c = ax_c.get_position()
    pos_d = ax_d.get_position()
    pos_e = ax_e.get_position()
    pos_f = ax_f.get_position()

    # Leave clear vertical separation from the enlarged top-row panel titles.
    y_top_label = pos_a.y1 + 0.070
    y_bot_label = pos_d.y1 + 0.048
    dx = -0.012

    fig.text(pos_a.x0 + dx, y_top_label, r"\textbf{a}", fontsize=22, fontweight="bold", va="bottom", ha="left")
    fig.text(pos_b.x0 + dx, y_top_label, r"\textbf{b}", fontsize=22, fontweight="bold", va="bottom", ha="left")
    fig.text(pos_c.x0 + dx, y_top_label, r"\textbf{c}", fontsize=22, fontweight="bold", va="bottom", ha="left")
    fig.text(pos_d.x0 + dx, y_bot_label, r"\textbf{d}", fontsize=22, fontweight="bold", va="bottom", ha="left")
    fig.text(pos_e.x0 + dx, y_bot_label, r"\textbf{e}", fontsize=22, fontweight="bold", va="bottom", ha="left")
    fig.text(pos_f.x0 + dx, y_bot_label, r"\textbf{f}", fontsize=22, fontweight="bold", va="bottom", ha="left")

    return fig


# =============================================================================
# FIGURE 4PRE: Saturation Appearances vs Sequence Rank
# =============================================================================
FIGURE4PRE_EXPERIMENT = (
    "memo_jan20_zipf_onelayerattention_vary_num_hidden_features_num_heads_doresample"
)
FIGURE4PRE_RECORD_GLOB = (
    f"{RESULTS_DIR}/cache/"
    f"{FIGURE4PRE_EXPERIMENT}/*"
)
FIGURE4PRE_CACHE = "./cache/df_fig4pre_saturation.csv"


def _extract_figure4pre_folder_saturation(folder):
    record_paths = [
        path for path in glob.glob(os.path.join(folder, "*.pkl"))
        if not path.endswith(("_all_sequences.pkl", "_model_final.pkl")) and "_model_iter_" not in os.path.basename(path)
    ]
    if len(record_paths) != 1:
        return []
    try:
        with open(record_paths[0], "rb") as file:
            data = FastParamUnpickler(file).load()
    except Exception:
        return []
    if "logs" not in data or len(data["logs"]) < 10:
        return []
    model_params = sum(
        value for value in data.get("model", {}).values()
        if isinstance(value, (int, np.integer))
    )
    if model_params > 1e6 and data["logs"][-1].get("train_loss", 0) > 0.3:
        return []

    rows = []
    THRESHOLD = 0.9
    for log in data["logs"]:
        if "num_apppearances" not in log or "test_metrics" not in log:
            continue
        appearances = log["num_apppearances"][::10]
        test_metrics = log["test_metrics"]
        sequence_ranks = np.asarray(test_metrics["sequence_rank"]) * 10
        accuracy = np.asarray(test_metrics["accuracy"])
        epoch = log["epoch"]
        switch_number = int(epoch // (np.round(len(data["logs"]) / 2.0)) + 1.0)

        mask = accuracy > THRESHOLD
        if not np.any(mask):
            continue
        for app, rank, acc in zip(
            np.asarray(appearances)[mask], sequence_ranks[mask], accuracy[mask]
        ):
            rows.append({
                "num_appearances": int(app),
                "sequence_rank": int(rank),
                "accuracy": float(acc),
                "num_parameters": model_params,
                "epoch": int(epoch),
                "switch_number": int(switch_number),
            })
    return rows


def regenerate_figure4pre_cache(workers=None):
    """Extract and rebuild the Figure 4pre saturation cache."""
    folders = sorted(glob.glob(FIGURE4PRE_RECORD_GLOB))
    if not folders:
        raise FileNotFoundError(
            f"No Figure 4pre records match {FIGURE4PRE_RECORD_GLOB}."
        )
    num_workers = min(get_default_workers(workers), len(folders))
    print(
        f"Regenerating Figure 4pre saturation cache from {FIGURE4PRE_EXPERIMENT} "
        f"({len(folders)} runs) with {num_workers} workers..."
    )
    with Pool(processes=num_workers) as pool:
        all_results = pool.map(_extract_figure4pre_folder_saturation, folders)

    all_rows = [row for run_rows in all_results if run_rows for row in run_rows]
    if not all_rows:
        raise RuntimeError("No usable Figure 4pre saturation records found.")
    df = pd.DataFrame(all_rows)
    results = (
        df.sort_values("epoch", ascending=True)
        .drop_duplicates(subset=["num_parameters", "switch_number", "sequence_rank"])
        .sort_values(["num_parameters", "switch_number", "sequence_rank"])
        .reset_index(drop=True)
    )
    results["sequence_rank_plus_one"] = results["sequence_rank"] + 1
    os.makedirs("./cache", exist_ok=True)
    results.to_csv(FIGURE4PRE_CACHE, index=False)
    print(f"Wrote {len(results)} rows to {FIGURE4PRE_CACHE}")


def ensure_figure4pre_cache(force_regenerate=False, workers=None):
    if force_regenerate or not os.path.isfile(FIGURE4PRE_CACHE):
        regenerate_figure4pre_cache(workers=workers)


def _bin_figure4pre_log_x(df_subset, num_bins=80):
    """Bin sequence rank in log space and summarize each bin.

    The central curve uses the geometric mean of positive appearances; the
    lower envelope uses the minimum observed appearance count in each bin.
    """
    y = df_subset["num_appearances"].values
    x = df_subset["sequence_rank_plus_one"].values
    if len(y) == 0:
        return pd.DataFrame({
            "sequence_rank_plus_one": [],
            "num_appearances": [],
            "num_appearances_min": [],
        })

    valid = (x > 0) & (y > 0)
    x = x[valid]
    y = y[valid]
    if len(y) == 0:
        return pd.DataFrame({
            "sequence_rank_plus_one": [],
            "num_appearances": [],
            "num_appearances_min": [],
        })

    bins = np.logspace(np.log10(x.min()), np.log10(x.max()), num_bins + 1)
    bin_idx = np.digitize(x, bins) - 1
    bin_idx = np.clip(bin_idx, 0, num_bins - 1)

    binned_x = []
    binned_y = []
    binned_min = []
    for b in range(num_bins):
        mask = (bin_idx == b)
        if np.any(mask):
            gm_x = np.exp(np.mean(np.log(x[mask])))
            gm_y = np.exp(np.mean(np.log(y[mask])))
            binned_x.append(gm_x)
            binned_y.append(gm_y)
            binned_min.append(np.min(y[mask]))

    return pd.DataFrame({
        "sequence_rank_plus_one": binned_x,
        "num_appearances": binned_y,
        "num_appearances_min": binned_min,
    }).sort_values("sequence_rank_plus_one")


def _plot_figure4pre_saturation_panel(ax, results, num_params, num_bins=80):
    """Plot one former Figure 4pre saturation panel on ``ax``."""
    for switch_number in [1, 2]:
        sub = results[
            (results["num_parameters"] == num_params)
            & (results["switch_number"] == switch_number)
        ].copy()
        sub["sequence_rank_plus_one"] = pd.to_numeric(sub["sequence_rank_plus_one"])
        binned = _bin_figure4pre_log_x(sub, num_bins=num_bins)
        label = "Pre-switch" if switch_number == 1 else "Post-switch"
        line, = ax.plot(
            binned["sequence_rank_plus_one"],
            binned["num_appearances"],
            linewidth=2.0,
            alpha=0.85,
            label=label,
            marker="+",
            markeredgecolor="black",
        )
        # The translucent lower envelope reaches the smallest observed
        # saturation-appearance count in each log-rank bin.
        ax.fill_between(
            binned["sequence_rank_plus_one"],
            binned["num_appearances_min"],
            binned["num_appearances"],
            color=line.get_color(),
            alpha=0.35,
            linewidth=0,
        )

    ax.set_title(f"{num_params:,} Parameters", pad=10)
    ax.set_xscale("log")
    ax.set_xlim(0.5, 100000)
    ax.set_yscale("log")
    ax.set_ylim(0.5, 100000)
    ax.set_xlabel("Sequence rank")
    ax.set_ylabel("Saturation appearances")
    ax.tick_params(which="both", direction="in")
    ax.legend(loc="upper right")


def create_figure4pre(force_regenerate_cache=False, num_bins=80):
    """Create publication-quality Figure 4pre: saturation appearances vs rank."""
    ensure_figure4pre_cache(force_regenerate=force_regenerate_cache)
    results = pd.read_csv(FIGURE4PRE_CACHE)

    apply_figure_style()
    params_to_analyze = [25399298, 56973314]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, num_params in zip(axes, params_to_analyze):
        _plot_figure4pre_saturation_panel(ax, results, num_params, num_bins=num_bins)

    fig.canvas.draw()
    pos_a = axes[0].get_position()
    pos_b = axes[1].get_position()

    t_a = axes[0].title.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
    t_b = axes[1].title.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())

    dx = -0.025
    y_label = max(t_a.y1, t_b.y1) + 0.035

    fig.text(pos_a.x0 + dx, y_label, r"\textbf{a}", fontsize=22, fontweight="bold", va="bottom", ha="left")
    fig.text(pos_b.x0 + dx, y_label, r"\textbf{b}", fontsize=22, fontweight="bold", va="bottom", ha="left")

    return fig


def force_regenerate_all_figure_caches(parallel=False, workers=None):
    """Rebuild every cache used by the camera-ready figures."""
    print("Figure 1: no data cache to regenerate.")
    ensure_figure2_caches(force_regenerate=True, parallel=parallel, workers=workers)
    # Figure 3 includes the former Figure 4pre saturation panels.
    ensure_figure4pre_cache(force_regenerate=True, workers=workers)
    ensure_figure4_caches(force_regenerate=True, workers=workers)
    print("Figure 5: no data cache to regenerate.")
    print("Figure 6: reads DAM records directly; no data cache to regenerate.")
    regenerate_figure_s4_caches()
    regenerate_figure_s5_cache(workers=workers)
    regenerate_figure_s6_cache()
    regenerate_figure_s7_cache(workers=workers)
    regenerate_figure_s8_cache(workers=workers)
    regenerate_figure_s2_cache(workers=workers)
    print("Figure S3: reads GPT model-size sweep records directly; no data cache to regenerate.")


# =============================================================================
# FIGURE 2: Full Assembly (2 rows x 3 columns)
# =============================================================================
def create_figure2(force_regenerate_cache=False, cache_parallel=False, workers=None):
    """Create publication-quality Figure 2 for NeurIPS submission.
    Combines:
      Column (a): Final train loss vs parameter count (12-layer top, 1-layer bottom)
      Column (b): Maximum rank memorized vs parameter count with power law fits
                  (12-layer top at 60k, 120k, 190k; 1-layer bottom at 20k, 50k, 79k)
      Column (c): Average accuracy over first 3000 ranked sequences across data switch
                  with broken axis [0, 1.0] and [4.0, 5.0] x 10^4 (12-layer top, 1-layer bottom)
    """
    ensure_figure2_caches(force_regenerate=force_regenerate_cache, parallel=cache_parallel, workers=workers)
    apply_figure_style()
    print(
        "Figure 2 top-row panels (a, b) experiments: "
        f"{', '.join(FIGURE2_TOP_AB_EXPERIMENTS)}"
    )
    print(
        "Figure 2 top-row panel (c) experiments: "
        f"{', '.join(FIGURE2_TOP_C_EXPERIMENTS)}"
    )
    print(
        "Figure 2 bottom-row panel (a) experiments: "
        f"{', '.join(FIGURE2_BOTTOM_A_EXPERIMENTS)}"
    )
    print(
        "Figure 2 bottom-row panel (b) experiment: "
        f"{FIGURE2_BOTTOM_B_EXPERIMENT}"
    )
    print(
        "Figure 2 bottom-row panel (c) experiment: "
        f"{FIGURE2_BOTTOM_C_EXPERIMENT}"
    )

    fig = plt.figure(figsize=(19, 11))

    # Standardize panel sizes across all columns so panels a, b, and c have identical dimensions.
    w_plot = 1.0
    w_cbar = 0.045
    w_pad = 0.025
    w_col_ac = w_plot + w_pad + w_cbar
    w_col_b = w_plot

    outer_gs = fig.add_gridspec(
        2, 3,
        width_ratios=[w_col_ac, w_col_b, w_col_ac],
        height_ratios=[1.0, 1.0],
        wspace=0.28,
        hspace=0.45,
        left=0.05,
        right=0.96,
        top=0.91,
        bottom=0.07,
    )

    cbar_wspace = w_pad / ((w_plot + w_cbar) / 2)

    # -------------------------------------------------------------------------
    # COLUMN 1: Panel (a) - Train Loss vs. Parameters
    # -------------------------------------------------------------------------
    gs_a0 = outer_gs[0, 0].subgridspec(1, 2, width_ratios=[w_plot, w_cbar], wspace=cbar_wspace)
    ax_a0 = fig.add_subplot(gs_a0[0])
    cax_a0 = fig.add_subplot(gs_a0[1])
    cax_a0.set_label("<colorbar>")

    gs_a1 = outer_gs[1, 0].subgridspec(1, 2, width_ratios=[w_plot, w_cbar], wspace=cbar_wspace)
    ax_a1 = fig.add_subplot(gs_a1[0])
    cax_a1 = fig.add_subplot(gs_a1[1])
    cax_a1.set_label("<colorbar>")

    # 12-layer train loss data (freshly extracted from memo_apr19)
    df_tl_12 = pd.read_csv(FIGURE2_TOP_A_CACHE)
    ok_12 = df_tl_12[~df_tl_12["failed"]]
    plot_train_loss_panel(
        ax_a0,
        ok_12,
        title="Train Loss vs. Number of Parameters",
        d_min=min(ok_12["num_hidden_features"]),
        d_max=max(ok_12["num_hidden_features"]),
        cbar_ax=cax_a0,
    )

    # 1-layer train loss data from FIGURE2_BOTTOM_A_EXPERIMENTS.
    df_tl_1 = pd.read_csv(FIGURE2_BOTTOM_A_CACHE)
    df_tl_1 = df_tl_1[df_tl_1["epoch"] == 50]
    df_tl_1 = filter_figure2_bottom_a_mlp_widths(df_tl_1)
    plot_train_loss_panel(
        ax_a1,
        df_tl_1,
        title="Train Loss vs. Number of Parameters",
        d_min=min(df_tl_1["num_hidden_features"]),
        d_max=max(df_tl_1["num_hidden_features"]),
        cbar_ax=cax_a1,
    )

    # -------------------------------------------------------------------------
    # COLUMN 2: Panel (b) - Max Rank Memorized vs. Parameters
    # -------------------------------------------------------------------------
    ax_b0 = fig.add_subplot(outer_gs[0, 1])
    ax_b1 = fig.add_subplot(outer_gs[1, 1])

    # 12-layer capacity data (freshly extracted from memo_apr19, excluding memo_may1)
    df_cap_12 = pd.read_csv(FIGURE2_TOP_B_CACHE)
    df_cap_12_ok = df_cap_12[~df_cap_12["failed"]] if "failed" in df_cap_12.columns else df_cap_12
    plot_power_law_panel(
        ax_b0,
        df_cap_12_ok,
        iters_list=[60000, 120000, 190000],
        title="Max Learned Rank vs. Number of Parameters",
        is_12layer=True,
    )
    ax_b0.set_ylim(top=2e5)

    # 1-layer capacity data from FIGURE2_BOTTOM_B_EXPERIMENT.
    df_cap_1 = pd.read_csv(FIGURE2_BOTTOM_B_CACHE)
    df_cap_1 = filter_figure2_bottom_row_models(df_cap_1)
    plot_power_law_panel(
        ax_b1,
        df_cap_1,
        iters_list=[20000, 50000, 79000],
        title="Max Learned Rank vs. Number of Parameters",
        is_12layer=False,
    )
    ax_b1.set_ylim(top=2e5)

    # -------------------------------------------------------------------------
    # COLUMN 3: Panel (c) - Meta-Memorization Broken Axis Lineplots
    # -------------------------------------------------------------------------
    gs_c0 = outer_gs[0, 2].subgridspec(1, 2, width_ratios=[w_plot, w_cbar], wspace=cbar_wspace)
    gs_c0_broken = gs_c0[0].subgridspec(1, 2, width_ratios=[1, 1], wspace=0.08)
    ax_c0_left = fig.add_subplot(gs_c0_broken[0])
    ax_c0_right = fig.add_subplot(gs_c0_broken[1], sharey=ax_c0_left)
    cax_c0 = fig.add_subplot(gs_c0[1])
    cax_c0.set_label("<colorbar>")

    gs_c1 = outer_gs[1, 2].subgridspec(1, 2, width_ratios=[w_plot, w_cbar], wspace=cbar_wspace)
    gs_c1_broken = gs_c1[0].subgridspec(1, 2, width_ratios=[1, 1], wspace=0.08)
    ax_c1_left = fig.add_subplot(gs_c1_broken[0])
    ax_c1_right = fig.add_subplot(gs_c1_broken[1], sharey=ax_c1_left)
    cax_c1 = fig.add_subplot(gs_c1[1])
    cax_c1.set_label("<colorbar>")

    # 12-layer data switch data from FIGURE2_TOP_C_EXPERIMENTS (memo_may1_zipf_gpt2_vary_num_hidden_features_num_heads_doresample)
    plot_df = pd.read_csv(FIGURE2_TOP_C_CACHE)
    plot_df = plot_df[plot_df["num_parameters"] < FIGURE2_TOP_C_MAX_PARAMETERS].copy()
    avg_df_12 = plot_df

    # 1-layer data-switch panel from FIGURE2_BOTTOM_C_EXPERIMENT.
    bottom_c_df = pd.read_csv(FIGURE2_BOTTOM_C_CACHE)
    bottom_c_df = filter_figure2_bottom_row_models(bottom_c_df)
    avg_df_1 = prepare_figure2_panel_c_data(bottom_c_df)

    norm_c0 = LogNorm(
        vmin=avg_df_12["num_parameters"].min(),
        vmax=avg_df_12["num_parameters"].max(),
    )
    norm_c1 = LogNorm(
        vmin=avg_df_1["num_parameters"].min(),
        vmax=avg_df_1["num_parameters"].max(),
    )

    plot_broken_axis_panel(ax_c0_left, ax_c0_right, avg_df_12, norm=norm_c0,
                           title="Average Accuracy Over First 3000 Ranked Sequences")
    plot_broken_axis_panel(ax_c1_left, ax_c1_right, avg_df_1, norm=norm_c1,
                           title="Average Accuracy Over First 3000 Ranked Sequences")

    # Colorbars for Column (c)
    sm_c0 = mpl.cm.ScalarMappable(cmap=NUM_PARAMETERS_CMAP, norm=norm_c0)
    sm_c0.set_array([])
    cbar_c0 = fig.colorbar(sm_c0, cax=cax_c0)
    cbar_c0.set_label("Number of Parameters", fontsize=14)
    cbar_c0.ax.tick_params(labelsize=11)
    cbar_c0.formatter = mticker.LogFormatterSciNotation()
    cbar_c0.update_ticks()

    sm_c1 = mpl.cm.ScalarMappable(cmap=NUM_PARAMETERS_CMAP, norm=norm_c1)
    sm_c1.set_array([])
    cbar_c1 = fig.colorbar(sm_c1, cax=cax_c1)
    cbar_c1.set_label("Number of Parameters", fontsize=14)
    cbar_c1.ax.tick_params(labelsize=11)
    cbar_c1.formatter = mticker.LogFormatterSciNotation()
    cbar_c1.update_ticks()

    for ax in [ax_a0, ax_a1, ax_b0, ax_b1, ax_c0_left, ax_c0_right, ax_c1_left, ax_c1_right]:
        ax.tick_params(which="both", top=False, right=False)

    # -------------------------------------------------------------------------
    # Row Headers: 12-layer models (top row) & 1-layer models (bottom row)
    # -------------------------------------------------------------------------
    fig.text(0.5, 0.985, r"\textbf{12-layer models}", fontsize=20, ha="center", va="top")
    fig.text(0.5, 0.505, r"\textbf{1-layer models}", fontsize=20, ha="center", va="top")

    # -------------------------------------------------------------------------
    # Subpanel Labels: a, b, c
    # -------------------------------------------------------------------------
    pos_a = ax_a0.get_position()
    pos_b = ax_b0.get_position()
    pos_c = ax_c0_left.get_position()

    fig.text(pos_a.x0 - 0.015, 0.98, r"\textbf{a}", fontsize=22, fontweight="bold", va="top")
    fig.text(pos_b.x0 - 0.015, 0.98, r"\textbf{b}", fontsize=22, fontweight="bold", va="top")
    fig.text(pos_c.x0 - 0.015, 0.98, r"\textbf{c}", fontsize=22, fontweight="bold", va="top")

    return fig


create_figure_2 = create_figure2

# =============================================================================
# FIGURE 3: Rescaled Memorization Rates vs Sequence Rank
# =============================================================================
FIGURE3_12LAYER_CSV = os.path.join(
    "./cache", "saturation_results_gpt2.csv"
)
FIGURE3_1LAYER_CSV = os.path.join(
    "./cache", "saturation_results_onelayer.csv"
)


def _ensure_saturation_source(path, record_globs, architecture, force=False):
    """Extract first observed >90% crossings from newly generated records."""
    if os.path.isfile(path) and not force:
        return
    rows = []
    folders = sorted({folder for pattern in record_globs for folder in glob.glob(pattern)})
    for folder in folders:
        record_path = os.path.join(folder, os.path.basename(folder) + ".pkl")
        try:
            with open(record_path, "rb") as stream:
                record = FastParamUnpickler(stream).load()
            logs, args = record["logs"], record["args"]
            if len(logs) < (200 if architecture == "12-layer" else 10):
                continue
            if architecture == "12-layer":
                width = int(args["num_heads"]) * int(args["num_hidden_features"])
                count = (int(args["num_layers"]) * (12 * width ** 2 + 13 * width)
                         + (int(args["len_context"]) + 2 * int(args.get("vocab_size", 2)) + 2) * width
                         + int(args.get("vocab_size", 2)))
                loss = float(logs[-1]["train_loss"])
                if (count > 3e6 and loss > 0.21) or (count > 1e6 and loss > 0.4):
                    continue
            else:
                count = _one_layer_parameter_count_from_args(args)
                if count > 7e7 or (count > 1e6 and float(logs[-1]["train_loss"]) > 0.3):
                    continue
            seen = set()
            for log in sorted(logs, key=lambda item: item["epoch"]):
                metrics = log.get("test_metrics", {})
                if "accuracy" not in metrics or "sequence_rank" not in metrics or "num_apppearances" not in log:
                    continue
                appearances = np.asarray(log["num_apppearances"])[::10]
                ranks = np.asarray(metrics["sequence_rank"])
                if architecture == "12-layer":
                    ranks = ranks * 10
                accuracy = np.asarray(metrics["accuracy"])
                if not (len(appearances) == len(ranks) == len(accuracy)):
                    raise ValueError("rank, appearance, and accuracy lengths differ")
                for rank, appearance, value in zip(ranks, appearances, accuracy):
                    rank = int(rank)
                    if value > 0.9 and rank not in seen:
                        seen.add(rank)
                        rows.append({"num_parameters": count, "sequence_rank_plus_one": rank + 1,
                                     "num_appearances_at_saturation": int(appearance),
                                     "epoch_at_saturation": int(log["epoch"])})
        except (OSError, KeyError, ValueError, TypeError, EOFError, pickle.UnpicklingError) as error:
            print(f"Skipping saturation record {record_path}: {error}", flush=True)
    if not rows:
        raise FileNotFoundError(f"No usable {architecture} saturation records in {record_globs}.")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Wrote {len(rows)} saturation events to {path}", flush=True)


def ensure_saturation_sources(force=False):
    _ensure_saturation_source(FIGURE3_12LAYER_CSV, FIGURE2_TOP_AB_RECORD_GLOBS, "12-layer", force)
    _ensure_saturation_source(FIGURE3_1LAYER_CSV, [FIGURE4PRE_RECORD_GLOB], "1-layer", force)


def _plot_rescaled_saturation_panel(
    ax, csv_path, slice_idx, alpha_exp, alpha_label, title, cbar_ax=None
):
    """Plot binned rescaled saturation appearances vs sequence rank for specified model slice and power law."""
    df = pd.read_csv(csv_path)
    model_sizes = np.sort(np.array(df["num_parameters"].unique()))

    log_sizes = np.log10(model_sizes)
    norm = mpl.colors.Normalize(vmin=log_sizes.min(), vmax=log_sizes.max())
    cmap = sns.color_palette(NUM_PARAMETERS_CMAP, as_cmap=True)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    results = []
    last_bin_centers = None

    for i, m in enumerate(model_sizes[slice_idx:]):
        indices = np.array((df["num_parameters"] == m))
        if indices.sum() < 5:
            continue
        ranks = df["sequence_rank_plus_one"][indices].values
        presentations = df["num_appearances_at_saturation"][indices].values

        # bin on log scale
        log_bins = np.logspace(np.log10(ranks.min()), np.log10(ranks.max()), 20)
        bin_idx = np.digitize(ranks, log_bins)

        bin_centers = []
        bin_mean_presentations = []
        for j in range(1, len(log_bins)):
            mask = bin_idx == j
            if mask.sum() > 0:
                bin_centers.append(np.sqrt(log_bins[j - 1] * log_bins[j]))
                bin_mean_presentations.append(np.mean(presentations[mask]))

        bin_centers = np.array(bin_centers)
        bin_mean_presentations = np.array(bin_mean_presentations)

        if len(bin_centers) < 3:
            continue

        # fit power law: presentations = a * rank^b via polyfit in log-log space
        b, log_a = np.polyfit(np.log(bin_centers), np.log(bin_mean_presentations), 1)
        a = np.exp(log_a)

        results.append({"num_parameters": m, "a": a, "b": b})

        color = cmap(norm(np.log10(m)))
        ax.loglog(bin_centers, bin_mean_presentations / a, "o-", color=color, ms=4)
        last_bin_centers = bin_centers

    if last_bin_centers is not None:
        ax.loglog(last_bin_centers, last_bin_centers**(alpha_exp), "k--", lw=3, label=alpha_label)

    ax.set_xlabel("Sequence rank (binned)", fontsize=14)
    ax.set_ylabel("Rescaled number of appearances", fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.set_xlim(1, 1e5)
    ax.tick_params(axis="both", which="both", labelsize=14)
    if cbar_ax is None:
        cbar = ax.figure.colorbar(sm, ax=ax)
    else:
        cbar = ax.figure.colorbar(sm, cax=cbar_ax)
    cbar.set_label(r"$\log_{10}(\text{Num Parameters})$")
    ax.legend(fontsize=12)

    return sm, last_bin_centers, pd.DataFrame(results)


def create_figure3(
    csv_12=FIGURE3_12LAYER_CSV,
    csv_1=FIGURE3_1LAYER_CSV,
    force_regenerate_cache=False,
):
    """Create publication-quality Figure 3 for NeurIPS camera-ready submission.

    Combines:
      (a) 12-layer GPT-2 rescaled saturation appearances (alpha=0.7)
      (b) 1-layer rescaled saturation appearances (alpha=0.85)
      (c-d) pre/post-switch saturation appearances for two model sizes
    """
    apply_figure_style()
    ensure_saturation_sources(force=force_regenerate_cache)
    ensure_figure4pre_cache(force_regenerate=force_regenerate_cache)
    figure4pre_results = pd.read_csv(FIGURE4PRE_CACHE)

    # The original Figure 3 panels were each approximately 4.5 inches tall.
    # Retain that panel scale in the two-row composite so its typography remains
    # comparable with the rest of the camera-ready figures.
    fig = plt.figure(figsize=(12, 9.5))
    outer_grid = fig.add_gridspec(
        2, 2,
        left=0.075, right=0.95, bottom=0.075, top=0.90,
        wspace=0.36, hspace=0.30,
    )

    # Reserve a narrow, identical colorbar slot beside every panel.  The lower
    # slots remain empty, so c/d have the same plotting-rectangle dimensions
    # as a/b without adding superfluous colorbars.
    panel_grids = [
        outer_grid[row, col].subgridspec(
            1, 2, width_ratios=[1.0, 0.055], wspace=0.08
        )
        for row in range(2)
        for col in range(2)
    ]
    axes = np.array([
        fig.add_subplot(panel_grid[0]) for panel_grid in panel_grids
    ]).reshape(2, 2)
    colorbar_axes = [
        fig.add_subplot(panel_grids[index][1]) for index in range(2)
    ]

    _plot_rescaled_saturation_panel(
        axes[0, 0],
        csv_12,
        slice_idx=5,
        alpha_exp=-0.7,
        alpha_label=r"$\alpha = 0.7$",
        title="12-layer models",
        cbar_ax=colorbar_axes[0],
    )
    _plot_rescaled_saturation_panel(
        axes[0, 1],
        csv_1,
        slice_idx=1,
        alpha_exp=-0.85,
        alpha_label=r"$\alpha = 0.85$",
        title="1-Layer models",
        cbar_ax=colorbar_axes[1],
    )
    for axis, num_params in zip(axes[1], [25399298, 56973314]):
        _plot_figure4pre_saturation_panel(axis, figure4pre_results, num_params)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for label, axis in zip("abcd", axes.ravel()):
        title_bbox = axis.title.get_window_extent(renderer).transformed(
            fig.transFigure.inverted()
        )
        position = axis.get_position()
        fig.text(
            position.x0 - 0.035,
            title_bbox.y1 + 0.018,
            rf"\textbf{{{label}}}",
            fontsize=22,
            fontweight="bold",
            va="bottom",
            ha="left",
        )
    return fig


def generate_figure3(savefigdir="./figs_camera_ready", force_regenerate_cache=False):
    utils.ensure_dir(savefigdir)
    print("Generating Figure 3...", flush=True)
    fig3 = create_figure3(force_regenerate_cache=force_regenerate_cache)
    standardize_figure_typography(fig3)
    for name in ["Figure3_combined", "Figure3", "saturation_rates_gpt2", "saturation_rates_onelayer"]:
        fig3.savefig(f"{savefigdir}/{name}.pdf", format="pdf", bbox_inches="tight")
        fig3.savefig(f"{savefigdir}/{name}.eps", format="eps", bbox_inches="tight")
        fig3.savefig(f"{savefigdir}/{name}.png", format="png", dpi=300, bbox_inches="tight")
    print(f"Saved Figure 3 to {savefigdir}", flush=True)
    plt.close(fig3)


# =============================================================================
# SUPPLEMENTAL FIGURE 4: Saturation trajectories across architectures
# =============================================================================
# These established source tables already hold the first >90%-accuracy event
# per sequence.  S4 writes normalized copies to ./cache so it does not depend
# on the historical cache_oct3 directory at render time.
FIGURE_S4_12LAYER_SOURCE_CACHE = FIGURE3_12LAYER_CSV
FIGURE_S4_1LAYER_SOURCE_CACHE = FIGURE3_1LAYER_CSV
FIGURE_S4_12LAYER_CACHE = "./cache/df_figure_s4_12layer_saturation.csv"
FIGURE_S4_1LAYER_CACHE = "./cache/df_figure_s4_1layer_saturation.csv"
FIGURE_S4_REQUIRED_COLUMNS = {
    "num_appearances_at_saturation", "num_parameters", "epoch_at_saturation",
    "sequence_rank_plus_one",
}


def _prepare_figure_s4_saturation(source_cache, architecture):
    """Normalize a legacy saturation table for one S4 architecture column."""
    frame = pd.read_csv(source_cache)
    if not FIGURE_S4_REQUIRED_COLUMNS.issubset(frame.columns):
        missing = sorted(FIGURE_S4_REQUIRED_COLUMNS.difference(frame.columns))
        raise ValueError(
            f"Figure S4 source cache {source_cache!r} is missing columns {missing}."
        )
    if "failed" in frame.columns:
        frame = frame.loc[~frame["failed"].astype(bool)].copy()

    numeric_columns = list(FIGURE_S4_REQUIRED_COLUMNS)
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=numeric_columns)
    frame = frame.loc[
        (frame["num_parameters"] > 0)
        & (frame["sequence_rank_plus_one"] > 0)
        & (frame["num_appearances_at_saturation"] > 0)
        & (frame["epoch_at_saturation"] > 0)
    ].copy()
    if architecture == "1-layer":
        # Match the supplied one-layer code: retain the earliest threshold
        # crossing over the run rather than treating the cache's switch labels
        # as separate panels, and apply its explicit model-size cutoff.
        frame = frame.loc[frame["num_parameters"] <= 7e7].copy()

    frame = (
        frame.sort_values("epoch_at_saturation")
        .drop_duplicates(["num_parameters", "sequence_rank_plus_one"])
        .sort_values(["num_parameters", "sequence_rank_plus_one"])
        .reset_index(drop=True)
    )
    frame["saturation_iteration"] = frame["epoch_at_saturation"] * 1000
    frame["architecture"] = architecture
    return frame[
        [
            "num_appearances_at_saturation", "num_parameters",
            "epoch_at_saturation", "saturation_iteration",
            "sequence_rank_plus_one", "architecture",
        ]
    ]


def regenerate_figure_s4_caches():
    """Normalize the existing saturation-event tables for Supplemental Figure S4."""
    ensure_saturation_sources()
    source_specs = [
        (FIGURE_S4_12LAYER_SOURCE_CACHE, "12-layer", FIGURE_S4_12LAYER_CACHE),
        (FIGURE_S4_1LAYER_SOURCE_CACHE, "1-layer", FIGURE_S4_1LAYER_CACHE),
    ]
    utils.ensure_dir("./cache")
    for source_cache, architecture, target_cache in source_specs:
        if not os.path.isfile(source_cache):
            raise FileNotFoundError(
                f"Figure S4 cannot find its {architecture} source cache: {source_cache!r}."
            )
        frame = _prepare_figure_s4_saturation(source_cache, architecture)
        if frame.empty:
            raise RuntimeError(
                f"Figure S4 has no valid {architecture} saturation events."
            )
        frame.to_csv(target_cache, index=False)
        print(
            f"Figure S4: wrote {len(frame)} {architecture} events to {target_cache}.",
            flush=True,
        )


def ensure_figure_s4_caches(force_regenerate=False):
    """Create the normalized S4 caches when absent or explicitly refreshed."""
    cache_paths = [FIGURE_S4_12LAYER_CACHE, FIGURE_S4_1LAYER_CACHE]
    if force_regenerate or any(not os.path.isfile(path) for path in cache_paths):
        regenerate_figure_s4_caches()


def _figure_s4_log_norm(frame):
    """Return a stable log parameter normalization for one architecture column."""
    values = frame["num_parameters"].to_numpy(dtype=float)
    vmin, vmax = values.min(), values.max()
    if vmin == vmax:
        vmin /= 10.0
        vmax *= 10.0
    return LogNorm(vmin=vmin, vmax=vmax)


def _plot_figure_s4_iteration_panel(ax, frame, norm, x_limits, y_limits):
    """Draw saturation iteration versus rank, with one viridis trace per model."""
    sns.lineplot(
        data=frame.sort_values(["num_parameters", "sequence_rank_plus_one"]),
        x="sequence_rank_plus_one",
        y="saturation_iteration",
        hue="num_parameters",
        palette=NUM_PARAMETERS_CMAP,
        hue_norm=norm,
        linewidth=1.4,
        errorbar=None,
        legend=False,
        ax=ax,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.set_xlabel("Sequence Rank")
    ax.set_ylabel("Saturation Iteration")


def _plot_figure_s4_appearances_panel(ax, frame, norm, x_limits, y_limits):
    """Draw saturation appearances versus rank using viridis '+' markers."""
    ax.scatter(
        frame["sequence_rank_plus_one"],
        frame["num_appearances_at_saturation"],
        c=frame["num_parameters"],
        cmap=NUM_PARAMETERS_CMAP,
        norm=norm,
        marker="+",
        s=28,
        linewidths=1.05,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.set_xlabel("Sequence Rank")
    ax.set_ylabel("Saturation Appearances")


def _add_figure_s4_colorbar(fig, ax, norm):
    """Add a parameter colorbar with the same normalization as its panel."""
    scalar_mappable = mpl.cm.ScalarMappable(cmap=NUM_PARAMETERS_CMAP, norm=norm)
    scalar_mappable.set_array([])
    colorbar = fig.colorbar(scalar_mappable, ax=ax, pad=0.02, fraction=0.048)
    colorbar.set_label("Parameters")
    colorbar.ax.tick_params(labelsize=10)
    colorbar.formatter = mticker.LogFormatterMathtext(base=10)
    colorbar.update_ticks()


def create_figure_s4(force_regenerate_cache=False):
    """Create Supplemental Figure S4 with 12-layer and 1-layer saturation panels."""
    ensure_figure_s4_caches(force_regenerate=force_regenerate_cache)
    apply_figure_style()
    frame_12 = pd.read_csv(FIGURE_S4_12LAYER_CACHE)
    frame_1 = pd.read_csv(FIGURE_S4_1LAYER_CACHE)
    norm_12 = _figure_s4_log_norm(frame_12)
    norm_1 = _figure_s4_log_norm(frame_1)

    fig, axes = plt.subplots(
        2, 2, figsize=(12.0, 10.2), facecolor="white",
        gridspec_kw={"left": 0.09, "right": 0.96, "bottom": 0.09, "top": 0.88,
                     "wspace": 0.34, "hspace": 0.32},
    )
    ax_top_12, ax_top_1 = axes[0]
    ax_bottom_12, ax_bottom_1 = axes[1]
    _plot_figure_s4_iteration_panel(
        ax_top_12, frame_12, norm_12, x_limits=(1, 1e5), y_limits=(900, 2e5)
    )
    _plot_figure_s4_iteration_panel(
        ax_top_1, frame_1, norm_1, x_limits=(1, 1.1e5), y_limits=(900, 1.1e5)
    )
    _plot_figure_s4_appearances_panel(
        ax_bottom_12, frame_12, norm_12, x_limits=(1, 1e5), y_limits=(10, 2e5)
    )
    _plot_figure_s4_appearances_panel(
        ax_bottom_1, frame_1, norm_1, x_limits=(1, 1.1e5), y_limits=(1, 2e5)
    )
    for ax, norm in [
        (ax_top_12, norm_12), (ax_top_1, norm_1),
        (ax_bottom_12, norm_12), (ax_bottom_1, norm_1),
    ]:
        ax.tick_params(which="both", direction="in")
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
        _add_figure_s4_colorbar(fig, ax, norm)

    fig.canvas.draw()
    for label, top_axis, bottom_axis in [
        ("12-layer models", ax_top_12, ax_bottom_12),
        ("1-layer models", ax_top_1, ax_bottom_1),
    ]:
        top_position = top_axis.get_position()
        bottom_position = bottom_axis.get_position()
        fig.text(
            (top_position.x0 + top_position.x1) / 2,
            max(top_position.y1, bottom_position.y1) + 0.035,
            label,
            fontsize=18,
            fontweight="normal",
            ha="center",
            va="bottom",
        )
    for label, axis in zip("ab", [ax_top_12, ax_bottom_12]):
        position = axis.get_position()
        fig.text(
            position.x0 - 0.060,
            position.y1 + 0.002,
            rf"\textbf{{{label}}}",
            fontsize=22,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
    return fig


# =============================================================================
# SUPPLEMENTAL FIGURE 5: 12-layer meta-memorization after a data switch
# =============================================================================
FIGURE_S5_RECORD_GLOB = FIGURE2_TOP_C_RECORD_GLOB
FIGURE_S5_CACHE = "./cache/df_figure_s5_dataswitch_12layer_ranked.csv"
FIGURE_S5_MAX_PARAMETERS = 2e6
FIGURE_S5_MAX_RANK = 3000
META_MEMORIZATION_PANEL_WSPACE = 0.12
FIGURE_S5_CACHE_COLUMNS = {
    "num_parameters", "epoch", "iteration", "sequence_rank", "accuracy",
}


def _figure_s5_parameter_count_from_folder(folder):
    """Compute a 12-layer GPT parameter count from a May 1 run directory name."""
    match = re.search(r"_hidden_(\d+)_nheads_(\d+)_nlayers_(\d+)_", os.path.basename(folder))
    if match is None:
        return None
    hidden_features, num_heads, num_layers = map(int, match.groups())
    width = hidden_features * num_heads
    # Matches the GPT parameter accounting used by the Figure 2 top-row cache.
    return (
        num_layers * (12 * width ** 2 + 13 * width)
        + (100 + 2 * 2 + 2) * width
        + 2
    )


def _extract_figure_s5_run(folder):
    """Extract rank-resolved May 1 data-switch accuracy from one checkpoint."""
    estimated_parameters = _figure_s5_parameter_count_from_folder(folder)
    if (
        estimated_parameters is not None
        and estimated_parameters >= FIGURE_S5_MAX_PARAMETERS
    ):
        return [], None

    record_paths = [
        path for path in glob.glob(os.path.join(folder, "*.pkl"))
        if not path.endswith(("_all_sequences.pkl", "_model_final.pkl")) and "_model_iter_" not in os.path.basename(path)
    ]
    if len(record_paths) != 1:
        return [], None
    record_path = record_paths[0]
    try:
        with open(record_path, "rb") as file:
            # FastParamUnpickler keeps the rank/accuracy NumPy arrays in logs
            # while replacing model tensors with lightweight shape metadata.
            data = FastParamUnpickler(file).load()
        logs = data.get("logs", [])
        if len(logs) < 10:
            return [], None
        args = data["args"]
        width = int(args["num_heads"]) * int(args["num_hidden_features"])
        num_layers = int(args.get("num_layers", 12))
        context_length = int(args.get("len_context", 100))
        vocab_size = int(args.get("vocab_size", 2))
        num_parameters = (
            num_layers * (12 * width ** 2 + 13 * width)
            + (context_length + 2 * vocab_size + 2) * width
            + vocab_size
        )
        if num_parameters >= FIGURE_S5_MAX_PARAMETERS:
            return [], None

        rows = []
        for log in logs:
            epoch = log.get("epoch")
            test_metrics = log.get("test_metrics", {})
            accuracy = test_metrics.get("accuracy")
            sequence_rank = test_metrics.get("sequence_rank")
            if epoch is None or accuracy is None or sequence_rank is None:
                continue
            accuracy = np.asarray(accuracy).reshape(-1)
            sequence_rank = np.asarray(sequence_rank).reshape(-1) * 10
            valid = (
                (sequence_rank >= 0)
                & (sequence_rank <= FIGURE_S5_MAX_RANK)
                & np.isfinite(accuracy)
            )
            rows.extend({
                "num_parameters": num_parameters,
                "epoch": int(epoch),
                "iteration": int(epoch) * 1000,
                "sequence_rank": int(rank),
                "accuracy": float(value),
            } for rank, value in zip(sequence_rank[valid], accuracy[valid]) )
        return rows, None
    except (KeyError, TypeError, ValueError, OSError, EOFError,
            pickle.UnpicklingError, RuntimeError) as error:
        return [], f"{record_path}: {error}"
    finally:
        if "data" in locals():
            del data
        gc.collect()


def regenerate_figure_s5_cache(workers=None):
    """Build S5's rank-resolved cache from the May 1 data-switch checkpoints."""
    all_folders = sorted(glob.glob(FIGURE_S5_RECORD_GLOB))
    eligible_folders = [
        folder for folder in all_folders
        if (
            _figure_s5_parameter_count_from_folder(folder) is None
            or _figure_s5_parameter_count_from_folder(folder) < FIGURE_S5_MAX_PARAMETERS
        )
    ]
    if not eligible_folders:
        raise FileNotFoundError(
            "Figure S5 found no May 1 checkpoints below its "
            f"{FIGURE_S5_MAX_PARAMETERS:g}-parameter limit."
        )
    # Recycled workers bound memory while processing checkpoint log arrays.
    requested_workers = get_default_workers(workers, max_cap=2)
    num_workers = min(requested_workers, len(eligible_folders))
    print(
        f"Regenerating Figure S5 cache from {len(eligible_folders)} of "
        f"{len(all_folders)} May 1 runs with {num_workers} workers...",
        flush=True,
    )
    rows = []
    skipped = []
    with Pool(processes=num_workers, maxtasksperchild=1) as pool:
        for run_rows, error in pool.imap_unordered(
            _extract_figure_s5_run, eligible_folders
        ):
            rows.extend(run_rows)
            if error is not None:
                skipped.append(error)
                print(f"Figure S5: skipping {error}", flush=True)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("Figure S5 found no usable rank-resolved data-switch logs.")
    frame = frame.sort_values(
        ["num_parameters", "iteration", "sequence_rank"]
    ).reset_index(drop=True)
    utils.ensure_dir("./cache")
    frame.to_csv(FIGURE_S5_CACHE, index=False)
    print(
        f"Figure S5: wrote {len(frame)} rows from "
        f"{frame['num_parameters'].nunique()} models to {FIGURE_S5_CACHE}"
        + (f"; skipped {len(skipped)} unreadable runs." if skipped else "."),
        flush=True,
    )


def ensure_figure_s5_cache(force_regenerate=False, workers=None):
    """Create S5's rank-resolved cache if absent, stale, or refreshed."""
    cache_is_usable = False
    if os.path.isfile(FIGURE_S5_CACHE):
        try:
            cache_is_usable = FIGURE_S5_CACHE_COLUMNS.issubset(
                pd.read_csv(FIGURE_S5_CACHE, nrows=0).columns
            )
        except (OSError, pd.errors.EmptyDataError):
            cache_is_usable = False
    if force_regenerate or not cache_is_usable:
        regenerate_figure_s5_cache(workers=workers)


def _figure_s5_plot_data(frame, rank_lower, rank_upper):
    """Average sequence accuracy by model and iteration in one rank interval."""
    subframe = frame.loc[
        (frame["sequence_rank"] >= rank_lower)
        & (frame["sequence_rank"] < rank_upper)
    ]
    return (
        subframe.groupby(["num_parameters", "iteration"], as_index=False)["accuracy"]
        .mean()
        .sort_values(["num_parameters", "iteration"])
    )


def create_figure_s5(force_regenerate_cache=False):
    """Create Supplemental Figure S5: 12-layer accuracy by ranked-sequence bin."""
    ensure_figure_s5_cache(force_regenerate=force_regenerate_cache)
    apply_figure_style()
    frame = pd.read_csv(FIGURE_S5_CACHE)
    for column in FIGURE_S5_CACHE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=FIGURE_S5_CACHE_COLUMNS)
    frame = frame.loc[
        (frame["num_parameters"] < FIGURE_S5_MAX_PARAMETERS)
        & (frame["sequence_rank"] >= 0)
        & (frame["sequence_rank"] <= FIGURE_S5_MAX_RANK)
    ].copy()
    if frame.empty:
        raise RuntimeError("Figure S5 cache has no data after its model/rank filters.")
    norm = LogNorm(
        vmin=frame["num_parameters"].min(),
        vmax=frame["num_parameters"].max(),
    )

    fig, axes = plt.subplots(
        1, 3, figsize=(15.5, 4.7), facecolor="white",
        gridspec_kw={"left": 0.075, "right": 0.89, "bottom": 0.17, "top": 0.80,
                     "wspace": META_MEMORIZATION_PANEL_WSPACE},
    )
    rank_edges = [(0, 1000), (1000, 2000), (2000, 3000)]
    for index, (axis, (rank_lower, rank_upper)) in enumerate(zip(axes, rank_edges)):
        panel_frame = _figure_s5_plot_data(frame, rank_lower, rank_upper)
        if panel_frame.empty:
            raise RuntimeError(
                f"Figure S5 has no observations for ranks [{rank_lower}, {rank_upper})."
            )
        sns.lineplot(
            data=panel_frame,
            x="iteration",
            y="accuracy",
            hue="num_parameters",
            palette=NUM_PARAMETERS_CMAP,
            hue_norm=norm,
            errorbar=None,
            legend=False,
            linewidth=1.6,
            ax=axis,
        )
        axis.axvline(40000, color="red", linestyle="--", linewidth=1.25)
        axis.text(
            42000,
            0.94,
            "Data Switch",
            color="red",
            fontsize=PANEL_TITLE_FONT_SIZE * MAX_TYPOGRAPHY_SCALE,
        )
        axis.set_title(f"Ranks Between {rank_lower} and {rank_upper}", pad=5)
        axis.set_xlabel(r"Iteration ($\times 10^4$)")
        axis.set_ylim(0.45, 1.0)
        axis.set_xticks([0, 20000, 40000, 60000, 80000])
        axis.set_xticklabels(["0", "2", "4", "6", "8"])
        axis.tick_params(which="both", direction="in")
        for spine in axis.spines.values():
            spine.set_linewidth(1.0)
        if index == 0:
            axis.set_ylabel("Accuracy")
        else:
            axis.set_ylabel("")
            axis.tick_params(labelleft=False)

    scalar_mappable = mpl.cm.ScalarMappable(cmap=NUM_PARAMETERS_CMAP, norm=norm)
    scalar_mappable.set_array([])
    colorbar = fig.colorbar(scalar_mappable, ax=axes, pad=0.018, fraction=0.035)
    colorbar.set_label("Number of Parameters")
    colorbar.ax.tick_params(labelsize=10)
    colorbar.formatter = mticker.LogFormatterMathtext(base=10)
    colorbar.update_ticks()
    return fig


# =============================================================================
# SUPPLEMENTAL FIGURE 6: 1-layer meta-memorization after a data switch
# =============================================================================
FIGURE_S6_SOURCE_CACHE = FIGURE2_BOTTOM_C_CACHE
FIGURE_S6_CACHE = "./cache/df_figure_s6_dataswitch_1layer_ranked.csv"
FIGURE_S6_MAX_PARAMETERS = FIGURE2_BOTTOM_MAX_PARAMETERS
FIGURE_S6_MAX_RANK = 3000
FIGURE_S6_SOURCE_COLUMNS = {
    "num_parameters", "epoch", "sequence_rank", "accuracy",
}
FIGURE_S6_CACHE_COLUMNS = FIGURE_S6_SOURCE_COLUMNS | {"iteration"}


def regenerate_figure_s6_cache(chunksize=1_000_000):
    """Derive S6's compact rank-resolved cache from the Figure 2c source cache.

    The source cache already applies the Figure 2c checkpoint-level filters:
    successful Jan 20 one-layer runs, no more than 70M parameters, and the
    pre-switch/post-switch epoch windows.  Here we retain ranks 0--3000 for
    the three S6 rank bands.
    """
    if not os.path.isfile(FIGURE_S6_SOURCE_CACHE):
        raise FileNotFoundError(
            "Figure S6 requires the Figure 2c source cache at "
            f"{FIGURE_S6_SOURCE_CACHE!r}. Regenerate it with --cache fig2_bottom_c."
        )
    try:
        source_columns = set(pd.read_csv(FIGURE_S6_SOURCE_CACHE, nrows=0).columns)
    except (OSError, pd.errors.EmptyDataError) as error:
        raise RuntimeError(
            f"Figure S6 could not read {FIGURE_S6_SOURCE_CACHE!r}: {error}"
        ) from error
    if not FIGURE_S6_SOURCE_COLUMNS.issubset(source_columns):
        missing = sorted(FIGURE_S6_SOURCE_COLUMNS.difference(source_columns))
        raise ValueError(
            f"Figure S6 source cache is missing required columns {missing}."
        )

    utils.ensure_dir("./cache")
    temporary_cache = f"{FIGURE_S6_CACHE}.tmp"
    if os.path.exists(temporary_cache):
        os.remove(temporary_cache)
    rows_written = 0
    first_chunk = True
    for chunk in pd.read_csv(FIGURE_S6_SOURCE_CACHE, chunksize=chunksize):
        for column in FIGURE_S6_SOURCE_COLUMNS:
            chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
        filtered = chunk.dropna(subset=FIGURE_S6_SOURCE_COLUMNS)
        filtered = filtered.loc[
            (filtered["num_parameters"] <= FIGURE_S6_MAX_PARAMETERS)
            & (filtered["sequence_rank"] >= 0)
            & (filtered["sequence_rank"] <= FIGURE_S6_MAX_RANK)
        ].copy()
        if filtered.empty:
            continue
        filtered["iteration"] = filtered["epoch"] * 1000
        filtered.to_csv(
            temporary_cache,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
        )
        first_chunk = False
        rows_written += len(filtered)
    if rows_written == 0:
        if os.path.exists(temporary_cache):
            os.remove(temporary_cache)
        raise RuntimeError("Figure S6 found no Figure 2c records in ranks 0--3000.")
    os.replace(temporary_cache, FIGURE_S6_CACHE)
    print(f"Figure S6: wrote {rows_written} rows to {FIGURE_S6_CACHE}.", flush=True)


def ensure_figure_s6_cache(force_regenerate=False):
    """Create S6's compact rank-resolved cache when missing or stale."""
    cache_is_usable = False
    if os.path.isfile(FIGURE_S6_CACHE):
        try:
            cache_is_usable = FIGURE_S6_CACHE_COLUMNS.issubset(
                pd.read_csv(FIGURE_S6_CACHE, nrows=0).columns
            )
        except (OSError, pd.errors.EmptyDataError):
            cache_is_usable = False
    if force_regenerate or not cache_is_usable:
        regenerate_figure_s6_cache()


def _figure_s6_plot_data(frame, rank_lower, rank_upper):
    """Average rank-resolved accuracy for one S6 sequence-rank interval."""
    panel_frame = frame.loc[
        (frame["sequence_rank"] >= rank_lower)
        & (frame["sequence_rank"] < rank_upper)
    ]
    return (
        panel_frame.groupby(["num_parameters", "iteration"], as_index=False)["accuracy"]
        .mean()
        .sort_values(["num_parameters", "iteration"])
    )


def _plot_figure_s6_broken_panel(ax_left, ax_right, frame, norm, title,
                                  show_y_label):
    """Plot one rank bin on pre-/post-switch iteration windows with an axis break."""
    left_frame = frame.loc[(frame["iteration"] >= 0) & (frame["iteration"] <= 10000)]
    right_frame = frame.loc[
        (frame["iteration"] >= 40000) & (frame["iteration"] <= 50000)
    ]
    line_kwargs = dict(
        y="accuracy",
        hue="num_parameters",
        palette=NUM_PARAMETERS_CMAP,
        hue_norm=norm,
        errorbar=None,
        legend=False,
        linewidth=1.5,
    )
    sns.lineplot(
        data=left_frame, x="iteration", ax=ax_left, **line_kwargs
    )
    sns.lineplot(
        data=right_frame, x="iteration", ax=ax_right, **line_kwargs
    )
    ax_right.axvline(40000, color="red", linestyle="--", linewidth=1.25)
    ax_right.text(
        40230,
        0.94,
        "Data Switch",
        color="red",
        fontsize=PANEL_TITLE_FONT_SIZE * MAX_TYPOGRAPHY_SCALE,
    )

    ax_left.set_xlim(0, 10000)
    ax_right.set_xlim(40000, 50000)
    ax_left.set_ylim(0.45, 1.0)
    ax_left.set_xticks([0, 5000, 10000])
    ax_right.set_xticks([40000, 45000, 50000])
    ax_left.set_xticklabels(["0.0", "0.5", "1.0"])
    ax_right.set_xticklabels(["4.0", "4.5", "5.0"])
    ax_left.get_xticklabels()[-1].set_horizontalalignment("right")
    ax_right.get_xticklabels()[0].set_horizontalalignment("left")
    ax_left.set_title(title, x=1.04, pad=5)
    ax_left.set_ylabel("Accuracy" if show_y_label else "")
    ax_left.set_xlabel("")
    ax_right.set_xlabel("")
    ax_right.set_ylabel("")
    if not show_y_label:
        ax_left.tick_params(labelleft=False)
    ax_right.tick_params(labelleft=False, left=False)

    ax_left.spines["right"].set_visible(False)
    ax_right.spines["left"].set_visible(False)
    for axis in [ax_left, ax_right]:
        axis.tick_params(which="both", direction="in")
        for spine in axis.spines.values():
            if spine.get_visible():
                spine.set_linewidth(1.0)
    d = 0.015
    slash_kwargs = dict(
        transform=ax_left.transAxes, color="black", clip_on=False, linewidth=1.0
    )
    ax_left.plot((1 - d, 1 + d), (-d, d), **slash_kwargs)
    ax_left.plot((1 - d, 1 + d), (1 - d, 1 + d), **slash_kwargs)
    slash_kwargs["transform"] = ax_right.transAxes
    ax_right.plot((-d, d), (-d, d), **slash_kwargs)
    ax_right.plot((-d, d), (1 - d, 1 + d), **slash_kwargs)
    ax_left.text(
        1.04, -0.14, r"Iteration ($\times 10^4$)",
        transform=ax_left.transAxes,
        ha="center",
        va="top",
        fontsize=AXIS_LABEL_FONT_SIZE * _figure5_relative_font_scale(
            ax_left, ax_left.figure
        ),
    )


def create_figure_s6(force_regenerate_cache=False):
    """Create Supplemental Figure S6: broken-axis one-layer data-switch panels."""
    ensure_figure_s6_cache(force_regenerate=force_regenerate_cache)
    apply_figure_style()
    frame = pd.read_csv(FIGURE_S6_CACHE)
    for column in FIGURE_S6_CACHE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=FIGURE_S6_CACHE_COLUMNS)
    if frame.empty:
        raise RuntimeError("Figure S6 cache has no usable rank-resolved records.")
    norm = LogNorm(
        vmin=frame["num_parameters"].min(),
        vmax=frame["num_parameters"].max(),
    )
    fig = plt.figure(figsize=(16.0, 4.8), facecolor="white")
    outer_grid = fig.add_gridspec(
        1, 3, left=0.065, right=0.90, bottom=0.22, top=0.79,
        wspace=META_MEMORIZATION_PANEL_WSPACE,
    )
    axes = []
    for index, (rank_lower, rank_upper) in enumerate([(0, 1000), (1000, 2000), (2000, 3000)]):
        panel_grid = outer_grid[index].subgridspec(1, 2, wspace=0.08)
        ax_left = fig.add_subplot(panel_grid[0])
        ax_right = fig.add_subplot(panel_grid[1], sharey=ax_left)
        panel_frame = _figure_s6_plot_data(frame, rank_lower, rank_upper)
        if panel_frame.empty:
            raise RuntimeError(
                f"Figure S6 has no observations for ranks [{rank_lower}, {rank_upper})."
            )
        _plot_figure_s6_broken_panel(
            ax_left,
            ax_right,
            panel_frame,
            norm,
            f"Ranks Between {rank_lower} and {rank_upper}",
            show_y_label=index == 0,
        )
        axes.extend([ax_left, ax_right])

    scalar_mappable = mpl.cm.ScalarMappable(cmap=NUM_PARAMETERS_CMAP, norm=norm)
    scalar_mappable.set_array([])
    colorbar = fig.colorbar(scalar_mappable, ax=axes, pad=0.025, fraction=0.025)
    colorbar.set_label("Number of Parameters")
    colorbar.ax.tick_params(labelsize=10)
    colorbar.formatter = mticker.LogFormatterMathtext(base=10)
    colorbar.update_ticks()
    # Treat each broken left/right pair as one logical panel for typography.
    # Otherwise, its text scales from only half the visual panel width and is
    # smaller than S5's corresponding full-width panels.
    fig.canvas.draw()
    for ax_left, ax_right in zip(axes[::2], axes[1::2]):
        left_position = ax_left.get_position()
        right_position = ax_right.get_position()
        logical_width = (right_position.x1 - left_position.x0) * fig.get_figwidth()
        logical_height = left_position.height * fig.get_figheight()
        logical_scale = float(np.clip(
            np.sqrt(logical_width * logical_height) / FIGURE5_REFERENCE_PANEL_INCHES,
            1.0,
            MAX_TYPOGRAPHY_SCALE,
        ))
        for axis in (ax_left, ax_right):
            axis._typography_scale_override = logical_scale

        # These labels are manually positioned because each rank panel has a
        # broken axis, so standardize_figure_typography does not manage them.
        for text in ax_left.texts:
            if text.get_text() == r"Iteration ($\times 10^4$)":
                text.set_fontsize(AXIS_LABEL_FONT_SIZE * logical_scale)
        for text in ax_right.texts:
            if text.get_text() == "Data Switch":
                text.set_fontsize(PANEL_TITLE_FONT_SIZE * logical_scale)
    return fig


# =============================================================================
# SUPPLEMENTAL FIGURE 7: Fine-tuning accuracy on novel sequences
# =============================================================================
# Figure S7 uses the Apr. 3 fork-progress sweep already extracted for Figure
# 4. S7 writes a compact copy in ./cache with only the early fine-tuning
# checkpoints and its plotted accuracy field.
FIGURE_S7_SOURCE_CACHE = FIGURE4_FORK_PROGRESS_CACHE
FIGURE_S7_CACHE = "./cache/df_figure_s7_finetuning_accuracy.csv"
FIGURE_S7_MAX_ITERATION = 5000
FIGURE_S7_SOURCE_COLUMNS = {
    "num_heads", "num_hidden_features", "num_parameters",
    "probe_accuracy", "epoch", "irun",
}
FIGURE_S7_CACHE_COLUMNS = {
    "num_heads", "num_hidden_features", "num_parameters",
    "fine_tuning_accuracy", "epoch", "iteration", "irun",
}


def regenerate_figure_s7_cache(workers=None):
    """Derive the compact S7 table from the Apr. 3 fork-progress cache."""
    source_is_usable = False
    if os.path.isfile(FIGURE_S7_SOURCE_CACHE):
        try:
            source_is_usable = FIGURE_S7_SOURCE_COLUMNS.issubset(
                pd.read_csv(FIGURE_S7_SOURCE_CACHE, nrows=0).columns
            )
        except (OSError, pd.errors.EmptyDataError):
            source_is_usable = False
    if not source_is_usable:
        regenerate_figure4_caches(workers=workers)

    frame = pd.read_csv(FIGURE_S7_SOURCE_CACHE)
    missing = FIGURE_S7_SOURCE_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(
            f"Figure S7 source cache is missing required columns {sorted(missing)}."
        )
    frame = frame.loc[:, sorted(FIGURE_S7_SOURCE_COLUMNS)].copy()
    for column in FIGURE_S7_SOURCE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=FIGURE_S7_SOURCE_COLUMNS)
    frame["iteration"] = frame["epoch"] * 1000
    frame = frame.loc[
        (frame["iteration"] >= 0)
        & (frame["iteration"] <= FIGURE_S7_MAX_ITERATION)
    ].rename(columns={"probe_accuracy": "fine_tuning_accuracy"})
    if frame.empty:
        raise RuntimeError("Figure S7 found no initial fine-tuning probe records.")
    # Preserve the source row order: this matches the supplied seaborn recipe
    # (estimator=None, sort=False) when several runs share a model size.
    frame = frame.reset_index(drop=True)
    utils.ensure_dir("./cache")
    frame.to_csv(FIGURE_S7_CACHE, index=False)
    print(
        f"Figure S7: wrote {len(frame)} rows from {frame['irun'].nunique()} Apr. 3 runs "
        f"to {FIGURE_S7_CACHE}.",
        flush=True,
    )


def ensure_figure_s7_cache(force_regenerate=False, workers=None):
    """Create S7's compact cache when it is missing or stale."""
    cache_is_usable = False
    if os.path.isfile(FIGURE_S7_CACHE):
        try:
            cache_is_usable = FIGURE_S7_CACHE_COLUMNS.issubset(
                pd.read_csv(FIGURE_S7_CACHE, nrows=0).columns
            )
        except (OSError, pd.errors.EmptyDataError):
            cache_is_usable = False
    if force_regenerate or not cache_is_usable:
        regenerate_figure_s7_cache(workers=workers)


def create_figure_s7(force_regenerate_cache=False):
    """Create Supplemental Figure S7: fine-tuning accuracy by model size."""
    ensure_figure_s7_cache(force_regenerate=force_regenerate_cache)
    apply_figure_style()
    frame = pd.read_csv(FIGURE_S7_CACHE)
    for column in FIGURE_S7_CACHE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=FIGURE_S7_CACHE_COLUMNS)
    frame = frame.loc[
        (frame["iteration"] >= 0)
        & (frame["iteration"] <= FIGURE_S7_MAX_ITERATION)
    ].copy()
    if frame.empty:
        raise RuntimeError("Figure S7 cache has no data in its initial fine-tuning window.")
    parameters = np.sort(frame["num_parameters"].unique())
    norm = LogNorm(vmin=parameters.min(), vmax=parameters.max())

    fig, axis = plt.subplots(figsize=(7.0, 6.0), facecolor="white")
    sns.lineplot(
        data=frame,
        x="iteration",
        y="fine_tuning_accuracy",
        hue="num_parameters",
        hue_order=parameters,
        palette=NUM_PARAMETERS_CMAP,
        hue_norm=norm,
        estimator=None,
        marker="+",
        markersize=4,
        linewidth=1.5,
        legend=False,
        sort=False,
        ax=axis,
    )
    _match_figure4_marker_colors(axis)
    # Keep the scientific multiplier adjacent to the label rather than as a
    # detached Matplotlib offset at the far edge of the axis.
    axis.set_xlabel(r"Iteration ($\times 10^3$)")
    axis.set_ylabel("Fine-Tuning Accuracy")
    axis.set_xlim(0, FIGURE_S7_MAX_ITERATION)
    axis.set_ylim(0.45, 1.0)
    axis.set_xticks(np.arange(0, FIGURE_S7_MAX_ITERATION + 1, 1000))
    axis.set_xticklabels([str(value) for value in range(6)])
    axis.tick_params(which="both", direction="in")
    for spine in axis.spines.values():
        spine.set_linewidth(1.0)

    scalar_mappable = mpl.cm.ScalarMappable(cmap=NUM_PARAMETERS_CMAP, norm=norm)
    scalar_mappable.set_array([])
    colorbar = fig.colorbar(scalar_mappable, ax=axis, pad=0.02)
    colorbar.set_label("Number of Parameters")
    colorbar.formatter = mticker.LogFormatterMathtext(base=10)
    colorbar.update_ticks()
    return fig


# =============================================================================
# SUPPLEMENTAL FIGURE 8: Novel-sequence probing with zero attention weights
# =============================================================================
FIGURE_S8_EXPERIMENT = (
    "memo_jan4_fork_progress_set_zero_all_attn_allow_learning_K_100000"
)
FIGURE_S8_RECORD_GLOB = (
    f"{RESULTS_DIR}/cache/{FIGURE_S8_EXPERIMENT}/*"
)
FIGURE_S8_CACHE = "./cache/df_figure_s8_zero_attention_probe.csv"
FIGURE_S8_MIN_PARAMETERS = 1e4
FIGURE_S8_MAX_PARAMETERS = 3e7
FIGURE_S8_TARGET_ITERATION = 1000
FIGURE_S8_CACHE_COLUMNS = {
    "num_heads", "num_hidden_features", "num_parameters", "probe_accuracy",
    "train_loss", "num_iters_required", "epoch", "iteration", "irun",
}


def _extract_figure_s8_run(task):
    """Extract zero-attention probe records from a single Jan. 4 checkpoint."""
    irun, folder = task
    record_path = _figure4_record_path(folder)
    if record_path is None:
        return [], None
    try:
        with open(record_path, "rb") as file:
            data = FastParamUnpickler(file).load()
        args = data["args"]
        if args.get("is_initialize_attention_weights_to_zero") != "zero_all_attn_allow_learning":
            return [], (
                f"{record_path}: expected zero_all_attn_allow_learning checkpoint"
            )
        num_parameters = _one_layer_parameter_count_from_args(args)
        rows = []
        for log in data.get("logs", []):
            probe_metrics = log.get("probe_metrics", {})
            accuracies = probe_metrics.get("accuracy")
            epoch = log.get("epoch")
            if accuracies is None or epoch is None:
                continue
            accuracy_values = np.asarray(accuracies).reshape(-1)
            if len(accuracy_values) < 10:
                continue
            probe_accuracy = float(np.mean(accuracy_values[-10:]))
            if not np.isfinite(probe_accuracy):
                continue
            rows.append({
                "num_heads": int(args["num_heads"]),
                "num_hidden_features": int(args["num_hidden_features"]),
                "num_parameters": int(num_parameters),
                "probe_accuracy": probe_accuracy,
                "train_loss": float(log["train_loss"]),
                "num_iters_required": float(probe_metrics["num_iters_required"]),
                "epoch": int(epoch),
                # Preserve the iteration convention in the supplied analysis.
                "iteration": (int(epoch) + 1) * 50,
                "irun": int(irun),
            })
        return rows, None
    except (KeyError, TypeError, ValueError, IndexError, OSError, EOFError,
            pickle.UnpicklingError, RuntimeError) as error:
        return [], f"{record_path}: {error}"
    finally:
        if "data" in locals():
            del data
        gc.collect()


def regenerate_figure_s8_cache(workers=None):
    """Build S8's zero-attention probe cache from the Jan. 4 checkpoints."""
    folders = sorted(glob.glob(FIGURE_S8_RECORD_GLOB))
    if not folders:
        raise FileNotFoundError(
            f"Figure S8 found no experiment directories matching {FIGURE_S8_RECORD_GLOB!r}."
        )
    # The checkpoints retain many probe arrays. A single recycled worker keeps
    # memory bounded while making a full cache rebuild safe on login nodes.
    num_workers = min(get_default_workers(workers, max_cap=1), len(folders))
    print(
        f"Regenerating Figure S8 cache from {len(folders)} Jan. 4 runs "
        f"with {num_workers} worker...",
        flush=True,
    )
    rows = []
    skipped = []
    with Pool(processes=num_workers, maxtasksperchild=1) as pool:
        for run_rows, error in pool.imap_unordered(
            _extract_figure_s8_run, enumerate(folders)
        ):
            rows.extend(run_rows)
            if error is not None:
                skipped.append(error)
                print(f"Figure S8: skipping {error}", flush=True)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("Figure S8 found no usable zero-attention probe records.")
    frame = frame.sort_values(
        ["num_parameters", "irun", "iteration"]
    ).reset_index(drop=True)
    utils.ensure_dir("./cache")
    frame.to_csv(FIGURE_S8_CACHE, index=False)
    print(
        f"Figure S8: wrote {len(frame)} rows from {frame['irun'].nunique()} runs "
        f"to {FIGURE_S8_CACHE}"
        + (f"; skipped {len(skipped)} unreadable runs." if skipped else "."),
        flush=True,
    )


def ensure_figure_s8_cache(force_regenerate=False, workers=None):
    """Create S8's zero-attention probe cache when absent or stale."""
    cache_is_usable = False
    if os.path.isfile(FIGURE_S8_CACHE):
        try:
            cache_is_usable = FIGURE_S8_CACHE_COLUMNS.issubset(
                pd.read_csv(FIGURE_S8_CACHE, nrows=0).columns
            )
        except (OSError, pd.errors.EmptyDataError):
            cache_is_usable = False
    if force_regenerate or not cache_is_usable:
        regenerate_figure_s8_cache(workers=workers)


def create_figure_s8(force_regenerate_cache=False):
    """Create Supplemental Figure S8: zero-attention probe accuracy at 1k steps."""
    ensure_figure_s8_cache(force_regenerate=force_regenerate_cache)
    apply_figure_style()
    frame = pd.read_csv(FIGURE_S8_CACHE)
    for column in FIGURE_S8_CACHE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=FIGURE_S8_CACHE_COLUMNS)
    frame = frame.loc[
        (frame["num_parameters"] > FIGURE_S8_MIN_PARAMETERS)
        & (frame["num_parameters"] < FIGURE_S8_MAX_PARAMETERS)
        & (frame["iteration"] == FIGURE_S8_TARGET_ITERATION)
    ].copy()
    if frame.empty:
        raise RuntimeError(
            "Figure S8 cache has no records after its parameter and iteration filters."
        )
    plot_frame = (
        frame.groupby("num_parameters", as_index=False)["probe_accuracy"]
        .mean()
        .sort_values("num_parameters")
    )

    fig, axis = plt.subplots(figsize=(7.0, 6.0), facecolor="white")
    sns.lineplot(
        data=plot_frame,
        x="num_parameters",
        y="probe_accuracy",
        marker="o",
        markersize=8,
        linewidth=1.8,
        color=sns.color_palette("deep")[0],
        errorbar=None,
        ax=axis,
    )
    axis.set_xscale("log")
    axis.set_ylim(0.45, 1.0)
    axis.set_xlabel("Number of Parameters")
    axis.set_ylabel("Probe Accuracy")
    axis.tick_params(which="both", direction="in")
    for spine in axis.spines.values():
        spine.set_linewidth(1.0)
    return fig


# =============================================================================
# FIGURE 5: Learned Attention Patterns
# =============================================================================
# This is the trained one-layer, 16-head model used for the attention-pattern
# visualization.  Keep the checkpoint selection explicit so that the figure is
# reproducible and cannot silently switch to a different experiment.
FIGURE5_RECORD_GLOB = (
    f"{RESULTS_DIR}/cache/"
    "memo_jan20_zipf_onelayerattention_vary_num_hidden_features_num_heads_doresample/"
    "memo_jan20_zipf_onelayerattention_vary_num_hidden_features_num_heads_doresample_"
    "transformer_K_100000_L_100_hidden_64_nheads_16*"
)


def create_figure5(sequence_idx=0, num_heads_to_plot=8,
                   record_glob=FIGURE5_RECORD_GLOB):
    """Create the camera-ready learned-attention visualization (Figure 5).

    The panel shows causal attention maps from the first held-out sequence of
    the 16-head, width-64 one-layer model.  A common [0, 1] colour scale makes
    attention magnitudes directly comparable across heads; the strict upper
    triangle is white because causal attention cannot attend to future keys.

    Args:
        sequence_idx: Index in the saved sequence batch to visualize.
        num_heads_to_plot: Number of heads to draw, up to eight (two rows of
            four panels).
        record_glob: Glob identifying the experiment directory containing the
            model record and its ``_all_sequences.pkl`` data file.

    Returns:
        The Matplotlib figure.  Saving is deliberately handled by ``main`` so
        callers can also embed the panel in a larger composition.
    """
    apply_figure_style()

    experiment_dirs = sorted(glob.glob(record_glob))
    if len(experiment_dirs) != 1:
        raise RuntimeError(
            "Figure 5 requires exactly one checkpoint directory matching "
            f"{record_glob!r}; found {len(experiment_dirs)}."
        )

    experiment_dir = experiment_dirs[0]
    experiment_name = os.path.basename(experiment_dir)
    record_path = os.path.join(experiment_dir, f"{experiment_name}.pkl")
    data_path = os.path.join(experiment_dir, f"{experiment_name}_all_sequences.pkl")
    if not os.path.isfile(record_path) or not os.path.isfile(data_path):
        raise FileNotFoundError(
            "Figure 5 checkpoint is incomplete. Expected both "
            f"{record_path!r} and {data_path!r}."
        )

    with open(record_path, "rb") as file:
        record = utils.CPU_Unpickler(file).load()
    with open(data_path, "rb") as file:
        data = pickle.load(file)

    args = record["args"]
    model = gpt.OneLayerAttention(
        args["len_context"],
        args["num_heads"],
        args["num_hidden_features"],
        args["vocab_size"],
        args["num_mlp_layers"],
    )
    model.load_state_dict(record["model"])
    model.eval()

    sequences = data["sequences"][-1]
    if not 0 <= sequence_idx < len(sequences):
        raise IndexError(
            f"sequence_idx={sequence_idx} is outside the saved batch of "
            f"{len(sequences)} sequences."
        )

    # Run only the selected sequence: this avoids an unnecessary full-batch
    # forward pass while retaining the batch dimension expected by the model.
    selected_sequence = sequences[sequence_idx : sequence_idx + 1]
    with torch.no_grad():
        _, attention = model.get_attention_weights(selected_sequence, None)

    if attention.ndim != 4:
        raise ValueError(
            "Expected attention weights with shape (batch, head, query, key), "
            f"but received {tuple(attention.shape)}."
        )
    num_heads_to_plot = min(num_heads_to_plot, attention.shape[1], 8)
    if num_heads_to_plot < 1:
        raise ValueError("The checkpoint has no attention heads to plot.")

    # Sized for a double-column NeurIPS figure.  Axis-number labels are kept
    # smaller than the descriptive labels so the eight dense panels stay clear.
    fig, axes = plt.subplots(
        2,
        4,
        figsize=(7.6, 4.0),
        facecolor="white",
        gridspec_kw={"hspace": 0.12, "wspace": 0.24},
    )
    axes = axes.ravel()

    cmap = plt.get_cmap("rocket").copy()
    cmap.set_bad(color="white")
    norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
    image = None
    for head in range(num_heads_to_plot):
        attention_map = attention[0, head].detach().cpu().numpy()
        future_mask = np.triu(np.ones_like(attention_map, dtype=bool), k=1)
        masked_attention_map = np.ma.array(attention_map, mask=future_mask)

        image = axes[head].imshow(
            masked_attention_map,
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
            aspect="equal",
        )
        axes[head].set_title(f"Head {head + 1}", fontsize=10, pad=3)
        if head >= 4:
            axes[head].set_xlabel("Key position", fontsize=9.5, labelpad=2)
        if head % 4 == 0:
            axes[head].set_ylabel("Query position", fontsize=9.5, labelpad=2)
        axes[head].set_xticks([0, 50])
        axes[head].set_yticks([0, 25, 50, 75])
        axes[head].tick_params(
            axis="both", which="major", labelsize=7.5, length=2.5, width=0.7,
            pad=1.5,
        )
        for spine in axes[head].spines.values():
            spine.set_linewidth(0.8)

    for axis in axes[num_heads_to_plot:]:
        axis.set_visible(False)

    # Reserve the right margin before placing the colourbar.  Applying this
    # adjustment afterwards would shift the last-column panels underneath it.
    fig.subplots_adjust(left=0.065, right=0.88, bottom=0.10, top=0.94)

    # A single colourbar, rather than one per panel, preserves panel area and
    # makes the shared normalization apparent at a glance.
    colorbar = fig.colorbar(
        image,
        ax=axes[:num_heads_to_plot].tolist(),
        fraction=0.024,
        pad=0.018,
        aspect=30,
        ticks=[0.0, 0.25, 0.50, 0.75, 1.0],
    )
    colorbar.set_label("Attention weight", fontsize=9.5, labelpad=5)
    colorbar.ax.tick_params(labelsize=7.5, length=2.5, width=0.7, pad=1.5)
    colorbar.outline.set_linewidth(0.8)

    return fig


# =============================================================================
# FIGURE 6: DAM schematic and data-switch accuracy
# =============================================================================
FIGURE6_DAM_RECORD_GLOB = (
    f"{RESULTS_DIR}/"
    "memo_dam_apr10_K_100_is_freeze_A_LearnAB/*.pt"
)
FIGURE6_INFONCE_RECORD_GLOB = (
    f"{RESULTS_DIR}/"
    "infonce_jan21_is_freeze_A_LearnAB/*.pt"
)
FIGURE6_SWITCH_STEP = 10_000
FIGURE6_REFERENCE_PDF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Figure6.pdf")


def _figure6_scalar(value):
    """Convert a scalar log value, including a zero-dimensional tensor, to float."""
    return float(value.item()) if isinstance(value, torch.Tensor) else float(value)


def load_figure6_accuracy_records(record_glob):
    """Load DAM-style records into the accuracy table used in Figure 6.

    The records contain one logging entry per natural training step.  Their
    second phase restarts ``step`` at zero, so ``natural_step`` is retained for
    the broken axis and ``step`` is shifted solely for provenance/debugging.
    """
    paths = sorted(glob.glob(record_glob))
    if not paths:
        raise FileNotFoundError(f"No Figure 6 records match {record_glob!r}.")

    rows = []
    for path in paths:
        record = torch.load(path, map_location=torch.device("cpu"), weights_only=False)
        args = record["args"]
        hidden_size = getattr(args, "H")
        num_steps = getattr(args, "NUM_STEPS", FIGURE6_SWITCH_STEP)
        for natural_step, log in enumerate(record["logs"], start=1):
            phase = "before" if log["phase"] == "train" else "after"
            phase_offset = 0 if phase == "before" else num_steps
            rows.append({
                "record_path": path,
                "step": _figure6_scalar(log["step"]) + phase_offset,
                "natural_step": natural_step,
                "loss": _figure6_scalar(log["loss"]),
                "accuracy": _figure6_scalar(log["accuracy"]),
                "phase": phase,
                "H": _figure6_scalar(hidden_size),
            })
        # Checkpoints retain many tensor-valued diagnostics.  Figure 6 needs
        # only the scalar rows above, so free each record before loading the
        # next one rather than accumulating the full sweep in memory.
        del args, record
        gc.collect()

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(f"Figure 6 records matched {record_glob!r} but contained no logs.")
    return frame


def _smooth_figure6_phase(frame, switch_step, epoch_bin, rolling_window):
    """Bin and causally smooth one side of Figure 6's data switch."""
    if frame.empty:
        return pd.DataFrame(columns=["natural_step", "accuracy"])

    smoothed = frame.copy()
    smoothed["epoch_bin"] = (smoothed["natural_step"] // epoch_bin).astype(int)
    smoothed = (
        smoothed.groupby("epoch_bin", as_index=False)
        .agg(natural_step=("natural_step", "mean"), accuracy=("accuracy", "mean"))
        .sort_values("natural_step")
    )
    smoothed["accuracy"] = (
        smoothed["accuracy"].rolling(window=rolling_window, min_periods=1).mean()
    )
    return smoothed


def _plot_figure6_broken_accuracy_panel(
    fig,
    grid_spec,
    frame,
    y_limits,
    switch_step=FIGURE6_SWITCH_STEP,
    epoch_bin=100,
    rolling_window=5,
):
    """Draw one DAM/InfoNCE accuracy panel with a discontinuity at the switch."""
    panel_grid = grid_spec.subgridspec(
        1,
        3,
        width_ratios=[1.0, 1.0, 0.075],
        wspace=0.08,
    )
    ax_left = fig.add_subplot(panel_grid[0, 0])
    ax_right = fig.add_subplot(panel_grid[0, 1], sharey=ax_left)
    cbar_ax = fig.add_subplot(panel_grid[0, 2])

    h_values = np.sort(frame["H"].unique())
    positive_h_values = h_values[h_values > 0]
    if len(positive_h_values) == 0:
        raise ValueError("Figure 6 requires strictly positive H values for its log color scale.")
    norm = mpl.colors.LogNorm(vmin=positive_h_values.min(), vmax=positive_h_values.max())
    # H is the model-size proxy in this sweep, so use the shared parameter
    # color encoding rather than the categorical turbo palette.
    cmap = plt.get_cmap(NUM_PARAMETERS_CMAP)

    for hidden_size in h_values:
        subset = frame[frame["H"] == hidden_size].sort_values("natural_step")
        pre_switch = subset[subset["natural_step"] < switch_step]
        post_switch = subset[subset["natural_step"] >= switch_step]
        pre_smooth = _smooth_figure6_phase(
            pre_switch, switch_step, epoch_bin, rolling_window
        )
        post_smooth = _smooth_figure6_phase(
            post_switch, switch_step, epoch_bin, rolling_window
        )
        color = cmap(norm(hidden_size))

        left_plot = pre_smooth[
            (pre_smooth["natural_step"] >= 0) & (pre_smooth["natural_step"] <= 2000)
        ]
        right_plot = post_smooth[
            (post_smooth["natural_step"] >= switch_step)
            & (post_smooth["natural_step"] <= switch_step + 2000)
        ]
        if not left_plot.empty:
            ax_left.plot(left_plot["natural_step"], left_plot["accuracy"], color=color, lw=2.0)
        if not right_plot.empty:
            ax_right.plot(right_plot["natural_step"], right_plot["accuracy"], color=color, lw=2.0)

    ax_right.axvline(switch_step, color="red", linestyle="--", linewidth=1.2)
    y_min, y_max = y_limits
    ax_right.text(
        switch_step + 55,
        y_min + 0.82 * (y_max - y_min),
        "Data\nSwitch",
        color="red",
        fontsize=PANEL_TITLE_FONT_SIZE * MAX_TYPOGRAPHY_SCALE,
        va="bottom",
    )

    ax_left.set_xlim(0, 2000)
    ax_right.set_xlim(switch_step, switch_step + 2000)
    ax_left.set_ylim(y_min, y_max)
    ax_left.set_ylabel("Accuracy", fontsize=13)
    ax_left.set_xticks([0, 1000, 2000])
    ax_right.set_xticks([switch_step, switch_step + 1000, switch_step + 2000])
    ax_left.set_xticklabels(["0.0", "0.1", "0.2"])
    ax_right.set_xticklabels(["1.0", "1.1", "1.2"])
    ax_left.get_xticklabels()[-1].set_horizontalalignment("right")
    ax_right.get_xticklabels()[0].set_horizontalalignment("left")
    ax_left.text(
        1.04,
        -0.12,
        r"Iteration ($\times 10^4$)",
        transform=ax_left.transAxes,
        ha="center",
        va="top",
        # This shared label is a text artist, so it bypasses the axis-label
        # standardizer used by the other panels.
        fontsize=AXIS_LABEL_FONT_SIZE * MAX_TYPOGRAPHY_SCALE,
    )

    for axis in [ax_left, ax_right]:
        axis.tick_params(which="both", direction="in", top=False, right=False, labelsize=11)
        for spine in axis.spines.values():
            spine.set_linewidth(1.2)
    ax_left.spines["right"].set_visible(False)
    ax_right.spines["left"].set_visible(False)
    ax_right.tick_params(labelleft=False, left=False)

    diagonal = 0.015
    slash_kwargs = dict(transform=ax_left.transAxes, color="black", clip_on=False, linewidth=1.0)
    ax_left.plot((1 - diagonal, 1 + diagonal), (-diagonal, +diagonal), **slash_kwargs)
    ax_left.plot((1 - diagonal, 1 + diagonal), (1 - diagonal, 1 + diagonal), **slash_kwargs)
    slash_kwargs.update(transform=ax_right.transAxes)
    ax_right.plot((-diagonal, +diagonal), (-diagonal, +diagonal), **slash_kwargs)
    ax_right.plot((-diagonal, +diagonal), (1 - diagonal, 1 + diagonal), **slash_kwargs)

    colorbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
    colorbar.set_label(r"$H$", fontsize=13)
    colorbar.ax.tick_params(labelsize=11)
    colorbar.formatter = mticker.LogFormatterMathtext(base=10)
    colorbar.update_ticks()
    _add_figure4_spines(cbar_ax, linewidth=1.2)
    return ax_left, ax_right, cbar_ax


def _draw_figure6_reference_panel_a(ax, reference_pdf=FIGURE6_REFERENCE_PDF):
    """Place the untouched Panel A artwork from the supplied Figure 6 PDF.

    Only the original panel label is blanked before drawing: ``create_figure6``
    adds a consistent analysis5-style ``a`` label outside the artwork.
    """
    if not os.path.isfile(reference_pdf):
        raise FileNotFoundError(
            "Figure 6 Panel A requires the supplied reference PDF at "
            f"{reference_pdf!r}."
        )
    ghostscript = shutil.which("gs")
    if ghostscript is None:
        raise RuntimeError("Figure 6 Panel A requires the Ghostscript executable 'gs'.")

    with tempfile.TemporaryDirectory(prefix="figure6_panel_a_") as temporary_dir:
        rendered_pdf = os.path.join(temporary_dir, "figure6_reference.png")
        try:
            subprocess.run(
                [
                    ghostscript, "-q", "-dSAFER", "-dBATCH", "-dNOPAUSE",
                    "-dFirstPage=1", "-dLastPage=1", "-sDEVICE=pngalpha", "-r300",
                    f"-sOutputFile={rendered_pdf}", reference_pdf,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                "Ghostscript could not render the Figure 6 reference PDF: "
                f"{error.stderr.strip()}"
            ) from error
        reference_image = plt.imread(rendered_pdf)

    height, width = reference_image.shape[:2]
    # These bounds isolate the left panel of the three-panel reference page.
    panel_a = reference_image[: int(0.92 * height), : int(0.278 * width)].copy()
    # The source panel's top-left label is the sole intended alteration.  The
    # replacement is placed by create_figure6 with the global figure styling.
    panel_a[: int(0.11 * panel_a.shape[0]), : int(0.12 * panel_a.shape[1])] = 1.0
    ax.imshow(panel_a, aspect="auto", interpolation="none")
    ax.axis("off")


def create_figure6(
    dam_record_glob=FIGURE6_DAM_RECORD_GLOB,
    infonce_record_glob=FIGURE6_INFONCE_RECORD_GLOB,
):
    """Create Figure 6: DAM schematic, DAM accuracy, and InfoNCE-DAM accuracy."""
    apply_figure_style()
    dam_frame = load_figure6_accuracy_records(dam_record_glob)
    infonce_frame = load_figure6_accuracy_records(infonce_record_glob)

    fig = plt.figure(figsize=(16.5, 5.0))
    outer_grid = fig.add_gridspec(
        1, 3, width_ratios=[1.05, 1.35, 1.35], wspace=0.30,
        left=0.04, right=0.985, bottom=0.16, top=0.90,
    )
    ax_a = fig.add_subplot(outer_grid[0, 0])
    _draw_figure6_reference_panel_a(ax_a)
    ax_b, ax_b_right, _ = _plot_figure6_broken_accuracy_panel(
        fig, outer_grid[0, 1], dam_frame, y_limits=(0.45, 1.0)
    )
    ax_c, ax_c_right, _ = _plot_figure6_broken_accuracy_panel(
        fig, outer_grid[0, 2], infonce_frame, y_limits=(0.0, 1.0)
    )

    fig.canvas.draw()
    for title, left_axis, right_axis in [
        ("DAM", ax_b, ax_b_right),
        ("InfoNCE DAM", ax_c, ax_c_right),
    ]:
        left_position = left_axis.get_position()
        right_position = right_axis.get_position()
        fig.text(
            (left_position.x0 + right_position.x1) / 2,
            max(left_position.y1, right_position.y1) + 0.010,
            title,
            fontsize=PANEL_TITLE_FONT_SIZE * MAX_TYPOGRAPHY_SCALE,
            ha="center",
            va="bottom",
        )
    for label, axis in zip("abc", [ax_a, ax_b, ax_c]):
        position = axis.get_position()
        fig.text(
            position.x0 - 0.018,
            position.y1 + 0.018,
            rf"\textbf{{{label}}}",
            fontsize=22,
            fontweight="bold",
            va="bottom",
            ha="left",
        )
    return fig


# =============================================================================
# SUPPLEMENTAL FIGURE 2: Uniform-distribution model-size sweep
# =============================================================================
FIGURE_S2_RECORD_GLOB = (
    f"{RESULTS_DIR}/cache/memo_jan14_uniformdist_modelsize/*"
)
FIGURE_S2_CACHE = "./cache/df_figure_s2_uniformdist_modelsize.csv"
FIGURE_S2_CACHE_COLUMNS = {
    "record_path", "num_heads", "num_layers", "num_parameters",
    "train_loss", "test_accuracy", "test_accuracy_rank_limit",
}
# A large run that ends at the binary chance baseline has not begun to learn.
# Restrict this screen to models whose capacity is otherwise sufficient, so
# small, genuinely capacity-limited architectures remain visible in S2.
FIGURE_S2_FAILURE_MIN_PARAMETERS = 1e5
FIGURE_S2_FAILURE_LOSS_THRESHOLD = 0.68
FIGURE_S2_FAILURE_ACCURACY_THRESHOLD = 0.55


def _figure_s2_argument(args, name):
    """Read an experiment argument stored as either a mapping or a namespace."""
    if isinstance(args, dict):
        return args[name]
    return getattr(args, name)


def _figure_s2_mean_scalar(value):
    """Reduce a scalar or array-like checkpoint value to a Python float."""
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().float().mean().item())
    return float(np.asarray(value, dtype=float).mean())


def _figure_s2_top_10000_accuracy(test_metrics):
    """Average accuracy over ranks 0--9,990 in the uniform-sweep evaluation.

    The evaluation records every tenth rank in ascending order, so the first
    1,000 accuracy values correspond exactly to the top 10,000 sequences.
    This is the same sampled-rank convention used for S1's 12-layer panel.
    """
    accuracy = np.asarray(test_metrics["accuracy"]).reshape(-1)
    if len(accuracy) < 1000:
        raise ValueError(
            "Figure S2 evaluation has fewer than 1,000 sampled ranks; "
            "cannot compute top-10,000 recall accuracy."
        )
    return float(accuracy[:1000].mean())


def _figure_s2_parameter_count(record, args):
    """Count GPT parameters, including checkpoints without a saved state dict."""
    try:
        parameters = list(record["model"].values())
        if not parameters or not all(hasattr(parameter, "numel") for parameter in parameters):
            raise ValueError("checkpoint does not retain tensor-valued model parameters")
        return sum(int(parameter.numel()) for parameter in parameters)
    except (KeyError, TypeError, ValueError, AttributeError):
        width = (
            int(_figure_s2_argument(args, "num_heads"))
            * int(_figure_s2_argument(args, "num_hidden_features"))
        )
        num_layers = int(_figure_s2_argument(args, "num_layers"))
        context_length = int(_figure_s2_argument(args, "len_context"))
        vocab_size = int(_figure_s2_argument(args, "vocab_size"))
        return (
            num_layers * (12 * width ** 2 + 13 * width)
            + (context_length + 2 * vocab_size + 2) * width
            + vocab_size
        )


def load_figure_s2_records(record_glob=FIGURE_S2_RECORD_GLOB):
    """Load completed uniform-distribution model-size sweep records for Figure S2.

    Each run stores its checkpoint as ``<run directory>/<run directory name>.pkl``.
    Runs with fewer than 100 logging entries are retained out of neither panel,
    matching the completion filter used for the original analysis.
    """
    run_paths = sorted(glob.glob(record_glob))
    if not run_paths:
        raise FileNotFoundError(
            "Figure S2 found no experiment directories matching "
            f"{record_glob!r}."
        )

    rows = []
    skipped = []
    missing_checkpoint_runs = 0
    for run_path in run_paths:
        if not os.path.isdir(run_path):
            skipped.append(run_path)
            continue
        record_paths = [
            path for path in glob.glob(os.path.join(run_path, "*.pkl"))
            if not path.endswith(("_all_sequences.pkl", "_model_final.pkl")) and "_model_iter_" not in os.path.basename(path)
        ]
        # Running jobs write the auxiliary sequence file before the model
        # checkpoint. This is expected, so omit those directories without a
        # per-run warning; a later forced regeneration will include them once
        # their main checkpoint appears.
        if len(record_paths) != 1:
            missing_checkpoint_runs += 1
            skipped.append(run_path)
            continue
        record_path = record_paths[0]
        try:
            with open(record_path, "rb") as file:
                record = utils.CPU_Unpickler(file).load()
        except (OSError, EOFError, pickle.UnpicklingError, RuntimeError, ValueError) as error:
            print(f"Figure S2: skipping unreadable record {record_path!r}: {error}", flush=True)
            skipped.append(record_path)
            continue

        logs = record.get("logs", [])
        if len(logs) < 100:
            skipped.append(record_path)
            continue
        try:
            args = record["args"]
            final_log = logs[-1]
            model_parameters = _figure_s2_parameter_count(record, args)
            if model_parameters <= 0:
                raise ValueError("checkpoint model has no tensor parameters")
            rows.append({
                "record_path": record_path,
                "num_heads": int(_figure_s2_argument(args, "num_heads")),
                "num_layers": int(_figure_s2_argument(args, "num_layers")),
                "num_parameters": model_parameters,
                "train_loss": _figure_s2_mean_scalar(final_log["train_loss"]),
                "test_accuracy": _figure_s2_top_10000_accuracy(
                    final_log["test_metrics"]
                ),
                "test_accuracy_rank_limit": 10_000,
            })
        except (KeyError, TypeError, ValueError, AttributeError) as error:
            print(f"Figure S2: skipping malformed record {record_path!r}: {error}", flush=True)
            skipped.append(record_path)
        finally:
            # The records include full model state dicts; retain only the scalar
            # panel inputs before moving to the next checkpoint.
            del record
            gc.collect()

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(
            "Figure S2 did not find a completed, readable record with at least "
            f"100 log entries under {record_glob!r}."
        )
    message = f"Figure S2: loaded {len(frame)} completed runs"
    if missing_checkpoint_runs:
        message += f"; ignored {missing_checkpoint_runs} run directories without a main checkpoint"
    other_skipped = len(skipped) - missing_checkpoint_runs
    if other_skipped:
        message += f"; skipped {other_skipped} incomplete/unreadable records"
    print(message + ".", flush=True)
    return frame


def regenerate_figure_s2_cache(workers=None):
    """Cache the scalar Uniform-distribution sweep inputs used by Figure S2."""
    # ``workers`` is accepted so this function can be invoked by the common
    # cache CLI. The checkpoints are large, so loading them serially avoids
    # multiplying their in-memory model-state footprint.
    del workers
    frame = load_figure_s2_records()
    utils.ensure_dir(os.path.dirname(FIGURE_S2_CACHE))
    frame.to_csv(FIGURE_S2_CACHE, index=False)
    print(f"Wrote {len(frame)} rows to {FIGURE_S2_CACHE}", flush=True)


def load_figure_s2_cache(force_regenerate=False):
    """Return a schema-validated S2 cache, rebuilding it only when needed."""
    cache_is_stale = force_regenerate or not os.path.isfile(FIGURE_S2_CACHE)
    if not cache_is_stale:
        try:
            cached = pd.read_csv(FIGURE_S2_CACHE)
            cache_is_stale = (
                cached.empty
                or not FIGURE_S2_CACHE_COLUMNS.issubset(cached.columns)
                or not cached["test_accuracy_rank_limit"].eq(10_000).all()
            )
        except (OSError, pd.errors.EmptyDataError):
            cache_is_stale = True
    if cache_is_stale:
        regenerate_figure_s2_cache()
    return pd.read_csv(FIGURE_S2_CACHE)


def filter_figure_s2_failed_runs(frame):
    """Drop large S2 models that end at the binary chance baseline.

    A run is excluded only when it has more than 100,000 parameters, final
    train loss at least 0.68 (near ``log(2)``), and top-10,000 recall accuracy
    at most 0.55.  The cached table remains complete for auditability.
    """
    failed = (
        (frame["num_parameters"] > FIGURE_S2_FAILURE_MIN_PARAMETERS)
        & (frame["train_loss"] >= FIGURE_S2_FAILURE_LOSS_THRESHOLD)
        & (frame["test_accuracy"] <= FIGURE_S2_FAILURE_ACCURACY_THRESHOLD)
    )
    return frame.loc[~failed].copy(), int(failed.sum())


def _plot_figure_s2_metric(ax, frame, metric, ylabel, title, legend_loc):
    """Plot one Figure S2 metric with a shared viridis head-count encoding."""
    mean_frame = (
        frame.groupby(["num_heads", "num_parameters"], as_index=False)[metric]
        .mean()
        .sort_values(["num_heads", "num_parameters"])
    )
    head_counts = np.sort(mean_frame["num_heads"].unique())
    colors = sns.color_palette(NUM_PARAMETERS_CMAP, n_colors=len(head_counts))
    for num_heads, color in zip(head_counts, colors):
        subset = mean_frame[mean_frame["num_heads"] == num_heads]
        ax.plot(
            subset["num_parameters"],
            subset[metric],
            color=color,
            linewidth=1.8,
            marker="+",
            markersize=6,
            markeredgecolor=color,
            markerfacecolor=color,
            label=rf"${num_heads:g}$",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Number of Parameters")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=7)
    ax.legend(title=r"$H$", loc=legend_loc, frameon=True)
    ax.tick_params(which="both", direction="in")
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)


def _plot_figure_s1_panel(ax, df_data, legend_loc=(0.41, 0.98)):
    """Plot recall accuracy vs parameter count with markers for H and viridis colormap for D."""
    H_values = sorted(df_data["num_heads"].unique())
    markers = ["o", "X", "s", "P", "D", "^", "v", "<", ">", "*", "h", "8"]
    marker_map = {h: markers[i % len(markers)] for i, h in enumerate(H_values)}

    D_values = sorted(df_data["num_hidden_features"].unique())
    cmap = mpl.cm.viridis
    norm = mpl.colors.LogNorm(vmin=min(D_values), vmax=max(D_values))

    for h in H_values:
        dh = df_data[df_data["num_heads"] == h]
        if len(dh) == 0:
            continue
        ax.scatter(
            dh["num_parameters"],
            dh["test_accuracy"],
            c=dh["num_hidden_features"],
            cmap=cmap,
            norm=norm,
            marker=marker_map[h],
            s=55,
            alpha=0.9,
            edgecolors="none",
        )

    ax.set_xlabel("Number of Parameters", fontsize=14)
    ax.set_ylabel("Recall Accuracy", fontsize=14)
    ax.set_xscale("log")
    ax.tick_params(which="both", direction="in", top=False, right=False, labelsize=11)

    # Colorbar for D
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = ax.figure.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(r"$D$", fontsize=14)
    cbar.ax.tick_params(labelsize=11)
    cbar.ax.yaxis.set_major_formatter(mpl.ticker.LogFormatterMathtext(base=10))

    # Legend for H
    style_handles = [
        Line2D([0], [0], marker=marker_map[h], linestyle="None", color="black",
               markerfacecolor="black", markersize=6.5, label=str(h))
        for h in H_values
    ]
    ax.legend(
        handles=style_handles,
        title=r"$H$",
        loc="upper right",
        bbox_to_anchor=legend_loc,
        bbox_transform=ax.transAxes,
        framealpha=0.9,
        ncol=2,
        fontsize=9,
        title_fontsize=10.5,
        labelspacing=0.25,
        handletextpad=0.2,
        borderpad=0.25,
        columnspacing=0.5,
    )


# The 1-layer S1 panel intentionally reuses Figure 2(a)'s Jan. 20
# no-resampling cache.  Keeping one cache prevents the two figures from
# silently diverging while preserving the exact experiment provenance.
FIGURE_S1_RIGHT_PANEL_CACHE = FIGURE2_BOTTOM_A_CACHE


def create_figure_s1(force_regenerate_cache=False, cache_parallel=False, workers=None):
    """Create Figure S1: Recall accuracy scales with parameter count.

    Caption:
        Figure S1: Recall accuracy scales with parameter count. Average next-bit
        accuracy on the top 10000 sequences increases with total parameters
        across architectures that vary in H and D.
    Left panel (a): 12-layer models
    Right panel (b): 1-layer models from
        ``memo_jan20_zipf_onelayerattention_vary_num_hidden_features_num_heads_noresample``.
    """
    ensure_figure2_caches(force_regenerate=force_regenerate_cache, parallel=cache_parallel, workers=workers)
    apply_figure_style()

    df_tl_12 = pd.read_csv(FIGURE2_TOP_A_CACHE)
    ok_12 = df_tl_12[~df_tl_12["failed"]]

    df_tl_1 = pd.read_csv(FIGURE_S1_RIGHT_PANEL_CACHE)
    last_epoch = df_tl_1["epoch"].max()
    df_tl_1 = df_tl_1[df_tl_1["epoch"] == last_epoch]
    df_tl_1 = filter_figure2_bottom_a_mlp_widths(df_tl_1)

    fig, axes = plt.subplots(
        1, 2, figsize=(14.0, 5.5), facecolor="white",
        gridspec_kw={"wspace": 0.32, "left": 0.08, "right": 0.96, "top": 0.85, "bottom": 0.14},
    )

    _plot_figure_s1_panel(
        axes[0],
        ok_12,
        legend_loc=(0.41, 0.98),
    )
    _plot_figure_s1_panel(
        axes[1],
        df_tl_1,
        legend_loc=(0.41, 0.98),
    )
    axes[1].set_ylim(0.45, 0.95)
    axes[0].set_title("12-layer models", pad=7)
    axes[1].set_title("1-layer models", pad=7)

    fig.canvas.draw()
    position_a = axes[0].get_position()
    position_b = axes[1].get_position()

    # Place panel labels just above the panel titles.
    fig.text(
        position_a.x0 - 0.035, position_a.y1 + 0.010, r"\textbf{a}",
        fontsize=22, fontweight="bold", va="bottom", ha="left",
    )
    fig.text(
        position_b.x0 - 0.035, position_b.y1 + 0.010, r"\textbf{b}",
        fontsize=22, fontweight="bold", va="bottom", ha="left",
    )

    return fig


def create_figure_s2(record_glob=FIGURE_S2_RECORD_GLOB, force_regenerate_cache=False):
    """Create Supplemental Figure S2 from the uniform-distribution sweep.

    Panel a reports final train loss and panel b reports final recall accuracy
    on the top 10,000 sequences, both against model parameter count. Line
    colour encodes the number of attention heads using the same viridis palette
    used for quantitative colour encodings elsewhere in this script.
    """
    apply_figure_style()
    if record_glob != FIGURE_S2_RECORD_GLOB:
        frame = load_figure_s2_records(record_glob)
    else:
        frame = load_figure_s2_cache(force_regenerate=force_regenerate_cache)
    frame, num_failed = filter_figure_s2_failed_runs(frame)
    if frame.empty:
        raise RuntimeError("Figure S2 has no runs after filtering failed models.")
    if num_failed:
        print(f"Figure S2: excluded {num_failed} stalled large-model runs.", flush=True)

    fig, axes = plt.subplots(
        1, 2, figsize=(14.0, 5.5), facecolor="white",
        gridspec_kw={"left": 0.08, "right": 0.98, "bottom": 0.14, "top": 0.88, "wspace": 0.16},
    )
    _plot_figure_s2_metric(
        axes[0], frame, "train_loss", "Train Loss",
        "Train Loss vs. Number of Parameters", legend_loc="upper right",
    )
    _plot_figure_s2_metric(
        axes[1], frame, "test_accuracy", "Recall Accuracy",
        "Recall Accuracy vs. Number of Parameters", legend_loc="upper left",
    )
    axes[0].set_ylim(0.1, 0.72)
    axes[1].set_ylim(0.49, 0.94)
    return fig


# =============================================================================
# SUPPLEMENTAL FIGURE 3: GPT training loss across model sizes
# =============================================================================
FIGURE_S3_RECORD_GLOB = (
    f"{RESULTS_DIR}/cache/"
    "memo_apr19_zipf_gpt2_vary_num_hidden_features_num_heads_noresample/*"
)


def _figure_s3_parameter_count(record, args):
    """Return a GPT checkpoint's parameter count, reconstructing it if needed."""
    try:
        parameters = list(record["model"].values())
        if not parameters or not all(hasattr(parameter, "numel") for parameter in parameters):
            raise ValueError("checkpoint does not retain tensor-valued model parameters")
        return sum(int(parameter.numel()) for parameter in parameters)
    except (KeyError, TypeError, ValueError, AttributeError):
        config = gpt.GPTConfig(
            block_size=int(_figure_s2_argument(args, "len_context")),
            n_embd=int(_figure_s2_argument(args, "num_heads"))
            * int(_figure_s2_argument(args, "num_hidden_features")),
            n_layer=int(_figure_s2_argument(args, "num_layers")),
            n_head=int(_figure_s2_argument(args, "num_heads")),
            bias=True,
            is_initialize_attention_weights_to_zero=_figure_s2_argument(
                args, "is_initialize_attention_weights_to_zero"
            ),
        )
        model = gpt.GPT(config, torch.nn.NLLLoss(reduction="none"))
        return sum(int(parameter.numel()) for parameter in model.state_dict().values())


def _figure_s3_first_thousand_mean(value):
    """Compute the recorded held-out accuracy over its first 1,000 sequences."""
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().reshape(-1)[:1000].float().mean().item())
    return float(np.asarray(value, dtype=float).reshape(-1)[:1000].mean())


def load_figure_s3_records(record_glob=FIGURE_S3_RECORD_GLOB):
    """Load per-epoch GPT losses and final held-out accuracy for Figure S3.

    The April 19 checkpoint directories also contain ``*_all_sequences.pkl``.
    Those auxiliary files are deliberately excluded so every retained row is
    tied to the model checkpoint from which it was measured.
    """
    run_paths = sorted(glob.glob(record_glob))
    if not run_paths:
        raise FileNotFoundError(
            "Figure S3 found no experiment directories matching "
            f"{record_glob!r}."
        )

    rows = []
    skipped = []
    for run_path in run_paths:
        if not os.path.isdir(run_path):
            skipped.append(run_path)
            continue
        record_paths = [
            path for path in glob.glob(os.path.join(run_path, "*.pkl"))
            if not path.endswith(("_all_sequences.pkl", "_model_final.pkl")) and "_model_iter_" not in os.path.basename(path)
        ]
        if len(record_paths) != 1:
            skipped.append(run_path)
            continue
        record_path = record_paths[0]
        try:
            with open(record_path, "rb") as file:
                record = utils.CPU_Unpickler(file).load()
        except (OSError, EOFError, pickle.UnpicklingError, RuntimeError, ValueError) as error:
            print(f"Figure S3: skipping unreadable record {record_path!r}: {error}", flush=True)
            skipped.append(record_path)
            continue

        logs = record.get("logs", [])
        if len(logs) < 100:
            skipped.append(record_path)
            del record
            gc.collect()
            continue
        try:
            args = record["args"]
            train_losses = [_figure_s2_mean_scalar(log["train_loss"]) for log in logs]
            num_parameters = _figure_s3_parameter_count(record, args)
            final_accuracy = _figure_s3_first_thousand_mean(
                logs[-1]["test_metrics"]["accuracy"]
            )
            final_loss = train_losses[-1]
            failed = (
                (num_parameters > 3e6 and final_loss > 0.21)
                or (num_parameters > 1e6 and final_loss > 0.4)
            )
            num_heads = int(_figure_s2_argument(args, "num_heads"))
            num_hidden_features = int(_figure_s2_argument(args, "num_hidden_features"))
            iterations_per_epoch = int(
                _figure_s2_argument(args, "num_iters_per_epoch")
            )
            rows.extend({
                "record_path": record_path,
                "epoch": epoch,
                "iteration": epoch * iterations_per_epoch,
                "train_loss": train_loss,
                "test_accuracy": final_accuracy,
                "num_parameters": num_parameters,
                "num_hidden_features": num_hidden_features * num_heads,
                "num_heads": num_heads,
                "failed": failed,
            } for epoch, train_loss in enumerate(train_losses))
        except (KeyError, TypeError, ValueError, AttributeError) as error:
            print(f"Figure S3: skipping malformed record {record_path!r}: {error}", flush=True)
            skipped.append(record_path)
        finally:
            del record
            gc.collect()

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(
            "Figure S3 did not find a completed, readable record with at least "
            f"100 log entries under {record_glob!r}."
        )
    print(
        f"Figure S3: loaded {frame['record_path'].nunique()} completed runs "
        f"({len(frame)} loss records)"
        + (f"; skipped {len(skipped)} incomplete/unreadable runs." if skipped else "."),
        flush=True,
    )
    return frame


def create_figure_s3(record_glob=FIGURE_S3_RECORD_GLOB):
    """Create Supplemental Figure S3: successful GPT train-loss trajectories.

    Each curve is a model configuration in the April 19 Zipfian GPT sweep;
    line color represents its parameter count on a logarithmic viridis scale.
    Models that did not meet the original final-loss criteria are omitted.
    """
    apply_figure_style()
    frame = load_figure_s3_records(record_glob)
    plot_frame = frame.loc[~frame["failed"]].copy()
    if plot_frame.empty:
        raise RuntimeError("Figure S3 has no models that pass the final-loss filter.")

    parameter_counts = plot_frame["num_parameters"].to_numpy(dtype=float)
    vmin = parameter_counts.min()
    vmax = parameter_counts.max()
    if vmin == vmax:
        vmin /= 10.0
        vmax *= 10.0
    norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6), facecolor="white")
    sns.lineplot(
        x="iteration",
        y="train_loss",
        data=plot_frame.sort_values(["num_parameters", "iteration"]),
        hue="num_parameters",
        palette=NUM_PARAMETERS_CMAP,
        hue_norm=norm,
        errorbar=None,
        legend=False,
        ax=ax,
    )
    ax.set_title("Train Loss vs. Iteration", pad=7)
    ax.set_xlabel(r"Iteration ($\times 10^4$)")
    ax.set_ylabel("Train Loss")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(50_000))
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda value, _: f"{value / 1e4:g}")
    )
    ax.tick_params(which="both", direction="in")
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    colorbar = fig.colorbar(
        mpl.cm.ScalarMappable(cmap=NUM_PARAMETERS_CMAP, norm=norm), ax=ax, pad=0.02
    )
    colorbar.set_label("Number of Parameters")
    colorbar.ax.tick_params(labelsize=11)
    colorbar.formatter = mticker.LogFormatterMathtext(base=10)
    colorbar.update_ticks()
    fig.tight_layout()
    return fig


def create_figure_s9(sequence_idx=9, num_heads_to_plot=16,
                     record_glob=FIGURE5_RECORD_GLOB):
    """Create the camera-ready Supplemental Figure 9 visualization (all 16 heads).

    Caption:
        Figure S9: Learned attention maps in a trained one-layer transformer for
        Sequence 10 (16 heads, 64 features per head). Each panel shows attention
        maps of one head, evaluated on the lowest-ranked sequence. Rows are query
        positions and columns are causal key positions.

    Args:
        sequence_idx: Sequence index to visualize (0-indexed; default 9 corresponds
            to Sequence 10, the lowest-ranked sequence in the evaluation set).
        num_heads_to_plot: Number of heads to plot (up to 16, arranged 4x4).
        record_glob: Glob matching the 16-head width-64 checkpoint.

    Returns:
        The Matplotlib figure object.
    """
    apply_figure_style()

    experiment_dirs = sorted(glob.glob(record_glob))
    if len(experiment_dirs) != 1:
        raise RuntimeError(
            "Figure S9 requires exactly one checkpoint directory matching "
            f"{record_glob!r}; found {len(experiment_dirs)}."
        )

    experiment_dir = experiment_dirs[0]
    experiment_name = os.path.basename(experiment_dir)
    record_path = os.path.join(experiment_dir, f"{experiment_name}.pkl")
    data_path = os.path.join(experiment_dir, f"{experiment_name}_all_sequences.pkl")
    if not os.path.isfile(record_path) or not os.path.isfile(data_path):
        raise FileNotFoundError(
            "Figure S9 checkpoint is incomplete. Expected both "
            f"{record_path!r} and {data_path!r}."
        )

    with open(record_path, "rb") as file:
        record = utils.CPU_Unpickler(file).load()
    with open(data_path, "rb") as file:
        data = pickle.load(file)

    args = record["args"]
    model = gpt.OneLayerAttention(
        args["len_context"],
        args["num_heads"],
        args["num_hidden_features"],
        args["vocab_size"],
        args["num_mlp_layers"],
    )
    model.load_state_dict(record["model"])
    model.eval()

    sequences = data["sequences"][-1]
    if not 0 <= sequence_idx < len(sequences):
        raise IndexError(
            f"sequence_idx={sequence_idx} is outside the saved batch of "
            f"{len(sequences)} sequences."
        )

    selected_sequence = sequences[sequence_idx : sequence_idx + 1]
    with torch.no_grad():
        _, attention = model.get_attention_weights(selected_sequence, None)

    if attention.ndim != 4:
        raise ValueError(
            "Expected attention weights with shape (batch, head, query, key), "
            f"but received {tuple(attention.shape)}."
        )
    num_heads_to_plot = min(num_heads_to_plot, attention.shape[1], 16)
    if num_heads_to_plot < 1:
        raise ValueError("The checkpoint has no attention heads to plot.")

    fig, axes = plt.subplots(
        4,
        4,
        figsize=(7.6, 7.6),
        facecolor="white",
        gridspec_kw={"hspace": 0.14, "wspace": 0.24},
    )
    axes = axes.ravel()

    cmap = plt.get_cmap("rocket").copy()
    cmap.set_bad(color="white")
    norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
    image = None
    for head in range(num_heads_to_plot):
        attention_map = attention[0, head].detach().cpu().numpy()
        future_mask = np.triu(np.ones_like(attention_map, dtype=bool), k=1)
        masked_attention_map = np.ma.array(attention_map, mask=future_mask)

        image = axes[head].imshow(
            masked_attention_map,
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
            aspect="equal",
        )
        axes[head].set_title(f"Head {head + 1}", fontsize=10, pad=3)
        if head >= 12:
            axes[head].set_xlabel("Key position", fontsize=9.5, labelpad=2)
        if head % 4 == 0:
            axes[head].set_ylabel("Query position", fontsize=9.5, labelpad=2)
        axes[head].set_xticks([0, 50])
        axes[head].set_yticks([0, 25, 50, 75])
        axes[head].tick_params(
            axis="both", which="major", labelsize=7.5, length=2.5, width=0.7,
            pad=1.5,
        )
        for spine in axes[head].spines.values():
            spine.set_linewidth(0.8)

    for axis in axes[num_heads_to_plot:]:
        axis.set_visible(False)

    fig.suptitle(
        r"Attention patterns for Sequence 10 -- 64 features, 16 heads",
        fontsize=12,
        y=0.965,
        fontweight="normal",
    )

    fig.subplots_adjust(left=0.065, right=0.88, bottom=0.06, top=0.925)

    colorbar = fig.colorbar(
        image,
        ax=axes[:num_heads_to_plot].tolist(),
        fraction=0.024,
        pad=0.018,
        aspect=30,
        ticks=[0.0, 0.25, 0.50, 0.75, 1.0],
    )
    colorbar.set_label("Attention weight", fontsize=9.5, labelpad=5)
    colorbar.ax.tick_params(labelsize=7.5, length=2.5, width=0.7, pad=1.5)
    colorbar.outline.set_linewidth(0.8)

    return fig


def create_figure_s10(head_idx=4, num_sequences=10,
                      record_glob=FIGURE5_RECORD_GLOB):
    """Create the camera-ready Supplemental Figure 10 visualization.

    Shows attention maps for a single attention head across multiple sequences:
    Head 5 (0-indexed head_idx=4) evaluated on Sequences 1 through 10.

    Suptitle:
        r"Attention patterns for Sequences 1 to 10 (Head 5 in a model with 64 features, 16 heads)",

    Args:
        head_idx: 0-indexed head index (default 4 corresponds to Head 5).
        num_sequences: Number of held-out sequences to visualize (default 10, arranged 2x5).
        record_glob: Glob pattern for the checkpoint.

    Returns:
        The Matplotlib figure object.
    """
    apply_figure_style()

    experiment_dirs = sorted(glob.glob(record_glob))
    if len(experiment_dirs) != 1:
        raise RuntimeError(
            "Figure S10 requires exactly one checkpoint directory matching "
            f"{record_glob!r}; found {len(experiment_dirs)}."
        )

    experiment_dir = experiment_dirs[0]
    experiment_name = os.path.basename(experiment_dir)
    record_path = os.path.join(experiment_dir, f"{experiment_name}.pkl")
    data_path = os.path.join(experiment_dir, f"{experiment_name}_all_sequences.pkl")
    if not os.path.isfile(record_path) or not os.path.isfile(data_path):
        raise FileNotFoundError(
            "Figure S10 checkpoint is incomplete. Expected both "
            f"{record_path!r} and {data_path!r}."
        )

    with open(record_path, "rb") as file:
        record = utils.CPU_Unpickler(file).load()
    with open(data_path, "rb") as file:
        data = pickle.load(file)

    args = record["args"]
    model = gpt.OneLayerAttention(
        args["len_context"],
        args["num_heads"],
        args["num_hidden_features"],
        args["vocab_size"],
        args["num_mlp_layers"],
    )
    model.load_state_dict(record["model"])
    model.eval()

    sequences = data["sequences"][-1]
    if len(sequences) < num_sequences:
        raise IndexError(
            f"num_sequences={num_sequences} exceeds available sequences ({len(sequences)})."
        )

    selected_sequences = sequences[:num_sequences]
    with torch.no_grad():
        _, attention = model.get_attention_weights(selected_sequences, None)

    fig, axes = plt.subplots(
        2,
        5,
        figsize=(9.2, 4.0),
        facecolor="white",
        gridspec_kw={"hspace": 0.14, "wspace": 0.24},
    )
    axes = axes.ravel()

    cmap = plt.get_cmap("rocket").copy()
    cmap.set_bad(color="white")
    norm = mpl.colors.PowerNorm(gamma=0.6, vmin=0.0, vmax=1.0)
    image = None
    for i in range(num_sequences):
        attn_map = attention[i, head_idx].detach().cpu().numpy()
        future_mask = np.triu(np.ones_like(attn_map, dtype=bool), k=1)
        masked_attention_map = np.ma.array(attn_map, mask=future_mask)

        image = axes[i].imshow(
            masked_attention_map,
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
            aspect="equal",
        )
        axes[i].set_title(f"Sequence {i + 1}", fontsize=10, pad=3)
        if i >= 5:  # bottom row
            axes[i].set_xlabel("Key position", fontsize=9.5, labelpad=2)
        if i % 5 == 0:  # leftmost column
            axes[i].set_ylabel("Query position", fontsize=9.5, labelpad=2)
        axes[i].set_xticks([0, 50])
        axes[i].set_yticks([0, 25, 50, 75])
        axes[i].tick_params(
            axis="both", which="major", labelsize=7.5, length=2.5, width=0.7,
            pad=1.5,
        )
        for spine in axes[i].spines.values():
            spine.set_linewidth(0.8)

    for axis in axes[num_sequences:]:
        axis.set_visible(False)

    fig.suptitle(
        r"Attention patterns for Sequences 1 to 10 (Head 5 in a model with 64 features, 16 heads)",
        fontsize=12,
        y=0.965,
        fontweight="normal",
    )

    fig.subplots_adjust(left=0.06, right=0.90, bottom=0.10, top=0.90)

    colorbar = fig.colorbar(
        image,
        ax=axes[:num_sequences].tolist(),
        fraction=0.020,
        pad=0.018,
        aspect=30,
        ticks=[0.0, 0.25, 0.50, 0.75, 1.0],
    )
    colorbar.set_label("Attention weight", fontsize=9.5, labelpad=5)
    colorbar.ax.tick_params(labelsize=7.5, length=2.5, width=0.7, pad=1.5)
    colorbar.outline.set_linewidth(0.8)

    return fig


# =============================================================================
# FIGURE 7: Effect of MLP width at matched attention width
# =============================================================================
# The canonical table lives with the other reusable camera-ready figure caches.
# The ``cache_oct3`` copy is retained only as a one-time migration source for
# the table that previously was (incorrectly) expected under ``figs/Figure4``.
FIGURE7_CACHE_DIR = "./cache"
FIGURE7_CACHE_FILENAME = (
    "train_loss_vs_num_parameters_"
    "memo_jan25_zipf_onelayerattention_vary_num_hidden_features_num_heads_noresample.csv"
)
FIGURE7_CACHE = os.path.join(FIGURE7_CACHE_DIR, FIGURE7_CACHE_FILENAME)
FIGURE7_LEGACY_CACHES = (
    os.path.join("./figs/Figure4", FIGURE7_CACHE_FILENAME),
    os.path.join("./cache/cache_oct3", FIGURE7_CACHE_FILENAME),
)
FIGURE7_RECORD_GLOBS = (
    f"{RESULTS_DIR}/cache/"
    "memo_jan24_zipf_onelayerattention_vary_num_hidden_features_num_heads_noresample/*",
    f"{RESULTS_DIR}/cache/"
    "memo_jan25_zipf_onelayerattention_vary_num_hidden_features_num_heads_noresample/*",
)
FIGURE7_SPECIAL_FH = (9, 16, 64, 128, 576, 1024, 4608)
FIGURE7_REQUIRED_COLUMNS = {
    "train_loss", "test_accuracy", "num_parameters", "num_hidden_features",
    "num_heads", "epoch", "num_hidden_features_mlp", "FH",
}


def _figure7_scalar(value):
    """Convert a scalar training metric from a checkpoint to ``float``."""
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _figure7_first_hundred_mean(value):
    """Compute the requested held-out accuracy summary without retaining tensors."""
    if isinstance(value, torch.Tensor):
        values = value.detach().cpu().reshape(-1)[:100].float().numpy()
    else:
        values = np.asarray(value, dtype=float).reshape(-1)[:100]
    return float(values.mean())


def _load_figure7_raw_records(record_globs=FIGURE7_RECORD_GLOBS):
    """Build the Figure 7 table from the Jan 24 and Jan 25 raw checkpoints."""
    folders = sorted({
        folder
        for record_glob in record_globs
        for folder in glob.glob(record_glob)
        if os.path.isdir(folder)
    })
    if not folders:
        raise FileNotFoundError(
            "Figure 7 found no experiment directories matching any of: "
            f"{record_globs!r}."
        )

    rows = []
    skipped = []
    for folder in folders:
        record_paths = [
            path for path in glob.glob(os.path.join(folder, "*.pkl"))
            if not path.endswith(("_all_sequences.pkl", "_model_final.pkl")) and "_model_iter_" not in os.path.basename(path)
        ]
        if len(record_paths) != 1:
            skipped.append(folder)
            continue
        record_path = record_paths[0]
        try:
            with open(record_path, "rb") as file:
                data = utils.CPU_Unpickler(file).load()
            logs = data.get("logs", [])
            if len(logs) < 10:
                skipped.append(record_path)
                continue

            args = data["args"]
            num_hidden_features = int(args["num_hidden_features"])
            num_heads = int(args["num_heads"])
            fh = num_hidden_features * num_heads
            if fh not in FIGURE7_SPECIAL_FH:
                continue
            num_hidden_features_mlp = int(
                args.get("num_hidden_features_mlp", fh)
            )
            num_parameters = sum(
                int(parameter.numel()) for parameter in data["model"].values()
            )
            test_accuracy = _figure7_first_hundred_mean(
                logs[9]["test_metrics"]["accuracy"]
            )
            rows.extend({
                "train_loss": _figure7_scalar(log["train_loss"]),
                "test_accuracy": test_accuracy,
                "num_parameters": num_parameters,
                "num_hidden_features": num_hidden_features,
                "num_heads": num_heads,
                "epoch": epoch,
                "num_hidden_features_mlp": num_hidden_features_mlp,
                "FH": fh,
            } for epoch, log in enumerate(logs) if "train_loss" in log)
        except (KeyError, TypeError, ValueError, OSError, EOFError,
                pickle.UnpicklingError, RuntimeError) as error:
            print(
                f"Figure 7: skipping unreadable or malformed record "
                f"{record_path!r}: {error}",
                flush=True,
            )
            skipped.append(record_path)
        finally:
            if "data" in locals():
                del data
            gc.collect()

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(
            "Figure 7 did not find any readable runs with at least ten logs "
            f"and FH in {FIGURE7_SPECIAL_FH}."
        )
    print(
        f"Figure 7: loaded {len(frame)} log records from "
        f"{frame[['num_parameters', 'FH', 'num_hidden_features_mlp']].drop_duplicates().shape[0]} "
        f"model configurations"
        + (f"; skipped {len(skipped)} incomplete/unreadable runs." if skipped else "."),
        flush=True,
    )
    return frame


def load_figure7_records(force_regenerate=False):
    """Load Figure 7 data, migrating the legacy CSV or rebuilding it if needed."""
    if not force_regenerate and os.path.isfile(FIGURE7_CACHE):
        frame = pd.read_csv(FIGURE7_CACHE)
        if FIGURE7_REQUIRED_COLUMNS.issubset(frame.columns):
            return frame

    if not force_regenerate:
        for legacy_cache in FIGURE7_LEGACY_CACHES:
            if not os.path.isfile(legacy_cache):
                continue
            frame = pd.read_csv(legacy_cache)
            if FIGURE7_REQUIRED_COLUMNS.issubset(frame.columns):
                utils.ensure_dir(FIGURE7_CACHE_DIR)
                # Preserve the historical cache as provenance; all future
                # Figure 7 reads use the camera-ready location above.
                shutil.copy2(legacy_cache, FIGURE7_CACHE)
                print(
                    f"Figure 7: migrated data table from {legacy_cache} to "
                    f"{FIGURE7_CACHE}.",
                    flush=True,
                )
                return frame

    frame = _load_figure7_raw_records()
    utils.ensure_dir(FIGURE7_CACHE_DIR)
    frame.to_csv(FIGURE7_CACHE, index=False)
    print(f"Figure 7: wrote {len(frame)} rows to {FIGURE7_CACHE}.", flush=True)
    return frame


def create_figure7(fit_degree=None, force_regenerate_cache=False):
    """Create Figure 7: train loss at iteration 10,000 across MLP widths.

    Colour encodes the attention width ``FH`` using viridis.  Crosses mark
    models whose MLP width equals ``FH``; circles mark all other MLP widths.
    When ``fit_degree`` is supplied, a least-squares polynomial is fit to all
    epoch-10 points of each FH in log10-parameter space, matching the x-axis.
    """
    if fit_degree is not None and fit_degree not in (2, 3):
        raise ValueError("Figure 7 supports only quadratic (2) or cubic (3) fits.")
    apply_figure_style()
    frame = load_figure7_records(force_regenerate=force_regenerate_cache).copy()
    numeric_columns = [
        "train_loss", "num_parameters", "epoch", "num_hidden_features_mlp", "FH",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=numeric_columns)
    plot_frame = frame.loc[frame["epoch"] == 10].copy()
    if plot_frame.empty:
        raise RuntimeError("Figure 7 data contain no records at epoch 10.")

    plot_frame["num_hidden_features_mlp=FH"] = (
        plot_frame["num_hidden_features_mlp"] == plot_frame["FH"]
    )
    nonmatching = plot_frame.loc[
        ~plot_frame["num_hidden_features_mlp=FH"]
        & plot_frame["FH"].isin(FIGURE7_SPECIAL_FH)
    ]
    matching = plot_frame.loc[plot_frame["num_hidden_features_mlp=FH"]]
    if nonmatching.empty and matching.empty:
        raise RuntimeError("Figure 7 has no usable records after its epoch and FH filters.")

    palette = dict(zip(
        FIGURE7_SPECIAL_FH,
        sns.color_palette(NUM_PARAMETERS_CMAP, n_colors=len(FIGURE7_SPECIAL_FH)),
    ))
    fig, ax = plt.subplots(1, 1, figsize=(7, 6), facecolor="white")
    if not nonmatching.empty:
        sns.scatterplot(
            x="num_parameters", y="train_loss", hue="FH", data=nonmatching,
            marker="o", s=35, edgecolor="black", linewidth=0.8,
            palette=palette, legend="full", alpha=1.0, zorder=2, ax=ax,
        )
    if not matching.empty:
        sns.scatterplot(
            x="num_parameters", y="train_loss", hue="FH", data=matching,
            marker="X", s=55, edgecolor="black", linewidth=0.8,
            palette=palette, legend=False, alpha=1.0, zorder=2, ax=ax,
        )

    if fit_degree is not None:
        for fh in FIGURE7_SPECIAL_FH:
            fh_frame = plot_frame.loc[plot_frame["FH"] == fh]
            if fh_frame.empty:
                continue
            log_parameters = np.log10(
                fh_frame["num_parameters"].to_numpy(dtype=float)
            )
            train_loss = fh_frame["train_loss"].to_numpy(dtype=float)
            if len(np.unique(log_parameters)) <= fit_degree:
                raise RuntimeError(
                    f"Figure 7 FH={fh} has too few distinct parameter counts "
                    f"for a degree-{fit_degree} fit."
                )
            coefficients = np.polyfit(log_parameters, train_loss, deg=fit_degree)
            log_grid = np.linspace(log_parameters.min(), log_parameters.max(), 256)
            ax.plot(
                10 ** log_grid,
                np.polyval(coefficients, log_grid),
                color=palette[fh],
                linewidth=1.6,
                zorder=1,
            )

    if ax.legend_ is not None:
        ax.legend(loc="lower left", title=r"$FH$", ncol=2, frameon=True)
    fit_title = (
        ""
        if fit_degree is None
        else f" ({'Quadratic' if fit_degree == 2 else 'Cubic'} Fits)"
    )
    ax.set_title("Train Loss vs. Number of Parameters" + fit_title, pad=7)
    ax.set_xlabel("Number of Parameters")
    ax.set_ylabel("Train Loss")
    ax.set_ylim(0.15, 0.75)
    ax.set_xscale("log")
    ax.tick_params(which="both", direction="in")
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
    fig.tight_layout()
    return fig


# =============================================================================
# FIGURE GENERATORS AND MAIN EXECUTION
# =============================================================================
def generate_figure1(savefigdir="./figs_camera_ready", force_regenerate_cache=False):
    utils.ensure_dir(savefigdir)
    print("Generating Figure 1...", flush=True)
    fig1 = create_figure1()
    standardize_figure_typography(fig1)
    for name in ["Figure1_combined", "Figure1"]:
        fig1.savefig(f"{savefigdir}/{name}.pdf", format="pdf", bbox_inches="tight")
        fig1.savefig(f"{savefigdir}/{name}.eps", format="eps", bbox_inches="tight")
        fig1.savefig(f"{savefigdir}/{name}.png", format="png", dpi=300, bbox_inches="tight")
    print(f"Saved Figure 1 to {savefigdir}", flush=True)
    plt.close(fig1)


def generate_figure2(savefigdir="./figs_camera_ready", force_regenerate_cache=False):
    utils.ensure_dir(savefigdir)
    print("Generating Figure 2...", flush=True)
    fig2 = create_figure2(force_regenerate_cache=force_regenerate_cache)
    standardize_figure_typography(fig2)
    for name in ["Figure2_combined", "Figure2"]:
        fig2.savefig(f"{savefigdir}/{name}.pdf", format="pdf", bbox_inches="tight")
        fig2.savefig(f"{savefigdir}/{name}.eps", format="eps", bbox_inches="tight")
        fig2.savefig(f"{savefigdir}/{name}.png", format="png", dpi=300, bbox_inches="tight")
    print(f"Saved Figure 2 to {savefigdir}", flush=True)
    plt.close(fig2)


generate_figure_2 = generate_figure2


def generate_figure4(savefigdir="./figs_camera_ready", force_regenerate_cache=False):
    utils.ensure_dir(savefigdir)
    print("Generating Figure 4...", flush=True)
    fig4 = create_figure4(force_regenerate_cache=force_regenerate_cache)
    standardize_figure_typography(fig4)

    master_pdf = f"{savefigdir}/Figure4.pdf"
    master_eps = f"{savefigdir}/Figure4.eps"
    master_png = f"{savefigdir}/Figure4.png"

    fig4.savefig(master_pdf, format="pdf", bbox_inches="tight")
    fig4.savefig(master_eps, format="eps", bbox_inches="tight")
    fig4.savefig(master_png, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig4)

    utils.ensure_dir(f"{savefigdir}/Figure4")
    utils.ensure_dir("./figs/Figure4")
    for prefix in [
        f"{savefigdir}/Figure4_combined",
        f"{savefigdir}/Figure4/Figure4",
        f"{savefigdir}/Figure4/Figure4_combined",
        "./figs/Figure4/Figure4",
        "./figs/Figure4/Figure4_combined",
    ]:
        shutil.copyfile(master_pdf, f"{prefix}.pdf")
        shutil.copyfile(master_eps, f"{prefix}.eps")
        shutil.copyfile(master_png, f"{prefix}.png")

    print(f"Saved Figure 4 to {savefigdir}", flush=True)


def generate_figure4pre(savefigdir="./figs_camera_ready", force_regenerate_cache=False):
    utils.ensure_dir(savefigdir)
    print("Generating Figure 4pre...", flush=True)
    fig4pre = create_figure4pre(force_regenerate_cache=force_regenerate_cache)
    standardize_figure_typography(fig4pre)
    for name in ["Figure4pre_combined", "Figure4pre"]:
        fig4pre.savefig(f"{savefigdir}/{name}.pdf", format="pdf", bbox_inches="tight")
        fig4pre.savefig(f"{savefigdir}/{name}.eps", format="eps", bbox_inches="tight")
        fig4pre.savefig(f"{savefigdir}/{name}.png", format="png", dpi=300, bbox_inches="tight")
    print(f"Saved Figure 4pre to {savefigdir}", flush=True)
    plt.close(fig4pre)


def generate_figure5(savefigdir="./figs_camera_ready", force_regenerate_cache=False):
    utils.ensure_dir(savefigdir)
    print("Generating Figure 5...", flush=True)
    fig5 = create_figure5()
    standardize_figure_typography(fig5)
    for extension in ["pdf", "eps"]:
        fig5.savefig(
            f"{savefigdir}/Figure5.{extension}",
            format=extension,
            bbox_inches="tight",
            facecolor="white",
        )
    fig5.savefig(
        f"{savefigdir}/Figure5.png",
        format="png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    print(f"Saved Figure 5 to {savefigdir}", flush=True)
    plt.close(fig5)


def generate_figure6(savefigdir="./figs_camera_ready", force_regenerate_cache=False):
    """Render Figure 6 from its DAM and InfoNCE-DAM source records."""
    utils.ensure_dir(savefigdir)
    print("Generating Figure 6...", flush=True)
    fig6 = create_figure6()
    standardize_figure_typography(fig6)
    for extension in ["pdf", "eps"]:
        fig6.savefig(
            f"{savefigdir}/Figure6.{extension}",
            format=extension,
            bbox_inches="tight",
            facecolor="white",
        )
    fig6.savefig(
        f"{savefigdir}/Figure6.png",
        format="png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    print(f"Saved Figure 6 to {savefigdir}", flush=True)
    plt.close(fig6)


def generate_figure7(savefigdir="./figs_camera_ready", force_regenerate_cache=False):
    """Render quadratic and cubic Figure 7 variants from the shared cache."""
    utils.ensure_dir(savefigdir)
    for fit_degree, name in [(2, "Figure7_quadratic"), (3, "Figure7_cubic")]:
        print(f"Generating Figure 7 ({name.rsplit('_', 1)[1]} fit)...", flush=True)
        fig7 = create_figure7(
            fit_degree=fit_degree,
            force_regenerate_cache=force_regenerate_cache,
        )
        standardize_figure_typography(fig7)
        for extension in ["pdf", "eps"]:
            fig7.savefig(
                f"{savefigdir}/{name}.{extension}",
                format=extension,
                bbox_inches="tight",
                facecolor="white",
            )
        fig7.savefig(
            f"{savefigdir}/{name}.png",
            format="png",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(fig7)
    print(f"Saved quadratic and cubic Figure 7 variants to {savefigdir}", flush=True)


def generate_figure_s1(savefigdir="./figs_camera_ready", force_regenerate_cache=False):
    """Render Supplemental Figure S1 from the 12-layer and 1-layer caches."""
    utils.ensure_dir(savefigdir)
    print("Generating Figure Supplemental 1...", flush=True)
    fig_s1 = create_figure_s1(force_regenerate_cache=force_regenerate_cache)
    standardize_figure_typography(fig_s1)
    for name in ["FigureS1", "FigureS1_combined", "Figure_Supplemental_1"]:
        for extension in ["pdf", "eps"]:
            fig_s1.savefig(
                f"{savefigdir}/{name}.{extension}",
                format=extension,
                bbox_inches="tight",
                facecolor="white",
            )
        fig_s1.savefig(
            f"{savefigdir}/{name}.png",
            format="png",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
    print(f"Saved Figure Supplemental 1 to {savefigdir}", flush=True)
    plt.close(fig_s1)


def generate_figure_s2(savefigdir="./figs_camera_ready", force_regenerate_cache=False):
    """Render Supplemental Figure S2 from its cached model-size sweep data."""
    utils.ensure_dir(savefigdir)
    print("Generating Figure Supplemental 2...", flush=True)
    fig_s2 = create_figure_s2(force_regenerate_cache=force_regenerate_cache)
    standardize_figure_typography(fig_s2)
    for name in ["FigureS2", "FigureS2_combined", "Figure_Supplemental_2"]:
        for extension in ["pdf", "eps"]:
            fig_s2.savefig(
                f"{savefigdir}/{name}.{extension}",
                format=extension,
                bbox_inches="tight",
                facecolor="white",
            )
        fig_s2.savefig(
            f"{savefigdir}/{name}.png",
            format="png",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
    print(f"Saved Figure Supplemental 2 to {savefigdir}", flush=True)
    plt.close(fig_s2)


def generate_figure_s3(savefigdir="./figs_camera_ready", force_regenerate_cache=False):
    """Render Supplemental Figure S3 from its raw GPT model-size sweep records."""
    utils.ensure_dir(savefigdir)
    print("Generating Figure Supplemental 3...", flush=True)
    fig_s3 = create_figure_s3()
    standardize_figure_typography(fig_s3)
    for name in ["FigureS3", "FigureS3_combined", "Figure_Supplemental_3"]:
        for extension in ["pdf", "eps"]:
            fig_s3.savefig(
                f"{savefigdir}/{name}.{extension}",
                format=extension,
                bbox_inches="tight",
                facecolor="white",
            )
        fig_s3.savefig(
            f"{savefigdir}/{name}.png",
            format="png",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
    print(f"Saved Figure Supplemental 3 to {savefigdir}", flush=True)
    plt.close(fig_s3)


def generate_figure_s4(savefigdir="./figs_camera_ready", force_regenerate_cache=False):
    """Render Supplemental Figure S4 from normalized saturation-event caches."""
    utils.ensure_dir(savefigdir)
    print("Generating Figure Supplemental 4...", flush=True)
    fig_s4 = create_figure_s4(force_regenerate_cache=force_regenerate_cache)
    standardize_figure_typography(fig_s4)
    for name in ["FigureS4", "FigureS4_combined", "Figure_Supplemental_4"]:
        for extension in ["pdf", "eps"]:
            fig_s4.savefig(
                f"{savefigdir}/{name}.{extension}",
                format=extension,
                bbox_inches="tight",
                facecolor="white",
            )
        fig_s4.savefig(
            f"{savefigdir}/{name}.png",
            format="png",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
    print(f"Saved Figure Supplemental 4 to {savefigdir}", flush=True)
    plt.close(fig_s4)


def generate_figure_s5(savefigdir="./figs_camera_ready", force_regenerate_cache=False):
    """Render Supplemental Figure S5 from rank-resolved May 1 data-switch logs."""
    utils.ensure_dir(savefigdir)
    print("Generating Figure Supplemental 5...", flush=True)
    fig_s5 = create_figure_s5(force_regenerate_cache=force_regenerate_cache)
    standardize_figure_typography(fig_s5)
    for name in ["FigureS5", "FigureS5_combined", "Figure_Supplemental_5"]:
        for extension in ["pdf", "eps"]:
            fig_s5.savefig(
                f"{savefigdir}/{name}.{extension}",
                format=extension,
                bbox_inches="tight",
                facecolor="white",
            )
        fig_s5.savefig(
            f"{savefigdir}/{name}.png",
            format="png",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
    print(f"Saved Figure Supplemental 5 to {savefigdir}", flush=True)
    plt.close(fig_s5)


def generate_figure_s6(savefigdir="./figs_camera_ready", force_regenerate_cache=False):
    """Render Supplemental Figure S6 from its compact Figure 2c-derived cache."""
    utils.ensure_dir(savefigdir)
    print("Generating Figure Supplemental 6...", flush=True)
    fig_s6 = create_figure_s6(force_regenerate_cache=force_regenerate_cache)
    standardize_figure_typography(fig_s6)
    for name in ["FigureS6", "FigureS6_combined", "Figure_Supplemental_6"]:
        for extension in ["pdf", "eps"]:
            fig_s6.savefig(
                f"{savefigdir}/{name}.{extension}",
                format=extension,
                bbox_inches="tight",
                facecolor="white",
            )
        fig_s6.savefig(
            f"{savefigdir}/{name}.png",
            format="png",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
    print(f"Saved Figure Supplemental 6 to {savefigdir}", flush=True)
    plt.close(fig_s6)


def generate_figure_s7(savefigdir="./figs_camera_ready", force_regenerate_cache=False):
    """Render Supplemental Figure S7 from its Apr. 3 fine-tuning cache."""
    utils.ensure_dir(savefigdir)
    print("Generating Figure Supplemental 7...", flush=True)
    fig_s7 = create_figure_s7(force_regenerate_cache=force_regenerate_cache)
    standardize_figure_typography(fig_s7)
    for name in ["FigureS7", "FigureS7_combined", "Figure_Supplemental_7"]:
        for extension in ["pdf", "eps"]:
            fig_s7.savefig(
                f"{savefigdir}/{name}.{extension}",
                format=extension,
                bbox_inches="tight",
                facecolor="white",
            )
        fig_s7.savefig(
            f"{savefigdir}/{name}.png",
            format="png",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
    print(f"Saved Figure Supplemental 7 to {savefigdir}", flush=True)
    plt.close(fig_s7)


def generate_figure_s8(savefigdir="./figs_camera_ready", force_regenerate_cache=False):
    """Render Supplemental Figure S8 from zero-attention probe checkpoints."""
    utils.ensure_dir(savefigdir)
    print("Generating Figure Supplemental 8...", flush=True)
    fig_s8 = create_figure_s8(force_regenerate_cache=force_regenerate_cache)
    standardize_figure_typography(fig_s8)
    for name in ["FigureS8", "FigureS8_combined", "Figure_Supplemental_8"]:
        for extension in ["pdf", "eps"]:
            fig_s8.savefig(
                f"{savefigdir}/{name}.{extension}",
                format=extension,
                bbox_inches="tight",
                facecolor="white",
            )
        fig_s8.savefig(
            f"{savefigdir}/{name}.png",
            format="png",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
    print(f"Saved Figure Supplemental 8 to {savefigdir}", flush=True)
    plt.close(fig_s8)


def generate_figure_s9(savefigdir="./figs_camera_ready", force_regenerate_cache=False):
    utils.ensure_dir(savefigdir)
    print("Generating Figure Supplemental 9...", flush=True)
    fig_s9 = create_figure_s9()
    standardize_figure_typography(fig_s9)
    for name in ["FigureS9", "FigureS9_combined", "Figure_Supplemental_9"]:
        for extension in ["pdf", "eps"]:
            fig_s9.savefig(
                f"{savefigdir}/{name}.{extension}",
                format=extension,
                bbox_inches="tight",
                facecolor="white",
            )
        fig_s9.savefig(
            f"{savefigdir}/{name}.png",
            format="png",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
    print(f"Saved Figure Supplemental 9 to {savefigdir}", flush=True)
    plt.close(fig_s9)


def generate_figure_s10(savefigdir="./figs_camera_ready", force_regenerate_cache=False):
    utils.ensure_dir(savefigdir)
    print("Generating Figure Supplemental 10...", flush=True)
    fig_s10 = create_figure_s10()
    standardize_figure_typography(fig_s10)
    for name in ["FigureS10", "FigureS10_combined", "Figure_Supplemental_10"]:
        for extension in ["pdf", "eps"]:
            fig_s10.savefig(
                f"{savefigdir}/{name}.{extension}",
                format=extension,
                bbox_inches="tight",
                facecolor="white",
            )
        fig_s10.savefig(
            f"{savefigdir}/{name}.png",
            format="png",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
    print(f"Saved Figure Supplemental 10 to {savefigdir}", flush=True)
    plt.close(fig_s10)


FIGURE_GENERATORS = {
    "1": generate_figure1,
    "2": generate_figure2,
    "3": generate_figure3,
    "4": generate_figure4,
    "5": generate_figure5,
    "6": generate_figure6,
    "7": generate_figure7,
    "s1": generate_figure_s1,
    "s2": generate_figure_s2,
    "s3": generate_figure_s3,
    "s4": generate_figure_s4,
    "s5": generate_figure_s5,
    "s6": generate_figure_s6,
    "s7": generate_figure_s7,
    "s8": generate_figure_s8,
    "s9": generate_figure_s9,
    "s10": generate_figure_s10,
}

FIGURE_ALIASES = {
    "1": "1", "fig1": "1", "figure1": "1",
    "2": "2", "fig2": "2", "figure2": "2", "figure_2": "2", "fig_2": "2", "create_figure_2": "2",
    "3": "3", "fig3": "3", "figure3": "3",
    "4": "4", "fig4": "4", "figure4": "4",
    "5": "5", "fig5": "5", "figure5": "5",
    "6": "6", "fig6": "6", "figure6": "6",
    "7": "7", "fig7": "7", "figure7": "7",
    "s1": "s1", "figs1": "s1", "figures1": "s1", "s-1": "s1", "figure_s1": "s1",
    "s2": "s2", "figs2": "s2", "figures2": "s2", "s-2": "s2", "figure_s2": "s2",
    "s3": "s3", "figs3": "s3", "figures3": "s3", "s-3": "s3", "figure_s3": "s3",
    "s4": "s4", "figs4": "s4", "figures4": "s4", "s-4": "s4", "figure_s4": "s4",
    "s5": "s5", "figs5": "s5", "figures5": "s5", "s-5": "s5", "figure_s5": "s5",
    "s6": "s6", "figs6": "s6", "figures6": "s6", "s-6": "s6", "figure_s6": "s6",
    "s7": "s7", "figs7": "s7", "figures7": "s7", "s-7": "s7", "figure_s7": "s7",
    "s8": "s8", "figs8": "s8", "figures8": "s8", "s-8": "s8", "figure_s8": "s8",
    "s9": "s9", "figs9": "s9", "figures9": "s9", "s-9": "s9", "figure_s9": "s9",
    "s10": "s10", "figs10": "s10", "figures10": "s10", "s-10": "s10", "figure_s10": "s10",
}
ALL_FIGURE_KEYS = ["1", 
  "2", 
  "3", "4", "5", "6", "7", "s1",
  "s2",
  "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10"]


def _run_single_figure_task(fig_key, savefigdir, force_regenerate_cache=False):
    start = time.time()
    generator = FIGURE_GENERATORS[fig_key]
    generator(savefigdir=savefigdir, force_regenerate_cache=force_regenerate_cache)
    duration = time.time() - start
    return fig_key, duration


def parse_figure_list(figures_arg):
    if not figures_arg:
        return ALL_FIGURE_KEYS

    selected = []
    for item in figures_arg:
        item_lower = str(item).strip().lower()
        if item_lower == "all":
            return ALL_FIGURE_KEYS
        for sub in item_lower.split(","):
            sub = sub.strip()
            if not sub:
                continue
            if sub == "all":
                return ALL_FIGURE_KEYS
            if sub in FIGURE_ALIASES:
                canonical = FIGURE_ALIASES[sub]
                if canonical not in selected:
                    selected.append(canonical)
            else:
                valid = ", ".join(sorted(set(FIGURE_ALIASES.keys())))
                raise ValueError(f"Unknown figure identifier: '{sub}'. Valid options: {valid}, all")
    return selected or ALL_FIGURE_KEYS


CACHE_TARGET_REGISTRY = {
    "fig2_top_ab": regenerate_figure2_top_ab_cache,
    "fig2_top_c": regenerate_figure2_top_c_cache,
    "fig2_bottom_a": regenerate_figure2_bottom_a_cache,
    "fig2_bottom_b": regenerate_figure2_bottom_b_cache,
    # Retained as a backwards-compatible cache-target alias for panel (a).
    "fig2_bottom_ab": regenerate_figure2_bottom_a_cache,
    "fig2_bottom_c": regenerate_figure2_bottom_c_cache,
    "fig2_top": regenerate_figure2_top_cache,
    "fig2": lambda workers=None: ensure_figure2_caches(force_regenerate=True, parallel=False, workers=workers),
    "fig4": regenerate_figure4_caches,
    "fig4pre": regenerate_figure4pre_cache,
    "s4": lambda workers=None: regenerate_figure_s4_caches(),
    "s5": regenerate_figure_s5_cache,
    "s6": lambda workers=None: regenerate_figure_s6_cache(),
    "s7": regenerate_figure_s7_cache,
    "s8": regenerate_figure_s8_cache,
    "s2": regenerate_figure_s2_cache,
    "all": lambda workers=None: force_regenerate_all_figure_caches(parallel=False, workers=workers),
}


def run_cache_regeneration(cache_args, parallel=False, workers=None):
    """Run standalone cache regeneration for specified target tables."""
    if not cache_args:
        cache_args = ["fig2"]
    total_workers = get_default_workers(workers)
    print(f"Cache targets requested: {cache_args} with {total_workers} CPUs (parallel={parallel})", flush=True)
    for target in cache_args:
        t_clean = str(target).lower().strip()
        if t_clean in ["fig2", "figure2"]:
            ensure_figure2_caches(force_regenerate=True, parallel=parallel, workers=total_workers)
        elif t_clean in ["all"]:
            force_regenerate_all_figure_caches(parallel=parallel, workers=total_workers)
        elif t_clean in CACHE_TARGET_REGISTRY:
            CACHE_TARGET_REGISTRY[t_clean](workers=total_workers)
        else:
            valid = ", ".join(sorted(CACHE_TARGET_REGISTRY.keys()))
            raise ValueError(f"Unknown cache target: '{target}'. Valid targets: {valid}")


def main(figures=None, parallel=False, workers=None, savefigdir="./figs_camera_ready", force_regenerate_cache=False, cache_parallel=False):
    utils.ensure_dir(savefigdir)
    selected_keys = parse_figure_list(figures)

    if force_regenerate_cache:
        print(
            "Forcing cache regeneration only for the selected figure(s); "
            "use '--cache all' to rebuild every figure cache.",
            flush=True,
        )

    print(f"Figures selected to generate: {[k.upper() for k in selected_keys]}", flush=True)

    if parallel and len(selected_keys) > 1:
        num_workers = workers or min(len(selected_keys), os.cpu_count() or 8, 8)
        print(f"Generating {len(selected_keys)} figures in parallel using {num_workers} processes...", flush=True)
        start_all = time.time()
        failed = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(_run_single_figure_task, key, savefigdir, force_regenerate_cache): key
                for key in selected_keys
            }
            for future in as_completed(futures):
                key = futures[future]
                try:
                    f_key, dur = future.result()
                    print(f"[Figure {f_key.upper()}] Done in {dur:.2f}s", flush=True)
                except Exception as e:
                    print(f"[Figure {key.upper()}] ERROR: {e}", flush=True)
                    import traceback
                    traceback.print_exc()
                    failed.append(key)
        total_elapsed = time.time() - start_all
        if failed:
            raise RuntimeError(f"The following figures failed to generate: {failed}")
        print(f"All {len(selected_keys)} figures completed in parallel in {total_elapsed:.2f}s", flush=True)
    else:
        start_all = time.time()
        for key in selected_keys:
            f_key, dur = _run_single_figure_task(key, savefigdir, force_regenerate_cache)
            print(f"[Figure {f_key.upper()}] Done in {dur:.2f}s", flush=True)
        total_elapsed = time.time() - start_all
        print(f"All {len(selected_keys)} figures completed in {total_elapsed:.2f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate publication-quality camera-ready figures."
    )
    parser.add_argument(
        "--figures",
        "--figure",
        nargs="+",
        default=["all"],
        help="Figures to generate: 1 through 7, S1 through S10, or all. Default: all",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Generate selected figures concurrently using multiple worker processes.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel worker processes (default: min(num_figures, 8)).",
    )
    parser.add_argument(
        "--output_dir",
        default="./figs_camera_ready",
        help="Directory where output figures will be written (default: ./figs_camera_ready).",
    )
    parser.add_argument(
        "--force_regenerate_cache",
        action="store_true",
        help="Rebuild caches used by the selected figure(s) before rendering.",
    )
    parser.add_argument(
        "--regenerate_cache",
        "--cache",
        nargs="*",
        default=None,
        help=(
            "Directly rebuild specified caches without rendering figures. "
            "Options: fig2, fig2_top_ab, fig2_top_c, fig2_bottom_a, fig2_bottom_b, "
            "fig2_bottom_c, "
            "fig2_top, fig4, fig4pre, s2, all."
        ),
    )
    parser.add_argument(
        "--cache_parallel",
        action="store_true",
        help="Run independent cache regeneration stages concurrently.",
    )
    cli_args = parser.parse_args()

    if cli_args.regenerate_cache is not None:
        run_cache_regeneration(
            cli_args.regenerate_cache,
            parallel=cli_args.cache_parallel,
            workers=cli_args.workers,
        )
    else:
        main(
            figures=cli_args.figures,
            parallel=cli_args.parallel,
            workers=cli_args.workers,
            savefigdir=cli_args.output_dir,
            force_regenerate_cache=cli_args.force_regenerate_cache,
            cache_parallel=cli_args.cache_parallel,
        )
