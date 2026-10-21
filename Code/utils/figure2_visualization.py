"""Figure 2: spatial-block out-of-fold predictions and predictive performance.

Visualization only. This module never fits a model, recalculates R² or changes
observations. Data passed to plot_figure2 must come from the original workflow.

Figure 2a: 21 Complaint-only-model OOF observed/predicted scatter plots.
Figure 2b: 21 Complaint-only-model OOF R², grouped by health domain.

Compatible with matplotlib, pandas, and numpy; no seaborn dependency.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


# ORIGINAL Figure 2a palette (from the user-provided source):
# pd.factorize(df_pred["City"]) in first-appearance order, then tab20[:n_city].
# Do NOT replace this with tab10 or an alphabetical city-to-color mapping.
EXTRA_CITY_COLOR = "#777777"


def _original_city_palette(df_pred: pd.DataFrame) -> dict:
    """Exactly recover the user's original pd.factorize + tab20 city colors."""
    _, labels = pd.factorize(df_pred["City"])
    cmap = plt.get_cmap("tab20")
    if len(labels) > len(cmap.colors):
        raise ValueError("Original tab20 scheme supports at most 20 cities.")
    return dict(zip(labels, cmap.colors[:len(labels)]))


def _validate_inputs(df_pred: pd.DataFrame, df_comparison: pd.DataFrame,
                     measure_col: str, category_col: str, value_col: str) -> None:
    required_pred = {"City", "y_variable", "y_true", "y_pred"}
    required_perf = {measure_col, category_col, value_col}
    if required_pred - set(df_pred):
        raise ValueError(f"Missing prediction columns: {sorted(required_pred - set(df_pred))}")
    if required_perf - set(df_comparison):
        raise ValueError(f"Missing comparison columns: {sorted(required_perf - set(df_comparison))}")

    pred_measures = set(df_pred["y_variable"].dropna().astype(str))
    perf_measures = set(df_comparison[measure_col].dropna().astype(str))
    if len(pred_measures) != 21 or len(perf_measures) != 21:
        raise ValueError(
            f"Expected 21 health measures in each input; "
            f"found predictions={len(pred_measures)}, metrics={len(perf_measures)}."
        )
    if pred_measures != perf_measures:
        raise ValueError("The health-measure sets in the two dataframes do not match.")
    if df_comparison[measure_col].duplicated().any():
        raise ValueError("Figure 2b requires one performance row per health measure.")
    if df_pred[["City", "y_variable", "y_true", "y_pred"]].isna().any().any():
        raise ValueError("Figure 2a has missing city, health measure or prediction values.")
    if not np.isfinite(df_pred[["y_true", "y_pred"]].to_numpy(dtype=float)).all():
        raise ValueError("Figure 2a contains non-finite observed or predicted values.")
    if not np.isfinite(df_comparison[value_col].to_numpy(dtype=float)).all():
        raise ValueError("Figure 2b contains non-finite R² values.")


def _scatter_panel(ax, subset: pd.DataFrame, colors: dict,
                   point_size: float, point_alpha: float,
                   identity_linewidth: float,
                   tick_fontsize: float) -> None:
    observed = subset["y_true"].to_numpy(dtype=float)
    predicted = subset["y_pred"].to_numpy(dtype=float)
    low = float(min(observed.min(), predicted.min()))
    high = float(max(observed.max(), predicted.max()))
    if low == high:
        pad = max(0.5, abs(low) * 0.02)
        low, high = low - pad, high + pad
    else:
        pad = (high - low) * 0.035
        low, high = low - pad, high + pad

    # Exact 1:1 data scales AND physical axes box; no asymmetric clipping.
    ax.set(xlim=(low, high), ylim=(low, high))
    ax.set_aspect("equal", adjustable="box")
    ax.set_box_aspect(1.0)
    ax.plot([low, high], [low, high], linestyle=(0, (3, 2)),
            color="#555555", lw=identity_linewidth, zorder=1)

    # ORIGINAL NYC handling: non-New York first, New York overlaid last.
    # NYC's color is from the original factorize + tab20 palette; there is
    # no substitution using the new fixed tab10 color scheme.
    non_ny = subset.loc[subset["City"] != "New York"]
    ny = subset.loc[subset["City"] == "New York"]
    if not non_ny.empty:
        ax.scatter(non_ny["y_true"], non_ny["y_pred"],
                   s=point_size, alpha=point_alpha,
                   color=non_ny["City"].map(colors).tolist(),
                   edgecolors="none", rasterized=True, zorder=2)
    if not ny.empty:
        ax.scatter(ny["y_true"], ny["y_pred"],
                   s=point_size, alpha=point_alpha,
                   color=ny["City"].map(colors).tolist(),
                   edgecolors="none", rasterized=True, zorder=3)

    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.tick_params(axis="both", labelsize=tick_fontsize,
                   width=0.6, length=2.2, pad=1.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)
    ax.grid(False)


def _plot_figure2a_predictions(
    fig, subplot_spec, df_pred,
    n_rows=5, n_cols=5,
    wspace=0.34, hspace=0.42,
    point_size=1.15, point_alpha=0.32,
    identity_linewidth=0.85,
    title_fontsize=8.7, tick_fontsize=6.7,
    label_fontsize=10.5,
    legend_ncol=5, legend_fontsize=7.4, legend_marker_size=5.0,
    legend_columnspacing=0.85, legend_handletextpad=0.28,
):
    """Render Figure 2a; the grid's unused last cells hold the city legend."""
    health_order = sorted(df_pred["y_variable"].astype(str).unique(), key=str.upper)
    if n_rows * n_cols < len(health_order):
        raise ValueError("Prediction subplot grid is too small.")
    if n_rows * n_cols == len(health_order):
        raise ValueError("Reserve at least one empty cell for the city legend.")
    if n_rows != 5 or n_cols != 5:
        raise ValueError("This optimized layout uses exactly 5 rows × 5 columns.")

    colors = _original_city_palette(df_pred)
    inner = subplot_spec.subgridspec(n_rows, n_cols,
                                     wspace=wspace, hspace=hspace)
    axes = []
    for i, health in enumerate(health_order):
        ax = fig.add_subplot(inner[i // n_cols, i % n_cols])
        subset = df_pred.loc[df_pred["y_variable"] == health]
        _scatter_panel(ax, subset, colors, point_size, point_alpha,
                       identity_linewidth, tick_fontsize)
        ax.set_title(health, fontsize=title_fontsize, pad=3.0, color="#202020")
        axes.append(ax)

    # The last row has one scatter and four free slots (21 measures in 5×5).
    remainder = len(health_order) % n_cols
    start_col = remainder if remainder else n_cols  # safety; 21 => 1
    legend_ax = fig.add_subplot(inner[n_rows - 1, start_col:n_cols])
    legend_ax.axis("off")
    handles = [
        Line2D([0], [0], linestyle="None", marker="o",
               markerfacecolor=color, markeredgecolor="none",
               markersize=legend_marker_size, label=city)
        for city, color in sorted(colors.items())
    ]
    legend_ax.legend(handles=handles, loc="center", ncol=legend_ncol,
                     fontsize=legend_fontsize, frameon=False,
                     borderaxespad=0, columnspacing=legend_columnspacing,
                     handletextpad=legend_handletextpad,
                     handlelength=1.0, labelspacing=0.95)

    # One shared pair of axis labels avoids 21 repetitions.
    bbox = subplot_spec.get_position(fig)
    fig.text(bbox.x0 - 0.032, (bbox.y0 + bbox.y1) / 2,
             "Predicted PLACES estimate (%)", va="center", ha="center",
             rotation=90, fontsize=label_fontsize, color="#202020")
    fig.text((bbox.x0 + bbox.x1) / 2, bbox.y0 - 0.035,
             "Observed PLACES estimate (%)", va="center", ha="center",
             fontsize=label_fontsize, color="#202020")
    return axes, legend_ax, colors, health_order


def _plot_figure2b_on_axis(
    ax, df_comparison,
    value_col="r2_complaint", measure_col="y_variable", category_col="legends",
    bar_width=0.57,
    color_complaint="#769BB9", bar_fill_alpha=0.26, bar_linewidth=1.2,
    value_fontsize=8, value_offset=0.015, value_format="{:.2f}",
    ylim=(0, 1.0), ylabel="Spatial out-of-fold R²",
    label_fontsize=10.5, category_fontsize=10.0, tick_fontsize=8.5,
    separator_color="#AAAAAA", separator_linewidth=0.8,
    separator_linestyle=(0, (3, 3)),
    category_y_fraction=0.90,
    width_height_ratio=1.15,
):
    """Render Figure 2b; use supplied R² unchanged (no recalculation)."""
    # Preserve first-appearance health-domain order from the metadata file.
    category_order = df_comparison[category_col].drop_duplicates().tolist()
    grouped = []
    for category in category_order:
        group = (df_comparison.loc[df_comparison[category_col] == category]
                 .sort_values(value_col, ascending=False, kind="stable"))
        grouped.append(group)
    plot_df = pd.concat(grouped, ignore_index=True)
    values = plot_df[value_col].to_numpy(dtype=float)
    if ylim is not None:
        if values.min() < ylim[0] or values.max() > ylim[1]:
            raise ValueError("Figure 2b R² values exceed ylim; adjust limits instead of hiding bars.")

    xpos = np.arange(len(plot_df))
    bars = ax.bar(xpos, values, width=bar_width,
                  facecolor=to_rgba(color_complaint, bar_fill_alpha),
                  edgecolor=color_complaint, lw=bar_linewidth, zorder=2)
    for rect, value in zip(bars, values):
        ax.text(rect.get_x() + rect.get_width() / 2,
                value + (value_offset if value >= 0 else -value_offset),
                value_format.format(value), ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=value_fontsize, color="#303030", clip_on=False)

    ax.set_xlim(-0.65, len(plot_df) - 0.35)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel, fontsize=label_fontsize, labelpad=9)
    ax.set_xticks(xpos)
    ax.set_xticklabels(plot_df[measure_col], rotation=53, ha="right",
                       rotation_mode="anchor", fontsize=tick_fontsize)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.tick_params(axis="y", labelsize=tick_fontsize, length=2.5, width=0.6)
    ax.yaxis.grid(True, color="#777777", linewidth=0.45, alpha=0.14, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.7)
    ax.spines["left"].set_linewidth(0.7)

    # Group separators and category labels at y=0.90 (DATA units).
    index = 0
    for k, category in enumerate(category_order):
        count = int((plot_df[category_col] == category).sum())
        midpoint = index + (count - 1) / 2
        if k > 0:
            ax.axvline(index - 0.5, color=separator_color,
                       linestyle=separator_linestyle,
                       lw=separator_linewidth, zorder=1)
        display_category = {
            "Health Outcomes": "Health\nOutcomes",
            "Health Risk Behaviors": "Health Risk\nBehaviors",
            "Health Status": "Health\nStatus",
            "Prevention": "Prevention",
        }.get(str(category), str(category))
        ax.text(midpoint, category_y_fraction, display_category,
                transform=ax.transData, ha="center", va="center",
                fontsize=category_fontsize, color="#333333", clip_on=False,
                linespacing=1.0)
        index += count

    # Set box aspect AFTER labels. Aspect ratio = width / height.
    ax.set_box_aspect(1.0 / width_height_ratio)
    return ax, plot_df


def plot_figure2(
    df_pred: pd.DataFrame,
    df_comparison: pd.DataFrame,
    output_pdf: Optional[str | Path] = None,
    output_png: Optional[str | Path] = None,
    figsize=(16.8, 9.5),
    width_ratios=(1.0, 1.12),
    outer_wspace=0.18,
    left_margin=0.065, right_margin=0.985,
    bottom_margin=0.14, top_margin=0.91,
    prediction_n_rows=5, prediction_n_cols=5,
    prediction_wspace=0.34, prediction_hspace=0.42,
    prediction_point_size=1.15, prediction_point_alpha=0.32,
    prediction_identity_linewidth=0.85,
    prediction_title_fontsize=8.7,
    prediction_label_fontsize=10.5,
    prediction_tick_fontsize=6.7,
    prediction_legend_ncol=5,
    prediction_legend_fontsize=7.4,
    prediction_legend_marker_size=5.0,
    prediction_legend_columnspacing=0.85,
    prediction_legend_handletextpad=0.28,
    comparison_value_col="r2_complaint",
    comparison_measure_col="y_variable",
    comparison_category_col="legends",
    comparison_bar_width=0.57,
    comparison_color_complaint="#769BB9",
    comparison_bar_fill_alpha=0.26,
    comparison_bar_linewidth=1.2,
    comparison_value_fontsize=8,
    comparison_value_offset=0.015,
    comparison_value_format="{:.3f}",
    comparison_ylim=(0, 1),
    comparison_ylabel="Spatial out-of-fold R²",
    comparison_label_fontsize=10.5,
    comparison_category_fontsize=10.0,
    comparison_tick_fontsize=8.5,
    comparison_separator_color="#AAAAAA",
    comparison_separator_linewidth=0.8,
    comparison_separator_linestyle=(0, (3, 3)),
    comparison_category_y_fraction=0.90,
    comparison_width_height_ratio=1.15,
    panel_labels=("a", "b"),
    panel_label_fontsize=12.0,
    print_diagnostics=True,
):
    """Generate complete Figure 2; outputs match original function's return tuple.

    Important: This routine ONLY visualizes supplied values. It does not run
    cross-validation, refit a model, recalculate predictive metrics or trim outliers.
    """
    _validate_inputs(df_pred, df_comparison,
                     comparison_measure_col, comparison_category_col,
                     comparison_value_col)

    with plt.rc_context({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "pdf.fonttype": 42,
                         "svg.fonttype": "none",
                         "savefig.facecolor": "white"}):
        fig = plt.figure(figsize=figsize, facecolor="white")
        outer = fig.add_gridspec(1, 2, width_ratios=width_ratios,
                                 wspace=outer_wspace,
                                 left=left_margin, right=right_margin,
                                 bottom=bottom_margin, top=top_margin)

        axes_pred, legend_ax, city_colors, health_order = _plot_figure2a_predictions(
            fig, outer[0, 0], df_pred,
            n_rows=prediction_n_rows, n_cols=prediction_n_cols,
            wspace=prediction_wspace, hspace=prediction_hspace,
            point_size=prediction_point_size,
            point_alpha=prediction_point_alpha,
            identity_linewidth=prediction_identity_linewidth,
            title_fontsize=prediction_title_fontsize,
            tick_fontsize=prediction_tick_fontsize,
            label_fontsize=prediction_label_fontsize,
            legend_ncol=prediction_legend_ncol,
            legend_fontsize=prediction_legend_fontsize,
            legend_marker_size=prediction_legend_marker_size,
            legend_columnspacing=prediction_legend_columnspacing,
            legend_handletextpad=prediction_legend_handletextpad,
        )
        ax_comparison = fig.add_subplot(outer[0, 1])
        ax_comparison, df_sorted = _plot_figure2b_on_axis(
            ax_comparison, df_comparison,
            value_col=comparison_value_col,
            measure_col=comparison_measure_col,
            category_col=comparison_category_col,
            bar_width=comparison_bar_width,
            color_complaint=comparison_color_complaint,
            bar_fill_alpha=comparison_bar_fill_alpha,
            bar_linewidth=comparison_bar_linewidth,
            value_fontsize=comparison_value_fontsize,
            value_offset=comparison_value_offset,
            value_format=comparison_value_format,
            ylim=comparison_ylim,
            ylabel=comparison_ylabel,
            label_fontsize=comparison_label_fontsize,
            category_fontsize=comparison_category_fontsize,
            tick_fontsize=comparison_tick_fontsize,
            separator_color=comparison_separator_color,
            separator_linewidth=comparison_separator_linewidth,
            separator_linestyle=comparison_separator_linestyle,
            category_y_fraction=comparison_category_y_fraction,
            width_height_ratio=comparison_width_height_ratio,
        )

        if panel_labels is not None:
            if len(panel_labels) != 2:
                raise ValueError("panel_labels must be a pair, e.g. ('a', 'b').")
            for label, spec in zip(panel_labels, (outer[0, 0], outer[0, 1])):
                bb = spec.get_position(fig)
                fig.text(bb.x0 - 0.010, bb.y1 + 0.026,
                         f"({label})", fontsize=panel_label_fontsize,
                         fontweight="bold", va="bottom", ha="left")

        if print_diagnostics:
            print("Figure 2a order (A–Z):", health_order)
            print("Figure 2a layout:", prediction_n_rows, "×", prediction_n_cols)
            print("Figure 2b order: category first appearance, then descending R²")
            print(df_sorted[[comparison_category_col,
                             comparison_measure_col,
                             comparison_value_col]].to_string(index=False))
            print("City colors:", {k: tuple(np.round(v[:3], 2))
                                   if hasattr(v, '__len__') else v
                                   for k, v in city_colors.items()})

        for output_path, extension, kwargs in (
            (output_pdf, ".pdf", {"format": "pdf"}),
            (output_png, ".png", {"format": "png", "dpi": 600}),
        ):
            if output_path is None:
                continue
            p = Path(output_path)
            if p.suffix.lower() != extension:
                raise ValueError(f"Expected {extension} file, got {p.name}")
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(p, bbox_inches="tight", pad_inches=0.10,
                        facecolor="white", **kwargs)

    return (fig, axes_pred, legend_ax, ax_comparison,
            health_order, df_sorted)
