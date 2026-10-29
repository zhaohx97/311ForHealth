"""Figure 4 visualization: prior-health, annual-pair, reporting and city sensitivity.

Optimized presentation only; all statistical inputs are read from existing outputs.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from matplotlib.font_manager import fontManager


def plot_figure4_prior_and_temporal(
    df_prior,
    df_temporal,

    output_pdf=None,
    output_png=None,

    # ============================================================
    # Column names
    # ============================================================

    outcome_col=
        "y_variable",

    year_pair_col=
        "year_pair",

    delta_prior_col=
        "delta_r2_complaint_over_prior",

    delta_prior_acs_col=
        "delta_r2_complaint_over_prior_acs",

    acs_r2_col=
        "r2_acs_only",

    hybrid_r2_col=
        "r2_hybrid",

    # ============================================================
    # Temporal order
    # ============================================================

    temporal_pair_order=None,

    # ============================================================
    # Overall figure
    # ============================================================

    figsize=(
        18.5,
        8.5,
    ),

    outer_width_ratios=(
        1.0,
        1.25,
    ),

    outer_wspace=
        0.35,

    left_inner_wspace=
        0.72,

    left_height_fraction=
        0.95,

    # ============================================================
    # LEFT bars: shared visual style
    # ============================================================

    bar_color=
        "#8CAAC4",

    bar_fill_alpha=
        0.20,

    bar_linewidth=
        1.50,

    bar_height=
        0.66,

    # ============================================================
    # Bar value labels
    # ============================================================

    bar_value_fontsize=
        9.0,

    bar_value_decimals=
        3,

    bar_value_offset_fraction=
        0.018,

    bar_value_color=
        "#333333",

    # ============================================================
    # LEFT x-range control
    #
    # Make negative side visibly wider
    # ============================================================

    bar_left_min=
        -0.030,

    bar_left_padding_fraction=
        0.20,

    bar_right_padding_fraction=
        0.22,

    # ============================================================
    # LEFT plot 1
    # ============================================================

    bar1_xlim=None,

    bar1_title=
        "Prior health + Complaints\nvs Prior health",

    # ============================================================
    # LEFT plot 2
    # ============================================================

    bar2_xlim=None,

    bar2_title=
        "Prior health + ACS + Complaints\nvs Prior health + ACS",

    # ============================================================
    # LEFT common formatting
    # ============================================================

    bar_xlabel=
        "Incremental out-of-sample R² (ΔR²)",

    bar_title_fontsize=
        14,

    bar_label_fontsize=
        12,

    bar_tick_fontsize=
        10,

    bar_zero_linewidth=
        0.9,

    bar_zero_alpha=
        0.70,

    # ============================================================
    # RIGHT scatter
    # ============================================================

    scatter_xlim=(
        0.30,
        1.00,
    ),

    scatter_ylim=(
        0.30,
        1.00,
    ),

    scatter_s=
        80,

    scatter_alpha=
        0.90,

    scatter_edgecolor=
        "white",

    scatter_linewidth=
        1.25,

    scatter_title=
        "",

    scatter_xlabel=
        "ACS-only out-of-sample R²",

    scatter_ylabel=
        "Hybrid out-of-sample R²",

    scatter_title_fontsize=
        16,

    scatter_label_fontsize=
        14,

    scatter_tick_fontsize=
        12,

    parity_linewidth=
        0.9,

    parity_dashes=(
        6,
        3,
    ),

    # ============================================================
    # RIGHT legend
    # ============================================================

    legend_anchor=(
        0.965,
        0.075,
    ),

    legend_fontsize=
        9.0,

    legend_title_fontsize=
        9.5,

    legend_ncol=
        3,

    legend_marker_scale=
        0.90,

    # ============================================================
    # Common
    # ============================================================

    grid_linewidth=
        0.40,

    grid_alpha=
        0.30,

    figure_left=
        0.055,

    figure_right=
        0.985,

    figure_bottom=
        0.08,

    figure_top=
        0.94,
):
    """
    Final Figure 4.

    LEFT PLOT 1
    -----------
    ΔR² =
    R²(Prior health + Complaints)
    - R²(Prior health)

    LEFT PLOT 2
    -----------
    ΔR² =
    R²(Prior health + ACS + Complaints)
    - R²(Prior health + ACS)

    RIGHT
    -----
    x = ACS-only pooled spatial OOF R²
    y = Hybrid pooled spatial OOF R²
    """

    # ============================================================
    # Global style
    # ============================================================

    # Keep the original white background without globally changing
    # sizing or colors through a theme preset.
    plt.style.use("default")

    # Match the original Arial appearance where installed; fall back
    # cleanly on systems without Arial (e.g., headless Linux QA).
    _available_fonts = {f.name for f in fontManager.ttflist}
    _figure_font = "Arial" if "Arial" in _available_fonts else "DejaVu Sans"
    plt.rcParams["font.family"] = _figure_font

    prior = (
        df_prior.copy()
    )

    temporal = (
        df_temporal.copy()
    )

    # ============================================================
    # Validate columns
    # ============================================================

    required_prior = {
        outcome_col,
        delta_prior_col,
        delta_prior_acs_col,
    }

    missing_prior = (
        required_prior
        -
        set(
            prior.columns
        )
    )

    if missing_prior:

        raise ValueError(
            "Missing prior-sensitivity columns: "
            f"{sorted(missing_prior)}"
        )


    required_temporal = {
        outcome_col,
        year_pair_col,
        acs_r2_col,
        hybrid_r2_col,
    }

    missing_temporal = (
        required_temporal
        -
        set(
            temporal.columns
        )
    )

    if missing_temporal:

        raise ValueError(
            "Missing temporal-robustness columns: "
            f"{sorted(missing_temporal)}"
        )

    # ============================================================
    # Validate left height
    # ============================================================

    if not (
        0
        <
        left_height_fraction
        <
        1
    ):

        raise ValueError(
            "left_height_fraction must be within (0, 1)."
        )

    # ============================================================
    # Helper:
    # signed bar-value labels
    #
    # NEW:
    # abs(value) < 0.001:
    #     +0.000
    #     -0.000
    # ============================================================

    def _format_signed_bar_value(
        value,
        decimals,
    ):

        if abs(
            value
        ) < 0.001:

            if value < 0:

                return (
                    "-0.000"
                )

            else:

                return (
                    "+0.000"
                )

        return (
            f"{value:+.{decimals}f}"
        )

    # ============================================================
    # Temporal pair order
    # ============================================================

    if temporal_pair_order is None:

        temporal_pair_order = (
            temporal[
                year_pair_col
            ]
            .drop_duplicates()
            .tolist()
        )

    # ============================================================
    # Health outcome order
    # ============================================================

    health_order = (
        prior[
            outcome_col
        ]
        .drop_duplicates()
        .tolist()
    )

    temporal_health = set(
        temporal[
            outcome_col
        ]
        .dropna()
        .unique()
    )

    prior_health = set(
        health_order
    )

    if temporal_health != prior_health:

        missing_in_temporal = (
            prior_health
            -
            temporal_health
        )

        extra_in_temporal = (
            temporal_health
            -
            prior_health
        )

        raise ValueError(
            "Health outcomes differ between prior and temporal "
            f"results. Missing in temporal: "
            f"{sorted(missing_in_temporal)}; "
            f"extra in temporal: "
            f"{sorted(extra_in_temporal)}"
        )

    # ============================================================
    # Shared bar appearance
    # ============================================================

    bar_facecolor = to_rgba(
        bar_color,
        bar_fill_alpha,
    )

    bar_edgecolor = (
        bar_color
    )

    # ============================================================
    # Health colors
    # ============================================================

    palette = list(
        sns.color_palette(
            "tab20",
            20,
        )
    )

    palette.append(
        "#FF6347"
    )

    if len(
        health_order
    ) > len(
        palette
    ):

        raise ValueError(
            f"{len(health_order)} health outcomes found, "
            f"but only {len(palette)} colors are available."
        )

    health_palette = {
        outcome:
            palette[i]

        for i, outcome
        in enumerate(
            health_order
        )
    }

    # ============================================================
    # Year-pair marker styles
    # ============================================================

    marker_cycle = [
        "o",
        "X",
        "s",
        "P",
        "^",
        "D",
    ]

    if (
        len(
            temporal_pair_order
        )
        >
        len(
            marker_cycle
        )
    ):

        raise ValueError(
            "Too many temporal year pairs for marker set."
        )

    year_markers = {
        pair:
            marker_cycle[i]

        for i, pair
        in enumerate(
            temporal_pair_order
        )
    }

    # ============================================================
    # Figure
    # ============================================================

    fig = plt.figure(
        figsize=
            figsize
    )

    outer_grid = fig.add_gridspec(
        nrows=
            1,

        ncols=
            2,

        width_ratios=
            outer_width_ratios,

        wspace=
            outer_wspace,
    )

    # ============================================================
    # LEFT major region
    # ============================================================

    vertical_margin = (
        1.0
        -
        left_height_fraction
    ) / 2.0

    left_vertical_grid = (
        outer_grid[
            0,
            0,
        ]
        .subgridspec(
            nrows=
                3,

            ncols=
                1,

            height_ratios=[
                vertical_margin,
                left_height_fraction,
                vertical_margin,
            ],

            hspace=
                0.0,
        )
    )

    # ============================================================
    # Two independent left plots
    # ============================================================

    left_plot_grid = (
        left_vertical_grid[
            1,
            0,
        ]
        .subgridspec(
            nrows=
                1,

            ncols=
                2,

            width_ratios=[
                1,
                1,
            ],

            wspace=
                left_inner_wspace,
        )
    )

    ax_bar1 = fig.add_subplot(
        left_plot_grid[
            0,
            0,
        ]
    )

    ax_bar2 = fig.add_subplot(
        left_plot_grid[
            0,
            1,
        ]
    )

    # ============================================================
    # RIGHT
    # ============================================================

    ax_scatter = fig.add_subplot(
        outer_grid[
            0,
            1,
        ]
    )

    # ============================================================
    # LEFT PLOT 1
    # ============================================================

    bar1_data = (
        prior[
            [
                outcome_col,
                delta_prior_col,
            ]
        ]
        .dropna()
        .sort_values(
            delta_prior_col,
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    y1 = np.arange(
        len(
            bar1_data
        )
    )

    bar1_values = (
        bar1_data[
            delta_prior_col
        ]
        .to_numpy(
            dtype=float
        )
    )

    ax_bar1.barh(
        y1,
        bar1_values,

        height=
            bar_height,

        facecolor=
            bar_facecolor,

        edgecolor=
            bar_edgecolor,

        linewidth=
            bar_linewidth,

        zorder=
            2,
    )

    ax_bar1.set_yticks(
        y1
    )

    ax_bar1.set_yticklabels(
        bar1_data[
            outcome_col
        ],

        fontsize=
            bar_tick_fontsize,
    )

    ax_bar1.invert_yaxis()

    # ------------------------------------------------------------
    # Zero line
    # ------------------------------------------------------------

    ax_bar1.axvline(
        0,

        color=
            "black",

        linestyle=
            "--",

        linewidth=
            bar_zero_linewidth,

        alpha=
            bar_zero_alpha,

        zorder=
            1,
    )

    # ------------------------------------------------------------
    # Expanded automatic x range
    # ------------------------------------------------------------

    bar1_vmin = np.nanmin(
        bar1_values
    )

    bar1_vmax = np.nanmax(
        bar1_values
    )

    bar1_span = (
        bar1_vmax
        -
        bar1_vmin
    )

    if bar1_span <= 0:

        bar1_span = max(
            abs(
                bar1_vmax
            ),
            0.1,
        )

    if bar1_xlim is None:

        bar1_xlim = (
            min(
                bar_left_min,

                bar1_vmin
                -
                bar_left_padding_fraction
                *
                bar1_span,
            ),

            bar1_vmax
            +
            bar_right_padding_fraction
            *
            bar1_span,
        )

    ax_bar1.set_xlim(
        bar1_xlim
    )

    # ------------------------------------------------------------
    # Bar-end signed labels
    # ------------------------------------------------------------

    bar1_text_offset = (
        bar_value_offset_fraction
        *
        (
            bar1_xlim[1]
            -
            bar1_xlim[0]
        )
    )

    for y_pos, value in zip(
        y1,
        bar1_values,
    ):

        if value >= 0:

            x_text = (
                value
                +
                bar1_text_offset
            )

            horizontal_alignment = (
                "left"
            )

        else:

            x_text = (
                value
                -
                bar1_text_offset
            )

            horizontal_alignment = (
                "right"
            )

        ax_bar1.text(
            x_text,
            y_pos,

            _format_signed_bar_value(
                value,
                bar_value_decimals,
            ),

            ha=
                horizontal_alignment,

            va=
                "center",

            fontsize=
                bar_value_fontsize,

            color=
                bar_value_color,

            clip_on=
                False,

            zorder=
                4,
        )

    # ------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------

    ax_bar1.set_xlabel(
        bar_xlabel,

        fontsize=
            bar_label_fontsize,
    )

    ax_bar1.set_title(
        bar1_title,

        fontsize=
            bar_title_fontsize,

        y=
            1.02,
    )

    ax_bar1.tick_params(
        axis=
            "x",

        labelsize=
            bar_tick_fontsize,
    )

    ax_bar1.tick_params(
        axis=
            "y",

        labelsize=
            bar_tick_fontsize,
    )

    ax_bar1.grid(
        axis=
            "x",

        linewidth=
            grid_linewidth,

        alpha=
            grid_alpha,

        zorder=
            0,
    )

    ax_bar1.spines[
        "top"
    ].set_visible(
        False
    )

    ax_bar1.spines[
        "right"
    ].set_visible(
        False
    )

    # ============================================================
    # LEFT PLOT 2
    # ============================================================

    bar2_data = (
        prior[
            [
                outcome_col,
                delta_prior_acs_col,
            ]
        ]
        .dropna()
        .sort_values(
            delta_prior_acs_col,
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    y2 = np.arange(
        len(
            bar2_data
        )
    )

    bar2_values = (
        bar2_data[
            delta_prior_acs_col
        ]
        .to_numpy(
            dtype=float
        )
    )

    ax_bar2.barh(
        y2,
        bar2_values,

        height=
            bar_height,

        facecolor=
            bar_facecolor,

        edgecolor=
            bar_edgecolor,

        linewidth=
            bar_linewidth,

        zorder=
            2,
    )

    ax_bar2.set_yticks(
        y2
    )

    ax_bar2.set_yticklabels(
        bar2_data[
            outcome_col
        ],

        fontsize=
            bar_tick_fontsize,
    )

    ax_bar2.invert_yaxis()

    # ------------------------------------------------------------
    # Zero line
    # ------------------------------------------------------------

    ax_bar2.axvline(
        0,

        color=
            "black",

        linestyle=
            "--",

        linewidth=
            bar_zero_linewidth,

        alpha=
            bar_zero_alpha,

        zorder=
            1,
    )

    # ------------------------------------------------------------
    # Expanded automatic x range
    # ------------------------------------------------------------

    bar2_vmin = np.nanmin(
        bar2_values
    )

    bar2_vmax = np.nanmax(
        bar2_values
    )

    bar2_span = (
        bar2_vmax
        -
        bar2_vmin
    )

    if bar2_span <= 0:

        bar2_span = max(
            abs(
                bar2_vmax
            ),
            0.1,
        )

    if bar2_xlim is None:

        bar2_xlim = (
            min(
                bar_left_min,

                bar2_vmin
                -
                bar_left_padding_fraction
                *
                bar2_span,
            ),

            bar2_vmax
            +
            bar_right_padding_fraction
            *
            bar2_span,
        )

    ax_bar2.set_xlim(
        bar2_xlim
    )

    # ------------------------------------------------------------
    # Bar-end signed labels
    # ------------------------------------------------------------

    bar2_text_offset = (
        bar_value_offset_fraction
        *
        (
            bar2_xlim[1]
            -
            bar2_xlim[0]
        )
    )

    for y_pos, value in zip(
        y2,
        bar2_values,
    ):

        if value >= 0:

            x_text = (
                value
                +
                bar2_text_offset
            )

            horizontal_alignment = (
                "left"
            )

        else:

            x_text = (
                value
                -
                bar2_text_offset
            )

            horizontal_alignment = (
                "right"
            )

        ax_bar2.text(
            x_text,
            y_pos,

            _format_signed_bar_value(
                value,
                bar_value_decimals,
            ),

            ha=
                horizontal_alignment,

            va=
                "center",

            fontsize=
                bar_value_fontsize,

            color=
                bar_value_color,

            clip_on=
                False,

            zorder=
                4,
        )

    # ------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------

    ax_bar2.set_xlabel(
        bar_xlabel,

        fontsize=
            bar_label_fontsize,
    )

    ax_bar2.set_title(
        bar2_title,

        fontsize=
            bar_title_fontsize,

        y=
            1.02,
    )

    ax_bar2.tick_params(
        axis=
            "x",

        labelsize=
            bar_tick_fontsize,
    )

    ax_bar2.tick_params(
        axis=
            "y",

        labelsize=
            bar_tick_fontsize,
    )

    ax_bar2.grid(
        axis=
            "x",

        linewidth=
            grid_linewidth,

        alpha=
            grid_alpha,

        zorder=
            0,
    )

    ax_bar2.spines[
        "top"
    ].set_visible(
        False
    )

    ax_bar2.spines[
        "right"
    ].set_visible(
        False
    )

    # ============================================================
    # RIGHT PANEL
    # ============================================================

    temporal_plot = (
        temporal.rename(
            columns={
                outcome_col:
                    "Health Measure",

                year_pair_col:
                    "Year Pair",
            }
        )
        .copy()
    )

    # ============================================================
    # Scatter
    # ============================================================

    sns.scatterplot(
        data=
            temporal_plot,

        x=
            acs_r2_col,

        y=
            hybrid_r2_col,

        hue=
            "Health Measure",

        style=
            "Year Pair",

        hue_order=
            health_order,

        style_order=
            temporal_pair_order,

        palette=
            health_palette,

        markers=
            year_markers,

        s=
            scatter_s,

        alpha=
            scatter_alpha,

        edgecolor=
            scatter_edgecolor,

        linewidth=
            scatter_linewidth,

        ax=
            ax_scatter,

        legend="full",

        zorder=
            3,
    )

    # ============================================================
    # Fixed scatter limits
    # ============================================================

    if not np.allclose(
        np.asarray(
            scatter_xlim,
            dtype=float,
        ),

        np.asarray(
            scatter_ylim,
            dtype=float,
        ),
    ):

        raise ValueError(
            "scatter_xlim and scatter_ylim must be identical "
            "for a true 1:1 comparison."
        )

    ax_scatter.set_xlim(
        scatter_xlim
    )

    ax_scatter.set_ylim(
        scatter_ylim
    )

    # ============================================================
    # 45-degree parity line
    # ============================================================

    parity_line = (
        ax_scatter.plot(
            [
                scatter_xlim[0],
                scatter_xlim[1],
            ],

            [
                scatter_xlim[0],
                scatter_xlim[1],
            ],

            color=
                "black",

            linewidth=
                parity_linewidth,

            zorder=
                1,
        )[0]
    )

    parity_line.set_dashes(
        parity_dashes
    )

    # ============================================================
    # Strict 1:1 geometry
    # ============================================================

    ax_scatter.set_aspect(
        "equal",

        adjustable=
            "box",
    )

    # ============================================================
    # Right formatting
    # ============================================================

    ax_scatter.set_xlabel(
        scatter_xlabel,

        fontsize=
            scatter_label_fontsize,
    )

    ax_scatter.set_ylabel(
        scatter_ylabel,

        fontsize=
            scatter_label_fontsize,
    )

    ax_scatter.set_title(
        scatter_title,

        fontsize=
            scatter_title_fontsize,

        y=
            1.02,
    )

    ax_scatter.tick_params(
        axis=
            "both",

        labelsize=
            scatter_tick_fontsize,
    )

    ax_scatter.grid(
        linewidth=
            grid_linewidth,

        alpha=
            grid_alpha,

        zorder=
            0,
    )

    ax_scatter.spines[
        "top"
    ].set_visible(
        False
    )

    ax_scatter.spines[
        "right"
    ].set_visible(
        False
    )

    # ============================================================
    # Legend
    # ============================================================

    handles, labels = (
        ax_scatter.get_legend_handles_labels()
    )

    existing_legend = (
        ax_scatter.get_legend()
    )

    if existing_legend is not None:

        existing_legend.remove()

    legend = (
        ax_scatter.legend(
            handles=
                handles,

            labels=
                labels,

            loc=
                "lower right",

            bbox_to_anchor=
                legend_anchor,

            frameon=
                False,

            fontsize=
                legend_fontsize,

            ncol=
                legend_ncol,

            markerscale=
                legend_marker_scale,

            borderaxespad=
                0.0,

            handletextpad=
                0.42,

            labelspacing=
                0.28,

            columnspacing=
                0.75,
        )
    )

    for text in (
        legend.get_texts()
    ):

        if text.get_text() in {
            "Health Measure",
            "Year Pair",
        }:

            text.set_fontsize(
                legend_title_fontsize
            )

            text.set_fontweight(
                "normal"
            )

    # ============================================================
    # Final figure layout
    # ============================================================

    fig.subplots_adjust(
        left=
            figure_left,

        right=
            figure_right,

        bottom=
            figure_bottom,

        top=
            figure_top,

        wspace=
            outer_wspace,
    )

    # ============================================================
    # Save
    # ============================================================

    if output_pdf is not None:

        output_pdf = Path(
            output_pdf
        )

        output_pdf.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            output_pdf,

            format=
                "pdf",

            bbox_inches=
                "tight",
        )

    if output_png is not None:
        output_png = Path(output_png)
        output_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_png, dpi=600, bbox_inches="tight")

    return (
        fig,
        {
            "prior":
                ax_bar1,

            "prior_acs":
                ax_bar2,

            "temporal":
                ax_scatter,
        },
    )


def _validate_figure4_visualization_inputs(
    df_reporting,
    df_city_r2,
    df_city_summary,
    df_outliers,
):
    """
    Validate standardized Figure 4 analytical outputs.
    """

    required_reporting = {
        "Health Measure",
        "change_r2_hybrid_after_reporting_adjustment",
    }

    required_city_r2 = {
        "City",
        "Health Measure",
        "Model",
        "R²",
    }

    required_summary = {
        "City",
        "Model",
        "whisker_low",
        "whisker_high",
    }

    required_outliers = {
        "City",
        "Model",
        "outlier_direction",
    }


    missing = (
        required_reporting
        -
        set(
            df_reporting.columns
        )
    )

    if missing:

        raise KeyError(
            "df_reporting is missing columns: "
            f"{sorted(missing)}"
        )


    missing = (
        required_city_r2
        -
        set(
            df_city_r2.columns
        )
    )

    if missing:

        raise KeyError(
            "df_city_r2 is missing columns: "
            f"{sorted(missing)}"
        )


    missing = (
        required_summary
        -
        set(
            df_city_summary.columns
        )
    )

    if missing:

        raise KeyError(
            "df_city_summary is missing columns: "
            f"{sorted(missing)}"
        )


    missing = (
        required_outliers
        -
        set(
            df_outliers.columns
        )
    )

    if missing:

        raise KeyError(
            "df_outliers is missing columns: "
            f"{sorted(missing)}"
        )


    models = set(
        df_city_r2[
            "Model"
        ]
        .dropna()
        .unique()
    )

    expected_models = {
        "ACS",
        "Hybrid",
    }

    if not expected_models.issubset(
        models
    ):

        raise ValueError(
            "df_city_r2 must contain both "
            "'ACS' and 'Hybrid'. "
            f"Found: {sorted(models)}"
        )


def _plot_figure4_reporting_panel(
    ax,
    df_reporting,

    # ------------------------------------------------------------
    # X-axis
    # ------------------------------------------------------------

    xlim=None,

    x_padding_fraction=0.16,

    minimum_padding=0.0012,

    positive_padding_multiplier=1.25,

    # ------------------------------------------------------------
    # Bars
    # ------------------------------------------------------------

    bar_color="#8CAAC4",

    bar_fill_alpha=0.20,

    bar_linewidth=1.50,

    bar_height=0.65,

    # ------------------------------------------------------------
    # Value labels
    # ------------------------------------------------------------

    show_values=True,

    value_decimals=3,

    value_fontsize=9.0,

    value_offset_fraction=0.018,

    value_color="#333333",

    # ------------------------------------------------------------
    # Text
    # ------------------------------------------------------------

    title=(
        ""
    ),

    xlabel=(
        "Δ spatial OOF R² (alternative − original)"
    ),

    title_fontsize=13,

    label_fontsize=11,

    tick_fontsize=9,

    # ------------------------------------------------------------
    # Reference / grid
    # ------------------------------------------------------------

    zero_linewidth=0.90,

    zero_alpha=0.70,

    grid_linewidth=0.35,

    grid_alpha=0.22,
):
    """
    Plot the pooled Hybrid reporting-adjustment result.
    """

    d = (
        df_reporting[
            [
                "Health Measure",
                "change_r2_hybrid_after_reporting_adjustment",
            ]
        ]
        .copy()
        .sort_values(
            "change_r2_hybrid_after_reporting_adjustment",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


    values = (
        d[
            "change_r2_hybrid_after_reporting_adjustment"
        ]
        .to_numpy(
            dtype=float
        )
    )


    # ============================================================
    # Compact asymmetric x-range
    # ============================================================

    if xlim is None:

        data_min = min(
            np.nanmin(
                values
            ),
            0.0,
        )

        data_max = max(
            np.nanmax(
                values
            ),
            0.0,
        )


        data_span = (
            data_max
            -
            data_min
        )

        if data_span <= 0:

            data_span = 0.01


        padding = max(
            minimum_padding,

            x_padding_fraction
            *
            data_span,
        )


        xlim = (
            data_min
            -
            padding,

            data_max
            +
            positive_padding_multiplier
            *
            padding,
        )


    # ============================================================
    # Bars
    # ============================================================

    y = np.arange(
        len(
            d
        )
    )


    ax.barh(
        y,
        values,

        height=bar_height,

        facecolor=to_rgba(
            bar_color,
            bar_fill_alpha,
        ),

        edgecolor=bar_color,

        linewidth=bar_linewidth,

        zorder=3,
    )


    ax.set_yticks(
        y
    )

    ax.set_yticklabels(
        d[
            "Health Measure"
        ],

        fontsize=tick_fontsize,
    )

    ax.invert_yaxis()


    # ============================================================
    # Value annotations
    # ============================================================

    if show_values:

        axis_span = (
            xlim[
                1
            ]
            -
            xlim[
                0
            ]
        )

        offset = (
            axis_span
            *
            value_offset_fraction
        )


        for (
            yi,
            value
        ) in zip(
            y,
            values,
        ):

            if value >= 0:

                x_text = (
                    value
                    +
                    offset
                )

                ha = "left"

            else:

                x_text = (
                    value
                    -
                    offset
                )

                ha = "right"


            ax.text(
                x_text,
                yi,

                f"{value:+.{value_decimals}f}",

                va="center",

                ha=ha,

                fontsize=value_fontsize,

                color=value_color,

                zorder=5,
            )


    # ============================================================
    # Zero reference
    # ============================================================

    ax.axvline(
        0,

        linestyle="--",

        color="#555555",

        linewidth=zero_linewidth,

        alpha=zero_alpha,

        zorder=2,
    )


    # ============================================================
    # Formatting
    # ============================================================

    ax.set_xlim(
        xlim
    )


    ax.set_title(
        title,

        fontsize=title_fontsize,

        pad=14,
    )


    ax.set_xlabel(
        xlabel,

        fontsize=label_fontsize,
    )


    ax.tick_params(
        axis="x",

        labelsize=tick_fontsize,
    )


    ax.tick_params(
        axis="y",

        labelsize=tick_fontsize,

        length=0,
    )


    ax.grid(
        axis="x",

        linewidth=grid_linewidth,

        alpha=grid_alpha,

        zorder=0,
    )


    ax.spines[
        "top"
    ].set_visible(
        False
    )

    ax.spines[
        "right"
    ].set_visible(
        False
    )

    ax.spines[
        "left"
    ].set_linewidth(
        0.90
    )

    ax.spines[
        "bottom"
    ].set_linewidth(
        0.90
    )


    return ax


def _figure4_city_ylim(
    city,
    df_summary,

    lower_margin_fraction=0.13,

    upper_margin_fraction=0.11,

    minimum_span=0.20,
):
    """
    Determine each city's display range using Tukey whiskers
    rather than extreme outliers.
    """

    d = (
        df_summary.loc[
            df_summary[
                "City"
            ]
            ==
            city
        ]
    )


    if len(
        d
    ) == 0:

        raise ValueError(
            f"No summary rows found for city: {city}"
        )


    whisker_low = (
        d[
            "whisker_low"
        ]
        .min()
    )

    whisker_high = (
        d[
            "whisker_high"
        ]
        .max()
    )


    span = (
        whisker_high
        -
        whisker_low
    )


    if span < minimum_span:

        span = minimum_span


    y_min = (
        whisker_low
        -
        lower_margin_fraction
        *
        span
    )


    y_max = (
        whisker_high
        +
        upper_margin_fraction
        *
        span
    )


    # R² ≤ 1
    y_max = min(
        1.0,
        y_max,
    )


    return (
        y_min,
        y_max,
    )


def _figure4_outlier_lookup(
    cities,
    models,
    df_outliers,
):
    """
    Return:

    lookup[city][model]["lower"]
    lookup[city][model]["upper"]
    """

    lookup = {}


    for city in cities:

        lookup[
            city
        ] = {}


        for model in models:

            d = (
                df_outliers.loc[
                    (
                        df_outliers[
                            "City"
                        ]
                        ==
                        city
                    )
                    &
                    (
                        df_outliers[
                            "Model"
                        ]
                        ==
                        model
                    )
                ]
            )


            lookup[
                city
            ][
                model
            ] = {

                "lower":
                    (
                        d[
                            "outlier_direction"
                        ]
                        ==
                        "lower"
                    )
                    .any(),

                "upper":
                    (
                        d[
                            "outlier_direction"
                        ]
                        ==
                        "upper"
                    )
                    .any(),
            }


    return lookup


def _figure4_panel_rectangles(
    fig,

    left_margin=0.050,

    right_margin=0.020,

    bottom_margin=0.135,

    top_margin=0.045,

    horizontal_gap=0.060,

    width_ratio_left=1.0,

    width_ratio_right=2.0,

    left_width_height_ratio=3 / 4,

    right_width_height_ratio=4 / 3,
):
    """
    Compute physical rectangles while preserving:

        LEFT : RIGHT width = 1 : 2

        LEFT width : height = 3 : 4

        RIGHT width : height = 4 : 3
    """


    fig_w, fig_h = (
        fig.get_size_inches()
    )


    available_width = (
        1
        -
        left_margin
        -
        right_margin
        -
        horizontal_gap
    )


    available_height = (
        1
        -
        bottom_margin
        -
        top_margin
    )


    ratio_sum = (
        width_ratio_left
        +
        width_ratio_right
    )


    left_w = (
        available_width
        *
        width_ratio_left
        /
        ratio_sum
    )


    right_w = (
        available_width
        *
        width_ratio_right
        /
        ratio_sum
    )


    left_h = (
        left_w
        *
        fig_w
        /
        left_width_height_ratio
        /
        fig_h
    )


    right_h = (
        right_w
        *
        fig_w
        /
        right_width_height_ratio
        /
        fig_h
    )


    max_h = max(
        left_h,
        right_h,
    )


    if max_h > available_height:

        scale = (
            available_height
            /
            max_h
        )

        left_w *= scale
        right_w *= scale
        horizontal_gap *= scale

        left_h *= scale
        right_h *= scale


    used_width = (
        left_w
        +
        horizontal_gap
        +
        right_w
    )


    x_start = (
        left_margin
        +
        (
            available_width
            -
            used_width
        )
        /
        2
    )


    left_y = (
        bottom_margin
        +
        (
            available_height
            -
            left_h
        )
        /
        2
    )


    right_y = (
        bottom_margin
        +
        (
            available_height
            -
            right_h
        )
        /
        2
    )


    left_rect = [
        x_start,
        left_y,
        left_w,
        left_h,
    ]


    right_rect = [
        x_start
        +
        left_w
        +
        horizontal_gap,

        right_y,
        right_w,
        right_h,
    ]


    return (
        left_rect,
        right_rect,
    )


def _figure4_create_city_axes(
    fig,
    right_rect,

    nrows=2,

    ncols=5,

    column_gap_fraction=0.115,

    row_gap_fraction=0.155,
):
    """
    Create 2 × 5 independently positioned axes inside right panel.
    """

    x0, y0, width, height = (
        right_rect
    )


    total_col_gap = (
        column_gap_fraction
        *
        width
    )


    total_row_gap = (
        row_gap_fraction
        *
        height
    )


    col_gap = (
        total_col_gap
        /
        (
            ncols
            -
            1
        )
    )


    row_gap = (
        total_row_gap
        /
        (
            nrows
            -
            1
        )
    )


    ax_width = (
        width
        -
        total_col_gap
    ) / ncols


    ax_height = (
        height
        -
        total_row_gap
    ) / nrows


    axes = []


    for row in range(
        nrows
    ):

        for col in range(
            ncols
        ):

            x = (
                x0
                +
                col
                *
                (
                    ax_width
                    +
                    col_gap
                )
            )


            y = (
                y0
                +
                (
                    nrows
                    -
                    1
                    -
                    row
                )
                *
                (
                    ax_height
                    +
                    row_gap
                )
            )


            axes.append(
                fig.add_axes(
                    [
                        x,
                        y,
                        ax_width,
                        ax_height,
                    ]
                )
            )


    return axes


def _plot_figure4_city_panel(
    fig,
    right_rect,

    df_city_r2,
    df_summary,
    df_outliers,

    # ------------------------------------------------------------
    # Cities
    # ------------------------------------------------------------

    city_order=None,

    # ------------------------------------------------------------
    # Models
    # ------------------------------------------------------------

    model_order=(
        "ACS",
        "Hybrid",
    ),

    model_positions=None,

    # ------------------------------------------------------------
    # Colors
    # ------------------------------------------------------------

    color_dict=None,

    indicator_color=None,

    # ------------------------------------------------------------
    # Close ACS / Hybrid positions
    # ------------------------------------------------------------

    subplot_xlim=(
        0.20,
        0.80,
    ),

    box_width=0.10,

    # ------------------------------------------------------------
    # Box style
    # ------------------------------------------------------------

    box_linewidth=0.90,

    whisker_linewidth=0.55,

    cap_linewidth=0.90,

    median_linewidth=0.65,

    # ------------------------------------------------------------
    # Outlier indicators
    # ------------------------------------------------------------

    lower_triangle_position=0.035,

    upper_triangle_position=0.035,

    triangle_size=32,

    # ------------------------------------------------------------
    # Y range
    # ------------------------------------------------------------

    lower_margin_fraction=0.13,

    upper_margin_fraction=0.11,

    minimum_span=0.20,

    # Shared range is calculated from the city Tukey whiskers;
    # pass None to restore independent city-specific axes.
    shared_ylim=None,

    # ------------------------------------------------------------
    # Grid geometry
    # ------------------------------------------------------------

    column_gap_fraction=0.115,

    row_gap_fraction=0.155,

    # ------------------------------------------------------------
    # Typography
    # ------------------------------------------------------------

    city_title_fontsize=10.5,

    ylabel_fontsize=7.7,

    xtick_fontsize=7.4,

    ytick_fontsize=7.0,

    # ------------------------------------------------------------
    # Grid
    # ------------------------------------------------------------

    grid_linewidth=0.28,

    grid_alpha=0.22,

    # ------------------------------------------------------------
    # Legend
    # ------------------------------------------------------------

    legend_vertical_gap=0.050,

    legend_fontsize=7.7,
):
    """
    Draw the city-stratified ACS vs Hybrid panel.
    """


    if model_positions is None:

        model_positions = {
            "ACS":
                0.40,

            "Hybrid":
                0.60,
        }


    if color_dict is None:

        color_dict = {
            "ACS":
                "#DDDDDD",

            "Hybrid":
                "#8CAAC4",
        }


    if indicator_color is None:

        indicator_color = {
            "ACS":
                "#777777",

            "Hybrid":
                "#668EAE",
        }


    # ============================================================
    # Alphabetical city order by default
    # ============================================================

    if city_order is None:

        city_order = sorted(
            df_city_r2[
                "City"
            ]
            .dropna()
            .unique()
            .tolist()
        )


    if len(
        city_order
    ) > 10:

        raise ValueError(
            "Figure 4 right panel currently supports "
            "at most 10 cities."
        )


    # ============================================================
    # Outlier lookup
    # ============================================================

    outlier_lookup = (
        _figure4_outlier_lookup(
            cities=
                city_order,

            models=
                model_order,

            df_outliers=
                df_outliers,
        )
    )


    # ============================================================
    # Create axes
    # ============================================================

    axes = (
        _figure4_create_city_axes(
            fig=
                fig,

            right_rect=
                right_rect,

            nrows=
                2,

            ncols=
                5,

            column_gap_fraction=
                column_gap_fraction,

            row_gap_fraction=
                row_gap_fraction,
        )
    )


    # Validate a provided shared range; all 10 panels use the same bounds.
    if shared_ylim is not None:
        if not (len(shared_ylim) == 2 and shared_ylim[0] < shared_ylim[1]):
            raise ValueError("shared_ylim must be (lower, upper).")

    # ============================================================
    # Draw cities
    # ============================================================

    for i, city in enumerate(
        city_order
    ):

        ax = (
            axes[
                i
            ]
        )


        city_data = (
            df_city_r2.loc[
                df_city_r2[
                    "City"
                ]
                ==
                city
            ]
        )


        # --------------------------------------------------------
        # Independent city-specific y-range
        # --------------------------------------------------------

        y_min, y_max = (
            _figure4_city_ylim(
                city=
                    city,

                df_summary=
                    df_summary,

                lower_margin_fraction=
                    lower_margin_fraction,

                upper_margin_fraction=
                    upper_margin_fraction,

                minimum_span=
                    minimum_span,
            )
        )


        if shared_ylim is not None:
            y_min, y_max = shared_ylim

        y_span = (
            y_max
            -
            y_min
        )


        # ========================================================
        # Boxplots
        #
        # Explicit positions make ACS and Hybrid visually close.
        # ========================================================

        for model in model_order:

            values = (
                city_data.loc[
                    city_data[
                        "Model"
                    ]
                    ==
                    model,
                    "R²",
                ]
                .dropna()
                .to_numpy(
                    dtype=float
                )
            )


            if len(
                values
            ) == 0:

                continue


            ax.boxplot(
                [
                    values
                ],

                positions=[
                    model_positions[
                        model
                    ]
                ],

                widths=
                    box_width,

                patch_artist=
                    True,

                whis=
                    1.5,

                showfliers=
                    False,

                manage_ticks=
                    False,

                boxprops={
                    "linewidth":
                        box_linewidth,

                    "edgecolor":
                        "black",

                    "facecolor":
                        color_dict[
                            model
                        ],
                },

                whiskerprops={
                    "linewidth":
                        whisker_linewidth,

                    "color":
                        "black",
                },

                capprops={
                    "linewidth":
                        cap_linewidth,

                    "color":
                        "black",
                },

                medianprops={
                    "linewidth":
                        median_linewidth,

                    "color":
                        "black",
                },
            )


        # ========================================================
        # Lower outlier indicators
        # ========================================================

        lower_y = (
            y_min
            +
            lower_triangle_position
            *
            y_span
        )


        for model in model_order:

            if (
                outlier_lookup[
                    city
                ][
                    model
                ][
                    "lower"
                ]
            ):

                ax.scatter(
                    model_positions[
                        model
                    ],
                    lower_y,

                    marker="v",

                    s=
                        triangle_size,

                    facecolor=
                        "white",

                    edgecolor=
                        indicator_color[
                            model
                        ],

                    linewidth=
                        0.95,

                    zorder=
                        6,
                )


        # ========================================================
        # Upper outlier indicators
        # ========================================================

        upper_y = (
            y_max
            -
            upper_triangle_position
            *
            y_span
        )


        for model in model_order:

            if (
                outlier_lookup[
                    city
                ][
                    model
                ][
                    "upper"
                ]
            ):

                ax.scatter(
                    model_positions[
                        model
                    ],
                    upper_y,

                    marker="^",

                    s=
                        triangle_size,

                    facecolor=
                        "white",

                    edgecolor=
                        indicator_color[
                            model
                        ],

                    linewidth=
                        0.95,

                    zorder=
                        6,
                )


        # ========================================================
        # Title
        # ========================================================

        ax.set_title(
            city,

            fontsize=
                city_title_fontsize,

            pad=
                6,
        )


        # ========================================================
        # X axis
        # ========================================================

        ax.set_xlim(
            subplot_xlim
        )


        ax.set_xticks(
            [
                model_positions[
                    "ACS"
                ],

                model_positions[
                    "Hybrid"
                ],
            ]
        )


        ax.set_xticklabels(
            [
                "ACS",
                "Hybrid",
            ],

            fontsize=
                xtick_fontsize,
        )


        # ========================================================
        # Y axis
        # ========================================================

        ax.set_ylim(
            y_min,
            y_max,
        )


        ax.set_ylabel(
            "Spatial OOF R²" if i % 5 == 0 else "",
            fontsize=ylabel_fontsize,
            labelpad=4,
        )


        ax.yaxis.set_major_locator(
            MaxNLocator(
                nbins=
                    5,

                prune=
                    None,
            )
        )


        ax.yaxis.set_major_formatter(
            FormatStrFormatter(
                "%.1f"
            )
        )


        ax.tick_params(
            axis=
                "y",

            labelsize=
                ytick_fontsize,

            width=
                0.7,

            length=
                3,
        )


        ax.tick_params(
            axis=
                "x",

            labelsize=
                xtick_fontsize,

            width=
                0.7,

            length=
                3,
        )


        if shared_ylim is not None and i % 5 != 0:
            ax.tick_params(axis="y", labelleft=False, left=False)

        # ========================================================
        # Zero reference
        # ========================================================

        if (
            y_min
            <
            0
            <
            y_max
        ):

            ax.axhline(
                0,

                color=
                    "#777777",

                linestyle=
                    "--",

                linewidth=
                    0.55,

                alpha=
                    0.45,

                zorder=
                    1,
            )


        # ========================================================
        # Horizontal grid only
        # ========================================================

        ax.grid(
            axis=
                "y",

            linewidth=
                grid_linewidth,

            alpha=
                grid_alpha,

            zorder=
                0,
        )


        # ========================================================
        # Only LEFT and BOTTOM spines
        # ========================================================

        ax.spines[
            "top"
        ].set_visible(
            False
        )

        ax.spines[
            "right"
        ].set_visible(
            False
        )

        ax.spines[
            "left"
        ].set_visible(
            True
        )

        ax.spines[
            "bottom"
        ].set_visible(
            True
        )


        ax.spines[
            "left"
        ].set_linewidth(
            0.85
        )

        ax.spines[
            "bottom"
        ].set_linewidth(
            0.85
        )


    # ============================================================
    # Hide unused axes
    # ============================================================

    for i in range(
        len(
            city_order
        ),
        len(
            axes
        ),
    ):

        axes[
            i
        ].set_visible(
            False
        )


    # ============================================================
    # Legend BELOW right panel
    # ============================================================

    legend_handles = [

        Line2D(
            [0],
            [0],

            marker="s",

            markersize=7,

            markerfacecolor=
                color_dict[
                    "ACS"
                ],

            markeredgecolor=
                "black",

            markeredgewidth=
                0.8,

            linestyle="none",

            label="ACS",
        ),

        Line2D(
            [0],
            [0],

            marker="s",

            markersize=7,

            markerfacecolor=
                color_dict[
                    "Hybrid"
                ],

            markeredgecolor=
                "black",

            markeredgewidth=
                0.8,

            linestyle="none",

            label="Hybrid",
        ),

        Line2D(
            [0],
            [0],

            marker="v",

            markersize=5,

            markerfacecolor=
                "white",

            markeredgecolor=
                "#555555",

            markeredgewidth=
                0.8,

            linestyle="none",

            label=(
                "Tukey outliers omitted"
            ),
        ),
    ]


    x0, y0, width, height = (
        right_rect
    )


    fig.legend(
        handles=
            legend_handles,

        loc=
            "upper center",

        bbox_to_anchor=(
            x0
            +
            width
            /
            2,

            y0
            -
            legend_vertical_gap,
        ),

        bbox_transform=
            fig.transFigure,

        ncol=
            3,

        frameon=
            False,

        fontsize=
            legend_fontsize,

        handletextpad=
            0.55,

        columnspacing=
            1.35,

        borderaxespad=
            0.0,
    )


    return axes


def plot_figure4(
    df_reporting,
    df_city_r2,
    df_city_summary,
    df_outliers,

    # ------------------------------------------------------------
    # Output
    # ------------------------------------------------------------

    output_pdf=None,

    output_png=None,

    # ------------------------------------------------------------
    # Overall canvas
    # ------------------------------------------------------------

    figsize=(
        18,
        10,
    ),

    # ------------------------------------------------------------
    # Main panel geometry
    # ------------------------------------------------------------

    left_right_width_ratio=(
        1.0,
        2.0,
    ),

    left_width_height_ratio=
        3 / 4,

    right_width_height_ratio=
        4 / 3,

    left_margin=
        0.050,

    right_margin=
        0.020,

    bottom_margin=
        0.135,

    top_margin=
        0.045,

    horizontal_gap=
        0.060,

    # ------------------------------------------------------------
    # LEFT
    # ------------------------------------------------------------

    left_xlim=None,

    left_x_padding_fraction=
        0.16,

    left_minimum_padding=
        0.0012,

    left_positive_padding_multiplier=
        1.25,

    left_bar_color=
        "#8CAAC4",

    left_bar_fill_alpha=
        0.20,

    left_bar_linewidth=
        1.50,

    left_bar_height=
        0.65,

    left_show_values=
        True,

    left_value_decimals=
        3,

    left_value_fontsize=
        9.0,

    left_value_offset_fraction=
        0.018,

    left_title=(
        ""
    ),

    left_xlabel=(
        "Δ spatial OOF R² (alternative − original)"
    ),

    left_title_fontsize=
        13,

    left_label_fontsize=
        11,

    left_tick_fontsize=
        9,

    # ------------------------------------------------------------
    # RIGHT city order
    # ------------------------------------------------------------

    city_order=None,

    # ------------------------------------------------------------
    # RIGHT:
    # close ACS / Hybrid
    # ------------------------------------------------------------

    right_model_positions=None,

    right_subplot_xlim=(
        0.20,
        0.80,
    ),

    right_box_width=
        0.10,

    # ------------------------------------------------------------
    # RIGHT styles
    # ------------------------------------------------------------

    right_color_dict=None,

    right_indicator_color=None,

    right_box_linewidth=
        0.90,

    right_whisker_linewidth=
        0.55,

    right_cap_linewidth=
        0.90,

    right_median_linewidth=
        0.65,

    # ------------------------------------------------------------
    # RIGHT outlier marker
    # ------------------------------------------------------------

    right_lower_triangle_position=
        0.035,

    right_upper_triangle_position=
        0.035,

    right_triangle_size=
        32,

    # ------------------------------------------------------------
    # RIGHT y-range
    # ------------------------------------------------------------

    right_lower_margin_fraction=
        0.13,

    right_upper_margin_fraction=
        0.11,

    right_minimum_span=
        0.20,

    # Shared vertical scale improves direct comparison across cities.
    right_shared_y_axis=True,
    right_shared_ylim=None,

    # ------------------------------------------------------------
    # RIGHT internal spacing
    # ------------------------------------------------------------

    right_column_gap_fraction=
        0.115,

    right_row_gap_fraction=
        0.155,

    # ------------------------------------------------------------
    # RIGHT typography
    # ------------------------------------------------------------

    right_city_title_fontsize=
        10.5,

    right_ylabel_fontsize=
        7.7,

    right_xtick_fontsize=
        7.4,

    right_ytick_fontsize=
        7.0,

    # ------------------------------------------------------------
    # RIGHT grid
    # ------------------------------------------------------------

    right_grid_linewidth=
        0.28,

    right_grid_alpha=
        0.22,

    # ------------------------------------------------------------
    # RIGHT legend
    # ------------------------------------------------------------

    right_legend_vertical_gap=
        0.050,

    right_legend_fontsize=
        7.7,

    # ------------------------------------------------------------
    # Optional panel labels
    # ------------------------------------------------------------

    panel_labels=None,

    panel_label_fontsize=
        14,

    swap_panels=False,
):
    """
    Final integrated Figure 4.

    LEFT
        pooled reporting-adjustment sensitivity

    RIGHT
        city-stratified ACS vs Hybrid spatial OOF R² distributions
        across 21 health measures

    No analytical quantities are recomputed here.
    """


    # ============================================================
    # Validate standardized inputs
    # ============================================================

    _validate_figure4_visualization_inputs(
        df_reporting=
            df_reporting,

        df_city_r2=
            df_city_r2,

        df_city_summary=
            df_city_summary,

        df_outliers=
            df_outliers,
    )


    # ============================================================
    # Default model positions
    # ============================================================

    if right_model_positions is None:

        right_model_positions = {
            "ACS":
                0.40,

            "Hybrid":
                0.60,
        }


    # ============================================================
    # Figure
    # ============================================================

    fig = plt.figure(
        figsize=
            figsize
    )


    # ============================================================
    # Main geometry
    # ============================================================

    (
        left_rect,
        right_rect,
    ) = (
        _figure4_panel_rectangles(
            fig=
                fig,

            left_margin=
                left_margin,

            right_margin=
                right_margin,

            bottom_margin=
                bottom_margin,

            top_margin=
                top_margin,

            horizontal_gap=
                horizontal_gap,

            width_ratio_left=
                left_right_width_ratio[
                    0
                ],

            width_ratio_right=
                left_right_width_ratio[
                    1
                ],

            left_width_height_ratio=
                left_width_height_ratio,

            right_width_height_ratio=
                right_width_height_ratio,
        )
    )

    if swap_panels:
        original_left_x = left_rect[0]
        panel_gap = right_rect[0] - (left_rect[0] + left_rect[2])
        left_rect[0] = original_left_x + right_rect[2] + panel_gap
        right_rect[0] = original_left_x


    # ============================================================
    # LEFT
    # ============================================================

    ax_left = fig.add_axes(
        left_rect
    )


    _plot_figure4_reporting_panel(
        ax=
            ax_left,

        df_reporting=
            df_reporting,

        xlim=
            left_xlim,

        x_padding_fraction=
            left_x_padding_fraction,

        minimum_padding=
            left_minimum_padding,

        positive_padding_multiplier=
            left_positive_padding_multiplier,

        bar_color=
            left_bar_color,

        bar_fill_alpha=
            left_bar_fill_alpha,

        bar_linewidth=
            left_bar_linewidth,

        bar_height=
            left_bar_height,

        show_values=
            left_show_values,

        value_decimals=
            left_value_decimals,

        value_fontsize=
            left_value_fontsize,

        value_offset_fraction=
            left_value_offset_fraction,

        title=
            left_title,

        xlabel=
            left_xlabel,

        title_fontsize=
            left_title_fontsize,

        label_fontsize=
            left_label_fontsize,

        tick_fontsize=
            left_tick_fontsize,
    )


    # Determine one vertical scale from the supplied Tukey summaries.
    # The raw distributions and Tukey whiskers are not modified.
    if right_shared_y_axis:
        if right_shared_ylim is None:
            _summary_lows = df_city_summary["whisker_low"].to_numpy(dtype=float)
            _summary_highs = df_city_summary["whisker_high"].to_numpy(dtype=float)
            _lows = _summary_lows[np.isfinite(_summary_lows)]
            _highs = _summary_highs[np.isfinite(_summary_highs)]
            if _lows.size == 0 or _highs.size == 0:
                raise ValueError("No finite city Tukey whiskers available.")
            _minimum, _maximum = float(_lows.min()), float(_highs.max())
            _range = max(_maximum - _minimum, 0.10)
            _low = _minimum - 0.08 * _range
            _high = min(1.0, _maximum + 0.08 * _range)
            right_shared_ylim = (_low, _high)
    else:
        right_shared_ylim = None

    # ============================================================
    # RIGHT
    # ============================================================

    axes_right = (
        _plot_figure4_city_panel(
            fig=
                fig,

            right_rect=
                right_rect,

            df_city_r2=
                df_city_r2,

            df_summary=
                df_city_summary,

            df_outliers=
                df_outliers,

            city_order=
                city_order,

            model_positions=
                right_model_positions,

            color_dict=
                right_color_dict,

            indicator_color=
                right_indicator_color,

            subplot_xlim=
                right_subplot_xlim,

            box_width=
                right_box_width,

            box_linewidth=
                right_box_linewidth,

            whisker_linewidth=
                right_whisker_linewidth,

            cap_linewidth=
                right_cap_linewidth,

            median_linewidth=
                right_median_linewidth,

            lower_triangle_position=
                right_lower_triangle_position,

            upper_triangle_position=
                right_upper_triangle_position,

            triangle_size=
                right_triangle_size,

            lower_margin_fraction=
                right_lower_margin_fraction,

            upper_margin_fraction=
                right_upper_margin_fraction,

            minimum_span=
                right_minimum_span,

            shared_ylim=right_shared_ylim,

            column_gap_fraction=
                right_column_gap_fraction,

            row_gap_fraction=
                right_row_gap_fraction,

            city_title_fontsize=
                right_city_title_fontsize,

            ylabel_fontsize=
                right_ylabel_fontsize,

            xtick_fontsize=
                right_xtick_fontsize,

            ytick_fontsize=
                right_ytick_fontsize,

            grid_linewidth=
                right_grid_linewidth,

            grid_alpha=
                right_grid_alpha,

            legend_vertical_gap=
                right_legend_vertical_gap,

            legend_fontsize=
                right_legend_fontsize,
        )
    )


    # ============================================================
    # Optional panel labels
    # ============================================================

    if panel_labels is not None:

        if len(
            panel_labels
        ) != 2:

            raise ValueError(
                "panel_labels must contain exactly two labels."
            )


        fig.text(
            left_rect[
                0
            ]
            -
            0.018,

            left_rect[
                1
            ]
            +
            left_rect[
                3
            ]
            +
            0.006,

            panel_labels[
                1 if swap_panels else 0
            ],

            fontsize=
                panel_label_fontsize,

            fontweight=
                "bold",

            ha=
                "left",

            va=
                "bottom",
        )


        fig.text(
            right_rect[
                0
            ]
            -
            0.012,

            right_rect[
                1
            ]
            +
            right_rect[
                3
            ]
            +
            0.006,

            panel_labels[
                0 if swap_panels else 1
            ],

            fontsize=
                panel_label_fontsize,

            fontweight=
                "bold",

            ha=
                "left",

            va=
                "bottom",
        )


    # ============================================================
    # Geometry diagnostics
    # ============================================================

    fig_w, fig_h = (
        fig.get_size_inches()
    )


    left_width_physical = (
        left_rect[
            2
        ]
        *
        fig_w
    )


    left_height_physical = (
        left_rect[
            3
        ]
        *
        fig_h
    )


    right_width_physical = (
        right_rect[
            2
        ]
        *
        fig_w
    )


    right_height_physical = (
        right_rect[
            3
        ]
        *
        fig_h
    )


    print(
        "\nFigure 4 visualization geometry"
    )

    print(
        "Right / Left width ratio:",
        round(
            right_width_physical
            /
            left_width_physical,
            3,
        )
    )

    print(
        "Left width / height:",
        round(
            left_width_physical
            /
            left_height_physical,
            3,
        )
    )

    print(
        "Right width / height:",
        round(
            right_width_physical
            /
            right_height_physical,
            3,
        )
    )

    print(
        "Left xlim:",
        tuple(
            round(
                x,
                4
            )
            for x
            in ax_left.get_xlim()
        )
    )

    print(
        "Right box positions:",
        right_model_positions
    )

    print(
        "Right subplot xlim:",
        right_subplot_xlim
    )


    # ============================================================
    # Save
    #
    # Do NOT call tight_layout().
    # ============================================================

    if output_pdf is not None:

        output_pdf = Path(
            output_pdf
        )

        output_pdf.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            output_pdf,

            bbox_inches=
                "tight",
        )


    if output_png is not None:

        output_png = Path(
            output_png
        )

        output_png.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            output_png,

            dpi=
                600,

            bbox_inches=
                "tight",
        )


    return (
        fig,
        ax_left,
        axes_right,
    )
