#!/usr/bin/env python3
"""Figure 4a and 4b: temporal, prior-health, reporting and city robustness.

Run: python Code/Figure4.py
Project layout: Code/, Data/, Result/, Figure/.
All plotting and calculation settings are retained from the final notebook.
Only file paths and module imports are adapted (plus Figure 2 OOF source).
"""

from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Save plots without blocking on GUI windows.
import matplotlib.pyplot as plt

from utils.figure4_visualization import (
    plot_figure4_prior_and_temporal,
    plot_figure4,
)

# Resolve paths from this script, never from the shell working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Data"
RESULT_DIR = PROJECT_ROOT / "Result"
FIGURE_DIR = PROJECT_ROOT / "Figure"


def main():
    # ============================================================
    # 1. Create common Figure output directory
    # ============================================================
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # 2. Figure 4a - prior-health and annual robustness
    # ============================================================
    # ============================================================
    # 1. Load analysis results
    # ============================================================

    df_prior = pd.read_csv(
        RESULT_DIR / "PriorSensitivity_2022for2023_Comparison.csv"
    )


    df_temporal = pd.read_csv(
        RESULT_DIR / "TemporalRobustness_Comparison.csv"
    )


    # ============================================================
    # 2. Basic checks
    # ============================================================

    assert len(
        df_prior
    ) == 21

    assert len(
        df_temporal
    ) == 84


    print(
        "Prior sensitivity:",
        df_prior.shape
    )

    print(
        "Temporal robustness:",
        df_temporal.shape
    )


    # ============================================================
    # 3. Temporal year-pair order
    # ============================================================

    TEMPORAL_PAIR_ORDER = [
        "2019 for 2020",
        "2020 for 2021",
        "2021 for 2022",
        "2022 for 2023",
    ]


    # ============================================================
    # 4. Plot
    # ============================================================

    fig, axes = (
        plot_figure4_prior_and_temporal(

            # ====================================================
            # Data
            # ====================================================

            df_prior=
                df_prior,

            df_temporal=
                df_temporal,

            # ====================================================
            # Output
            # ====================================================

            output_pdf=
                FIGURE_DIR / "Figure4a.pdf",

            output_png=
                FIGURE_DIR / "Figure4a.png",

            # ====================================================
            # Temporal ordering
            # ====================================================

            temporal_pair_order=
                TEMPORAL_PAIR_ORDER,

            # ====================================================
            # Overall figure
            # ====================================================

            figsize=(
                15,
                11,
            ),

            outer_width_ratios=(
                1.0,
                1.75,
            ),

            outer_wspace=
                0.2,

            # ====================================================
            # Left internal layout
            # ====================================================

            left_inner_wspace=
                0.72,

            left_height_fraction=
                0.95,

            # ====================================================
            # Both left bar plots
            # ====================================================

            bar_color=
                "#8CAAC4",

            bar_fill_alpha=
                0.20,

            bar_linewidth=
                1.50,

            bar_height=
                0.66,

            # ====================================================
            # Bar-end signed values
            # ====================================================

            bar_value_fontsize=
                9.0,

            bar_value_decimals=
                3,

            bar_value_offset_fraction=
                0.018,

            bar_value_color=
                "#333333",

            # ====================================================
            # Expanded negative-side x range
            # ====================================================

            bar_left_min=
                -0.015,

            bar_left_padding_fraction=
                0.20,

            bar_right_padding_fraction=
                0.22,

            # ====================================================
            # Left plot 1
            # ====================================================

            bar1_xlim=
                None,

            bar1_title=
                "Prior health + Complaints\nvs Prior health",

            # ====================================================
            # Left plot 2
            # ====================================================

            bar2_xlim=
                None,

            bar2_title=
                "Prior health + ACS + Complaints\nvs Prior health + ACS",

            # ====================================================
            # Left text
            # ====================================================

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

            # ====================================================
            # Right scatter
            # ====================================================

            scatter_xlim=(
                0.31,
                0.99,
            ),

            scatter_ylim=(
                0.31,
                0.99,
            ),

            scatter_s=
                82,

            scatter_alpha=
                0.90,

            scatter_edgecolor=
                "white",

            scatter_linewidth=
                1.25,

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

            # ====================================================
            # Legend
            # ====================================================

            legend_anchor=(
                0.965,
                0.075,
            ),

            legend_fontsize=
                7.6,

            legend_title_fontsize=
                9.5,

            legend_ncol=
                3,

            legend_marker_scale=
                0.90,

            # ====================================================
            # Grid / margins
            # ====================================================

            grid_linewidth=
                0.40,

            grid_alpha=
                0.30,

            figure_left=
                0.055,

            figure_right=
                0.985,

            figure_bottom=
                0.33,

            figure_top=
                0.94,
        )
    )

    plt.show()

    # ============================================================
    # 3. Figure 4b - input paths
    # ============================================================
    REPORTING_PATH = RESULT_DIR / "ReportingAdjustment.csv"
    CITY_R2_PATH = RESULT_DIR / "CityStratified_R2.csv"
    CITY_SUMMARY_PATH = RESULT_DIR / "CityStratified_R2_Summary.csv"
    OUTLIER_PATH = RESULT_DIR / "CityStratified_R2_TukeyOutliers.csv"

    # ============================================================
    # 4. Figure 4b - load and verify existing results
    # ============================================================
    # ================================================================
    # 2. Read standardized Figure 4 results
    # ================================================================

    df_reporting = pd.read_csv(
        REPORTING_PATH
    )

    df_city_r2 = pd.read_csv(
        CITY_R2_PATH
    )

    df_city_summary = pd.read_csv(
        CITY_SUMMARY_PATH
    )

    df_outliers = pd.read_csv(
        OUTLIER_PATH
    )


    print(
        "Reporting:",
        df_reporting.shape
    )

    print(
        "City R²:",
        df_city_r2.shape
    )

    print(
        "City summary:",
        df_city_summary.shape
    )

    print(
        "Tukey outliers:",
        df_outliers.shape
    )

    # ============================================================
    # 5. Figure 4b - original plotting parameters and export
    # ============================================================
    # ================================================================
    # 3. Final Figure 4
    # ================================================================

    fig, ax_left, axes_right = (
        plot_figure4(

            # --------------------------------------------------------
            # Standardized analytical outputs
            # --------------------------------------------------------

            df_reporting=
                df_reporting,

            df_city_r2=
                df_city_r2,

            df_city_summary=
                df_city_summary,

            df_outliers=
                df_outliers,

            # --------------------------------------------------------
            # Outputs
            # --------------------------------------------------------

            output_pdf=
                FIGURE_DIR
                /
                "Figure4b.pdf",

            output_png=
                FIGURE_DIR
                /
                "Figure4b.png",

            # ========================================================
            # OVERALL CANVAS
            # ========================================================

            figsize=(
                15,
                11,
            ),

            # ========================================================
            # MAIN PANEL GEOMETRY
            #
            # LEFT : RIGHT = 1 : 2
            # ========================================================

            left_right_width_ratio=(
                1.0,
                1.85,
            ),

            # --------------------------------------------------------
            # LEFT width : height = 3 : 4
            # --------------------------------------------------------

            left_width_height_ratio=
                3 / 4,

            # --------------------------------------------------------
            # RIGHT width : height = 4 : 3
            # --------------------------------------------------------

            right_width_height_ratio=
                4 / 3,

            # --------------------------------------------------------
            # Outer margins
            # --------------------------------------------------------

            left_margin=
                0.050,

            right_margin=
                0.020,

            bottom_margin=
                0.135,

            top_margin=
                0.045,

            # --------------------------------------------------------
            # LARGE gap between LEFT and RIGHT
            # --------------------------------------------------------

            horizontal_gap=
                0.10,

            # ========================================================
            # LEFT PANEL
            # ========================================================

            # None:
            # automatically derive compact asymmetric x-range
            # from Hybrid reporting-adjustment results
            left_xlim=
                None,

            left_x_padding_fraction=
                0.16,

            left_minimum_padding=
                0.0012,

            left_positive_padding_multiplier=
                1.25,

            # --------------------------------------------------------
            # Bars
            # --------------------------------------------------------

            left_bar_color=
                "#8CAAC4",

            left_bar_fill_alpha=
                0.20,

            left_bar_linewidth=
                1.50,

            left_bar_height=
                0.65,

            # --------------------------------------------------------
            # Value labels
            # --------------------------------------------------------

            left_show_values=
                True,

            left_value_decimals=
                3,

            left_value_fontsize=
                9.0,

            left_value_offset_fraction=
                0.018,

            # --------------------------------------------------------
            # Text
            # --------------------------------------------------------

            left_title=
                "",

            left_xlabel=(
                "Δ spatial OOF R² (alternative − original)"
            ),

            left_title_fontsize=
                13,

            left_label_fontsize=
                11,

            left_tick_fontsize=
                9,

            # ========================================================
            # RIGHT PANEL
            # ========================================================

            # --------------------------------------------------------
            # None = alphabetical city order:
            #
            # Austin
            # Baltimore
            # Boston
            # Chicago
            # Los Angeles
            # Memphis
            # New York
            # Philadelphia
            # San Diego
            # San Francisco
            # --------------------------------------------------------

            city_order=
                None,

            # ========================================================
            # KEY FINAL DESIGN:
            #
            # ACS and Hybrid are intentionally close together
            # within each city subplot.
            # ========================================================

            right_model_positions={
                "ACS":
                    0.40,

                "Hybrid":
                    0.60,
            },

            # --------------------------------------------------------
            # Tight x-window around two model boxes
            # --------------------------------------------------------

            right_subplot_xlim=(
                0.20,
                0.80,
            ),

            # --------------------------------------------------------
            # Narrow boxplots
            # --------------------------------------------------------

            right_box_width=
                0.10,

            # --------------------------------------------------------
            # Box style
            # --------------------------------------------------------

            right_color_dict={
                "ACS":
                    "#DDDDDD",

                "Hybrid":
                    "#8CAAC4",
            },

            right_indicator_color={
                "ACS":
                    "#777777",

                "Hybrid":
                    "#668EAE",
            },

            right_box_linewidth=
                0.90,

            right_whisker_linewidth=
                0.55,

            right_cap_linewidth=
                0.90,

            right_median_linewidth=
                0.65,

            # ========================================================
            # TUKEY OUTLIER INDICATORS
            #
            # Triangles are model-specific:
            # ACS triangle sits exactly under/above ACS;
            # Hybrid triangle sits exactly under/above Hybrid.
            # ========================================================

            right_lower_triangle_position=
                0.035,

            right_upper_triangle_position=
                0.035,

            right_triangle_size=
                32,

            # ========================================================
            # CITY-SPECIFIC Y AXES
            #
            # Based on Tukey whiskers rather than raw extreme outliers.
            # ========================================================

            right_lower_margin_fraction=
                0.13,

            right_upper_margin_fraction=
                0.11,

            right_minimum_span=
                0.20,

            # One R² range across all cities for visual comparability.
            right_shared_y_axis=True,
            right_shared_ylim=None,

            # ========================================================
            # LARGER SPACING BETWEEN CITY SUBPLOTS
            # ========================================================

            right_column_gap_fraction=
                0.115,

            right_row_gap_fraction=
                0.155,

            # ========================================================
            # RIGHT TYPOGRAPHY
            # ========================================================

            right_city_title_fontsize=
                12,

            right_ylabel_fontsize=
                10,

            right_xtick_fontsize=
                8,

            right_ytick_fontsize=
                8,

            # ========================================================
            # RIGHT GRID
            # ========================================================

            right_grid_linewidth=
                0.28,

            right_grid_alpha=
                0.22,

            # ========================================================
            # LEGEND
            #
            # Kept below the right panel with enough clearance.
            # ========================================================

            right_legend_vertical_gap=
                0.035,

            right_legend_fontsize=
                10,

            # ========================================================
            # PANEL LABELS
            #
            # Keep None if added later in Illustrator/manuscript.
            # Use ("a", "b") if desired here.
            # ========================================================

            panel_label_fontsize=
                14,

            swap_panels=True,
        )
    )

    plt.show()


if __name__ == "__main__":
    main()
