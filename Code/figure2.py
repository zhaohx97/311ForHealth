#!/usr/bin/env python3
"""Figure 2: 21 health-measure spatial OOF predictions and CC model R².

Run: python Code/Figure2.py
Project layout: Code/, Data/, Result/, Figure/.
All plotting and calculation settings are retained from the final notebook.
Only file paths and module imports are adapted (plus Figure 2 OOF source).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Save plots without blocking on GUI windows.
import matplotlib
matplotlib.use("Agg")  # Non-interactive: save all final figures without a GUI.
import matplotlib.pyplot as plt

from utils.figure2_visualization import plot_figure2

# Figure 2 uses the same Times-Bold PDF panel labels as Figures 1 and 4.
# These settings affect only the (a)/(b) labels; the plots remain unchanged.
LABEL_FONTSIZE_FRACTION = 0.002
MIN_LABEL_FONTSIZE = 16.0
BOTTOM_GAP_MULTIPLIER = 1.00
LABEL_POSITION_FRACTION = 0.50
COMBINED_PNG_DPI = 600  # Retain the original Figure 2 PNG resolution.


def _apply_consistent_panel_labels(
    pdf_path, png_path, fig, axes_pred, legend_ax, ax_comparison
):
    """Place (a)/(b) under the two panels using Figure 1/4 label styling.

    The input PDF is preserved as a vector page. Only the existing panel labels
    are replaced by centered Times-Bold labels in the bottom annotation band.
    No axes, data, legends, or drawing settings are changed.
    """
    try:
        import fitz  # PyMuPDF, as in Figures 1 and 4.
    except ImportError as exc:
        raise ImportError(
            "Figure 2's matching PDF labels require PyMuPDF. "
            "Install it with: pip install pymupdf"
        ) from exc

    # Calculate each panel's center from its original Matplotlib layout.
    # Match bbox_inches='tight' used by the unchanged plotting function.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tight = fig.get_tightbbox(renderer).padded(0.1)
    width_inches = fig.get_figwidth()
    left_axes = list(np.asarray(axes_pred, dtype=object).flat) + [legend_ax]
    left_x0 = min(ax.get_position().x0 for ax in left_axes)
    left_x1 = max(ax.get_position().x1 for ax in left_axes)
    right_bbox = ax_comparison.get_position()
    centers_in_inches = (
        (left_x0 + left_x1) / 2.0 * width_inches,
        (right_bbox.x0 + right_bbox.x1) / 2.0 * width_inches,
    )

    with fitz.open(str(pdf_path)) as original:
        if len(original) != 1:
            raise ValueError("Figure 2 must be one PDF page.")
        old_page = original[0]
        old_width = old_page.rect.width
        old_height = old_page.rect.height

        # Identical family, weight and size rule to the Figure 1/4 composers.
        # The original figure width is used (not a resized or rasterized panel).
        label_fontsize = max(
            MIN_LABEL_FONTSIZE,
            LABEL_FONTSIZE_FRACTION * old_width / (1 + 2 * 0.012),
        )
        bottom_gap = BOTTOM_GAP_MULTIPLIER * label_fontsize
        # A small outer margin matches the other composite figure exports.
        outer_margin = 0.012 * old_width
        page_height = old_height + bottom_gap + outer_margin

        # Convert figure-space center coordinates to cropped PDF coordinates.
        # Using the saved PDF's width accounts for possible minor text-bound
        # differences between Agg and PDF rendering.
        tight_width_pt = tight.width * 72
        left_x, right_x = (
            ((x - tight.x0) * 72) * (old_width / tight_width_pt)
            for x in centers_in_inches
        )

        with fitz.open() as result:
            page = result.new_page(width=old_width, height=page_height)
            page.show_pdf_page(
                fitz.Rect(0, 0, old_width, old_height), original, 0,
                keep_proportion=True,
            )
            baseline = old_height + bottom_gap * LABEL_POSITION_FRACTION
            for marker, center_x in (("(a)", left_x), ("(b)", right_x)):
                marker_width = fitz.get_text_length(
                    marker, fontname="tibo", fontsize=label_fontsize
                )
                page.insert_text(
                    (center_x - marker_width / 2, baseline), marker,
                    fontname="tibo", fontsize=label_fontsize,
                    color=(0, 0, 0),
                )
            result.save(str(pdf_path) + ".tmp.pdf", garbage=4, deflate=True)

    tmp_path = Path(str(pdf_path) + ".tmp.pdf")
    tmp_path.replace(pdf_path)
    # Retain the previously configured 600 dpi image output. The PDF remains
    # vector-based; rendering occurs only when producing the PNG companion.
    if png_path is not None:
        with fitz.open(str(pdf_path)) as doc:
            pix = doc[0].get_pixmap(dpi=COMBINED_PNG_DPI, alpha=False)
            pix.save(str(png_path))


# Resolve paths from this script, never from the shell working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Data"
RESULT_DIR = PROJECT_ROOT / "Result"
FIGURE_DIR = PROJECT_ROOT / "Figure"


def main():
    # ============================================================
    # 1. Resolve required inputs and outputs
    # ============================================================
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    COMPLAINT_METRIC_PATH = RESULT_DIR / "Metrics_complaint_2022for2023.csv"
    HEALTH_TYPE_PATH = DATA_DIR / "Dataset_healthtype.csv"
    # Confirmed primary CC-model predictions, instead of the Figure 4b robustness export.
    OOF_PATH = RESULT_DIR / "Predictions_complaint_2022for2023.csv"
    FIGURE2_PDF_PATH = FIGURE_DIR / "Figure2.pdf"
    FIGURE2_PNG_PATH = FIGURE_DIR / "Figure2.png"

    # ============================================================
    # 2. Confirm input files
    # ============================================================
    # ================================================================
    # 2. Confirm source files
    # ================================================================

    source_paths = [
        COMPLAINT_METRIC_PATH,
        HEALTH_TYPE_PATH,
        OOF_PATH,
    ]


    for path in source_paths:

        print(
            f"{path.exists():5}  {path}"
        )


    if not all(
        path.exists()
        for path
        in source_paths
    ):

        raise FileNotFoundError(
            "One or more Figure 2 source files are missing."
        )

    # ============================================================
    # 3. Read primary CC metrics
    # ============================================================
    # ================================================================
    # 3. Read latest Complaint-model primary metrics
    # ================================================================

    df_metric_complaint = pd.read_csv(
        COMPLAINT_METRIC_PATH
    )


    print(
        "Complaint metric shape:",
        df_metric_complaint.shape
    )

    print(
        "Complaint metric columns:",
        df_metric_complaint.columns.tolist()
    )


    print(df_metric_complaint.head().to_string(index=False))

    # ============================================================
    # 4. Validate metric table
    # ============================================================
    # ================================================================
    # 4. Validate Complaint metric file
    #
    # Expected current structure includes:
    #
    #   y_variable
    #   r2_complaint
    #   rmse_complaint
    #   mae_complaint
    #   n_observations
    # ================================================================

    required_metric_cols = {
        "y_variable",
        "r2_complaint",
    }


    missing_metric_cols = (
        required_metric_cols
        -
        set(
            df_metric_complaint.columns
        )
    )


    if missing_metric_cols:

        raise ValueError(
            "Complaint metric file is missing required columns: "
            f"{sorted(missing_metric_cols)}"
        )


    if not (
        df_metric_complaint[
            "y_variable"
        ]
        .is_unique
    ):

        raise ValueError(
            "Duplicate y_variable values found in Complaint metrics."
        )


    if len(
        df_metric_complaint
    ) != 21:

        raise ValueError(
            "Expected 21 health measures in Complaint metrics; "
            f"found {len(df_metric_complaint)}."
        )


    df_metric_complaint[
        "r2_complaint"
    ] = pd.to_numeric(
        df_metric_complaint[
            "r2_complaint"
        ],
        errors="raise",
    )

    # ============================================================
    # 5. Verify CC model metrics
    # ============================================================
    # ================================================================
    # 5. Sanity check:
    # confirm latest primary Complaint spatial OOF R²
    # ================================================================

    mean_complaint_r2 = (
        df_metric_complaint[
            "r2_complaint"
        ]
        .mean()
    )


    print(
        "Mean Complaint spatial OOF R²:",
        mean_complaint_r2
    )


    EXPECTED_COMPLAINT_MEAN_R2 = (
        0.45478906711880734
    )


    np.testing.assert_allclose(
        mean_complaint_r2,
        EXPECTED_COMPLAINT_MEAN_R2,
        atol=1e-6,
    )


    print(
        "✓ Latest Complaint spatial-block-CV metrics confirmed."
    )

    # ============================================================
    # 6. Read health type metadata
    # ============================================================
    # ================================================================
    # 6. Read health-measure category metadata
    # ================================================================

    df_type = pd.read_csv(
        HEALTH_TYPE_PATH,
        index_col=0,
    )


    df_type.columns = [
        "y_variable",
        "legends",
    ]


    print(
        "Health-type shape:",
        df_type.shape
    )


    print(df_type.to_string(index=False))

    # ============================================================
    # 7. Validate health metadata
    # ============================================================
    # ================================================================
    # 7. Validate health-measure metadata
    # ================================================================

    required_type_cols = {
        "y_variable",
        "legends",
    }


    missing_type_cols = (
        required_type_cols
        -
        set(
            df_type.columns
        )
    )


    if missing_type_cols:

        raise ValueError(
            "Health-type file is missing columns: "
            f"{sorted(missing_type_cols)}"
        )


    if not (
        df_type[
            "y_variable"
        ]
        .is_unique
    ):

        raise ValueError(
            "Duplicate y_variable values found in health-type metadata."
        )


    if len(
        df_type
    ) != 21:

        raise ValueError(
            "Expected 21 health measures in Dataset_healthtype.csv; "
            f"found {len(df_type)}."
        )

    # ============================================================
    # 8. Merge Figure 2b metrics
    # ============================================================
    # ================================================================
    # 8. Build Figure 2b input dataframe
    #
    # IMPORTANT:
    # Keep df_type on the LEFT of the merge.
    #
    # This preserves the original category first-appearance order.
    # The plotting function will then sort r2_complaint descending
    # WITHIN each category.
    # ================================================================

    df_comparison = (
        df_type
        .merge(
            df_metric_complaint[
                [
                    "y_variable",
                    "r2_complaint",
                ]
            ],

            on=
                "y_variable",

            how=
                "inner",

            validate=
                "one_to_one",
        )
    )


    if len(
        df_comparison
    ) != 21:

        raise ValueError(
            "Expected 21 matched health measures for Figure 2b; "
            f"found {len(df_comparison)}."
        )


    print(df_comparison.to_string(index=False))

    # ============================================================
    # 9. Verify Figure 2b health domain order
    # ============================================================
    # ================================================================
    # 9. Confirm Figure 2b category order
    # ================================================================

    category_order = (
        df_comparison[
            "legends"
        ]
        .drop_duplicates()
        .tolist()
    )


    print(
        "Figure 2b category order:"
    )

    print(
        category_order
    )


    EXPECTED_CATEGORY_ORDER = [
        "Health Outcomes",
        "Health Risk Behaviors",
        "Health Status",
        "Prevention",
    ]


    if (
        category_order
        !=
        EXPECTED_CATEGORY_ORDER
    ):

        raise ValueError(
            "Unexpected Figure 2b category order.\n"
            f"Expected: {EXPECTED_CATEGORY_ORDER}\n"
            f"Found:    {category_order}"
        )


    print(
        "✓ Figure 2b category order confirmed."
    )

    # ============================================================
    # 10. Preview Figure 2b sorting
    # ============================================================
    # ================================================================
    # 10. Preview Figure 2b final ordering
    #
    # Category first-appearance order
    # +
    # descending Complaint R² within category
    # ================================================================

    sorted_groups = []


    for category in category_order:

        group = (
            df_comparison.loc[
                df_comparison[
                    "legends"
                ]
                ==
                category
            ]
            .sort_values(
                by=
                    "r2_complaint",

                ascending=
                    False,
            )
            .copy()
        )


        sorted_groups.append(
            group
        )


    df_figure2b_preview = pd.concat(
        sorted_groups,
        ignore_index=True,
    )


    print(df_figure2b_preview[["legends", "y_variable", "r2_complaint"]].to_string(index=False))

    # ============================================================
    # 11. Read primary CC OOF predictions
    # ============================================================
    # Original pooled held-out CC-model predictions, not Figure 4b's alternative feature representation.
    df_oof = pd.read_csv(OOF_PATH, dtype={"TractFIPS": str})
    print("Primary CC OOF prediction shape:", df_oof.shape)
    print("Prediction columns:", df_oof.columns.tolist())

    # ============================================================
    # 12. Select CC OOF fields
    # ============================================================
    # The primary prediction export is already complaint-only.
    # Select the same plotting fields as the original notebook; no model_type filter.
    required_oof_cols = {"City", "y_variable", "y_true", "y_pred"}
    missing_oof_cols = required_oof_cols - set(df_oof.columns)
    if missing_oof_cols:
        raise ValueError(
            "Primary CC OOF file is missing required columns: "
            f"{sorted(missing_oof_cols)}"
        )

    keep_cols = [
        col for col in [
            "TractFIPS", "City", "spatial_fold", "y_variable", "y_true", "y_pred"
        ]
        if col in df_oof.columns
    ]
    df_pred_complaint = df_oof[keep_cols].copy()

    # ============================================================
    # 13. Clean OOF prediction values
    # ============================================================
    # ================================================================
    # 13. Clean Figure 2a prediction data
    # ================================================================

    df_pred_complaint[
        "y_true"
    ] = pd.to_numeric(
        df_pred_complaint[
            "y_true"
        ],
        errors="coerce",
    )


    df_pred_complaint[
        "y_pred"
    ] = pd.to_numeric(
        df_pred_complaint[
            "y_pred"
        ],
        errors="coerce",
    )


    df_pred_complaint = (
        df_pred_complaint
        .dropna(
            subset=[
                "City",
                "y_variable",
                "y_true",
                "y_pred",
            ]
        )
        .copy()
    )


    print(
        "Complaint OOF rows:",
        len(
            df_pred_complaint
        )
    )

    print(
        "Cities:",
        df_pred_complaint[
            "City"
        ]
        .nunique()
    )

    print(
        "Health measures:",
        df_pred_complaint[
            "y_variable"
        ]
        .nunique()
    )

    # ============================================================
    # 14. Validate OOF prediction coverage
    # ============================================================
    # ================================================================
    # 14. Validate Figure 2a structure
    # ================================================================

    if (
        df_pred_complaint[
            "City"
        ]
        .nunique()
        !=
        10
    ):

        raise ValueError(
            "Expected 10 cities in Figure 2a."
        )


    if (
        df_pred_complaint[
            "y_variable"
        ]
        .nunique()
        !=
        21
    ):

        raise ValueError(
            "Expected 21 health measures in Figure 2a."
        )


    print(
        "\nCities:"
    )

    print(
        sorted(
            df_pred_complaint[
                "City"
            ]
            .unique()
            .tolist()
        )
    )

    # ============================================================
    # 15. Compare health-measure sets and verify model R²
    # ============================================================
    # ================================================================
    # 15. Confirm same health-measure SET
    #
    # Figure 2a ORDER:
    #     alphabetical A -> Z
    #
    # Figure 2b ORDER:
    #     category
    #     +
    #     Complaint R² descending within category
    # ================================================================

    health_set_2a = set(
        df_pred_complaint[
            "y_variable"
        ]
        .unique()
    )


    health_set_2b = set(
        df_comparison[
            "y_variable"
        ]
        .unique()
    )


    if (
        health_set_2a
        !=
        health_set_2b
    ):

        print(
            "Only in Figure 2a:",
            sorted(
                health_set_2a
                -
                health_set_2b
            )
        )

        print(
            "Only in Figure 2b:",
            sorted(
                health_set_2b
                -
                health_set_2a
            )
        )


        raise ValueError(
            "Figure 2a and Figure 2b do not contain "
            "the same set of health measures."
        )


    print(
        "✓ Figure 2a and Figure 2b use the same 21 health measures."
    )

    # Extra verification for the confirmed OOF-source replacement: the prediction
    # values must reproduce the existing, unmodified primary CC-model R² metrics.
    # This checks inputs; it does not modify the data or figure.
    def _pooled_r2(group):
        y_true = group["y_true"].to_numpy(dtype=float)
        y_pred = group["y_pred"].to_numpy(dtype=float)
        return 1.0 - np.square(y_true - y_pred).sum() / np.square(y_true - y_true.mean()).sum()

    r2_from_predictions = df_pred_complaint.groupby("y_variable", sort=False).apply(_pooled_r2)
    r2_expected = df_metric_complaint.set_index("y_variable")["r2_complaint"]
    r2_diffs = (r2_from_predictions - r2_expected).abs()
    if (r2_diffs > 5e-5).any():
        bad = r2_diffs[r2_diffs > 5e-5]
        raise ValueError(
            "Primary OOF predictions do not match the Figure 2b CC model metrics: "
            f"{bad.to_dict()}. Verify that the files come from the same final run."
        )
    print("✓ Primary CC predictions reproduce all 21 reported pooled OOF R² values.")

    # ============================================================
    # 16. Preview alphabetical Figure 2a order
    # ============================================================
    # ================================================================
    # 16. Preview Figure 2a alphabetical order
    # ================================================================

    figure2a_order_preview = sorted(
        df_pred_complaint[
            "y_variable"
        ]
        .unique()
        .tolist(),

        key=lambda x: x.upper(),
    )


    print(
        "Figure 2a order:"
    )


    for i, health in enumerate(
        figure2a_order_preview,
        start=1,
    ):

        print(
            f"{i:02d}. {health}"
        )

    # ============================================================
    # 17. Render Figure 2 using unchanged plotting settings
    # ============================================================
    # ================================================================
    # 18. FINAL FIGURE 2 — VISUAL OPTIMIZATION ONLY
    # ================================================================
    # Figure 2a: 21 outcomes sorted alphabetically; 5 rows × 5 columns;
    # the four remaining cells show the ten-city legend.
    # Figure 2b: same R² values, same health-domain order and within-domain ranking.
    # Both panels are rendered from the original spatial-block-CV outputs.

    (
        fig,
        axes_pred,
        legend_ax,
        ax_comparison,
        health_order_2a,
        df_sorted_2b,
    ) = plot_figure2(
        df_pred=df_pred_complaint,
        df_comparison=df_comparison,
        output_pdf=FIGURE2_PDF_PATH,
        output_png=FIGURE2_PNG_PATH,

        # Overall geometry and whitespace
        figsize=(15, 9),
        width_ratios=(1.0, 1.5),
        outer_wspace=0.18,
        left_margin=0.065,
        right_margin=0.985,
        bottom_margin=0.14,
        top_margin=0.91,

        # Figure 2a: predictions, 5 × 5
        prediction_n_rows=5,
        prediction_n_cols=5,
        prediction_wspace=0.34,
        prediction_hspace=0.42,
        prediction_point_size=1.15,
        prediction_point_alpha=0.32,
        prediction_title_fontsize=8.7,
        prediction_tick_fontsize=6.7,
        prediction_legend_ncol=5,

        # Figure 2b: unchanged data, improved presentation
        comparison_value_col='r2_complaint',
        comparison_measure_col='y_variable',
        comparison_category_col='legends',
        comparison_ylim=(0, 1),
        comparison_bar_width=0.5,
        comparison_color_complaint='#769BB9',
        comparison_bar_fill_alpha=0.26,
        comparison_category_y_fraction=0.90,
        comparison_width_height_ratio=1.15,

        panel_labels=None,  # Labels added below, matching Figures 1 and 4.
    )

    # Replace only the panel markers with centered Times-Bold labels.
    # All Figure 2 plot content is preserved as existing PDF vector graphics.
    _apply_consistent_panel_labels(
        FIGURE2_PDF_PATH, FIGURE2_PNG_PATH,
        fig, axes_pred, legend_ax, ax_comparison,
    )

    plt.show()

    # ============================================================
    # 18. Final diagnostics
    # ============================================================
    # ================================================================
    # 19. Final Figure 2 checks
    # ================================================================

    print(
        "Figure 2a — alphabetical order:"
    )

    print(
        health_order_2a
    )


    print(
        "\nFigure 2b — category + descending Complaint R²:"
    )


    print(df_sorted_2b[["legends", "y_variable", "r2_complaint"]].to_string(index=False))


    print(
        "\nOutput files:"
    )

    print(
        FIGURE2_PDF_PATH
    )

    print(
        FIGURE2_PNG_PATH
    )


if __name__ == "__main__":
    main()
