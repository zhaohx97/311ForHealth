#!/usr/bin/env python3
"""Figure 3: CC/ACS/Hybrid performance comparisons and OOF SHAP attribution.

Run: python Code/Figure3.py
Project layout: Code/, Data/, Result/, Figure/.
All plotting and calculation settings are retained from the final notebook.
Only file paths and module imports are adapted (plus Figure 2 OOF source).
"""

from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Save plots without blocking on GUI windows.
import matplotlib.pyplot as plt
from utils.figure3_visualization import plot_figure3_revised


# Identical Times-Bold panel-label styling to Figures 1, 2 and 4.
# Only the panel markers are changed; all Figure 3 drawing settings remain intact.
LABEL_FONTSIZE_FRACTION = 0.002
MIN_LABEL_FONTSIZE = 16.0
OUTER_MARGIN_FRACTION = 0.012
BOTTOM_GAP_MULTIPLIER = 1.00
LABEL_POSITION_FRACTION = 1.00
LABEL_B_GAP_FRACTION = 0.30  # (b) baseline within the two right-panel gap.
COMBINED_PNG_DPI = 600


def _apply_consistent_panel_labels(pdf_path, png_path, fig, axes):
    """Place (a), (b), (c) in Times-Bold, preserving the source PDF vectors.

    (a) is centered beneath the left panel; (b) sits between the upper and
    lower right panels; and (c) is centered beneath the lower right panel.
    As in Figure 2, an annotation band is appended under the *existing* PDF
    rather than modifying the underlying axes, data, geometry, or plot code.
    """
    try:
        import fitz  # PyMuPDF; required for vector-preserving PDF labels.
    except ImportError as exc:
        raise ImportError(
            "Matching Figure 3 labels require PyMuPDF: pip install pymupdf"
        ) from exc

    # The plotting utility returns named original Matplotlib axes.
    left = axes["left"]
    top_right = axes["right_top"]
    bottom_right = axes["right_bottom"]

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # Depending on the utility's savefig setting, the PDF may be tight-cropped
    # or use the original full figure bounding box. Match its actual dimensions.
    tight_bbox = fig.get_tightbbox(renderer).padded(0.1)  # inches
    from matplotlib.transforms import Bbox
    full_bbox = Bbox.from_bounds(0, 0, *fig.get_size_inches())

    with fitz.open(str(pdf_path)) as original:
        if len(original) != 1:
            raise ValueError("Figure 3 must be a single-page PDF.")
        old_page = original[0]
        old_width = old_page.rect.width
        old_height = old_page.rect.height

        def size_error(bbox):
            return (abs(old_width / 72 - bbox.width) / max(1, bbox.width)
                    + abs(old_height / 72 - bbox.height) / max(1, bbox.height))

        crop = min((tight_bbox, full_bbox), key=size_error)

        def pdf_x(center_fraction):
            x_inches = center_fraction * fig.get_figwidth()
            return (x_inches - crop.x0) / crop.width * old_width

        def pdf_y(display_y):
            # Matplotlib display y increases upward; PDF y increases downward.
            y_inches = display_y / fig.dpi
            return (crop.y1 - y_inches) / crop.height * old_height

        label_fontsize = max(
            MIN_LABEL_FONTSIZE,
            LABEL_FONTSIZE_FRACTION * old_width / (1 + 2 * OUTER_MARGIN_FRACTION),
        )
        outer_margin = OUTER_MARGIN_FRACTION * old_width
        bottom_gap = BOTTOM_GAP_MULTIPLIER * label_fontsize
        page_height = old_height + bottom_gap + outer_margin

        # Calculate centers from each subplot's unchanged axes position.
        def ax_center_x(ax):
            p = ax.get_position()
            return pdf_x((p.x0 + p.x1) / 2)

        # The (b) marker is placed only in the whitespace already separating
        # the two right-hand subplots; their positions are left unchanged.
        top_right_bottom = pdf_y(top_right.get_tightbbox(renderer).y0)
        bottom_right_top = pdf_y(bottom_right.get_tightbbox(renderer).y1)
        gap = bottom_right_top - top_right_bottom
        if gap < 1.35 * label_fontsize:
            raise ValueError(
                "Not enough whitespace between Figure 3b and 3c for a "
                "non-overlapping label. Keep all original layout parameters "
                "unchanged and review the current panel annotations. "
                f"Gap: {gap:.1f} pt; label: {label_fontsize:.1f} pt."
            )
        baseline_b = top_right_bottom + gap * LABEL_B_GAP_FRACTION
        baseline_bottom = old_height + bottom_gap * LABEL_POSITION_FRACTION

        with fitz.open() as assembled:
            page = assembled.new_page(width=old_width, height=page_height)
            page.show_pdf_page(
                fitz.Rect(0, 0, old_width, old_height), original, 0,
                keep_proportion=True,
            )
            for marker, center_x, baseline in (
                ("(a)", ax_center_x(left), baseline_bottom),
                ("(b)", ax_center_x(top_right), baseline_b),
                ("(c)", ax_center_x(bottom_right), baseline_bottom),
            ):
                marker_width = fitz.get_text_length(
                    marker, fontname="tibo", fontsize=label_fontsize
                )
                page.insert_text(
                    (center_x - marker_width / 2, baseline), marker,
                    fontname="tibo", fontsize=label_fontsize, color=(0, 0, 0),
                )
            temporary = Path(str(pdf_path) + ".tmp.pdf")
            assembled.save(str(temporary), garbage=4, deflate=True)

    temporary.replace(pdf_path)
    # The final PDF remains a single vector page. Only the companion PNG
    # is rasterized, at the same 600-dpi setting used for Figure 2.
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
    # 1. Paths, file checks and original source data loading
    # ============================================================
    # Make the expected project structure explicit.
    project_root = PROJECT_ROOT
    inputs = {
        "complaint": project_root / "Result/Metrics_complaint_2022for2023.csv",
        "acs": project_root / "Result/Metrics_demographics_2022for2023.csv",
        "hybrid": project_root / "Result/Metrics_hybrid_2022for2023.csv",
        "shap": project_root / "Result/SHAP_feature_summary_hybrid_2022for2023.csv",
        "health_type": project_root / "Data/Dataset_healthtype.csv",
    }
    missing = [str(p) for p in inputs.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Required input files not found in the project Data/ or Result/ directories:\n"
            + "\n".join(missing)
        )

    # Original source tables, unchanged.
    df_complaint_metrics = pd.read_csv(inputs["complaint"])
    df_demographics_metrics = pd.read_csv(inputs["acs"])
    df_hybrid_metrics = pd.read_csv(inputs["hybrid"])
    df_shap_feature = pd.read_csv(inputs["shap"])
    df_type = pd.read_csv(inputs["health_type"], index_col=0)
    df_type.columns = ["y_variable", "legends"]

    # ============================================================
    # 2. Original Figure 3 plotting and export parameters
    # ============================================================
    output_dir = FIGURE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    (
        fig,
        axes,
        df_comparison,
        df_dumbbell,
        df_feature_rank,
        df_shap_domain,
    ) = plot_figure3_revised(
        df_complaint_metrics=df_complaint_metrics,
        df_demographics_metrics=df_demographics_metrics,
        df_hybrid_metrics=df_hybrid_metrics,
        df_shap_feature=df_shap_feature,
        df_type=df_type,
        output_pdf=output_dir / "Figure3.pdf",
        output_png=output_dir / "Figure3.png",
        # Three-panel layout
        figsize=(15, 11),
        width_ratios=(1.25, 1.0),
        wspace=0.35,
        right_height_ratios=(1.55, 1.0),
        right_hspace=0.40,
        panel_labels=None,  # Add matching Times-Bold panel markers below.
        # Left — same values and calculation as original
        left_scatter_size=185,
        left_scatter_linewidth=1.5,
        left_fill_alpha=0.27,
        left_label_fontsize=9.5,
        left_xlim=(0.125, 0.875),
        left_ylim=(0, 100),
        left_box_aspect=1.0,
        # Top-right — do not clip lower ACS R² values
        dumbbell_point_size=36,
        dumbbell_label_fontsize=8.7,
        dumbbell_xlim=(0.3, 1.0),
        right_top_box_aspect=None,
        show_delta_text=False,
        # Bottom-right — original top-eight SHAP ranking
        shap_top_n=8,
        shap_xlim=None,
        shap_category_marker_size=12,
        shap_category_marker_width=2.35,
        shap_overall_marker_size=6.7,
        shap_feature_fontsize=9.0,
        legend_fontsize=9,
    )

    # Apply only the shared Figure 1/2/4 panel-label styling to the PDF.
    _apply_consistent_panel_labels(
        output_dir / "Figure3.pdf", output_dir / "Figure3.png", fig, axes,
    )

    print("Saved:")
    print(output_dir / "Figure3.pdf")
    print(output_dir / "Figure3.png")
    plt.show()


if __name__ == "__main__":
    main()
