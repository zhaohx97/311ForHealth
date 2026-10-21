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
        panel_labels=("a", "b", "c"),
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

    print("Saved:")
    print(output_dir / "Figure3.pdf")
    print(output_dir / "Figure3.png")
    plt.show()


if __name__ == "__main__":
    main()
