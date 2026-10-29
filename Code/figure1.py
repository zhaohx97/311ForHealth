#!/usr/bin/env python3
"""Figure 1: spatial and thematic 311 complaint patterns (one-page composite).

Run: python Code/Figure1.py
Project layout: Code/, Data/, Result/, Figure/.
All data processing and plotting settings for panels (a) and (b) are unchanged.
The original panel PDFs are placed together as one vector-preserving PDF page.
"""

from pathlib import Path
from io import BytesIO

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Save plots without blocking on GUI windows.
import matplotlib
matplotlib.use("Agg")  # Non-interactive: save all final figures without a GUI.
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import rcParams
import geopandas as gpd

from utils.figure1_visualization import (
    visualize_spatial_distribution,
    visualize_thematic_structures,
)

# Resolve paths from this script, never from the shell working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Data"
RESULT_DIR = PROJECT_ROOT / "Result"
FIGURE_DIR = PROJECT_ROOT / "Figure"


# Combined-file output options (these do not affect either panel's data or drawing).
COMBINED_PNG_DPI = 300  # The PDF is vector-preserving and recommended for submission.


def _to_pdf_bytes(fig):
    """Render an already-created Matplotlib figure to vector PDF in memory."""
    buffer = BytesIO()
    fig.savefig(buffer, format="pdf", dpi=600, bbox_inches="tight")
    return buffer.getvalue()


def _save_combined_figure(panel_a_pdf, panel_b_pdf, pdf_path, png_path=None):
    """Place the two unchanged vector panels on one PDF page; add (a) and (b).

    This is a *layout-only* operation: no maps, bars, axes, colors, legends,
    geographic data, values, or panel-internal coordinates are recalculated.

    Dependency for combining PDFs: pip install pymupdf
    """
    try:
        import fitz  # PyMuPDF; used only for lossless/vector PDF composition.
    except ImportError as exc:
        raise ImportError(
            "Creating the one-page Figure 1 PDF requires PyMuPDF. "
            "Install it with: pip install pymupdf"
        ) from exc

    with fitz.open(stream=panel_a_pdf, filetype="pdf") as document_a, \
         fitz.open(stream=panel_b_pdf, filetype="pdf") as document_b:
        if len(document_a) != 1 or len(document_b) != 1:
            raise ValueError("Each original subfigure must be a one-page PDF.")

        source_a = document_a[0].rect
        source_b = document_b[0].rect

        # Both source pages are placed at the same display width, preserving
        # their original aspect ratios and all individual panel layouts.
        panel_width = max(source_a.width, source_b.width)
        height_a = panel_width * source_a.height / source_a.width
        height_b = panel_width * source_b.height / source_b.width

        label_fontsize = max(16.0, 0.019 * panel_width)
        outer_margin = 0.012 * panel_width
        label_gap = 4 * label_fontsize
        bottom_gap = 2 * label_fontsize

        page_width = panel_width + 2 * outer_margin
        page_height = (
            2 * outer_margin + height_a + label_gap + height_b + bottom_gap
        )

        with fitz.open() as assembled:
            page = assembled.new_page(width=page_width, height=page_height)

            top_a = outer_margin
            bottom_a = top_a + height_a
            top_b = bottom_a + label_gap
            bottom_b = top_b + height_b

            # show_pdf_page embeds the original PDF graphics as vector content.
            page.show_pdf_page(
                fitz.Rect(outer_margin, top_a, outer_margin + panel_width, bottom_a),
                document_a, 0, keep_proportion=True,
            )
            page.show_pdf_page(
                fitz.Rect(outer_margin, top_b, outer_margin + panel_width, bottom_b),
                document_b, 0, keep_proportion=True,
            )

            for marker, baseline in (
                ("(a)", bottom_a + label_gap * 0.4),
                ("(b)", bottom_b + bottom_gap * 0.6),
            ):
                marker_width = fitz.get_text_length(
                    marker, fontname="tibo", fontsize=label_fontsize
                )
                page.insert_text(
                    ((page_width - marker_width) / 2, baseline),
                    marker,
                    fontname="tibo",  # Serif / bold, consistent with panel labels.
                    fontsize=label_fontsize,
                    color=(0, 0, 0),
                )

            # This PDF is precisely one page and contains panels (a) and (b).
            assembled.save(str(pdf_path), garbage=4, deflate=True)

        if png_path is not None:
            with fitz.open(str(pdf_path)) as combined:
                pixmap = combined[0].get_pixmap(dpi=COMBINED_PNG_DPI, alpha=False)
                pixmap.save(str(png_path))


def main():
    # ============================================================
    # 1. Original visualization configuration
    # ============================================================
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42
    rcParams['font.family'] = 'Arial'
    pd.set_option('display.max_rows', 3)

    import warnings
    warnings.filterwarnings('ignore')

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # 2. Figure 1a - source data, original per-capita aggregation and map join
    # ============================================================
    df_complaint_year = pd.read_csv(DATA_DIR / 'Dataset_311.csv', index_col=0)
    df_complaint_year['TractFIPS'] = df_complaint_year['TractFIPS'].astype(float)
    df_complaint_year['TractFIPS'] = df_complaint_year['TractFIPS'].astype(str)
    df_complaint_year['TractFIPS'] = df_complaint_year['TractFIPS'].str.zfill(13)

    complaint_cols = df_complaint_year.columns[df_complaint_year.columns.get_loc('AirQuality') : df_complaint_year.columns.get_loc('Water') + 1]
    df_complaint_year[complaint_cols] = df_complaint_year[complaint_cols].div(df_complaint_year['TotalPopulation'], axis=0)

    df_complaint = df_complaint_year.groupby('TractFIPS')[complaint_cols].sum().reset_index()
    df_complaint['TotalComplaints'] = df_complaint[complaint_cols].sum(axis=1)

    gdf_ct_ = gpd.read_file(DATA_DIR / 'SHP' / 'CT' / 'Dataset_CT.shp')
    df_complaint_sum = df_complaint[['TractFIPS','TotalComplaints']]

    gdf_ct_sum = pd.merge(gdf_ct_, df_complaint_sum, on='TractFIPS')
    gdf_ct_sum

    # ============================================================
    # 3. Figure 1a - city order
    # ============================================================
    city_list = np.unique(gdf_ct_sum['City']) 

    # ============================================================
    # 4. Figure 1a - unchanged plotting parameters and PDF export
    # ============================================================
    colors_truncated = plt.cm.RdBu_r
    new_colors = mcolors.LinearSegmentedColormap.from_list(
        'new_colors', colors_truncated(np.linspace(0.15, 0.9, 256))
    )

    fig = visualize_spatial_distribution(
        gdf=gdf_ct_sum,
        city_list=city_list,
        column='TotalComplaints',
        figsize=(20, 10),
        cmap=new_colors,
        k=10,
        linewidth=0.001,
        title_y_row1=0.885,
        title_y_row2=0.455,
    )

    # Preserve the complete original Figure 1a vector graphic in memory.
    panel_a_pdf = _to_pdf_bytes(fig)
    plt.close(fig)

    # ============================================================
    # 5. Figure 1b - original city-level category means
    # ============================================================
    city_sum = df_complaint_year.groupby('City')[complaint_cols].mean().reset_index()

    # ============================================================
    # 6. Figure 1b - unchanged plotting parameters and PDF export
    # ============================================================
    fig = visualize_thematic_structures(
        city_sum=city_sum,
        num_top_types=8,
        top_n_to_color=8,
        width=0.6,
        alpha=0.7,
    )

    # Preserve the complete original Figure 1b vector graphic in memory.
    panel_b_pdf = _to_pdf_bytes(fig)
    plt.close(fig)

    # Final submission figure: one single-page file, with both panels and
    # centrally placed (a) / (b) labels. No Figure title or caption is added.
    _save_combined_figure(
        panel_a_pdf, panel_b_pdf,
        FIGURE_DIR / 'Figure1.pdf',
        FIGURE_DIR / 'Figure1.png',
    )


if __name__ == "__main__":
    main()
