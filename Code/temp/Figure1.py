#!/usr/bin/env python3
"""Figure 1a and 1b: spatial and thematic 311 complaint patterns.

Run: python Code/Figure1.py
Project layout: Code/, Data/, Result/, Figure/.
All plotting and calculation settings are retained from the final notebook.
Only file paths and module imports are adapted (plus Figure 2 OOF source).
"""

from pathlib import Path

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

    fig.savefig(FIGURE_DIR / 'Figure1a.pdf', dpi=600, bbox_inches='tight')
    fig.savefig(FIGURE_DIR / 'Figure1a.png', dpi=600, bbox_inches='tight')

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

    fig.savefig(FIGURE_DIR / 'Figure1b.pdf', dpi=600, bbox_inches='tight')
    fig.savefig(FIGURE_DIR / 'Figure1b.png', dpi=600, bbox_inches='tight')


if __name__ == "__main__":
    main()
