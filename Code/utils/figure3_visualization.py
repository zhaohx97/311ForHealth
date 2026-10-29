"""Figure 3 — publication-oriented visualization of spatial OOF model results.

This module changes *presentation only*. It preserves:
  • original model-comparison merges and the ACS error-reduction formula;
  • Figure 3a's 21 measure-level values;
  • Figure 3b's ranking by Hybrid minus ACS out-of-fold R²;
  • Figure 3c's complaint-only SHAP ranking and group-wise medians.

Put this file alongside Revision_SpatialCV_9_Figure3_optimized.ipynb.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib import patheffects, font_manager

try:
    from adjustText import adjust_text
except ImportError:
    adjust_text = None


# Matches the original tab20 indices [0, 2, 4, 6].
DEFAULT_DOMAIN_COLORS = {
    "Health Outcomes": "#1f77b4",
    "Health Risk Behaviors": "#ff7f0e",
    "Health Status": "#2ca02c",
    "Prevention": "#d62728",
}


def _check_columns(frame, cols, name):
    missing = set(cols) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")


def _clean_axis(ax, grid_axis="x", grid_alpha=0.14):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#555555")
    ax.spines["bottom"].set_color("#555555")
    ax.spines["left"].set_linewidth(0.75)
    ax.spines["bottom"].set_linewidth(0.75)
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, color="#8A8A8A", alpha=grid_alpha, lw=0.55)


def _repel_text_fallback(ax, labels, anchors):
    """Place 21 labels with deterministic collision-aware candidate search.

    Avoids any dependency on adjustText. Positions use display coordinates and
    never change marker values or positions.
    """
    import math
    fig = ax.figure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    bounds = ax.get_window_extent(rend).padded(-6)
    xy_px = np.array([ax.transData.transform(xy) for xy in anchors], dtype=float)
    wh = np.array([(t.get_window_extent(rend).width,
                    t.get_window_extent(rend).height) for t in labels], dtype=float)
    n = len(labels)
    density = []
    for i in range(n):
        d = np.linalg.norm(xy_px - xy_px[i], axis=1)
        density.append(int(np.count_nonzero((d < 85) & (d > 0))))
    placement_order = np.argsort(-np.asarray(density), kind="stable")
    placed = {}
    angles = [90, -90, 45, 135, -45, -135, 0, 180,
              65, 115, -65, -115, 20, 160, -20, -160]
    radii = [16, 30, 46, 64, 83, 104]
    for i in placement_order:
        w, h = wh[i]
        best_cost, best_xy = float("inf"), xy_px[i]
        for radius in radii:
            for angle in angles:
                rad = math.radians(angle)
                cx = xy_px[i, 0] + radius * math.cos(rad)
                cy = xy_px[i, 1] + radius * math.sin(rad)
                b = [cx-w/2-3, cy-h/2-2, cx+w/2+3, cy+h/2+2]
                excess = sum([max(0, bounds.x0-b[0]), max(0, b[2]-bounds.x1),
                              max(0, bounds.y0-b[1]), max(0, b[3]-bounds.y1)])
                cost = 90*excess + radius * .45
                for j, q in placed.items():
                    wj, hj = wh[j]
                    ox = max(0, min(b[2], q[0]+wj/2+3)-max(b[0],q[0]-wj/2-3))
                    oy = max(0, min(b[3], q[1]+hj/2+2)-max(b[1],q[1]-hj/2-2))
                    cost += 8 * ox * oy
                # Gently avoid covering another actual data point.
                nearby = (np.abs(xy_px[:,0]-cx)<w/2+5)&(np.abs(xy_px[:,1]-cy)<h/2+5)
                nearby[i] = False
                cost += 48 * np.count_nonzero(nearby)
                if cost < best_cost:
                    best_cost, best_xy = cost, np.array([cx,cy])
        placed[int(i)] = best_xy
        labels[i].set_position(ax.transData.inverted().transform(best_xy))
    for i, txt in enumerate(labels):
        if np.linalg.norm(placed[i]-xy_px[i]) > 22:
            ax.annotate("", xy=anchors[i], xytext=txt.get_position(),
                        arrowprops={"arrowstyle":"-","color":"0.68","lw":0.48},
                        zorder=3)

def plot_figure3_revised(
    df_complaint_metrics,
    df_demographics_metrics,
    df_hybrid_metrics,
    df_shap_feature,
    df_type,
    output_pdf=None,
    output_png=None,
    measure_col="y_variable",
    category_col="legends",
    category_order=None,
    complaint_r2_col="r2_complaint",
    acs_r2_col="r2_demographics",
    hybrid_r2_col="r2_hybrid",
    shap_feature_col="feature",
    shap_group_col="feature_group",
    shap_group_label="Complaint",
    shap_metric_col="mean_abs_shap_std_y",
    # Publication layout (the right panels use all available column width).
    figsize=(15.5, 9.2),
    width_ratios=(1.14, 1.0),
    wspace=0.35,
    right_height_ratios=(1.55, 1.0),
    right_hspace=0.40,
    # Left: preserve marker appearance/values; improve labels.
    left_scatter_size=185,
    left_scatter_linewidth=1.5,
    left_fill_alpha=0.27,
    left_label_fontsize=9.5,
    left_tick_fontsize=9.5,
    left_xlim=(0.125, 0.875),
    left_ylim=(0, 100),
    left_xlabel="Complaint-only out-of-sample R²",
    left_ylabel="Reduction in ACS out-of-sample\nsquared error (%)",
    left_box_aspect=1.0,
    # Right top.
    dumbbell_point_size=36,
    dumbbell_linewidth=1.75,
    dumbbell_line_alpha=0.65,
    acs_edgecolor="#777777",
    acs_facecolor="white",
    dumbbell_label_fontsize=8.7,
    dumbbell_tick_fontsize=9.0,
    dumbbell_xlim=None,  # Auto-enclose *all* data; avoids clipping ACS-only observations.
    dumbbell_xlabel="Spatial out-of-sample R²",
    right_top_box_aspect=None,  # None removes unnecessary square constraint.
    show_delta_text=False,
    delta_fontsize=7.5,
    # Right bottom: preserve feature ranking and original SHAP definition.
    shap_top_n=8,
    shap_range_linewidth=1.65,
    shap_range_alpha=0.55,
    shap_range_color="#A0A0A0",
    shap_category_marker_size=12,
    shap_category_marker_width=2.35,
    shap_overall_marker_size=6.7,
    shap_overall_edge_width=1.25,
    shap_overall_edgecolor="#777777",
    shap_overall_facecolor="white",
    shap_feature_fontsize=9.0,
    shap_tick_fontsize=9.0,
    shap_xlim=None,
    shap_xlabel="Median standardized |SHAP| across outcomes",
    legend_fontsize=9,
    grid_alpha=0.13,
    panel_labels=("a", "b", "c"),
):
    """Draw Figure 3, return (fig, axes, comparison, dumbbell, ranks, domain_SHAP)."""
    for frame, cols, name in (
        (df_complaint_metrics, [measure_col, complaint_r2_col], "df_complaint_metrics"),
        (df_demographics_metrics, [measure_col, acs_r2_col], "df_demographics_metrics"),
        (df_hybrid_metrics, [measure_col, hybrid_r2_col], "df_hybrid_metrics"),
        (df_type, [measure_col, category_col], "df_type"),
        (df_shap_feature, [measure_col, shap_feature_col, shap_group_col, shap_metric_col], "df_shap_feature"),
    ):
        _check_columns(frame, cols, name)

    df_type_use = df_type[[measure_col, category_col]].drop_duplicates().copy()
    if df_type_use[measure_col].duplicated().any():
        raise ValueError("The health-domain table assigns multiple domains to a health measure.")
    if category_order is None:
        categories_present = df_type_use[category_col].drop_duplicates().tolist()
        category_order = [c for c in DEFAULT_DOMAIN_COLORS if c in categories_present]
    else:
        category_order = list(category_order)
    if len(category_order) != 4 or set(category_order) != set(df_type_use[category_col]):
        raise ValueError("Expected exactly the four health domains present in df_type.")
    # Use EXACT original tab20 palette, independent of the input-row order.
    colors = {c: DEFAULT_DOMAIN_COLORS.get(c, plt.get_cmap("tab20")(2*i))
              for i, c in enumerate(category_order)}

    comp = (df_complaint_metrics[[measure_col, complaint_r2_col]]
            .merge(df_demographics_metrics[[measure_col, acs_r2_col]], on=measure_col, validate="one_to_one")
            .merge(df_hybrid_metrics[[measure_col, hybrid_r2_col]], on=measure_col, validate="one_to_one")
            .merge(df_type_use, on=measure_col, validate="one_to_one")
            .replace([np.inf, -np.inf], np.nan).dropna().copy())
    if len(comp) != 21:
        raise ValueError(f"Figure 3 expects 21 complete health measures; got {len(comp)}.")
    comp["delta_r2"] = comp[hybrid_r2_col] - comp[acs_r2_col]
    comp["acs_remaining_error"] = 1 - comp[acs_r2_col]
    if (comp["acs_remaining_error"] <= 0).any():
        raise ValueError("ACS R² must be below 1 for the error-reduction calculation.")
    # Preserve formula from the original module exactly.
    comp["acs_error_reduction_percent"] = 100 * comp["delta_r2"] / comp["acs_remaining_error"]

    feat = (df_shap_feature.loc[df_shap_feature[shap_group_col] == shap_group_label,
                                 [measure_col, shap_feature_col, shap_metric_col]]
            .merge(df_type_use, on=measure_col, validate="many_to_one")
            .replace([np.inf, -np.inf], np.nan).dropna().copy())
    if feat.empty:
        raise ValueError("No non-null complaint-feature SHAP results were found.")
    rank = (feat.groupby(shap_feature_col, as_index=False)
            .agg(overall_median_shap=(shap_metric_col, "median"))
            .sort_values("overall_median_shap", ascending=False)
            .reset_index(drop=True))
    top_features = rank[shap_feature_col].head(shap_top_n).tolist()
    domain = (feat.loc[feat[shap_feature_col].isin(top_features)]
              .groupby([shap_feature_col, category_col], as_index=False)
              .agg(domain_median_shap=(shap_metric_col, "median"))
              .merge(rank[[shap_feature_col, "overall_median_shap"]],
                     on=shap_feature_col, validate="many_to_one"))

    available_fonts = {f.name for f in font_manager.fontManager.ttflist}
    plot_font = "Arial" if "Arial" in available_fonts else "DejaVu Sans"
    with plt.rc_context({"font.family": plot_font, "font.sans-serif": [plot_font],
                         "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig = plt.figure(figsize=figsize, facecolor="white")
        fig.subplots_adjust(left=0.068, right=0.985, bottom=0.08, top=0.96)
        grid = fig.add_gridspec(1, 2, width_ratios=width_ratios, wspace=wspace)
        ax_left = fig.add_subplot(grid[0, 0])
        rgrid = grid[0, 1].subgridspec(2, 1, height_ratios=right_height_ratios,
                                       hspace=right_hspace)
        ax_top = fig.add_subplot(rgrid[0, 0])
        ax_bottom = fig.add_subplot(rgrid[1, 0])
        if left_box_aspect is not None:
            ax_left.set_box_aspect(left_box_aspect)
        # By default the right-top chart is NOT forced into a square box.
        if right_top_box_aspect is not None:
            ax_top.set_box_aspect(right_top_box_aspect)

        # (a) Scatter of CC predictive fit vs residual-error reduction.
        for cat in category_order:
            d = comp.loc[comp[category_col] == cat]
            ax_left.scatter(d[complaint_r2_col], d["acs_error_reduction_percent"],
                            s=left_scatter_size, linewidth=left_scatter_linewidth,
                            facecolor=to_rgba(colors[cat], left_fill_alpha),
                            edgecolor=colors[cat], zorder=4)
        mx = comp[complaint_r2_col].median()
        my = comp["acs_error_reduction_percent"].median()
        ax_left.axvline(mx, color="0.62", linewidth=0.9, linestyle="--", zorder=1)
        ax_left.axhline(my, color="0.62", linewidth=0.9, linestyle="--", zorder=1)
        # Set final limits BEFORE moving text. The original code adjusted text first.
        if left_xlim is None:
            mn, ma = comp[complaint_r2_col].agg(["min", "max"])
            ax_left.set_xlim(mn - 0.08, ma + 0.08)
        else:
            ax_left.set_xlim(left_xlim)
        if left_ylim is None:
            mn, ma = comp["acs_error_reduction_percent"].agg(["min", "max"])
            ax_left.set_ylim(min(0, mn - 6), max(100, ma + 6))
        else:
            ax_left.set_ylim(left_ylim)
        if ((comp[complaint_r2_col] < ax_left.get_xlim()[0]) | (comp[complaint_r2_col] > ax_left.get_xlim()[1]) |
            (comp["acs_error_reduction_percent"] < ax_left.get_ylim()[0]) |
            (comp["acs_error_reduction_percent"] > ax_left.get_ylim()[1])).any():
            raise ValueError("Left plot limits would hide observations. Adjust left_xlim / left_ylim.")
        ax_left.set_xlabel(left_xlabel, fontsize=11.2, labelpad=9)
        ax_left.set_ylabel(left_ylabel, fontsize=11.2, labelpad=9)
        ax_left.tick_params(labelsize=left_tick_fontsize, length=3)
        _clean_axis(ax_left, grid_axis="both", grid_alpha=grid_alpha)
        handles = [Line2D([0],[0], marker="o", linestyle="", markersize=7,
                          markerfacecolor=to_rgba(colors[c], left_fill_alpha),
                          markeredgecolor=colors[c], markeredgewidth=1.4, label=c)
                   for c in category_order]
        ax_left.legend(handles=handles, loc="upper left", frameon=False,
                       fontsize=legend_fontsize, labelspacing=0.38, borderaxespad=0.6)
        # Labels: position above each observation initially; avoid collisions.
        lab_txt, coords = [], []
        for i, (_, r) in enumerate(comp.iterrows()):
            x, y = float(r[complaint_r2_col]), float(r["acs_error_reduction_percent"])
            coords.append((x,y))
            offset = (0.0, (3.5 if i % 2 == 0 else -3.5))
            t = ax_left.text(x + offset[0], y + offset[1], str(r[measure_col]),
                             ha="center", va="center", fontsize=left_label_fontsize,
                             color="#252525", zorder=5)
            t.set_path_effects([patheffects.withStroke(linewidth=2.0, foreground="white")])
            lab_txt.append(t)
        if adjust_text is not None:
            adjust_text(lab_txt, x=comp[complaint_r2_col].to_numpy(),
                        y=comp["acs_error_reduction_percent"].to_numpy(),
                        ax=ax_left, expand=(1.14, 1.25), force_text=(0.55, 0.8),
                        force_static=(0.40, 0.55), force_pull=(0.012, 0.015),
                        ensure_inside_axes=True, prevent_crossings=True, iter_lim=450,
                        arrowprops={"arrowstyle": "-", "color": "0.6", "lw": 0.5, "alpha": 0.7})
        else:
            _repel_text_fallback(ax_left, lab_txt, coords)

        # (b) ACS -> Hybrid dumbbell, sorted by ΔR², descending.
        dumbbell = comp.sort_values("delta_r2", ascending=False).reset_index(drop=True).copy()
        dumbbell["_y"] = np.arange(len(dumbbell))
        for row_i, row in dumbbell.iterrows():
            y = row["_y"]
            col = colors[row[category_col]]
            if row_i % 2:
                ax_top.axhspan(y - 0.5, y + 0.5, color="#909090", alpha=0.035, lw=0, zorder=0)
            acs = row[acs_r2_col]
            hybrid = row[hybrid_r2_col]
            ax_top.plot([acs, hybrid], [y,y], color=to_rgba(col,dumbbell_line_alpha),
                        lw=dumbbell_linewidth, solid_capstyle="round", zorder=2)
            ax_top.scatter(acs, y, s=dumbbell_point_size, facecolor=acs_facecolor,
                           edgecolor=acs_edgecolor, linewidth=1.1, zorder=3)
            ax_top.scatter(hybrid, y, s=dumbbell_point_size, facecolor=col,
                           edgecolor=col, linewidth=0.9, zorder=4)
            if show_delta_text:
                ax_top.annotate(f"{row['delta_r2']:+.2f}", xy=(hybrid,y), xytext=(5,0),
                                textcoords="offset points", va="center",
                                fontsize=delta_fontsize, color="#333333")
        ax_top.set_yticks(dumbbell["_y"])
        ax_top.set_yticklabels(dumbbell[measure_col], fontsize=dumbbell_label_fontsize)
        ax_top.invert_yaxis()
        data_min = float(dumbbell[[acs_r2_col, hybrid_r2_col]].min().min())
        data_max = float(dumbbell[[acs_r2_col, hybrid_r2_col]].max().max())
        if dumbbell_xlim is None:
            ax_top.set_xlim(min(0, data_min - 0.04), max(1.0, data_max + 0.04))
        else:
            ax_top.set_xlim(dumbbell_xlim)
            if data_min < dumbbell_xlim[0] or data_max > dumbbell_xlim[1]:
                warnings.warn("Supplied dumbbell_xlim hides some model results!", RuntimeWarning)
        ax_top.set_xlabel(dumbbell_xlabel, fontsize=10.6, labelpad=7)
        ax_top.tick_params(axis="y", length=0, pad=5)
        ax_top.tick_params(axis="x", labelsize=dumbbell_tick_fontsize)
        _clean_axis(ax_top, grid_axis="x", grid_alpha=grid_alpha)
        ax_top.spines["left"].set_visible(False)
        model_handles = [
            Line2D([0],[0], marker="o", linestyle="", markersize=6, markerfacecolor="white",
                   markeredgecolor=acs_edgecolor, label="ACS only"),
            Line2D([0],[0], marker="o", linestyle="", markersize=6,
                   markerfacecolor="#444444", markeredgecolor="#444444", label="Hybrid"),
        ]
        # Outside the data area, in the extra inter-panel space.
        ax_top.legend(
    handles=model_handles,
    loc="lower left",
    frameon=False,
    fontsize=legend_fontsize,
    ncol=2,
    handletextpad=0.4,
    columnspacing=1.3,
    borderaxespad=0.6,
)

        # (c) Top complaint SHAP: same values and ranking as source code.
        ymax_vals = []
        for i, feat_name in enumerate(top_features):
            d = domain.loc[domain[shap_feature_col] == feat_name]
            vals = d["domain_median_shap"].to_numpy(dtype=float)
            ymax_vals.extend(vals.tolist())
            med = float(d["overall_median_shap"].iloc[0])
            ymax_vals.append(med)
            if i % 2:
                ax_bottom.axhspan(i - 0.5, i + 0.5, color="#909090", alpha=0.035, lw=0, zorder=0)
            ax_bottom.plot([vals.min(),vals.max()], [i,i], color=shap_range_color,
                           lw=shap_range_linewidth, alpha=shap_range_alpha,
                           solid_capstyle="round", zorder=1)
            ax_bottom.plot(med, i, marker="s", markersize=shap_overall_marker_size,
                           markerfacecolor=shap_overall_facecolor,
                           markeredgecolor=shap_overall_edgecolor,
                           markeredgewidth=shap_overall_edge_width,
                           linestyle="None", zorder=3)
            for cat in category_order:
                dcat = d.loc[d[category_col] == cat]
                if not dcat.empty:
                    ax_bottom.plot(float(dcat["domain_median_shap"].iloc[0]), i,
                                   marker="|", markersize=shap_category_marker_size,
                                   markeredgewidth=shap_category_marker_width,
                                   color=colors[cat], linestyle="None", zorder=4)
        ax_bottom.set_yticks(np.arange(len(top_features)))
        ax_bottom.set_yticklabels(top_features, fontsize=shap_feature_fontsize)
        ax_bottom.invert_yaxis()
        max_shap = max(ymax_vals)
        if shap_xlim is None:
            ax_bottom.set_xlim(0, max_shap * 1.09)
        else:
            ax_bottom.set_xlim(shap_xlim)
            if (min(ymax_vals) < shap_xlim[0]) or (max_shap > shap_xlim[1]):
                warnings.warn("Supplied shap_xlim hides some plotted values!", RuntimeWarning)
        ax_bottom.set_xlabel(shap_xlabel, fontsize=10.6, labelpad=7)
        ax_bottom.tick_params(axis="y", length=0, pad=5)
        ax_bottom.tick_params(axis="x", labelsize=shap_tick_fontsize)
        _clean_axis(ax_bottom, grid_axis="x", grid_alpha=grid_alpha)
        ax_bottom.spines["left"].set_visible(False)
        shap_handles = [Line2D([0],[0], marker="|", linestyle="", color=colors[c],
                               markersize=11, markeredgewidth=2.0, label=c) for c in category_order]
        shap_handles.append(Line2D([0],[0], marker="s", linestyle="", markersize=6.0,
                                   markerfacecolor="white", markeredgecolor=shap_overall_edgecolor,
                                   markeredgewidth=shap_overall_edge_width, label="Overall median"))
        ax_bottom.legend(handles=shap_handles, loc="lower right", frameon=False,
                         fontsize=legend_fontsize-0.4, handletextpad=0.25,
                         labelspacing=0.3, borderaxespad=0.4)

        # Panel labels, positioned consistently above the three chart areas.
        if panel_labels is not None:
            if len(panel_labels) != 3:
                raise ValueError("panel_labels must contain 3 labels, or be None.")
            for ax, letter in zip((ax_left, ax_top, ax_bottom), panel_labels):
                ax.text(-0.01, 1.035, f"({letter})", transform=ax.transAxes,
                        fontsize=12, fontweight="bold", ha="right", va="bottom",
                        clip_on=False, color="#222222")

        if output_pdf:
            p = Path(output_pdf)
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(p, format="pdf", bbox_inches="tight", pad_inches=0.09)
        if output_png:
            p = Path(output_png)
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(p, format="png", dpi=400, bbox_inches="tight", pad_inches=0.09)

    axes = {"left": ax_left, "right_top": ax_top, "right_bottom": ax_bottom}
    return fig, axes, comp, dumbbell, rank, domain
