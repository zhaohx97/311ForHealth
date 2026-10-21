"""Figure 1 visualization utilities (original rendering logic preserved).

Figure 1a: city-by-city Jenks natural-break maps of aggregate per-capita 311 counts.
Figure 1b: city-specific leading complaint categories based on the provided
           city-level input table. No input data or statistics are recomputed here.

This module is used by ``Revision_SpatialCV_9_Figure1.ipynb``.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from mapclassify import NaturalBreaks


def visualize_spatial_distribution(
                                  gdf,
                                  city_list,
                                  column,
                                  figsize=(16, 8),
                                  cmap='Blues',
                                  k=10,
                                  linewidth=0.1,
                                  title_y_row1=0.875,
                                  title_y_row2=0.475
                                  ):
    
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=figsize)
    axes = axes.flatten()

    row1_cbar_y_bar = 0.535   
    row2_cbar_y_bar = 0.1     

    row1_cbar_y_label = row1_cbar_y_bar - 0.021   
    row2_cbar_y_label = row2_cbar_y_bar - 0.021  

    cmap_obj = plt.get_cmap(cmap)



    for idx, city in enumerate(city_list):
        ax = axes[idx]
        gdf_city = gdf[gdf['City'] == city].copy()

        nb = NaturalBreaks(gdf_city[column], k=k)
        gdf_city["jenks_bin"] = nb.yb

        colors = cmap_obj((gdf_city["jenks_bin"] / (k - 1)))

        gdf_city.plot(
            color=colors,
            edgecolor='white',
            linewidth=linewidth,
            ax=ax
        )

        ax.axis('off')

        bbox = ax.get_position()
        title_x = bbox.x0 + bbox.width / 2
        title_y = title_y_row1 if idx < 5 else title_y_row2
        fig.text(title_x, title_y - 0.01, city, ha='center', va='bottom', fontsize=16)

        vmin = gdf_city[column].min()
        vmax = gdf_city[column].max()
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

        if idx < 5:
            cbar_y_bar = row1_cbar_y_bar
            cbar_y_label = row1_cbar_y_label
        else:
            cbar_y_bar = row2_cbar_y_bar
            cbar_y_label = row2_cbar_y_label

        cbar_width = bbox.width * 0.5
        cbar_x = bbox.x0 + (bbox.width - cbar_width) / 2

        cax = fig.add_axes([
            cbar_x,
            cbar_y_bar,
            cbar_width,
            0.0075
        ])
        cb = mpl.colorbar.ColorbarBase(
            cax,
            cmap=cmap_obj,
            norm=norm,
            orientation='horizontal'
        )

        cb.set_ticks([vmin, vmax])
        cb.set_ticklabels([
            f"{vmin:.1f}",
            f"{vmax:.1f}"
        ])
        cb.ax.tick_params(labelsize=10)

        fig.text(
            cbar_x + cbar_width / 2,
            cbar_y_label,
            '# per capita',
            ha='center',
            va='bottom',
            fontsize=10
        )

    plt.subplots_adjust(hspace=0.3, wspace=0.15)
    plt.show()

    return fig


def visualize_thematic_structures(
                                  city_sum, 
                                  num_top_types=8, top_n_to_color=8, 
                                  width=0.6, alpha=0.8
                                  ):
    
    cities = city_sum['City'].unique()

    type_rank_counter = {}
    for city in cities:
        city_data = city_sum[city_sum['City'] == city].drop(columns='City').T
        city_data.columns = ['count']
        top_types = city_data.sort_values('count', ascending=False).head(top_n_to_color).index
        for t in top_types:
            type_rank_counter[t] = type_rank_counter.get(t, 0) + 1

    colored_types = sorted(type_rank_counter, key=type_rank_counter.get, reverse=True)[:top_n_to_color]
    palette = sns.color_palette('tab20', top_n_to_color)
    color_dict = {typ: palette[i] for i, typ in enumerate(colored_types)}

    #plt.style.use('seaborn-white')
    #plt.style.use('seaborn-v0_8-white')

    n_cols = 5
    n_rows = (len(cities) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4 + 2))
    axes = axes.flatten()

    city_top_type_dict = {}



    for idx, city in enumerate(cities):
        ax = axes[idx]
        city_data = city_sum[city_sum['City'] == city].drop(columns='City').T
        city_data.columns = ['count']
        city_data = city_data.sort_values('count', ascending=False)

        top_type = city_data.index[0]
        city_top_type_dict[city] = top_type

        city_data = city_data.head(num_top_types).reset_index().rename(columns={'index': 'type'})

        bar_colors = []
        for typ in city_data['type']:
            if typ in color_dict:
                bar_colors.append(color_dict[typ])
            else:
                bar_colors.append('#dddddd')    

        ax.bar(
            city_data['type'],
            city_data['count'],
            color=bar_colors,
            edgecolor='none',
            width=width,
            alpha=alpha
        )

        ax.set_title(f'{city}', fontsize=20, y=1.05)
        ax.set_xticklabels(city_data['type'], rotation=45, ha='right', fontsize=13.5)
        ax.tick_params(axis='y', labelsize=13.5)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))

    for j in range(len(cities), len(axes)):
        fig.delaxes(axes[j])

    legend_handles = [mpatches.Patch(color=color_dict[t], label=t) for t in colored_types]
    legend_handles.append(mpatches.Patch(facecolor='#dddddd', edgecolor='none', label='Other Types'))

    fig.legend(handles=legend_handles, loc='upper center', 
               ncol=9, bbox_to_anchor=(0.5, -0.01), 
               fontsize=15, frameon=False)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.6, wspace=0.3)
    plt.show()

    return fig
