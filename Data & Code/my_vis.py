import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import gridspec
from matplotlib.patches import Patch
from matplotlib.patches import ConnectionPatch
import matplotlib.patches as mpatches
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter

from matplotlib import rcParams
rcParams['font.family'] = 'Arial'

import seaborn as sns

from mapclassify import NaturalBreaks
import textwrap

pd.set_option('display.max_rows', 3)

import warnings
warnings.filterwarnings('ignore')





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
        gdf_city['jenks_bin'] = nb.yb

        colors = cmap_obj((gdf_city['jenks_bin'] / (k - 1)))

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
                    f'{vmin:.1f}',
                    f'{vmax:.1f}'
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
                                num_top_types=8, 
                                top_n_to_color=8, 
                                width=0.6, 
                                alpha=0.8
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
    plt.style.use('seaborn-v0_8-white')

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

    fig.legend(handles=legend_handles, loc='upper center', ncol=9, bbox_to_anchor=(0.5, -0.01), fontsize=15)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.6, wspace=0.3)
    plt.show()

    return fig





def visualize_complaints_predictions(
                                    df_pred, 
                                    df_metric
                                    ):

    unique_vars = df_pred['y_variable'].unique()

    n_cols = 7
    n_rows = int(np.ceil(len(unique_vars) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4 + 2))
    axes = axes.flatten()

    city_codes, city_labels = pd.factorize(df_pred['City'])
    cmap = plt.get_cmap('tab20')
    city_color_dict = dict(zip(city_labels, cmap.colors[:len(city_labels)]))
    sorted_city_items = sorted(city_color_dict.items(), key=lambda x: x[0])

    for i, var in enumerate(unique_vars):
        subset = df_pred[df_pred['y_variable'] == var]
        ax = axes[i]

        non_ny = subset[subset['City'] != 'New York']
        ny = subset[subset['City'] == 'New York']
        non_ny_colors = non_ny['City'].map(city_color_dict)
        ny_colors = ny['City'].map(city_color_dict)

        ax.scatter(non_ny['y_true'], non_ny['y_pred'], s=2, alpha=1, color=non_ny_colors)
        ax.scatter(ny['y_true'], ny['y_pred'], s=2, alpha=1, color=ny_colors)

        min_val = min(subset['y_true'].min(), subset['y_pred'].min())
        max_val = max(subset['y_true'].max(), subset['y_pred'].max())
        ax.plot([min_val, max_val], [min_val, max_val], color='k', linestyle='--', linewidth=1)

        ax.set_title(f'{var}', fontsize=20)
        ax.set_xlabel('True', fontsize=12)
        ax.set_ylabel('Predicted', fontsize=12)
        ax.tick_params(axis='both', labelsize=10)

        for spine_name in ['top', 'right']:
            ax.spines[spine_name].set_visible(False)
        for spine_name in ['bottom', 'left']:
            ax.spines[spine_name].set_visible(True)
            ax.spines[spine_name].set_linewidth(1.5)

        ax.grid(False)

        adjusted_r2 = df_metric.loc[df_metric['y_variable'] == var, 'adjusted_r2'].values[0]
        rmse = df_metric.loc[df_metric['y_variable'] == var, 'rmse'].values[0]
        mae = df_metric.loc[df_metric['y_variable'] == var, 'mae'].values[0]

        ax.text(0.95, 0.17, f'Adj.R$^2$: {adjusted_r2:.2f}', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=15, color='black')
        ax.text(0.95, 0.10, f'RMSE: {rmse:.2f}', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=15, color='black')
        ax.text(0.95, 0.03, f'MAE: {mae:.2f}', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=15, color='black')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=color, markersize=15, label=city)
        for city, color in sorted_city_items
    ]
    fig.legend(handles=handles, loc='lower center', ncol=10,
               fontsize=18, frameon=False, bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.show()

    return fig





def visualize_model_comparison(
                            df_comparison,
                            figsize=(20, 10),
                            width=0.5,
                            color_complaint='#8CAAC4',
                            color_demographics='#DDDDDD',
                            color_improvement='#BBBBBB'
                            ):
    
    legend_order = df_comparison['legends'].drop_duplicates().tolist()

    df_sorted = pd.concat([
                        group.sort_values(by='adjusted_r2_complaint', ascending=False)
                        for legend in legend_order
                        for _, group in df_comparison[df_comparison['legends'] == legend].groupby('legends')
                        ], 
                        ignore_index=True)

    x = np.arange(len(df_sorted['y_variable']))

    fig, ax = plt.subplots(figsize=figsize)

    ax.bar(x - width / 2, df_sorted['adjusted_r2_complaint'], width, color=color_complaint)
    ax.bar(x + width / 2, df_sorted['adjusted_r2_demographics'], width, color=color_demographics)
    ax.bar(x + width / 2, df_sorted['r2_improvement'], width, bottom=df_sorted['adjusted_r2_demographics'], color=color_improvement)

    ax.set_ylabel('Adjusted R²', fontsize=16)
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(df_sorted['y_variable'], rotation=45, ha='right', fontsize=13)
    ax.tick_params(axis='both', labelsize=13)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    previous_legend = df_sorted['legends'].iloc[0]
    start_index = 0



    for i in range(1, len(df_sorted)):
        current_legend = df_sorted['legends'].iloc[i]
        
        if current_legend != previous_legend:

            ax.axvline(x = i - 0.5, color='black', linewidth=0.75, linestyle=(0, (8, 4)))
            
            end_index = i - 1
            mid_x = (start_index + end_index) / 2
            
            wrapped_text = textwrap.fill(previous_legend, width=10) 
            ax.text(mid_x, 0.99, wrapped_text, ha='center', va='top', fontsize=20)
            
            start_index = i
            previous_legend = current_legend

    end_index = len(df_sorted) - 1
    mid_x = (start_index + end_index) / 2

    wrapped_text = textwrap.fill(previous_legend, width=10)
    ax.text(mid_x, 0.99, wrapped_text, ha='center', va='top', fontsize=20)

    legend_elements = [
                    Patch(facecolor=color_complaint, edgecolor=color_complaint, label='CC Model'),
                    Patch(facecolor=color_demographics, edgecolor=color_demographics, label='DF Model'),
                    Patch(facecolor=color_improvement, edgecolor=color_improvement, label='Hybrid Model over DF Model')
                    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.15),
              ncol=3, frameon=False, fontsize=16)

    plt.tight_layout()
    plt.show()

    return fig





def visualize_shap_ratio_lollipop(
                                df_shap_, 
                                measure_list,
                                figsize=(10, 10), 
                                alpha=0.8,
                                ):

    df_shap = df_shap_.copy()
    demographic_features = ['TotalPopulation', 'PercentMale', 'PercentElderly', 'PercentWhite']
    
    df_shap['feature_group'] = df_shap['feature'].apply(
        lambda x: 'Demographic' if x in demographic_features else 'Complaint'
    )

    df_group = df_shap.groupby(['y_variable', 'feature_group'])['shap_ratio'].sum().reset_index()
    df_group_wide = df_group.pivot(index='y_variable', columns='feature_group', values='shap_ratio').fillna(0)
    #df_group_wide = df_group_wide.sort_values(by='y_variable')
    df_group_wide = df_group_wide.reindex(measure_list)

    x_labels = df_group_wide.index
    complaint_vals = df_group_wide['Complaint']
    demo_vals = df_group_wide['Demographic']

    fig, ax = plt.subplots(figsize=figsize)
    x_pos = list(range(len(x_labels)))

    min_marker_size = 30
    max_marker_size = 55

    c_min, c_max = complaint_vals.min(), complaint_vals.max()
    complaint_sizes = min_marker_size + (complaint_vals - c_min) / max(c_max - c_min, 1e-6) * (max_marker_size - min_marker_size)



    for i, (x, val, size) in enumerate(zip(x_pos, complaint_vals, complaint_sizes)):
        label = 'Complaint types' if i == 0 else None
        ax.vlines(x, 0, -val, color='#8CAAC4', linewidth=5, alpha=alpha, label=label)

        ax.plot(x, -val, 'o',
                markerfacecolor='white',
                markeredgecolor='#8CAAC4',
                alpha=1, markersize=size, markeredgewidth=2)

        ax.text(x, -val+0.04, f'{val:.2f}',
                va='top', ha='center',
                fontsize=11, color='k', alpha=1)

    for i, (x, c_val, d_val) in enumerate(zip(x_pos, complaint_vals, demo_vals)):
        label = 'Demographic features' if i == 0 else None
        ax.vlines(x, -c_val, -(c_val + d_val),
                  color='#BBBBBB', linewidth=2.5, alpha=alpha, label=label)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, ha='center', fontsize=13, rotation=90)
    ax.xaxis.set_ticks_position('top')
    ax.tick_params(axis='x', top=True, labeltop=True, bottom=False, labelbottom=False)

    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)

    max_total = (complaint_vals + demo_vals).max()
    bottom_line_y = -max_total
    ax.axhline(y=bottom_line_y, color='black', 
               #linestyle='--', 
               linewidth=0.25)

    group_bins = [0, 11, 15, 17, len(x_labels)] 
    n_groups = len(group_bins) - 1
    x_min, x_max = x_pos[2], x_pos[-3]
    group_refs = np.linspace(x_min, x_max, n_groups)

    for grp_idx in range(n_groups):
        start, end = group_bins[grp_idx], group_bins[grp_idx+1]
        x_center = group_refs[grp_idx]

        for x, c_val, d_val in zip(x_pos[start:end], complaint_vals[start:end], demo_vals[start:end]):
            if x < x_center:
                rad = -0.075
            elif x > x_center:
                rad = 0.075
            else:
                rad = 0

            con = ConnectionPatch(
                xyA=(x_center, bottom_line_y-0.2), 
                xyB=(x, bottom_line_y),         
                coordsA='data', coordsB='data',
                axesA=ax, axesB=ax,
                color='gray', linestyle='--',
                linewidth=0.5, alpha=0.75,
                arrowstyle='-',
                connectionstyle=f'arc3,rad={rad}'
            )
            ax.add_artist(con)

    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5)
    ax.set_ylim(bottom_line_y-0.225, 0)
    ax.legend(loc='lower right', frameon=False, 
              bbox_to_anchor=(1, 0.02), 
              fontsize=12)

    plt.tight_layout()
    plt.close()

    return fig





def visualize_top_feature(
                          df_shap_, 
                          df_healthtype, 
                          figsize=(10, 10), 
                          top_k=8, 
                          cmap_name='Set3', 
                          ncol=6, 
                          ): 

    demographic_features = ['TotalPopulation', 'PercentMale', 'PercentElderly', 'PercentWhite']
    demographic_colors = ['#d9d9d9', '#636363', '#bdbdbd', '#969696']
    demographic_color_map = dict(zip(demographic_features, demographic_colors))

    df_shap_wide = df_shap_.pivot(index='y_variable', columns='feature', values='shap_ratio').fillna(0).reset_index()
    df_shap_wide_ = pd.merge(df_shap_wide, df_healthtype, on='y_variable')

    feature_cols = df_shap_wide_.columns.drop(['y_variable', 'category'])
    df_category_feature_mean = df_shap_wide_.groupby('category')[feature_cols].mean().reset_index()

    value_cols = df_category_feature_mean.columns.drop('category')
    categories_to_plot = df_category_feature_mean['category'].tolist()

    selected_features = set()
    for category in categories_to_plot[:4]:
        row = df_category_feature_mean[df_category_feature_mean['category'] == category].iloc[0]
        top_feats = row[value_cols].sort_values(ascending=False).head(top_k).index
        selected_features.update(top_feats)
    final_selected_features = sorted(selected_features)

    non_demo_features = [f for f in final_selected_features if f not in demographic_features]
    tab_cmap = plt.cm.get_cmap(cmap_name, len(non_demo_features))
    non_demo_colors = [tab_cmap(i) for i in range(len(non_demo_features))]
    non_demo_color_map = dict(zip(non_demo_features, non_demo_colors))

    color_map = {**demographic_color_map, **non_demo_color_map}



    fig, axs = plt.subplots(1, 4, figsize=figsize)
    axs = axs.flatten()

    for i, category in enumerate(categories_to_plot[:4]):
        row = df_category_feature_mean[df_category_feature_mean['category'] == category].iloc[0]
        features = value_cols.tolist()
        values = row[features].values

        sorted_feats_vals = sorted(zip(features, values), key=lambda x: x[1], reverse=True)
        top_feats_vals = sorted_feats_vals[:top_k]
        other_feats_vals = sorted_feats_vals[top_k:]
        other_sum = sum(val for _, val in other_feats_vals)

        sizes = [val for _, val in top_feats_vals] + [other_sum]
        colors = [color_map.get(feat, '#f9f9f9') if feat in final_selected_features else '#f9f9f9'
                  for feat, _ in top_feats_vals] + ['#f9f9f9']

        wedgeprops = dict(linewidth=0.6, edgecolor='white', alpha=0.85)
        wedges, texts, autotexts = axs[i].pie(
                                            sizes, labels=None, colors=colors, startangle=0, counterclock=False,
                                            wedgeprops=wedgeprops,
                                            autopct='%1.1f%%', pctdistance=0.85, textprops={'fontsize': 10, 'color': 'black'}
                                             )

        outer_circle = plt.Circle((0, 0), 1.0, color='black', fill=False, linewidth=0.75)
        axs[i].add_artist(outer_circle)

        centre_circle = plt.Circle((0, 0), 0.7, fc='white', ec='black', linewidth=0.75)
        axs[i].add_artist(centre_circle)

        category_text = category
        if category == 'Health Risk Behaviors':
            parts = category.split(' ')
            category_text = parts[0] + ' ' + parts[1] + '\n' + ' '.join(parts[2:])

        axs[i].text(0, 0, category_text, ha='center', va='center', fontsize=16, fontweight='bold', color='black')

        axs[i].set_aspect('equal')
        axs[i].set_xticks([])
        axs[i].set_yticks([])

    legend_handles = []
    for feat in final_selected_features:
        color = color_map.get(feat, '#ffffff')
        patch = mpatches.Patch(color=color, label=feat, edgecolor='black')
        legend_handles.append(patch)
    legend_handles.append(mpatches.Patch(facecolor='white', edgecolor='black',
                                         label=f'Other {len(value_cols)-top_k} features'))

    fig.legend(handles=legend_handles, loc='upper center', 
               ncol=9, bbox_to_anchor=(0.5, -0.01), 
               fontsize=15, frameon=False)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.close()

    return fig





def visualize_year_robustness(
                             df_metric_all, 
                             xy_left, 
                             xy_right, 
                             s=250
                             ):

    sns.set(style='white', font_scale=1.2)

    palette = sns.color_palette('tab20', 20)
    custom_color = ['#FF6347']
    palette = palette + custom_color

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    sns.scatterplot(
        data=df_metric_all,
        x=xy_left[0],
        y=xy_left[1],
        hue='Health Measure',
        style='Year',
        palette=palette,
        s=s,
        ax=axes[0],
        legend=False
    )

    axes[0].plot([0.2, 0.9], [0.2, 0.9], color='black', dashes=(6, 3), linewidth=0.75)
    axes[0].set_xlim(0.15, 0.95)
    axes[0].set_ylim(0.15, 0.95)
    axes[0].set_xlabel('DF Model', fontsize=15)
    axes[0].set_ylabel('CC Model', fontsize=15)
    axes[0].set_aspect('equal', adjustable='box')
    axes[0].tick_params(axis='both', labelsize=12)
    axes[0].grid(linewidth = 0.2, zorder = 0)
    axes[0].set_title('Adjusted R² Comparison', fontsize=18, y=1.015)

    sns.scatterplot(
        data=df_metric_all,
        x=xy_right[0],
        y=xy_right[1],
        hue='Health Measure',
        style='Year',
        palette=palette,
        s=s,
        ax=axes[1],
        legend='brief',
        zorder = 1
    )

    axes[1].plot([0, 7], [0, 7], color='black', linestyle='--', dashes=(6, 3), linewidth=0.75)
    axes[1].set_xlim(-0.5, 7.5)
    axes[1].set_ylim(-0.5, 7.5)
    axes[1].set_xlabel('DF Model', fontsize=15)
    axes[1].set_ylabel('CC Model', fontsize=15)
    axes[1].set_aspect('equal', adjustable='box')
    axes[1].set_xticks(np.arange(0, 8, 1))
    axes[1].set_yticks(np.arange(0, 8, 1))
    axes[1].tick_params(axis='both', labelsize=12)
    axes[1].grid(linewidth = 0.2, zorder = 0)
    axes[1].set_title('RMSE Comparison', fontsize=18, y=1.015)

    legend = axes[1].legend(
        loc='center left',
        bbox_to_anchor=(-0.425, 0.5),
        frameon=False,
        fontsize=11
    )
    axes[1].add_artist(legend)

    plt.subplots_adjust(wspace=0.5, bottom=0.15)
    plt.show()

    return fig





def plot_metric_by_city(
                        r2_by_city, 
                        rmse_by_city, 
                        cities=None, 
                        figsize=(20, 10)
                        ):

    df_long_rmse = rmse_by_city.melt(
        id_vars=['Health Measure', 'City'],
        value_vars=['DF Model', 'CC Model', 'Hybrid Model'],
        var_name='model_type',
        value_name='metric_value'
    )
    df_long_rmse['Metric'] = 'RMSE'

    df_long_r2 = r2_by_city.melt(
        id_vars=['Health Measure', 'City'],
        value_vars=['DF Model', 'CC Model', 'Hybrid Model'],
        var_name='model_type',
        value_name='metric_value'
    )
    df_long_r2['Metric'] = 'R²'

    df_long_all = pd.concat([df_long_rmse, df_long_r2], ignore_index=True)

    model_name_map = {
        'DF Model': 'DF\nModel',
        'CC Model': 'CC\nModel',
        'Hybrid Model': 'Hybrid\nModel'
    }
    df_long_all['Model Display'] = df_long_all['model_type'].map(model_name_map)

    color_dict = {
        'CC\nModel': '#8CAAC4',
        'DF\nModel': '#DDDDDD',
        'Hybrid\nModel': '#BBBBBB'
    }

    if cities is None:
        cities = df_long_all['City'].unique()[:10]



    fig = plt.figure(figsize=figsize)
    outer = gridspec.GridSpec(2, 5, wspace=0.4, hspace=0.3)

    for i, city in enumerate(cities):
        inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[i], height_ratios=[1, 1], hspace=0.1)
        
        for j, metric in enumerate(['R²', 'RMSE']):
            ax = plt.Subplot(fig, inner[j])
            fig.add_subplot(ax)

            data = df_long_all[(df_long_all['City'] == city) & (df_long_all['Metric'] == metric)]

            sns.boxplot(
                    data=data,
                    x='Model Display',
                    y='metric_value',
                    order=['CC\nModel', 'DF\nModel', 'Hybrid\nModel'],
                    palette=color_dict,
                    ax=ax,
                    width=0.5,
                    linewidth=1,
                    fliersize=5,
                    flierprops=dict(marker='o', markersize=5, markerfacecolor='none',
                                    markeredgecolor='black', markeredgewidth=0.5, zorder=2),
                    boxprops=dict(linewidth=1, edgecolor='black', zorder=2),
                    whiskerprops=dict(linewidth=0.5, color='black', zorder=2),
                    capprops=dict(linewidth=1, color='black', zorder=2),
                    medianprops=dict(linewidth=0.5, color='black', zorder=2),
                    zorder=2
                    )

            if j == 0:
                ax.set_title(f'{city}', fontsize=15, y=1.075)

            ax.set_ylabel(metric, fontsize=10)

            if metric == 'R²':
                ax.set_xlabel('')
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('')

            ax.set_xlim(-0.95, 2.95)
            ax.grid(linewidth=0.3, zorder=0)
            ax.tick_params(axis='both', labelsize=8)

            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

            for spine in ax.spines.values():
                spine.set_linewidth(1)

            ymin = data['metric_value'].min()
            ymax = data['metric_value'].max()
            margin = (ymax - ymin) * 0.1 if ymax > ymin else 0.05

            if metric == 'R²' and city.lower() in ['boston', 'memphis', 'san francisco']:
                if city.lower() == 'san francisco':
                    ymin_fixed = -1.0
                elif city.lower() == 'boston':
                    ymin_fixed = -0.2
                else:  
                    ymin_fixed = -0.5
                ymax_fixed = min(ymax + margin, 1.0)
                ax.set_ylim(ymin_fixed, ymax_fixed)
            else:
                ax.set_ylim(ymin - margin, ymax + margin)

    plt.tight_layout()
    plt.show()
    
    return fig





def visualize_architecture_robustness(
                                      df_fusion, 
                                      models_to_show=['LGB', 'XGB', 'RF', 'LR'],
                                      linewidth=1.5, 
                                      markersize=5, 
                                      hspace=0.5, 
                                      wspace=0.3
                                      ):    

    y_variables = df_fusion['y_variable'].tolist()

    all_r2_cols = {
        'LGB': {
            'complaint': 'adjusted_r2_complaint_LGB',
            'demographics': 'adjusted_r2_demographics_LGB',
            'hybrid': 'adjusted_r2_hybrid_LGB'
        },
        'XGB': {
            'complaint': 'adjusted_r2_complaint_XGB',
            'demographics': 'adjusted_r2_demographics_XGB',
            'hybrid': 'adjusted_r2_hybrid_XGB'
        },
        'RF': {
            'complaint': 'adjusted_r2_complaint_RF',
            'demographics': 'adjusted_r2_demographics_RF',
            'hybrid': 'adjusted_r2_hybrid_RF'
        },
        'LR': {
            'complaint': 'adjusted_r2_complaint_LR',
            'demographics': 'adjusted_r2_demographics_LR',
            'hybrid': 'adjusted_r2_hybrid_LR'
        }
    }

    model_order = ['complaint', 'demographics', 'hybrid']
    model_labels = ['CC\nModel', 'DF\nModel', 'Hybrid\nModel']
    x_positions = np.arange(len(model_order))

    custom_colors = {
            'LGB': '#8CAAC4',
            'XGB': '#BBBBBB',
            'RF': '#DDDDDD',
            'LR': 'k'
            }

    custom_linestyles = {
            'LGB': '-',
            'XGB': '-',
            'RF': '-',
            'LR': '--'
            }



    fig, axes = plt.subplots(3, 7, figsize=(20, 10))
    axes = axes.flatten()

    bar_width = 0.185
    bar_offsets = {
        'LGB': -bar_width,
        'XGB': 0,
        'RF': bar_width
    }

    for i, y_var in enumerate(y_variables):
        ax = axes[i]

        for model_key in models_to_show:
            r2_cols = all_r2_cols.get(model_key)
            if r2_cols is None:
                continue

            r2_values = [df_fusion.loc[i, r2_cols[m]] for m in model_order]
            color = custom_colors.get(model_key, 'black')
            linestyle = custom_linestyles.get(model_key, '-')

            if model_key in ['LGB', 'XGB', 'RF']:
                offset = bar_offsets[model_key]
                ax.bar(x_positions + offset, r2_values, width=bar_width, color=color,
                       label=model_key, zorder=2, edgecolor='white', linewidth=0.5)

            elif model_key == 'LR':
                ax.plot(x_positions, r2_values, color=color, linestyle=linestyle, dashes=(10, 5),
                        linewidth=linewidth, label='Linear Regression')
                for j, val in enumerate(r2_values):
                    ax.plot(x_positions[j], val, marker='o', markersize=markersize,
                            markerfacecolor='none', markeredgecolor=color, zorder=2)

        ax.set_title(y_var, fontsize=12)
        ax.set_xlim(-0.75, len(model_order) - 0.25)
        ax.set_ylim(0, 1)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(model_labels, fontsize=8)
        ax.tick_params(labelsize=8)

        if i % 7 == 0:
            ax.set_ylabel('Adjusted R²', fontsize=10)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    for j in range(len(y_variables), len(axes)):
        fig.delaxes(axes[j])

    custom_legend = [
                Patch(facecolor=custom_colors['LGB'], edgecolor='white', label='LightGBM'),
                Patch(facecolor=custom_colors['XGB'], edgecolor='white', label='XGBoost'),
                Patch(facecolor=custom_colors['RF'], edgecolor='white', label='Random Forest'),
                Line2D([0], [0], color=custom_colors['LR'], linestyle='--', lw=2,
                    marker='o', markerfacecolor='none', markeredgecolor=custom_colors['LR'],
                    markersize=6, label='Linear Regression')
                ]
    
    fig.legend(handles=custom_legend, loc='lower center', ncol=4, fontsize=12, frameon=False)



    plt.subplots_adjust(hspace=hspace, wspace=wspace)
    plt.show()

    return fig
