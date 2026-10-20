# 311 for Health — Reproducibility Guide

Companion repository for *Emergent Behavioral Signals in Civic Feedback Decode Neighborhood Health Phenotypes*. The study combines 21,211,941 service requests from ten US cities (2019–2022), 27 standardized complaint categories, eight ACS variables, and 21 CDC PLACES health measures across 5,252 census tracts. The primary analysis pairs 2022 complaints with the 2023 PLACES release.

## 1. Repository contents and reproducing the released figures

```text
311ForHealth/
├── README.md
├── Code/
│   ├── Figure1.py ... Figure4.py
│   └── utils/figure1_visualization.py ... figure4_visualization.py
├── Data/       # Processed figure inputs and tract geometries
├── Result/     # Precomputed model predictions, metrics, and summaries
└── Figure/     # Figure outputs
```

From the repository root, run:

```bash
python Code/Figure1.py
python Code/Figure2.py
python Code/Figure3.py
python Code/Figure4.py
```

The scripts read inputs from `Data/` and **`Result/`** and save figures to `Figure/`. Key inputs are:

| Script | Key files |
|---|---|
| `Figure1.py` | `Data/Dataset_311.csv`; `Data/SHP/CT/Dataset_CT.shp` (with shapefile companion files) |
| `Figure2.py` | `Result/Predictions_complaint_2022for2023.csv`, `Result/Metrics_complaint_2022for2023.csv`; `Data/Dataset_healthtype.csv` |
| `Figure3.py` | `Result/Metrics_{complaint,demographics,hybrid}_2022for2023.csv`, `Result/SHAP_feature_summary_hybrid_2022for2023.csv`; `Data/Dataset_healthtype.csv` |
| `Figure4.py` | `Result/PriorSensitivity_2022for2023_Comparison.csv`, `TemporalRobustness_Comparison.csv`, `ReportingAdjustment.csv`, `CityStratified_R2.csv`, `CityStratified_R2_Summary.csv`, `CityStratified_R2_TukeyOutliers.csv` |

Figure generation uses Python packages including `pandas`, `numpy`, `matplotlib`, `geopandas`, `seaborn`, and `mapclassify`, as applicable to each script.

## 2. Spatial cross-validation

The analysis uses a **fixed five-fold spatial-block CV** with the following settings:

| Setting | Value |
|---|---|
| Spatial units | Census tract polygons |
| Projected CRS | `EPSG:5070` |
| Clustering coordinates | Projected tract centroids |
| Clustering scope / estimator | Within each city; `sklearn.cluster.KMeans` |
| `n_clusters` | `5` per city |
| `random_state` | `42` |
| `n_init` | `50` |
| Global fold IDs | `0–4` |
| Fold assignment | One local cluster per city per global fold; assignments balance total tract counts |
| Saved assignments | `SpatialFoldAssignments.csv` (fixed across CC, ACS, Hybrid, SHAP, and robustness analyses) |
| Training / evaluation | Four folds for training; one held-out fold for prediction |

## 3. Model specification and metrics

The three feature sets are **CC** (27 category-specific complaint counts divided by tract population), **ACS** (eight variables), and **Hybrid** (CC + ACS). The ACS variables are `TotalPopulation`, `PercentMale`, `PercentElderly`, `PercentWhite`, `PercentEducation`, `PercentUnemployed`, `PercentPoverty`, and `MedianIncome`. Data are aligned by standardized 11-character `TractFIPS`; rows with missing target values are excluded for the corresponding health measure.

For each of the 21 outcomes, the primary model uses these fixed settings:

| Setting | Value |
|---|---|
| Estimator | `lightgbm.LGBMRegressor` |
| `n_estimators` | `100` |
| `random_state` | `42` |
| `verbosity` | `-1` |
| `n_jobs` | `-1` |
| Other parameters | Library defaults |
| Cross-validation | Five fixed spatial folds; four for training, one held out |
| Hyperparameter selection | No fold-specific search |

Predictions from all five held-out folds are pooled before computing **OOF R²** (`1 − SSE/SST`), **MAE**, and **RMSE**. The incremental performance of complaint features is `R²(Hybrid) − R²(ACS)`.


## 4. Additional algorithms

Algorithmic sensitivity compares the complaint-only model under the same input features and frozen spatial folds. Alongside the primary LightGBM OOF predictions, the analysis fits:

| Algorithm | Explicit settings |
|---|---|
| **XGBoost** | `n_estimators=100`, `random_state=42`, `n_jobs=-1`, `objective="reg:squarederror"`, `tree_method="hist"` |
| **Random Forest** | `n_estimators=100`, `random_state=42`, `n_jobs=-1` |

Other parameters use library defaults. The LightGBM comparison series uses the saved primary complaint-model OOF predictions (`LGB_MODE="reuse_primary"`). Performance is evaluated from pooled held-out predictions for each of the 21 outcomes (Supplementary Figure S2).

## 5. LLM classification and validation sampling

The workflow uses **GLM-4-Long** (`temperature=0.1`) to map **1,685 distinct complaint types** into 30 predefined categories plus an `Others` option; 27 categories are retained for analysis. Manual validation samples **300 types (17.8%)**, stratified by category. Two annotators independently label the selected types, followed by review to establish reference labels (see Supplementary Figure S3).

**Category coverage (sampled/total):**

Abandoned Vehicles (10/25); Animals (10/79); Air Quality (8/8); Building (10/106); Bulky Items (10/22); Code Enforcement (10/25); Construction (10/43); Flooding (10/29).  
Fire (10/27); Garbage (10/80); Graffiti (10/19); Hazardous Materials (7/7); Homelessness (9/9); Household (6/6); Illegal Dumping (10/17); Noise (10/24).  
Parking (10/55); Parks (10/26); Recycling (10/28); Rodents/Pests (10/19); Stormwater Drainage (10/33); Street & Sidewalk Conditions (10/155); Streetlights (10/45); Sewage (10/19).  
Snow/Ice Removal (10/16); Trees (10/41); Traffic Signals (10/39); Transportation (10/92); Unsanitary Conditions (10/10); Water (10/35); uncategorized (10/546).

**City coverage (sampled/total; coverage):**

Austin (32/246; 13.01%); Baltimore (69/342; 20.18%); Boston (41/180; 22.78%); Chicago (21/106; 19.81%); Los Angeles (2/12; 16.67%).  
Memphis (34/198; 17.17%); New York (55/264; 20.83%); Philadelphia (11/72; 15.28%); San Diego (31/198; 15.66%); San Francisco (9/91; 9.89%).

**Cities covered: 10/10.** City-level counts are non-additive because a complaint type can occur in more than one city.
