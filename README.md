# IS 5213: Data Science & Big Data — Capstone Day

**Instructor:** Andrew Madson
**Trine University — 2026 Spring 1**

An 8-hour hands-on capstone experience using the **Cleveland Heart Disease** dataset (303 patients, 14 attributes) from the UCI Machine Learning Repository.

---

## Quick Start

1. **Clone this repo** (or download as ZIP):
   ```
   git clone https://github.com/andymadson/5218.3p1-capstone.git
   ```
2. Open the folder in **RStudio**
3. Run **`00_setup.R`** first to install all required packages
4. Open the script for the current session and follow along

## Scripts

| File | Session | Time | Topics |
|------|---------|------|--------|
| `00_setup.R` | Setup | Start of day | Install & load all packages |
| `01_data_loading_and_eda.R` | Session 1 | 8:00–9:15 AM | Data loading, EDA, histograms, correlation heatmap, box plots, scatter plots |
| `02_data_scrubbing.R` | Session 2 | 9:30–10:45 AM | Missing values, imputation, feature engineering, one-hot encoding, train/test split |
| `03_trees_and_ensembles.R` | Session 3 | 11:00 AM–12:00 PM | Decision tree, random forest, gradient boosting, model evaluation |
| `04_regression_and_comparison.R` | Session 4 | 1:00–2:15 PM | Logistic regression, stepwise selection, ROC curves, model comparison |
| `05_unsupervised.R` | Session 5 | 2:30–3:30 PM | PCA, tSNE, K-means clustering |
| `full_script.R` | All | — | Complete day's code in one file (run top to bottom) |

## Prerequisites

- **R** and **RStudio** installed
- Internet connection (to download the dataset from UCI)
- Packages from Weeks 1–8: `rpart`, `rpart.plot`, `randomForest`, `gbm`, `ggplot2`, `caret`, `e1071`, `Rtsne`, `cluster`, `factoextra` (installed by `00_setup.R`)

## Dataset

The Cleveland Heart Disease dataset contains 303 patient records with 14 clinical attributes including age, sex, chest pain type, resting blood pressure, cholesterol, and a binary target indicating presence/absence of heart disease. It is loaded directly from the UCI repository in the scripts.
