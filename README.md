# FMCG Demand Forecasting (2022–2024)

This project simulates weekly demand forecasting in the FMCG (Fast-Moving Consumer Goods) industry, based on synthetic transactional data from 2022 to 2024. It includes two full modeling pipelines:

- A **lightweight version** built with Pandas and scikit-learn (Kaggle-compatible)
- A **production-style version** built in Databricks using PySpark, MLflow, and Feature Store

##  Business context

FMCG companies operate on tight margins and depend on accurate demand forecasts to optimize supply chains, reduce waste, and increase product availability. This project explores how weekly SKU-level sales can be predicted using historical data, promotional activity, seasonality, pricing, and logistics-related features.

---

## 📁 Repository structure

```

fmcg\_forecasting\_repo/
├── dataset/                     # Synthetic daily-level sales data (CSV & Parquet)
├── kaggle\_notebooks/           # Lightweight Python notebooks for EDA and modeling
├── databricks\_version/         # Spark-based forecasting pipeline in Databricks
│   ├── 01\_data\_loading.py
│   ├── 02\_feature\_engineering.py
│   ├── 03\_modeling.py
│   ├── 04\_feature\_store.py
│   ├── 05\_weekly\_prediction\_job.py
│   ├── 06\_optional\_retraining\_job.ipynb
│   └── utils/
│       ├── load\_data.py
│       ├── eda\_utils.py
│       ├── feature\_engineering\_utils.py
│       └── forecasting\_utils.py

```

---

##  Project components

### 1. `kaggle_notebooks/` 

| Notebook | Description |
|----------|-------------|
| [`01_eda_insights.ipynb`](notebooks/01_eda_insights.ipynb) | Full exploratory data analysis: seasonality, lifecycle, stockouts, promotion impact |
| [`02_demand_forecasting.ipynb`](notebooks/02_demand_forecasting.ipynb) | Forecasting model using LightGBM + feature engineering, lagging, and daily-to-weekly aggregation |

### 2. `databricks_version/`  
*Scalable Spark-based version with production-ready workflow*

- `01_data_loading.py` – Reads and checks raw data
- `02_feature_engineering.py` – Generates calendar, pricing, promotion, and logistics features
- `03_modeling.py` – Trains Random Forest model using PySpark and logs to MLflow
- `04_feature_store.py` – Registers features in the Databricks Feature Store
- `05_weekly_prediction_job.py` – Simulates new data arrival and weekly prediction pipeline
- `06_optional_retraining_job.ipynb` – Optional retraining pipeline (model decay handling)

---

## 🛠️ Technologies used

- **Databricks**
- **PySpark**
- **MLflow**
- **Databricks Feature Store**
- **scikit-learn**
- **pandas / matplotlib / seaborn**
- **XGBoost** – for comparative modeling in the Kaggle version
- **SHAP** – for feature importance and explainability
- **Python (3.10)** – core scripting and data handling
- **Kaggle Notebooks** – hosted development environment for exploratory and lightweight modeling

---
## 📊 Key Highlights

- Time-based train/test split (weekly)
- Lifecycle-aware features and custom proxies
- Stockout risk and promotion responsiveness by channel
- Visual storytelling and markdown-driven EDA
- Final model: **MAE = 3.57**, **R² = 0.60**


## 💡 Key skills demonstrated

- Time series forecasting in a production-style environment
- Working with high-dimensional SKU-level sales data
- Automating feature engineering and prediction jobs
- Registering features and models for reuse via Feature Store
- Bridging business context with technical solutions

---

## 🔗 Useful links

- [Kaggle notebooks](https://www.kaggle.com/beatafaron/code)
- [LinkedIn profile](https://www.linkedin.com/in/beata-faron-24764832/)
- [Dataset on Kaggle](https://www.kaggle.com/datasets/beatafaron/fmcg-daily-sales-2022-2024)

---


## 📎 Related Kaggle Notebooks

-  [EDA + Business Insights](https://www.kaggle.com/code/beatafaron/fmcg-2022-2024-eda-business-insights)  
-  [Forecasting Models](https://www.kaggle.com/code/beatafaron/forcasting-fmcg-demand-struggle-to-signal-ml)

---

## 📬 Author

**Beata Faron**  
[LinkedIn](https://www.linkedin.com/in/beata-faron-24764832/) | [Kaggle](https://www.kaggle.com/beatafaron)
