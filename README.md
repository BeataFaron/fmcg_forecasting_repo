# 📦 FMCG Demand Forecasting (2022–2024)

This project simulates a real-world forecasting workflow in the FMCG (Fast-Moving Consumer Goods) industry, using synthetic daily sales data for a single SKU (MI-006) across multiple channels and regions.

---

## 🔍 Project Structure

| Notebook | Description |
|----------|-------------|
| [`01_eda_insights.ipynb`](notebooks/01_eda_insights.ipynb) | Full exploratory data analysis: seasonality, lifecycle, stockouts, promotion impact |
| [`02_demand_forecasting.ipynb`](notebooks/02_demand_forecasting.ipynb) | Forecasting model using LightGBM + feature engineering, lagging, and daily-to-weekly aggregation |

---

## 📊 Key Highlights

- Time-based train/test split (weekly)
- Lifecycle-aware features and custom proxies
- Stockout risk and promotion responsiveness by channel
- Visual storytelling and markdown-driven EDA
- Final model: **MAE = 3.57**, **R² = 0.60**

---

## 🧱 Upcoming: Databricks Version (PySpark)

This repo is also connected to a Databricks workspace.  
An upcoming version of this project will reimplement the pipeline using:
- PySpark DataFrames
- MLflow tracking
- Feature Store integration
- Model registry & evaluation at scale

Stay tuned!

---

## 🛠️ Tech Stack

- Python, Pandas, Matplotlib, Seaborn
- LightGBM, Scikit-learn
- (Planned) PySpark, MLflow (Databricks)

---

## 📎 Related Kaggle Notebooks

- 📘 [EDA + Business Insights](https://www.kaggle.com/code/beatafaron/fmcg-2022-2024-eda-business-insights)  
- 🤖 [Forecasting Models](https://www.kaggle.com/code/beatafaron/forcasting-fmcg-demand-struggle-to-signal-ml)

---

## 📬 Author

**Beata Faron**  
[LinkedIn](https://www.linkedin.com/in/beata-faron-24764832/) | [Kaggle](https://www.kaggle.com/beatafaron)
