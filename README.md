# 📡 Telco Customer Churn Prediction

A machine learning project that predicts whether a telecom customer will churn (cancel their service), using the [IBM Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle.

> **Course:** Data Analysis with Python — Final Project No. 02
> **Group 9** | Supervised by Dr. Tran Hung

---

## 📌 Problem Statement

Customer churn refers to the event where a subscriber cancels or stops using a company's service. Retaining an existing customer is 5×–25× cheaper than acquiring a new one, making churn prediction one of the most commercially impactful applications of machine learning in the telecom industry.

**Task:** Binary classification — predict whether a customer will churn (`Yes`) or not (`No`).

---

## 📂 Dataset

- **Source:** [Kaggle — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (`blastchar/telco-customer-churn`)
- **Size:** 7,043 rows × 21 columns
- **Target variable:** `Churn` (~26% churn rate — imbalanced)

### Feature Categories

| Category | Features |
|---|---|
| Customer demographics | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| Account information | `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges` |
| Phone services | `PhoneService`, `MultipleLines` |
| Internet services | `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |
| Target | `Churn` |

---

## 🔬 Project Workflow

### Step 1 — Understanding the Data
- Downloaded dataset via Kaggle API
- Explored shape, dtypes, class distribution, and feature overview
- Identified key data quality issues (`TotalCharges` stored as string, `SeniorCitizen` as int)

### Step 2 — Data Cleaning
- Standardized missing value representations (whitespace strings → `NaN`)
- Converted `TotalCharges` to `float64`; imputed 11 missing values with the median
- Encoded `SeniorCitizen` as a categorical variable
- Dropped `customerID` (identifier, not a feature)

### Step 3 — Exploratory Data Analysis (EDA)
Generated 9 charts covering:
- Target class distribution
- Churn by tenure, monthly charges, contract type, internet service, payment method, senior citizen status, and combined contract × payment method breakdown

**Key findings:**
- Median tenure of churned customers: **10 months** vs. 38 months for retained customers
- Month-to-month contracts churn at **42.7%** vs. only 2.8% for two-year contracts
- Fiber optic users churn at **41.9%** — significantly above the 26% baseline
- Electronic check payment method has the highest churn rate at **45.3%**
- Customers without `OnlineSecurity` or `TechSupport` churn at ~42% vs. ~15% for subscribers

### Step 4 — Feature Engineering & Preprocessing
- Label-encoded binary categorical columns
- One-hot encoded multi-class categorical columns
- Applied `StandardScaler` for Logistic Regression (tree models don't require scaling)
- Stratified 80/20 train/test split (`random_state=42`)

### Step 5 — Modeling
Two models were trained:

| Model | Role | Notes |
|---|---|---|
| Logistic Regression | Baseline | `class_weight='balanced'` to handle imbalance |
| Gradient Boosting | Advanced | 200 estimators, `learning_rate=0.05`, `max_depth=4` |

### Step 6 — Evaluation

| Metric | Logistic Regression | Gradient Boosting |
|---|---|---|
| Accuracy | 0.7388 | **0.8041** |
| Precision | 0.5052 | **0.6678** |
| Recall | **0.7807** | 0.5214 |
| F1-Score | **0.6134** | 0.5856 |
| ROC-AUC | 0.8412 | **0.8430** |

**Neither model shows significant overfitting** (train–test accuracy gap < 0.05 for both).

---

## 🏆 Conclusions

- **Gradient Boosting** achieves higher accuracy, precision, and AUC overall.
- **Logistic Regression** catches more actual churners (Recall = 0.78) — making it arguably more suitable when the business cost of a missed churner outweighs false alarms.
- The most important churn predictors are: **tenure**, **MonthlyCharges**, **Contract type**, **InternetService (Fiber optic)**, absence of **OnlineSecurity / TechSupport**, and **Electronic check** payment.

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn kaggle
```

### Download the Dataset

1. Get your Kaggle API credentials from [kaggle.com/settings](https://www.kaggle.com/settings) → **Create Legacy API Key**
2. Place `kaggle.json` in `~/.kaggle/`
3. Run the dataset download cell in the notebook

### Run the Notebook

Open `telco_customer_churn.ipynb` in Jupyter or Google Colab and run all cells sequentially.

---

## 🛠️ Tech Stack

- **Python 3**
- `pandas`, `numpy` — data manipulation
- `matplotlib`, `seaborn` — visualization
- `scikit-learn` — modeling and evaluation (`LogisticRegression`, `GradientBoostingClassifier`, `StandardScaler`, metrics)

---

## 📁 Repository Structure

```
telco-customer-churn-prediction/
├── figs/
├── telco_customer_churn.ipynb   # Main analysis notebook
└── README.md
```

---

## 📄 License

This project was created for academic purposes as part of a Data Analysis with Python course assignment.
