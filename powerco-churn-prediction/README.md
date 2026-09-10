# ⚡ Customer Churn Prediction — Energy Utility

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Pandas](https://img.shields.io/badge/pandas-2.0%2B-150458)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E)
![License](https://img.shields.io/badge/License-MIT-green)

A machine learning project to predict **customer churn** for an energy/utility company using historical consumption, pricing, and contract data.

---

## 📌 Project Overview

Customer churn is a critical business problem for energy providers. This project analyzes customer data and applies machine learning techniques to identify customers who are more likely to churn.

**Goal:** Predict the binary target `churn`:

* `1` = Customer churned
* `0` = Customer retained

The dataset contains information related to customer behavior, consumption, pricing, contracts, products, margins, and tenure.

---

## 📂 Dataset

**File:** `data/raw/client_data.csv`

* **Records:** ~1,000 customers
* **Features:** 44 columns
* **Target:** `churn`

### Feature Groups

| Group                       | Columns                                                                                                                                              |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Identifiers**             | `id`                                                                                                                                                 |
| **Sales Channel**           | `channel_sales`, `origin_up`                                                                                                                         |
| **Electricity Consumption** | `cons_12m`, `cons_last_month`, `forecast_cons_12m`, `forecast_cons_year`, `imp_cons`                                                                 |
| **Gas Consumption**         | `cons_gas_12m`, `has_gas`                                                                                                                            |
| **Contract Dates**          | `date_activ`, `date_end`, `date_modif_prod`, `date_renewal`                                                                                          |
| **Pricing Forecasts**       | `forecast_price_energy_off_peak`, `forecast_price_energy_peak`, `forecast_price_pow_off_peak`, `forecast_discount_energy`, `forecast_meter_rent_12m` |
| **Margins & Products**      | `margin_gross_pow_ele`, `margin_net_pow_ele`, `net_margin`, `nb_prod_act`, `pow_max`                                                                 |
| **Tenure**                  | `num_years_antig`                                                                                                                                    |
| **Target**                  | `churn`                                                                                                                                              |

### ⚠️ Data Notes

* `channel_sales` and `origin_up` contain missing values.
* `id` contains anonymized customer identifiers.
* `has_gas` uses categorical `t` / `f` values.
* Date columns are stored as `YYYY-MM-DD` strings.
* The customer ID should not be used as a predictive feature.

---

## 🎯 Project Objectives

The project aims to:

* Understand customer churn patterns
* Explore customer consumption behavior
* Analyze pricing-related variables
* Clean and preprocess customer data
* Handle missing values
* Engineer useful machine learning features
* Train classification models
* Evaluate model performance
* Identify potential churn drivers
* Generate business-oriented insights

---

## 🔄 Machine Learning Workflow

```text
Raw Customer Data
        ↓
Data Cleaning
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Categorical Encoding
        ↓
Train/Test Split
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Churn Prediction
        ↓
Business Insights
```

---

## 📊 Exploratory Data Analysis

The analysis investigates:

* Churn distribution
* Customer tenure
* Electricity consumption
* Gas consumption
* Pricing variables
* Number of products
* Sales channels
* Customer margins
* Contract information
* Relationships between customer characteristics and churn

Visualizations are used to identify patterns and potential churn drivers.

---

## 🧹 Data Preprocessing

The preprocessing stage includes:

* Handling missing values
* Converting date columns to datetime
* Encoding categorical variables
* Removing irrelevant identifiers
* Checking numerical features
* Preparing data for machine learning
* Splitting data into training and testing sets

---

## 🤖 Machine Learning Models

Classification algorithms can include:

* Logistic Regression
* Decision Tree
* Random Forest
* Gradient Boosting

### Evaluation Metrics

Models can be evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion Matrix

---

## 💼 Business Application

A customer churn prediction model can help an energy provider:

* Identify customers with higher churn risk
* Prioritize retention campaigns
* Understand customer behavior
* Investigate pricing-related churn
* Support proactive customer engagement
* Make data-driven retention decisions

The model should be treated as a **decision-support tool**, not the sole basis for customer decisions.

---

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Seaborn**
* **Scikit-learn**
* **Jupyter Notebook**

---

## 📁 Project Structure

```text
powerco-churn-prediction/
│
├── data/
│   └── raw/
│       └── client_data.csv
│
├── notebooks/
│   └── churn_analysis.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   └── evaluate_model.py
│
├── models/
│   └── trained_model.pkl
│
├── requirements.txt
├── .gitignore
└── README.md
```

> Update this structure to match the actual files in your repository.

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/powerco-churn-prediction.git
```

### 2. Open the project directory

```bash
cd powerco-churn-prediction
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the environment

**Windows PowerShell:**

```powershell
venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

If the project uses Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
notebooks/churn_analysis.ipynb
```

Run the notebook cells from beginning to end.

---

## 📈 Results

The final results should compare the trained models and identify the model with the strongest performance.

Example:

| Model               | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| ------------------- | -------: | --------: | -----: | -------: | ------: |
| Logistic Regression |      XX% |       XX% |    XX% |      XX% |     XX% |
| Decision Tree       |      XX% |       XX% |    XX% |      XX% |     XX% |
| Random Forest       |      XX% |       XX% |    XX% |      XX% |     XX% |
| Gradient Boosting   |      XX% |       XX% |    XX% |      XX% |     XX% |

> Replace the placeholder values with the actual results from your project.

---

## 🧠 Skills Demonstrated

* Python Programming
* Data Cleaning
* Exploratory Data Analysis
* Data Visualization
* Feature Engineering
* Categorical Encoding
* Date Feature Engineering
* Machine Learning
* Classification
* Model Evaluation
* Business Analytics
* Customer Churn Analysis

---

## 🎓 Project Context

This project was completed as part of the **BCG X Data Science job simulation**, focusing on the application of data science and machine learning techniques to a customer churn prediction problem.

---

## ⚠️ Disclaimer

This is a personal implementation created for learning and portfolio purposes as part of a BCG X job simulation.

It is **not an official BCG X product**, does not represent BCG's internal systems, and does not contain proprietary BCG information.

---

## 👤 Author

**Muhammad Salman Sarwar**

GitHub: `YOUR-GITHUB-USERNAME`
