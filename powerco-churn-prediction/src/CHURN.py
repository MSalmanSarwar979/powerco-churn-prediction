```python
"""
PowerCo Customer Churn and Price Sensitivity Analysis

Forage / BCG X Data Science Job Simulation
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

CLIENT_FILE = BASE_DIR / "client_data (1).csv"
PRICE_FILE = BASE_DIR / "price_data (1).csv"

DATE_COLUMNS = [
    "date_activ",
    "date_end",
    "date_modif_prod",
    "date_renewal",
]

PRICE_COLUMNS = [
    "price_off_peak_var",
    "price_peak_var",
    "price_mid_peak_var",
    "price_off_peak_fix",
    "price_peak_fix",
    "price_mid_peak_fix",
]


# ------------------------------------------------------------
# Data loading
# ------------------------------------------------------------

def load_data():
    """Load customer and historical price data."""
    client_df = pd.read_csv(CLIENT_FILE)
    price_df = pd.read_csv(PRICE_FILE)

    print(f"Client data: {client_df.shape}")
    print(f"Price data:  {price_df.shape}")

    return client_df, price_df


# ------------------------------------------------------------
# Data quality checks
# ------------------------------------------------------------

def inspect_data(client_df, price_df):
    """Display basic data-quality information."""

    print("\nClient columns:")
    print(client_df.columns.tolist())

    print("\nClient data types:")
    print(client_df.dtypes)

    print("\nMissing values in client data:")
    missing_client = client_df.isnull().sum()
    print(missing_client[missing_client > 0])

    print("\nMissing values in price data:")
    missing_price = price_df.isnull().sum()
    print(missing_price[missing_price > 0])

    duplicate_ids = client_df["id"].duplicated().sum()

    print("\nCustomer ID check:")
    print(f"Total rows:       {len(client_df)}")
    print(f"Unique IDs:       {client_df['id'].nunique()}")
    print(f"Duplicate IDs:    {duplicate_ids}")


# ------------------------------------------------------------
# Churn analysis
# ------------------------------------------------------------

def analyze_churn(client_df):
    """Analyze the distribution of customer churn."""

    churn_counts = client_df["churn"].value_counts()
    churn_percent = (
        client_df["churn"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    print("\nChurn counts:")
    print(churn_counts)

    print("\nChurn percentage:")
    print(churn_percent)

    plt.figure(figsize=(7, 5))
    sns.countplot(data=client_df, x="churn")
    plt.title("Customer Churn Distribution")
    plt.xlabel("Churn (0 = Stayed, 1 = Churned)")
    plt.ylabel("Number of Customers")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# Date preparation
# ------------------------------------------------------------

def prepare_dates(client_df, price_df):
    """Convert date columns to pandas datetime objects."""

    for column in DATE_COLUMNS:
        client_df[column] = pd.to_datetime(
            client_df[column],
            errors="coerce",
        )

    price_df["price_date"] = pd.to_datetime(
        price_df["price_date"],
        errors="coerce",
    )

    return client_df, price_df


# ------------------------------------------------------------
# Customer analysis
# ------------------------------------------------------------

def analyze_customer_features(client_df):
    """Visualize important customer characteristics."""

    plt.figure(figsize=(8, 5))
    sns.boxplot(
        data=client_df,
        x="churn",
        y="cons_12m",
    )
    plt.title("Annual Consumption vs Churn")
    plt.xlabel("Churn")
    plt.ylabel("12-Month Consumption")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    sns.boxplot(
        data=client_df,
        x="churn",
        y="net_margin",
    )
    plt.title("Net Margin vs Churn")
    plt.xlabel("Churn")
    plt.ylabel("Net Margin")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# Price sensitivity features
# ------------------------------------------------------------

def create_price_features(price_df):
    """
    Create customer-level historical price features.

    Average price is calculated across the available
    variable and fixed price components.
    """

    price_df = price_df.copy()

    price_df["avg_price"] = price_df[PRICE_COLUMNS].mean(axis=1)

    price_df = price_df.sort_values(
        ["id", "price_date"]
    )

    price_df["price_change"] = (
        price_df
        .groupby("id")["avg_price"]
        .pct_change()
    )

    price_df["abs_price_change"] = (
        price_df["price_change"].abs()
    )

    price_features = (
        price_df
        .groupby("id")
        .agg(
            avg_price=("avg_price", "mean"),
            max_price=("avg_price", "max"),
            min_price=("avg_price", "min"),
            avg_price_change=("price_change", "mean"),
            max_price_change=("price_change", "max"),
            avg_abs_price_change=("abs_price_change", "mean"),
            max_abs_price_change=("abs_price_change", "max"),
        )
        .reset_index()
    )

    return price_df, price_features


# ------------------------------------------------------------
# Combine customer and price information
# ------------------------------------------------------------

def build_analysis_dataset(client_df, price_features):
    """Merge customer information with price features."""

    analysis_df = client_df.merge(
        price_features,
        on="id",
        how="left",
    )

    print("\nMerged dataset:")
    print(analysis_df.shape)

    return analysis_df


# ------------------------------------------------------------
# Price sensitivity analysis
# ------------------------------------------------------------

def analyze_price_sensitivity(analysis_df):
    """Compare price-related features between churn groups."""

    summary = (
        analysis_df
        .groupby("churn")[
            [
                "avg_price",
                "avg_abs_price_change",
                "max_abs_price_change",
            ]
        ]
        .mean()
    )

    print("\nPrice sensitivity by churn:")
    print(summary)

    plt.figure(figsize=(8, 5))
    sns.boxplot(
        data=analysis_df,
        x="churn",
        y="avg_abs_price_change",
    )
    plt.title("Price Change vs Customer Churn")
    plt.xlabel("Churn")
    plt.ylabel("Average Absolute Price Change")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# Correlation analysis
# ------------------------------------------------------------

def analyze_correlations(analysis_df):
    """Display correlations between numerical variables and churn."""

    numeric_columns = analysis_df.select_dtypes(
        include=np.number
    ).columns

    correlation = analysis_df[numeric_columns].corr()

    print("\nCorrelation with churn:")
    print(
        correlation["churn"]
        .sort_values(ascending=False)
    )


# ------------------------------------------------------------
# Machine learning preparation
# ------------------------------------------------------------

def prepare_model_data(analysis_df):
    """Prepare features and target for machine learning."""

    model_df = analysis_df.copy()

    # Customer ID is an identifier, not a predictive feature.
    model_df = model_df.drop(
        columns=["id"],
        errors="ignore",
    )

    # Original dates are excluded from this baseline model.
    model_df = model_df.drop(
        columns=DATE_COLUMNS,
        errors="ignore",
    )

    X = model_df.drop(columns=["churn"])
    y = model_df["churn"]

    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numerical_features = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    return (
        X,
        y,
        numerical_features,
        categorical_features,
    )


# ------------------------------------------------------------
# Model training
# ------------------------------------------------------------

def train_model(
    X_train,
    y_train,
    numerical_features,
    categorical_features,
):
    """Build preprocessing pipeline and train Logistic Regression."""

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_pipeline,
                numerical_features,
            ),
            (
                "cat",
                categorical_pipeline,
                categorical_features,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    return model


# ------------------------------------------------------------
# Model evaluation
# ------------------------------------------------------------

def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model."""

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    print("\nModel performance")
    print("-" * 40)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC:  {roc_auc:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
        )
    )

    cm = confusion_matrix(
        y_test,
        predictions,
    )

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    return accuracy, roc_auc


# ------------------------------------------------------------
# Main workflow
# ------------------------------------------------------------

def main():

    print("\nPowerCo Customer Churn Analysis")
    print("=" * 45)

    client_df, price_df = load_data()

    inspect_data(
        client_df,
        price_df,
    )

    analyze_churn(client_df)

    client_df, price_df = prepare_dates(
        client_df,
        price_df,
    )

    analyze_customer_features(client_df)

    price_df, price_features = create_price_features(
        price_df
    )

    print("\nPrice feature summary:")
    print(
        price_features[
            [
                "avg_price",
                "avg_price_change",
                "avg_abs_price_change",
            ]
        ].describe()
    )

    analysis_df = build_analysis_dataset(
        client_df,
        price_features,
    )

    analyze_price_sensitivity(
        analysis_df
    )

    analyze_correlations(
        analysis_df
    )

    (
        X,
        y,
        numerical_features,
        categorical_features,
    ) = prepare_model_data(
        analysis_df
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print("\nTraining data:", X_train.shape)
    print("Testing data: ", X_test.shape)

    model = train_model(
        X_train,
        y_train,
        numerical_features,
        categorical_features,
    )

    accuracy, roc_auc = evaluate_model(
        model,
        X_test,
        y_test,
    )

    print("\nBusiness interpretation")
    print("-" * 40)
    print(
        "The analysis combines customer characteristics "
        "with historical pricing information to investigate "
        "potential relationships between price changes and churn."
    )

    print(
        "The Logistic Regression model provides a baseline "
        "for identifying customers with higher predicted "
        "churn probability."
    )

    print(
        f"\nFinal baseline Accuracy: {accuracy:.4f}"
    )

    print(
        f"Final baseline ROC-AUC:  {roc_auc:.4f}"
    )

    print("\nAnalysis completed successfully.")


if __name__ == "__main__":
    main()
```
