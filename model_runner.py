import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import statsmodels.api as sm

## @file model_runner.py
#  @brief Trains and evaluates a logistic regression model using experimental data.
#         Returns model predictions, metrics, and statistics.

## @brief Runs the logistic regression model and evaluates its performance.
#  Trains the model, predicts outcomes, computes accuracy, and extracts coefficient statistics.
#  @param data_file Path to the CSV dataset file.
#  @return Tuple containing model, test data, predictions, accuracy, coefficients, p-values, and feature names.
def run_model(data_file="materials_data.csv"):
    df = pd.read_csv(data_file).dropna()

    features = [
        "temperature", "spin_speed", "solvent_ratio",
        "deposition_time", "solute_concentration", "annealing_time"
    ]
    X = df[features]
    y = df["success"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Statsmodels for p-values
    X_train_sm = sm.add_constant(X_train)
    sm_model = sm.Logit(y_train, X_train_sm).fit(disp=0)
    p_values = sm_model.pvalues[1:]

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)

    return model, X_test, y_test, y_pred, y_prob, acc, model.coef_[0], p_values, features
