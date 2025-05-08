import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

# model_runner.py
# Loads data, trains a logistic regression model, and saves evaluation plots.

def run_model(data_file="materials_data.csv", save_dir="figures"):
    """
    Trains a logistic regression model on the dataset and saves plots to disk.

    Parameters:
        data_file (str): Path to the CSV file with experiment data.
        save_dir (str): Folder where plots will be saved.
    """
    df = pd.read_csv(data_file)

    features = ["temperature", "spin_speed", "solvent_ratio", "deposition_time"]
    X = df[features]
    y = df["success"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    print("Logistic Regression Accuracy:", round(acc, 2))

    # Make sure the output folder exists
    os.makedirs(save_dir, exist_ok=True)

    # === Plot 1: Confusion Matrix ===
    cm = confusion_matrix(y_test, y_pred, labels=[1, 0])
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=300)
    plt.close()

    # === Plot 2: ROC Curve ===
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "roc_curve.png"), dpi=300)
    plt.close()

    # === Plot 3: Feature Importance ===
    coeffs = model.coef_[0]
    importance = pd.Series(coeffs, index=features).sort_values()

    plt.figure()
    importance.plot(kind="barh", color="skyblue")
    plt.title("Feature Importance (Model Coefficients)")
    plt.xlabel("Coefficient Value")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "feature_importance.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    run_model()
