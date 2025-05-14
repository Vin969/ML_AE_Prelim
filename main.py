import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from data_generator import make_data, write_to_csv
from model_runner import run_model

## @file main.py
#  @brief Entry point for running the autonomous experimentation pipeline.
#         Handles data generation, model training, evaluation, and output generation.

## @brief Deletes the dataset file if it exists.
#  Used to reset the pipeline to a clean state.
#  @param file_path Path to the dataset file.
def reset_dataset(file_path="materials_data.csv"):
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"{file_path} has been reset.")

## @brief Saves model coefficients and p-values as a formatted figure.
#  @param coef_df DataFrame containing feature names, coefficients, and p-values.
#  @param accuracy Logistic regression accuracy score.
#  @param save_path Output file path for the saved figure.
def save_results_table(coef_df, accuracy, save_path="figures/model_summary.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig_height = 1.5 + 0.4 * len(coef_df)
    fig, ax = plt.subplots(figsize=(8,fig_height))
    ax.axis('off')

    data = [["Feature", "Coefficient", "P-Value"]] + coef_df.round(4).values.tolist()

    table = ax.table(
        cellText=data,
        colLabels=None,
        cellLoc='center',
        loc='center',
        colWidths=[0.3, 0.3, 0.3]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    plt.title(f"Logistic Regression Accuracy: {accuracy:.2f}", fontsize=14, pad=0)

    plt.tight_layout(rect = [0.0, 0.0, 1.0, 1.0])
    plt.savefig(save_path, dpi=300)
    plt.close()

## @brief Main function that runs the full experiment pipeline.
#  Generates synthetic data, trains a logistic regression model,
#  evaluates performance, and saves all relevant outputs.
def main():
    reset_dataset()

    new_data = make_data(100)
    write_to_csv(new_data)

    model, X_test, y_test, y_pred, y_prob, acc, coefficients, p_values, features = run_model()

    print("Logistic Regression Accuracy:", round(acc, 2))

    os.makedirs("figures", exist_ok=True)

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("figures/confusion_matrix.png", dpi=300)
    plt.close()

    # ROC curve plot
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
    plt.savefig("figures/roc_curve.png", dpi=300)
    plt.close()

    # Coefficient and p-value table
    coef_df = pd.DataFrame({
        "Feature": features,
        "Coefficient": coefficients,
        "P-Value": p_values.values
    })

    print("\nLogistic Regression Coefficients and P-Values:")
    print(coef_df.to_string(index=False))

    save_results_table(coef_df, acc)

if __name__ == "__main__":
    main()
