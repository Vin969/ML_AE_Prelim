from data_generator import make_data, write_to_csv
from model_runner import run_model

# main.py
# Generates a batch of synthetic data and runs the model training + evaluation.

if __name__ == "__main__":
    """
    Generates a new batch of synthetic experiments, saves them,
    and trains the logistic regression model on the full dataset.
    """
    new_batch = make_data(10)
    write_to_csv(new_batch)

    run_model()
