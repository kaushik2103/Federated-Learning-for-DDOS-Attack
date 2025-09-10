import flwr as fl
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, classification_report, confusion_matrix
import pickle
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import grpc

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id, x_train, y_train, x_test, y_test):
        self.client_id = client_id
        self.x_train = x_train
        self.y_train = y_train.values.ravel()
        self.x_test = x_test
        self.y_test = y_test.values.ravel()
        self.all_possible_classes = np.array([0, 1, 2, 3, 4])
        self.model = None

    def get_parameters(self, config):
        return []

    def fit(self, parameters, config):
        metrics = {}

        if not parameters:
            print(f"[Client {self.client_id}] Round 1: Training a small initial model...")
            self.model = RandomForestClassifier(
                n_estimators=15,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=4,
                max_features="sqrt",
                bootstrap=True,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=42
            )
        else:
            print(f"[Client {self.client_id}] Re-training the received global model...")
            model_bytes = parameters[0].tobytes()
            self.model = pickle.loads(model_bytes)

            self.model.n_estimators += 20

        self.model.fit(self.x_train, self.y_train)

        print(f"[Client {self.client_id}] Training complete. Model now has {self.model.n_estimators} trees.")

        metrics["n_features_in"] = self.model.n_features_in_

        model_bytes = pickle.dumps(self.model)

        return [np.frombuffer(model_bytes, dtype=np.uint8)], len(self.x_train), metrics

    def evaluate(self, parameters, config):
        print(f"[Client {self.client_id}] Evaluating global model...")

        try:
            model_bytes = parameters[0].tobytes()
            model: RandomForestClassifier = pickle.loads(model_bytes)

            y_pred = model.predict(self.x_test)
            y_proba = model.predict_proba(self.x_test)

            loss = log_loss(self.y_test, y_proba, labels=self.all_possible_classes)
            accuracy = model.score(self.x_test, self.y_test)

            print(f"[Client {self.client_id}] Evaluation complete: Accuracy={accuracy:.4f}, Loss={loss:.4f}")

            report = classification_report(
                self.y_test, y_pred, labels=self.all_possible_classes, output_dict=True
            )
            report_df = pd.DataFrame(report).transpose()
            report_df.to_csv(f"reports/client_{self.client_id}_warm_start_report.csv")

            cm = confusion_matrix(self.y_test, y_pred, labels=self.all_possible_classes)
            plt.figure(figsize=(10, 7))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=self.all_possible_classes, yticklabels=self.all_possible_classes)
            plt.title(f'Confusion Matrix - Client {self.client_id} (Warm Start Global Model)')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.savefig(f"reports/client_{self.client_id}_warm_start_confusion_matrix.png")
            plt.close()

            return loss, len(self.x_test), {"accuracy": accuracy}

        except Exception as e:
            print(f"[Client {self.client_id}] Evaluation failed with exception: {e}")
            return float('inf'), len(self.x_test), {"accuracy": 0.0}


def main():
    parser = argparse.ArgumentParser(description="Flower client")
    parser.add_argument("--client-id", type=int, required=True, help="Client ID (e.g., 1)")
    args = parser.parse_args()

    if not os.path.exists('reports'):
        os.makedirs('reports')

    client_id = args.client_id
    data_path = f"client_{client_id}/processed-dataset"
    print(f"Loading data for client {client_id} from {data_path}")

    try:
        x_train = pd.read_csv(os.path.join(data_path, 'x_train.csv'))
        y_train = pd.read_csv(os.path.join(data_path, 'y_train.csv'))
        x_test = pd.read_csv(os.path.join(data_path, 'x_test.csv'))
        y_test = pd.read_csv(os.path.join(data_path, 'y_test.csv'))
    except FileNotFoundError:
        print(f"Error: Data for client {client_id} not found.")
        print("Please run `prepare_dataset.py` first.")
        return

    client = FlowerClient(client_id, x_train, y_train, x_test, y_test)

    max_grpc_message_length = (2 ** 31) - 1
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=client,
        grpc_max_message_length=max_grpc_message_length
    )


if __name__ == "__main__":
    main()