import flwr as fl
import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, classification_report
import argparse
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class SvmClient(fl.client.NumPyClient):
    def __init__(self, client_id, x_train, y_train, x_test, y_test, num_classes):
        self.client_id = client_id
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.num_classes = num_classes
        self.model = SVC(random_state=42, kernel='rbf', max_iter=2000)

    def get_parameters(self, config):
        n_features = self.x_train.shape[1]

        initial_coef = np.zeros((self.num_classes, n_features))
        initial_intercept = np.zeros((self.num_classes,))

        if self.num_classes == 2:
            initial_coef = np.zeros((1, n_features))
            initial_intercept = np.zeros((1,))

        return [initial_coef, initial_intercept]

    def fit(self, parameters, config):
        print(f"[Client {self.client_id}] Training LinearSVC model...")
        self.model = LinearSVC(random_state=42, dual="auto", max_iter=2000)
        self.model.fit(self.x_train, self.y_train)
        print(f"[Client {self.client_id}] Training complete.")

        return [self.model.coef_, self.model.intercept_], len(self.x_train), {}

    def evaluate(self, parameters, config):
        print(f"[Client {self.client_id}] Evaluating global SVM model...")

        eval_model = LinearSVC(random_state=42, dual="auto", max_iter=2000)
        eval_model.coef_ = parameters[0]
        eval_model.intercept_ = parameters[1]
        eval_model.classes_ = np.arange(self.num_classes)

        y_pred = eval_model.predict(self.x_test)

        report = classification_report(self.y_test, y_pred, output_dict=True, zero_division=0)

        metrics = {
            "accuracy": report["accuracy"],
            "precision": report["weighted avg"]["precision"],
            "recall": report["weighted avg"]["recall"],
            "f1_score": report["weighted avg"]["f1-score"],
        }

        loss = 1.0 - metrics["accuracy"]

        print(f"[Client {self.client_id}] Evaluation: Accuracy={metrics['accuracy']:.4f}")
        return float(loss), len(self.x_test), metrics


def main():
    parser = argparse.ArgumentParser(description="Flower SVM client")
    parser.add_argument("--client-id", type=int, required=True, help="Client ID (e.g., 1)")
    args = parser.parse_args()

    client_id = args.client_id
    data_path = f"client_{client_id}/processed-dataset-svm"
    print(f"Loading SVM data for client {client_id} from {data_path}")

    try:
        x_train = np.load(os.path.join(data_path, 'x_train.npy'))
        y_train = np.load(os.path.join(data_path, 'y_train.npy'))
        x_test = np.load(os.path.join(data_path, 'x_test.npy'))
        y_test = np.load(os.path.join(data_path, 'y_test.npy'))
    except FileNotFoundError:
        print(f"Error: Data for client {client_id} not found.")
        print("Please run `prepare_dataset_svm.py` first.")
        return

    num_classes = len(np.unique(np.concatenate((y_train, y_test))))

    client = SvmClient(client_id, x_train, y_train, x_test, y_test, num_classes)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)


if __name__ == "__main__":
    main()

