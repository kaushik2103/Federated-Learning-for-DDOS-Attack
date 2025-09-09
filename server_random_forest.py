import flwr as fl
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
from typing import Dict, List, Tuple, Optional, Union
from flwr.common import FitRes, Scalar, ndarrays_to_parameters, parameters_to_ndarrays, Parameters
from flwr.server.client_proxy import ClientProxy
import os
import time
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


def plot_history(history: fl.server.history.History):
    print("Generating loss and accuracy plots...")

    loss = history.losses_distributed
    accuracy = history.metrics_distributed.get("accuracy")

    if not accuracy:
        print("No accuracy history found to plot.")
        return

    rounds = [r for r, _ in loss]
    loss_values = [l for _, l in loss]
    acc_values = [a for _, a in accuracy]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    ax1.plot(rounds, loss_values, marker='o', linestyle='-')
    ax1.set_title("Aggregated Loss vs. Rounds")
    ax1.set_xlabel("Federated Round")
    ax1.set_ylabel("Loss")
    ax1.grid(True)

    ax2.plot(rounds, acc_values, marker='o', linestyle='-', color='r')
    ax2.set_title("Aggregated Accuracy vs. Rounds")
    ax2.set_xlabel("Federated Round")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True)

    plt.tight_layout()
    plot_filename = "reports/training_history_warm_start.png"
    plt.savefig(plot_filename)
    print(f"Training history plot saved to {plot_filename}")
    plt.close()


class FedAvgWarmStart(fl.server.strategy.Strategy):
    def __init__(self, min_fit_clients: int, **kwargs):
        super().__init__(**kwargs)
        self.min_fit_clients = min_fit_clients
        self.global_model: Optional[RandomForestClassifier] = None
        self.global_model_params: Optional[Parameters] = None

    def initialize_parameters(self, client_manager: fl.server.client_manager.ClientManager) -> Optional[Parameters]:
        return None

    def configure_fit(
            self, server_round: int, parameters: Parameters, client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[ClientProxy, fl.common.FitIns]]:
        while client_manager.num_available() < self.min_fit_clients:
            print(f"Waiting for clients... {client_manager.num_available()}/{self.min_fit_clients} connected.")
            time.sleep(3)

        print(f"Enough clients connected ({client_manager.num_available()}). Starting round {server_round}.")

        return [(client, fl.common.FitIns(parameters, {})) for client in
                client_manager.sample(num_clients=self.min_fit_clients)]

    def aggregate_fit(
            self, server_round: int, results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        old_n_estimators = self.global_model.n_estimators if self.global_model else 0

        newly_trained_estimators = []
        all_classes_lists = []

        for _, fit_res in results:
            model_bytes = parameters_to_ndarrays(fit_res.parameters)[0].tobytes()
            updated_model: RandomForestClassifier = pickle.loads(model_bytes)

            newly_trained_estimators.extend(updated_model.estimators_[old_n_estimators:])
            all_classes_lists.append(updated_model.classes_)

        if self.global_model is None:
            self.global_model = RandomForestClassifier(n_estimators=len(newly_trained_estimators), random_state=42,
                                                       warm_start=True)
            self.global_model.estimators_ = newly_trained_estimators
            dummy_X, dummy_y = make_blobs(n_samples=2, n_features=results[0][1].metrics["n_features_in"],
                                          centers=len(np.unique(np.concatenate(all_classes_lists))))
            self.global_model.fit(dummy_X, dummy_y)
        else:
            self.global_model.estimators_.extend(newly_trained_estimators)
            self.global_model.n_estimators = len(self.global_model.estimators_)

        self.global_model.classes_ = np.unique(np.concatenate(all_classes_lists))
        self.global_model.n_classes_ = len(self.global_model.classes_)

        print(f"Aggregation complete. Global model now has {self.global_model.n_estimators} trees.")

        model_bytes = pickle.dumps(self.global_model)
        params_ndarray = np.frombuffer(model_bytes, dtype=np.uint8)
        self.global_model_params = ndarrays_to_parameters([params_ndarray])

        return self.global_model_params, {}

    def configure_evaluate(
            self, server_round: int, parameters: Parameters, client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[ClientProxy, fl.common.EvaluateIns]]:
        if self.global_model_params is None:
            return []

        return [(client, fl.common.EvaluateIns(parameters, {})) for client in
                client_manager.sample(num_clients=client_manager.num_available())]

    def aggregate_evaluate(
            self, server_round: int, results: List[Tuple[ClientProxy, fl.common.EvaluateRes]],
            failures: List[Union[Tuple[ClientProxy, fl.common.EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        if not results:
            return None, {}

        loss = sum(res.num_examples * res.loss for _, res in results) / sum(res.num_examples for _, res in results)
        accuracy = sum(res.num_examples * res.metrics["accuracy"] for _, res in results) / sum(
            res.num_examples for _, res in results)

        print(f"Round {server_round} evaluation aggregated: Loss={loss:.4f}, Accuracy={accuracy:.4f}")
        return loss, {"accuracy": accuracy}

    def evaluate(self, server_round: int, parameters: Parameters) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        if self.global_model_params:
            model_bytes = parameters_to_ndarrays(parameters)[0].tobytes()
            model = pickle.loads(model_bytes)
            model_filename = f"reports/global_model_round_{server_round}.pkl"
            with open(model_filename, "wb") as f:
                pickle.dump(model, f)
            print(f"Global model from round {server_round} saved to {model_filename}")
        return None


def main():
    if not os.path.exists('reports'):
        os.makedirs('reports')

    strategy = FedAvgWarmStart(min_fit_clients=5)

    history = fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=25),
        strategy=strategy,
    )

    print("Server finished.")
    plot_history(history)


if __name__ == "__main__":
    main()