import flwr as fl
import numpy as np
from sklearn.svm import LinearSVC
from typing import Dict, List, Tuple, Optional, Union
from flwr.common import FitRes, Scalar, ndarrays_to_parameters, parameters_to_ndarrays, Parameters, EvaluateRes
from flwr.server.client_proxy import ClientProxy
import os
import matplotlib.pyplot as plt
import pickle
import time


class SvmStrategy(fl.server.strategy.Strategy):
    def __init__(self, num_classes=5, min_fit_clients=5):
        super().__init__()
        self.global_params: Optional[Parameters] = None
        self.num_classes = num_classes
        self.min_fit_clients = min_fit_clients

    def initialize_parameters(self, client_manager: fl.server.client_manager.ClientManager) -> Optional[Parameters]:
        return None

    def configure_fit(self, server_round: int, parameters: Parameters,
                      client_manager: fl.server.client_manager.ClientManager
                      ) -> List[Tuple[ClientProxy, fl.common.FitIns]]:
        self.global_params = parameters

        while client_manager.num_available() < self.min_fit_clients:
            print(f"Waiting for clients... {client_manager.num_available()}/{self.min_fit_clients} connected.")
            time.sleep(3)

        print(f"Enough clients connected ({client_manager.num_available()}). Starting round {server_round}.")

        clients = client_manager.sample(num_clients=self.min_fit_clients, min_num_clients=self.min_fit_clients)
        return [(client, fl.common.FitIns(parameters, {})) for client in clients]

    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]],
                      failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
                      ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        client_weights = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
        coefs = [weights[0] for weights in client_weights]
        intercepts = [weights[1] for weights in client_weights]

        avg_coef = np.mean(coefs, axis=0)
        avg_intercept = np.mean(intercepts, axis=0)

        print(f"Saving global SVM model for round {server_round}...")
        global_model = LinearSVC()
        global_model.coef_ = avg_coef
        global_model.intercept_ = avg_intercept
        global_model.classes_ = np.arange(self.num_classes)

        model_filename = f"report_svm/global_model_round_{server_round}.pkl"
        with open(model_filename, "wb") as f:
            pickle.dump(global_model, f)

        new_global_params = ndarrays_to_parameters([avg_coef, avg_intercept])
        self.global_params = new_global_params

        print(f"Round {server_round}: Aggregation of SVM models complete.")
        return new_global_params, {}

    def configure_evaluate(self, server_round: int, parameters: Parameters,
                           client_manager: fl.server.client_manager.ClientManager
                           ) -> List[Tuple[ClientProxy, fl.common.EvaluateIns]]:
        clients = client_manager.sample(num_clients=self.min_fit_clients, min_num_clients=self.min_fit_clients)
        return [(client, fl.common.EvaluateIns(self.global_params, {})) for client in clients]

    def aggregate_evaluate(self, server_round: int, results: List[Tuple[ClientProxy, EvaluateRes]],
                           failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]]
                           ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        if not results:
            return None, {}

        loss = np.mean([res.loss for _, res in results])
        accuracy = np.mean([res.metrics["accuracy"] for _, res in results])
        precision = np.mean([res.metrics["precision"] for _, res in results])
        recall = np.mean([res.metrics["recall"] for _, res in results])
        f1_score = np.mean([res.metrics["f1_score"] for _, res in results])

        print(
            f"Round {server_round} evaluation matrices: \n"
            f"  Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
            f"Recall: {recall:.4f}, F1-score: {f1_score:.4f}"
        )

        return loss, {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
        }

    def evaluate(self, server_round: int, parameters: Parameters):
        return None


def main():
    if not os.path.exists('report_svm'):
        os.makedirs('report_svm')

    # Instantiate the strategy with the minimum number of clients
    strategy = SvmStrategy(min_fit_clients=5)

    history = fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=20),
        strategy=strategy
    )
    print("Server finished.")

    # --- ROBUST PLOTTING LOGIC ---
    if history and history.metrics_distributed and history.losses_distributed:
        loss_data = history.losses_distributed

        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Federated SVM Training History')

        # Extract and Plot Loss
        rounds_loss = [r for r, _ in loss_data]
        losses = [l for _, l in loss_data]
        axs[0, 0].plot(rounds_loss, losses, marker='o')
        axs[0, 0].set_title('Training Loss')
        axs[0, 0].set_xlabel('Round')
        axs[0, 0].set_ylabel('Loss')
        axs[0, 0].grid(True)

        metrics_to_plot = {
            'accuracy': (0, 1),
            'precision': (1, 0),
            'f1_score': (1, 1)
        }

        for metric, pos in metrics_to_plot.items():
            if metric in history.metrics_distributed:
                metric_data = history.metrics_distributed[metric]
                rounds = [r for r, _ in metric_data]
                values = [v for _, v in metric_data]

                ax = axs[pos]
                ax.plot(rounds, values, marker='o',
                        color='r' if metric == 'accuracy' else 'g' if metric == 'precision' else 'b')
                ax.set_title(f'Training {metric.capitalize()}')
                ax.set_xlabel('Round')
                ax.set_ylabel(metric.capitalize())
                ax.grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('report_svm/training_history_svm.png')
        print("SVM training history plot saved to report_svm/training_history_svm.png")
    else:
        print("No training history recorded to plot.")


if __name__ == "__main__":
    main()