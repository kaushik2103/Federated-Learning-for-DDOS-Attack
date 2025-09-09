import flwr as fl
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, InputLayer
import matplotlib.pyplot as plt
import os
from typing import List, Tuple, Dict, Optional, Union
from flwr.common import Metrics, EvaluateRes, Scalar, FitRes, Parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy


# Define a function to create the model for 5-class classification.
def create_model(num_features=50, num_classes=5):
    model = Sequential([
        InputLayer(input_shape=(1, num_features)),
        LSTM(64, return_sequences=False),
        Dense(32, activation='relu'),
        # The output layer has 5 neurons and 'softmax' activation for 5 classes.
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    return {"accuracy": sum(accuracies) / sum(examples)}


class FedAvgWithLogging(fl.server.strategy.FedAvg):
    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            print(f"Saving global model for round {server_round}...")
            aggregated_weights = parameters_to_ndarrays(aggregated_parameters)
            model = create_model()
            model.set_weights(aggregated_weights)
            model.save(f"reports_dl/global_model_round_{server_round}.keras")

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, EvaluateRes]],
            failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)
        if aggregated_metrics is not None:
            accuracy = aggregated_metrics.get("accuracy")
            if accuracy is not None:
                print(f"Round {server_round} evaluation matrices: Accuracy={accuracy:.4f}, Loss={aggregated_loss:.4f}")

        return aggregated_loss, aggregated_metrics


def main():
    if not os.path.exists('reports_dl'):
        os.makedirs('reports_dl')

    model = create_model()
    initial_parameters = fl.common.ndarrays_to_parameters(model.get_weights())

    strategy = FedAvgWithLogging(
        fraction_fit=1.0,
        min_fit_clients=5,
        min_available_clients=5,
        initial_parameters=initial_parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=weighted_average,
    )

    history = fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=20),
        strategy=strategy,
    )
    print("Server finished.")

    if history and hasattr(history, 'metrics_distributed') and 'accuracy' in history.metrics_distributed:
        loss_data = history.losses_distributed
        acc_data = history.metrics_distributed.get('accuracy')

        if not acc_data:
            print("Accuracy metrics were not recorded.")
            return

        rounds = [r for r, _ in acc_data]
        accuracies = [acc for _, acc in acc_data]

        losses = [l for r, l in loss_data if r in rounds]

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(rounds, losses, marker='o')
        plt.title('Federated Training Loss')
        plt.xlabel('Round')
        plt.ylabel('Loss')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(rounds, accuracies, marker='o', color='r')
        plt.title('Federated Training Accuracy')
        plt.xlabel('Round')
        plt.ylabel('Accuracy')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('reports_dl/training_history_dl.png')
        print("Deep learning training history plot saved to reports_dl/training_history_dl.png")
    else:
        print("No training history recorded to plot.")


if __name__ == "__main__":
    main()

