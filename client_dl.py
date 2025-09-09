import flwr as fl
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, InputLayer
import argparse
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id, x_train, y_train, x_test, y_test, num_features, num_classes):
        self.client_id = client_id
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        self.model = Sequential([
            InputLayer(input_shape=(1, num_features)),
            LSTM(64, return_sequences=False),
            Dense(32, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def get_parameters(self, config):
        print(f"[Client {self.client_id}] get_parameters")
        return self.model.get_weights()

    def fit(self, parameters, config):
        print(f"[Client {self.client_id}] fit, config: {config}")
        self.model.set_weights(parameters)

        history = self.model.fit(
            self.x_train,
            self.y_train,
            epochs=30,
            batch_size=16,
            validation_split=0.1,
            verbose=2
        )

        return self.model.get_weights(), len(self.x_train), {"accuracy": history.history['accuracy'][-1]}

    def evaluate(self, parameters, config):
        print(f"[Client {self.client_id}] evaluate, config: {config}")
        self.model.set_weights(parameters)

        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=2)
        return loss, len(self.x_test), {"accuracy": accuracy}


def main():
    parser = argparse.ArgumentParser(description="Flower DL client")
    parser.add_argument("--client-id", type=int, required=True, help="Client ID (e.g., 1)")
    args = parser.parse_args()

    client_id = args.client_id
    data_path = f"client_{client_id}/processed-dataset-dl"
    print(f"Loading DL data for client {client_id} from {data_path}")

    try:
        x_train = np.load(os.path.join(data_path, 'x_train.npy'))
        y_train = np.load(os.path.join(data_path, 'y_train.npy'))
        x_test = np.load(os.path.join(data_path, 'x_test.npy'))
        y_test = np.load(os.path.join(data_path, 'y_test.npy'))
    except FileNotFoundError:
        print(f"Error: Data for client {client_id} not found.")
        print("Please run `prepare_dataset_dl.py` first.")
        return

    num_features = x_train.shape[2]
    num_classes = len(np.unique(np.concatenate((y_train, y_test))))

    client = FlowerClient(client_id, x_train, y_train, x_test, y_test, num_features, num_classes)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)


if __name__ == "__main__":
    main()

