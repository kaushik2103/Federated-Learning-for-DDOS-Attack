import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import shutil


def prepare_dataset_for_svm(num_clients=5, num_features=50, num_samples=20000, num_classes=5, alpha=50.0):
    print(f"Generating and preparing dataset for SVM (alpha={alpha})...")

    X, y = make_classification(
        n_samples=num_samples, n_features=num_features, n_informative=30, n_redundant=5,
        n_classes=num_classes, n_clusters_per_class=2, flip_y=0.01, random_state=42,
    )
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(num_features)])
    y = pd.Series(y, name='target')

    client_data_indices = [[] for _ in range(num_clients)]
    class_indices = [np.where(y == i)[0] for i in range(num_classes)]
    for k_idx in class_indices:
        np.random.shuffle(k_idx)
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        samples_per_client = (proportions * len(k_idx)).astype(int)
        remainder = len(k_idx) - samples_per_client.sum()
        for i in range(remainder):
            samples_per_client[i % num_clients] += 1
        current_pos = 0
        for client_id in range(num_clients):
            num_samples = samples_per_client[client_id]
            client_data_indices[client_id].extend(k_idx[current_pos: current_pos + num_samples])
            current_pos += num_samples

    print("Processing and saving data for each SVM client...")
    for client_id in range(num_clients):
        client_indices = client_data_indices[client_id]
        X_client, y_client = X.iloc[client_indices], y.iloc[client_indices]

        X_train, X_test, y_train, y_test = train_test_split(
            X_client, y_client, test_size=0.2, random_state=42, stratify=y_client
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        dir_path = f"client_{client_id + 1}/processed-dataset-svm"
        os.makedirs(dir_path, exist_ok=True)

        # Save as NumPy arrays
        np.save(os.path.join(dir_path, "x_train.npy"), X_train_scaled)
        np.save(os.path.join(dir_path, "y_train.npy"), y_train.values)
        np.save(os.path.join(dir_path, "x_test.npy"), X_test_scaled)
        np.save(os.path.join(dir_path, "y_test.npy"), y_test.values)

        print(f"Client {client_id + 1} SVM data saved. Train shape: {X_train_scaled.shape}")


if __name__ == "__main__":
    print("Cleaning up old client directories...")
    for i in range(1, 6):
        if os.path.exists(f'client_{i}'):
            shutil.rmtree(f'client_{i}')

    prepare_dataset_for_svm()
