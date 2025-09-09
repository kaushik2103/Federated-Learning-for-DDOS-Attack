import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import os
import shutil


def process_client_datasets(num_clients=5):
    print("--- Starting Data Generation and Splitting Process ---")

    print("Step 1: Generating a central synthetic dataset...")
    X, y = make_classification(
        n_samples=20000, n_features=50, n_informative=30, n_redundant=5,
        n_classes=5, n_clusters_per_class=2, flip_y=0.01, random_state=42
    )
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(50)])
    y = pd.Series(y, name='target')

    client_data_indices = [[] for _ in range(num_clients)]
    class_indices = [np.where(y == i)[0] for i in range(5)]
    for k_idx in class_indices:
        np.random.shuffle(k_idx)
        proportions = np.random.dirichlet(np.repeat(50.0, num_clients))
        samples_per_client = (proportions * len(k_idx)).astype(int)
        remainder = len(k_idx) - samples_per_client.sum()
        for i in range(remainder):
            samples_per_client[i % num_clients] += 1
        current_pos = 0
        for client_id in range(num_clients):
            num_samples = samples_per_client[client_id]
            client_data_indices[client_id].extend(k_idx[current_pos : current_pos + num_samples])
            current_pos += num_samples

    print("\nStep 2: Processing each client's dataset individually...")
    for i in range(num_clients):
        client_id = i + 1
        print(f"\n--- Processing Client {client_id} ---")
        client_dir = f"client_{client_id}"
        os.makedirs(client_dir, exist_ok=True)

        indices = client_data_indices[i]
        X_client = X.iloc[indices]
        y_client = y.iloc[indices]
        full_client_data = pd.concat([X_client, y_client], axis=1)
        full_data_path = os.path.join(client_dir, "full_dataset.csv")
        full_client_data.to_csv(full_data_path, index=False)
        print(f"  - Full, unsplit dataset saved to: {full_data_path}")

        print(f"  - Reading full dataset from file...")
        data_to_split = pd.read_csv(full_data_path)
        X_split = data_to_split.drop(columns=['target'])
        y_split = data_to_split['target']

        print("  - Splitting data into 80% training and 20% testing...")
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_split, y_split, test_size=0.2, random_state=42, stratify=y_split
            )
        except ValueError:
            print(f"  - Warning: Stratification failed for Client {client_id}. Using random split.")
            X_train, X_test, y_train, y_test = train_test_split(
                X_split, y_split, test_size=0.2, random_state=42
            )

        processed_dir = os.path.join(client_dir, "processed-dataset")
        os.makedirs(processed_dir, exist_ok=True)

        X_train.to_csv(os.path.join(processed_dir, "x_train.csv"), index=False)
        y_train.to_csv(os.path.join(processed_dir, "y_train.csv"), index=False)
        X_test.to_csv(os.path.join(processed_dir, "x_test.csv"), index=False)
        y_test.to_csv(os.path.join(processed_dir, "y_test.csv"), index=False)

        print(f"  - Split data saved to the '{processed_dir}' folder.")
        print(f"  - Train size: {len(X_train)}, Test size: {len(X_test)}")


if __name__ == "__main__":
    print("Cleaning up old client directories...")
    for i in range(1, 6):
        if os.path.exists(f'client_{i}'):
            shutil.rmtree(f'client_{i}')

    # Run the main processing function
    process_client_datasets()