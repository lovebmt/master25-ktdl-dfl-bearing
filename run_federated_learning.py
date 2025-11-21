import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from collections import OrderedDict
import urllib.request

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import flwr as fl
from flwr.common import Context, Metrics, NDArrays, Scalar
from flwr.server.strategy import FedAvg
from flwr.simulation import start_simulation

import ray
import logging

os.environ["RAY_LOG_TO_STDERR"] = "0"
os.environ["RAY_BACKEND_LOG_LEVEL"] = "ERROR"
os.environ["RAY_LOG_LEVEL"] = "ERROR"

ray.init(ignore_reinit_error=True, log_to_driver=False)

os.makedirs("reports", exist_ok=True)

csv_filename = None

if csv_filename is None:
    local_paths = [
        "processed/bearing_merged_2.csv",
        "processed/bearing_merged_1.csv",
    ]
    for path in local_paths:
        if os.path.exists(path):
            csv_filename = path
            break

if csv_filename is None:
    github_urls = [
        "https://raw.githubusercontent.com/lovebmt/master25-ktdl-dfl-bearing/refs/heads/main/processed/bearing_merged_2.csv",
        "https://raw.githubusercontent.com/lovebmt/master25-ktdl-dfl-bearing/refs/heads/main/processed/bearing_merged_1.csv"
    ]
    os.makedirs("processed", exist_ok=True)
    for url in github_urls:
        try:
            filename = url.split('/')[-1]
            filename = os.path.join("processed", filename)
            urllib.request.urlretrieve(url, filename)
            csv_filename = filename
            break
        except Exception as e:
            continue

df = pd.read_csv(csv_filename)
num_df = df.select_dtypes(include=[np.number])
sensor_names = list(num_df.columns)[:8]

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()
for i, col in enumerate(sensor_names):
    axes[i].plot(df[col][:1000], linewidth=0.5, alpha=0.7)
    axes[i].set_title(f'{col} (first 1000 samples)', fontweight='bold')
    axes[i].set_xlabel('Sample')
    axes[i].set_ylabel('Value')
    axes[i].grid(True, alpha=0.3)
plt.tight_layout()
plt.suptitle('Bearing Sensor Data Visualization', fontsize=14, fontweight='bold', y=1.002)
plt.savefig('reports/01_sensor_data_visualization.png', dpi=300, bbox_inches='tight')
plt.close()

class BearingAutoencoder(nn.Module):
    def __init__(self, input_size: int = 8, latent_size: int = 4, hidden_size: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_size),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

class BearingDataset(Dataset):
    def __init__(self, X: np.ndarray):
        self.X = torch.from_numpy(X).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]
        return {"x": x, "y": x}

def _split_partitions_balanced(values: np.ndarray, partition_id: int, num_partitions: int):
    N = values.shape[0]
    part_size = math.ceil(N / num_partitions)
    start = partition_id * part_size
    end = min(start + part_size, N)
    if start >= N:
        raise RuntimeError(f"partition_id={partition_id} exceeds data size")
    return values[start:end]

def _split_partitions_imbalanced(values: np.ndarray, partition_id: int, num_partitions: int):
    N = values.shape[0]
    if num_partitions == 10:
        ratios = [0.30, 0.05, 0.12, 0.10, 0.09, 0.08, 0.08, 0.07, 0.1, 0.01]
    elif num_partitions == 5:
        ratios = [0.35, 0.25, 0.20, 0.12, 0.08]
    elif num_partitions == 3:
        ratios = [0.50, 0.30, 0.20]
    else:
        ratios = [2 ** (-i) for i in range(num_partitions)]
        ratios = [r / sum(ratios) for r in ratios]
    sizes = [int(N * r) for r in ratios]
    sizes[-1] = N - sum(sizes[:-1])
    start = sum(sizes[:partition_id])
    end = start + sizes[partition_id]
    return values[start:end]

def load_data(partition_id: int, num_partitions: int, batch_size: int = 128, 
              csv_path: str = None, data_distribution: str = "balanced"):
    if csv_path is None:
        csv_path = csv_filename
    df = pd.read_csv(csv_path)
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 8:
        raise ValueError(f"CSV has {num_df.shape[1]} numeric columns, expected >= 8")
    num_df = num_df.iloc[:, :8]
    num_df.columns = ['B1_a', 'B1_b', 'B2_a', 'B2_b', 'B3_a', 'B3_b', 'B4_a', 'B4_b']
    num_df = num_df.round(4)
    values = num_df.values
    N = values.shape[0]
    if N == 0:
        raise RuntimeError("CSV has no data")
    if data_distribution == "balanced":
        values_part = _split_partitions_balanced(values, partition_id, num_partitions)
    elif data_distribution == "imbalanced":
        values_part = _split_partitions_imbalanced(values, partition_id, num_partitions)
    else:
        raise ValueError(f"Unknown distribution: {data_distribution}")
    n_train = int(len(values_part) * 0.8)
    train_vals = values_part[:n_train]
    test_vals = values_part[n_train:]
    train_dataset = BearingDataset(train_vals)
    test_dataset = BearingDataset(test_vals)
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return trainloader, testloader

def train(net, trainloader, epochs, device, lr=0.001):
    net.to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in trainloader:
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            optimizer.zero_grad()
            output = net(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
    avg_loss = epoch_loss / len(trainloader)
    return avg_loss

def test(net, testloader, device):
    net.to(device)
    criterion = torch.nn.MSELoss()
    net.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch in testloader:
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            output = net(x)
            loss = criterion(output, y)
            test_loss += loss.item()
    avg_loss = test_loss / len(testloader)
    return avg_loss, avg_loss

class BearingClient(fl.client.NumPyClient):
    def __init__(self, partition_id: int, num_partitions: int, local_epochs: int = 1, 
                 lr: float = 0.001, data_distribution: str = "balanced"):
        self.partition_id = partition_id
        self.num_partitions = num_partitions
        self.local_epochs = local_epochs
        self.lr = lr
        self.data_distribution = data_distribution
        self.model = BearingAutoencoder()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trainloader, self.testloader = load_data(
            partition_id=self.partition_id,
            num_partitions=self.num_partitions,
            batch_size=128,
            data_distribution=self.data_distribution,
        )

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters: NDArrays) -> None:
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        self.set_parameters(parameters)
        train_loss = train(self.model, self.trainloader, epochs=self.local_epochs, 
                          device=self.device, lr=self.lr)
        return (
            self.get_parameters(config={}),
            len(self.trainloader.dataset),
            {"train_loss": train_loss},
        )

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        self.set_parameters(parameters)
        test_loss, _ = test(self.model, self.testloader, device=self.device)
        return (test_loss, len(self.testloader.dataset), {"eval_loss": test_loss})

_num_clients = 10
_local_epochs = 1
_learning_rate = 0.001
_data_distribution = "balanced"

def client_fn(cid: str) -> fl.client.Client:
    partition_id = int(cid)
    return BearingClient(
        partition_id=partition_id,
        num_partitions=_num_clients,
        local_epochs=_local_epochs,
        lr=_learning_rate,
        data_distribution=_data_distribution,
    ).to_client()

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    if not metrics:
        return {}
    total_examples = sum([num_examples for num_examples, _ in metrics])
    aggregated = {}
    metric_keys = metrics[0][1].keys()
    for key in metric_keys:
        weighted_sum = sum([
            num_examples * m[key]
            for num_examples, m in metrics
            if key in m
        ])
        aggregated[key] = weighted_sum / total_examples
    return aggregated

def create_strategy():
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    return strategy

NUM_CLIENTS = 10
NUM_ROUNDS = 20
LOCAL_EPOCHS = 1
LEARNING_RATE = 0.001

_num_clients = NUM_CLIENTS
_local_epochs = LOCAL_EPOCHS
_learning_rate = LEARNING_RATE

with open('reports/data_distribution_analysis.txt', 'w') as f:
    f.write("="*70 + "\n")
    f.write("DATA DISTRIBUTION ANALYSIS\n")
    f.write("="*70 + "\n\n")
    
    f.write("BALANCED Distribution:\n")
    for i in range(NUM_CLIENTS):
        train_loader, test_loader = load_data(i, NUM_CLIENTS, data_distribution="balanced")
        f.write(f"   Client {i}: Train={len(train_loader.dataset):5d}, Test={len(test_loader.dataset):4d}\n")
    
    f.write("\nIMBALANCED Distribution:\n")
    imbalanced_train_sizes = []
    for i in range(NUM_CLIENTS):
        train_loader, test_loader = load_data(i, NUM_CLIENTS, data_distribution="imbalanced")
        imbalanced_train_sizes.append(len(train_loader.dataset))
        pct = len(train_loader.dataset)/sum(imbalanced_train_sizes)*100 if sum(imbalanced_train_sizes) > 0 else 0
        f.write(f"   Client {i}: Train={len(train_loader.dataset):5d}, Test={len(test_loader.dataset):4d} ({pct:.1f}%)\n")

print("Running Experiment 1: FedAvg + Balanced Data...")
_data_distribution = "balanced"
strategy_balanced = create_strategy()
history_balanced = start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    strategy=strategy_balanced,
    client_resources={"num_cpus": 1, "num_gpus": 0.0},
)
train_losses_balanced = [loss for _, loss in history_balanced.metrics_distributed_fit.get("train_loss", [])]
eval_losses_balanced = [loss for _, loss in history_balanced.metrics_distributed.get("eval_loss", [])]

print("Running Experiment 2: FedAvg + Imbalanced Data...")
_data_distribution = "imbalanced"
strategy_imbalanced = create_strategy()
history_imbalanced = start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    strategy=strategy_imbalanced,
    client_resources={"num_cpus": 1, "num_gpus": 0.0},
)
train_losses_imbalanced = [loss for _, loss in history_imbalanced.metrics_distributed_fit.get("train_loss", [])]
eval_losses_imbalanced = [loss for _, loss in history_imbalanced.metrics_distributed.get("eval_loss", [])]

experiments = [
    ("Exp 1: FedAvg\n(Balanced)", train_losses_balanced, eval_losses_balanced, "steelblue"),
    ("Exp 2: FedAvg\n(Imbalanced)", train_losses_imbalanced, eval_losses_imbalanced, "orange"),
]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax = axes[0]
for name, train, _, color in experiments:
    if train:
        ax.plot(range(1, len(train) + 1), train, marker='o', label=name, color=color, linewidth=2)
ax.set_xlabel("Round", fontsize=12)
ax.set_ylabel("Training Loss (MSE)", fontsize=12)
ax.set_title("Training Loss Comparison", fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
for name, _, eval_loss, color in experiments:
    if eval_loss:
        ax.plot(range(1, len(eval_loss) + 1), eval_loss, marker='s', label=name, color=color, linewidth=2)
ax.set_xlabel("Round", fontsize=12)
ax.set_ylabel("Evaluation Loss (MSE)", fontsize=12)
ax.set_title("Evaluation Loss Comparison", fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[2]
exp_names = ["Exp 1:\nBalanced", "Exp 2:\nImbalanced"]
final_losses = [
    train_losses_balanced[-1] if train_losses_balanced else 0,
    train_losses_imbalanced[-1] if train_losses_imbalanced else 0,
]
colors_bar = ["steelblue", "orange"]
bars = ax.bar(exp_names, final_losses, color=colors_bar, alpha=0.7, edgecolor='black')
ax.set_ylabel("Final Training Loss", fontsize=12)
ax.set_title("Final Loss Comparison", fontsize=14, fontweight='bold')
ax.grid(True, axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('reports/02_experiments_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

with open('reports/experiments_summary.txt', 'w') as f:
    f.write("="*70 + "\n")
    f.write("SUMMARY TABLE\n")
    f.write("="*70 + "\n")
    f.write(f"{'Experiment':<30} {'Final Train Loss':<20} {'Final Eval Loss':<20}\n")
    f.write("-"*70 + "\n")
    for name, train, eval_loss, _ in experiments:
        train_final = f"{train[-1]:.6f}" if train else "N/A"
        eval_final = f"{eval_loss[-1]:.6f}" if eval_loss else "N/A"
        f.write(f"{name.replace(chr(10), ' '):<30} {train_final:<20} {eval_final:<20}\n")
    f.write("="*70 + "\n")
    if train_losses_balanced and train_losses_imbalanced:
        difference = ((train_losses_imbalanced[-1] - train_losses_balanced[-1]) / train_losses_balanced[-1]) * 100
        f.write(f"\nPerformance difference (Exp 2 vs Exp 1): {difference:+.2f}%\n")

final_model = BearingAutoencoder()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
final_model.to(device)

if train_losses_balanced[-1] < train_losses_imbalanced[-1]:
    _data_distribution = "balanced"
    best_exp = "Balanced"
else:
    _data_distribution = "imbalanced"
    best_exp = "Imbalanced"

strategy_final = create_strategy()
history_final = start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    strategy=strategy_final,
    client_resources={"num_cpus": 1, "num_gpus": 0.0},
)

train_loader_full, test_loader_full = load_data(0, 1, batch_size=128, data_distribution="balanced")

criterion = torch.nn.MSELoss()
final_model.eval()
all_errors = []
with torch.no_grad():
    for batch in test_loader_full:
        x = batch['x'].to(device)
        output = final_model(x)
        errors = torch.mean((x - output) ** 2, dim=1).cpu().numpy()
        all_errors.extend(errors)

all_errors = np.array(all_errors)
threshold_95 = np.percentile(all_errors, 95)
threshold_mean_2std = np.mean(all_errors) + 2 * np.std(all_errors)
anomaly_threshold = threshold_95

with open('reports/anomaly_threshold.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("ANOMALY THRESHOLD\n")
    f.write("="*80 + "\n")
    f.write(f"\n95th Percentile: {threshold_95:.6f}\n")
    f.write(f"Mean + 2×Std:    {threshold_mean_2std:.6f}\n")
    f.write(f"\nSelected threshold: {threshold_95:.6f} (95th percentile)\n")
    f.write(f"\nRULE:\n")
    f.write(f"   Error < {threshold_95:.6f} -> NORMAL\n")
    f.write(f"   Error > {threshold_95:.6f} -> ANOMALY\n")

def test_bearing_sample(sensor_values, model, threshold, device, sample_name="Test Sample"):
    x = torch.tensor(sensor_values, dtype=torch.float32).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        reconstructed = model(x)
    x_np = x.cpu().numpy()[0]
    recon_np = reconstructed.cpu().numpy()[0]
    error = np.mean((x_np - recon_np) ** 2)
    is_anomaly = error > threshold
    return error, is_anomaly, x_np, recon_np

normal_sample = num_df.iloc[100].values
error_normal, is_anomaly_normal, input_normal, output_normal = test_bearing_sample(
    normal_sample, final_model, anomaly_threshold, device, "Normal Sample"
)

anomaly_sample_1 = normal_sample.copy()
anomaly_sample_1[0] = anomaly_sample_1[0] * 10
error_anomaly_1, is_anomaly_1, input_anomaly_1, output_anomaly_1 = test_bearing_sample(
    anomaly_sample_1, final_model, anomaly_threshold, device, "Scenario 1: Sensor Error"
)

anomaly_sample_2 = normal_sample.copy() * 3
error_anomaly_2, is_anomaly_2, input_anomaly_2, output_anomaly_2 = test_bearing_sample(
    anomaly_sample_2, final_model, anomaly_threshold, device, "Scenario 2: High Vibration"
)

anomaly_sample_3 = normal_sample.copy()
anomaly_sample_3[2] = -0.5
anomaly_sample_3[3] = -0.3
error_anomaly_3, is_anomaly_3, input_anomaly_3, output_anomaly_3 = test_bearing_sample(
    anomaly_sample_3, final_model, anomaly_threshold, device, "Scenario 3: Negative Values"
)

test_results = [
    ("Normal Sample", error_normal, is_anomaly_normal),
    ("Scenario 1: Sensor Error", error_anomaly_1, is_anomaly_1),
    ("Scenario 2: High Vibration", error_anomaly_2, is_anomaly_2),
    ("Scenario 3: Negative Values", error_anomaly_3, is_anomaly_3)
]

with open('reports/anomaly_detection_results.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("ANOMALY DETECTION TEST RESULTS\n")
    f.write("="*80 + "\n\n")
    f.write(f"{'#':<5} {'Test Case':<30} {'Error (MSE)':<15} {'Result':<20}\n")
    f.write("-"*80 + "\n")
    for i, (name, error, is_anomaly) in enumerate(test_results, 1):
        status = "ANOMALY" if is_anomaly else "NORMAL"
        f.write(f"{i:<5} {name:<30} {error:<15.6f} {status:<20}\n")
    f.write("-"*80 + "\n")
    f.write(f"\nThreshold: {anomaly_threshold:.6f}\n")
    f.write(f"\nAnalysis:\n")
    f.write(f"   Normal samples: {sum(1 for _, _, a in test_results if not a)}\n")
    f.write(f"   Anomaly samples: {sum(1 for _, _, a in test_results if a)}\n")
    f.write(f"   Min error: {min(e for _, e, _ in test_results):.6f}\n")
    f.write(f"   Max error: {max(e for _, e, _ in test_results):.6f}\n")

fig, ax = plt.subplots(figsize=(12, 6))
names = [name for name, _, _ in test_results]
errors = [error for _, error, _ in test_results]
colors = ['green' if not is_anom else 'red' for _, _, is_anom in test_results]
bars = ax.bar(range(len(names)), errors, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax.axhline(y=anomaly_threshold, color='orange', linestyle='--', linewidth=2,
          label=f'Threshold: {anomaly_threshold:.6f}')
for i, (bar, error) in enumerate(zip(bars, errors)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{error:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_xlabel('Test Case', fontsize=12, fontweight='bold')
ax.set_ylabel('Reconstruction Error (MSE)', fontsize=12, fontweight='bold')
ax.set_title('Reconstruction Error: Normal vs Anomaly', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=15, ha='right', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='green', alpha=0.7, label='Normal'),
    Patch(facecolor='red', alpha=0.7, label='Anomaly'),
    plt.Line2D([0], [0], color='orange', linestyle='--', linewidth=2, label=f'Threshold: {anomaly_threshold:.6f}')
]
ax.legend(handles=legend_elements, fontsize=10, loc='upper left')
plt.tight_layout()
plt.savefig('reports/03_anomaly_detection_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

model_path = "reports/final_bearing_autoencoder.pt"
torch.save(final_model.state_dict(), model_path)

print("\n" + "="*80)
print("FEDERATED LEARNING COMPLETED!")
print("="*80)
print(f"\nAll reports and charts saved to: reports/")
print(f"  - 01_sensor_data_visualization.png")
print(f"  - 02_experiments_comparison.png")
print(f"  - 03_anomaly_detection_comparison.png")
print(f"  - data_distribution_analysis.txt")
print(f"  - experiments_summary.txt")
print(f"  - anomaly_threshold.txt")
print(f"  - anomaly_detection_results.txt")
print(f"  - final_bearing_autoencoder.pt")
print("="*80)
