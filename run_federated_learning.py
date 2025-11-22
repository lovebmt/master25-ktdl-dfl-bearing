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
import os
os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"
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

def train_model(net, trainloader, epochs, device, lr=0.001):
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

def test_model(net, testloader, device):
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
        train_loss = train_model(self.model, self.trainloader, epochs=self.local_epochs, 
                          device=self.device, lr=self.lr)
        return (
            self.get_parameters(config={}),
            len(self.trainloader.dataset),
            {"train_loss": train_loss},
        )

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        self.set_parameters(parameters)
        test_loss, _ = test_model(self.model, self.testloader, device=self.device)
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

balanced_train_sizes = []
balanced_test_sizes = []
imbalanced_train_sizes = []
imbalanced_test_sizes = []

with open('reports/data_distribution_analysis.txt', 'w') as f:
    f.write("="*70 + "\n")
    f.write("DATA DISTRIBUTION ANALYSIS\n")
    f.write("="*70 + "\n\n")
    
    f.write("BALANCED Distribution:\n")
    for i in range(NUM_CLIENTS):
        train_loader, test_loader = load_data(i, NUM_CLIENTS, data_distribution="balanced")
        balanced_train_sizes.append(len(train_loader.dataset))
        balanced_test_sizes.append(len(test_loader.dataset))
        f.write(f"   Client {i}: Train={len(train_loader.dataset):5d}, Test={len(test_loader.dataset):4d}\n")
    
    f.write("\nIMBALANCED Distribution:\n")
    for i in range(NUM_CLIENTS):
        train_loader, test_loader = load_data(i, NUM_CLIENTS, data_distribution="imbalanced")
        imbalanced_train_sizes.append(len(train_loader.dataset))
        imbalanced_test_sizes.append(len(test_loader.dataset))
        pct = len(train_loader.dataset)/sum(imbalanced_train_sizes)*100 if sum(imbalanced_train_sizes) > 0 else 0
        f.write(f"   Client {i}: Train={len(train_loader.dataset):5d}, Test={len(test_loader.dataset):4d} ({pct:.1f}%)\n")

# 1. Data Distribution Visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Bar chart for balanced distribution
ax = axes[0, 0]
client_ids = list(range(NUM_CLIENTS))
ax.bar(client_ids, balanced_train_sizes, color='steelblue', alpha=0.7, edgecolor='black')
ax.set_xlabel('Client ID', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Training Samples', fontsize=12, fontweight='bold')
ax.set_title('IID (Balanced) Data Distribution', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(balanced_train_sizes):
    ax.text(i, v, str(v), ha='center', va='bottom', fontsize=9, fontweight='bold')

# Pie chart for balanced distribution
ax = axes[0, 1]
colors_pie = plt.cm.Set3(np.linspace(0, 1, NUM_CLIENTS))
wedges, texts, autotexts = ax.pie(balanced_train_sizes, labels=[f'C{i}' for i in range(NUM_CLIENTS)],
                                    autopct='%1.1f%%', colors=colors_pie, startangle=90)
for autotext in autotexts:
    autotext.set_color('black')
    autotext.set_fontsize(9)
    autotext.set_fontweight('bold')
ax.set_title('IID (Balanced) Distribution Percentage', fontsize=14, fontweight='bold')

# Bar chart for imbalanced distribution
ax = axes[1, 0]
ax.bar(client_ids, imbalanced_train_sizes, color='orange', alpha=0.7, edgecolor='black')
ax.set_xlabel('Client ID', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Training Samples', fontsize=12, fontweight='bold')
ax.set_title('Non-IID (Imbalanced) Data Distribution', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(imbalanced_train_sizes):
    ax.text(i, v, str(v), ha='center', va='bottom', fontsize=9, fontweight='bold')

# Pie chart for imbalanced distribution
ax = axes[1, 1]
wedges, texts, autotexts = ax.pie(imbalanced_train_sizes, labels=[f'C{i}' for i in range(NUM_CLIENTS)],
                                    autopct='%1.1f%%', colors=colors_pie, startangle=90)
for autotext in autotexts:
    autotext.set_color('black')
    autotext.set_fontsize(9)
    autotext.set_fontweight('bold')
ax.set_title('Non-IID (Imbalanced) Distribution Percentage', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('reports/04_data_distribution_visualization.png', dpi=300, bbox_inches='tight')
plt.close()

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

# 2. Convergence Analysis Details
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Detailed training loss plot
ax = axes[0, 0]
if train_losses_balanced:
    ax.plot(range(1, len(train_losses_balanced) + 1), train_losses_balanced, 
            marker='o', label='Balanced', color='steelblue', linewidth=2, markersize=6)
if train_losses_imbalanced:
    ax.plot(range(1, len(train_losses_imbalanced) + 1), train_losses_imbalanced, 
            marker='s', label='Imbalanced', color='orange', linewidth=2, markersize=6)
ax.set_xlabel('Round', fontsize=12, fontweight='bold')
ax.set_ylabel('Training Loss (MSE)', fontsize=12, fontweight='bold')
ax.set_title('Training Loss Convergence Over Rounds', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Detailed evaluation loss plot
ax = axes[0, 1]
if eval_losses_balanced:
    ax.plot(range(1, len(eval_losses_balanced) + 1), eval_losses_balanced, 
            marker='o', label='Balanced', color='steelblue', linewidth=2, markersize=6)
if eval_losses_imbalanced:
    ax.plot(range(1, len(eval_losses_imbalanced) + 1), eval_losses_imbalanced, 
            marker='s', label='Imbalanced', color='orange', linewidth=2, markersize=6)
ax.set_xlabel('Round', fontsize=12, fontweight='bold')
ax.set_ylabel('Evaluation Loss (MSE)', fontsize=12, fontweight='bold')
ax.set_title('Evaluation Loss Convergence Over Rounds', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Convergence speed comparison (loss decrease rate)
ax = axes[1, 0]
if train_losses_balanced and len(train_losses_balanced) > 1:
    loss_decrease_balanced = [train_losses_balanced[i] - train_losses_balanced[i+1] 
                               for i in range(len(train_losses_balanced)-1)]
    ax.plot(range(1, len(loss_decrease_balanced) + 1), loss_decrease_balanced, 
            marker='o', label='Balanced', color='steelblue', linewidth=2, markersize=6)
if train_losses_imbalanced and len(train_losses_imbalanced) > 1:
    loss_decrease_imbalanced = [train_losses_imbalanced[i] - train_losses_imbalanced[i+1] 
                                 for i in range(len(train_losses_imbalanced)-1)]
    ax.plot(range(1, len(loss_decrease_imbalanced) + 1), loss_decrease_imbalanced, 
            marker='s', label='Imbalanced', color='orange', linewidth=2, markersize=6)
ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax.set_xlabel('Round', fontsize=12, fontweight='bold')
ax.set_ylabel('Loss Decrease (Current - Next)', fontsize=12, fontweight='bold')
ax.set_title('Convergence Speed Comparison', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Cumulative improvement
ax = axes[1, 1]
if train_losses_balanced:
    initial_balanced = train_losses_balanced[0]
    improvement_balanced = [(initial_balanced - loss) / initial_balanced * 100 
                             for loss in train_losses_balanced]
    ax.plot(range(1, len(improvement_balanced) + 1), improvement_balanced, 
            marker='o', label='Balanced', color='steelblue', linewidth=2, markersize=6)
if train_losses_imbalanced:
    initial_imbalanced = train_losses_imbalanced[0]
    improvement_imbalanced = [(initial_imbalanced - loss) / initial_imbalanced * 100 
                               for loss in train_losses_imbalanced]
    ax.plot(range(1, len(improvement_imbalanced) + 1), improvement_imbalanced, 
            marker='s', label='Imbalanced', color='orange', linewidth=2, markersize=6)
ax.set_xlabel('Round', fontsize=12, fontweight='bold')
ax.set_ylabel('Improvement from Initial Loss (%)', fontsize=12, fontweight='bold')
ax.set_title('Cumulative Improvement Over Time', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('reports/05_convergence_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Federated Learning Architecture Diagram
fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(5, 9.5, 'Federated Learning Architecture', 
        fontsize=20, fontweight='bold', ha='center', va='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', edgecolor='black', linewidth=2))

# Central Server
server_rect = plt.Rectangle((4, 7), 2, 1, facecolor='lightcoral', edgecolor='black', linewidth=2)
ax.add_patch(server_rect)
ax.text(5, 7.5, 'Central Server\n(Aggregator)', fontsize=12, fontweight='bold', ha='center', va='center')
ax.text(5, 6.7, 'FedAvg Strategy', fontsize=9, ha='center', va='top', style='italic')

# Global Model
model_rect = plt.Rectangle((4, 5.5), 2, 0.6, facecolor='lightyellow', edgecolor='black', linewidth=1.5)
ax.add_patch(model_rect)
ax.text(5, 5.8, 'Global Model\n(Autoencoder)', fontsize=10, fontweight='bold', ha='center', va='center')

# Clients in a circle
num_display_clients = 10
radius = 2.5
center_x, center_y = 5, 3
for i in range(num_display_clients):
    angle = 2 * np.pi * i / num_display_clients - np.pi/2
    x = center_x + radius * np.cos(angle)
    y = center_y + radius * np.sin(angle)
    
    # Client box
    client_rect = plt.Rectangle((x-0.3, y-0.2), 0.6, 0.4, facecolor='lightgreen', 
                                 edgecolor='black', linewidth=1.5)
    ax.add_patch(client_rect)
    ax.text(x, y, f'Client {i}', fontsize=8, fontweight='bold', ha='center', va='center')
    
    # Local data indicator
    ax.text(x, y-0.35, f'Data: {imbalanced_train_sizes[i]}', fontsize=6, ha='center', va='top')
    
    # Arrow from server to client (send model)
    arrow_to_client = plt.Arrow(5, 7, x-5, y-7, width=0.15, color='blue', alpha=0.4)
    ax.add_patch(arrow_to_client)
    
    # Arrow from client to server (send weights)
    arrow_to_server = plt.Arrow(x, y, 5-x, 7-y, width=0.15, color='red', alpha=0.4)
    ax.add_patch(arrow_to_server)

# Legend for data flow
ax.text(0.5, 8, 'Data Flow:', fontsize=11, fontweight='bold')
ax.arrow(0.5, 7.6, 0.5, 0, head_width=0.1, head_length=0.1, fc='blue', ec='blue')
ax.text(1.2, 7.6, 'Model Distribution', fontsize=9, va='center')
ax.arrow(0.5, 7.2, 0.5, 0, head_width=0.1, head_length=0.1, fc='red', ec='red')
ax.text(1.2, 7.2, 'Weight Updates', fontsize=9, va='center')

# Process flow on the right
ax.text(8.5, 8, 'Training Process:', fontsize=11, fontweight='bold')
steps = [
    '1. Server broadcasts\n   global model',
    '2. Clients train\n   on local data',
    '3. Clients send\n   weight updates',
    '4. Server aggregates\n   using FedAvg',
    '5. Update global\n   model'
]
for i, step in enumerate(steps):
    y_pos = 7.5 - i * 0.6
    ax.text(8.5, y_pos, step, fontsize=8, va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', edgecolor='black', linewidth=1))

# Model architecture details
ax.text(0.5, 2.5, 'Model Architecture:', fontsize=11, fontweight='bold')
arch_text = 'Input (8) → Hidden (16) →\nLatent (4) → Hidden (16) → Output (8)'
ax.text(0.5, 2, arch_text, fontsize=8, va='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lavender', edgecolor='black', linewidth=1))

# Data distribution info
ax.text(0.5, 1, f'Total Clients: {NUM_CLIENTS}', fontsize=9)
ax.text(0.5, 0.6, f'Training Rounds: {NUM_ROUNDS}', fontsize=9)
ax.text(0.5, 0.2, f'Local Epochs: {LOCAL_EPOCHS}', fontsize=9)

plt.tight_layout()
plt.savefig('reports/06_federated_learning_architecture.png', dpi=300, bbox_inches='tight')
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

# MSE Distribution Visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Histogram of MSE distribution
ax = axes[0, 0]
n, bins, patches = ax.hist(all_errors, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
ax.axvline(threshold_95, color='red', linestyle='--', linewidth=2, label=f'95th Percentile: {threshold_95:.6f}')
ax.axvline(threshold_mean_2std, color='orange', linestyle='--', linewidth=2, label=f'Mean + 2σ: {threshold_mean_2std:.6f}')
ax.axvline(np.mean(all_errors), color='green', linestyle='-', linewidth=2, label=f'Mean: {np.mean(all_errors):.6f}')
ax.set_xlabel('Reconstruction Error (MSE)', fontsize=12, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax.set_title('MSE Distribution with Threshold', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# Cumulative distribution
ax = axes[0, 1]
sorted_errors = np.sort(all_errors)
cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
ax.plot(sorted_errors, cumulative, color='steelblue', linewidth=2)
ax.axvline(threshold_95, color='red', linestyle='--', linewidth=2, label=f'95th Percentile')
ax.axhline(95, color='red', linestyle='--', linewidth=2, alpha=0.5)
ax.axvline(threshold_mean_2std, color='orange', linestyle='--', linewidth=2, label=f'Mean + 2σ')
ax.set_xlabel('Reconstruction Error (MSE)', fontsize=12, fontweight='bold')
ax.set_ylabel('Cumulative Percentage (%)', fontsize=12, fontweight='bold')
ax.set_title('Cumulative Distribution of MSE', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Box plot
ax = axes[1, 0]
bp = ax.boxplot([all_errors], vert=True, patch_artist=True, widths=0.5)
bp['boxes'][0].set_facecolor('lightblue')
bp['boxes'][0].set_edgecolor('black')
bp['boxes'][0].set_linewidth(2)
ax.axhline(threshold_95, color='red', linestyle='--', linewidth=2, label=f'95th Percentile: {threshold_95:.6f}')
ax.axhline(threshold_mean_2std, color='orange', linestyle='--', linewidth=2, label=f'Mean + 2σ: {threshold_mean_2std:.6f}')
ax.set_ylabel('Reconstruction Error (MSE)', fontsize=12, fontweight='bold')
ax.set_title('MSE Distribution Box Plot', fontsize=14, fontweight='bold')
ax.set_xticklabels(['All Errors'])
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# MSE Formula and Statistics
ax = axes[1, 1]
ax.axis('off')

# MSE Formula
formula_text = r'$MSE = \frac{1}{n} \sum_{i=1}^{n} (x_i - \hat{x}_i)^2$'
ax.text(0.5, 0.85, 'Mean Squared Error (MSE) Formula:', fontsize=14, fontweight='bold', 
        ha='center', va='top', transform=ax.transAxes)
ax.text(0.5, 0.75, formula_text, fontsize=16, ha='center', va='top', 
        transform=ax.transAxes, bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='black', linewidth=2))

# Where clause
where_text = (
    r'where:' + '\n'
    r'$x_i$ = original input value' + '\n'
    r'$\hat{x}_i$ = reconstructed value' + '\n'
    r'$n$ = number of features (8 sensors)'
)
ax.text(0.5, 0.60, where_text, fontsize=11, ha='center', va='top', 
        transform=ax.transAxes, family='monospace')

# Statistics
stats_text = (
    f'Statistics:\n'
    f'━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n'
    f'Total samples:     {len(all_errors):,}\n'
    f'Mean MSE:          {np.mean(all_errors):.6f}\n'
    f'Std deviation:     {np.std(all_errors):.6f}\n'
    f'Min MSE:           {np.min(all_errors):.6f}\n'
    f'Max MSE:           {np.max(all_errors):.6f}\n'
    f'Median:            {np.median(all_errors):.6f}\n'
    f'━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n'
    f'95th Percentile:   {threshold_95:.6f}\n'
    f'Mean + 2σ:         {threshold_mean_2std:.6f}\n'
    f'Selected Threshold: {threshold_95:.6f}'
)
ax.text(0.5, 0.40, stats_text, fontsize=10, ha='center', va='top', 
        transform=ax.transAxes, family='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcyan', edgecolor='black', linewidth=1.5))

plt.tight_layout()
plt.savefig('reports/07_mse_distribution_threshold.png', dpi=300, bbox_inches='tight')
plt.close()

with open('reports/anomaly_threshold.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("ANOMALY THRESHOLD\n")
    f.write("="*80 + "\n")
    f.write(f"\nMSE Formula: MSE = (1/n) * Σ(xi - x̂i)²\n")
    f.write(f"  where xi = original value, x̂i = reconstructed value, n = features\n\n")
    f.write(f"Statistics:\n")
    f.write(f"  Total samples:     {len(all_errors):,}\n")
    f.write(f"  Mean MSE:          {np.mean(all_errors):.6f}\n")
    f.write(f"  Std deviation:     {np.std(all_errors):.6f}\n")
    f.write(f"  Min MSE:           {np.min(all_errors):.6f}\n")
    f.write(f"  Max MSE:           {np.max(all_errors):.6f}\n")
    f.write(f"  Median:            {np.median(all_errors):.6f}\n\n")
    f.write(f"95th Percentile: {threshold_95:.6f}\n")
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
print(f"  - 04_data_distribution_visualization.png")
print(f"  - 05_convergence_analysis.png")
print(f"  - 06_federated_learning_architecture.png")
print(f"  - 07_mse_distribution_threshold.png")
print(f"  - data_distribution_analysis.txt")
print(f"  - experiments_summary.txt")
print(f"  - anomaly_threshold.txt")
print(f"  - anomaly_detection_results.txt")
print(f"  - final_bearing_autoencoder.pt")
print("="*80)
