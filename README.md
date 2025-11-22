# DFL-demo

## Hướng dẫn sử dụng

### 1. Cài đặt môi trường và các thư viện cần thiết

- Tạo môi trường ảo (nên dùng):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

- Cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

### 2. Chạy script `run_dfl.py`

- Chạy script ở thư mục gốc hoặc thư mục `release/`:

```bash
python run_dfl.py
```

- Nếu chạy trong thư mục `release/`:

```bash
cd release
python run_dfl.py
```

OUTPUT
```bash

================================================================================
EXPERIMENT 1: DFL with BALANCED data
================================================================================

================================================================================
DECENTRALIZED FEDERATED LEARNING EXPERIMENT
================================================================================
Configuration:
  - Number of Peers: 10
  - Communication: Peer-to-Peer (P2P)
  - Topology: Ring
  - Aggregation: Local at each peer
  - Rounds: 50
  - Local Epochs: 1
  - Learning Rate: 0.001
  - Data Distribution: balanced
  - Device: cpu
================================================================================

Ring topology visualization saved to reports_dfl/ring_topology.png
Sensor data visualization saved to reports_dfl/sensor_data_visualization.png
Data distribution visualization saved to reports_dfl/data_distribution_visualization.png

################################################################################
STARTING DECENTRALIZED FEDERATED LEARNING
Peers: 10 | Rounds: 50 | Topology: Ring
################################################################################


Round 0: Avg Train Loss=0.038657, Avg Eval Loss=0.027762

Round 5: Avg Train Loss=0.006291, Avg Eval Loss=0.006558

Round 10: Avg Train Loss=0.005935, Avg Eval Loss=0.006093

Round 15: Avg Train Loss=0.004783, Avg Eval Loss=0.004923

Round 20: Avg Train Loss=0.003818, Avg Eval Loss=0.003942

Round 25: Avg Train Loss=0.002916, Avg Eval Loss=0.002989

Round 30: Avg Train Loss=0.002752, Avg Eval Loss=0.002869

Round 35: Avg Train Loss=0.002715, Avg Eval Loss=0.002829

Round 40: Avg Train Loss=0.002696, Avg Eval Loss=0.002811

Round 45: Avg Train Loss=0.002676, Avg Eval Loss=0.002790

################################################################################
DECENTRALIZED FL COMPLETED!
################################################################################


================================================================================
Experiment (balanced) completed
================================================================================


================================================================================
EXPERIMENT 2: DFL with IMBALANCED data
================================================================================

================================================================================
DECENTRALIZED FEDERATED LEARNING EXPERIMENT
================================================================================
Configuration:
  - Number of Peers: 10
  - Communication: Peer-to-Peer (P2P)
  - Topology: Ring
  - Aggregation: Local at each peer
  - Rounds: 50
  - Local Epochs: 1
  - Learning Rate: 0.001
  - Data Distribution: imbalanced
  - Device: cpu
================================================================================

Data distribution visualization saved to reports_dfl/data_distribution_visualization.png

################################################################################
STARTING DECENTRALIZED FEDERATED LEARNING
Peers: 10 | Rounds: 50 | Topology: Ring
################################################################################


Round 0: Avg Train Loss=0.033712, Avg Eval Loss=0.024514

Round 5: Avg Train Loss=0.006587, Avg Eval Loss=0.006262

Round 10: Avg Train Loss=0.006010, Avg Eval Loss=0.005632

Round 15: Avg Train Loss=0.005108, Avg Eval Loss=0.004805

Round 20: Avg Train Loss=0.004284, Avg Eval Loss=0.003966

Round 25: Avg Train Loss=0.003754, Avg Eval Loss=0.003512

Round 30: Avg Train Loss=0.003210, Avg Eval Loss=0.002975

Round 35: Avg Train Loss=0.002863, Avg Eval Loss=0.002719

Round 40: Avg Train Loss=0.002829, Avg Eval Loss=0.002689

Round 45: Avg Train Loss=0.002805, Avg Eval Loss=0.002673

################################################################################
DECENTRALIZED FL COMPLETED!
################################################################################


================================================================================
Experiment (imbalanced) completed
================================================================================

Peer losses comparison plot saved to reports_dfl/peer_losses.png

================================================================================
Creating Anomaly Detection Comparison...
================================================================================

Anomaly detection comparison saved to reports_dfl/anomaly_detection_comparison.png

================================================================================
Creating MSE Distribution Comparison...
================================================================================

MSE distribution comparison saved to reports_dfl/mse_distribution_threshold.png

================================================================================
📊 JSON REPORT GENERATED
================================================================================
  📄 reports_dfl/dfl_results.json
================================================================================


Comparison plots saved to: reports_dfl/
  - experiments_comparison.png
  - final_loss_comparison.png

================================================================================
ALL EXPERIMENTS COMPLETED!
================================================================================


```


### 3. Xem báo cáo

- Báo cáo PDF:
  - File: `report.pdf` (mở trực tiếp để xem báo cáo chính thức)
- Báo cáo kết quả DFL:
  - File: `reports_dfl/dfl_results.json`

### 4. Xem slide trình bày

- Mở file `presentation.html` bằng trình duyệt web để xem slide trình bày.

### 5. Xem hình ảnh kết quả

- Các file ảnh kết quả được lưu trong thư mục `reports_dfl/` với định dạng `.png` (ví dụ: `reports_dfl/*.png`).

---