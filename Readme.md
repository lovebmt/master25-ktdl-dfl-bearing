# Federated Learning for Bearing Anomaly Detection with Flower 🌸

This notebook demonstrates federated learning using Flower (flwr) framework with PyTorch to train an autoencoder model on bearing sensor data. The autoencoder learns to reconstruct normal bearing behavior, which can be used for anomaly detection.

## 📋 Table of Contents
1. **Setup & Installation** - Install dependencies
2. **Data Preparation** - Load and prepare bearing sensor data
   - Understanding data structure
   - Input/Output examples
   - Data statistics and visualization
3. **Model Definition** - Define the autoencoder architecture
4. **Dataset Class** - Create PyTorch dataset for autoencoder
5. **Data Loading Functions** - Partition data for federated learning
6. **Training & Testing Functions** - Define training and evaluation logic
7. **Flower Client Definition** - Define federated learning client
8. **Flower Server Strategy** - Define server aggregation strategy
9. **Run Federated Learning Simulation** - Execute FL training
10. **Evaluate Final Model** - Test the global model
11. **Visualize Results** - Plot training metrics and reconstructions
    - Loss and RMSE plots
    - Accuracy metrics and improvements
    - Reconstruction quality analysis
    - Input → Output testing with examples
12. **Save Final Model** - Export trained model
13. **Summary & Next Steps** - Key insights and future directions
## 📖 Project Overview

This notebook demonstrates **Federated Learning for Bearing Anomaly Detection** using an Autoencoder architecture.

### **What We'll Build:**
1. **Autoencoder Model** 🧠
   - Input: 8 sensor readings from bearing vibration data
   - Architecture: Encoder (8 → 4 → 2) + Decoder (2 → 4 → 8)
   - Purpose: Learn normal bearing patterns and detect anomalies

2. **Federated Learning with Flower** 🌸
   - Multiple clients (simulating edge devices)
   - Collaborative training without sharing raw data
   - Privacy-preserving machine learning

3. **Two Key Experiments** 🔬
   - **Experiment 1**: FedAvg with **balanced data** (IID) - Baseline
   - **Experiment 2**: FedAvg with **imbalanced data** (Non-IID) - Real-world scenario

### **Key Research Questions:**
- ❓ How does data distribution affect federated learning?
- ❓ Does imbalanced data degrade FedAvg performance?

### **Technology Stack:**
- **PyTorch**: Deep learning framework
- **Flower**: Federated learning framework
- **Pandas**: Data manipulation
- **Matplotlib**: Visualization

### **Dataset:**
- **NASA IMS Bearing Dataset** (vibration sensor data)
- 8 channels of vibration measurements
- Multiple bearing failures recorded over time
- Perfect for demonstrating edge device scenarios

### **Why This Matters:**
In real-world Industrial IoT:
- 🏭 Different factories collect different amounts of data
- 📊 Data distribution is naturally imbalanced
- 🔒 Data privacy regulations prevent centralized collection
- 🌐 Federated learning enables collaborative model training

Let's explore how different data distributions affect federated learning performance!
## 8. Flower Server Strategy 🎯

Define aggregation strategy for Federated Learning.

### **FedAvg (Federated Averaging):**
- ✅ **Simple**: Weighted average of model parameters from clients
- ✅ **Fast**: No additional regularization term
- ✅ **Good with IID data**: Works well when data is evenly distributed
- ⚠️ **May struggle with non-IID data**: Performance can degrade with imbalanced data

**Formula:**
```
Aggregated_θ = Σ(n_i/N × θ_i)

Where:
- θ_i: Model parameters from client i
- n_i: Number of samples in client i
- N: Total number of samples
```

We'll test FedAvg with both **balanced** and **imbalanced** data distributions!

### 💡 Key Insights from Experiments

**What We Learned:**

1. **Balanced Data (Exp 1) - Baseline Performance** ✅
   - FedAvg works well when clients have equal data
   - Stable convergence
   - Good final performance
   
2. **Imbalanced Data (Exp 2) - Real-World Challenge** ⚠️
   - Performance may degrade with non-IID data
   - Some clients with less data may overfit
   - Convergence can be slower and less stable
   - Shows the real-world challenges of federated learning

**Recommendations for Production:**

| Scenario | Recommended Strategy | Reason |
|----------|---------------------|---------|
| Balanced data across clients | FedAvg | Simple, efficient, works well |
| Imbalanced data (real-world) | FedAvg or FedProx | Standard approaches for production |
| High data heterogeneity | FedProx or other advanced methods | Better handles non-IID |
| Limited communication | FedAvg | Lower overhead |

**Industrial IoT Applications:**
- 🏭 Predictive maintenance with edge devices
- 🌐 Distributed quality control systems
- 📱 Mobile device anomaly detection
- 🚗 Vehicle fleet health monitoring

**Key Takeaway:**
Data distribution significantly impacts federated learning performance. In real-world deployments, understanding your data distribution is critical for successful model training.

### 💡 Giải Thích: Tại Sao Dùng MSE và Cách Phát Hiện Bất Thường?

#### 🤔 **Câu hỏi: Tại sao cần tính MSE?**

**MSE (Mean Squared Error)** đo lường **sự khác biệt** giữa:
- **Input** (giá trị cảm biến gốc)
- **Output** (giá trị model reconstruct)

**Công thức MSE:**
```
MSE = (1/8) × Σ(input_i - output_i)²

Trong đó:
- 8 = số cảm biến (B1_a, B1_b, B2_a, B2_b, B3_a, B3_b, B4_a, B4_b)
- input_i = giá trị cảm biến thứ i
- output_i = giá trị reconstruct thứ i
```

#### 🎯 **MSE Liên Quan Đến Bất Thường Như Thế Nào?**

**Nguyên lý hoạt động:**

1. **Model học từ dữ liệu BÌNH THƯỜNG** (training):
   - Model học "pattern" của bearing hoạt động bình thường
   - VD: B1_a ≈ 0.12, B1_b ≈ 0.15, tương quan giữa các cảm biến...
   
2. **Khi gặp dữ liệu BÌNH THƯỜNG** (testing):
   - Model **NHẬN RA** pattern quen thuộc
   - Reconstruct **CHÍNH XÁC**
   - **MSE THẤP** ✅
   
3. **Khi gặp dữ liệu BẤT THƯỜNG** (testing):
   - Model **KHÔNG NHẬN RA** pattern này (chưa học bao giờ)
   - Reconstruct **SAI LỆCH**
   - **MSE CAO** ❌

---

#### 📊 **Ví Dụ Cụ Thể:**

**Case 1: Mẫu Bình Thường**
```
Input:  [0.12, 0.15, 0.11, 0.13, 0.14, 0.12, 0.10, 0.13]
Output: [0.12, 0.15, 0.11, 0.13, 0.14, 0.12, 0.10, 0.13]
        ↓
MSE = ((0.12-0.12)² + (0.15-0.15)² + ... ) / 8 = 0.0001
      ↓
✅ MSE THẤP → BÌNH THƯỜNG
```

**Case 2: Mẫu Bất Thường (Cảm biến lỗi)**
```
Input:  [1.20, 0.15, 0.11, 0.13, 0.14, 0.12, 0.10, 0.13]  ← B1_a = 1.20 (quá cao!)
Output: [0.18, 0.15, 0.11, 0.13, 0.14, 0.12, 0.10, 0.13]  ← Model cố reconstruct
        ↓
MSE = ((1.20-0.18)² + (0.15-0.15)² + ... ) / 8 = 0.1302
      ↓
❌ MSE CAO → BẤT THƯỜNG
```

---

#### 🎓 **Tóm Tắt:**

| Tình Huống | Pattern | Model Reconstruct | MSE | Kết Luận |
|------------|---------|-------------------|-----|----------|
| **Bình thường** | Model đã học | Chính xác ✅ | **Thấp** (< threshold) | ✅ Normal |
| **Bất thường** | Model chưa học | Sai lệch ❌ | **Cao** (> threshold) | 🚨 Anomaly |

---

#### 🔧 **Ứng Dụng Thực Tế:**

**Bearing bị hỏng → Rung bất thường → MSE cao → Phát hiện kịp thời!**

1. **Bình thường**: Bearing hoạt động ổn định
   - Rung đều → MSE thấp → Không cảnh báo
   
2. **Bắt đầu hỏng**: Rung bắt đầu thay đổi
   - MSE tăng dần → Cảnh báo sớm
   
3. **Hỏng nặng**: Rung rất bất thường
   - MSE rất cao → Cảnh báo ngay lập tức!

💡 **Ưu điểm**: Không cần nhãn "bất thường", chỉ cần học từ dữ liệu bình thường!