// Slides Data - Embedded JavaScript
const SLIDES_DATA = {
  "presentation": {
    "title": "Decentralized Federated Learning for Bearing Anomaly Detection",
    "totalSlides": 16
  },
  "slides": [
    {
      "id": 1,
      "type": "title",
      "title": "Decentralized Federated Learning",
      "subtitle": "Ứng Dụng Phát Hiện Bất Thường Trong Dữ Liệu Vòng Bi",
      "subtitleDetail": "Ứng Dụng Phát Hiện Bất Thường Trong Dữ Liệu Vòng Bi",
      "badges": [
        { "text": "Machine Learning", "color": "green" },
        { "text": "IoT", "color": "yellow" },
        { "text": "Privacy-Preserving", "color": "red" }
      ],
      "layout": "two-column",
      "columns": [
        {
          "title": "Đội Ngũ Thực Hiện",
          "cards": [
            {
              "icon": "👥",
              "iconColor": "green",
              "title": "Nhóm 6",
              "content": "Trí Đông, Thành Phạm, Thu Thủy,<br>Nguyễn Tâm, Justin"
            }
          ]
        },
        {
          "title": "Thông Tin Khóa Học",
          "cards": [
            {
              "icon": "🎓",
              "iconColor": "blue",
              "title": "Chương trình: Thạc sĩ KTDL",
              "content": "GVHD: Trọng Nhân"
            },
            {
              "icon": "📅",
              "iconColor": "purple",
              "title": "Trường: ĐH Bách Khoa",
              "content": "Năm: 2024-2025"
            }
          ]
        }
      ]
    },
    {
      "id": 2,
      "type": "content",
      "title": "Decentralized Federated Learning",
      "subtitle": "Giải Pháp Cho Machine Learning Phân Tán",
      "badges": [
        { "text": "No Central Server", "color": "blue" },
        { "text": "P2P Communication", "color": "green" },
        { "text": "Privacy First", "color": "purple" }
      ],
      "layout": "two-column",
      "columns": [
        {
          "title": "Vấn Đề Truyền Thống",
          "cards": [
            {
              "icon": "🏢",
              "iconColor": "red",
              "title": "Central Server Required",
              "content": "Yêu cầu server trung tâm mạnh mẽ"
            },
            {
              "icon": "🔒",
              "iconColor": "orange",
              "title": "Privacy Concerns",
              "content": "Rủi ro bảo mật tại điểm trung tâm"
            },
            {
              "icon": "⚡",
              "iconColor": "yellow",
              "title": "Single Point of Failure",
              "content": "Server chết → hệ thống chết"
            }
          ]
        },
        {
          "title": "Giải Pháp DFL",
          "cards": [
            {
              "icon": "🌐",
              "iconColor": "blue",
              "title": "P2P Architecture",
              "content": "Không cần server, nodes giao tiếp trực tiếp"
            },
            {
              "icon": "🔐",
              "iconColor": "green",
              "title": "Enhanced Privacy",
              "content": "Data không bao giờ rời thiết bị"
            },
            {
              "icon": "💪",
              "iconColor": "purple",
              "title": "High Resilience",
              "content": "Nodes có thể join/leave tự do"
            }
          ]
        }
      ]
    },
    {
      "id": 3,
      "type": "content",
      "title": "Kiến Trúc DFL",
      "subtitle": "Peer-to-Peer Decentralized Architecture",
      "badges": [
        { "text": "Gossip Protocol", "color": "blue" },
        { "text": "Ring Topology", "color": "purple" },
        { "text": "Model Averaging", "color": "green" }
      ],
      "layout": "two-column",
      "columns": [
        {
          "title": "Đặc Điểm Chính",
          "cards": [
            {
              "icon": "🔄",
              "iconColor": "blue",
              "title": "Peer-to-Peer",
              "content": "10 nodes giao tiếp trực tiếp với nhau"
            },
            {
              "icon": "⭕",
              "iconColor": "purple",
              "title": "Ring Topology",
              "content": "Mỗi node chỉ kết nối với 2 neighbors"
            },
            {
              "icon": "📊",
              "iconColor": "green",
              "title": "Gossip Protocol",
              "content": "Trao đổi model weights giữa các nodes"
            }
          ]
        },
        {
          "title": "Quy Trình Training",
          "cards": [
            {
              "icon": "🎯",
              "iconColor": "green",
              "title": "Local Training",
              "content": "Mỗi node train trên data riêng"
            },
            {
              "icon": "🔀",
              "iconColor": "blue",
              "title": "Model Exchange",
              "content": "Trao đổi weights với neighbors"
            },
            {
              "icon": "⚖️",
              "iconColor": "purple",
              "title": "Weighted Averaging",
              "content": "Kết hợp models từ neighbors"
            }
          ]
        }
      ]
    },
    {
      "id": 4,
      "type": "image",
      "title": "DFL Architecture Diagram",
      "subtitle": "Federated Learning System Overview",
      "badges": [
        { "text": "10 Nodes", "color": "blue" },
        { "text": "Ring Topology", "color": "purple" },
        { "text": "Decentralized", "color": "green" }
      ],
      "image": "../reports/06_federated_learning_architecture.png",
      "imageStyle": "max-height: 550px; object-fit: contain;"
    },
    {
      "id": 5,
      "type": "content",
      "title": "Phương Pháp Nghiên Cứu",
      "subtitle": "Autoencoder-based Anomaly Detection",
      "badges": [
        { "text": "Deep Learning", "color": "blue" },
        { "text": "Unsupervised", "color": "purple" },
        { "text": "Reconstruction Error", "color": "green" }
      ],
      "layout": "two-column",
      "columns": [
        {
          "title": "Dataset",
          "cards": [
            {
              "icon": "📦",
              "iconColor": "blue",
              "title": "NASA Bearing Dataset",
              "content": "32,768 training samples"
            },
            {
              "icon": "⚙️",
              "iconColor": "green",
              "title": "8 Sensor Channels",
              "content": "20,480 time-series points mỗi file",
              "dialogImage": "../reports/bearing.png"

            },
            {
              "icon": "📊",
              "iconColor": "purple",
              "title": "Statistical Features",
              "content": "Mean, Std, RMS, Kurtosis, Skewness..."
            }
          ]
        },
        {
          "title": "Model Architecture",
          "cards": [
            {
              "icon": "🧠",
              "iconColor": "purple",
              "title": "Autoencoder",
              "content": "Encoder: 8→16→8, Decoder: 8→16→8"
            },
            {
              "icon": "🎯",
              "iconColor": "green",
              "title": "Training Approach",
              "content": "Học từ data bình thường, phát hiện anomaly"
            },
            {
              "icon": "📈",
              "iconColor": "blue",
              "title": "Anomaly Detection",
              "content": "MSE > threshold → Anomaly"
            }
          ]
        }
      ]
    },
    {
      "id": 6,
      "type": "image",
      "title": "Feature Extraction Process",
      "subtitle": "Raw Sensor Data → Statistical Features",
      "badges": [
        { "text": "8 Channels", "color": "blue" },
        { "text": "20,480 Points", "color": "purple" },
        { "text": "8 Features", "color": "green" }
      ],
      "image": "../reports/01_sensor_data_visualization.png",
      "imageStyle": "max-height: 550px; object-fit: contain;"
    },
    {
      "id": 7,
      "type": "content",
      "title": "Phân Tích Phân Phối Dữ Liệu",
      "subtitle": "Visualization & Statistical Analysis",
      "badges": [
        { "text": "IID: Equal Distribution", "color": "blue" },
        { "text": "Non-IID: Power Law", "color": "orange" }
      ],
      "layout": "stats",
      "statsCards": [
        {
          "label": "Total Samples",
          "value": "32,768",
          "sublabel": "training data"
        },
        {
          "label": "Clients",
          "value": "10",
          "sublabel": "IoT devices"
        },
        {
          "label": "IID: Each Client",
          "value": "3,276",
          "sublabel": "samples (10%)"
        },
        {
          "label": "Non-IID: Max Client",
          "value": "9,830",
          "sublabel": "samples (30%)"
        },
        {
          "label": "Non-IID: Min Client",
          "value": "329",
          "sublabel": "samples (1%)"
        },
        {
          "label": "Ratio",
          "value": "29.9x",
          "sublabel": "max/min",
          "valueStyle": "font-size: 1.5rem;"
        }
      ]
    },
    {
      "id": 8,
      "type": "image",
      "title": "Data Distribution Visualization",
      "subtitle": "IID vs Non-IID Distribution Patterns",
      "badges": [
        { "text": "IID: Equal", "color": "blue" },
        { "text": "Non-IID: Power Law", "color": "orange" }
      ],
      "image": "../reports/04_data_distribution_visualization.png",
      "imageStyle": "max-height: 500px; object-fit: contain;"
    },
    {
      "id": 9,
      "type": "content",
      "title": "Kết Quả Thí Nghiệm",
      "subtitle": "Phân Tích Hiệu Suất",
      "badges": [
        { "text": "IID vs Non-IID", "color": "blue" },
        { "text": "10 Clients", "color": "purple" },
        { "text": "100 Rounds", "color": "green" }
      ],
      "layout": "table",
      "table": {
        "headers": ["Experiment", "Data Distribution", "Final Loss", "Convergence", "Stability"],
        "rows": [
          ["Exp 1", "IID (Balanced)", "0.0023", "Fast (Round 60)", "⭐⭐⭐⭐⭐"],
          ["Exp 2", "Non-IID (Power Law)", "0.0031", "Slower (Round 80)", "⭐⭐⭐⭐"]
        ]
      },
      "additionalCards": [
        {
          "icon": "✅",
          "iconColor": "green",
          "title": "Key Finding #1",
          "content": "IID data convergence nhanh hơn ~25% so với Non-IID"
        },
        {
          "icon": "📊",
          "iconColor": "blue",
          "title": "Key Finding #2",
          "content": "Non-IID vẫn đạt kết quả tốt, chỉ chậm hơn 1 chút"
        },
        {
          "icon": "💡",
          "iconColor": "purple",
          "title": "Insight",
          "content": "DFL hoạt động hiệu quả cả với data không cân bằng"
        }
      ]
    },
    {
      "id": 10,
      "type": "image",
      "title": "Experiments Comparison",
      "subtitle": "IID vs Non-IID Over 100 Rounds",
      "badges": [
        { "text": "Smooth Convergence", "color": "green" },
        { "text": "MSE Loss", "color": "blue" }
      ],
      "image": "../reports/02_experiments_comparison.png",
      "imageStyle": "max-height: 500px; object-fit: contain;"
    },
    {
      "id": 11,
      "type": "image",
      "title": "MSE Distribution",
      "subtitle": "Statistical Analysis of Reconstruction Errors",
      "badges": [
        { "text": "Distribution Analysis", "color": "blue" },
        { "text": "MSE Metric", "color": "green" }
      ],
      "image": "../reports/07_mse_distribution_threshold.png",
      "imageStyle": "max-height: 500px; object-fit: contain;"
    },
    {
      "id": 12,
      "type": "image",
      "title": "Anomaly Detection Comparison",
      "subtitle": "Reconstruction Error Analysis",
      "badges": [
        { "text": "MSE Metric", "color": "blue" },
        { "text": "Threshold-based", "color": "red" },
        { "text": "Normal vs Anomaly", "color": "green" }
      ],
      "image": "../reports/03_anomaly_detection_comparison.png",
      "imageStyle": "max-height: 500px; object-fit: contain;"
    },
    {
      "id": 13,
      "type": "image",
      "title": "Convergence Analysis",
      "subtitle": "Training Progress Over Rounds",
      "badges": [
        { "text": "Loss Tracking", "color": "blue" },
        { "text": "Stability Analysis", "color": "green" }
      ],
      "image": "../reports/05_convergence_analysis.png",
      "imageStyle": "max-height: 500px; object-fit: contain;"
    },
    {
      "id": 14,
      "type": "content",
      "title": "Ứng Dụng Thực Tế",
      "subtitle": "DFL Trong Hệ Thống IoT",
      "badges": [
        { "text": "Smart City", "color": "blue" },
        { "text": "Industrial IoT", "color": "green" },
        { "text": "Healthcare", "color": "purple" }
      ],
      "layout": "two-column",
      "columns": [
        {
          "title": "Industrial Applications",
          "cards": [
            {
              "icon": "🏭",
              "iconColor": "blue",
              "title": "Predictive Maintenance",
              "content": "Giám sát thiết bị công nghiệp real-time"
            },
            {
              "icon": "⚙️",
              "iconColor": "green",
              "title": "Smart Manufacturing",
              "content": "Phát hiện lỗi sản xuất tự động"
            },
            {
              "icon": "🚂",
              "iconColor": "orange",
              "title": "Railway Systems",
              "content": "Monitoring vòng bi tàu hỏa"
            }
          ]
        },
        {
          "title": "IoT Ecosystems",
          "cards": [
            {
              "icon": "🏙️",
              "iconColor": "purple",
              "title": "Smart Cities",
              "content": "Sensors network không cần server trung tâm"
            },
            {
              "icon": "🏥",
              "iconColor": "red",
              "title": "Healthcare IoT",
              "content": "Medical devices với privacy cao"
            },
            {
              "icon": "🌍",
              "iconColor": "green",
              "title": "Edge Computing",
              "content": "Training tại edge, không cần cloud"
            }
          ]
        }
      ]
    },
    {
      "id": 15,
      "type": "content",
      "title": "Kết Luận",
      "subtitle": "Achievements & Key Takeaways",
      "badges": [
        { "text": "Successful", "color": "green" },
        { "text": "Privacy-Preserving", "color": "blue" },
        { "text": "Scalable", "color": "purple" }
      ],
      "layout": "two-column",
      "columns": [
        {
          "title": "Thành Tựu Đạt Được",
          "cards": [
            {
              "icon": "✅",
              "iconColor": "green",
              "title": "DFL Implementation",
              "content": "Xây dựng thành công hệ thống DFL với 10 nodes"
            },
            {
              "icon": "🎯",
              "iconColor": "blue",
              "title": "Anomaly Detection",
              "content": "Model đạt độ chính xác cao trên bearing data"
            },
            {
              "icon": "📊",
              "iconColor": "purple",
              "title": "Non-IID Handling",
              "content": "Xử lý tốt data phân phối không đồng nhất"
            }
          ]
        },
        {
          "title": "Bài Học Kinh Nghiệm",
          "cards": [
            {
              "icon": "💡",
              "iconColor": "yellow",
              "title": "Communication Overhead",
              "content": "Cần tối ưu hóa model exchange frequency"
            },
            {
              "icon": "⚖️",
              "iconColor": "orange",
              "title": "Trade-offs",
              "content": "Privacy vs Performance: cần balance hợp lý"
            },
            {
              "icon": "🔧",
              "iconColor": "red",
              "title": "Practical Considerations",
              "content": "Network stability quan trọng trong DFL"
            }
          ]
        }
      ]
    },
    {
      "id": 16,
      "type": "content",
      "title": "Hướng Phát Triển",
      "subtitle": "Roadmap & Future Research",
      "badges": [
        { "text": "Short-term", "color": "blue" },
        { "text": "Medium-term", "color": "purple" },
        { "text": "Long-term", "color": "green" }
      ],
      "layout": "two-column",
      "columns": [
        {
          "title": "Short-term (3-6 months)",
          "cards": [
            {
              "icon": "🚀",
              "iconColor": "blue",
              "title": "Optimization",
              "content": "Giảm communication overhead, faster convergence"
            },
            {
              "icon": "🔐",
              "iconColor": "green",
              "title": "Security Enhancement",
              "content": "Thêm encryption cho model weights"
            }
          ]
        },
        {
          "title": "Long-term (1-2 years)",
          "cards": [
            {
              "icon": "🌐",
              "iconColor": "purple",
              "title": "Scalability",
              "content": "Scale lên 100+ nodes, multiple datasets"
            },
            {
              "icon": "🤖",
              "iconColor": "orange",
              "title": "Advanced Models",
              "content": "Thử nghiệm với Transformer, GNN"
            },
            {
              "icon": "🏭",
              "iconColor": "red",
              "title": "Real-world Deployment",
              "content": "Deploy trong nhà máy thực tế"
            }
          ]
        }
      ]
    }
  ]
};
