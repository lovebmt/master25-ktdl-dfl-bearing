// Slides Data - Embedded JavaScript
const SLIDES_DATA = {
  "presentation": {
    "title": "Decentralized Federated Learning for Bearing Anomaly Detection",
    "totalSlides": 17
  },
  "slides": [
    {
      "id": 1,
      "type": "title",
      "title": "Decentralized Federated Learning",
      "subtitle": "Ứng Dụng Phát Hiện Bất Thường Trong Dữ Liệu Vòng Bi",
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
              "title": "Nhóm TEAM6",
              "content": "Lê Đức Phương<br> Phạm Văn Thành<br> Đồng Quang Trí <br> Nguyễn Văn Tâm<br> Đinh Thị Thu Thủy"
            }
          ]
        },
        {
          "title": "Thông Tin Khóa Học",
          "cards": [
            {
              "icon": "🎓",
              "iconColor": "blue",
              "title": "Môn học: KTDL",
              "content": "GVHD: TS. Phan Trọng Nhân"
            },
            {
              "icon": "📅",
              "iconColor": "purple",
              "title": "Trường: ĐH Bách Khoa",
              "content": "Tháng 11/2025"
            }
          ]
        }
      ]
    },
    {
      "id": 2,
      "type": "content",
      "title": "Decentralized Federated Learning",
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
              "title": "Central Server Required"
            },
            {
              "icon": "🔒",
              "iconColor": "orange",
              "title": "Privacy Concerns"
            },
            {
              "icon": "⚡",
              "iconColor": "yellow",
              "title": "Single Point of Failure"
            }
          ]
        },
        {
          "title": "Giải Pháp DFL",
          "cards": [
            {
              "icon": "🌐",
              "iconColor": "blue",
              "title": "P2P Architecture"
            },
            {
              "icon": "🔐",
              "iconColor": "green",
              "title": "Enhanced Privacy"
            },
            {
              "icon": "💪",
              "iconColor": "purple",
              "title": "High Resilience"
            }
          ]
        }
      ]
    },
    {
      "id": 3,
      "type": "image",
      "title": "Learning Types Comparison",
      "subtitle": "Centralized vs Federated vs Decentralized Learning",
      "badges": [
        { "text": "Centralized", "color": "blue" },
        { "text": "Federated", "color": "green" },
        { "text": "Decentralized", "color": "purple" }
      ],
      "image": "../reports_dfl/learning_type.png",
      "imageStyle": "max-height: 550px; object-fit: contain;"
    },
    {
      "id": 4,
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
              "content": "peers giao tiếp trực tiếp, không có server"
            },
            {
              "icon": "⭕",
              "iconColor": "purple",
              "title": "Ring Topology",
              "content": "Mỗi peer kết nối với 2 peers lân cận"
            },
            {
              "icon": "📊",
              "iconColor": "green",
              "title": "Model Exchange",
              "content": "Trao đổi model weights giữa các peers"
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
              "content": "Mỗi peer train trên data riêng"
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
      "id": 5,
      "type": "image",
      "title": "DFL Architecture Diagram",
      "subtitle": "Federated Learning System Overview",
      "badges": [
        { "text": "ring topology", "color": "blue" },
        { "text": "Network Topology", "color": "purple" },
        { "text": "Decentralized", "color": "green" }
      ],
      "image": "../reports_dfl/network_topology.png",
      "imageStyle": "max-height: 550px; object-fit: contain;"
    },
    {
      "id": 6,
      "type": "content",
      "title": "Autoencoder-based Anomaly Detection",
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
              "content": "32,760 training samples"
            },
            {
              "icon": "⚙️",
              "iconColor": "green",
              "title": "8 Sensor Channels",
              "content": "20,480 time-series points mỗi file",
              "dialogImage": "../reports_dfl/bearing.png"

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
              "content": "Encoder: 8→4→2 (bottleneck), Decoder: 2→4→8"
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
      "id": 7,
      "type": "image",
      "title": "Feature Extraction Process",
      "subtitle": "Raw Sensor Data → Statistical Features",
      "badges": [
        { "text": "8 Channels", "color": "blue" },
        { "text": "20,480 Points", "color": "purple" },
        { "text": "8 Features", "color": "green" }
      ],
      "image": "../reports_dfl/sensor_data_visualization.png",
      "imageStyle": "max-height: 550px; object-fit: contain;"
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
      "image": "../reports_dfl/data_distribution_visualization.png",
      "imageStyle": "max-height: 500px; object-fit: contain;"
    },
    {
      "id": 9,
      "type": "image",
      "title": "Ring Topology Network Architecture",
      "badges": [
        { "text": "Ring Topology", "color": "blue" },
        { "text": "10 Peers", "color": "purple" },
        { "text": "P2P Communication", "color": "green" }
      ],
      "image": "../reports_dfl/ring_topology.png",
      "imageStyle": "max-height: 550px; object-fit: contain;"
    },
    {
      "id": 10,
      "type": "content",
      "title": "Kết Quả",
      "badges": [
        { "text": "IID vs Non-IID", "color": "blue" },
        { "text": "10 Peers", "color": "purple" },
        { "text": "50 Rounds", "color": "green" }
      ],
      "layout": "table",
      "table": {
        "headers": ["Experiment", "Data Distribution", "Final Loss", "Convergence", "Stability"],
        "rows": [
          ["Exp 1", "IID (Balanced)", "0.002425", "Fast (Round 30-40)", "⭐⭐⭐⭐⭐"],
          ["Exp 2", "Non-IID (Power Law)", "0.002705", "Slower (Round 40-50)", "⭐⭐⭐⭐"]
        ]
      },
      "additionalCards": [
        {
          "icon": "✅",
          "iconColor": "green",
          "title": "Key Finding #1",
          "content": "IID đạt final eval loss 0.002425, thấp hơn Non-IID (0.002705) khoảng 10.4%"
        },
        {
          "icon": "📊",
          "iconColor": "blue",
          "title": "Key Finding #2",
          "content": "Train loss reduction: Balanced 94.19% vs Imbalanced 92.74%"
        },
        {
          "icon": "💡",
          "iconColor": "purple",
          "title": "Insight",
          "content": "DFL P2P Ring hoạt động hiệu quả với cả IID và Non-IID data"
        }
      ]
    },
    {
      "id": 11,
      "type": "image",
      "title": "Experiments Comparison",
      "subtitle": "IID vs Non-IID Over 50 Rounds",
      "badges": [
        { "text": "Smooth Convergence", "color": "green" },
        { "text": "MSE Loss", "color": "blue" }
      ],
      "image": "../reports_dfl/experiments_comparison.png",
      "imageStyle": "max-height: 500px; object-fit: contain;"
    },
    {
      "id": 12,
      "type": "image",
      "title": "MSE Distribution",
      "subtitle": "Statistical Analysis of Reconstruction Errors",
      "badges": [
        { "text": "Distribution Analysis", "color": "blue" },
        { "text": "MSE Metric", "color": "green" }
      ],
      "image": "../reports_dfl/mse_distribution_threshold.png",
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
      "image": "../reports_dfl/peer_losses.png",
      "imageStyle": "max-height: 500px; object-fit: contain;"
    },
    {
      "id": 14,
      "type": "image",
      "title": "Anomaly Detection Comparison",
      "subtitle": "Reconstruction Error Analysis",
      "badges": [
        { "text": "MSE Metric", "color": "blue" },
        { "text": "Threshold-based", "color": "red" },
        { "text": "Normal vs Anomaly", "color": "green" }
      ],
      "image": "../reports_dfl/anomaly_detection_comparison.png",
      "imageStyle": "max-height: 500px; object-fit: contain;"
    },
    {
      "id": 15,
      "type": "content",
      "title": "Ứng Dụng & Hướng Phát Triển",
      "subtitle": "Applications & Future Directions",
      "badges": [
        { "text": "Applications", "color": "blue" },
        { "text": "Future Work", "color": "purple" },
        { "text": "Scalable", "color": "green" }
      ],
      "layout": "two-column",
      "columns": [
        {
          "title": "Ứng Dụng Thực Tế",
          "cards": [
            {
              "icon": "🏭",
              "iconColor": "blue",
              "title": "Predictive Maintenance"
            },
            {
              "icon": "⚙️",
              "iconColor": "green",
              "title": "Smart Manufacturing"
            },
            {
              "icon": "🏙️",
              "iconColor": "purple",
              "title": "Smart City IoT"
            },
            {
              "icon": "🏥",
              "iconColor": "red",
              "title": "Healthcare IoT"
            }
          ]
        },
        {
          "title": "Hướng Phát Triển",
          "cards": [
            {
              "icon": "🚀",
              "iconColor": "blue",
              "title": "Alternative Topologies"
            },
            {
              "icon": "🔐",
              "iconColor": "green",
              "title": "Security Enhancement"
            },
            {
              "icon": "🌐",
              "iconColor": "purple",
              "title": "Large-scale Deployment"
            },
            {
              "icon": "🤖",
              "iconColor": "orange",
              "title": "Hardware Integration"
            }
          ]
        }
      ]
    },
    {
      "id": 16,
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
              "content": "Xây dựng thành công hệ thống DFL với 10 peers"
            },
            {
              "icon": "🎯",
              "iconColor": "blue",
              "title": "Anomaly Detection",
              "content": "Model phát hiện tốt bất thường với threshold dựa trên 95th percentile"
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
      "id": 17,
      "type": "thank-you",
      "title": "Cảm ơn Quý Thầy Cô và Các Bạn Đã Lắng Nghe!",
      "subtitle": "Questions & Discussion",
      "badges": [
        { "text": "Thank You!", "color": "green" },
        { "text": "Q&A", "color": "blue" }
      ],
      "questions": [
        "Nếu neighbor mất kết nối/không gửi model thì xử lý thế nào?",
        "Có timeout/retry khi chờ message từ neighbor không?",
        "Thiếu model từ neighbor thì aggregate dùng weights còn lại hay giữ model cũ?",
        "Làm sao tránh cập nhật bằng model cũ/đã lỗi (stale/faulty)?",
        "Có xác thực/khóa chữ ký để ngăn model giả mạo không?",
        "Thiết kế cho bất đồng bộ hoàn toàn — làm sao đảm bảo hội tụ (convergence)?",
        "Nếu peer bị reset/restart, có cơ chế rejoin và đồng bộ state không?",
        "Có giới hạn băng thông/chiến lược nén model khi network yếu không?",
        "Tại sao chọn ring topology thay vì mesh/star topology?",
        "Ring topology có ưu/nhược điểm gì so với các topology khác?",
        "Nếu một peer trong ring fail, toàn bộ vòng có bị đứt không?",
        "Có cơ chế backup path hoặc redundant links trong ring không?"
      ]
    }
  ]
};
