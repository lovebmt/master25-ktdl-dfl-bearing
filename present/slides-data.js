// Slides Data - Embedded JavaScript
const SLIDES_DATA = {
  "presentation": {
    "title": "Decentralized Federated Learning for Bearing Anomaly Detection",
    "totalSlides": 20
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
      "title": "Tóm Tắt Đóng Góp & Kết Luận",
      "badges": [
        { "text": "Successful", "color": "green" },
        { "text": "Privacy-Preserving", "color": "blue" },
        { "text": "IoT-Ready", "color": "purple" }
      ],
      "layout": "two-column",
      "columns": [
        {
          "title": "Đóng Góp Chính",
          "cards": [
            {
              "icon": "📚",
              "iconColor": "blue",
              "title": "Tổng Quan DFL",
              "content": "Trình bày tầm quan trọng của DFL trong IoT, phân tích thách thức và giải pháp"
            },
            {
              "icon": "📊",
              "iconColor": "purple",
              "title": "So Sánh IID vs Non-IID",
              "content": "IID đạt 0.002425 loss, thấp hơn 10.4% so với Non-IID (0.002705)"
            },
            {
              "icon": "🎯",
              "iconColor": "orange",
              "title": "100% Độ Chính Xác",
              "content": "Phát hiện bất thường với ngưỡng MSE dựa trên phân vị 95"
            },
            {
              "icon": "📈",
              "iconColor": "red",
              "title": "Phân Tích Hội Tụ",
              "content": "IID hội tụ ổn định hơn, đạt giảm 94% mất mát so với ban đầu"
            }
          ]
        },
        {
          "title": "Hạn Chế",
          "cards": [
            {
              "icon": "💻",
              "iconColor": "orange",
              "title": "Hạn Chế: Mô Phỏng",
              "content": "Chưa triển khai trên thiết bị IoT thật với ràng buộc phần cứng"
            },
            {
              "icon": "🔐",
              "iconColor": "red",
              "title": "Hạn Chế: Bảo Mật",
              "content": "Chưa triển khai cơ chế phòng thủ chống tấn công Byzantine"
            },
            {
              "icon": "🌐",
              "iconColor": "blue",
              "title": "Hạn Chế: Topology",
              "content": "Chỉ kiểm thử ring topology, chưa thử mesh/gossip"
            },
            {
              "icon": "📏",
              "iconColor": "purple",
              "title": "Hạn Chế: Quy Mô",
              "content": "Chỉ 10 peers, chưa test với hàng trăm/nghìn node"
            }
          ]
        }
      ]
    },
    {
      "id": 16,
      "type": "content",
      "title": "Thách Thức Dữ Liệu IoT trong Bearing Monitoring",
      "subtitle": "Challenges in IoT Data & DFL Implementation",
      "badges": [
        { "text": "IoT Challenges", "color": "red" },
        { "text": "DFL Challenges", "color": "orange" },
        { "text": "Data Complexity", "color": "blue" }
      ],
      "layout": "two-column",
      "columns": [
        {
          "title": "Thách Thức Dữ Liệu IoT Nói Chung",
          "cards": [
            {
              "icon": "🌊",
              "iconColor": "blue",
              "title": "Volume & Velocity",
              "content": "Sensor stream liên tục (ms-level), 20,480 điểm/file × 8 channels → không thể upload hết lên cloud để train"
            },
            {
              "icon": "🔀",
              "iconColor": "orange",
              "title": "Non-IID Cực Mạnh",
              "content": "Mỗi bearing: tuổi thọ khác nhau, tải khác nhau, môi trường khác nhau → phân phối hoàn toàn khác biệt (khác xa dữ liệu ảnh/văn bản thường IID hơn)"
            },
            {
              "icon": "📡",
              "iconColor": "red",
              "title": "Nhiễu Cao & Thiếu Dữ Liệu",
              "content": "Sensor drift, rung môi trường, hỏng cảm biến, mất mẫu → dữ liệu IoT rất 'bẩn'"
            }
          ]
        },
        {
          "title": "Thách Thức DFL với Dữ Liệu Bearing",
          "cards": [
            {
              "icon": "📶",
              "iconColor": "yellow",
              "title": "Mạng Yếu, Mất Kết Nối",
              "content": "IoT ở dây chuyền, phân xưởng → mạng không ổn định, không sync liên tục. DFL phải tự hoạt động khi mất server"
            },
            {
              "icon": "💻",
              "iconColor": "purple",
              "title": "Edge Device Yếu",
              "content": "Năng lực tính toán nhỏ → DFL cần model nhẹ (Autoencoder 8→4→2→4→8) thay vì cloud-like workload"
            },
            {
              "icon": "⚡",
              "iconColor": "green",
              "title": "Yêu Cầu Real-time",
              "content": "Model phải chạy tại edge với latency thấp → DFL train liên tục không dùng pipeline offline → deploy"
            }
          ]
        }
      ]
    },
    {
      "id": 17,
      "type": "content",
      "title": "DFL Giải Quyết Thách Thức IoT Như Thế Nào?",
      "subtitle": "Why DFL is Different for IoT Bearing Data?",
      "badges": [
        { "text": "DFL Solutions", "color": "green" },
        { "text": "IoT-Optimized", "color": "blue" },
        { "text": "Real-world Ready", "color": "purple" }
      ],
      "layout": "two-column",
      "columns": [
        {
          "title": "Xử Lý Data Streaming & Liên Tục",
          "cards": [
            {
              "icon": "🎯",
              "iconColor": "green",
              "title": "Local Training Real-time",
              "content": "DFL cho phép local training ở edge theo thời gian thực → không cần dataset tĩnh như NLP/CV"
            },
            {
              "icon": "🔄",
              "iconColor": "blue",
              "title": "Thiết Kế Cho Non-IID Mạnh",
              "content": "Mỗi node học theo dữ liệu của nó → model thích nghi cho từng máy. Neighbor consensus giúp ổn định hơn khi không có global master"
            },
            {
              "icon": "🔐",
              "iconColor": "purple",
              "title": "Bảo Mật Dữ Liệu Cảm Biến",
              "content": "Không đưa raw vibration/temperature lên cloud → chỉ chia sẻ boundary/update nhẹ → tránh lộ bí mật sản xuất"
            }
          ]
        },
        {
          "title": "Phù Hợp Môi Trường IoT Thực Tế",
          "cards": [
            {
              "icon": "🌐",
              "iconColor": "orange",
              "title": "Tự Hoạt Động Khi Mất Mạng",
              "content": "Không phụ thuộc server → peer tự tìm hàng xóm, tự healing khi node join/leave (decentralized)"
            },
            {
              "icon": "⭕",
              "iconColor": "red",
              "title": "Hỗ Trợ Topology Linh Hoạt",
              "content": "Ring, mesh, tree → IoT deployment thật có thể map dễ dàng. Hệ thống vẫn hoạt động khi một phần mạng bị đứt"
            },
            {
              "icon": "💪",
              "iconColor": "green",
              "title": "Tối Ưu Edge Device Yếu",
              "content": "DFL không yêu cầu update lớn hoặc mô hình lớn như FL truyền thống → chỉ cần CPU nhỏ vẫn chạy liên tục"
            }
          ]
        }
      ]
    },
    {
      "id": 18,
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
      "id": 19,
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
