Outline for word      

1. Giới thiệu
 

1.1 Bối cảnh dữ liệu phân tán trong IoT
1.2 Vấn đề bảo mật, băng thông và tính sẵn sàng
1.3 Tại sao cần Decentralized Federated Learning cho IoT
1.4 Mục tiêu và phạm vi báo cáo
Tam

 

 

2. IoT và Thách thức đối với Machine Learning
2.1 Cấu trúc hệ thống IoT
2.2 Đặc trưng của dữ liệu IoT
2.3 Hạn chế của phương pháp học tập trung trong IoT
2.4 Yêu cầu đối với mô hình học phân tán
Thuy

 

 

3. Tổng quan Federated Learning (nền tảng để hiểu DFL)
 

3.1 Khái niệm Federated Learning
3.2 Quy trình hoạt động và FedAvg
3.3 Ưu điểm khi áp dụng vào IoT
3.4 Các hạn chế của FL khiến cần DFL
 

 

4. Decentralized Federated Learning (DFL)
4.1 Khái niệm FL, DFL -> đưa ra mục tiêu
4.2 Kiến trúc DFL (peer-to-peer, gossip, ring, blockchain-based)
4.3 Quy trình hoạt động (không có server trung tâm)
4.4 Ưu điểm DFL so với FL trong IoT
    4.5 Thách thức kỹ thuật của DFL

Tam

 

5. Mô hình DFL trong bối cảnh IoT
 

5.1 Cấu trúc mạng IoT phù hợp DFL
5.2 Các thiết bị IoT đóng vai trò node ngang hàng
5.3 Cơ chế trao đổi trọng số giữa các IoT nodes
5.4 Xử lý trường hợp lỗi node hoặc kết nối không ổn định
5.5 Tích hợp edge computing trong DFL
 

Thuy

 

6. Mô phỏng DFL cho IoT (không dùng thiết bị thật)
 

6.1 Lý do và yêu cầu mô phỏng
6.2 Công cụ mô phỏng (Flower, FedML, mô phỏng P2P)
6.3 Dataset mô phỏng IoT (weather, smart home, air quality…)
6.4 Cách chia dataset thành nhiều IoT node
6.5 Thiết kế các node mô phỏng (client process / docker)
6.6 Cấu hình mô phỏng DFL (topology, peer list, round…)
 

Pending: a Thành

7. Triển khai mô hình DFL
 

7.1 Định nghĩa bài toán (vd: dự đoán nhiệt độ)
7.2 Lựa chọn mô hình ML phù hợp IoT (nhẹ, nhanh)
7.3 Huấn luyện cục bộ tại mỗi node
 

7.4 Giao tiếp P2P và đồng bộ hóa trọng số
7.5 Tiêu chí đánh giá mô hình trong bối cảnh DFL
 

Pending: a Thành

 

 

8. Kết quả mô phỏng & Đánh giá
 

8.1 Cấu hình mô phỏng (số node, topology…)
8.2 Hiệu suất mô hình (RMSE, loss…)
8.3 Tác động topology đến chất lượng training
8.4 So sánh: Centralized vs FL vs DFL
8.5 Phân tích ưu/nhược điểm khi dùng DFL trong IoT
 

Pending: a Thành

 

 

9. Ứng dụng DFL trong các hệ thống IoT thực tế
 

9.1 Smart City (khí tượng, giao thông, môi trường)
9.2 Smart Home / Smart Grid
9.3 Industrial IoT
9.4 Hệ thống cảm biến phân tán diện rộng (WSN)
 

 

10. Kết luận và Hướng phát triển
 

10.1 Tóm tắt đóng góp
10.2 Giới hạn của DFL trong mô phỏng
10.3 Hướng phát triển: bảo mật, decentralization sâu hơn, mô phỏng quy mô lớn
Tam

reference

 

 

 

 

2