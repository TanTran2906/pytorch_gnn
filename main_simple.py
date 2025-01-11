# Sử dụng PyTorch để xây dựng và huấn luyện mô hình
import torch
import torch.nn as nn
import numpy as np  # Xử lý dữ liệu số học, như ma trận và mảng
import argparse
import utils
import dataloader
from gnn_wrapper import GNNWrapper, SemiSupGNNWrapper
# Visualization đồ thị (graphs)
import matplotlib.pyplot as plt
import networkx as nx

# GRAPH #1
# mỗi cạnh gồm 3 phần tử [node1, node2, graph_id]. graph_id = 0 xác định đây là đồ thị đầu tiên
e = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [
    1, 2, 0], [1, 3, 0], [2, 3, 0], [2, 4, 0]]
# Với mỗi cạnh [node1, node2, graph_id] trong danh sách e, thêm một cạnh ngược [node2, node1, graph_id] => iến đồ thị thành đồ thị không có hướng
e.extend([[i, j, num] for j, i, num in e])
# Sắp xếp các cạnh theo thứ tự để dễ quản lý
e = sorted(e)
E = np.asarray(e)  # Mảng NumPy chứa danh sách cạnh của đồ thị đầu tiên
# array([[0, 1, 0],
#    [0, 2, 0],
#    [0, 4, 0],]])

# number of nodes
edges = 5
# Tạo ma trận đặc trưng cho nút dưới dạng One-Hot Encoding.
# Ma trận N có kích thước (5, 5) vì đồ thị có 5 nút
N = np.eye(edges, dtype=np.float32)
# N = [[1, 0, 0, 0, 0],
#      [0, 1, 0, 0, 0],
#      [0, 0, 1, 0, 0],
#      [0, 0, 0, 1, 0],
#      [0, 0, 0, 0, 1]]
# Thêm cột graph_id vào ma trận đặc trưng nút. Vì đây là đồ thị đầu tiên, tất cả giá trị graph_id
# Tạo một mảng kích thước (edges, 1) với giá trị toàn 0 (np.zeros)
# Nối mảng này vào ma trận N ở cột cuối cùng
N = np.concatenate((N, np.zeros((edges, 1), dtype=np.float32)), axis=1)
# N = [[1, 0, 0, 0, 0, 0],
#      [0, 1, 0, 0, 0, 0],
#      [0, 0, 1, 0, 0, 0],
#      [0, 0, 0, 1, 0, 0],
#      [0, 0, 0, 0, 1, 0]]


# Hàm plot_graph dùng NetworkX để vẽ đồ thị
def plot_graph(E, N):
    g = nx.Graph()  # Khởi tạo một đồ thị trống để thêm các node và cạnh
    # Thêm các node vào đồ thị, dựa trên số hàng trong ma trận đặc trưng N
    # Nếu N có 5 hàng, thì các node từ 0 đến 4 sẽ được thêm vào đồ thị
    g.add_nodes_from(range(N.shape[0]))
    # Thêm các cạnh vào đồ thị từ danh sách E. Cột đầu tiên và thứ hai trong E đại diện cho các cặp nodes kết nối với nhau
    g.add_edges_from(E[:, :2])
    # Vẽ đồ thị với bố cục "spring layout", vị trí các nodes được xác định dựa trên lực đẩy và hút (mô phỏng vật lý)
    # cmap=plt.get_cmap('Set1'): Chỉ định màu sắc cho các node
    # with_labels=True: Hiển thị nhãn của các node
    nx.draw_spring(g, cmap=plt.get_cmap('Set1'), with_labels=True)
    plt.show()


plot_graph(E, N)

# GRAPH #2

e1 = [[0, 2, 1], [0, 3, 1], [1, 2, 1], [1, 3, 1], [2, 3, 1]]
e1.extend([[i, j, num] for j, i, num in e1])
# Các node trong e2 được đánh lại ID để không bị trùng với e1
# N.shape[0]: Số lượng node trong GRAPH #1
e2 = [[a + N.shape[0], b + N.shape[0], num] for a, b, num in e1]
# reorder
e2 = sorted(e2)
edges_2 = 4

# Plot second graph

# E1 = np.asarray(e1)
# N1 = np.eye(edges_2, dtype=np.float32)

# # Thêm một cột graph_id = 1(np.ones) vào ma trận đặc trưng nút N1 để xác định GRAPH #2
# N1 = np.concatenate((N1, np.ones((edges_2, 1), dtype=np.float32)), axis=1)
# # Hiển thị GRAPH #2
# plot_graph(E1, N1)

# Hợp nhất danh sách cạnh E của GRAPH #1 và e2
E = np.concatenate((E, np.asarray(e2)), axis=0)
# Hợp nhất ma trận đặc trưng nút:
N_tot = np.eye(edges + edges_2, dtype=np.float32)
N_tot = np.concatenate(
    (N_tot, np.zeros((edges + edges_2, 1), dtype=np.float32)), axis=1)

plot_graph(E, N_tot)

# Create Input to GNN
# Trong thực tế: Nhãn (labels) thường không được tạo ngẫu nhiên mà dựa vào dữ liệu thực tế

# np.random.randint(low, high, size)
# low: Giá trị nhỏ nhất
# high: Giá trị lớn nhất
# size: Kích thước của mảng đầu ra
# low=0 và high=2: Hàm sẽ sinh ra các số nguyên ngẫu nhiên là 0 hoặc 1
# N_tot.shape[0]: Tổng số node trong đồ thị hợp nhất
# Mảng có kích thước tương ứng với số lượng node, trong đó mỗi phần tử là 0 hoặc 1
#  [1, 0, 1, 1, 0, 0, 1, 0,1]
labels = np.random.randint(2, size=(N_tot.shape[0]))
# labels = np.eye(max(labels)+1, dtype=np.int32)[labels]  # one-hot encoding of labels =>cần thiết cho bài toán multiclass


# Tạo một đối tượng cấu hình (cfg) từ lớp Config trong gói GNNWrapper
cfg = GNNWrapper.Config()
# Cho phép sử dụng GPU (CUDA) để tăng tốc quá trình tính toán nếu hệ thống hỗ trợ
cfg.use_cuda = True
# trả về thiết bị tương ứng (cuda:0 nếu có GPU, hoặc cpu nếu không có GPU)
cfg.device = utils.prepare_device(n_gpu_use=1, gpu_id=0)
# Không kích hoạt TensorBoard để theo dõi trực quan quá trình huấn luyện
cfg.tensorboard = False
cfg.epochs = 500

# Hàm kích hoạt chuẩn hóa đầu ra của mỗi node trong khoảng (-1, 1)
cfg.activation = nn.Tanh()
# Kích thước tầng ẩn của hàm chuyển trạng thái
# [5, ] nghĩa là tầng ẩn có kích thước 5 đơn vị (neurons)
# Dùng trong quá trình cập nhật trạng thái node
cfg.state_transition_hidden_dims = [5, ]
# Kích thước tầng ẩn của hàm đầu ra (output function)
# [5] nghĩa là tầng ẩn có kích thước 5 đơn vị
# Dùng để tính toán đầu ra cuối cùng từ trạng thái node
cfg.output_function_hidden_dims = [5]
cfg.state_dim = 5  # Mỗi node trong đồ thị sẽ được biểu diễn bằng một vector 5 chiều
# Số vòng lặp tối đa để cập nhật trạng thái của node trong quá trình lan truyền thông tin
cfg.max_iterations = 50
# Ngưỡng hội tụ: Nếu sự thay đổi trong trạng thái node giữa các vòng lặp nhỏ hơn 0.01, quá trình lan truyền sẽ dừng lại trước khi đạt đến max_iterations
cfg.convergence_threshold = 0.01
# False: Sử dụng mô hình node-level tasks (nhiệm vụ dựa trên từng node, như phân loại node)
# Nếu là True, mô hình sẽ tập trung vào graph-level tasks (nhiệm vụ dựa trên toàn bộ đồ thị, như phân loại đồ thị)
cfg.graph_based = False
# Ghi log sau mỗi 10 epoch để hiển thị các chỉ số huấn luyện như loss, accuracy
cfg.log_interval = 10
# Loại nhiệm vụ huấn luyện,"multiclass": Phân loại đa lớp,Các node sẽ được gán nhãn thuộc một trong nhiều lớp (ví dụ: 0, 1, 2,...)
cfg.task_type = "multiclass"
cfg.lrw = 0.001  # Learning rate (tốc độ học)

# model creation: Tạo một mô hình GNN,sử dụng cấu hình đã thiết lập trong cfg
model = GNNWrapper(cfg)
# dataset creation: chuyển đổi dữ liệu đầu vào (danh sách cạnh và nút) thành định dạng phù hợp để đưa vào mô hình GNN
# aggregation_type="sum": Phương pháp gộp thông tin từ các nút lân cận:
# "sum": Tổng hợp thông tin từ các nút lân cận bằng cách cộng giá trị
# sparse_matrix=True: Sử dụng ma trận thưa (sparse matrix) để tiết kiệm bộ nhớ và tăng tốc tính toán
dset = dataloader.from_EN_to_GNN(E, N_tot, targets=labels, aggregation_type="sum",
                                 sparse_matrix=True)  # generate the dataset

model(dset)  # Khởi tạo dữ liệu (dset) vào mô hình GNN (model)

# training code
for epoch in range(1, cfg.epochs + 1):
    # Gọi hàm huấn luyện cho mô hình tại epoch hiện tại
    model.train_step(epoch)

    # Sau mỗi 10 epoch, In các chỉ số (metrics) như độ chính xác (accuracy) hoặc độ lỗi (Average loss) tại epoch đó
    if epoch % 10 == 0:
        model.test_step(epoch)

    # Theo dõi kết quả
    if epoch % 100 == 0:  # Show graph visualization every 100 epochs
        plot_graph(E, N_tot)
