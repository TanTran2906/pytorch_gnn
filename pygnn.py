import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torch.nn import init
import math
from net import MLP, StateTransition


class GNN(nn.Module):
    # Hàm khởi tạo lớp GNN
    def __init__(self, config, state_net=None, out_net=None):
        super(GNN, self).__init__()

        # config: Chứa các tham số cấu hình như:
        # convergence_threshold: Ngưỡng hội tụ
        # max_iterations: Số vòng lặp tối đa cho quá trình cập nhật trạng thái
        # n_nodes: Số lượng nút trong đồ thị
        # state_dim: Kích thước vector trạng thái của mỗi nút
        # label_dim: Kích thước vector nhãn của mỗi nút
        # output_dim: Kích thước vector đầu ra
        # state_transition_hidden_dims: Danh sách các kích thước ẩn của mạng chuyển trạng thái
        # output_function_hidden_dims: Danh sách các kích thước ẩn của mạng tính đầu ra
        # device: Thiết bị sử dụng (CPU hoặc GPU)
        # activation: Hàm kích hoạt (ví dụ: ReLU, Tanh, v.v.)
        # graph_based: Chỉ định có sử dụng ma trận tổng hợp đồ thị hay không
        self.config = config
        # hyperparameters and general properties
        self.convergence_threshold = config.convergence_threshold
        self.max_iterations = config.max_iterations
        self.n_nodes = config.n_nodes
        self.state_dim = config.state_dim
        self.label_dim = config.label_dim
        self.output_dim = config.output_dim
        self.state_transition_hidden_dims = config.state_transition_hidden_dims
        self.output_function_hidden_dims = config.output_function_hidden_dims

        # node state initialization
        # Tạo hai tensor để lưu trữ trạng thái của các nút:
        # Trạng thái ban đầu của các nút,tất cả là 0,lưu trên thiết bị cấu hình như CPU hoặc GPU
        self.node_state = torch.zeros(
            *[self.n_nodes, self.state_dim]).to(self.config.device)  # (n,d_n)
        # Lưu trữ trạng thái hội tụ cuối cùng của các nút sau khi thực hiện vòng lặp cập nhật
        self.converged_states = torch.zeros(
            *[self.n_nodes, self.state_dim]).to(self.config.device)
        # state and output transition functions
        # Xác định hàm chuyển trạng thái giữa các nút, tức là cách thông tin được lan truyền qua đồ thị
        if state_net is None:
            self.state_transition_function = StateTransition(self.state_dim, self.label_dim,
                                                             mlp_hidden_dim=self.state_transition_hidden_dims,
                                                             activation_function=config.activation)
        else:  # Sử dụng một mạng tùy chỉnh do người dùng cung cấp, từ tham số đầu vào
            self.state_transition_function = state_net
        # Xác định cách tính đầu ra từ trạng thái hội tụ của các nút
        if out_net is None:
            self.output_function = MLP(
                self.state_dim, self.output_function_hidden_dims, self.output_dim)
        else:
            self.output_function = out_net
        # Xác định có sử dụng ma trận tổng hợp trên toàn đồ thị (graph_agg) trong tính toán đầu ra hay không
        # graph_based = True:Trạng thái hội tụ của nút sẽ được tổng hợp thêm thông tin từ toàn đồ thị trước khi tính đầu ra
        self.graph_based = self.config.graph_based

    # khởi tạo lại các tham số của các lớp mạng con trong mô hình
    # đảm bảo rằng các trọng số của mô hình không bị lệch hoặc không bị phụ thuộc vào trạng thái ban đầu
    def reset_parameters(self):

        self.state_transition_function.mlp.init()
        self.output_function.init()

    # các trạng thái của các nút được cập nhật qua nhiều lần lặp cho đến khi hội tụ hoặc đạt số lần lặp tối đa
    # node_states, node_labels, edges dạng tensor
    # agg_matrix: Ma trận tổng hợp (aggregation matrix) giúp tập hợp thông tin từ các nút láng giềng
    # graph_agg: Một ma trận tổng hợp đồ thị (nếu có), dùng để tổng hợp thông tin từ các nút trong đồ thị
    def forward(self,
                edges,
                agg_matrix,
                node_labels,
                node_states=None,
                graph_agg=None
                ):
        # Khởi tạo số lần lặp,trạng thái nút

        n_iterations = 0
        # convergence loop
        # state initialization
        # Nếu node_states được truyền vào, nó sẽ được sử dụng; nếu không, sử dụng trạng thái mặc định (self.node_state)
        node_states = self.node_state if node_states is None else node_states
        # Nếu node_states được truyền vào, nó sẽ được sử dụng; nếu không, sử dụng trạng thái mặc định (self.node_state)
        while n_iterations < self.max_iterations:
            # Tính toán trạng thái mới của các nút
            new_state = self.state_transition_function(
                node_states, node_labels, edges, agg_matrix)
            n_iterations += 1
            # convergence condition
            # Điều kiện hội tụ:
            with torch.no_grad():
                # torch.norm tính toán độ lớn của sự khác biệt giữa trạng thái mới (new_state) và trạng thái cũ (node_states) theo chiều dim=1 (theo các hàng)
                distance = torch.norm(input=new_state - node_states,
                                      dim=1)  # checked, they are the same (in cuda, some bug)
                # kiểm tra xem khoảng cách giữa trạng thái mới và cũ có nhỏ hơn ngưỡng hội tụ hay không
                check_min = distance < self.convergence_threshold
            node_states = new_state  # Cập nhật trạng thái nút

            # Kiểm tra hội tụ: Nếu tất cả các khoảng cách giữa trạng thái mới và cũ đều nhỏ hơn ngưỡng hội tụ, vòng lặp sẽ dừng lại
            if check_min.all():
                break
        # Lưu trạng thái hội tụ, sau khi vòng lặp hội tụ hoặc đạt số lần lặp tối đa
        states = node_states
        self.converged_states = states
        # Tổng hợp thông tin theo đồ thị (nếu có): Nếu cấu hình graph_based là True, sử dụng graph_agg để tổng hợp trạng thái của các nút
        if self.graph_based:
            states = torch.matmul(graph_agg, node_states)

        # Dùng output_function (một MLP) để tính toán đầu ra của mô hình từ trạng thái của các nút.
        output = self.output_function(states)
        # n_iterations: Số lần lặp mà mô hình đã thực hiện trước khi hội tụ hoặc đạt số lần lặp tối đa
        return output, n_iterations
