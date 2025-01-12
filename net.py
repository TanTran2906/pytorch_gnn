import torch
import torch.nn as nn
import torch.nn.functional as F
import typing


class MLP(nn.Module):
    # input_dim: Số chiều của đầu vào
    # hidden_sizes: Danh sách các kích thước (số nút) của từng lớp ẩn
    # out_dim: Số chiều của đầu ra
    # activation_function: Hàm kích hoạt được áp dụng sau mỗi lớp ẩn (mặc định: nn.Sigmoid)
    # activation_out: Hàm kích hoạt áp dụng sau lớp đầu ra (nếu có, mặc định là None)
    def __init__(self, input_dim, hidden_sizes: typing.Iterable[int], out_dim, activation_function=nn.Sigmoid(),
                 activation_out=None):
        super(MLP, self).__init__()

        # Danh sách chứa số nút của các lớp, bao gồm đầu vào và các lớp ẩn
        # Nếu input_dim = 4, hidden_sizes = [8, 16], thì i_h_sizes = [4, 8, 16]
        i_h_sizes = [input_dim] + hidden_sizes  # add input dim to the iterable
        # Đối tượng nn.Sequential, dùng để tạo một chuỗi các lớp mạng tuần tự(gọi theo thứ tự chúng được thêm vào)
        self.mlp = nn.Sequential()
        # Vòng lặp xây dựng các lớp Linear và thêm hàm kích hoạt
        # Vòng lặp chạy qua từng cặp chiều của i_h_sizes, tạo ra các lớp kết nối tuyến tính (Linear).
        for idx in range(len(i_h_sizes) - 1):
            # Bước 1: Tạo các lớp Linear
            # in_features: Số chiều đầu vào của lớp
            # out_features: Số chiều đầu ra của lớp
            # i_h_sizes=[4, 8, 16] => Lớp 1: nn.Linear(in_features=4, out_features=8),Lớp 2: nn.Linear(in_features=8, out_features=16)
            self.mlp.add_module("layer_{}".format(idx),
                                nn.Linear(in_features=i_h_sizes[idx], out_features=i_h_sizes[idx + 1]))
            # Bước 2: Thêm hàm kích hoạt
            # Sau mỗi lớp Linear, thêm một hàm kích hoạt (activation_function), mặc định là nn.Sigmoid()
            self.mlp.add_module("act_{}".format(idx), activation_function)
            # 2 bước trên sẽ tạo ra:
            # "layer_0": nn.Linear(4, 8)
            # "act_0": activation_function (e.g., nn.ReLU())
            # "layer_1": nn.Linear(8, 16)
            # "act_1": activation_function

        # Thêm lớp đầu ra (out_layer), i_h_sizes[-1]: Số nút của lớp ẩn cuối cùng
        # Lớp này kết nối lớp ẩn cuối cùng với lớp đầu ra
        # i_h_sizes=[4, 8, 16], out_dim=3 => "out_layer": nn.Linear(16, 3)
        self.mlp.add_module("out_layer", nn.Linear(i_h_sizes[-1], out_dim))

        # Thêm hàm kích hoạt cho lớp đầu ra (nếu có)
        if activation_out is not None:
            self.mlp.add_module("out_layer_activation", activation_out)

    # Khởi tạo trọng số của các lớp Linear trong mạng bằng Xavier initialization
    # Xavier initialization: Đảm bảo giá trị khởi tạo phù hợp với kích thước của các lớp, giúp tăng tốc hội tụ
    # nn.init.xavier_normal_: Khởi tạo trọng số với phân phối chuẩn
    def init(self):
        # Duyệt qua tất cả các lớp trong self.mlp, chỉ áp dụng khởi tạo nếu lớp là nn.Linear
        for i, l in enumerate(self.mlp):
            if type(l) == nn.Linear:
                nn.init.xavier_normal_(l.weight)

    # x: Đầu vào của mạng
    # Truyền tiến: Dữ liệu x được truyền qua tất cả các lớp trong self.mlp (theo thứ tự)
    # Kết quả: Trả về đầu ra của mạng
    def forward(self, x):
        return self.mlp(x)


# code from Pedro H. Avelar

class StateTransition(nn.Module):

    # node_state_dim: Kích thước vector trạng thái của mỗi nút
    # node_label_dim: Kích thước vector nhãn của mỗi nút
    # mlp_hidden_dim: Danh sách kích thước các lớp ẩn của MLP
    # activation_function: Hàm kích hoạt, mặc định là Tanh
    def __init__(self,
                 node_state_dim: int,
                 node_label_dim: int,
                 mlp_hidden_dim: typing.Iterable[int],
                 activation_function=nn.Tanh()
                 ):
        super(type(self), self).__init__()
        # Kích thước đầu vào của MLP
        # arc state computation f(l_v: Nhãn nút nguồn, l_n: Nhãn nút đích, x_n: Trạng thái nút đích)
        d_i = node_state_dim + 2 * node_label_dim
        d_o = node_state_dim  # Kích thước đầu ra của MLP
        d_h = list(mlp_hidden_dim)  # if already a list, no change
        self.mlp = MLP(input_dim=d_i, hidden_sizes=d_h, out_dim=d_o, activation_function=activation_function,
                       activation_out=activation_function)  # state transition function, non-linearity also in output

    # thực hiện tính toán trạng thái mới của các nút trong đồ thị dựa trên các cạnh và trạng thái hiện tại
    # node_states, node_labels, edges dạng tensor
    def forward(
            self,
            node_states,
            node_labels,
            edges,
            agg_matrix,
    ):
        # Lấy nhãn và trạng thái của nút từ các cạnh
        src_label = node_labels[edges[:, 0]]
        tgt_label = node_labels[edges[:, 1]]
        tgt_state = node_states[edges[:, 1]]
        # Kết hợp thông tin của cạnh
        # Kết hợp nhãn của nút nguồn, nút đích, và trạng thái của nút đích bằng torch.cat theo chiều cuối cùng (-1)
        edge_states = self.mlp(
            torch.cat(
                [src_label, tgt_label, tgt_state],
                -1
            )
        )

        # Trạng thái mới của các nút sau khi tổng hợp thông tin từ các cạnh
        new_state = torch.matmul(agg_matrix, edge_states)
        return new_state


class GINTransition(nn.Module):

    def __init__(self,
                 node_state_dim: int,
                 node_label_dim: int,
                 mlp_hidden_dim: typing.Iterable[int],
                 activation_function=nn.Tanh()
                 ):
        super(type(self), self).__init__()
        d_i = node_state_dim + node_label_dim
        d_o = node_state_dim
        d_h = list(mlp_hidden_dim)
        self.mlp = MLP(input_dim=d_i, hidden_sizes=d_h, out_dim=d_o, activation_function=activation_function,
                       activation_out=activation_function)  # state transition function, non-linearity also in output

    def forward(
            self,
            node_states,
            node_labels,
            edges,
            agg_matrix,

    ):
        state_and_label = torch.cat(
            [node_states, node_labels],
            -1
        )
        aggregated_neighbourhood = torch.matmul(
            agg_matrix, state_and_label[edges[:, 1]])
        node_plus_neighbourhood = state_and_label + aggregated_neighbourhood
        new_state = self.mlp(node_plus_neighbourhood)
        return new_state


class GINPreTransition(nn.Module):

    def __init__(self,
                 node_state_dim: int,
                 node_label_dim: int,
                 mlp_hidden_dim: typing.Iterable[int],
                 activation_function=nn.Tanh()
                 ):
        super(type(self), self).__init__()
        d_i = node_state_dim + node_label_dim
        d_o = node_state_dim
        d_h = list(mlp_hidden_dim)
        self.mlp = MLP(input_dim=d_i, hidden_sizes=d_h, out_dim=d_o, activation_function=activation_function,
                       activation_out=activation_function)

    def forward(
            self,
            node_states,
            node_labels,
            edges,
            agg_matrix,
    ):
        intermediate_states = self.mlp(
            torch.cat(
                [node_states, node_labels],
                -1
            )
        )
        new_state = (
            torch.matmul(agg_matrix, intermediate_states[edges[:, 1]])
            + torch.matmul(agg_matrix, intermediate_states[edges[:, 0]])
        )
        return new_state
