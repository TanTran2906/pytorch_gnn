import torch
import torch.nn as nn
import torch.nn.functional as F
import dataloader
import torch.optim as optim
from abc import ABCMeta, abstractmethod
from utils import Accuracy
from torch.utils.tensorboard import SummaryWriter
import torchvision
from utils import matplotlib_imshow
import utils
from pygnn import GNN


class GNNWrapper:
    # Lưu trữ tất cả các tham số cấu hình của mô hình GNN
    class Config:
        def __init__(self):
            # lưu trữ thông tin về thiết bị mà mô hình sẽ chạy trên đó (CPU hoặc GPU)
            self.device = None
            # xác định xem có sử dụng CUDA (GPU) hay không
            self.use_cuda = None
            self.dataset_path = None
            # xác định khoảng thời gian giữa các lần ghi log trong quá trình huấn luyện,nếu log_interval = 10, thì mỗi 10 epoch
            self.log_interval = None
            self.tensorboard = None
            # Xác định loại bài toán mà mô hình đang giải quyết
            self.task_type = None

            # hyperparams
            # tốc độ học của mô hình. Nó quyết định mức độ thay đổi của các trọng số trong quá trình huấn luyện
            self.lrw = None
            # hàm mất mát (loss function) sẽ được sử dụng trong quá trình huấn luyện
            self.loss_f = None
            self.epochs = None
            self.convergence_threshold = None
            # Số lượng tối đa các phép lặp trong quá trình huấn luyện
            self.max_iterations = None
            self.n_nodes = None  # Số lượng các nút (nodes)
            self.state_dim = None  # số chiều của các đặc trưng của các nút trong đồ thị
            # Số chiều của nhãn (labels) mà mô hình dự đoán
            self.label_dim = None
            self.output_dim = None  # Số chiều của đầu ra mô hình
            self.graph_based = False  # xác định liệu mô hình có dựa trên đồ thị hay không
            self.activation = torch.nn.Tanh()
            # Các kích thước của các lớp ẩn (hidden layers) cho phần chuyển trạng thái của mô hình
            self.state_transition_hidden_dims = None
            # Các kích thước của các lớp ẩn cho phần đầu ra của mô hình

            self.output_function_hidden_dims = None
            # học bán giám sát
            self.task_type = "semisupervised"

            # optional
            # self.loss_w = 1.
            # self.energy_weight = 0.
            # self.l2_weight = 0.

    def __init__(self, config: Config):
        self.config = config

        # to be populated
        # đối tượng tối ưu hóa (optimizer) sử dụng để cập nhật trọng số trong quá trình huấn luyện
        self.optimizer = None
        # hàm mất mát (loss function) dùng để tính toán lỗi trong quá trình huấn luyện

        self.criterion = None
        # đối tượng để tải dữ liệu huấn luyện và kiểm tra
        self.train_loader = None
        self.test_loader = None

        # SummaryWriter là một lớp trong PyTorch dùng để ghi lại các giá trị như độ chính xác, loss, histogram của các trọng số, v.v. vào thư mục
        if self.config.tensorboard:
            self.writer = SummaryWriter('logs/tensorboard')
        self.first_flag_writer = True

    # cho phép đối tượng GNNWrapper được gọi như một hàm,thực hiện các bước khởi tạo và cấu hình mô hình
    def __call__(self, dset, state_net=None, out_net=None):
        # handle the dataset info
        self._data_loader(dset)
        self.gnn = GNN(self.config, state_net, out_net).to(self.config.device)
        self._criterion()
        self._optimizer()
        self._accuracy()

    def _data_loader(self, dset):  # handle dataset data and metadata
        # xác định thiết bị mà dữ liệu sẽ được chuyển tới
        self.dset = dset.to(self.config.device)
        # Lưu các thuộc tính như  số lượng nhãn (node_label_dim), số lượng nút (num_nodes), và số lớp đầu ra (num_classes) từ dữ liệu vào cấu hình
        self.config.label_dim = self.dset.node_label_dim
        self.config.n_nodes = self.dset.num_nodes
        self.config.output_dim = self.dset.num_classes

    def _optimizer(self):
        # for name, param in self.gnn.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data)
        # exit()
        # Sử dụng thuật toán tối ưu hóa Adam với learning rate lấy từ self.config.lrw để tối ưu hóa các tham số của mô hình
        self.optimizer = optim.Adam(self.gnn.parameters(), lr=self.config.lrw)
        # self.optimizer = optim.SGD(self.gnn.parameters(), lr=self.config.lrw)

    def _criterion(self):
        # hàm mất mát: tính toán mức độ sai khác giữa nhãn thực tế và dự đoán của mô hình
        self.criterion = nn.CrossEntropyLoss()

    def _accuracy(self):
        self.TrainAccuracy = Accuracy(type=self.config.task_type)
        self.ValidAccuracy = Accuracy(type=self.config.task_type)
        self.TestAccuracy = Accuracy(type=self.config.task_type)

    def train_step(self, epoch):
        # Đặt mô hình vào chế độ huấn luyện: Trong PyTorch, train() cho phép mô hình sử dụng các cơ chế như dropout và batch normalization
        self.gnn.train()
        data = self.dset
        # Đặt lại gradient của các tham số trong mô hình về 0 trước khi tính toán lại gradient trong quá trình huấn luyện
        self.optimizer.zero_grad()
        # Đặt lại các chỉ số độ chính xác trước mỗi bước huấn luyện mới
        self.TrainAccuracy.reset()
        # Tính toán đầu ra của mô hình
        # Kiểm tra xem có sử dụng phương pháp dựa trên đồ thị hay không
        # output là kết quả đầu ra của mô hình ( giá trị dự đoán),
        # iterations là số lần lặp trong quá trình tính toán đầu ra
        if self.config.graph_based:
            output, iterations = self.gnn(
                data.edges, data.agg_matrix, data.node_labels, graph_agg=data.graph_node)
        else:
            output, iterations = self.gnn(
                data.edges, data.agg_matrix, data.node_labels)
        # loss computation - semisupervised
        loss = self.criterion(output, data.targets)
        # Lan truyền ngược và tối ưu hóa
        # Tính toán gradient của hàm mất mát đối với các tham số của mô hình
        loss.backward()
        # Cập nhật các tham số của mô hình bằng cách sử dụng các gradient đã tính toán trong bước trước
        self.optimizer.step()

        # # Cập nhật và tính toán độ chính xác
        # batch_acc = self.TrainAccuracy.update((output, target), batch_compute=True)
        with torch.no_grad():  # Bật chế độ không tính toán gradient
            # accuracy_train = torch.mean(
            #     (torch.argmax(output[data.idx_train], dim=-1) == data.targets[data.idx_train]).float())
            # Cập nhật độ chính xác huấn luyện với kết quả dự đoán (output) và nhãn thực tế (data.targets)
            self.TrainAccuracy.update(output, data.targets)
            # Tính toán độ chính xác từ các cập nhật trước đó
            accuracy_train = self.TrainAccuracy.compute()

            # In kết quả và ghi vào TensorBoard
            if epoch % self.config.log_interval == 0:
                print(
                    'Train Epoch: {} \t Mean Loss: {:.6f}\tAccuracy Full Batch: {:.6f} \t  Best Accuracy : {:.6f}  \t Iterations: {}'.format(
                        epoch, loss, accuracy_train, self.TrainAccuracy.get_best(), iterations))

                if self.config.tensorboard:
                    self.writer.add_scalar('Training Accuracy',
                                           accuracy_train,
                                           epoch)
                    self.writer.add_scalar('Training Loss',
                                           loss,
                                           epoch)
                    self.writer.add_scalar('Training Iterations',
                                           iterations,
                                           epoch)

                    for name, param in self.gnn.named_parameters():
                        self.writer.add_histogram(name, param, epoch)
        # self.TrainAccuracy.reset()

    # dự đoán đầu ra từ mô hình GNN

    def predict(self, edges, agg_matrix, node_labels):
        return self.gnn(edges, agg_matrix, node_labels)

    def predict(self, edges, agg_matrix, node_labels, graph_node):
        return self.gnn(edges, agg_matrix, node_labels, graph_agg=graph_node)

    """Tương tự train_step
        Mô hình không cập nhật gradient(torch.no_grad()), chỉ sử dụng các tham số đã học để thực hiện dự đoán
    """

    def test_step(self, epoch):
        # TEST
        # Đặt mô hình ở chế độ đánh giá (eval), các cơ chế như dropout và batch normalization bị tắt
        self.gnn.eval()
        data = self.dset
        self.TestAccuracy.reset()
        with torch.no_grad():
            if self.config.graph_based:
                output, iterations = self.gnn(
                    data.edges, data.agg_matrix, data.node_labels, graph_agg=data.graph_node)
            else:
                output, iterations = self.gnn(
                    data.edges, data.agg_matrix, data.node_labels)
            test_loss = self.criterion(output, data.targets)

            self.TestAccuracy.update(output, data.targets)
            acc_test = self.TestAccuracy.compute()
            # acc_test = torch.mean(
            #     (torch.argmax(output[data.idx_test], dim=-1) == data.targets[data.idx_test]).float())

            if epoch % self.config.log_interval == 0:
                print('Test set: Average loss: {:.4f}, Accuracy:  ({:.4f}%) , Best Accuracy:  ({:.4f}%)'.format(
                    test_loss, acc_test, self.TestAccuracy.get_best()))

                if self.config.tensorboard:
                    self.writer.add_scalar('Test Accuracy',
                                           acc_test,
                                           epoch)
                    self.writer.add_scalar('Test Loss',
                                           test_loss,
                                           epoch)
                    self.writer.add_scalar('Test Iterations',
                                           iterations,
                                           epoch)

    def valid_step(self, epoch):
        # TEST
        self.gnn.eval()
        data = self.dset
        self.ValidAccuracy.reset()
        with torch.no_grad():
            if self.config.graph_based:
                output, iterations = self.gnn(
                    data.edges, data.agg_matrix, data.node_labels, graph_agg=data.graph_node)
            else:
                output, iterations = self.gnn(
                    data.edges, data.agg_matrix, data.node_labels)
            test_loss = self.criterion(output, data.targets)

            self.ValidAccuracy.update(output, data.targets)
            acc_valid = self.ValidAccuracy.compute()
            # acc_test = torch.mean(
            #     (torch.argmax(output[data.idx_test], dim=-1) == data.targets[data.idx_test]).float())

            if epoch % self.config.log_interval == 0:
                print('Valid set: Average loss: {:.4f}, Accuracy:  ({:.4f}%) , Best Accuracy:  ({:.4f}%)'.format(
                    test_loss, acc_valid, self.ValidAccuracy.get_best()))

                if self.config.tensorboard:
                    self.writer.add_scalar('Valid Accuracy',
                                           acc_valid,
                                           epoch)
                    self.writer.add_scalar('Valid Loss',
                                           test_loss,
                                           epoch)
                    self.writer.add_scalar('Valid Iterations',
                                           iterations,
                                           epoch)


class SemiSupGNNWrapper(GNNWrapper):
    class Config:
        def __init__(self):
            self.device = None
            self.use_cuda = None
            self.dataset_path = None
            self.log_interval = None
            self.tensorboard = None
            self.task_type = None

            # hyperparams
            self.lrw = None
            self.loss_f = None
            self.epochs = None
            self.convergence_threshold = None
            self.max_iterations = None
            self.n_nodes = None
            self.state_dim = None
            self.label_dim = None
            self.output_dim = None
            self.graph_based = False
            self.activation = torch.nn.Tanh()
            self.state_transition_hidden_dims = None
            self.output_function_hidden_dims = None

            # optional
            # self.loss_w = 1.
            # self.energy_weight = 0.
            # self.l2_weight = 0.

    def __init__(self, config: Config):
        super().__init__(config)

    def _data_loader(self, dset):  # handle dataset data and metadata
        self.dset = dset.to(self.config.device)
        self.config.label_dim = self.dset.node_label_dim
        self.config.n_nodes = self.dset.num_nodes
        self.config.output_dim = self.dset.num_classes

    def _accuracy(self):
        self.TrainAccuracy = Accuracy(type="semisupervised")
        self.ValidAccuracy = Accuracy(type="semisupervised")
        self.TestAccuracy = Accuracy(type="semisupervised")

    def train_step(self, epoch):
        self.gnn.train()
        data = self.dset
        self.optimizer.zero_grad()
        self.TrainAccuracy.reset()
        # output computation
        if self.config.graph_based:
            output, iterations = self.gnn(
                data.edges, data.agg_matrix, data.node_labels, data.graph_node)
        else:
            output, iterations = self.gnn(
                data.edges, data.agg_matrix, data.node_labels)
        # loss computation - semisupervised
        loss = self.criterion(
            output[data.idx_train], data.targets[data.idx_train])

        loss.backward()

        # with torch.no_grad():
        #     for name, param in self.gnn.named_parameters():
        #         if "state_transition_function" in name:
        #             #self.writer.add_histogram("gradient " + name, param.grad, epoch)
        #             param.grad = 0*  param.grad

        self.optimizer.step()

        # # updating accuracy
        # batch_acc = self.TrainAccuracy.update((output, target), batch_compute=True)
        with torch.no_grad():  # Accuracy computation
            # accuracy_train = torch.mean(
            #     (torch.argmax(output[data.idx_train], dim=-1) == data.targets[data.idx_train]).float())
            self.TrainAccuracy.update(output, data.targets, idx=data.idx_train)
            accuracy_train = self.TrainAccuracy.compute()

            if epoch % self.config.log_interval == 0:
                print(
                    'Train Epoch: {} \t Mean Loss: {:.6f}\tAccuracy Full Batch: {:.6f} \t  Best Accuracy : {:.6f}  \t Iterations: {}'.format(
                        epoch, loss, accuracy_train, self.TrainAccuracy.get_best(), iterations))

                if self.config.tensorboard:
                    self.writer.add_scalar('Training Accuracy',
                                           accuracy_train,
                                           epoch)
                    self.writer.add_scalar('Training Loss',
                                           loss,
                                           epoch)
                    self.writer.add_scalar('Training Iterations',
                                           iterations,
                                           epoch)
                    for name, param in self.gnn.named_parameters():
                        self.writer.add_histogram(name, param, epoch)
                        self.writer.add_histogram(
                            "gradient " + name, param.grad, epoch)
        # self.TrainAccuracy.reset()
        return output  # used for plotting

    def predict(self, edges, agg_matrix, node_labels):
        return self.gnn(edges, agg_matrix, node_labels)

    def test_step(self, epoch):
        # TEST
        self.gnn.eval()
        data = self.dset
        self.TestAccuracy.reset()
        with torch.no_grad():
            if self.config.graph_based:
                output, iterations = self.gnn(
                    data.edges, data.agg_matrix, data.node_labels, data.graph_node)
            else:
                output, iterations = self.gnn(
                    data.edges, data.agg_matrix, data.node_labels)
            test_loss = self.criterion(
                output[data.idx_test], data.targets[data.idx_test])

            self.TestAccuracy.update(output, data.targets, idx=data.idx_test)
            acc_test = self.TestAccuracy.compute()
            # acc_test = torch.mean(
            #     (torch.argmax(output[data.idx_test], dim=-1) == data.targets[data.idx_test]).float())

            if epoch % self.config.log_interval == 0:
                print('Test set: Average loss: {:.4f}, Accuracy:  ({:.4f}%) , Best Accuracy:  ({:.4f}%)'.format(
                    test_loss, acc_test, self.TestAccuracy.get_best()))

                if self.config.tensorboard:
                    self.writer.add_scalar('Test Accuracy',
                                           acc_test,
                                           epoch)
                    self.writer.add_scalar('Test Loss',
                                           test_loss,
                                           epoch)
                    self.writer.add_scalar('Test Iterations',
                                           iterations,
                                           epoch)

    def valid_step(self, epoch):
        # TEST
        self.gnn.eval()
        data = self.dset
        self.ValidAccuracy.reset()
        with torch.no_grad():
            if self.config.graph_based:
                output, iterations = self.gnn(
                    data.edges, data.agg_matrix, data.node_labels, data.graph_node)
            else:
                output, iterations = self.gnn(
                    data.edges, data.agg_matrix, data.node_labels)
            test_loss = self.criterion(
                output[data.idx_valid], data.targets[data.idx_valid])

            self.ValidAccuracy.update(output, data.targets, idx=data.idx_valid)
            acc_valid = self.ValidAccuracy.compute()
            # acc_test = torch.mean(
            #     (torch.argmax(output[data.idx_test], dim=-1) == data.targets[data.idx_test]).float())

            if epoch % self.config.log_interval == 0:
                print('Valid set: Average loss: {:.4f}, Accuracy:  ({:.4f}%) , Best Accuracy:  ({:.4f}%)'.format(
                    test_loss, acc_valid, self.ValidAccuracy.get_best()))

                if self.config.tensorboard:
                    self.writer.add_scalar('Valid Accuracy',
                                           acc_valid,
                                           epoch)
                    self.writer.add_scalar('Valid Loss',
                                           test_loss,
                                           epoch)
                    self.writer.add_scalar('Valid Iterations',
                                           iterations,
                                           epoch)
