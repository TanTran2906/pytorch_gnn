# Thực hiện việc huấn luyện một mô hình GNN sử dụng dữ liệu lấy từ một file .mat trong thư mục data
import torch
import torchvision  # Hỗ trợ làm việc với dữ liệu hình ảnh
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import argparse  # Xử lý các tham số dòng lệnh
import utils
import dataloader

from gnn_wrapper import GNNWrapper

import scipy.io  # Thêm thư viện để làm việc với file .mat
# Cấu hình ngẫu nhiên liên quan đến GPU
# # fix random seeds for reproducibility
# SEED = 123
# torch.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# np.random.seed(SEED)


def main():
    # Training settings
    # Tạo một đối tượng ArgumentParser để định nghĩa và xử lý các tham số dòng lệnh mà người dùng có thể truyền khi chạy chương trình
    # description: Chuỗi mô tả về mục đích của chương trình, khi chạy python script.py --help
    parser = argparse.ArgumentParser(description='PyTorch')
    # --epochs: Định nghĩa tham số cho số epoch
    # mặc định là 100 nếu người dùng không chỉ định
    # metavar='N': Tên gợi ý cho tham số trong tài liệu trợ giúp
    # help: Mô tả ý nghĩa của tham số để hiển thị khi chạy lệnh --help
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    # --lr: Tốc độ học (learning rate)
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    # CUDA (Compute Unified Device Architecture) là một nền tảng và mô hình lập trình được NVIDIA phát triển, cho phép sử dụng GPU (Graphics Processing Unit) để tăng tốc các tính toán song song
    # Mặc định: CUDA được bật (default=False), nghĩa là nếu GPU khả dụng, mô hình sẽ được huấn luyện trên GPU
    # action='store_true': Gán giá trị True nếu tham số xuất hiện, ngược lại là False
    parser.add_argument('--no-cuda', action='store_true', default=True,
                        help='disables CUDA training')
    # Chọn GPU cụ thể (theo ID) để huấn luyện
    parser.add_argument('--cuda_dev', type=int, default=0,
                        help='select specific CUDA device for training')
    parser.add_argument('--n_gpu_use', type=int, default=1,
                        help='select number of CUDA device for training')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='logging training status cadency')
    parser.add_argument('--tensorboard', action='store_true', default=True,
                        help='For logging the model in tensorboard')

    # args: Biến chứa tất cả các tham số được truyền từ dòng lệnh dưới dạng thuộc tính
    args = parser.parse_args()

    # use_cuda: Biến boolean cho biết có sử dụng GPU hay không.
    # not args.no_cuda: GPU sẽ được sử dụng nếu --no-cuda không được kích hoạt.
    # torch.cuda.is_available(): Kiểm tra xem GPU có khả dụng không
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if not use_cuda:
        args.n_gpu_use = 0
    # Xác định thiết bị sử dụng (CPU hoặc GPU)
    device = utils.prepare_device(
        n_gpu_use=args.n_gpu_use, gpu_id=args.cuda_dev)

    # Đọc file .mat (Giả sử file có tên 'data.mat')
    # Thay 'path_to_file.mat' bằng đường dẫn chính xác
    # mat_data = scipy.io.loadmat('data\subcli\sub_10_5_200.mat')
    # # In ra các khóa trong file .mat
    # # dict_keys(['__header__', '__version__', '__globals__', 'dataSet'])
    # print("Các biến trong file .mat:", mat_data.keys())

    # # truy cập các dữ liệu trong file .mat như sau:
    # variable = mat_data['dataSet']
    # print("Dữ liệu của biến:", variable)
    # kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # torch.manual_seed(args.seed)
    # # fix random seeds for reproducibility
    # SEED = 123
    # torch.manual_seed(SEED)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # np.random.seed(SEED)

    # configugations
    cfg = GNNWrapper.Config()
    cfg.use_cuda = use_cuda
    cfg.device = device

    cfg.log_interval = args.log_interval
    cfg.tensorboard = args.tensorboard

    # cfg.batch_size = args.batch_size
    # cfg.test_batch_size = args.test_batch_size
    # cfg.momentum = args.momentum

    cfg.dataset_path = './data'  # Đường dẫn đến thư mục chứa tập dữ liệu
    cfg.epochs = args.epochs
    cfg.lrw = args.lr
    cfg.activation = nn.Sigmoid()
    cfg.state_transition_hidden_dims = [10, ]
    cfg.output_function_hidden_dims = [5]
    cfg.state_dim = 10
    cfg.max_iterations = 50
    cfg.convergence_threshold = 0.01
    cfg.graph_based = False
    cfg.log_interval = 10
    cfg.lrw = 0.01
    cfg.task_type = "multiclass"

    # model creation
    model_tr = GNNWrapper(cfg)  # huấn luyện
    model_val = GNNWrapper(cfg)  # kiểm định
    model_tst = GNNWrapper(cfg)  # kiểm tra
    # dataset creation
    # Hàm lấy một tập con (subgraph) từ tập dữ liệu chính
    # set="sub_10_5_200": Định danh của tập con, có thể liên quan đến cấu hình hoặc kích thước tập dữ liệu.
    dset = dataloader.get_subgraph(
        set="sub_10_5_200", aggregation_type="sum", sparse_matrix=True)  # generate the dataset
    model_tr(dset["train"])  # dataset initalization into the GNN
    # state_net Hàm chuyển trạng thái từ mô hình huấn luyện (model_tr).
    # out_net Hàm đầu ra từ mô hình huấn luyện
    # Đồng bộ các hàm chuyển trạng thái và hàm đầu ra giữa các mô hình
    model_val(dset["validation"], state_net=model_tr.gnn.state_transition_function,
              out_net=model_tr.gnn.output_function)  # dataset initalization into the GNN
    model_tst(dset["test"], state_net=model_tr.gnn.state_transition_function,
              out_net=model_tr.gnn.output_function)  # dataset initalization into the GNN

    # training code
    for epoch in range(1, args.epochs + 1):
        model_tr.train_step(epoch)
        if epoch % 10 == 0:
            model_tst.test_step(epoch)
            model_val.valid_step(epoch)
            # model_tst.test_step(epoch)

    # if args.save_model:
    #     torch.save(model.gnn.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
