import torch
import torch.nn as nn
import numpy as np
import argparse
import dataloader
from gnn_wrapper import GNNWrapper, SemiSupGNNWrapper
import matplotlib.pyplot as plt
import networkx as nx
import utils

# simple  visualization of the graph given the arcs


def plot_graph(E):
    g = nx.Graph()
    g.add_nodes_from(range(np.max(E) + 1))
    g.add_edges_from(E[:, :2])
    nx.draw_spring(g, cmap=plt.get_cmap('Set1'), with_labels=True)
    plt.show()


###############################################################################################
### SIMPLE EXAMPLE ON EN_Matrix usage for the creation of a dataset for graph classification ##
###############################################################################################
# Mỗi nút trong đồ thị sẽ được gán một vector đặc trưng ngẫu nhiên có 10 chiều
feature_dims = 10

# GRAPH #1

# List of edges in the first graph - NOTICE: the last column contains the id of the graph to which the arc belongs
# syntax for every row:  [starting_node, ending_node, graph_id]
e = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [
    1, 2, 0], [1, 3, 0], [2, 3, 0], [2, 4, 0]]
# it is an undirected graph, so we need to add to the list also the arcs in the other  direction
e.extend([[i, j, num] for j, i, num in e])

# convert to numpy
E = np.asarray(e)

# Tìm số lượng nút trong đồ thị bằng cách lấy giá trị lớn nhất trong E
nodes1 = np.max(E) + 1
# Tạo ma trận đặc trưng N cho các nút, Kích thước của N là (nodes1, feature_dims)
N = np.random.rand(nodes1, feature_dims)
# Thêm một cột mới vào ma trận đặc trưng N để biểu diễn ID của đồ thị
N = np.concatenate((N, np.zeros((nodes1, 1), dtype=np.float32)), axis=1)


plot_graph(E)

################################################
# create another GRAPH #2


# List of edges in the second graph - NOTICE: the last column contains the id of the graph to which the arc belongs
# syntax for every row:  [starting_node, ending_node, graph_id]
e1 = [[0, 2, 1], [0, 3, 1], [1, 2, 1], [1, 3, 1], [2, 3, 1]]

# number of nodes in this graph is equal to the max id + 1
nodes2 = np.max(e1) + 1

# undirected graph, adding other direction
e1.extend([[i, j, num] for j, i, num in e1])
# reindexing node ids based on the dimension of previous graph (using unique ids)
e2 = [[a + nodes1, b + nodes1, num] for a, b, num in e1]

plot_graph(np.asarray(e1))

E1 = np.asarray(e2)

# create a matrix containing the features of each node (random in this simple case)
N1 = np.random.rand(nodes2, feature_dims)

# adding the last column that represent the id (1) of the graph to which the node belongs
N1 = np.concatenate((N1, np.ones((nodes2, 1), dtype=np.float32)), axis=1)

# Kết hợp danh sách cạnh của đồ thị #1 (E) và đồ thị #2 (E1) để tạo ra một đồ thị duy nhất chứa cả hai đồ thị

E = np.concatenate((E, E1), axis=0)

# plot the graph containing the two separate graphs
plot_graph(E)


# Kết hợp ma trận đặc trưng của 2 đồ thị

N_tot = np.concatenate((N, N1), axis=0)

# tạo ra một mảng ngẫu nhiên gồm 2 phần tử, với mỗi phần tử là một số nguyên thuộc khoảng [0, 2) (tức là 0 hoặc 1, ví dụ: [0, 1], [1,1],[0,0])
target = np.random.randint(2, size=(2,))

cfg = GNNWrapper.Config()
cfg.use_cuda = True
cfg.device = utils.prepare_device(n_gpu_use=1, gpu_id=0)
cfg.tensorboard = False
cfg.epochs = 500

cfg.activation = nn.Tanh()
cfg.state_transition_hidden_dims = [5, ]
cfg.output_function_hidden_dims = [5]
cfg.state_dim = 5
cfg.max_iterations = 50
cfg.convergence_threshold = 0.01

# NOTICE:  graph-focused task, set this to TRUE!
# Đặt chế độ phân loại đồ thị (mỗi nhãn gắn với một đồ thị, không phải từng nút)
cfg.graph_based = True

cfg.log_interval = 10
cfg.task_type = "multiclass"
cfg.lrw = 0.001

# model creation
model = GNNWrapper(cfg)
# dataset creation
dset = dataloader.from_EN_to_GNN(E, N_tot, targets=target, aggregation_type="sum",
                                 sparse_matrix=True)  # generate the dataset

model(dset)  # dataset initalization into the GNN

# training code
for epoch in range(1, cfg.epochs + 1):
    model.train_step(epoch)

    if epoch % 10 == 0:
        model.test_step(epoch)
