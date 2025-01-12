import numpy as np
from abc import abstractmethod
from torch.utils.data import DataLoader
import torch
from torchvision import datasets, transforms
import networkx as nx
import typing
import scipy
import scipy.io as spio
import numpy as np
import os


def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''

    def _check_keys(d):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in d:
            if isinstance(d[key], spio.matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        '''
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        '''
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list

    data = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


# train = loadmat('multi1')

# thanks Pedro H. Avelar
"""
 Nhận đầu vào là đồ thị G, loại tổng hợp thông tin các nút (tức là cách tính tổng hợp giữa các hàng xóm của mỗi nút) 
 và một cờ sparse_matrix cho biết liệu ma trận kết quả có nên là ma trận thưa (sparse matrix) hay không
"""

# chuyển đổi một đồ thị networkx thành dạng phù hợp với GNN


def nx_to_format(G, aggregation_type, sparse_matrix=True):
    e = len(G.edges)  # số cạnh
    n = len(G.nodes)  # số node

    # edges = torch.LongTensor(list(G.edges)),nếu G.edges có các cạnh là [(2, 3), (1, 4), (2, 1)], thì edg sau khi sắp xếp sẽ là [(1, 4), (2, 1), (2, 3)]
    # Sắp xếp cạnh (mỗi cạnh là một tuple (node_i, node_j)) và chuyển đổi thành torch.LongTensor, giúp việc xử lý và tính toán sau này dễ dàng hơn
    # tensor([[1, 4],
    #     [2, 1],
    #     [2, 3]])
    edg = sorted(list(G.edges))
    edges = torch.LongTensor(edg)
    # nx.to_numpy_array(G) chuyển đổi đồ thị G thành một ma trận kề, với các giá trị trong ma trận là 1 nếu có cạnh giữa hai nút và 0 nếu không có cạnh
    adj_matrix = np.asarray(nx.to_numpy_array(G))

    if aggregation_type == "sum":
        pass  # Không thay đổi ma trận kề, giữ nguyên các giá trị
    elif aggregation_type == "degreenorm":
        # Áp dụng chuẩn hóa theo bậc của các nút. Cụ thể, tính tổng các giá trị trong mỗi cột (tương ứng với bậc của các nút) và chia các giá trị trong ma trận kề cho tổng này
        row_sum = np.sum(adj_matrix, axis=0, keepdims=True)
        adj_matrix = adj_matrix / row_sum
    elif aggregation_type == "symdegreenorm":
        # Phương pháp chuẩn hóa bậc đối xứng chưa được triển khai, nên ném ra lỗi NotImplementedError
        raise NotImplementedError(
            "Symmetric degree normalization not yet implemented")
    else:
        raise ValueError("Invalid neighbour aggregation type")
    # Tạo ra một ma trận tổng hợp (aggregation matrix) từ ma trận kề của đồ thị
    # Ma trận thưa (Sparse matrix): Dùng khi đồ thị rất lớn và có nhiều cạnh có trọng số bằng 0 =>Ma trận đặc (Dense matrix): Dùng khi đồ thị nhỏ hơn hoặc khi không quan tâm đến việc tiết kiệm bộ nhớ
    if sparse_matrix:
        # Tạo chỉ số (index) của ma trận thưa
        # agg_matrix_i là một tensor chứa:
        # Phần đầu tiên là danh sách các nút nguồn (s)
        # Phần tử thứ hai là một danh sách các chỉ số từ 0 đến e-1
        # tensor([[0, 1, 2],
        # [0, 1, 2]])
        agg_matrix_i = torch.LongTensor(
            [[s for s, t in G.edges], list(range(e))])
        # Tạo giá trị (value) của ma trận thưa
        # adj_matrix[s, t] là giá trị của ma trận kề tại vị trí (s, t)
        agg_matrix_v = torch.FloatTensor(
            [adj_matrix[s, t] for s, t in G.edges])
        # torch.sparse.FloatTensor() tạo ra một ma trận thưa có kích thước [n, e], trong đó n là số lượng nút và e là số lượng cạnh.
        # Ma trận này chứa các trọng số từ ma trận kề
        agg_matrix = torch.sparse.FloatTensor(
            agg_matrix_i, agg_matrix_v, torch.Size([n, e]))
    # Ma trận đặc (Dense matrix): Dùng khi đồ thị nhỏ hơn hoặc khi không quan tâm đến việc tiết kiệm bộ nhớ
    else:
        # tạo ra một ma trận đặc có kích thước [n, e] (số nút và số cạnh). Mỗi phần tử trong ma trận này được khởi tạo bằng 0
        agg_matrix = torch.zeros(*[n, e])
        # Điền giá trị của ma trận kề vào ma trận đặc tại vị trí tương ứng với nút nguồn s và cạnh i
        for i, (s, t) in enumerate(edg):
            agg_matrix[s, i] = adj_matrix[s, t]

    return edges, agg_matrix


class Dataset:
    def __init__(
            self,
            name,
            num_nodes,
            num_edges,
            label_dim,
            is_multiclass,
            num_classes,
            edges,
            agg_matrix,
            node_labels,
            targets,
            idx_train=None,
            idx_valid=None,
            idx_test=None,
            graph_node=None
    ):
        self.name = name
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.node_label_dim = label_dim
        self.num_classes = num_classes
        self.is_multiclass = is_multiclass
        self.edges = edges
        self.agg_matrix = agg_matrix
        self.node_labels = node_labels
        self.targets = targets
        self.idx_train = idx_train
        self.idx_valid = idx_valid
        self.idx_test = idx_test
        self.graph_node = graph_node

    def cuda(self):
        self.edges, self.agg_matrix, self.node_labels, self.targets, self.idx_train, self.idx_test, self.graph_node = map(
            lambda x: x.cuda() if x is not None else None,
            [self.edges, self.agg_matrix, self.node_labels, self.targets, self.idx_train, self.idx_test,
             self.graph_node]
        )
        return self

    def cpu(self):

        return self

    def to(self, device):
        if "cuda" in device.type:
            torch.cuda.set_device(device)
            return self.cuda()
        else:
            return self.cpu()


def get_twochains(num_nodes_per_graph=50, pct_labels=.1, pct_valid=.5, aggregation_type="sum", sparse_matrix=True):
    G1 = nx.generators.classic.path_graph(num_nodes_per_graph)
    G2 = nx.generators.classic.path_graph(num_nodes_per_graph)

    G = nx.disjoint_union(G1, G2)
    G = G.to_directed()

    e = len(G.edges)
    n = len(G.nodes)

    edges, agg_matrix = nx_to_format(G, aggregation_type, sparse_matrix)

    is_multilabel = False
    n_classes = 2
    d_l = 1
    node_labels = torch.zeros(*[n, d_l])
    # node_labels = torch.eye(n)
    targets = torch.tensor(np.array(
        ([0] * (n // 2)) + ([1] * (n // 2)), dtype=np.int64), dtype=torch.long)

    idx = np.random.permutation(np.arange(n))
    idx_trainval = idx[:int(n * pct_labels)]
    idx_train = torch.LongTensor(
        idx_trainval[:-int(len(idx_trainval) * pct_valid)])
    idx_valid = torch.LongTensor(
        idx_trainval[-int(len(idx_trainval) * pct_valid):])  # TODO wht is he doing, why with BoolTensro is strange?
    idx_test = torch.LongTensor(idx[int(n * pct_labels):])

    return Dataset(
        "twochains",
        n,
        e,
        d_l,
        is_multilabel,
        n_classes,
        edges,
        agg_matrix,
        node_labels,
        targets,
        idx_train,
        idx_valid,
        idx_test,
    )


############## SSE ################

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def read_sse_ids(percentage=None, dataset=None):
    def _internal(file):
        ids = []
        with open(os.path.join(dataset, file), 'r') as f:
            for line in f:
                ids.append(int(line.strip()))
        return ids

    if percentage:
        train_ids = _internal(
            "train_idx-{}.txt".format(
                percentage))  # list, each element a row of the file => id of the graph belonging to train set
        test_ids = _internal("test_idx-{}.txt".format(percentage))

    return train_ids, test_ids


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def get_twochainsSSE(aggregation_type, percentage=0.9, dataset="data/n-chains-connect", node_has_feature=False,
                     train_file="train_idx-", test_file="test_idx-", sparse_matrix=True):
    import os
    print('Loading dataset: {}'.format(dataset))
    graph_info = "meta.txt"
    neigh = "adj_list.txt"
    labels_file = "label.txt"
    # loading targets

    targets = np.loadtxt(os.path.join(dataset, labels_file))
    targets = torch.tensor(np.argmax(targets, axis=1), dtype=torch.long)

    with open(os.path.join(dataset, graph_info), 'r') as f:
        # (ex. MUTAG - 23 2) number of nodes in the graph, target of the graph
        info = f.readline().strip().split()
        if node_has_feature:
            # n == number of nodes, l label (target) of the graph
            n_nodes, l, n_feat = [int(w) for w in info]
        else:
            # n == number of nodes, l label (target) of the graph
            n_nodes, l = [int(w) for w in info]
    # load adj_list
    if node_has_feature:
        features = np.loadtxt(os.path.join(dataset, "features.txt"))
    else:
        # zero feature else
        features = np.zeros((n_nodes, 1), dtype=np.float32)

    with open(os.path.join(dataset, neigh), 'r') as f:

        g = nx.Graph()  # netxgraph
        node_features = []
        # n_edges = 0  # edges in the graph
        for j in range(n_nodes):
            # for every row of the current graph  create the graph itself
            g.add_node(j)  # add node to networkx graph

            row = [int(w) for w in
                   f.readline().strip().split()]  # composition of each row : number of neighbors, id_neigh_1, id_neigh_2 ...
            # increment edge counter with number of neighbors => number of arcs
            n_edges = row[0]
            for k in range(1, n_edges + 1):
                # add edge in graph to all nodes from current one
                g.add_edge(j, row[k])

        g = g.to_directed()  # every arc  # in this example, state of
        # e = [list(pair) for pair in g.edges()]  # [[0, 1], [0, 5], [1, 2], ... list containing lists of edge pair

        edges, agg_matrix = nx_to_format(g, aggregation_type, sparse_matrix)
        e = len(g.edges)
        n = len(g.nodes)
        d_l = 1
        is_multilabel = False
        n_classes = 2
        node_labels = torch.tensor(features, dtype=torch.float)
        # targets = torch.tensor(np.clip(target, 0, 1), dtype=torch.long)  # convert -1 to 0

        # creation of N matrix - [node_features, graph_id (to which the node belongs)] #here there is a unique graph
        # create mask for training
        train_ids, test_ids = read_sse_ids(
            percentage=percentage, dataset=dataset)
        # train_mask = sample_mask(train_ids, n)
        test_ids_temp = range(0, 2000)
        test_ids = [i for i in test_ids_temp if i not in train_ids]
        idx_train = torch.LongTensor(train_ids)
        idx_test = torch.LongTensor(test_ids)
        idx_valid = torch.LongTensor(test_ids)

        return Dataset(
            "two_chainsSSE",
            n,
            e,
            d_l,
            is_multilabel,
            n_classes,
            edges,
            agg_matrix,
            node_labels,
            targets,
            idx_train,
            idx_valid,
            idx_test,
        )


def get_subgraph(set="sub_10_5_200", aggregation_type="sum", sparse_matrix=False):
    from scipy.sparse import coo_matrix
    import scipy.sparse as sp
    import pandas as pd

    types = ["train", "validation", "test"]
    set_name = set
    train = loadmat("./data/subcli/{}.mat".format(set_name))
    train = train["dataSet"]
    dset = {}
    for set_type in types:
        adj = coo_matrix(train['{}Set'.format(set_type)]['connMatrix'].T)
        edges = np.array([adj.row, adj.col]).T

        G = nx.DiGraph()
        G.add_nodes_from(range(0, np.max(edges) + 1))
        G.add_edges_from(edges)

        # G = nx.from_edgelist(edges)
        lab = np.asarray(train['{}Set'.format(set_type)]['nodeLabels']).T
        if len(lab.shape) < 2:
            lab = lab.reshape(lab.shape[0], 1)
        lab = torch.tensor(lab, dtype=torch.float)
        target = np.asarray(train['{}Set'.format(set_type)]['targets']).T
        targets = torch.tensor(np.clip(target, 0, 1),
                               dtype=torch.long)  # convert -1 to 0

        edges, agg_matrix = nx_to_format(G, aggregation_type, sparse_matrix)

        e = len(G.edges)
        n = len(G.nodes)
        d_l = lab.shape[1]
        is_multilabel = False
        n_classes = 2
        node_labels = lab
        dset[set_type] = Dataset(
            "subgraph_{}".format(set_type),
            n,
            e,
            d_l,
            is_multilabel,
            n_classes,
            edges,
            agg_matrix,
            node_labels,
            targets)

    return dset


def get_karate(num_nodes_per_graph=None, aggregation_type="sum", sparse_matrix=True):
    # F = nx.read_edgelist("./data/karate/edges.txt", nodetype=int)
    G = nx.karate_club_graph()

    # edge = np.loadtxt("./data/karate/edges.txt", dtype=np.int32)   # 0-based indexing
    # edge_inv = np.flip(edge, axis=1)
    # edges = np.concatenate((edge, edge_inv))
    # G = nx.DiGraph()
    # G.add_edges_from(edges)
    G = G.to_directed()
    e = len(G.edges)
    n = len(G.nodes)
    # F = nx.Graph()
    # F.add_edges_from(G.edges)

    edges, agg_matrix = nx_to_format(
        G, aggregation_type, sparse_matrix=sparse_matrix)

    is_multilabel = False
    n_classes = 4

    targets = [0] * n
    # class_nodes = [[]] * n_classes # NB keeps broadcasting also at append time
    class_nodes = [[], [], [], []]
    with open("./data/karate/classes.txt") as f:
        for line in f:
            node, node_class = map(int, line.split(" "))
            targets[node] = node_class
            class_nodes[node_class].append(node)

    d_l = n
    # node_labels = torch.zeros(*[n, d_l])
    node_labels = torch.eye(n)
    targets = torch.tensor(targets, dtype=torch.long)

    idx_train = []
    idx_test = []
    for c in class_nodes:
        perm = np.random.permutation(c)
        idx_train += list(perm[:1])  # first index for training
        idx_test += list(perm[1:])  # all other indexes for testing
        # idx_train += list(perm)  # first index for training
        # idx_test += list(perm)  # all other indexes for testing

    idx_valid = torch.LongTensor(idx_train)
    idx_train = torch.LongTensor(idx_train)
    idx_test = torch.LongTensor(idx_test)

    return Dataset(
        "karate",
        n,
        e,
        d_l,
        is_multilabel,
        n_classes,
        edges,
        agg_matrix,
        node_labels,
        targets,
        idx_train,
        idx_valid,
        idx_test,
    )


def collate(samples):
    import dgl
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)


def get_dgl_minigc(aggregation_type="sum", ):
    import dgl
    from dgl.data import MiniGCDataset
    tr_set = MiniGCDataset(80, 10, 20)
    test_set = MiniGCDataset(20, 10, 20)
    data_loader = DataLoader(tr_set, batch_size=80, shuffle=True,
                             collate_fn=collate)
    dataiter = iter(data_loader)
    images, labels = dataiter.next()  # get all the dataset
    G = images.to_networkx()

    e = len(G.edges)
    n = len(G.nodes)

    edges, agg_matrix = nx_to_format(G, aggregation_type)

    print("ciao")


def get_dgl_cora(aggregation_type="sum", sparse_matrix=False):
    import dgl
    from dgl.data import CoraDataset

    tr_set = CoraDataset()
    G = tr_set.graph

    e = len(G.edges)
    n = len(G.nodes)
    d_l = tr_set.features.shape[1]
    is_multilabel = False
    n_classes = tr_set.num_labels
    node_labels = torch.tensor(tr_set.features)
    targets = torch.tensor(tr_set.labels)
    # in this case, there are msk => convert to boolean mask
    idx_train = torch.BoolTensor(tr_set.train_mask)
    idx_valid = torch.BoolTensor(tr_set.val_mask)
    idx_test = torch.BoolTensor(tr_set.test_mask)
    edges, agg_matrix = nx_to_format(G, aggregation_type, sparse_matrix)

    return Dataset(
        "cora",
        n,
        e,
        d_l,
        is_multilabel,
        n_classes,
        edges,
        agg_matrix,
        node_labels,
        targets,
        idx_train,
        idx_valid,
        idx_test,
    )


def get_dgl_citation(aggregation_type="sum", dataset="pubmed"):
    import dgl
    from dgl.data import CitationGraphDataset

    tr_set = CitationGraphDataset(dataset)
    G = tr_set.graph

    e = len(G.edges)
    n = len(G.nodes)
    d_l = tr_set.features.shape[1]
    is_multilabel = False
    n_classes = tr_set.num_labels
    node_labels = torch.tensor(tr_set.features)
    targets = torch.tensor(tr_set.labels)
    idx_train = torch.BoolTensor(tr_set.train_mask)
    idx_valid = torch.BoolTensor(tr_set.val_mask)
    idx_test = torch.BoolTensor(tr_set.test_mask)
    edges, agg_matrix = nx_to_format(G, aggregation_type)

    return Dataset(
        "cora",
        n,
        e,
        d_l,
        is_multilabel,
        n_classes,
        edges,
        agg_matrix,
        node_labels,
        targets,
        idx_train,
        idx_valid,
        idx_test,
    )


def get_dgl_karate(aggregation_type="sum"):
    import dgl
    from dgl.data import KarateClub

    tr_set = KarateClub()
    G = tr_set.graph

    e = len(G.edges)
    n = len(G.nodes)
    d_l = tr_set.features.shape[1]
    is_multilabel = False
    n_classes = tr_set.num_labels
    node_labels = torch.tensor(tr_set.features)
    targets = torch.tensor(tr_set.labels)
    idx_train = torch.BoolTensor(tr_set.train_mask)
    idx_valid = torch.BoolTensor(tr_set.val_mask)
    idx_test = torch.BoolTensor(tr_set.test_mask)
    edges, agg_matrix = nx_to_format(G, aggregation_type)

    return Dataset(
        "cora",
        n,
        e,
        d_l,
        is_multilabel,
        n_classes,
        edges,
        agg_matrix,
        node_labels,
        targets,
        idx_train,
        idx_valid,
        idx_test,
    )

# INPUT
# E: Ma trận cạnh chứa các cặp chỉ số nút, mô tả các cạnh trong đồ thị. Mỗi hàng trong E chứa 3 cột: id_p, id_c, và graph_id
# N: Ma trận chứa các đặc trưng của các nút và một cột bổ sung là graph_id xác định đồ thị mà nút này thuộc về
# targets: Các nhãn mục tiêu cho các nút trong đồ thị
# aggregation_type: Loại phương pháp tổng hợp được sử dụng (ví dụ: "sum" hoặc "degreenorm")
# sparse_matrix: Cờ xác định liệu có sử dụng ma trận thưa (sparse matrix) hay không


def from_EN_to_GNN(E, N, targets, aggregation_type, sparse_matrix=True):
    """
    :param E: # E matrix - matrix of edges : [[id_p, id_c, graph_id],...]
    :param N: # N matrix - [node_features, graph_id (to which the node belongs)]
    :return: # L matrix - list of graph targets [tar_g_1, tar_g_2, ...]
    """

    N_full = N
    E_full = E
    N = N[:, :-1]  # bỏ cột graph_id từ ma trận N
    e = E[:, :2]  # Chỉ lấy hai cột đầu tiên của ma trận cạnh => id_p, id_c

    # creating input for gnn => [id_p, id_c, label_p, label_c]

    # creating arcnode matrix, but transposed
    """
    1 1 0 0 0 0 0 
    0 0 1 1 0 0 0
    0 0 0 0 1 1 1    

    """  # for the indices where to insert the ones, stack the id_p and the column id (single 1 for column)
    # Tạo một đồ thị có hướng sử dụng thư viện networkx
    G = nx.DiGraph()
    # tạo ra một đồ thị với các nút có chỉ số từ 0 đến np.max(e), và thêm các cạnh dựa trên các chỉ số nút trong e
    G.add_nodes_from(range(0, np.max(e) + 1))
    G.add_edges_from(e)
    # Chuyển đồ thị thành dạng phù hợp với GNN
    edges, agg_matrix = nx_to_format(G, aggregation_type, sparse_matrix)

    # Lấy số lượng đồ thị từ các graph_id trong ma trận N
    num_graphs = int(max(N_full[:, -1]) + 1)
    # get all graph_ids
    g_ids = N_full[:, -1]  # lấy cột cuối cùng trong ma trận N_full
    g_ids = g_ids.astype(np.int32)  # chuyển mảng g_ids sang kiểu dữ liệu int32

    # xử lý việc tạo ma trận đồ thị cho các nút trong đồ thị với hai trường hợp: một là khi sử dụng ma trận thưa (sparse matrix) và một là khi không sử dụng ma trận thưa
    # creating graphnode matrix => create identity matrix get row corresponding to id of the graph
    # graphnode = np.take(np.eye(num_graphs), g_ids, axis=0).T
    # substitued with same code as before
    if sparse_matrix:
        # unique: Mảng các giá trị duy nhất của g_ids (tức là các graph_id khác nhau), [0, 1, 2]
        # counts: Mảng chứa số lần xuất hiện của mỗi graph_id,counts = [2, 2, 3] Điều này có nghĩa là graph_id 0 xuất hiện 2 lần
        unique, counts = np.unique(g_ids, return_counts=True)
        # Tạo một mảng values_matrix có kích thước bằng số phần tử trong g_ids và giá trị mặc định là 1.0
        values_matrix = np.ones([len(g_ids)]).astype(
            np.float32)  # [1.0, 1.0, 1.0]
        if aggregation_type == "degreenorm":
            # giá trị tại mỗi chỉ số i trong values_matrix sẽ được chia cho số lần xuất hiện của g_ids[i]
            values_matrix_normalized = values_matrix[g_ids] / counts[g_ids]
        else:
            values_matrix_normalized = values_matrix
        # graphnode = SparseMatrix(indices=np.stack((g_ids, np.arange(len(g_ids))), axis=1),
        #                          values=np.ones([len(g_ids)]).astype(np.float32),
        #                          dense_shape=[num_graphs, len(N)])

        # torch.LongTensor([[0, 1, 0, 2, 1, 2, 2], [0, 1, 2, 3, 4, 5, 6]])
        # list(range(len(g_ids))): Là các chỉ số của các nút (từ 0 đến len(g_ids)-1)
        agg_matrix_i = torch.LongTensor([g_ids, list(range(len(g_ids)))])
        agg_matrix_v = torch.FloatTensor(values_matrix_normalized)
        graphnode = torch.sparse.FloatTensor(
            agg_matrix_i, agg_matrix_v, torch.Size([num_graphs, len(N)]))
    else:
        # Chọn các hàng từ ma trận đơn vị
        # np.eye(num_graphs) tạo ra ma trận đơn vị có kích thước [num_graphs, num_graphs]
        # num_graphs = 3 và g_ids = [0, 1, 0, 2, 1, 2, 2]
        """
        np.eye(3)
        [[1. 0. 0.]
        [0. 1. 0.]
        [0. 0. 1.]]
        np.take(np.eye(3), g_ids, axis=0) =
        [[1. 0. 0.]  # Lấy hàng 0
        [0. 1. 0.]  # Lấy hàng 1
        [1. 0. 0.]  # Lấy hàng 0
        [0. 0. 1.]  # Lấy hàng 2
        [0. 1. 0.]  # Lấy hàng 1
        [0. 0. 1.]  # Lấy hàng 2
        [0. 0. 1.]] # Lấy hàng 2
        Áp dụng .T lên ma trận, các hàng của ma trận sẽ trở thành các cột   và ngược lại
        [[1. 0. 1. 0. 0. 0. 0.]
        [0. 1. 0. 0. 1. 0. 0.]
        [0. 0. 0. 1. 0. 1. 1.]]
        """
        graphnode = torch.FloatTensor(
            np.take(np.eye(num_graphs), g_ids, axis=0).T)  # chọn các hàng của ma trận đơn vị tương ứng với chỉ số đồ thị trong g_ids, .T chuyển vị ma trận

    # print(graphnode.shape)

    # Xác định số lượng cạnh và số lượng nút
    e = E_full.shape[0]
    n = N_full.shape[0]
    # Lấy chiều của ma trận N (số lượng đặc trưng cho mỗi nút)
    d_l = N.shape[1]
    # không phải là phân loại đa nhãn, tức là mỗi nút chỉ thuộc một lớp duy nhất
    is_multilabel = False
    # Tính số lượng lớp phân loại: np.max(targets) tìm giá trị lớn nhất trong mảng targets (mảng này chứa các nhãn mục tiêu cho các nút)
    n_classes = (np.max(targets).astype(np.int) + 1)
    # Chuyển ma trận đặc trưng nút và nhãn mục tiêu thành tensor
    node_labels = torch.FloatTensor(N)
    targets = torch.tensor(targets, dtype=torch.long)

    return Dataset(
        "name",          # Tên của dataset
        n,               # Số lượng nút
        e,               # Số lượng cạnh
        d_l,             # Số chiều của đặc trưng nút
        is_multilabel,   # Biến chỉ việc có phải phân loại đa nhãn hay không
        n_classes,       # Số lớp phân loại
        edges,           # Các cạnh của đồ thị
        agg_matrix,      # Ma trận đại diện các cạnh (agg_matrix)
        node_labels,     # Các nhãn của các nút
        targets,         # Nhãn mục tiêu
        graph_node=graphnode  # Ma trận đại diện đồ thị (graph_node)
    )


def old_load_karate(path="data/karate/"):
    """Load karate club dataset"""
    print('Loading karate club dataset...')
    import random
    import scipy.sparse as sp

    # Tải danh sách cạnh từ file edges.txt. Mỗi cạnh biểu diễn một kết nối giữa hai nút trong đồ thị
    edges = np.loadtxt("{}edges.txt".format(
        path), dtype=np.int32)  # 0-based indexing

    # edge_inv = np.flip(edges, axis=1) # add also archs in opposite direction
    # edges = np.concatenate((edges, edge_inv))
    # reorder list of edges also by second column
    # Sắp xếp danh sách cạnh: sắp xếp danh sách cạnh theo thứ tự tăng dần theo cột thứ hai, sau đó theo cột thứ nhất nếu cần
    # [i,j]: chỉ định số dòng(: là lấy tất cả)/dòng, chỉ định cột/số cột cần lấy
    # Trước sắp xếp: [0 31], [0 21], [0 19], ...
    # Sau sắp xếp: [0 1], [0 2], [0 3], ....
    edges = edges[np.lexsort((edges[:, 1], edges[:, 0]))]
    # Tạo đặc trưng cho các nút, ma trận one-hot kích thước nxn (n số nút)
    # tocsr(): Chuyển đổi ma trận sang dạng ma trận thưa (CSR - Compressed Sparse Row) để tối ưu bộ nhớ
    features = sp.eye(np.max(edges + 1), dtype=np.float).tocsr()
    # Tải nhãn từ file classes.txt: Cột 0: ID của nút, Cột 1: Nhãn tương ứng của nút đó
    # idx_labels[:, 0].argsort(): Sắp xếp các nhãn theo ID nút để đồng bộ với danh sách đặc trưng
    idx_labels = np.loadtxt("{}classes.txt".format(path), dtype=np.float32)
    idx_labels = idx_labels[idx_labels[:, 0].argsort()]
    # Lấy nhãn (cột thứ 2) của các nút
    labels = idx_labels[:, 1]
    # labels = np.eye(max(idx_labels[:, 1])+1, dtype=np.int32)[idx_labels[:, 1]]  # one-hot encoding of labels

    # E: Danh sách cạnh (edges) được mở rộng thêm một cột giá trị 0
    E = np.concatenate(
        (edges, np.zeros((len(edges), 1), dtype=np.int32)), axis=1)
    # Ma trận đặc trưng (features) được chuyển thành mảng NumPy và thêm một cột giá trị 0
    N = np.concatenate((features.toarray(), np.zeros(
        (features.shape[0], 1), dtype=np.int32)), axis=1)

    # Tạo một mảng có kích thước bằng số nút trong đồ thị (34 nút), khởi tạo toàn bộ giá trị là 0
    mask_train = np.zeros(shape=(34,), dtype=np.float32)

    # np.argwhere(labels == 0): Lấy chỉ mục của các nút có nhãn là 0
    # random.choices(..., k=4): Chọn ngẫu nhiên 4 chỉ mục từ danh sách
    id_0, id_4, id_5, id_12 = random.choices(np.argwhere(labels == 0), k=4)
    id_1, id_6, id_7, id_13 = random.choices(np.argwhere(labels == 1), k=4)
    id_2, id_8, id_9, id_14 = random.choices(np.argwhere(labels == 2), k=4)
    id_3, id_10, id_11, id_15 = random.choices(np.argwhere(labels == 3), k=4)

    # Đặt giá trị 1 cho các nút đã chọn trong mảng mask_train, đánh dấu đây là các nút thuộc tập huấn luyện
    mask_train[id_0] = 1.  # class 1
    mask_train[id_1] = 1.  # class 2
    mask_train[id_2] = 1.  # class 0
    mask_train[id_3] = 1.  # class 3

    # Tạo mặt nạ kiểm tra (mask_test) bằng cách lấy phần bù của mask_train
    mask_test = 1. - mask_train

    return E, N, labels, torch.BoolTensor(mask_train), torch.BoolTensor(mask_test)
