import numpy as np
from scipy.special import expit as sigmoid
import igraph as ig
import random
import argparse
import os
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def is_dag(W):
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    return G.is_dag()


def simulate_dag(d, s0, graph_type):
    """Simulate random DAG with some expected number of edges.

    Args:
        d (int): num of nodes
        s0 (int): expected num of edges
        graph_type (str): ER, SF, BP

    Returns:
        B (np.ndarray): [d, d] binary adj matrix of DAG
    """
    def _random_permutation(M):
        # np.random.permutation permutes first axis only
        P = np.random.permutation(np.eye(M.shape[0]))
        return P.T @ M @ P

    def _random_acyclic_orientation(B_und):
        return np.tril(_random_permutation(B_und), k=-1)

    def _graph_to_adjmat(G):
        return np.array(G.get_adjacency().data)

    if graph_type == 'ER':
        # Erdos-Renyi
        # G_und = ig.Graph.Erdos_Renyi(n=d, m=s0)
        G_und = ig.Graph.Erdos_Renyi(n=d, p=0.5)
        B_und = _graph_to_adjmat(G_und)
        B = _random_acyclic_orientation(B_und)
    elif graph_type == 'SF':
        # Scale-free, Barabasi-Albert
        G = ig.Graph.Barabasi(n=d, m=int(round(s0 / d)), directed=True)
        B = _graph_to_adjmat(G)
    elif graph_type == 'BP':
        # Bipartite, Sec 4.1 of (Gu, Fu, Zhou, 2018)
        top = int(0.2 * d)
        G = ig.Graph.Random_Bipartite(top, d - top, m=s0, directed=True, neimode=ig.OUT)
        B = _graph_to_adjmat(G)
    else:
        raise ValueError('unknown graph type')
    B_perm = _random_permutation(B)
    assert ig.Graph.Adjacency(B_perm.tolist()).is_dag()
    return B_perm


def simulate_parameter(B, w_ranges=((-2.0, -0.5), (0.5, 2.0))):
    """Simulate SEM parameters for a DAG.

    Args:
        B (np.ndarray): [d, d] binary adj matrix of DAG
        w_ranges (tuple): disjoint weight ranges

    Returns:
        W (np.ndarray): [d, d] weighted adj matrix of DAG
    """
    W = np.zeros(B.shape)
    S = np.random.randint(len(w_ranges), size=B.shape)  # which range
    for i, (low, high) in enumerate(w_ranges):
        U = np.random.uniform(low=low, high=high, size=B.shape)
        W += B * (S == i) * U
    return W


def simulate_linear_sem(W, n, sem_type, noise_scale=None):
    """Simulate samples from linear SEM with specified type of noise.

    For uniform, noise z ~ uniform(-a, a), where a = noise_scale.

    Args:
        W (np.ndarray): [d, d] weighted adj matrix of DAG
        n (int): num of samples, n=inf mimics population risk
        sem_type (str): gauss, exp, gumbel, uniform, logistic, poisson
        noise_scale (np.ndarray): scale parameter of additive noise, default all ones

    Returns:
        X (np.ndarray): [n, d] sample matrix, [d, d] if n=inf
    """
    def _simulate_single_equation(X, w, scale):
        """X: [n, num of parents], w: [num of parents], x: [n]"""
        if sem_type == 'gauss':
            z = np.random.normal(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'exp':
            z = np.random.exponential(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'gumbel':
            z = np.random.gumbel(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'uniform':
            z = np.random.uniform(low=-scale, high=scale, size=n)
            x = X @ w + z
        elif sem_type == 'logistic':
            x = np.random.binomial(1, sigmoid(X @ w)) * 1.0
        elif sem_type == 'poisson':
            lam = np.exp(np.clip(X @ w, a_min=None, a_max=10))
            x = np.random.poisson(lam) * 1.0
        else:
            raise ValueError('unknown sem type')
        return x

    d = W.shape[0]
    if noise_scale is None:
        scale_vec = np.ones(d)
    elif np.isscalar(noise_scale):
        scale_vec = noise_scale * np.ones(d)
    else:
        if len(noise_scale) != d:
            raise ValueError('noise scale must be a scalar or has length d')
        scale_vec = noise_scale
    if not is_dag(W):
        raise ValueError('W must be a DAG')
    if np.isinf(n):  # population risk for linear gauss SEM
        if sem_type == 'gauss':
            # make 1/d X'X = true cov
            X = np.sqrt(d) * np.diag(scale_vec) @ np.linalg.inv(np.eye(d) - W)
            return X
        else:
            raise ValueError('population risk not available')
    # empirical risk
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == d
    X = np.zeros([n, d])
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        X[:, j] = _simulate_single_equation(X[:, parents], W[parents, j], scale_vec[j])
    return X


def simulate_nonlinear_sem(B, n, sem_type='mlp', noise_scale=None):
    """Simulate samples from nonlinear SEM.

    Args:
        B (np.ndarray): [d, d] binary adj matrix of DAG
        n (int): num of samples
        sem_type (str): mlp, mim, gp, gp-add
        noise_scale (np.ndarray): scale parameter of additive noise, default all ones

    Returns:
        X (np.ndarray): [n, d] sample matrix
    """
    def _simulate_single_equation(X, scale):
        """X: [n, num of parents], x: [n]"""
        z = np.random.normal(scale=scale, size=n)
        pa_size = X.shape[1]
        if pa_size == 0:
            return z
        if sem_type == 'mlp':
            hidden = 100
            W1 = np.random.uniform(low=0.5, high=2.0, size=[pa_size, hidden])
            W1[np.random.rand(*W1.shape) < 0.5] *= -1
            W2 = np.random.uniform(low=0.5, high=2.0, size=hidden)
            W2[np.random.rand(hidden) < 0.5] *= -1
            x = sigmoid(X @ W1) @ W2 + z
        elif sem_type == 'mim':
            w1 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
            w1[np.random.rand(pa_size) < 0.5] *= -1
            w2 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
            w2[np.random.rand(pa_size) < 0.5] *= -1
            w3 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
            w3[np.random.rand(pa_size) < 0.5] *= -1
            x = np.tanh(X @ w1) + np.cos(X @ w2) + np.sin(X @ w3) + z
        elif sem_type == 'gp':
            from sklearn.gaussian_process import GaussianProcessRegressor
            gp = GaussianProcessRegressor()
            x = gp.sample_y(X, random_state=None).flatten() + z
        elif sem_type == 'gp-add':
            from sklearn.gaussian_process import GaussianProcessRegressor
            gp = GaussianProcessRegressor()
            x = sum([gp.sample_y(X[:, i, None], random_state=None).flatten()
                     for i in range(X.shape[1])]) + z
        else:
            raise ValueError('unknown sem type')
        return x

    d = B.shape[0]
    scale_vec = noise_scale if noise_scale else np.ones(d)
    X = np.zeros([n, d])
    G = ig.Graph.Adjacency(B.tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == d
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        X[:, j] = _simulate_single_equation(X[:, parents], scale_vec[j])
    return X


def generate_data(n, d, s0, graph_type, linear_sem_type, nonlinear_sem_type, type, save_dir):
    # make data folder
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    existing_folders = sorted([f for f in os.listdir(save_dir) if os.path.isdir(os.path.join(save_dir, f))])
    if existing_folders:
        new_folder_num = int(existing_folders[-1]) + 1
    else:
        new_folder_num = 1
    new_folder_name = f'{new_folder_num:06}'
    new_folder_path = os.path.join(save_dir, new_folder_name)    
    os.makedirs(new_folder_path)

    # linear sem
    if type == 'linear':
        B_true = simulate_dag(d, s0, graph_type)
        W_true = simulate_parameter(B_true)
        noise_scale = [random.uniform(0.1, 1.0) for _ in range(B_true.shape[0])]
        X = simulate_linear_sem(W_true, n, linear_sem_type, noise_scale)
        np.savetxt(os.path.join(new_folder_path, f'B_true_{graph_type}_{linear_sem_type}.csv'), B_true, delimiter=',')
        # 필요한지 확인..?
        np.savetxt(os.path.join(new_folder_path, f'W_true_{graph_type}_{linear_sem_type}.csv'), W_true, delimiter=',')
        np.savetxt(os.path.join(new_folder_path, f'X_{graph_type}_{linear_sem_type}.csv'), X, delimiter=',')
      
    # nonlinear sem
    else:
        B_true = simulate_dag(d, s0, graph_type)
        noise_scale = [random.uniform(0.1, 1.0) for _ in range(B_true.shape[0])]
        X = simulate_nonlinear_sem(B_true, n, nonlinear_sem_type, noise_scale)
        np.savetxt(os.path.join(new_folder_path, f'B_true_{graph_type}_{nonlinear_sem_type}.csv'), B_true, delimiter=',')
        np.savetxt(os.path.join(new_folder_path, f'X_{graph_type}_{nonlinear_sem_type}.csv'), X, delimiter=',')
    
    
def visualize_dag(csv_path):
    adj_matrix = np.loadtxt(csv_path, delimiter=',')
    G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)  
    pos = nx.spring_layout(G, k=1.5)  
    plt.figure(figsize=(5, 5))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=1000, arrows=True)
    plt.title("DAG Visualization")
    plt.show()


def visualize_matrix(csv_path):
    adj_matrix = np.loadtxt(csv_path, delimiter=',')
    plt.figure(figsize=(5, 5))
    sns.heatmap(adj_matrix, annot=True, cmap='Blues', cbar=False)
    plt.gca().xaxis.set_label_position('top') 
    plt.gca().xaxis.tick_top()  
    plt.title('Adjacency Matrix Visualization')
    plt.show()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--save_dir', type=str, default = '/home/jina/reprod/data/train')
    parser.add_argument('--num_graph', type=int, default = 200)
    parser.add_argument('--graph_type', type=str, default = 'ER')

    args = parser.parse_args()

    num_graph = args.num_graph

    n_list = [500, 1000, 2000]
    d_list = [10, 20, 50, 100]
    # s0_list = [1, 2, 4]
    graph_list = ['ER', 'SF']
    type_list = ['linear', 'nonlinear']
    linear_type = ['gauss', 'exp', 'gumbel', 'uniform', 'logistic', 'poisson']
    nonlinear_sem_type = 'mlp'
    linear_sem_type = 'exp'

    for _ in range(num_graph):
        seed = random.randint(0, 10000)
        set_random_seed(seed)

        n = random.choice(n_list)
        d = random.choice(d_list)
        max_edge = d * (d-1) // 2
        s0 = int(max_edge * random.uniform(0.3, 0.8))
        # graph_type = random.choice(graph_list)
        sem_type = random.choice(type_list)
        if sem_type == 'linear':
            linear_sem_type = random.choice(linear_type)

        generate_data(n=n, d=d, s0=s0, graph_type=args.graph_type, linear_sem_type=linear_sem_type, nonlinear_sem_type=nonlinear_sem_type, type=sem_type, save_dir=args.save_dir)

