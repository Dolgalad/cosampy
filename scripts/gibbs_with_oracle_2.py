import sys
sys.path.append("..")
import os
import copy
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.utils.convert import from_networkx
from torch_geometric.nn import GCNConv
from torch_geometric.nn.norm import LayerNorm
from torch_geometric.nn import global_mean_pool

import cosampy
import cosampy.co
from cosampy.co.tsp import *

# torch device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

epoch_max = 1000
iter_max = 500
num_nodes = 10
exp_len_max = 2000
lr = 0.0001
batch_size = 512

output_dir = f"gibbs_with_oracle_2_tsp{num_nodes}_4layers"
if os.path.isdir(output_dir):
    os.system(f"rm {output_dir}/*")
os.makedirs(output_dir, exist_ok=True)

class HammingBall:
    def __init__(self, x0, r=1):
        self.x0 = x0
        self.r = r
    def __iter__(self):
        yield self.x0
        for i in np.ndindex(*self.x0.shape):
            y = self.x0.copy()
            y[i] = not y[i]
            yield y
class RollBall:
    def __init__(self, x0):
        self.x0=x0
    def __iter__(self):
        yield self.x0
        for i in range(self.x0.shape[0]):
            for j in range(self.x0.shape[1]-1):
                y = self.x0.copy()
                y[i,:] = np.roll(y[i,:], j+1)
                yield y
class OracleHamming:
    def __init__(self, x0, pb, solmap):
        self.x0 = x0
        self.pb = pb
        self.map = solmap
        self.neighbors, self.weights = self.initialize_neighbors()
    def __len__(self):
        return len(self.neighbors)
    def initialize_neighbors(self):
        H = RollBall(self.x0[:,1:])
        neighbors, weights = [], []
        solution_counts = {}
        c = self.pb.cost_vector()
        sol0 = nearest_neighbor_route_2(c, [], self.pb)
        E0 = solution_cost(sol0, self.pb)
        for y in H:
            cy = c.copy()
            cy[y[:,1]==1] = 0.0
            cy[y[:,2]==1] = np.sum(c)
            tmp_sol = self.map.get(cy)
            if tmp_sol is None:
                # solve with oracle
                yp = nearest_neighbor_route_2(cy, [], self.pb)
                if tuple(yp.sequence) in solution_counts:
                    solution_counts[tuple(yp.sequence)] += 1
                else:
                    solution_counts[tuple(yp.sequence)] = 1
                ypf = edges_to_flow(yp, self.pb)
                y = np.concatenate((np.expand_dims(ypf,1), y), axis=1)
                E1 = np.dot(c,ypf)
                self.map.set(cy, (y, yp, E1))
            else:
                y, yp, E1 = tmp_sol
                if tuple(yp.sequence) in solution_counts:
                    solution_counts[tuple(yp.sequence)] += 1
                else:
                    solution_counts[tuple(yp.sequence)] = 1

            neighbors.append(y)
            weights.append(np.exp(0.5 * (E0 - E1)))
        for i,n in enumerate(neighbors):
            el = flow_to_edges(n[:,0],self.pb)
            r = edge_list_to_route(el)
            sol = ETSPSolution(r)
            weights[i] /= solution_counts[tuple(sol.sequence)]
        return neighbors, np.array(weights)
    def Z(self):
        return np.sum(self.weights)
    def sample(self):
        p = self.weights / self.Z()
        idx = np.random.choice(range(len(self)), p=p)
        return self.neighbors[idx]
    def __iter__(self):
        for n in self.neighbors:
            yield n

"""Map cost vectors to solutions
"""
class SolutionMap:
    def __init__(self):
        self.map= {}
    def set(self, c, s):
        self.map[c.tobytes()] = s
    def get(self, c):
        if c.tobytes() in self.map:
            return self.map[c.tobytes()]
        return None


class GCN(torch.nn.Module):
    def __init__(self, in_dim=3, hidden_dim=16, out_dim=1):
        super().__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.ln1 = LayerNorm(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.ln2 = LayerNorm(hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.ln3 = LayerNorm(hidden_dim)
        self.conv4 = GCNConv(hidden_dim, out_dim)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.dropout(x, 0.1)
        x = self.ln1(x)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.dropout(x, 0.1)
        x = self.ln2(x)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.dropout(x, 0.1)
        x = self.ln3(x)
        x = F.relu(x)
        x = self.conv4(x, edge_index)
        x = F.dropout(x, 0.1)
        return x[data.decision_variable_mask]

gcn = GCN(in_dim=6, out_dim=2, hidden_dim=64).to(device)

optimizer = torch.optim.Adam(gcn.parameters(), lr=lr)

all_losses = []

# experience replay
exp_x, exp_y = [], []

for epoch in range(epoch_max):
    # random TSP
    tsp = random_euclidean_tsp(num_nodes)
    N = tsp.number_of_edges()
    # solution map
    sol_map = SolutionMap()
    # initial solution
    sol = nearest_neighbor_route(tsp)
    c = tsp.cost_vector()
    x = edges_to_flow(sol, tsp)
    x = np.expand_dims(x, axis=1)
    f = np.zeros((x.shape[0], 3))
    f[:,0] = 1
    xx = np.concatenate((x,f), axis=1)
    best_cost = solution_cost(sol, tsp)
    best_sol = copy.deepcopy(sol)
    # create GCN input data structure
    xt = np.pad(xx, ((0, tsp.number_of_nodes()), (0,0)))
    fg = tsp.factor_graph(xt, formulation="mtz")
    data = from_networkx(fg)
    data.x = data.x.to(torch.float)
    mask = np.zeros(data.x.shape[0])
    mask[:N] = 1
    data.decision_variable_mask = torch.tensor(mask, dtype=torch.bool)


    # start sampling
    cost_values = []
    loss_values = []
    for it in range(iter_max):
        t0 = time.time()
        xneighbors = OracleHamming(xx, tsp, sol_map)
        t0 = time.time() - t0
        # sample candidate
        t2 = time.time()
        y = xneighbors.sample()
        t2 = time.time() - t2
        t1 = time.time()
        yneighbors = OracleHamming(y, tsp, sol_map)
        t1 = time.time() - t1
        # accept or reject
        Zx = xneighbors.Z()
        Zy = yneighbors.Z()
        accept_rate = min(1.0, Zx/Zy)
        if np.random.rand() < accept_rate:
            xx = y
            if np.dot(c, xx[:,0]) < best_cost:
                best_cost = np.dot(c, xx[:,0])
                el = flow_to_edges(xx[:,0],tsp)
                r = edge_list_to_route(el)
                best_sol = ETSPSolution(r)
               

        cost = np.dot(c, xx[:,0])
        s = np.sum(xx[:,1])
        print(f"Epoch = {epoch}, it = {it}, cost = {cost}, ar = {accept_rate}, s = {s}")
        print(f"\tt0 = {t0}, t1 = {t1}, t2 = {t2}")

        cost_values.append(cost)

        # training
        t3 = time.time()
        xdata, ydata = data.clone(), data.clone()
        xdata.x[:xx.shape[0], 2:] = torch.tensor(xx, dtype=torch.float32)
        xdata.Z = Zx
        ydata.x[:y.shape[0], 2:] = torch.tensor(y, dtype=torch.float32)
        ydata.Z = Zy
        xlabels, ylabels = torch.tensor(xneighbors.weights[1:], dtype=torch.float32), torch.tensor(yneighbors.weights[1:], dtype=torch.float32)
        xdata.y = xlabels.clone().unsqueeze_(0)
        ydata.y = ylabels.clone().unsqueeze_(0)
        exp_x.append(xdata)
        exp_x.append(ydata)

        if len(exp_x) > exp_len_max:
            del exp_x[1]
            del exp_x[0]

        train_loader = DataLoader(exp_x, batch_size=batch_size, shuffle=True)
        epoch_loss = []
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            a = torch.flatten(gcn(batch)).reshape(batch.y.shape)

            loss = F.kl_div(F.log_softmax(a), F.log_softmax(batch.y), log_target=True, reduction="batchmean")
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
        t3 = time.time() - t3
        print(f"\tt3 = {t3}")
        loss_values.append(np.mean(epoch_loss, axis=0))
        all_losses.append(np.mean(epoch_loss, axis=0))
        print(f"\tloss = {loss_values[-1]}")


    fig,ax = plt.subplots(1,1)
    ax.hist(cost_values, bins=20)
    ax.set_xlabel("Cost")
    ax.set_ylabel("#")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, str(epoch)+"_costs.png"))
    plt.close(fig)

    loss_values_ = np.array(loss_values)
    all_losses_ = np.array(all_losses)

    fig,ax = plt.subplots(1,1, sharex=True)
    ax.plot(loss_values_)
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, str(epoch)+"_loss.png"))
    plt.close(fig)

    fig,ax = plt.subplots(1,1, sharex=True)
    ax.plot(all_losses_)
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "all_loss.png"))
    plt.close(fig)


    fig,ax = plt.subplots(1,2)
    plot_solution(sol, tsp, ax=ax[0], color="green", alpha=.4)
    plot_solution(best_sol, tsp, ax=ax[1], color="red", alpha=.4)
    c0,c1 = solution_cost(sol,tsp), solution_cost(best_sol, tsp)
    ax[0].set_title(f"Cost = {c0}")
    ax[1].set_title(f"Cost = {c1}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, str(epoch)+"_sol.png"))
    plt.close(fig)

    # save model
    model_path = os.path.join(output_dir, str(epoch)+"_model.pt")
    torch.save(gcn.state_dict(), model_path)





