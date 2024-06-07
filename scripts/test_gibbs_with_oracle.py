import sys
sys.path.append("..")
import os
import copy
import time
import numpy as np
import matplotlib.pyplot as plt
from progress.bar import Bar

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.utils.convert import from_networkx
from torch_geometric.nn import GCNConv
from torch_geometric.nn.norm import LayerNorm
from torch_geometric.nn import global_mean_pool


from cosampy.co.tsp import *

# torch device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

epoch_max = 100
iter_max = 1000
num_nodes = 20

output_dir = f"test_gibbs_with_oracle_tsp{num_nodes}_4layers"
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
class OracleHamming:
    def __init__(self, x0, pb, solmap):
        self.x0 = x0
        self.pb = pb
        self.map = solmap
        self.neighbors, self.weights = self.initialize_neighbors()
    def __len__(self):
        return len(self.neighbors)
    def initialize_neighbors(self):
        H = HammingBall(self.x0, r=1)
        neighbors, weights = [], []
        solution_counts = {}
        c = self.pb.cost_vector()
        E0 = np.dot(c, self.x0[:,0])
        for y in H:
            #if np.all(y[:,1]==self.x0[:,1]) and np.all(y[y[:,1]==1,0] == self.x0[y[:,1]==1,0]):
            #    continue
            cy = c.copy()
            cy[ (y[:,0]==0) & (y[:,1]==1) ] = np.sum(c)
            cy[ (y[:,0]==1) & (y[:,1]==1) ] = 0.0
            tmp_sol = self.map.get(cy)
            if tmp_sol is None:
                # solve with oracle
                yp = nearest_neighbor_route_2(cy, [], self.pb)
                if tuple(yp.sequence) in solution_counts:
                    solution_counts[tuple(yp.sequence)] += 1
                else:
                    solution_counts[tuple(yp.sequence)] = 1
                ypf = edges_to_flow(yp, self.pb)
                y[:,0] = ypf
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

checkpoint_epochs = [1, 10, 20, 40, 80, 160]
checkpoint_cost_means = []

# generate the test instances
test_instances = [random_euclidean_tsp(num_nodes) for _ in range(epoch_max)]
for chk_epoch in checkpoint_epochs:
    # checkpoint_path
    checkpoint_path = f"gibbs_with_oracle_tsp20_4layers_transfer/{chk_epoch}_model.pt"

    gcn = GCN(in_dim=4, out_dim=2, hidden_dim=64).to(device)
    gcn.load_state_dict(torch.load(checkpoint_path))
    
    all_losses = []
    all_epoch_costs = []
    
    bar = Bar(f"{chk_epoch} Solving TSP{num_nodes}: ", max=epoch_max)
    for epoch in range(epoch_max):
        # random TSP
        #tsp = random_euclidean_tsp(num_nodes)
        tsp = test_instances[epoch]
        N = tsp.number_of_edges()
        # solution map
        sol_map = SolutionMap()
        # initial solution
        sol = nearest_neighbor_route(tsp)
        c = tsp.cost_vector()
        x = edges_to_flow(sol, tsp)
        x = np.expand_dims(x, axis=1)
        f = np.zeros_like(x)
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
        initial_cost = solution_cost(sol, tsp)
    
    
        # start sampling
        cost_values = []
        min_cost_values = []
        for it in range(iter_max):
            # prediction
            ydata = copy.deepcopy(data)
            pred = gcn(data.to(device))
            nidx = list(np.ndindex(pred.shape))
            # sample candidate
            w = F.softmax(torch.flatten(pred.detach())).cpu().numpy()
            #xidx = np.random.choice(range(len(nidx)), p=w)
            xidx = np.argmax(w)
            sample_idx = nidx[xidx]
            ydata.x[sample_idx[0],sample_idx[1]+2] = not ydata.x[sample_idx[0],sample_idx[1]+2]
    
            cy = c.copy()
            y = ydata.x[ydata.x[:,0]==0,2:][:N,:].cpu().numpy()
    
            cy[ (y[:,0]==0) & (y[:,1]==1) ] = np.sum(c)
            cy[ (y[:,0]==1) & (y[:,1]==1) ] = 0.0
            #tmp_sol = self.map.get(cy)
            #if tmp_sol is None:
            # solve with oracle
            yp = nearest_neighbor_route_2(cy, [], tsp)
            ypf = edges_to_flow(yp, tsp)
            ydata.x[:ypf.shape[0],2] = torch.tensor(ypf)
            ydata.x[:ypf.shape[0],3] = torch.tensor(y[:,1])
    
            # accept rate
            Zx = pred.exp().sum()
            ypred = gcn(ydata.to(device))
            Zy = ypred.exp().sum()
    
            xx = data.x[data.x[:,0]==0, 2][:N].cpu().numpy()
            yy = ydata.x[ydata.x[:,0]==0, 2][:N].cpu().numpy()
            Ex = np.dot(c, xx)
            Ey = np.dot(c, yy)
    
    
            ar = min(1.0, np.exp(Ex - Ey) * Zx/Zy)
            if np.random.rand() < ar:
                # accept
                data = ydata
    
    
            cost_values.append(np.dot(c, yy))
    
            if cost_values[-1] < best_cost:
                best_sol = yp
                best_cost = cost_values[-1]
            min_cost_values.append(best_cost)
    
        all_epoch_costs.append((initial_cost - np.array(min_cost_values)) / initial_cost)
    
    
        fig,ax = plt.subplots(1,1)
        ax.hist(cost_values, bins=20)
        ax.set_xlabel("Cost")
        ax.set_ylabel("#")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, str(chk_epoch)+"_"+str(epoch)+"_costs.png"))
        plt.close(fig)
    
        fig,ax = plt.subplots(1,1)
        ax.plot(min_cost_values)
        ax.set_xlabel("It")
        ax.set_ylabel("Cost")
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, str(chk_epoch)+"_"+str(epoch)+"_min_costs.png"))
        plt.close(fig)
    
        fig,ax = plt.subplots(1,2)
        plot_solution(sol, tsp, ax=ax[0], color="green", alpha=.4)
        plot_solution(best_sol, tsp, ax=ax[1], color="red", alpha=.4)
        c0,c1 = solution_cost(sol,tsp), solution_cost(best_sol, tsp)
        ax[0].set_title(f"Cost = {c0}")
        ax[1].set_title(f"Cost = {c1}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, str(chk_epoch)+"_"+str(epoch)+"_sol.png"))
        plt.close(fig)
    
        bar.next()
    bar.finish()

    all_epoch_costs = np.array(all_epoch_costs)
    checkpoint_cost_means.append(np.mean(all_epoch_costs,axis=0))
    
    fig,ax = plt.subplots(1,1)
    ax.plot(100 * np.mean(all_epoch_costs, axis=0))
    ax.set_xlabel("It")
    ax.set_ylabel("Delta Cost (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "all_epoch_costs.png"))
    plt.close(fig)

fig,ax = plt.subplots(1,1)
for i,vals in enumerate(checkpoint_cost_means):
    ax.plot(100 * vals, label=str(checkpoint_epochs[i]))
ax.set_xlabel("It")
ax.set_ylabel("Delta Cost (%)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "checkpoint_cost_mean.png"))
plt.close(fig)
