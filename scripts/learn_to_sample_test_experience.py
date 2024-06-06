import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.data import Data
from torch_geometric.utils.convert import from_networkx
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt

from sampco.co.tsp import *

import time
import copy

# max iterations per iteration
max_iter = 100
num_samples = 100
noise_std = 0.01
epochs = 1000
nV = 6
replay_len = 10

# create a random ETSP instance
tsp = random_euclidean_tsp(nV)

# initial solution given by oracle
sol = nearest_neighbor_route(tsp)
solx = edges_to_flow(sol, tsp)
solx = np.concatenate((solx, np.zeros(tsp.number_of_nodes())))
fg = tsp.factor_graph(solx, formulation="mtz")
data = from_networkx(fg)
data.x = data.x.to(torch.float)

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(3, 16)
        self.conv2 = GCNConv(16, 1)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, 0.5)
        x = self.conv2(x, edge_index)
        return F.elu(x)

class AcceptancePredictor(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(AcceptancePredictor, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(3, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, 1)

    def forward(self, xdata, ydata):
        # 1. Obtain node embeddings
        x = self.conv1(xdata.x - ydata.x, xdata.edge_index)
        x = x.relu()
        x = self.conv2(x, xdata.edge_index)
        # 2. Readout layer
        x = global_mean_pool(x, torch.zeros(x.size()[0], dtype=torch.long))  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5)
        x = self.lin(x)

        return x


def normalize_costs(d):
    gcosts = d.x[d.x[:,0]==0, 1]
    d.x[d.x[:,0]==0, 1] = (gcosts - torch.mean(gcosts)) / torch.std(gcosts)
    return d

def sample_neighbors(c, gdata, tsp):
    neighbors, noise = [], []
    noise_std_idx = np.ones(tsp.ilp_number_of_variables(formulation="mtz"))
    noise_std_idx[-tsp.number_of_nodes():] = 0
    for _ in range(num_samples):
        n_ = np.random.normal(0,noise_std*noise_std_idx)
        solp = nearest_neighbor_route_2(c + n_,
                                        gdata.edge_index, tsp)
        neighbors.append(solp)
        noise.append(n_)
    return neighbors, noise

def get_unique_neighbors(neighbors, noise):
    unique_neighbors = []
    unique_noise= []
    unique_counts = []
    for ii, n in enumerate(neighbors):
        if not n in unique_neighbors:
            unique_neighbors.append(n)
            unique_noise.append(noise[ii])
            unique_counts.append(1)
        else:
            idx = unique_neighbors.index(n)
            unique_noise[idx] += noise[ii]
            unique_counts[idx] += 1
    for ii in range(len(unique_neighbors)):
        unique_noise[ii] /= unique_counts[ii]
    return unique_neighbors, unique_noise, unique_counts



def compute_loss(exp):
    losses = []
    for xdata,ydata,tsp in exp:
        # predict new costs
        new_costs = torch.flatten(gcn(xdata)[xdata.x[:,0]==0])
        # sample from neighborhood of x
        xneighbors, xnoise = sample_neighbors(new_costs.clone().detach(), xdata, tsp)
        unique_neighbors, unique_noise, unique_counts = get_unique_neighbors(xneighbors, xnoise)
        Zopt = np.sum([np.exp(0.5 * (E0 - solution_cost(n,tsp))) for n in unique_neighbors])
        # evaluate expected and resulting probabilities and compute loss
        loss = 0.0
        dy = np.zeros_like(unique_noise[0])
        for ii,n in enumerate(unique_neighbors):
            # evaluate optimal probability
            qopt = np.exp(0.5 * (E0 - solution_cost(n,tsp))) / Zopt
            qapp = np.sum([n == ni for ni in xneighbors]) / len(xneighbors)
            loss += qopt * (np.log(qopt) - np.log(qapp))
            dy -= (1.0/Zopt) * (1.0/noise_std) * unique_noise[ii] / qapp
        losses.append(loss)
        # call backward
        new_costs.backward(torch.tensor(dy).requires_grad_(True))
    return np.mean(losses)

# Experience replay
experience_replay = []
all_loss = []
all_accept_loss = []
mean_costs, std_costs, skew_costs, kurtosis_costs = [], [], [], []

gcn = GCN()
acceptance_predictor = AcceptancePredictor(hidden_channels=64)

optimizer = torch.optim.Adam(gcn.parameters(), lr=0.01)
acceptance_optimizer = torch.optim.Adam(acceptance_predictor.parameters(), lr=0.001)

for kk in range(epochs):

    # create a random ETSP instance
    tsp = random_euclidean_tsp(nV+int(kk/50))
    #tsp = random_euclidean_tsp(np.random.choice(range(5,10)))

    
    # initial solution given by oracle
    sol = nearest_neighbor_route(tsp)
    solx = edges_to_flow(sol, tsp, formulation="mtz")
    #solx = np.concatenate((solx, np.zeros(tsp.number_of_nodes())))

    best_sol = copy.deepcopy(sol)
    best_cost = solution_cost(sol, tsp)
    fig, ax = plt.subplots(1,1)
    plot_solution(sol, tsp, ax=ax)
    plt.tight_layout()
    plt.savefig(f"test_{kk}_sol.png")
    plt.close(fig)

    
    data = from_networkx(tsp.factor_graph(solx, formulation="mtz"))
    data.x = data.x.to(torch.float)
    data = normalize_costs(data)

    cost_values = []
    loss_values = []
    accept_loss_values = []
    noise_std_idx = np.ones(tsp.ilp_number_of_variables(formulation="mtz"))
    noise_std_idx[-tsp.number_of_nodes():] = 0
    for it in range(max_iter):
        print("Iteration ", it)
        E0 = solution_cost(sol, tsp)
    
        # perform forward pass of GCN
        new_costs = torch.flatten(gcn(data)[data.x[:,0]==0])
    
        # generate samples of the neighborhood of current solution
        neighbors = []
        noise = []
        for _ in range(num_samples):
            n_ = np.random.normal(0,noise_std*noise_std_idx)
            solp = nearest_neighbor_route_2(new_costs.clone().detach() + n_,
                                            data.edge_index, tsp)
            if solp != sol or True:
                neighbors.append(solp)
                noise.append(n_)
    
        # sample neighbor
        weights = np.array([np.exp(0.5 * (E0 - solution_cost(n,tsp))) for n in neighbors])
        Zx0 = np.sum(weights)
        weights /= Zx0
    
        if len(neighbors)==0:
            print("No neighbors")
            break
    
        xidx = np.random.choice(range(len(neighbors)), p=weights)
        x = neighbors[xidx]
        c2 = new_costs.clone().detach() + noise[xidx]
    
        x_neighbors = []
        xx = edges_to_flow(x, tsp, formulation="mtz")
    
        ydata = data.clone()
        ydata.x[ydata.x[:,0]==0, 2] = torch.tensor(xx).to(torch.float)
        ydata.x[ydata.x[:,0]==0, 1] = torch.tensor(c2).to(torch.float)
        ydata.x = ydata.x.to(torch.float)
        ydata = normalize_costs(ydata)
        tmp_costs = torch.flatten(gcn(ydata))
        new_costs2 = torch.flatten(tmp_costs[ydata.x[:,0]==0].clone().detach())
        for _ in range(num_samples):
            solp = nearest_neighbor_route_2(new_costs2 + np.random.normal(0,noise_std*noise_std_idx),
                                            data.edge_index, tsp)
            x_neighbors.append(solp)
        Zx = np.sum([np.exp(0.5 * (solution_cost(x,tsp) - solution_cost(n,tsp))) for n in x_neighbors])

        # evaluate the loss function
        # approximation of the optimal proposal distribution
        unique_neighbors, unique_noise, unique_count = get_unique_neighbors(neighbors, noise)

        if len(unique_neighbors)<=0:
            print("Unique neighbors : ", len(unique_neighbors))
            break
        Zopt = np.sum([np.exp(0.5 * (E0 - solution_cost(n,tsp))) for n in unique_neighbors])

        # loss function and update weights
        experience_replay.append((data.clone(), ydata.clone(), copy.deepcopy(tsp)))
        if len(experience_replay) > replay_len:
            del experience_replay[0]
        optimizer.zero_grad()
        loss = compute_loss(experience_replay)
        optimizer.step()
        loss_values.append(loss)
        all_loss.append(loss)
        print("\tloss = ", loss)


        print("Weights : ")
        print(gcn.conv1.lin.weight)
        print(gcn.conv2.lin.weight)

    
        # acceptance rate
        pred = torch.sigmoid(acceptance_predictor(data, ydata))
        print("PRED : ", pred.shape)
        print(pred)
        accept_rate = min(1.0, Zx0/Zx)
        print(torch.tensor([accept_rate], dtype=torch.float32))
        accept_loss = F.kl_div(pred.log(), torch.tensor([accept_rate], dtype=torch.float32))
        print("Accept loss : ", accept_loss)
        acceptance_optimizer.zero_grad()
        accept_loss.backward()
        acceptance_optimizer.step()
        accept_loss_values.append(accept_loss.item())
        all_accept_loss.append(accept_loss.item())
        print("\tc(x), c(y) = ", E0, solution_cost(x, tsp))
        print("\tZx0, Zx    = ", Zx0, Zx)
        print("\taccept rate : ", accept_rate)
        if np.random.rand() < accept_rate:
            print("\t\taccept")
            sol = x
            solx = edges_to_flow(sol, tsp, formulation="mtz")
            data = ydata

            solc = solution_cost(sol, tsp)
            if solc < best_cost:
                best_cost = solc
                fig, ax = plt.subplots(1,1)
                plot_solution(sol, tsp, ax=ax)
                plt.tight_layout()
                plt.savefig(f"test_{kk}_sol.png")
                plt.close(fig)
        else:
            print("\t\treject")
    
        print("\tCandidate ", sol.sequence, solution_cost(sol,tsp))
        cost_values.append(solution_cost(sol, tsp))
    
    
    
    
    fig,ax = plt.subplots(2,2)
    ax[0,0].plot(cost_values)
    ax[0,0].set_title("Cost")
    ax[0,1].plot(loss_values)
    ax[0,1].set_title("Loss")
    ax[1,0].hist(cost_values, bins=20)
    ax[1,0].set_title("Cost")
    ax[1,1].plot(accept_loss_values)
    ax[1,1].set_title("accept loss")
    plt.tight_layout()
    plt.savefig(f"test_{kk}.png")
    plt.close(fig)

    fig,ax = plt.subplots(2,1)
    ax[0].plot(all_loss)
    ax[0].set_title("Loss")
    ax[1].plot(all_accept_loss)
    ax[1].set_title("Accept Loss")
    plt.tight_layout()
    plt.savefig(f"test_loss.png")
    plt.close(fig)

    mean_costs.append(np.mean(cost_values))
    std_costs.append(np.mean(cost_values))
    skew_costs.append(skew(cost_values))
    kurtosis_costs.append(kurtosis(cost_values))

    fig,ax = plt.subplots(4,1)
    ax[0].plot(mean_costs)
    ax[0].set_title("Mean cost")
    ax[1].plot(std_costs)
    ax[1].set_title("Std")
    ax[2].plot(skew_costs)
    ax[2].set_title("Skewness")
    ax[3].plot(kurtosis_costs)
    ax[3].set_title("Kurtosis")

    plt.tight_layout()
    plt.savefig(f"test_stats.png")
    plt.close(fig)
   




