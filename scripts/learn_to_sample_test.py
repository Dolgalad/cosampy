import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils.convert import from_networkx
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt

from sampco.co.tsp import *

import time
import copy

# max iterations per iteration
max_iter = 1000
num_samples = 100
noise_std = 0.01
epochs = 1000
nV = 6
replay_len = 1000

# create a random ETSP instance
tsp = random_euclidean_tsp(nV)

# initial solution given by oracle
sol = nearest_neighbor_route(tsp)
solx = edges_to_flow(sol, tsp)
solx = np.concatenate((solx, np.zeros(tsp.number_of_nodes())))
print("N = ", tsp.number_of_edges())
print("solx shape = ", solx.shape)
fg = tsp.factor_graph(solx, formulation="mtz")
print("nv,ne = ", nx.number_of_nodes(fg), nx.number_of_edges(fg))
data = from_networkx(fg)
data.x = data.x.to(torch.float)
print(data.x[data.x[:,0]==0, 2].shape)
print(solx.shape)
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(3, 16)
        self.conv2 = GCNConv(16, 1)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, 0.5)
        x = self.conv2(x, edge_index)
        return F.elu(x)

def normalize_costs(d):
    gcosts = d.x[d.x[:,0]==0, 1]
    d.x[d.x[:,0]==0, 1] = (gcosts - torch.mean(gcosts)) / torch.std(gcosts)
    return d


# Experience replay
experience_replay = []

gcn = GCN()

optimizer = torch.optim.Adam(gcn.parameters(), lr=0.0001)

print(gcn(data)[data.x[:,0] == 0].shape)

print(nearest_neighbor_route_2(tsp.cost_vector(), data.edge_index, tsp).sequence)
print(nearest_neighbor_route_2(gcn(data)[data.x[:,0]==0].detach(), data.edge_index, tsp).sequence)

for kk in range(epochs):

    # create a random ETSP instance
    tsp = random_euclidean_tsp(nV+int(kk/50))
    
    # initial solution given by oracle
    sol = nearest_neighbor_route(tsp)
    solx = edges_to_flow(sol, tsp, formulation="mtz")
    #solx = np.concatenate((solx, np.zeros(tsp.number_of_nodes())))

    best_sol = copy.deepcopy(sol)
    best_cost = solution_cost(sol, tsp)
    fig, ax = plt.subplots(1,1)
    plot_solution(sol, tsp, ax=ax)
    plt.savefig(f"test_{kk}_sol.png")
    plt.close(fig)

    
    data = from_networkx(tsp.factor_graph(solx, formulation="mtz"))
    data.x = data.x.to(torch.float)
    data = normalize_costs(data)

    cost_values = []
    loss_values = []
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
            if solp!=x:
                x_neighbors.append(solp)
        Zx = np.sum([np.exp(0.5 * (solution_cost(x,tsp) - solution_cost(n,tsp))) for n in x_neighbors])

        # evaluate the loss function
        # approximation of the optimal proposal distribution
        unique_neighbors = []
        unique_noise= []
        for ii, n in enumerate(neighbors):
            if n != sol and not n in unique_neighbors:
                unique_neighbors.append(n)
                unique_noise.append(noise[ii])
        if len(unique_neighbors)<=0:
            print("Unique neighbors : ", len(unique_neighbors))
            break
        Zopt = np.sum([np.exp(0.5 * (E0 - solution_cost(n,tsp))) for n in unique_neighbors])
        loss = 0.0

        dy = np.zeros_like(unique_noise[0])
        for ii,n in enumerate(unique_neighbors):
            # evaluate optimal probability
            qopt = np.exp(0.5 * (E0 - solution_cost(n,tsp))) / Zopt
            qapp = np.sum([n == ni for ni in neighbors]) / len(neighbors)
            #print("\t", n.sequence, qopt, qapp)
            loss += qopt * (np.log(qopt) - np.log(qapp))
            dy -= (1.0/Zopt) * (1.0/noise_std) * unique_noise[ii] / qapp
        #print("BB ", np.sum([np.sum([n == ni for ni in unique_neighbors]) for n in neighbors]), len(neighbors))
        if loss==0:
            print("\n# neighbors        : ", len(neighbors))
            print("\n# unique neighbors : ", len(unique_neighbors))
            for n in unique_neighbors:
                E1 = solution_cost(n,tsp)
                qopt = np.exp(0.5 * (E0 - solution_cost(n,tsp))) / Zopt
                qapp = np.sum([n == ni for ni in neighbors]) / len(neighbors)
                print("\t", E0, E1, qopt, qapp)

        loss_values.append(loss)
        print("\tloss = ", loss)
        # update the parameters of the model
        y = torch.ones(tmp_costs.size()).requires_grad_(True)
        #print(y.size(), data.x[:,0].size())
        #tmp_costs[data.x[:,0]==0].backward(torch.sum(torch.tensor(noise).requires_grad_(True), dim=0))

        optimizer.zero_grad()
        #tmp_costs[data.x[:,0]==0].backward(torch.tensor(dy).requires_grad_(True))
        new_costs.backward(torch.tensor(dy).requires_grad_(True))

        optimizer.step()

        #print(gcn.conv1.lin.weight)
        #with torch.no_grad():
        #    gcn.conv1.lin.weight -= 0.00001 * gcn.conv1.lin.weight.grad
        #    gcn.conv2.lin.weight -= 0.00001 * gcn.conv2.lin.weight.grad

    
        # acceptance rate
        accept_rate = min(1.0, Zx0/Zx)
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
                plt.savefig(f"test_{kk}_sol.png")
                plt.close(fig)
        else:
            print("\t\treject")
    
        print("\tCandidate ", sol.sequence, solution_cost(sol,tsp))
        cost_values.append(solution_cost(sol, tsp))
    
    
    
    
    fig,ax = plt.subplots(1,3)
    ax[0].plot(cost_values)
    ax[0].set_title("Cost")
    ax[1].plot(loss_values)
    ax[1].set_title("Loss")
    ax[2].hist(cost_values, bins=20)
    ax[2].set_title("Cost")
    
    plt.savefig(f"test_{kk}.png")
    plt.close(fig)




