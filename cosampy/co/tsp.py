"""
Definition of a Traveling Salesman Problem data container
"""
import networkx as nx
import numpy as np
from itertools import product, chain, combinations
import matplotlib.pyplot as plt

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

class ETSP:
    def __init__(self, pos):
        self.pos = pos
    def number_of_nodes(self):
        return self.pos.shape[0]
    def number_of_edges(self):
        N = self.number_of_nodes()
        return N*(N-1)
    def node_distance(self, n1, n2):
        if n1==n2:
            return 0.0
        return np.linalg.norm(self.pos[n1] - self.pos[n2])
    def edge_iter(self):
        N = self.number_of_nodes()
        for i,j in product(range(N), range(N)):
            if i!=j:
                yield (i,j)
    def edge_idx(self, i, j):
        N = self.number_of_nodes()
        return i*(N-1) + j - (i<j)
    def ilp_number_of_variables(self, formulation="dfj"):
        if formulation=="dfj":
            return self.number_of_edges()
        if formulation=="mtz":
            return self.number_of_edges() + self.number_of_nodes()
    def cost_vector(self, formulation="dfj"):
        """Returns cost vector c = \{ c_ij \}, (i,j)\in A
        """
        if formulation=="dfj":
            return np.array([self.node_distance(i,j) for i,j in self.edge_iter()])
        if formulation=="mtz":
            return np.concatenate((np.array([self.node_distance(i,j) for i,j in self.edge_iter()]),
                                   np.zeros(self.number_of_nodes())
                                 ))

    def entering_flow_constraints(self, formulation="dfj"):
        """Constraint matrix and right hand side ensuring all nodes are visited exactly once
        """
        N = self.number_of_nodes()
        rhs = np.ones(N)
        C = np.zeros((N, self.ilp_number_of_variables(formulation=formulation)))
        for j in range(N):
            for i in range(N):
                if i!=j:
                    C[j,self.edge_idx(i,j)] = 1
        return C, rhs
    def outgoing_flow_constraints(self, formulation="dfj"):
        """Constraint matrix and right hand side ensuring all nodes are visited exactly once
        """
        N = self.number_of_nodes()
        rhs = np.ones(N)
        C = np.zeros((N, self.ilp_number_of_variables(formulation=formulation)))
        for i in range(N):
            for j in range(N):
                if i!=j:
                    C[i,self.edge_idx(i,j)] = 1
        return C, rhs
    def subtour_breaking_constraints(self, formulation="dfj"):
        N = self.number_of_nodes()
        C, rhs = [], []
        if formulation=="dfj":
            for S in powerset(range(N)):
                if len(S)<=1 or len(S)==N:
                    continue
                row = np.zeros(N*(N-1))
                for (i,j) in product(S,S):
                    if i==j:
                        continue
                    row[self.edge_idx(i,j)] = 1
                C.append(row)
                rhs.append(len(S)-1)
        if formulation=="mtz":
            for i in range(1,N):
                for j in range(1,N):
                    if i==j:
                        continue
                    row = np.zeros(N*(N-1) + N)
                    row[N*(N-1) + i] = 1
                    row[N*(N-1) + j] = -1
                    row[self.edge_idx(i,j)] = N
                    C.append(row)
                    rhs.append(N-1)

        return np.array(C), np.array(rhs)
    def ilp_constraints(self, formulation="dfj"):
        """TSP ILP formulations: returns the constraint matrix and right hand side and the sense of the constraint
        """
        eC, eC_rhs = self.entering_flow_constraints(formulation=formulation)
        oC, oC_rhs = self.outgoing_flow_constraints(formulation=formulation)
        stC, stC_rhs = self.subtour_breaking_constraints(formulation=formulation)
        sense = np.array(["=" for i in range(eC_rhs.shape[0])]+
                         ["=" for i in range(oC_rhs.shape[0])]+
                         ["<=" for i in range(stC_rhs.shape[0])]
                        )
        print(eC.shape, oC.shape, stC.shape)
        print(eC_rhs.shape, oC_rhs.shape, stC_rhs.shape)
        return np.concatenate((eC,oC,stC)), np.concatenate((eC_rhs,oC_rhs,stC_rhs)), sense
    def ilp_formulation(self, formulation="dfj"):
        C,rhs,sense = self.ilp_constraints(formulation=formulation)
        return self.cost_vector(formulation=formulation), C, rhs, sense
    def factor_graph(self, sol=None, formulation="dfj"):
        B = nx.Graph()
        c = self.cost_vector(formulation=formulation)
        for i,ci in enumerate(c):
            if sol is None:
                B.add_node(f"x{i}", x=[0,ci])
            else:
                B.add_node(f"x{i}", x=[0,ci,*sol[i]])
        C,rhs,s = self.ilp_constraints(formulation=formulation)
        if sol is not None:
            slack = rhs - np.dot(C, sol[:,0])
            print("slack ", slack.shape)
        for i,rhsi in enumerate(rhs):
            if sol is None:
                B.add_node(f"c{i}", x=[1,rhsi])
            else:
                B.add_node(f"c{i}", x=[1,rhsi,slack[i], 0])
        # add node-constraint edges
        for i in range(C.shape[0]):
            for j in range(C.shape[1]):
                if C[i,j]!=0:
                    B.add_edge(f"x{j}", f"c{i}")
        return B






class ETSPSolution:
    def __init__(self, seq):
        self.sequence = seq
    def edge_iter(self):
        for i in range(1,len(self.sequence)):
            yield self.sequence[i-1], self.sequence[i]
        yield self.sequence[-1], self.sequence[0]
    def __eq__(self, other):
        #print(self.sequence)
        #print(other.sequence)
        #print(np.array(self.sequence) == np.array(other.sequence))
        #print(np.all(np.array(self.sequence) == np.array(other.sequence)))

        return np.all(np.array(self.sequence) == np.array(other.sequence))


def check_solution(sol: ETSPSolution, pb: ETSP) -> bool:
    nV = pb.number_of_nodes()
    if len(sol.sequence) != nV:
        print("Bad number of nodes : ", len(sol.sequence), nV)
        return False
    # mark all nodes as non visited
    Q = np.zeros(nV, dtype=bool)
    # check that all nodes are visited once
    for n in sol.sequence:
        if Q[n] == True:
            print("Visiting node ", n, " twice")
            return False
        else:
            Q[n] = True
    print("all nodes visited : ", np.all(Q))
    return np.all(Q)

def random_euclidean_tsp(n: int) -> ETSP:
    return ETSP(np.random.rand(n,2))

def nearest_neighbor_route(pb: ETSP) -> ETSPSolution:
    nV = pb.number_of_nodes()
    nodes = np.arange(nV)
    # initialize sequence of visited nodes by [0] (start at node 0)
    sequence = [0]
    # mark all other nodes as non visited
    visited = np.zeros(nV, dtype=bool)
    visited[0] = True
    while np.any(visited == False):
        current_position = pb.pos[sequence[-1]]
        # candidate indices
        candidates = nodes[ visited == False ]
        candidates_pos = pb.pos[ visited == False]
        distances = np.linalg.norm( candidates_pos - current_position, axis=1)
        nearest_neighbor = candidates[np.argmin(distances)]
        sequence.append(nearest_neighbor)
        visited[nearest_neighbor] = True
    return ETSPSolution(sequence)

def nearest_neighbor_route_2(c, edge_index, pb):
    nV = pb.number_of_nodes()
    nodes = np.arange(nV)
    sequence = [0]
    visited = np.zeros(nV, dtype=bool)
    visited[0] = True
    while np.any(visited == False):
        candidates = nodes[ visited == False ]
        candidate_edges = []
        candidate_edge_costs = []
        for j in range(nV):
            if not visited[j]:
                e = (sequence[-1], j)
                candidate_edges.append(e)
                candidate_edge_costs.append(c[pb.edge_idx(e[0],e[1])])

        selected_edge = candidate_edges[np.argmin(candidate_edge_costs)]
        sequence.append(selected_edge[1])
        visited[selected_edge[1]] = True
    return ETSPSolution(sequence)


def solution_cost(sol: ETSPSolution, pb: ETSP) -> float:
    nV = len(sol.sequence)
    c = 0.0
    for i in range(1, nV):
        c += pb.node_distance(sol.sequence[i-1], sol.sequence[i])
    c += pb.node_distance(sol.sequence[-1], sol.sequence[0])
    return c

def two_opt(sol: ETSPSolution, pb: ETSP) -> ETSPSolution:
    n = pb.number_of_nodes()
    H = np.copy(sol.sequence)
    improvement = True
    while improvement:
        improvement = False
        for i,xi in enumerate(H):
            for j,xj in enumerate(H):
                if abs( i - j ) <= 1:
                    continue
                if pb.node_distance(xi, H[(i+1)%n]) + pb.node_distance(xj, H[(j+1)%n]) > pb.node_distance(xi, xj) + pb.node_distance(H[(i+1)%n], H[(j+1)%n]):
                    xi1 = H[(i+1)%n]
                    xi1pos = np.argwhere(H == xi1)
                    xjpos = np.argwhere(H == xj)
                    H[xi1pos] = xj
                    H[xjpos] = xi1
                    improvement = True
    return ETSPSolution(H)


def plot_solution(sol: ETSPSolution, pb: ETSP, ax=None, color="blue", alpha=1.0):
    if ax is None:
        fig, ax = plt.subplots()
    x, y = pb.pos[:,0], pb.pos[:,1]
    n = np.arange(len(sol.sequence))
    ax.scatter(x, y)
    
    for i, txt in enumerate(n):
        ax.annotate(txt, (x[i], y[i]))

    for i in range(1, len(sol.sequence)):
        ax.arrow(*pb.pos[sol.sequence[i-1]], *(pb.pos[sol.sequence[i]] - pb.pos[sol.sequence[i-1]]), head_width=.01, fc='k', ec=color, length_includes_head=True, alpha=alpha)
    ax.arrow(*pb.pos[sol.sequence[-1]], *(pb.pos[sol.sequence[0]] - pb.pos[sol.sequence[-1]]), head_width=.01, fc='k', ec=color, length_includes_head=True,  alpha=alpha)
    return ax

def edge_list_to_route(el):
    r = [0]
    while len(el)>0:
        for e in el:
            if e[0] == r[-1]:
                r.append(e[1])
                el.remove(e)
                break
    return r[:-1]

def edges_to_flow(sol, pb, formulation="dfj"):
    N = pb.number_of_nodes()
    x = np.zeros(pb.ilp_number_of_variables(formulation=formulation))
    for e in sol.edge_iter():
        x[pb.edge_idx(e[0], e[1])] = 1
    return x

def flow_to_edges(x, pb):
    edge_list = [e for e in pb.edge_iter()]
    route = []
    return [edge_list[i[0]] for i in np.argwhere(x > 0)]

if __name__=="__main__":
    pb = random_euclidean_tsp(7)
    c = pb.cost_vector()
    print("Costs : ", c.shape)

    edge_list = list(pb.edge_iter())
    print(edge_list)
    redges_idx = np.random.choice(range(len(edge_list)), 15)
    redges = [edge_list[e] for e in redges_idx]
    print([edge_list.index(e) for e in redges])
    print([pb.edge_idx(i,j) for (i,j) in redges])

    print("Entering flow constraints")
    eC, e_rhs = pb.entering_flow_constraints()
    print(eC)
    print(e_rhs)
    print("Outgoind flow constraints")
    oC, o_rhs = pb.outgoing_flow_constraints()
    print(oC)
    print(o_rhs)
    print("Subtour breaking constraints")
    stC, st_rhs = pb.subtour_breaking_constraints()
    print(stC)
    print(st_rhs)

    C, rhs = pb.ilp_constraints()
    print(C.shape, rhs.shape)

    import scipy
    from scipy.optimize import LinearConstraint, milp
    constraints = LinearConstraint(C, rhs)
    print(constraints)
    integrality = np.ones_like(c)
    constraints = [
            LinearConstraint(eC, e_rhs, e_rhs),
            LinearConstraint(oC, o_rhs, e_rhs),
            LinearConstraint(stC, ub=st_rhs)
            ]
    res = milp(c=c, constraints=constraints, integrality=integrality)
    #res = milp(c=c, constraints=constraints)

    print(res.status)
    print(res.x)
    # make edge list
    el = [edge_list[i[0]] for i in np.argwhere(res.x > 1e-8)]
    print(el)

    sol = ETSPSolution(edge_list_to_route(el))
    print("Solution          : ", sol.sequence)
    print("Solution is valid : ", check_solution(sol, pb))
    print("Solution cost     : ", solution_cost(sol, pb))

    fig, ax = plt.subplots(1, 2)
    plot_solution(sol, pb, ax=ax[0])
    solnn = nearest_neighbor_route(pb)
    print("NN Solution cost  : ", solution_cost(solnn, pb))
    plot_solution(solnn, pb, ax=ax[1])


    # Gibbs
    import os
    from sampco.sampling.gibbs import *
    print("Gibbs 1 cost only")
    costs = []
    cost_f = lambda x: -np.dot(c,x)
    dirname = "gibbs_1_cost"
    if os.path.isdir(dirname):
        # remove contents
        os.system(f"rm {dirname}/*")
    os.makedirs(dirname, exist_ok=True)
    prevx = np.random.choice([0,1], np.shape(res.x), replace=True)
    for k,xx in enumerate(gibbs(cost_f, prevx, max_iter=1)):
        costs.append(cost_f(xx))
        if np.any(xx != prevx):
            prevx = xx
            # draw the result
            el = [edge_list[i[0]] for i in np.argwhere(xx > 1e-8)]
            fig, ax = plt.subplots(1, 1)
            x, y = pb.pos[:,0], pb.pos[:,1]
            n = np.arange(pb.number_of_nodes())
            ax.scatter(x, y)
            
            for i, txt in enumerate(n):
                ax.annotate(txt, (x[i], y[i]))
        
            for e in el:
                ax.arrow(*pb.pos[e[0]], *(pb.pos[e[1]] - pb.pos[e[0]]), head_width=.01, fc='k', ec='k', length_includes_head=True)

            plt.savefig(os.path.join(dirname, f"{k}.png"))

            plt.close(fig)


    fig = plt.figure()
    plt.plot(costs)

    # Gibbs with penalized constraints
    print("Gibbs 1 lagrangian unit penalty")

    stC2 = np.concatenate((stC, -stC))
    st_rhs2 = np.concatenate((st_rhs, -st_rhs))
    le = 20 * np.ones_like(e_rhs)
    lo = 20 * np.ones_like(o_rhs)
    lst = 20 * np.ones_like(st_rhs2)
    lagrangian = lambda x: -np.dot(c,x) - np.dot(le, np.abs(np.dot(eC, x) - e_rhs)) - np.dot(lo, np.abs(np.dot(oC,x) - o_rhs)) - np.dot(lst, np.abs(np.dot(stC2,x) - st_rhs2))
    costs2 = []
    dirname = "gibbs_1_lagrangin"
    if os.path.isdir(dirname):
        # remove contents
        os.system(f"rm {dirname}/*")
    os.makedirs(dirname, exist_ok=True)

    prevx = np.random.choice([0,1], np.shape(res.x), replace=True)
    for k,xx in enumerate(gibbs(lagrangian, prevx, max_iter=1)):
        costs2.append(cost_f(xx))
        if np.any(xx != prevx):
            prevx = xx
            # draw the result
            el = [edge_list[i[0]] for i in np.argwhere(xx > 1e-8)]
            fig, ax = plt.subplots(1, 1)
            x, y = pb.pos[:,0], pb.pos[:,1]
            n = np.arange(pb.number_of_nodes())
            ax.scatter(x, y)
            
            for i, txt in enumerate(n):
                ax.annotate(txt, (x[i], y[i]))
        
            for e in el:
                ax.arrow(*pb.pos[e[0]], *(pb.pos[e[1]] - pb.pos[e[0]]), head_width=.01, fc='k', ec='k', length_includes_head=True)
            plt.savefig(os.path.join(dirname, f"{k}.png"))
            plt.close(fig)


    fig = plt.figure()
    plt.plot(costs2)

    # primal dual sampling
    from sampco.sampling.gibbs import primal_dual_sampling
    CC = np.concatenate((eC, -eC, oC, -oC, stC))
    rhs = np.concatenate((e_rhs, -e_rhs, o_rhs, -o_rhs, st_rhs))
    print(CC.shape, rhs.shape, res.x.shape)

    lagrangian = lambda x,y: -np.dot(c,x) - np.dot(y, np.dot(CC, x) - rhs)
    grady_lagrangian = lambda x,y: -(rhs - np.dot(CC, x))
    costs3, lagrangian_vals = [], []
    dirname = "dual_sampling"
    if os.path.isdir(dirname):
        # remove contents
        os.system(f"rm {dirname}/*")
    os.makedirs(dirname, exist_ok=True)

    prevx = np.random.choice([0,1], np.shape(res.x), replace=True)
    prevy = np.zeros_like(rhs)
    for k,(xx,yy) in enumerate(primal_dual_sampling(lagrangian, grady_lagrangian, prevx, prevy, max_iter=10000)):
        costs3.append(np.dot(c,xx))
        lagrangian_vals.append(lagrangian(xx,yy))
        if np.any(xx != prevx):
            print(xx, yy)
            prevx = xx
            # draw the result
            el = [edge_list[i[0]] for i in np.argwhere(xx > 1e-8)]
            fig, ax = plt.subplots(1, 1)
            x, y = pb.pos[:,0], pb.pos[:,1]
            n = np.arange(pb.number_of_nodes())
            ax.scatter(x, y)
            
            for i, txt in enumerate(n):
                ax.annotate(txt, (x[i], y[i]))
        
            for e in el:
                ax.arrow(*pb.pos[e[0]], *(pb.pos[e[1]] - pb.pos[e[0]]), head_width=.01, fc='k', ec='k', length_includes_head=True)
            plt.savefig(os.path.join(dirname, f"{k}.png"))
            plt.close(fig)
    fig = plt.figure()
    plt.plot(costs3)
    fig = plt.figure()
    plt.plot(lagrangian_vals)


    plt.show()

    exit()

    # solve with nearest neighbor heuristic solver
    sol = nearest_neighbor_route(pb)
    # print solution path
    print("Solution          : ", sol.sequence)
    # check the validity of the solution
    print("Solution is valid : ", check_solution(sol, pb))
    # print solution cost
    print("Solution cost     : ", solution_cost(sol, pb))
    fig, ax = plt.subplots()
    x, y = pb.pos[:,0], pb.pos[:,1]
    n = np.arange(len(sol.sequence))
    ax.scatter(x, y)
    
    for i, txt in enumerate(n):
        ax.annotate(txt, (x[i], y[i]))

    for i in range(1, len(sol.sequence)):
        plt.arrow(*pb.pos[sol.sequence[i-1]], *(pb.pos[sol.sequence[i]] - pb.pos[sol.sequence[i-1]]), head_width=.01, fc='k', ec='k', length_includes_head=True)
    plt.arrow(*pb.pos[sol.sequence[-1]], *(pb.pos[sol.sequence[0]] - pb.pos[sol.sequence[-1]]), head_width=.01, fc='k', ec='k', length_includes_head=True)

    #pos = np.concatenate((pb.pos[sol.sequence,:], [pb.pos[0]]))
    #posx, posy = pos[:, 0], pos[:, 1]
    #plt.plot(posx, posy, "->")

    # correct with two_opt
    sol2 = two_opt(sol, pb)
    print("Solution          : ", sol2.sequence)
    print("Solution is valid : ", check_solution(sol2, pb))
    print("Solution cost     : ", solution_cost(sol2, pb))

    plot_solution(sol2, pb)
    plt.show()

    import time
    n_samples = 10
    sizes = 2 ** np.arange(1,6)
    nn_times = []
    nn_costs = []
    twoopt_times = []
    twoopt_costs = []
    twoopt_nn_cost_gaps = []
    for sz in sizes:
        print("Size : ", sz)
        tmp_nn_times = []
        tmp_nn_costs = []
        tmp_twoopt_times = []
        tmp_twoopt_costs = []
        tmp_twoopt_nn_cost_gaps = []
        
        for i in range(n_samples):
            pb = random_euclidean_tsp(sz)
            # heuristic solver
            t0 = time.time()
            sol = nearest_neighbor_route(pb)
            t0 = time.time() - t0
            tmp_nn_times.append(t0)
            tmp_nn_costs.append(solution_cost(sol, pb))
            # two-opt
            t1 = time.time()
            sol2opt = two_opt(sol, pb)
            t1 = time.time() - t1
            tmp_twoopt_times.append(t1)
            tmp_twoopt_costs.append(solution_cost(sol2opt, pb))
            tmp_twoopt_nn_cost_gaps.append((tmp_nn_costs[-1] - tmp_twoopt_costs[-1]) / tmp_nn_costs[-1])
        nn_times.append(np.mean(tmp_nn_times))
        nn_costs.append(np.mean(tmp_nn_costs))
        twoopt_times.append(np.mean(tmp_twoopt_times))
        twoopt_costs.append(np.mean(tmp_twoopt_costs))
        twoopt_nn_cost_gaps.append(np.mean(tmp_twoopt_nn_cost_gaps))

    fig, ax = plt.subplots(1, 3)
    ax[0].plot(sizes, nn_times, label="nn")
    ax[0].plot(sizes, twoopt_times, label="2opt")
    ax[0].set_title("Time (s)")
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].legend()
    ax[1].plot(sizes, nn_costs, label="nn")
    ax[1].plot(sizes, twoopt_costs, label="2opt")
    ax[1].set_title("Cost")
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")

    ax[1].legend()
    ax[2].plot(sizes, twoopt_nn_cost_gaps)
    plt.show()

    dim1,dim2 = 0, 1
    pb = random_euclidean_tsp(10)
    delta = 0.025
    x = np.arange(-1.0, 1.0, delta)
    y = np.arange(-1.0, 1.0, delta)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)
    Z2 = np.zeros(X.shape)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            tmp_c = np.copy(pb.pos)
            tmp_c[dim1, 0] = X[i,j]
            tmp_c[dim1, 1] = Y[i,j]
            pb1 = ETSP(tmp_c)
            sol = nearest_neighbor_route(pb1)
            Z[i,j] = solution_cost(sol, pb)
            sol = two_opt(sol, pb)
            Z2[i,j] = solution_cost(sol, pb)

    fig, ax = plt.subplots(1,2)
    CS = ax[0].contour(X, Y, Z)
    ax[0].clabel(CS, inline=True, fontsize=10)
    CS2 = ax[1].contour(X, Y, Z)
    ax[1].clabel(CS2, inline=True, fontsize=10)

    plt.show()







