import numpy as np
import networkx as nx

class CSP:
    def __init__(self, g, src, dst, ub):
        self.g = g
        self.src = src
        self.dst = dst
        self.ub = ub


def random_csp(graph_generator=lambda: nx.generators.grid_2d_graph(5,5)):
    # make graph directed
    g = nx.to_directed(graph_generator())
    # set weights and resource to 1
    w = {e: 1. for e in g.edges}
    r = {e: 1. for e in g.edges}
    nx.set_edge_attributes(g, w, "weight")
    nx.set_edge_attributes(g, r, "resource")
    # select source and destination at random
    [src,dst] = np.random.choice(range(nx.number_of_nodes(g)), 2, replace=False)

    return CSP(g, src, dst, 3.)

def factor_graph(csp):
    B = nx.Graph()
    # add variable nodes, one for each edge
    edge_list = list(csp.g.edges)
    nE = nx.number_of_edges(csp.g)
    B.add_nodes_from([f"x_{i}" for i in range(nE)], bipartite=0)
    B.set_node_attributes({f"x_{i}":csp.g.edges[edge_list[i]]["weight"] for i in range(nE)})
    # add constraint nodes, one for each flow constraint (ie number of nodes)
    B.add_nodes_from([f"c_{i}" for i in range(nx.number_of_nodes(csp.g))], bipartite=1)
    return B



if __name__=="__main__":
    csp = random_csp()

