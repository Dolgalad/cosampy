import torch
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing

class EdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='max') #  "Max" aggregation.
        self.mlp = Seq(Linear(3 * in_channels, out_channels),
                       ReLU(),
                       Linear(out_channels, out_channels))

    def forward(self, x, e, edge_index):
        # x has shape [N, in_channels]
        # e has shape [E, in_channels]
        # edge_index has shape [2, E]

        return self.propagate(edge_index, x=x, e=e)

    def message(self, x_i, x_j, e):
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]
        # e_ij has shape [E, in_channels]

        tmp = torch.cat([x_i, x_j - x_i, e], dim=1)  # tmp has shape [E, 2 * in_channels]
        return self.mlp(tmp)

if __name__=="__main__":
    from torch_geometric.data import Data
    import random
    import numpy as np
    #random.seed(123)
    #np.random.seed(123)
    #torch.use_deterministic_algorithms(mode=True)
    #torch.cuda.manual_seed(123)
    #torch.cuda.manual_seed_all(123)
    edge_index = torch.tensor([[0,1],
                               [1,0],
                               [1,2],
                               [2,1]], dtype=torch.long)
    x = torch.tensor([[0], [0], [0]], dtype=torch.float)
    e = torch.tensor([[-2], [-1], [0], [1]], dtype=torch.float)

    data = Data(x=x, e=e, edge_index=edge_index.t().contiguous())
    print(data)

    econv = EdgeConv(1, 2)
    y=econv(data.x, data.e, data.edge_index)
    print(y)

