import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import knn_graph

# message passing network
class EdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='max') #  "Max" aggregation.
        
        self.mlp = (
            nn.Sequential(
                nn.Linear(2 * in_channels, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels)
            )
        )

    def forward(self, x, edge_index):
        # x: [N, in_channels]
        # edge_index: [2, E]

        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        # x_i: [E, in_channels]
        # x_j: [E, in_channels]

        con = torch.cat([x_i, x_j - x_i], dim=1)  # tmp has shape [E, 2 * in_channels]
        return self.mlp(con)

class GraphModule(nn.Module):
    def __init__(self, in_channels, out_channels, k=6, layers=2):
        super().__init__()

        self.k = k
        self.mlp = nn.ModuleList()

        # for every message passing layer
        for _ in range(layers - 1):
            self.mlp.append(
              EdgeConv(in_channels, out_channels)
            )
            self.mlp.append(
                nn.ReLU(inplace=True)
            )
            self.mlp.append(
                nn.Dropout(0.5)
            )

        self.mlp.append(
          EdgeConv(in_channels, out_channels)
        )

    def forward(self, object_feat, data_dict):
        # unpack
        batch_size = object_feat.shape[0]
        object_masks = data_dict['bbox_mask']
        object_centers = data_dict['center']
        
        # loop for every sample
        for batch_id in range(batch_size):
            # unpack
            object_mask = object_masks[batch_id]

            # filter out the masked object proposals
            object_feat_new = object_feat[batch_id][object_mask > 0]
            
            # if there are any unmasked object proposals
            if object_feat_new.shape[0] != 0:
                # create a graph with KNN with k=6
                edge_index = knn_graph(object_centers[batch_id][object_mask > 0], self.k, loop=False)

                # forward pass the message passing layers
                for layer in self.mlp:
                    if isinstance(layer, MessagePassing):
                        object_feat_new = layer(object_feat_new, edge_index)
                    else:
                        object_feat_new = layer(object_feat_new)
                        
                # save back the contextualized object proposals
                object_feat[batch_id][object_mask > 0] = object_feat_new
                
        return object_feat