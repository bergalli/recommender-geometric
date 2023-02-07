import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import SAGEConv, to_hetero


def weighted_mse_loss(pred, target, weight=None):
    weight = 1.0 if weight is None else weight[target].to(pred.dtype)
    return (weight * (pred - target.to(pred.dtype)).pow(2)).mean()


def train_model(model, optimizer, train_data, weight=None):
    model.train()
    optimizer.zero_grad()

    edge_label_index_remapping = _remap_batch_node_ids(train_data["user", "movie"].edge_label_index)

    pred = model(train_data.x_dict, train_data.edge_index_dict, edge_label_index_remapping)
    target = train_data["user", "movie"].edge_label
    loss = weighted_mse_loss(pred, target, weight)
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test_model(model, data):
    model.eval()

    edge_label_index_remapping = _remap_batch_node_ids(data["user", "movie"].edge_label_index)

    pred = model(
        data.x_dict, data.edge_index_dict, edge_label_index_remapping
    )  # data["user", "movie"].edge_label_index

    pred = pred.clamp(min=0, max=5)
    target = data["user", "movie"].edge_label.float()
    rmse = F.mse_loss(pred, target).sqrt()
    return float(rmse)


def _remap_batch_node_ids(edge_label_index):
    user_node_ids_mapper = {
        int(old_id): new_id
        for new_id, old_id in enumerate(edge_label_index[0, :].unique())
    }
    movie_node_ids_mapper = {
        int(old_id): new_id
        for new_id, old_id in enumerate(edge_label_index[1, :].unique())
    }
    edge_label_index_remapping = torch.tensor(
        [
            [user_node_ids_mapper[int(id)] for id in edge_label_index[0, :]],
            [movie_node_ids_mapper[int(id)] for id in edge_label_index[1, :]],
        ]
    )
    return edge_label_index_remapping


class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        # these convolutions have been replicated to match the number of edge types
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        # concat user and movie embeddings
        z = torch.cat([z_dict["user"][row], z_dict["movie"][col]], dim=-1)
        # concatenated embeddings passed to linear layer
        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)


class Model(torch.nn.Module):
    def __init__(self, hidden_channels, data):
        super().__init__()
        self.encoder = GNNEncoder(hidden_channels, hidden_channels)
        self.encoder = to_hetero(self.encoder, data.metadata())
        self.decoder = EdgeDecoder(hidden_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        # z_dict contains dictionary of movie and user embeddings returned from GraphSage
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index)
