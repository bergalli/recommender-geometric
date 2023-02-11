"""
This is a boilerplate pipeline 'recommender_model'
generated using Kedro 0.18.4
"""

import torch
from petastorm.spark import SparkDatasetConverter, make_spark_converter
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader, DataLoader
from torch_geometric.transforms import RandomLinkSplit, ToUndirected
import torch.multiprocessing as mp
from .model import Model, train_model, test_model

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os


def make_graph_tensors(
    source_nodes,
    dest_nodes,
    weight,
    pivoted_nodes_genome_scores,
):
    source_nodes_converter = make_spark_converter(source_nodes)
    dest_nodes_converter = make_spark_converter(dest_nodes)
    weight_converter = make_spark_converter(weight)
    pivoted_nodes_genome_scores_converter = make_spark_converter(pivoted_nodes_genome_scores)

    edge_index = torch.tensor([source_nodes, dest_nodes])
    edge_label = torch.tensor(weight)

    movies_nodes_attr = pivoted_nodes_genome_scores.values.tolist()
    movies_nodes_attr = torch.tensor(movies_nodes_attr)

    return edge_index, edge_label, movies_nodes_attr


def create_pyg_network(edge_index, edge_label, movies_nodes_attr) -> HeteroData:

    data = HeteroData()

    data["movie"].x = movies_nodes_attr
    n_users = len(set(edge_index[0, :].tolist()))
    data["user"].x = torch.eye(n_users)

    data["user", "rates", "movie"].edge_index = edge_index
    data["user", "rates", "movie"].edge_label = edge_label

    # data_list = []
    # batch_size=1000
    # for idx in np.arange(0, edge_index.shape[1], batch_size):
    #     train_data = HeteroData()
    #
    #     train_data["movie"].x = movies_nodes_attr
    #     n_users = len(set(edge_index[0, :].tolist()))
    #     train_data["user"].x = torch.eye(n_users)
    #
    #     train_data["user", "rates", "movie"].edge_index = edge_index[:, idx:idx + batch_size]
    #     train_data["user", "rates", "movie"].edge_label = edge_label[idx:idx + batch_size]
    #
    #     # Add a reverse ('movie', 'rev_rates', 'user') relation for message passing.
    #     train_data = ToUndirected()(train_data)
    #     del train_data["movie", "rev_rates", "user"].edge_label  # Remove "reverse" label.
    #     data_list.append(train_data)

    return data


def make_graph_undirected(data: HeteroData):
    # Add a reverse ('movie', 'rev_rates', 'user') relation for message passing.
    data = ToUndirected()(data)
    del data["movie", "rev_rates", "user"].edge_label  # Remove "reverse" label.

    return data


def make_ttv_data(data: HeteroData):

    # Perform a link-level split into training, validation, and test edges.
    train_data, val_data, test_data = RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        neg_sampling_ratio=0.0,
        edge_types=[("user", "rates", "movie")],  # for heteroData
        rev_edge_types=[("movie", "rev_rates", "user")],  # for heteroData
    )(data)

    return train_data, val_data, test_data


def instanciate_model(train_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(hidden_channels=16, data=train_data).to(device)

    # Due to lazy initialization, we need to run one model step so the number
    # of parameters can be inferred:
    with torch.no_grad():
        model.encoder(train_data.x_dict, train_data.edge_index_dict)

    # We have an unbalanced dataset with different proportions of ratings.
    # We are predicting new edges and the rating associated, herefore we use a weighted MSE loss.
    weight = torch.bincount(train_data["user", "movie"].edge_label)
    weight = weight.max() / weight

    return model, weight


def get_sampler_dataloader(train_data):

    unique_user_node_id = torch.tensor(
        list(set(train_data[("user", "rates", "movie")].edge_index[0, :].tolist()))
    )

    subgraph_loader = NeighborLoader(
        train_data,
        batch_size=4096,
        shuffle=False,
        num_neighbors={key: [-1] * 2 for key in train_data.edge_types},
        input_nodes=("user", unique_user_node_id),
    )

    return subgraph_loader


def train_gcn_model(model, weight, train_dataloader, val_data, test_data):
    # model = config["model"]
    # weight = config["weight"]
    # train_dataloader = config["train_dataloader"]
    # val_data = config["val_data"]
    # test_data = config["test_data"]

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(15):
        for train_data in iter(train_dataloader):
            loss = train_model(model, optimizer, train_data, weight)
            train_rmse = test_model(model, train_data)
            val_rmse = test_model(model, val_data)
            test_rmse = test_model(model, test_data)
            print(
                f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_rmse:.4f}, "
                f"Val: {val_rmse:.4f}, Test: {test_rmse:.4f}"
            )

    return model


def train_gcn_model_distributed(
    model: Model,
    weight: torch.Tensor,
    train_dataloader: DataLoader,
    val_data,
    test_data,
):
    def train_gcn_fun(rank, world_size, model, weight, train_dataloader, val_data, test_data):

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"

        # initialize the process group
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

        # create model and move it to GPU with id rank
        model = model.to("cpu")
        ddp_model = DDP(model, device_ids=None)

        optimizer = torch.optim.Adam(ddp_model.parameters(), lr=0.01)
        for epoch in range(15):
            for train_data in iter(train_dataloader):
                loss = train_model(ddp_model, optimizer, train_data, weight)
                train_rmse = test_model(ddp_model, train_data)
                val_rmse = test_model(ddp_model, val_data)
                test_rmse = test_model(ddp_model, test_data)
                print(
                    f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_rmse:.4f}, "
                    f"Val: {val_rmse:.4f}, Test: {test_rmse:.4f}"
                )

    world_size = 3
    mp.spawn(
        train_gcn_fun,
        args=(world_size, model, weight, train_dataloader, val_data, test_data),
        nprocs=world_size,
        join=True,
    )

    return model
