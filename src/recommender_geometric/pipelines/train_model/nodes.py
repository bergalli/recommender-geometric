"""
This is a boilerplate pipeline 'train_model'
generated using Kedro 0.18.5
"""
import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader, DataLoader
from torch_geometric.transforms import RandomLinkSplit, ToUndirected
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os
from recommender_geometric.recommender_model.model import (
    Model,
    train_model,
    test_model,
    weighted_mse_loss,
)
import pyspark.sql


# import horovod.spark.torch as hvd
# hvd.init()
# ddaata = torch.utils.data.distributed.DistributedSampler(
#     source_nodes, num_replicas=hvd.size(), rank=hvd.rank()
# )

# def create_large_dataset():
#
#     return large_dataset


def make_graph_tensors(
    source_nodes: pyspark.sql.DataFrame,
    dest_nodes: pyspark.sql.DataFrame,
    edges_labels: pyspark.sql.DataFrame,
    pivoted_genome_scores: pyspark.sql.DataFrame,
):
    source_nodes = source_nodes.toPandas()
    dest_nodes = dest_nodes.toPandas()
    edges_labels = edges_labels.toPandas()
    pivoted_genome_scores = pivoted_genome_scores.toPandas()

    edge_index = torch.tensor(
        [source_nodes["user_node_id"].values, dest_nodes["movie_node_id"].values]
    )
    del source_nodes, dest_nodes

    edge_label = torch.tensor(edges_labels["rating"].values)
    del edges_labels

    movies_nodes_attr = pivoted_genome_scores
    movies_nodes_attr = torch.tensor(movies_nodes_attr.values)
    del pivoted_genome_scores

    return edge_index, edge_label, movies_nodes_attr


def create_pyg_network(
    edge_index,
    edge_label,
    movies_nodes_attr,
) -> HeteroData:
    data = HeteroData()

    data["movie"].x = movies_nodes_attr

    n_users = len(set(edge_index[0, :].tolist()))
    data["user"].x = torch.eye(n_users)

    data["user", "rates", "movie"].edge_index = edge_index

    data["user", "rates", "movie"].edge_label = edge_label

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
        rev_edge_types=[("movie", "rev_rates", "user")],  # for heteroData, prevents data leakage
    )(data)
    del data

    return train_data, val_data, test_data


def instanciate_model(train_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(hidden_channels=16, metadata=train_data.metadata()).to(device)

    # Due to lazy initialization, we need to run one recommender_model step so the number
    # of parameters of the encoder can be inferred:
    with torch.no_grad():
        # model(train_data.x_dict, train_data.edge_index_dict, train_data.edge_label_dict)
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
    """
    Vanilla training
    """
    # recommender_model = config["recommender_model"]
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


def train_gcn_model_multiproc(
    model: Model,
    weight: torch.Tensor,
    train_dataloader: DataLoader,
    val_data,
    test_data,
):
    """
    Multiprocessing training
    """

    def train_gcn_fun(rank, world_size, model, weight, train_dataloader, val_data, test_data):

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"

        # initialize the process group
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

        # create recommender_model and move it to GPU with id rank
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


def train_gcn_model_distributed_with_lightning(
    model: Model,
    weight: torch.Tensor,
    train_data: DataLoader,
    val_data,
    test_data,
):
    import horovod.spark.torch as hvd

    unique_train_user_node_id = torch.tensor(
        list(set(train_data[("user", "rates", "movie")].edge_index[0, :].tolist()))
    )

    train_subgraph_loader = NeighborLoader(
        train_data,
        batch_size=4096,
        shuffle=False,
        num_neighbors={key: [-1] * 2 for key in train_data.edge_types},
        input_nodes=("user", unique_train_user_node_id),
    )
    unique_test_user_node_id = torch.tensor(
        list(set(train_data[("user", "rates", "movie")].edge_index[0, :].tolist()))
    )
    test_subgraph_loader = NeighborLoader(
        test_data,
        batch_size=4096,
        shuffle=False,
        num_neighbors={key: [-1] * 2 for key in test_data.edge_types},
        input_nodes=("user", unique_test_user_node_id),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    store = (None,)  # store = HDFSStore('/user/username/experiments')
    loss = weighted_mse_loss

    torch_estimator = hvd.TorchEstimator(
        num_proc=4,
        store=store,
        model=model,
        optimizer=optimizer,
        loss=loss,
        feature_cols=["features"],
        label_cols=["y"],
        batch_size=32,
        epochs=10,
    )

    keras_model = torch_estimator.fit(train_subgraph_loader)
    predict_df = keras_model.transform(test_subgraph_loader)


def train_gcn_model_distributed_with_spark(
    model: Model,
    weight: torch.Tensor,
    train_data: DataLoader,
    val_data,
    test_data,
):
    import horovod.spark.torch as hvd

    unique_train_user_node_id = torch.tensor(
        list(set(train_data[("user", "rates", "movie")].edge_index[0, :].tolist()))
    )

    train_subgraph_loader = NeighborLoader(
        train_data,
        batch_size=4096,
        shuffle=False,
        num_neighbors={key: [-1] * 2 for key in train_data.edge_types},
        input_nodes=("user", unique_train_user_node_id),
    )
    unique_test_user_node_id = torch.tensor(
        list(set(train_data[("user", "rates", "movie")].edge_index[0, :].tolist()))
    )
    test_subgraph_loader = NeighborLoader(
        test_data,
        batch_size=4096,
        shuffle=False,
        num_neighbors={key: [-1] * 2 for key in test_data.edge_types},
        input_nodes=("user", unique_test_user_node_id),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    store = (None,)  # store = HDFSStore('/user/username/experiments')
    loss = weighted_mse_loss

    torch_estimator = hvd.TorchEstimator(
        num_proc=4,
        store=store,
        model=model,
        optimizer=optimizer,
        loss=loss,
        feature_cols=["features"],
        label_cols=["y"],
        batch_size=32,
        epochs=10,
    )

    keras_model = torch_estimator.fit(train_subgraph_loader)
    predict_df = keras_model.transform(test_subgraph_loader)


def train_gcn_model_distributed_with_ray(
    model: Model,
    weight: torch.Tensor,
    train_data: DataLoader,
    val_data,
    test_data,
):
    from raydp.torch import TorchEstimator

    unique_train_user_node_id = torch.tensor(
        list(set(train_data[("user", "rates", "movie")].edge_index[0, :].tolist()))
    )

    train_subgraph_loader = NeighborLoader(
        train_data,
        batch_size=4096,
        shuffle=False,
        num_neighbors={key: [-1] * 2 for key in train_data.edge_types},
        input_nodes=("user", unique_train_user_node_id),
    )
    unique_test_user_node_id = torch.tensor(
        list(set(train_data[("user", "rates", "movie")].edge_index[0, :].tolist()))
    )
    test_subgraph_loader = NeighborLoader(
        test_data,
        batch_size=4096,
        shuffle=False,
        num_neighbors={key: [-1] * 2 for key in test_data.edge_types},
        input_nodes=("user", unique_test_user_node_id),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    estimator = TorchEstimator(
        num_workers=2,
        model=model,
        optimizer=optimizer,
        loss=torch.nn.MSELoss(),  # weighted_mse_loss,
        metrics=["accuracy", "mse"],
        feature_columns=["x", "y"],
        label_column="z",
        batch_size=1000,
        num_epochs=2,
        use_gpu=False,
        config={"fit_config": {"steps_per_epoch": 2}},
    )
    estimator.fit_on_spark(train_subgraph_loader, test_subgraph_loader)
