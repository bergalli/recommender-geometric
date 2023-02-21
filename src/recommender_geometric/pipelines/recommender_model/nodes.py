"""
This is a boilerplate pipeline 'recommender_model'
generated using Kedro 0.18.4
"""
import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader, DataLoader
from torch_geometric.transforms import RandomLinkSplit, ToUndirected
import torch.multiprocessing as mp
from .model import Model, train_model, test_model, weighted_mse_loss

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os

# import horovod.spark.torch as hvd
# hvd.init()
# ddaata = torch.utils.data.distributed.DistributedSampler(
#     source_nodes, num_replicas=hvd.size(), rank=hvd.rank()
# )

def make_graph_tensors(
    source_nodes,
    dest_nodes,
    weight,
    pivoted_nodes_genome_scores,
):

    edge_index = torch.tensor([source_nodes.values, dest_nodes.values])
    edge_label = torch.tensor(weight)

    movies_nodes_attr = pivoted_nodes_genome_scores.values
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
        rev_edge_types=[("movie", "rev_rates", "user")],  # for heteroData, prevents data leakage
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
    """
    Vanilla training
    """
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
