"""
This is a boilerplate pipeline 'recommender_model'
generated using Kedro 0.18.4
"""
import torch
from torch_geometric.data import HeteroData
from .recommender_model import Model, train_model, test_model


def train_gcn_model(
    train_data: HeteroData,
    val_data,
    test_data,
):
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

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(1, 15):
        loss = train_model(model, optimizer, train_data, weight)
        train_rmse = test_model(model, train_data)
        val_rmse = test_model(model, val_data)
        test_rmse = test_model(model, test_data)
        print(
            f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_rmse:.4f}, "
            f"Val: {val_rmse:.4f}, Test: {test_rmse:.4f}"
        )

    return model
