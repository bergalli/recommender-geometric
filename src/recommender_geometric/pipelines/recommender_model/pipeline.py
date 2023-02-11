"""
This is a boilerplate pipeline 'recommender_model'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    make_graph_tensors,
    create_pyg_network,
    make_graph_undirected,
    make_ttv_data,
    instanciate_model,
    get_sampler_dataloader,
    train_gcn_model,
    train_gcn_model_distributed,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                make_graph_tensors,
                inputs=dict(
                    source_nodes="source_nodes",
                    dest_nodes="dest_nodes",
                    weight="weight",
                    pivoted_nodes_genome_scores="pivoted_nodes_genome_scores",
                ),
                outputs=["edge_index", "edge_label", "movies_nodes_attr"],
            ),
            node(
                create_pyg_network,
                inputs=dict(
                    edge_index="edge_index",
                    edge_label="edge_label",
                    movies_nodes_attr="movies_nodes_attr",
                ),
                outputs="users_rating_movies_network",
            ),
            node(
                make_graph_undirected,
                inputs=dict(
                    data="users_rating_movies_network",
                ),
                outputs="undirected_graph",
                name="pipe_start",
            ),
            node(
                make_ttv_data,
                inputs=dict(
                    data="undirected_graph",
                ),
                outputs=["train_data", "val_data", "test_data"],
            ),
            node(
                instanciate_model,
                inputs=dict(
                    train_data="train_data",
                ),
                outputs=["model", "weight"],
            ),
            node(
                get_sampler_dataloader,
                inputs=dict(train_data="train_data"),
                outputs="subgraph_sampler",
            ),
            node(
                train_gcn_model_distributed,
                inputs=dict(
                    model="model",
                    weight="weight",
                    train_dataloader="subgraph_sampler",
                    val_data="val_data",
                    test_data="test_data",
                ),
                outputs="trained_model",
            ),
        ],
        inputs={"source_nodes", "dest_nodes", "weight", "pivoted_nodes_genome_scores"},
        namespace="recommender_model",
    )
