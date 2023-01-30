"""
This is a boilerplate pipeline 'data_prep'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    create_dataframe_graph_edges,
    create_dataframe_users_nodes_attributes,
    create_dataframe_movies_nodes_attributes,
    create_edges_tensors,
    create_nodes_tensors,
    create_pyg_network,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                create_dataframe_graph_edges,
                inputs=dict(
                    movies="movies",
                    ratings="ratings",
                ),
                outputs="edges_dataframe",
            ),
            node(
                create_dataframe_users_nodes_attributes,
                inputs=dict(
                    edges_dataframe="edges_dataframe",
                    tags="tags",
                ),
                outputs="users_attributes_dataframe",
            ),
            node(
                create_dataframe_movies_nodes_attributes,
                inputs=dict(
                    edges_dataframe="edges_dataframe",
                    movies="movies",
                    genome_scores="genome_scores",
                    genome_tags="genome_tags",
                ),
                outputs="movies_attributes_dataframe",
            ),
            node(
                create_edges_tensors,
                inputs=dict(
                    edges_dataframe="edges_dataframe",
                ),
                outputs=["edges_index", "edges_attr"],
            ),
            node(
                create_nodes_tensors,
                inputs=dict(
                    users_attributes_dataframe="users_attributes_dataframe",
                    movies_attributes_dataframe="movies_attributes_dataframe",
                ),
                outputs=["movies_nodes_attr", "users_nodes_attr"],
            ),
            node(
                create_pyg_network,
                inputs=dict(
                    edges_index="edges_index",
                    edges_attr="edges_attr",
                    movies_nodes_attr="movies_nodes_attr",
                    users_nodes_attr="users_nodes_attr",
                ),
                outputs="users_rating_movies_network"
            ),
        ],
    )
