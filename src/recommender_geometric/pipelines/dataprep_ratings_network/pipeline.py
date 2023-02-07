"""
This is a boilerplate pipeline 'make_ratings_network'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    create_dataframe_graph_edges,
    # create_dataframe_users_nodes_attributes,
    # create_dataframe_movies_nodes_attributes,
    define_users_to_movies_edges,
    define_movies_attributes,
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
            # node(
            #     create_dataframe_users_nodes_attributes,
            #     inputs=dict(
            #         edges_dataframe="edges_dataframe",
            #         tags="tags",
            #     ),
            #     outputs="users_attributes_dataframe",
            # ),
            # node(
            #     create_dataframe_movies_nodes_attributes,
            #     inputs=dict(
            #         edges_dataframe="edges_dataframe",
            #         movies="movies",
            #         genome_scores="genome_scores",
            #         genome_tags="genome_tags",
            #     ),
            #     outputs="movies_attributes_dataframe",
            # ),
            node(
                define_users_to_movies_edges,
                inputs=dict(
                    edges_dataframe="edges_dataframe",
                ),
                outputs=["edge_index", "edge_label"],
            ),
            node(
                define_movies_attributes,
                inputs=dict(
                    edges_dataframe="edges_dataframe",
                    genome_scores="genome_scores",
                ),
                outputs="movies_nodes_attr",
            ),
            node(
                create_pyg_network,
                inputs=dict(
                    edge_index="edge_index",
                    edge_label="edge_label",
                    movies_nodes_attr="movies_nodes_attr",
                    # users_nodes_attr="users_nodes_attr",
                ),
                outputs="users_rating_movies_network",
            ),
        ],
        inputs={"movies", "ratings", "genome_scores"},
        outputs="users_rating_movies_network",
        namespace="data_prep",
    )
