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
                outputs=["source_nodes", "dest_nodes", "weight"],
            ),
            node(
                define_movies_attributes,
                inputs=dict(
                    edges_dataframe="edges_dataframe",
                    genome_scores="genome_scores",
                ),
                outputs="pivoted_nodes_genome_scores",
            ),
        ],
        inputs={"ratings", "genome_scores"},
        outputs={"source_nodes", "dest_nodes", "weight", "pivoted_nodes_genome_scores"},
        namespace="data_prep",
    )
