"""
This is a boilerplate pipeline 'make_ratings_network'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    create_dataframe_graph_edges,
    define_users_to_movies_edges,
    define_movies_attributes,
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
        # namespace="dataprep_ratings_nework",
    )
