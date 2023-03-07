"""
This is a boilerplate pipeline 'make_ratings_network'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    filter_low_viewers,
    create_dataframe_graph_edges,
    define_users_to_movies_edges,
    define_movies_attributes,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                filter_low_viewers,
                inputs=dict(
                    ratings="ratings",
                    n_users_to_keep="params:n_users_to_keep",
                ),
                outputs="ratings_filtered",
            ),
            node(
                create_dataframe_graph_edges,
                inputs=dict(
                    ratings="ratings_filtered",
                ),
                outputs=["edges_dataframe", "movie_id_to_node_id", "user_id_to_node_id"],
            ),
            node(
                define_users_to_movies_edges,
                inputs=dict(
                    edges_dataframe="edges_dataframe",
                ),
                outputs=["source_nodes", "dest_nodes", "edges_labels"],
            ),
            node(
                define_movies_attributes,
                inputs=dict(
                    movie_id_to_node_id="movie_id_to_node_id",
                    genome_scores="genome_scores",
                ),
                outputs="pivoted_genome_scores",
            ),
        ],
        inputs={"ratings", "genome_scores"},
        outputs={"source_nodes", "dest_nodes", "edges_labels", "pivoted_genome_scores"},
        # namespace="dataprep_ratings_nework",
    )
