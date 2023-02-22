"""
This is a boilerplate pipeline 'make_ratings_network'
generated using Kedro 0.18.4
"""

from typing import Tuple

import pyspark.pandas as pd


def create_dataframe_graph_edges(
    ratings: pd.DataFrame,
) -> pd.DataFrame:
    """
    Creates new columns node_movie_id and node_user_id going to 0 up to the number or users or movies, to
    label the nodes in the graph.
    The output dataframe contains all edges of users ratings to movies
    :param movies:
    :param ratings:
    :return:
    """

    # assign nodes ids as a range to movieId and userId
    unique_movie_ids = ratings["movieId"].unique().values
    movie_id_to_node_id = {movie_id: node_id for node_id, movie_id in enumerate(unique_movie_ids)}
    ratings["movie_node_id"] = ratings["movieId"].map(lambda x: movie_id_to_node_id[x])

    unique_user_ids = ratings["userId"].unique().values
    user_id_to_node_id = {user_id: node_id for node_id, user_id in enumerate(unique_user_ids)}
    ratings["user_node_id"] = ratings["userId"].map(lambda x: user_id_to_node_id[x])

    # convert ratings from a float [0,5] scale to an integer [0,100] scale
    ratings["rating"] = (ratings["rating"] / ratings["rating"].max()) * 100
    ratings["rating"] = ratings["rating"].astype(int)

    edges_dataframe = ratings
    return edges_dataframe


def define_movies_attributes(
    edges_dataframe: pd.DataFrame,
    genome_scores: pd.DataFrame,
) -> pd.DataFrame:

    # merge node ids to movies genome scores
    # right join to keep all nodes in the output dataframe
    movie_id_to_node_id = edges_dataframe[["movieId", "movie_node_id"]].drop_duplicates()
    nodes_genome_scores = pd.merge(
        genome_scores,
        movie_id_to_node_id,
        on="movieId",
        how="right",
    )
    nodes_genome_scores["movie_node_id"] = nodes_genome_scores["movie_node_id"].astype(str)
    nodes_genome_scores["tagId"] = nodes_genome_scores["tagId"].astype(str)

    pivoted_nodes_genome_scores = nodes_genome_scores.pivot(
        index="movie_node_id", columns="tagId", values="relevance"
    )
    # sort index so that rows are ordered according to movie node id
    pivoted_nodes_genome_scores = pivoted_nodes_genome_scores.set_index(
        keys=pivoted_nodes_genome_scores.index.astype(int)
    )
    pivoted_nodes_genome_scores = pivoted_nodes_genome_scores.sort_index()

    # fill missing genome relevances with 0, for movies without a defined genome # todo take forever
    pivoted_nodes_genome_scores = pivoted_nodes_genome_scores.fillna(0)

    return pivoted_nodes_genome_scores


def define_users_to_movies_edges(
    edges_dataframe: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Keep single column dataframes because kedro not handling pandas spark series
    """
    # source nodes
    src = edges_dataframe.loc[:, ["user_node_id"]]
    # destination nodes
    dst = edges_dataframe.loc[:, ["movie_node_id"]]
    # weight of each edge
    weight = edges_dataframe.loc[:, ["rating"]]

    return src, dst, weight

