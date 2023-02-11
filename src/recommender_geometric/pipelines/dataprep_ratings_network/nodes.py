"""
This is a boilerplate pipeline 'make_ratings_network'
generated using Kedro 0.18.4
"""


from typing import Tuple

import numpy as np
import pandas
import pyspark.pandas as pd
import torch
from torch_geometric.data import HeteroData
import warnings

warnings.filterwarnings("ignore")


def create_dataframe_graph_edges(
    ratings: pandas.DataFrame,
) -> pd.DataFrame:
    """
    Creates new columns node_movie_id and node_user_id going to 0 up to the number or users or movies, to
    label the nodes in the graph.
    The output dataframe contains all edges of users ratings to movies
    :param movies:
    :param ratings:
    :return:
    """

    # pandas to modin conversion
    # movies = movies.to_pandas_on_spark()
    edges_dataframe = ratings.to_pandas_on_spark()

    # edges_dataframe = pd.merge(ratings, movies, on="movieId", how="inner")

    # assign nodes ids as a range to movieId and userId
    movie_id_to_node_id = {
        movie_id: node_id
        for node_id, movie_id in enumerate(edges_dataframe["movieId"].unique().values)
    }
    edges_dataframe["movie_node_id"] = edges_dataframe["movieId"].map(
        lambda x: movie_id_to_node_id[x]
    )
    user_id_to_node_id = {
        user_id: node_id
        for node_id, user_id in enumerate(edges_dataframe["userId"].unique().values)
    }
    edges_dataframe["user_node_id"] = edges_dataframe["userId"].map(lambda x: user_id_to_node_id[x])

    # convert ratings from a float [0,5] scale to an integer [0,100] scale
    edges_dataframe["rating"] = (edges_dataframe["rating"] / edges_dataframe["rating"].max()) * 100
    edges_dataframe["rating"] = edges_dataframe["rating"].astype(int)

    return edges_dataframe[["userId", "movieId", "user_node_id", "movie_node_id", "rating"]]


# def create_dataframe_users_nodes_attributes(
#     edges_dataframe: pd.DataFrame,
#     tags: pandas.DataFrame,
# ) -> pd.DataFrame:
#     tags = pd.DataFrame(tags)
#     users_attributes_dataframe = pd.merge(
#         edges_dataframe, tags, on=["userId", "movieId"], how="inner"
#     )
#     return users_attributes_dataframe


# def create_dataframe_movies_nodes_attributes(
#     edges_dataframe: pd.DataFrame,
#     movies: pandas.DataFrame,
#     genome_scores: pandas.DataFrame,
#     genome_tags: pandas.DataFrame,
# ) -> pd.DataFrame:
#     movies = pd.DataFrame(movies)
#     genome_scores = pd.DataFrame(genome_scores)
#     genome_tags = pd.DataFrame(genome_tags)
#
#     movies_attributes = pd.merge(genome_scores, genome_tags, on="tagId", how="inner")
#     movies_attributes = pd.merge(
#         movies_attributes, edges_dataframe[["movieId", "movie_node_id"]], on="movieId", how="inner"
#     )
#     movies_attributes = movies_attributes.rename(columns={"relevance": "tag_relevance"})
#     movies_attributes = pd.merge(movies_attributes, movies, on="movieId", how="inner")
#
#     return movies_attributes[
#         ["movie_node_id", "movieId", "tagId", "tag_relevance", "tag", "title", "genres"]
#     ]


def define_movies_attributes(
    edges_dataframe: pd.DataFrame,
    genome_scores: pd.DataFrame,
) -> pd.DataFrame:
    genome_scores = genome_scores.to_pandas_on_spark()
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
    # sort index so that rows are ordered according to movie node id # todo spark pivot auto sort ?
    pivoted_nodes_genome_scores = pivoted_nodes_genome_scores.sort_index()

    # fill missing genome relevances with 0, for movies without a defined genome # todo take forever
    # pivoted_nodes_genome_scores = pivoted_nodes_genome_scores.fillna(0)

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
