"""
This is a boilerplate pipeline 'data_prep'
generated using Kedro 0.18.4
"""

import pandas
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from typing import Tuple
import numpy as np


def create_dataframe_graph_edges(
    movies: pandas.DataFrame,
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
    movies = pd.DataFrame(movies)
    ratings = pd.DataFrame(ratings)

    movie_id_to_node_id = {
        movie_id: node_id for node_id, movie_id in enumerate(movies["movieId"].unique())
    }
    movies["movie_node_id"] = movies["movieId"].map(lambda x: movie_id_to_node_id[x])

    user_id_to_node_id = {
        user_id: node_id for node_id, user_id in enumerate(ratings["userId"].unique())
    }
    ratings["user_node_id"] = ratings["userId"].map(lambda x: user_id_to_node_id[x])

    edges_dataframe = pd.merge(ratings, movies, on="movieId", how="inner")

    return edges_dataframe[
        ["userId", "movieId", "rating", "title", "user_node_id", "movie_node_id"]
    ]


# def create_dataframe_users_nodes_attributes(
#     edges_dataframe: pd.DataFrame,
#     tags: pandas.DataFrame,
# ) -> pd.DataFrame:
#     tags = pd.DataFrame(tags)
#     users_attributes_dataframe = pd.merge(
#         edges_dataframe, tags, on=["userId", "movieId"], how="inner"
#     )
#     return users_attributes_dataframe


def create_dataframe_movies_nodes_attributes(
    edges_dataframe: pd.DataFrame,
    movies: pandas.DataFrame,
    genome_scores: pandas.DataFrame,
    genome_tags: pandas.DataFrame,
) -> pd.DataFrame:
    movies = pd.DataFrame(movies)
    genome_scores = pd.DataFrame(genome_scores)
    genome_tags = pd.DataFrame(genome_tags)

    movies_attributes = pd.merge(genome_scores, genome_tags, on="tagId", how="inner")
    movies_attributes = pd.merge(
        movies_attributes, edges_dataframe[["movieId", "movie_node_id"]], on="movieId", how="inner"
    )
    movies_attributes = movies_attributes.rename(columns={"relevance": "tag_relevance"})
    movies_attributes = pd.merge(movies_attributes, movies, on="movieId", how="inner")

    return movies_attributes[
        ["movie_node_id", "movieId", "tagId", "tag_relevance", "tag", "title", "genres"]
    ]


def define_users_to_movies_edges(
    edges_dataframe: pd.DataFrame,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # source nodes
    src = edges_dataframe["user_node_id"].values.tolist()
    # destination nodes
    dst = edges_dataframe["movie_node_id"].values.tolist()
    # weight of each edge
    weight = edges_dataframe["rating"].values.tolist()

    edges_index = torch.tensor([src, dst], dtype=torch.long)
    edges_attr = torch.tensor(weight, dtype=torch.float)

    return edges_index, edges_attr


def define_movies_attributes(
    edges_dataframe: pd.DataFrame,
    genome_scores: pd.DataFrame,
) -> torch.Tensor:

    nodes_genome_scores = pd.merge(
        genome_scores,
        edges_dataframe[["movieId", "movie_node_id"]].drop_duplicates(),
        on="movieId",
        how="inner",
    )
    # memory efficient pivot
    n_movies_per_chunk, unique_movies = 100, nodes_genome_scores["movie_node_id"].unique()
    pivoted_chunks = []
    for movies_chunk in np.array_split(unique_movies, n_movies_per_chunk):
        chunk = nodes_genome_scores.loc[nodes_genome_scores["movie_node_id"].isin(movies_chunk), :]
        pivoted = chunk.pivot(index="movie_node_id", columns="tagId", values="relevance")
        pivoted_chunks.append(pivoted)
    pivoted_nodes_genome_scores = pd.concat(pivoted_chunks, axis=0)
    pivoted_nodes_genome_scores = pivoted_nodes_genome_scores.sort_index()

    movies_nodes_attr = pivoted_nodes_genome_scores.values.tolist()
    movies_nodes_attr = torch.tensor(movies_nodes_attr, dtype=torch.float)

    return movies_nodes_attr


def create_pyg_network(
    edges_index: torch.Tensor,
    edges_attr: torch.Tensor,
    movies_nodes_attr: torch.Tensor,
    # users_nodes_attr: torch.Tensor,
) -> HeteroData:
    user_movies_network = HeteroData()

    user_movies_network["movie"].x = movies_nodes_attr

    user_movies_network["user", "rates", "movie"].edges_index = edges_index
    user_movies_network["user", "rates", "movie"].edges_attr = edges_attr

    return user_movies_network
