"""
This is a boilerplate pipeline 'make_ratings_network'
generated using Kedro 0.18.4
"""

from typing import Tuple

from pyspark.context import SparkContext
from pyspark.sql import DataFrame, Row, Window
from pyspark.sql.functions import col, create_map, lit, broadcast, last
import pyspark.sql.functions as F
from pyspark.sql.types import IntegerType
import numpy as np
from itertools import chain
from pyspark.sql.window import Window



def filter_low_viewers(ratings: DataFrame, n_users_to_keep: int):
    """
    filter the initial dataset to limit usage of memory when training the dataset
    :param ratings:
    :param n_users_to_keep:
    :return:
    """
    if n_users_to_keep is not None:
        # drop duplicated user/movie ids as a safety, but raw dataset already not duplicated
        # todo go in tests
        unique_users_movies = ratings.select(["userId", "movieId"]).dropDuplicates()

        # for each movie, compute the number of times it has been watched/rated
        movie_watched_n_times = (
            unique_users_movies.groupBy("movieId")
            .count()
            .withColumnRenamed("count", "watched_n_times")
        )
        # for each user, get the average of ratings made on movies he watched
        user_watched_famous_movies = (
            unique_users_movies.join(movie_watched_n_times, on="movieId")
            .groupby("userId")
            .mean("watched_n_times")
        )

        # for each user, count the number of movies watched/rated
        user_watch_n_movies = (
            unique_users_movies.groupBy("userId")
            .count()
            .withColumnRenamed("count", "n_movies_watched")
        )

        # sort users according to most number of movies watched/rated
        #  so that those with most movies watched are on top
        # also sort according to the most famous movies watched, so that
        #  when filtering users, we reduce the chance to have isolated user nodes.
        users_sorted_by_watch_habits = (
            ratings.select("userId")
            .distinct()
            .join(user_watch_n_movies, on="userId")
            .join(user_watched_famous_movies, on="userId")
            .sort(["n_movies_watched", "avg(watched_n_times)"], ascending=[False, False])
        )
        w = Window.orderBy("userId")
        users_sorted_by_watch_habits = users_sorted_by_watch_habits.withColumn("index", F.row_number().over(w))
        users_to_keep = users_sorted_by_watch_habits.filter(f"index <= {n_users_to_keep}")

        # apply user filter on ratings dataframe
        ratings = ratings.join(users_to_keep.select("userId"), on="userId", how="right")

    return ratings


def create_dataframe_graph_edges(
    ratings: DataFrame,
) -> Tuple[DataFrame, dict, dict]:
    """
    Creates new columns node_movie_id and node_user_id going to 0 up to the number or users or movies, to
    label the nodes in the graph.
    The output dataframe contains all edges of users ratings to movies
    :param movies:
    :param ratings:
    :return:
    """

    # assign nodes ids as a range to movieId
    unique_movie_ids = ratings.select("movieId").distinct().rdd.map(lambda r: r[0]).collect()
    movie_id_to_node_id = {movie_id: node_id for node_id, movie_id in enumerate(unique_movie_ids)}
    mapping_expr = create_map([lit(x) for x in chain(*movie_id_to_node_id.items())])
    ratings = ratings.withColumn("movie_node_id", mapping_expr[col("movieId")])

    # assign nodes ids as a range to userId
    unique_user_ids = ratings.select("userId").distinct().rdd.map(lambda r: r[0]).collect()
    user_id_to_node_id = {user_id: node_id for node_id, user_id in enumerate(unique_user_ids)}
    mapping_expr = create_map([lit(x) for x in chain(*user_id_to_node_id.items())])
    ratings = ratings.withColumn("user_node_id", mapping_expr[col("userId")])

    # convert ratings from a float [0,5] scale to an integer [0,100] scale
    ratings = (
        ratings.crossJoin(ratings.select(F.max("rating").alias("max_rating")))
        .withColumn("rating", (col("rating") / col("max_rating")) * 100)
        .drop("max_rating")
        .withColumn("rating", col("rating").cast(IntegerType()))
    )

    edges_dataframe = ratings
    return edges_dataframe, movie_id_to_node_id, user_id_to_node_id


def define_movies_attributes(
    movie_id_to_node_id,
    genome_scores: DataFrame,
) -> DataFrame:

    movie_id_to_node_id_mapper = create_map([lit(x) for x in chain(*movie_id_to_node_id.items())])
    genome_scores = genome_scores.withColumn(
        "movie_node_id", movie_id_to_node_id_mapper[col("movieId")]
    )

    # fill in missing values for movies without a genome
    sc = SparkContext.getOrCreate()
    all_movie_node_id = (
        sc.range(min(movie_id_to_node_id.values()), max(movie_id_to_node_id.values()))
        .map(Row("movie_node_id"))
        .toDF()
    )
    # cross join all movie node ids with the unique genome tag values
    all_movie_node_id = all_movie_node_id.crossJoin(
        genome_scores.select("tagId").distinct()
    ).cache()
    # outer join of the initial dataframe with all movie node ids matched with tags
    # fill relavance missing values with 0
    genome_scores = genome_scores.join(
        broadcast(all_movie_node_id), on=["movie_node_id", "tagId"], how="outer"
    ).withColumn(
        "relevance",
        last("relevance", ignorenulls=True).over(
            Window.partitionBy("tagId")
            .orderBy("movie_node_id")
            .rowsBetween(Window.unboundedPreceding, 0)
        ),
    )

    pivoted_genome_scores = genome_scores.groupBy("movie_node_id").pivot("tagId").max("relevance")
    pivoted_genome_scores = pivoted_genome_scores.orderBy(col("movie_node_id"))
    pivoted_genome_scores = pivoted_genome_scores.drop("movie_node_id")

    return pivoted_genome_scores


def define_users_to_movies_edges(
    edges_dataframe: DataFrame,
) -> Tuple[DataFrame, DataFrame, DataFrame]:
    """
    Keep single column dataframes because kedro not handling pandas spark series
    """
    # source nodes
    src = edges_dataframe.select(col("user_node_id"))
    # destination nodes
    dst = edges_dataframe.select(col("movie_node_id"))
    # weight of each edge
    weight = edges_dataframe.select(col("rating"))

    return src, dst, weight
