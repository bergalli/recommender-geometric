# recommender-geometric

Recommender system using pytorch geomtric to learn and predict an heterogeneous graph representation.
Made for large networks:
    - Spark data management
    - distributed Deep graph network to predict new edges with labels (user rates a movie)
    - Sampling of the graph during epochs

Data source to unzip in `data/01_raw` : https://files.grouplens.org/datasets/movielens/ml-25m.zip

Run with `kedro run`