# recommender-geometric

Recommender system using pytorch geomtric to learn and predict an heterogeneous graph representation.
Made for large networks:
    - Spark data management
    - distributed Deep graph network to predict new edges with labels (user rates a movie)
    - Sampling of the graph during epochs

Data source to unzip in `data/01_raw` : https://files.grouplens.org/datasets/movielens/ml-25m.zip
Use `poetry install` to install project's dependancies.

## Preprocessing + training local or distributed

### Run in local mode
Run with `kedro run --params:"backend=local"`

Local mode doesn't load all rows of the dataframe to improve speed and out-of-memory error.
This parameter can be changed in `conf/base/catalog/input.yaml` under `local.ratings.load_args.nrows`

### Run in spark mode
Run with `kedro run --params:"backend=spark"`


## Skip preprocessing

Run with `kedro run --from_nodes training_start`

Add `--params:"backend=BACKEND"` independantly of how the backend during the previous preprocessing run.   
