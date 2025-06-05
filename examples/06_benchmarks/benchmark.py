#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Benchmark script for recommender algorithms on MovieLens 100k dataset.
Supports algorithms: sar, svd, als, bpr, ncf, lightgcn
"""

from recommenders.datasets.python_splitters import python_stratified_split
from recommenders.utils.general_utils import get_number_processors
from recommenders.datasets import movielens
from benchmark_utils import *
import time
import cornac
import surprise
import pandas as pd
import numpy as np
import sys
import os
import logging
import warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)


try:
    import pyspark
except ImportError:
    pass  # skip this import if we are not in a Spark environment

try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')  # only show error messages
    import torch
except ImportError:
    pass  # skip this import if we are not in a GPU environment

# Add the benchmark_utils path
current_path = os.path.join(os.getcwd(), "examples", "06_benchmarks")
sys.path.append(current_path)

try:
    from recommenders.utils.spark_utils import start_or_get_spark
except ImportError:
    pass  # skip this import if we are not in a Spark environment
try:
    from recommenders.utils.gpu_utils import get_cuda_version, get_cudnn_version
except ImportError:
    pass  # skip this import if we are not in a GPU environment

# Print system information
print(f"System version: {sys.version}")
print(f"Number of cores: {get_number_processors()}")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"Surprise version: {surprise.__version__}")
print(f"Cornac version: {cornac.__version__}")
try:
    print(f"PySpark version: {pyspark.__version__}")
except NameError:
    pass
try:
    print(f"CUDA version: {get_cuda_version()}")
    print(f"CuDNN version: {get_cudnn_version()}")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"PyTorch version: {torch.__version__}")
except NameError:
    pass

# Initialize Spark if available
try:
    spark = start_or_get_spark("PySpark", memory="32g")
    spark.conf.set("spark.sql.analyzer.failAmbiguousSelfJoin", "false")
except NameError:
    pass  # skip this if we are not in a Spark environment

# Fix random seeds for reproducibility
np.random.seed(SEED)
try:
    tf.random.set_seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
except NameError:
    pass  # skip this if we are not in a GPU environment

# ========================================
# CONFIGURATION SECTION - EDIT HERE
# ========================================

# Choose which algorithms to test by commenting/uncommenting lines
# Available algorithms: "als", "svd", "sar", "ncf", "bpr", "lightgcn"
algorithms = [
    # "als",
    # "svd",
    # "sar",
    "ncf",
    "bpr",
    "lightgcn",
]

# ========================================
# END CONFIGURATION SECTION
# ========================================

# Fixed to use only 100k dataset
data_sizes = ["1m"]

# Environment mapping
environments = {
    "als": "pyspark",
    "sar": "python_cpu",
    "svd": "python_cpu",
    "ncf": "python_gpu",
    "bpr": "python_cpu",
    "lightgcn": "python_gpu",
}

# Metrics mapping
metrics = {
    "als": ["rating", "ranking"],
    "sar": ["ranking"],
    "svd": ["rating", "ranking"],
    "ncf": ["ranking"],
    "bpr": ["ranking"],
    "lightgcn": ["ranking"]
}

# Algorithm parameters
als_params = {
    "rank": 10,
    "maxIter": 20,
    "implicitPrefs": False,
    "alpha": 0.1,
    "regParam": 0.05,
    "coldStartStrategy": "drop",
    "nonnegative": False,
    "userCol": DEFAULT_USER_COL,
    "itemCol": DEFAULT_ITEM_COL,
    "ratingCol": DEFAULT_RATING_COL,
}

sar_params = {
    "similarity_type": "jaccard",
    "time_decay_coefficient": 30,
    "time_now": None,
    "timedecay_formula": True,
    "col_user": DEFAULT_USER_COL,
    "col_item": DEFAULT_ITEM_COL,
    "col_rating": DEFAULT_RATING_COL,
    "col_timestamp": DEFAULT_TIMESTAMP_COL,
}

svd_params = {
    "n_factors": 150,
    "n_epochs": 30,
    "lr_all": 0.005,
    "reg_all": 0.02,
    "random_state": SEED,
    "verbose": False
}

ncf_params = {
    "model_type": "NeuMF",
    "n_factors": 4,
    "layer_sizes": [16, 8, 4],
    "n_epochs": 20,
    "batch_size": 256,
    "learning_rate": 1e-3,
    "verbose": 10
}

bpr_params = {
    "k": 200,
    "max_iter": 200,
    "learning_rate": 0.01,
    "lambda_reg": 1e-3,
    "seed": SEED,
    "verbose": False
}

lightgcn_param = {
    "model_type": "lightgcn",
    "n_layers": 3,
    "batch_size": 256,
    "embed_size": 64,
    "decay": 0.0001,
    "epochs": 20,
    "learning_rate": 0.005,
    "eval_epoch": 5,
    "top_k": DEFAULT_K,
    "metrics": ["recall", "ndcg", "precision", "map"],
    "save_model": False,
    "MODEL_DIR": ".",
}

params = {
    "als": als_params,
    "sar": sar_params,
    "svd": svd_params,
    "ncf": ncf_params,
    "bpr": bpr_params,
    "lightgcn": lightgcn_param,
}

# Function mappings
prepare_training_data = {
    "als": prepare_training_als,
    "sar": prepare_training_sar,
    "svd": prepare_training_svd,
    "ncf": prepare_training_ncf,
    "bpr": prepare_training_cornac,
    "lightgcn": prepare_training_lightgcn,
}

prepare_metrics_data = {
    "als": lambda train, test: prepare_metrics_als(train, test),
}

trainer = {
    "als": lambda params, data: train_als(params, data),
    "svd": lambda params, data: train_svd(params, data),
    "sar": lambda params, data: train_sar(params, data),
    "ncf": lambda params, data: train_ncf(params, data),
    "bpr": lambda params, data: train_bpr(params, data),
    "lightgcn": lambda params, data: train_lightgcn(params, data),
}

rating_predictor = {
    "als": lambda model, test: predict_als(model, test),
    "svd": lambda model, test: predict_svd(model, test),
}

ranking_predictor = {
    "als": lambda model, test, train: recommend_k_als(model, test, train),
    "sar": lambda model, test, train: recommend_k_sar(model, test, train),
    "svd": lambda model, test, train: recommend_k_svd(model, test, train),
    "ncf": lambda model, test, train: recommend_k_ncf(model, test, train),
    "bpr": lambda model, test, train: recommend_k_cornac(model, test, train),
    "lightgcn": lambda model, test, train: recommend_k_lightgcn(model, test, train),
}

rating_evaluator = {
    "als": lambda test, predictions: rating_metrics_pyspark(test, predictions),
    "svd": lambda test, predictions: rating_metrics_python(test, predictions),
}

ranking_evaluator = {
    "als": lambda test, predictions, k: ranking_metrics_pyspark(test, predictions, k),
    "sar": lambda test, predictions, k: ranking_metrics_python(test, predictions, k),
    "svd": lambda test, predictions, k: ranking_metrics_python(test, predictions, k),
    "ncf": lambda test, predictions, k: ranking_metrics_python(test, predictions, k),
    "bpr": lambda test, predictions, k: ranking_metrics_python(test, predictions, k),
    "lightgcn": lambda test, predictions, k: ranking_metrics_python(test, predictions, k),
}


def generate_summary(data, algo, k, train_time, time_rating, rating_metrics, time_ranking, ranking_metrics):
    summary = {"Data": data, "Algo": algo, "K": k, "Train time (s)": train_time,
               "Predicting time (s)": time_rating, "Recommending time (s)": time_ranking}
    if rating_metrics is None:
        rating_metrics = {
            "RMSE": np.nan,
            "MAE": np.nan,
            "R2": np.nan,
            "Explained Variance": np.nan,
        }
    if ranking_metrics is None:
        ranking_metrics = {
            "MAP": np.nan,
            "nDCG@k": np.nan,
            "Precision@k": np.nan,
            "Recall@k": np.nan,
        }
    summary.update(rating_metrics)
    summary.update(ranking_metrics)
    return summary


def main():
    """Main benchmark function"""

    # Start timing
    start_time = time.time()
    run_timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

    # Initialize results dataframe
    cols = ["Data", "Algo", "K", "Train time (s)", "Predicting time (s)",
            "RMSE", "MAE", "R2", "Explained Variance", "Recommending time (s)",
            "MAP", "nDCG@k", "Precision@k", "Recall@k", "Run Timestamp"]
    df_results = pd.DataFrame(columns=cols)

    for data_size in data_sizes:
        # Load the dataset
        df = movielens.load_pandas_df(
            size=data_size,
            header=[DEFAULT_USER_COL, DEFAULT_ITEM_COL,
                    DEFAULT_RATING_COL, DEFAULT_TIMESTAMP_COL]
        )
        print(f"Size of Movielens {data_size}: {df.shape}")

        # Split the dataset
        df_train, df_test = python_stratified_split(
            df,
            ratio=0.75,
            min_rating=1,
            filter_by="item",
            col_user=DEFAULT_USER_COL,
            col_item=DEFAULT_ITEM_COL
        )

        # Loop through the algorithms
        for algo in algorithms:
            print(f"\nComputing {algo} algorithm on Movielens {data_size}")

            # Data prep for training set
            train = prepare_training_data.get(
                algo, lambda x, y: (x, y))(df_train, df_test)

            # Get model parameters
            model_params = params[algo]

            # Train the model
            model, time_train = trainer[algo](model_params, train)
            print(f"Training time: {time_train}s")

            # Predict and evaluate
            train, test = prepare_metrics_data.get(
                algo, lambda x, y: (x, y))(df_train, df_test)

            if "rating" in metrics[algo]:
                # Predict for rating
                preds, time_rating = rating_predictor[algo](model, test)
                print(f"Rating prediction time: {time_rating}s")

                # Evaluate for rating
                ratings = rating_evaluator[algo](test, preds)
            else:
                ratings = None
                time_rating = np.nan

            if "ranking" in metrics[algo]:
                # Predict for ranking
                top_k_scores, time_ranking = ranking_predictor[algo](
                    model, test, train)
                print(f"Ranking prediction time: {time_ranking}s")

                # Evaluate for ranking
                rankings = ranking_evaluator[algo](
                    test, top_k_scores, DEFAULT_K)
            else:
                rankings = None
                time_ranking = np.nan

            # Record results
            summary = generate_summary(data_size, algo, DEFAULT_K, time_train,
                                       time_rating, ratings, time_ranking, rankings)
            summary["Run Timestamp"] = run_timestamp
            df_results.loc[df_results.shape[0] + 1] = summary

            # Save results for this algorithm and dataset
            RESULT_DIR = os.path.join(os.path.dirname(__file__), "results")
            os.makedirs(RESULT_DIR, exist_ok=True)
            algo_file = os.path.join(RESULT_DIR, f"{algo}_{data_size}.csv")
            # Append to file if exists, else write header
            df_to_save = pd.DataFrame([summary])
            write_header = not os.path.exists(algo_file)
            df_to_save.to_csv(algo_file, mode='a',
                              header=write_header, index=False)

    print("\nComputation finished")

    # Display results
    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)
    print(df_results.to_string())

    # Total time
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")


if __name__ == "__main__":
    main()
