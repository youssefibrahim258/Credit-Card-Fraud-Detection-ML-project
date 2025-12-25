"""
Main training pipeline for Credit Card Fraud Detection.
This script loads and preprocesses data, trains multiple ML models with
different preprocessing and sampling strategies, evaluates them on
train/validation sets, and logs metrics, parameters, figures, and models
to MLflow for experiment tracking and comparison.
"""

import argparse
import mlflow
import mlflow.sklearn

from src.data.load_data import load_data
from src.features.preprocessing import (apply_log_transform,choose_processor,get_sampler)

from src.models.model_registry import get_model_grids
from src.models.train import train_model
from src.models.evaluate import evaluate_model


def main(args):
    # Load & preprocess data
    df_train = apply_log_transform(load_data(args.train_path))
    df_val = apply_log_transform(load_data(args.val_path))

    x_train = df_train.iloc[:, :-1].values
    y_train = df_train.iloc[:, -1].values
    x_val = df_val.iloc[:, :-1].values
    y_val = df_val.iloc[:, -1].values

    # Preprocessing choices
    scaler = choose_processor(args.scaler)
    sampler = get_sampler(args.sampler, args.ratio)

    # MLflow setup 
    mlflow.set_experiment("Credit_Card_Fraud_Detection2")

    model_grids = get_model_grids()

    
    # Train each model
    for model_name in args.model_options:
        model_info = model_grids[model_name]

        with mlflow.start_run(run_name=model_name):

            grid = train_model(
                model=model_info["model"],
                param_grid=model_info["params"],
                scaler=scaler,
                sampler=sampler,
                x_train=x_train,
                y_train=y_train,
                sampler_option=args.sampler,
                ratio=args.ratio
            )

            best_model = grid.best_estimator_

            # Train Metrics 
            train_metrics, train_figs = evaluate_model(best_model,x_train,y_train,prefix="train")

            # Validation Metrics 
            val_metrics, val_figs = evaluate_model(best_model,x_val,y_val,prefix="val")

            # MLflow Logging
            # Metrics
            mlflow.log_metrics({**train_metrics,**val_metrics})

            # Params
            mlflow.log_params({"scaler": args.scaler,"sampler": args.sampler,"ratio": args.ratio})
            mlflow.log_params(grid.best_params_)

            # Figures
            for name, fig in train_figs.items():
                mlflow.log_figure(fig, name)

            for name, fig in val_figs.items():
                mlflow.log_figure(fig, name)

            # Model
            mlflow.sklearn.log_model(best_model, "model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Credit Card Fraud Detection with MLflow")
    parser.add_argument("--train_path",type=str,default="data_sets/train.csv")
    parser.add_argument("--val_path",type=str,default="data_sets/val.csv")
    parser.add_argument("--scaler",type=str,default="standardScaler",choices=["standardScaler", "MinMaxScaler"])
    parser.add_argument("--model_options",nargs="+",default=["LogisticRegression","RandomForest","MLP","VotingClassifier"])
    parser.add_argument("--sampler",type=int,default=1,help="1=Over, 2=Under, 3=Under+Over")
    parser.add_argument("--ratio",type=float,default=0.02)

    args = parser.parse_args()
    main(args)
