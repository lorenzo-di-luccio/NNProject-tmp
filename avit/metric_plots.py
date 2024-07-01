import plotly
import plotly.express
import plotly.graph_objects
import polars
from typing import *

# Various utility functions for plotting.
# Not important for understanding the main part of the project.


def compute_agg_metrics(metric_files: List[str], last_epoch: int) -> polars.DataFrame:
    df = polars.concat(
        map(
            lambda metric_file:
            polars.scan_csv(metric_file, has_header=True, separator=",")
            .cast({
                "epoch": polars.UInt32,
                "step": polars.UInt64,
                "train_loss_step": polars.Float32,
                "train_loss_epoch": polars.Float32,
                "train_distr_loss_step": polars.Float32,
                "train_distr_loss_epoch": polars.Float32,
                "train_ponder_loss_step": polars.Float32,
                "train_ponder_loss_epoch": polars.Float32,
                "train_task_loss_step": polars.Float32,
                "train_task_loss_epoch": polars.Float32,
                "train_accuracy_step": polars.Float32,
                "train_accuracy_epoch": polars.Float32,
                "train_f1_score_step": polars.Float32,
                "train_f1_score_step": polars.Float32,
                "val_loss_step": polars.Float32,
                "val_loss_epoch": polars.Float32,
                "val_distr_loss_step": polars.Float32,
                "val_distr_loss_epoch": polars.Float32,
                "val_ponder_loss_step": polars.Float32,
                "val_ponder_loss_epoch": polars.Float32,
                "val_task_loss_step": polars.Float32,
                "val_task_loss_epoch": polars.Float32,
                "val_accuracy_step": polars.Float32,
                "val_accuracy_epoch": polars.Float32,
                "val_f1_score_step": polars.Float32,
                "val_f1_score_step": polars.Float32
            }),
            metric_files
        ), how="vertical"
    )

    df_train = df.select(
        polars.col("epoch"),
        polars.col("train_loss_epoch"),
        polars.col("train_distr_loss_epoch"),
        polars.col("train_ponder_loss_epoch"),
        polars.col("train_task_loss_epoch"),
        polars.col("train_accuracy_epoch"),
        polars.col("train_f1_score_epoch")
    )
    df_train = df_train.cast({
        "epoch": polars.UInt32,
        "train_loss_epoch": polars.Float32,
        "train_distr_loss_epoch": polars.Float32,
        "train_ponder_loss_epoch": polars.Float32,
        "train_task_loss_epoch": polars.Float32,
        "train_accuracy_epoch": polars.Float32,
        "train_f1_score_epoch": polars.Float32
    })
    df_train = df_train.filter(polars.col("epoch") <= last_epoch)
    df_train = df_train.drop_nulls(subset="epoch")
    df_train = df_train.drop_nulls(subset="train_loss_epoch")
    df_train = df_train.drop_nulls(subset="train_distr_loss_epoch")
    df_train = df_train.drop_nulls(subset="train_ponder_loss_epoch")
    df_train = df_train.drop_nulls(subset="train_task_loss_epoch")
    df_train = df_train.drop_nulls(subset="train_accuracy_epoch")
    df_train = df_train.drop_nulls(subset="train_f1_score_epoch")

    df_validation = df.select(
        polars.col("epoch"),
        polars.col("val_loss_epoch"),
        polars.col("val_distr_loss_epoch"),
        polars.col("val_ponder_loss_epoch"),
        polars.col("val_task_loss_epoch"),
        polars.col("val_accuracy_epoch"),
        polars.col("val_f1_score_epoch")
    )
    df_validation = df_validation.cast({
        "epoch": polars.UInt32,
        "val_loss_epoch": polars.Float32,
        "val_distr_loss_epoch": polars.Float32,
        "val_ponder_loss_epoch": polars.Float32,
        "val_task_loss_epoch": polars.Float32,
        "val_accuracy_epoch": polars.Float32,
        "val_f1_score_epoch": polars.Float32
    })
    df_validation = df_validation.filter(polars.col("epoch") <= last_epoch)
    df_validation = df_validation.drop_nulls(subset="epoch")
    df_validation = df_validation.drop_nulls(subset="val_loss_epoch")
    df_validation = df_validation.drop_nulls(subset="val_distr_loss_epoch")
    df_validation = df_validation.drop_nulls(subset="val_ponder_loss_epoch")
    df_validation = df_validation.drop_nulls(subset="val_task_loss_epoch")
    df_validation = df_validation.drop_nulls(subset="val_accuracy_epoch")
    df_validation = df_validation.drop_nulls(subset="val_f1_score_epoch")

    df = df_train.join(df_validation, on="epoch", how="inner", validate="1:1")
    return df.collect()


def load_metrics(metrics_file: str) -> polars.DataFrame:
    return polars.scan_parquet(metrics_file).collect()


def save_metrics(metrics_df: polars.DataFrame, metrics_file: str) -> None:
    metrics_df.write_parquet(metrics_file)


def metrics_for_plot(
        metrics_file: str,
        stage: str
) -> Tuple[polars.DataFrame, polars.DataFrame]:
    df = polars.scan_parquet(metrics_file)

    df_loss = df.select(
        polars.col(stage),
        polars.col(f"train_loss_{stage}"),
        polars.col(f"val_loss_{stage}")
    )
    df_distr_loss = df.select(
        polars.col(stage),
        polars.col(f"train_distr_loss_{stage}"),
        polars.col(f"val_distr_loss_{stage}")
    )
    df_ponder_loss = df.select(
        polars.col(stage),
        polars.col(f"train_ponder_loss_{stage}"),
        polars.col(f"val_ponder_loss_{stage}")
    )
    df_task_loss = df.select(
        polars.col(stage),
        polars.col(f"train_task_loss_{stage}"),
        polars.col(f"val_task_loss_{stage}")
    )
    df_accuracy = df.select(
        polars.col(stage),
        polars.col(f"train_accuracy_{stage}"),
        polars.col(f"val_accuracy_{stage}")
    )
    df_f1_score = df.select(
        polars.col(stage),
        polars.col(f"train_f1_score_{stage}"),
        polars.col(f"val_f1_score_{stage}")
    )

    return (
        df_loss.collect(),
        df_distr_loss.collect(), df_ponder_loss.collect(), df_task_loss.collect(),
        df_accuracy.collect(), df_f1_score.collect()
    )


def plot(
        metrics_df: polars.DataFrame,
        stage: str,
        name: str
) -> plotly.graph_objects.Figure:
    fig = plotly.express.line(
        data_frame=metrics_df.to_pandas(),
        title=f"{name} {stage}",
        x=stage,
        y=[f"train_{name}_{stage}", f"val_{name}_{stage}"],
        labels={stage: stage, "value": name},
        markers=True
    )
    new_names = {
        f"train_{name}_{stage}": f"train {name}",
        f"val_{name}_{stage}": f"validation {name}"
    }
    fig.for_each_trace(lambda trace: trace.update(name=new_names[trace.name]))
    fig_params = fig.to_dict()
    fig_params["layout"]["legend"]["title"] = None
    fig_params["layout"]["autosize"] = False
    fig.update_layout(**fig_params["layout"])

    return fig
