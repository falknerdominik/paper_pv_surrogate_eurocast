from pathlib import Path
from typing import Any, Callable

import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.losses.numpy import mae

from pv_surrogate_eurocast.constants import Paths, SystemData
from pv_surrogate_eurocast.scripts.helper import load_static_data
from pv_surrogate_eurocast.typedef import NormalizedPVGISSchema


def index_series_loader(index: int, row: pd.Series, time_series_dir: Path) -> tuple[int, pd.DataFrame]:
    """
    Loads and returns a pandas DataFrame from a parquet file based on the given index.

    Args:
        row (pd.Series): The row containing the sample ID.

    Returns:
        pd.DataFrame: The loaded DataFrame.

    """
    return index, pd.read_parquet(time_series_dir / f"{index}.parquet")


def sample_id_series_loader(index: int, row: pd.Series, time_series_dir: Path) -> tuple[int, pd.DataFrame]:
    """
    Loads and returns a pandas DataFrame from a parquet file based on the given index.

    Args:
        row (pd.Series): The row containing the sample ID.

    Returns:
        pd.DataFrame: The loaded DataFrame.

    """
    return row["sample_id"], pd.read_parquet(time_series_dir / f"{row['sample_id']}.parquet")


class TimeSeriesLoader:
    def __init__(
        self,
        metadata_parquet_path: Path,
        time_series_dir: Path,
        series_loader: Callable[[int, Any, Path], pd.DataFrame] = index_series_loader,
        limit: int = None,
    ):
        self.metadata_parquet_path = metadata_parquet_path
        self.metadata = pd.read_parquet(metadata_parquet_path)
        if limit is not None:
            # limit provides an option to only load a subset of the data
            self.metadata = self.metadata[0:limit]
        self.data_length = len(self.metadata)

        self.time_series_dir = time_series_dir

        self.series_loader = series_loader

    def __getitem__(self, index: int):
        if index < 0 or index >= self.data_length:
            raise IndexError("Index out of bounds")

        row = self.metadata.iloc[index]
        return self.series_loader(index, row, self.time_series_dir)

    def __iter__(self):
        for i in range(self.data_length):
            yield self[i]

    def __len__(self):
        return self.data_length


def eval_for_dl(metadata_path: str, target_column: str, data_dir: Path, target_dir: Path, limit: int = None):
    loader = TimeSeriesLoader(metadata_path, data_dir, sample_id_series_loader)
    nf = NeuralForecast.load(path=Paths.model_checkpoints)
    static_data = load_static_data(target_column, metadata_path)
    evaluations = []
    for sample_id, module in loader:
        data = (
            module.loc[:, [NormalizedPVGISSchema.ds, target_column]]
            .assign(unique_id=sample_id)
            .rename(columns={target_column: "y"})
        )

        # predictions dataframe should contain the following columns:
        # ds, AutoNHITS, AutoTFT, AutoMLP, y
        predictions = nf.predict(df=data, static_df=static_data, horizon=24, freq="H").reset_index()
        # joining
        predictions = predictions.merge(data, on="ds", how="left")
        predictions.to_parquet(target_dir / f"{sample_id}.parquet")

        # calculate total MAE
        y_true = data["y"].values
        for model in ["AutoNHITS", "AutoTFT", "AutoMLP"]:
            y_hat = predictions[model].values
            error = mae(y_hat, y_true)
            evaluations.append({"model": model, **static_data.loc[sample_id].to_dict(), "mae": error})

    # evaluation dataframe should contain the following columns:
    # model, unique_id, kwP, orientation, tilt, mae
    pd.DataFrame(evaluations).to_parquet(target_dir / "evaluations.parquet")


def eval_for_symreg(metadata_path: str, target_column: str, data_dir: Path, target_dir: Path, limit: int = None):
    # TODO
    pass


def main():
    # deep learning
    eval_for_dl(
        SystemData.german_enriched_test_distribution,
        NormalizedPVGISSchema.power,
        Paths.pvgis_data_dir,
        Paths.general_test_results_dl,
        limit=1000,
    )
    eval_for_dl(
        SystemData.german_enriched_test_distribution,
        NormalizedPVGISSchema.power,
        Paths.pvgis_fixed_location,
        Paths.fixed_points_results_dl,
        limit=1000,
    )
    eval_for_dl(
        SystemData.german_enriched_test_distribution,
        NormalizedPVGISSchema.power,
        Paths.pvgis_outward_data_dir,
        Paths.outward_points_results_dl,
        limit=1000,
    )

    # symbolic regression
    eval_for_symreg(
        SystemData.german_enriched_test_distribution,
        NormalizedPVGISSchema.power,
        Paths.pvgis_data_dir,
        Paths.general_test_results_symreg,
        limit=1000,
    )
    eval_for_symreg(
        SystemData.german_enriched_test_distribution,
        NormalizedPVGISSchema.power,
        Paths.pvgis_fixed_location,
        Paths.fixed_points_results_symreg,
        limit=1000,
    )
    eval_for_symreg(
        SystemData.german_enriched_test_distribution,
        NormalizedPVGISSchema.power,
        Paths.pvgis_outward_data_dir,
        Paths.outward_points_results_symreg,
        limit=1000,
    )


if __name__ == "__main__":
    main()
