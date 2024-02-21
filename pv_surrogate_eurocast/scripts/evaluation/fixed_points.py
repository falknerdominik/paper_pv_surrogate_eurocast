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
    ):
        self.metadata_parquet_path = metadata_parquet_path
        self.metadata = pd.read_parquet(metadata_parquet_path)
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


def eval_for_dl(metadata_path: str, target_column: str, data_dir: str):
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
        predictions = nf.predict(df=data, static_df=static_data, horizon=24, freq="H").reset_index()
        predictions.to_parquet(Paths.fixed_points_results_dl / f"{sample_id}.parquet")

        # calculate total MAE
        y_true = data["y"].values
        for model in ["AutoNHITS", "AutoTFT", "AutoMLP"]:
            y_hat = predictions[model].values
            error = mae(y_hat, y_true)
            evaluations.append({"model": model, **static_data.loc[sample_id].to_dict(), "mae": error})

    pd.DataFrame(evaluations).to_parquet(Paths.fixed_points_results_dl / "evaluations.parquet")


def main():
    eval_for_dl(SystemData.german_enriched_test_distribution, Paths.pvgis_fixed_location, NormalizedPVGISSchema.power)


if __name__ == "__main__":
    main()
