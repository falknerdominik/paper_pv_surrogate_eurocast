
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.auto import AutoMLP, AutoNHITS, AutoTFT
from neuralforecast.losses.pytorch import MSE
from ray.tune.search.hyperopt import HyperOptSearch

from pv_surrogate_eurocast.constants import Paths, SystemData
from pv_surrogate_eurocast.scripts.helper import load_static_data
from pv_surrogate_eurocast.typedef import NormalizedPVGISSchema


def load_train_data(target_column: str):
    data = pd.read_parquet(SystemData.german_enriched_train_distribution)

    sum = pd.DataFrame()
    count_time_series = 100
    for i in range(count_time_series):
        series = data.iloc[i]
        target = pd.read_parquet(Paths.pvgis_data_dir / f"{series['sample_id']}.parquet")
        target = (
            target.loc[:, [NormalizedPVGISSchema.ds, target_column]]
            .assign(unique_id=series["sample_id"])
            .rename(columns={target_column: "y"})
        )
        sum = pd.concat([sum, target])
    return sum


def main():
    # load data for pretraining
    for target in [NormalizedPVGISSchema.global_irradiance, NormalizedPVGISSchema.power]:
        static_data = load_static_data(target, SystemData.german_enriched_train_distribution)
        data = load_train_data(target)

        # fit neuralforecast models
        horizon = 24
        models = [
            AutoNHITS(h=horizon, loss=MSE(), search_alg=HyperOptSearch(), num_samples=100, scaler_type="standard"),
            AutoTFT(h=horizon, loss=MSE(), search_alg=HyperOptSearch(), num_samples=100, scaler_type="standard"),
            AutoMLP(h=horizon, loss=MSE(), search_alg=HyperOptSearch(), num_samples=100, scaler_type="standard"),
        ]
        nf = NeuralForecast(models=models, freq="H")
        nf.fit(df=data, static_df=static_data, val_size=0, sort_df=True)

        path = Paths.model_checkpoints / f"{target}_pretraining"
        path.mkdir(parents=True, exist_ok=True)
        nf.save(path=str(Paths.model_checkpoints), model_index=None, overwrite=True, save_dataset=False)


if __name__ == "__main__":
    main()
