from math import cos, sin
import pandas as pd
import geopandas as gpd
from pysr import PySRRegressor

from pv_surrogate_eurocast.constants import Paths, SystemData
from pv_surrogate_eurocast.typedef import NormalizedPVGISSchema

def enrich_with_covariates(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data
        .assign(
            # Create covariates for time series forecasting, normalize them between -0.5 and 0.5
            hour_of_day=lambda df: df.ds.dt.hour / 23.0 - 0.5,
            day_of_month=lambda df: (df.ds.dt.day - 1) / 30.0 - 0.5,
            month=lambda df: (df.ds.dt.month - 1) / 11.0 - 0.5,
            day_of_year=lambda df: (df.ds.dt.dayofyear - 1) / 365.0 - 0.5,
            week_of_year=lambda df: (df.ds.dt.isocalendar().week - 1) / 52.0 - 0.5,
        )
    )


def load_data(target_column: str, limit: int = None) -> pd.DataFrame:
    data = gpd.read_parquet(SystemData.german_enriched_train_distribution)

    count_time_series = data.shape[0]
    if limit is not None:
        count_time_series = limit
    else:
        count_time_series = data.shape[0]

    sum = pd.DataFrame()
    for i in range(count_time_series):
        series = data.iloc[i]
        target = pd.read_parquet(Paths.pvgis_data_dir / f"{series['sample_id']}.parquet")
        target = (
            target.loc[:, [NormalizedPVGISSchema.ds, target_column]]
            .assign(unique_id=series["sample_id"])
            .rename(columns={target_column: "y"})
        )

        lon = series.geometry.x
        lat = series.geometry.y
        target = (
            # enrich with time based variables
            enrich_with_covariates(target)
            # enrich with known static values
            .assign(
                kwP=series['kwP'],
                orientation=series['orientation'],
                tilt=series['tilt'],
                # remap lon/lat to x,y,z coordinates
                x_pos=cos(lat) * cos(lon),
                y_pos=cos(lat) * sin(lon), 
                z_pos=sin(lat),
            )
        )
        print(f'Loaded {i+1}/{count_time_series} time series')
        sum = pd.concat([sum, target])
    return sum


def main():
    p = load_data(NormalizedPVGISSchema.power, limit=2000)
    y = p.y
    X = p.drop(columns=["y", "ds", 'unique_id'])

    model = PySRRegressor(
        binary_operators=["+", "-", "*", "/"], 
        unary_operators=["sin"],
        batching=True, batch_size=1000,
        nested_constraints={"sin": {"sin": 0}},
        maxsize=50,
        population_size=64,
        select_k_features=5,
    )
    model.fit(X, y)
    print(model)


if __name__ == '__main__':
    main()