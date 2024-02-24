from math import cos, sin, pow
from pathlib import Path
from typing import Callable
import numpy as np
import pandas as pd
import geopandas as gpd
from pysr import PySRRegressor

from pv_surrogate_eurocast.constants import Paths, SystemData
from pv_surrogate_eurocast.scripts.evaluate_models import TimeSeriesLoader, sample_id_series_loader
from pv_surrogate_eurocast.typedef import NormalizedPVGISSchema

import pandas as pd
import datetime
import math
import pvlib
from pvlib.solarposition import sun_rise_set_transit_spa

def is_sun_up_series(dates, latitude, longitude):
    """
    Calculate a boolean series indicating whether the sun is up for each hour in the input series.
    
    Args:
    - dates (pd.Series): Series of datetime values.
    - latitude (float): Latitude of the location in decimal degrees (positive for north, negative for south).
    - longitude (float): Longitude of the location in decimal degrees (positive for east, negative for west).
    
    Returns:
    - pd.Series: Boolean series indicating whether the sun is up for each hour.
    """
    # Calculate sunrise and sunset times for the series of datetime values
    # sun = calculate_sunrise_sunset_series(dates, latitude, longitude, tilt, azimuth)
    sun = sun_rise_set_transit_spa(
        pd.DatetimeIndex(pd.to_datetime(dates.dt.date, utc=True).drop_duplicates()), latitude, longitude
    )
    left = pd.DataFrame(dates)
    left['day'] = pd.to_datetime(dates.dt.tz_localize('utc').dt.date, utc=True)
    sun = pd.merge(left, sun, left_on='day', right_on='ds')
    sun['ds'] = sun['ds'].dt.tz_localize('utc')
    sun_up = (sun['ds'] > sun['sunrise']) & (sun['ds'] < sun['sunset'])
    sun_up = sun_up.astype(int)
    return sun_up

def enrich_with_covariates(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data
        .assign(
            # Create covariates for time series forecasting, normalize them between -0.5 and 0.5
            # hour_of_day=lambda df: df.ds.dt.hour - 12,
            # hour_of_day=lambda df: -0.05 * (df.ds.dt.hour - 12)**2 + 1,
            # week_of_year=lambda df: df.ds.dt.isocalendar().week,
            week_of_year=lambda df: (df.ds.dt.isocalendar().week - 1) / 52.0,
            # day_of_month=lambda df: (df.ds.dt.day - 1) / 30.0 - 0.5,
            month=lambda df: (df.ds.dt.month - 1) / 11.0 - 0.5,
            # day_of_year=lambda df: (df.ds.dt.dayofyear - 1) / 365.0 - 0.5,
        )
    )


def load_data(target_column: str, limit: int = None) -> pd.DataFrame:
    data = gpd.read_parquet(SystemData.german_enriched_train_distribution)

    count_time_series = data.shape[0]
    if limit is not None:
        count_time_series = limit
    else:
        count_time_series = min(limit, data.shape[0])

    sum = []
    for i in range(count_time_series):
        series = data.iloc[i]
        try:
            target = pd.read_parquet(Paths.pvgis_data_dir / f"{series['sample_id']}.parquet")
            target = (
                target.loc[:, [NormalizedPVGISSchema.ds, target_column]]
                .assign(unique_id=series["sample_id"])
                .rename(columns={target_column: "y"})
            )

            lon = series.geometry.x
            lat = series.geometry.y
            is_sun_up = is_sun_up_series(target.ds, lat, lon)
            target = target.assign(is_sun_up=is_sun_up)

            # normalize parameter and convert to kW
            if target_column == NormalizedPVGISSchema.power:
                target.y = target.y / 1000

            target = target.groupby('unique_id').resample('D', on='ds').sum().drop(columns=['unique_id']).reset_index()
            target = (
                # enrich with time based variables
                enrich_with_covariates(target)
                # enrich with known static values
                .assign(
                    kwP=series['kwP'],
                    orientation=series['orientation'],
                    tilt=series['tilt'],
                    # remap lon/lat to x,y,z coordinates
                    lat=lat,
                    lon=lon,
                    # x_pos=cos(lat) * cos(lon),
                    # y_pos=cos(lat) * sin(lon), 
                    # z_pos=sin(lat),
                )
            )
            target = target[:(365)]
            sum.append(target)
            print(f'Loaded {i+1}/{count_time_series} time series')
        except Exception as e:
            print(f'Could not load {i+1}/{count_time_series} time series')
            print(e)
    sum = pd.concat(sum)
    return sum

def file_name_path_standard(row):
    return f'{row["sample_id"]}.parquet'

def file_name_path_outward(row):
    return f'{row["sample_id"]}_{row["bearing"]}_{row["distance"]}.parquet'

def symreg_series_loader(target_column: str, path_builder: Callable = file_name_path_standard) -> tuple[int, pd.DataFrame, pd.Series]:

    def symreg_series_loader_intern(index: int, row: pd.Series, time_series_dir: Path) -> tuple[int, pd.DataFrame, pd.Series]:
        try:
            # if fixed point
            target = pd.read_parquet(time_series_dir / path_builder(row))
            # if normal
            # target = gpd.read_parquet(time_series_dir / path_builder(row))
            target = (
                target.loc[:, [NormalizedPVGISSchema.ds, target_column]]
                .assign(unique_id=row["sample_id"])
                .rename(columns={target_column: "y"})
            )

            # if fixed point
            # lon = row.lon
            # lat = row.lat
            # if normal
            lon = row.geometry.x
            lat = row.geometry.y
            # normalize parameter and convert to kW
            if target_column == NormalizedPVGISSchema.power:
                target.y = target.y / 1000

            is_sun_up = is_sun_up_series(target.ds, lat, lon)
            target = target.assign(is_sun_up=is_sun_up)
            target = target.groupby('unique_id').resample('D', on='ds').sum().drop(columns=['unique_id']).reset_index()
            target = (
                # enrich with time based variables
                enrich_with_covariates(target)
                # enrich with known static values
                .assign(
                    # if normal
                    kwP=row['kwP'],
                    orientation=row['orientation'],
                    tilt=row['tilt'],

                    # if fixed point
                    # kwP=row['peakpower'],
                    # orientation=row['aspect'],
                    # tilt=row['angle'],

                    # remap lon/lat to x,y,z coordinates
                    lat=lat,
                    lon=lon,
                )
            )
            target = target[:(365)]
            target = target.sample(frac=1)
            return row['sample_id'], target.drop(columns=["y", "ds", 'unique_id']), target.y
        except Exception as e:
            return row['sample_id'], None, None
        
    return symreg_series_loader_intern


def train(target_column: str, limit: int | None = None):
    p = load_data(target_column, limit=limit)
    p = p.sample(frac=1)
    y = p.y
    X = p.drop(columns=["y", "ds", 'unique_id'])

    model = PySRRegressor(
        binary_operators=["+", "*", '/', '-'], 
        unary_operators=["sin"],
        batching=True, batch_size=73,
        model_selection='accuracy',
        nested_constraints={"sin": {"sin": 0}},
        elementwise_loss='myloss(x, y) = abs(x-y) / max(abs(x), 1)',
        equation_file=str(Paths.symreg_model)
    )
    model.fit(X, y)
    print(model)
    print('h')


def test(
        equation_file: Path, metadata_path: str, target_column: str, data_dir: Path, target_path: Path, limit: int = None,
        path_builder: Callable = file_name_path_standard
):
    loader = TimeSeriesLoader(metadata_path, data_dir, symreg_series_loader(target_column, path_builder))
    model = PySRRegressor.from_file(str(equation_file))
    scores = []
    for index, (sample_id, X, y) in enumerate(loader):

        if X is None or y is None:
            print(f'Skipped {sample_id}')
            continue
        else:
            print(f'Sample {sample_id} loaded')

        try:
            prediction = model.predict(X)
        except Exception as e:
            print(f'Skipped Sample {sample_id}')
            continue

        result = pd.DataFrame({
            'SymReg': prediction,
            'y': y,
            'MAE': abs(prediction-y),
            'MAPE': abs(prediction-y) / y
        })
        results_with_params = pd.concat([X, result], axis=1).mean()
        # lat, lon = cartesian_to_spherical(**results_with_params[['x_pos', 'y_pos', 'z_pos']].to_dict())
        scores.append({
            'sample_id': sample_id,
            'model': 'SymReg',
            'lat': results_with_params.lat,
            'lon': results_with_params.lon,
            'kwP': results_with_params.kwP,
            'orientation': results_with_params.orientation,
            'tilt': results_with_params.tilt,
            'MAE': results_with_params.MAE,
            'MAPE': results_with_params.MAPE,
            'R2': model.score(X, y).mean(),
        })

        if index >= limit:
            break
    aggregated_results = pd.DataFrame.from_dict(scores)
    aggregated_results.to_parquet(target_path)


def main():
    target_column = NormalizedPVGISSchema.global_irradiance
    # train(target_column, limit=500)
    test(
        Paths.symreg_model, 
        SystemData.german_outward_points,
        target_column,
        Paths.pvgis_outward_data_dir,
        Paths.pvgis_outward_data_dir / 'evaluation.parquet',
        limit=1000,
        path_builder=file_name_path_outward
    )
    # test(
    #     Paths.symreg_model, 
    #     SystemData.german_fixed_location_points,
    #     target_column,
    #     Paths.pvgis_fixed_location,
    #     Paths.pvgis_fixed_location / 'evaluation.parquet',
    #     limit=1000,
    #     path_builder=file_name_path_standard,
    # )


if __name__ == '__main__':
    main()
    

