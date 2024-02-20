import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from joblib import Parallel, delayed
from prefect import task

from pv_surrogate_eurocast.constants import Paths, SystemData
from pv_surrogate_eurocast.typedef import (
    ConfigurationEntry,
    PVGISConfigurationSchema,
    map_pvgis_raw_to_normalized,
    map_system_data_to_pvgis_configuration,
)

hourly_uri = "https://re.jrc.ec.europa.eu/api/v5_2/seriescalc"

logger = logging.getLogger(__name__)


def query_pvgis(config_entry) -> dict | tuple[None, int]:
    response = requests.get(hourly_uri, params=config_entry.model_dump())
    if response.status_code == 200:
        return response.json()
    else:
        raise requests.HTTPError(f"Request failed with status code {response.status_code}. Reason {response.content}")


@task
def download_and_save_pvgis_data(record: PVGISConfigurationSchema, target_dir: Path = Paths.pvgis_data_dir):
    config = ConfigurationEntry(
        lat=record["lat"],
        lon=record["lon"],
        peakpower=record["peakpower"],
        angle=record["angle"],
        aspect=record["aspect"],
        loss=record["loss"],
        mounting=record["mounting"],
        startyear=2005,
        endyear=2020,
        outputformat="json",
        usehorizon=1,
        pvcalculation=1,
    )

    try:
        raw_data = query_pvgis(config)

        # normalize dataframe to standard structure
        data = pd.json_normalize(raw_data["outputs"]["hourly"])
        data = data.rename(columns=map_pvgis_raw_to_normalized).assign(
            ds=lambda df: pd.to_datetime(df["ds"], format="%Y%m%d:%H%M")
        )

        data.to_parquet(target_dir / f"{record['sample_id']}.parquet")
        json = config.model_dump_json()
        with open(target_dir / f"{record['sample_id']}.json", "w") as file:
            file.write(json)
        logger.info(f"Downloaded data for {record['sample_id']}")
    except requests.HTTPError as e:
        logger.error(f"Failed to download data for {record['sample_id']}. Reason {e}")
        error_file = target_dir / f"error_{record['sample_id']}.json"
        with open(error_file, "w") as file:
            file.write(str(e))


@task
def read_configurations(distribution_path: Path) -> list[dict]:
    distribution = gpd.read_parquet(distribution_path)
    pvgis_configurations: PVGISConfigurationSchema = (
        distribution.assign(lon=distribution["geometry"].x, lat=distribution["geometry"].y)
        .drop(columns=["geometry"])
        .rename(columns=map_system_data_to_pvgis_configuration)
    )
    # done = pd.read_csv("/Users/dfalkner/projects/pv-surrogate-eurocast/already_done.csv")
    # pvgis_configurations = pvgis_configurations[~pvgis_configurations["sample_id"].isin(done.iloc[:, 0])]
    return pvgis_configurations.to_dict(orient="records")


def remap_configuration_outward_points(config: dict) -> dict:
    return {
        "sample_id": f'{config["sample_id"]}_{config["bearing"]}_{config["distance"]}',
        "lat": config["lat"],
        "lon": config["lon"],
        "peakpower": config["peakpower"],
        "angle": config["angle"],
        "aspect": config["aspect"],
        "loss": config["loss"],
        "mounting": config["mounting"],
    }


def main():
    logging.basicConfig(level=logging.INFO)

    # train dataset
    # pvgis_configurations = read_configurations.fn(SystemData.german_enriched_train_distribution)
    # Parallel(n_jobs=4)(delayed(download_and_save_pvgis_data.fn)(config) for config in pvgis_configurations)

    # test dataset
    # pvgis_configurations = read_configurations.fn(SystemData.german_enriched_test_distribution)
    # Parallel(n_jobs=4)(delayed(download_and_save_pvgis_data.fn)(config) for config in pvgis_configurations)

    # download outward points
    pvgis_configurations = read_configurations.fn(SystemData.german_outward_points)
    Parallel(n_jobs=4)(
        delayed(download_and_save_pvgis_data.fn)(
            remap_configuration_outward_points(config), Paths.pvgis_outward_data_dir
        )
        for config in pvgis_configurations
    )


if __name__ == "__main__":
    main()
