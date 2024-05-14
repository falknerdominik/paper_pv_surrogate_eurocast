import logging
from pathlib import Path

import pandas as pd
from prefect import flow, task
from pvoutput import PVOutput
from pvoutput.grid_search import GridSearch

from pv_surrogate_eurocast.constants import Paths, SystemData

logger = logging.getLogger(__name__)


def search_using_grid(pvo: PVOutput, grid: pd.DataFrame, search_radius: float = 25, cache_dir=Paths.cache_dir):
    systems = None
    # object for searching sub grids
    grid_search = GridSearch(cache_dir=cache_dir)

    for i, loc in grid.iterrows():
        logger.info(
            f"Searching for systems within {search_radius}km of {loc.latitude}, {loc.longitude} ({i} of {len(grid)})..."
        )
        new_systems = pvo.search(
            query=f"{search_radius}km", lat=loc.latitude, lon=loc.longitude, wait_if_rate_limit_exceeded=True
        )
        if (
            len(new_systems) >= 30
        ):  # PVOutput returns a maximum of 30 systems, so we'll drill down and conduct a finer search
            logger.info("PVOutput returned 30 systems... drilling down")
            grid_ = grid_search.generate_grid(
                radial_clip=(loc.latitude, loc.longitude, 25), buffer=0, search_radius=5, show=False
            )
            # Call this function recursively with a finer search radius
            new_systems = search_using_grid(pvo, grid_, search_radius=search_radius / 5.0, cache_dir=cache_dir)
        if systems is None:
            systems = new_systems
        else:
            systems = pd.concat((systems, new_systems if not new_systems.empty else pd.DataFrame()))
        logger.info(f"Found {len(new_systems)} new systems")
    # Remove any duplicates due to overlapping search radii
    return systems[~systems.index.duplicated()]


@task
def download_system_metadata(pvo: PVOutput, system_data_path: Path) -> pd.DataFrame:
    systems = pd.read_parquet(system_data_path)

    def get_metadata(system_id):
        return pvo.get_metadata(system_id, wait_if_rate_limit_exceeded=True)

    metadata = systems.reset_index().system_id.apply(get_metadata)
    return metadata


@task
def search_for_systems(country: str, pvoutput: PVOutput, geo_cache_dir: Path = Paths.cache_dir) -> pd.DataFrame:
    # using geo pandas to create a grid
    grd = GridSearch(cache_dir=geo_cache_dir)
    country_grid = grd.generate_grid(
        countries=[country],  # List as many countries as you want, or set to None for world-wide
        # show=True,  # Gives a nice plot of the region and grid
    )

    systems = search_using_grid(pvoutput, country_grid)
    return (
        systems
        # index == system_id, reset it to normal column
        .reset_index()
    )


@flow
def main():
    # context
    api_key = None
    system_id = None
    CACHE_DIR = Paths.cache_dir

    # parameters
    to_download = [
        # (
        #     "Austria",
        #     SystemData.austrian_systems,
        #     SystemData.meta_austrian_systems,
        # ),
        # (
        #     "Germany",
        #     SystemData.german_systems,
        #     SystemData.meta_german_systems,
        # ),
        (
            "Italy",
            SystemData.italy_systems,
            SystemData.meta_italy_systems,
        )
    ]

    for country, system_target_path, system_meta_target_path in to_download:
        pvo = PVOutput(api_key, system_id)

        systems = search_for_systems(country, pvo, CACHE_DIR)
        systems.to_parquet(system_target_path)

        # Download metadata
        metadata = download_system_metadata(pvo, system_target_path)
        metadata.to_parquet(system_meta_target_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
