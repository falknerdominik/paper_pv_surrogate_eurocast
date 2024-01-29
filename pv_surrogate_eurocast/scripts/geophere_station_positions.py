import json

import geopandas as gpd
import pandas as pd
from matplotlib import pyplot as plt
from shapely.geometry import Point

from pv_surrogate_eurocast.constants import ModulesNearsGeoshpereReport


def filter_modules_near_station(
    stations: gpd.GeoDataFrame,
    modules: gpd.GeoDataFrame,
    radius: float = 5,
    crs: str = "EPSG:4326",
    distance_crs: int = 32633,
):
    """
    Filter modules based on proximity to a station.

    Parameters:
    -----------
    stations: gpd.GeoDataFrame
        Stations as a GeoDataFrame.
    modules: gpd.GeoDataFrame
        Modules as a GeoDataFrame.
    radius: float
        Buffer radius in kilometers.
    crs: str
        Coordinate reference system (CRS) of the input data.
    distance_crs: int
        Coordinate reference system (CRS) for distance calculation.
    """
    # stations_gdf = stations_gdf.set_crs(crs)
    # modules_gdf = modules_gdf.set_crs(crs)

    # convert CRS to a projected system for distance calculation (e.g., UTM)
    stations = stations.to_crs(epsg=distance_crs)  # UTM zone 33N, adjust as needed
    modules = modules.to_crs(epsg=distance_crs)  # UTM zone 33N, adjust as needed

    # buffer distance in kilometers (e.g., 10km)
    buffer_radius = radius * 1000  # 10 km in meters

    # create buffers around stations
    stations["buffer"] = stations.geometry.buffer(buffer_radius)

    # filter modules based on proximity to any station
    mask = modules.geometry.apply(lambda x: any(x.within(station.buffer) for station in stations.itertuples()))
    return modules[mask]


def plot_module_and_stations(
    country: gpd.GeoDataFrame, modules: gpd.GeoDataFrame, stations: gpd.GeoDataFrame, title: str | None = None
):
    """
    Plot the modules and stations on a map, along with a histogram of the number of outputs.

    Parameters:
        country (gpd.GeoDataFrame): GeoDataFrame representing the country boundaries.
        modules (gpd.GeoDataFrame): GeoDataFrame representing the module locations.
        stations (gpd.GeoDataFrame): GeoDataFrame representing the station locations.
        title (str, optional): Title for the plot. Defaults to None.

    Returns:
        fig (matplotlib.figure.Figure): The generated matplotlib figure.
    """
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))

    # first plot (map)
    country.plot(ax=ax[0], color="lightgray", edgecolor="black")
    stations.plot(ax=ax[0], marker="x", color="blue", markersize=5)
    modules.plot(ax=ax[0], marker="o", color="red", markersize=5)

    # second plot - histogram
    modules["num_outputs"].plot(kind="hist", ax=ax[1])
    ax[1].set_xlabel("Number of Outputs")
    ax[1].set_ylabel("Count")
    fig.suptitle(title)
    return fig


def main():
    # this script plots all geoshpere stations on a map and plots relevant pv modules near it from pvoutput
    path = "/Users/dfalkner/projects/pv-surrogate-eurocast/data/geoshpere/klima-v1-1h.json"
    data = None
    with open(path) as f:
        data = json.load(f)

    # convert to geopandas dataframe
    stations = pd.DataFrame.from_records(data["stations"])
    geometry = [Point(xy) for xy in zip(stations["lon"], stations["lat"])]
    stations = stations.drop(["lon", "lat"], axis=1)
    stations = gpd.GeoDataFrame(stations, crs="EPSG:4326", geometry=geometry)

    # load modules from austria
    austrian_modules = pd.read_parquet(
        "/Users/dfalkner/projects/pv-surrogate-eurocast/data/system_data/pvoutput_austrian_systems.parquet"
    )
    geometry = [Point(xy) for xy in zip(austrian_modules["longitude"], austrian_modules["latitude"])]
    austrian_modules = austrian_modules.drop(["longitude", "latitude"], axis=1)
    austrian_modules = gpd.GeoDataFrame(austrian_modules, crs="EPSG:4326", geometry=geometry)

    # load map file
    world = gpd.read_file("/Users/dfalkner/projects/pv-surrogate-eurocast/data/cache/ne_10m_admin_0_countries.zip")
    austria = world[world["NAME"] == "Austria"]

    # convert everything to the same coordinate reference system
    distance_crs: int = 32633
    stations = stations.to_crs(epsg=distance_crs)  # UTM zone 33N, adjust as needed
    austrian_modules = austrian_modules.to_crs(epsg=distance_crs)  # UTM zone 33N, adjust as needed
    austria = austria.to_crs(epsg=distance_crs)

    # === Plotting
    radius = 1
    modules_near_stations = filter_modules_near_station(stations, austrian_modules, radius=radius)
    fig = plot_module_and_stations(
        austria,
        modules_near_stations,
        stations,
        title=f"Modules within {radius}km of a geoshpere station (n={modules_near_stations.shape[0]})",
    )
    fig.savefig(str(ModulesNearsGeoshpereReport.near_1_km), bbox_inches="tight")

    radius = 5
    modules_near_stations = filter_modules_near_station(stations, austrian_modules, radius=radius)
    fig = plot_module_and_stations(
        austria,
        modules_near_stations,
        stations,
        title=f"Modules within {radius}km of a geoshpere station (n={modules_near_stations.shape[0]})",
    )
    fig.savefig(str(ModulesNearsGeoshpereReport.near_5_km), bbox_inches="tight")

    radius = 10
    modules_near_stations = filter_modules_near_station(stations, austrian_modules, radius=radius)
    fig = plot_module_and_stations(
        austria,
        modules_near_stations,
        stations,
        title=f"Modules within {radius}km of a geoshpere station (n={modules_near_stations.shape[0]})",
    )
    fig.savefig(str(ModulesNearsGeoshpereReport.near_10_km), bbox_inches="tight")

    radius = 20
    modules_near_stations = filter_modules_near_station(stations, austrian_modules, radius=radius)
    fig = plot_module_and_stations(
        austria,
        modules_near_stations,
        stations,
        title=f"Modules within {radius}km of a geoshpere station (n={modules_near_stations.shape[0]})",
    )
    fig.savefig(str(ModulesNearsGeoshpereReport.near_20_km), bbox_inches="tight")


if __name__ == "__main__":
    main()
