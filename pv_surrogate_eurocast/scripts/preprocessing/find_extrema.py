from pathlib import Path

import geopandas as gpd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from pv_surrogate_eurocast.constants import GeoData, Paths, SystemData
from pv_surrogate_eurocast.typedef import NormalizedPVGISSchema


def plot_histogram_for_day(
    data: pd.DataFrame,
    x: str,
    y: str,
    ax: plt.Axes,
    day: str,
    station: str,
    unit: str = None,
) -> None:
    """
    Plots a histogram for the given day using the specified DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        ax (plt.Axes): The axes on which to plot the histogram.
        day (str): The day for which the histogram will be plotted.
        station (str): The station for which the histogram will be plotted.
        unit (str): The unit of the y-axis.

    Returns:
        None
    """
    day_data = data[(data[x] >= pd.to_datetime(day)) & (data[x] < pd.to_datetime(day) + pd.Timedelta(days=1))]
    # ax.hist(day_data["global_irradiance"], bins=24, color="skyblue")  # Assuming hourly data for 24 bins
    ax.bar(day_data[x], day_data[y], width=0.03, color="blue")  # Adjust width as needed

    # axis formatting
    ax.xaxis.set_label_text("Hour of the Day")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))  # Adjust format as needed
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))  # Set interval to display every hour
    ax.tick_params(axis="x", rotation=45)  # Rotate x-axis labels for better readability

    ax.yaxis.set_label_text(unit)

    ax.set_title(f"Station {station}")


def find_extreme_points(points: gpd.GeoDataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Finds the single most northern, southern, western, and eastern points in a GeoDataFrame.

    This function uses GeoPandas to identify the most extreme points based on latitude and longitude.
    In case of multiple points sharing the same extreme coordinate, the first occurrence is returned
    for each direction.

    Args:
        points (GeoDataFrame): The GeoDataFrame containing a 'geometry' column with Point geometries.

    Returns:
        dict: A dictionary with keys 'north', 'south', 'west', 'east', each mapping to a GeoDataFrame
        row (as a pandas Series) representing the single specific point in each cardinal direction.

    Example:

    Note:
        The function assumes the 'geometry' column contains Shapely Point objects and the GeoDataFrame
        is not empty. It does not perform CRS transformations; ensure the GeoDataFrame's CRS is appropriate
        for latitude and longitude comparisons.
    """
    # Get the bounding box of all geometries
    minx, miny, maxx, maxy = points.geometry.total_bounds

    # Find the points that are most extreme and return only the first occurrence
    north = points[points.geometry.y == maxy].iloc[0]
    south = points[points.geometry.y == miny].iloc[0]
    west = points[points.geometry.x == minx].iloc[0]
    east = points[points.geometry.x == maxx].iloc[0]

    return (north, south, west, east)


def draw_country_points(
    points: gpd.GeoDataFrame,
    country_name: str,
    target_path: Path,
):
    """
    Draws the given points on a map of the specified country.

    Args:
        points (gpd.GeoDataFrame): A GeoDataFrame containing the points to be plotted.
        country_name (str): The name of the country to plot.
        target_path (Path): The path to save the resulting figure.

    Returns:
        None
    """
    # Assuming you have a GeoDataFrame for country boundaries
    countries_gdf = gpd.read_file(GeoData.natural_earth_data)
    germany = countries_gdf[countries_gdf["NAME"] == country_name].to_crs(epsg=4326)
    points.set_crs(epsg=4326, inplace=True)

    fig, ax = plt.subplots(figsize=(5, 5))
    germany.plot(ax=ax, color="white", edgecolor="black")
    points.plot(ax=ax, marker="x", color="red", markersize=30)

    for index, (x, y, label) in enumerate(zip(points.geometry.x, points.geometry.y, points.sample_id)):
        # this is a bit of a hack to make the labels look nice on the german map. It offsets the respective labels
        # to avoid the country border. Should be okay for most countries, but might need adjustment for others.
        match index:
            case 1:
                ax.annotate(label, xy=(x, y), xytext=(-3, -11), textcoords="offset points")
            case 2:
                ax.annotate(label, xy=(x, y), xytext=(9, -3), textcoords="offset points")
            case _:
                ax.annotate(label, xy=(x, y), xytext=(3, 3), textcoords="offset points")

    ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(target_path)


def draw_histograms_per_day(
    north: tuple[int, pd.DataFrame],
    south: tuple[int, pd.DataFrame],
    west: tuple[int, pd.DataFrame],
    east: tuple[int, pd.DataFrame],
    x: str,
    y: str,
    day: str,
    target_path: Path,
    day_name: str = None,
    unit: str = None,
):
    """
    Draws a 2x2 grid subplot with bar plots for each direction.

    Args:
        north: Tuple containing the hour and DataFrame for the North direction.
        south: Tuple containing the hour and DataFrame for the South direction.
        west: Tuple containing the hour and DataFrame for the West direction.
        east: Tuple containing the hour and DataFrame for the East direction.
        day: Day for which the histograms are plotted.
    """
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    plot_histogram_for_day(north[1], x, y, axs[0, 0], day, north[0], unit=unit)
    plot_histogram_for_day(south[1], x, y, axs[0, 1], day, south[0], unit=unit)
    plot_histogram_for_day(west[1], x, y, axs[1, 0], day, west[0], unit=unit)
    plot_histogram_for_day(east[1], x, y, axs[1, 1], day, east[0], unit=unit)
    if day_name is not None:
        plt.suptitle(f"{y} on {day_name} ({day})")
    else:
        plt.suptitle(f"{y} on {day}")

    plt.tight_layout()
    plt.savefig(target_path)


def main():
    points = gpd.read_parquet(SystemData.german_enriched_train_distribution)
    north, south, west, east = find_extreme_points(points)
    country_name = "Germany"
    draw_country_points(
        gpd.GeoDataFrame([north, south, west, east]),
        country_name,
        Paths.extrema_dir / f"{country_name}_extrema.pdf",
    )

    interesting_days = {
        "2019": {
            "spring_equinox": "2019-03-20",
            "summer_solstice": "2019-06-21",
            "autumn_equinox": "2019-09-23",
            "winter_solstice": "2019-12-22",
        },
        "2020": {
            "spring_equinox": "2020-03-19",
            "summer_solstice": "2020-06-20",
            "autumn_equinox": "2020-09-22",
            "winter_solstice": "2020-12-21",
        },
    }

    for target_variable, unit in [
        (NormalizedPVGISSchema.global_irradiance, "W/m^2"),
        (NormalizedPVGISSchema.power, "watts"),
    ]:
        target_path = Paths.extrema_dir / target_variable
        target_path.mkdir(parents=True, exist_ok=True)

        for year in interesting_days.keys():
            for name, day in interesting_days[year].items():
                draw_histograms_per_day(
                    (north["sample_id"], pd.read_parquet(Paths.pvgis_data_dir / f"{north['sample_id']}.parquet")),
                    (south["sample_id"], pd.read_parquet(Paths.pvgis_data_dir / f"{south['sample_id']}.parquet")),
                    (west["sample_id"], pd.read_parquet(Paths.pvgis_data_dir / f"{west['sample_id']}.parquet")),
                    (east["sample_id"], pd.read_parquet(Paths.pvgis_data_dir / f"{east['sample_id']}.parquet")),
                    NormalizedPVGISSchema.ds,
                    target_variable,
                    day,
                    target_path / f"{year}_{name}.pdf",
                    day_name=name,
                    unit=unit,
                )


if __name__ == "__main__":
    main()
