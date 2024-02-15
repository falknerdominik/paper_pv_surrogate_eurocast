import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

from pv_surrogate_eurocast.constants import Paths


def plot_histogram_for_day(data: pd.DataFrame, ax: plt.Axes, day: str) -> None:
    """
    Plots a histogram for the given day using the specified DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        ax (plt.Axes): The axes on which to plot the histogram.
        day (str): The day for which the histogram will be plotted.

    Returns:
        None
    """
    day_data = data[(data["ds"] >= pd.to_datetime(day)) & (data["ds"] < pd.to_datetime(day) + pd.Timedelta(days=1))]
    ax.hist(day_data["y"], bins=24, color="skyblue")  # Assuming hourly data for 24 bins
    ax.set_title(f"Histogram for {day}")


def find_extreme_points(points: gpd.GeoDataFrame):
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


def draw_country_points_and_histograms(
    points: gpd.GeoDataFrame,
    north: pd.DataFrame,
    south: pd.DataFrame,
    west: pd.DataFrame,
    east: pd.DataFrame,
    day: str,
    country_name: str,
):
    """
    Draws country points and histograms for a given day.

    Args:
        points (gpd.GeoDataFrame): GeoDataFrame containing the points to be plotted.
        north (pd.DataFrame): DataFrame containing data for the northern region.
        south (pd.DataFrame): DataFrame containing data for the southern region.
        west (pd.DataFrame): DataFrame containing data for the western region.
        east (pd.DataFrame): DataFrame containing data for the eastern region.
        day (str): The day for which the histograms will be plotted.
        country_name (str): The name of the country to be plotted.

    Returns:
        None
    """

    # Assuming you have a GeoDataFrame for country boundaries
    countries_gdf = gpd.read_file(Paths.natural_earth_data)
    germany = countries_gdf[countries_gdf["NAME"] == country_name]

    # Create figure and axes
    _, axs = plt.subplots(4, 2, figsize=(15, 20), gridspec_kw={"width_ratios": [1, 3]})

    # Plot Germany on the left side
    germany.plot(ax=axs[0, 0], color="lightblue")
    points.plot(ax=axs[0, 0], marker="o", color="red", markersize=5)
    axs[0, 0].axis("off")  # Remove axes
    for x, y, label in zip(points.geometry.x, points.geometry.y, points["sample_id"]):
        axs[0, 0].text(x, y, label)

    # Ensure the other left plots do not show anything
    for ax in axs[1:, 0]:
        ax.axis("off")

    # Plot histograms on the right side
    dataframes = [north, south, west, east]
    for i, df in enumerate(dataframes, start=1):
        plot_histogram_for_day(df, axs[i, 1], day)

    plt.tight_layout()
    plt.show()


def main():
    pass


if __name__ == "__main__":
    main()
