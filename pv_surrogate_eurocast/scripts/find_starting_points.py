"""
This script finds good locations for PV modules in the data. It filters any points not within the specified country and ensures that no two points are too close to each other.
It checks:
- That areas around points are within the specified country
- That areas around points do not overlap with each other
- Draws a map of the area using the points
"""

import geopandas as gpd

from pv_surrogate_eurocast.constants import GeoData, Paths

def filter_non_overlapping_points_within_country(natural_earth_data, country_name: str, points: gpd.GeoDataFrame, radius_km: float):
    # Load country boundaries and project to a suitable UTM zone
    world = gpd.read_file(natural_earth_data)
    country = world[world['NAME'] == country_name]
    country = country.to_crs(epsg=32632)  # Example for Germany, adjust UTM zone as needed
    
    # Ensure the points GeoDataFrame is in the same CRS as the country
    old_crs = points.crs
    points = points.to_crs(epsg=32632)
    
    # Initialize list to keep track of valid points
    valid_points = []

    for _, point in points.iterrows():
        buffer = point.geometry.buffer(radius_km * 1000)  # Create a buffer (circle) around the point
        # Check if the buffer is fully within the country
        if not country.geometry.contains(buffer).any():
            continue  # Skip points where the buffer is not fully contained
        
        # Check for overlap with previously accepted buffers
        overlap = False
        for valid_point in valid_points:
            if buffer.intersects(valid_point.geometry.buffer(radius_km * 1000)):
                overlap = True
                break
        
        if not overlap:
            valid_points.append(point)

    # Convert valid points list to GeoDataFrame
    filtered_points_gdf = gpd.GeoDataFrame(valid_points, crs=points.crs)
    
    # Optionally, transform back to geographic coordinates
    filtered_points_gdf = filtered_points_gdf.to_crs(old_crs)
    
    return filtered_points_gdf

def main():
    # Example usage
    # Assuming you have a GeoDataFrame of pre-selected points
    points_gdf = gpd.read_parquet("/Users/dfalkner/projects/pv-surrogate-eurocast/data/system_data/german_enriched_test_distribution.parquet")
    points_gdf.set_crs(epsg=4326, inplace=True)

    country_shapefile = GeoData.natural_earth_data
    country_name = 'Germany'
    radius_km = 65  # Define the radius in kilometers

    filtered_points = filter_non_overlapping_points_within_country(country_shapefile, country_name, points_gdf, radius_km)

    # Plotting for visualization
    import matplotlib.pyplot as plt

    # plotting
    world = gpd.read_file(country_shapefile).to_crs(epsg=4326)
    _, ax = plt.subplots(figsize=(10, 10))
    ax.set_axis_off()
    world[world['NAME'] == country_name].plot(ax=ax, color='white', edgecolor="black")
    filtered_points.plot(ax=ax, color='red', marker='x', markersize=20, label='Starting Points')

    for x, y, label in zip(filtered_points.geometry.x, filtered_points.geometry.y, filtered_points.sample_id):
        ax.annotate(label, xy=(x, y), xytext=(3, 3), textcoords="offset points")

    plt.legend(frameon=False, loc='upper right')
    plt.savefig(Paths.figure_dir / "starting_points.pdf")


if __name__ == '__main__':
    main()