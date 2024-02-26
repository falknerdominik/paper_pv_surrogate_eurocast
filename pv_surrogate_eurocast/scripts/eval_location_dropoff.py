from pathlib import Path
import numpy as np

import pandas as pd
from shapely.geometry import Point, Polygon
from shapely import affinity
from shapely.ops import transform
import geopandas as gpd
import numpy as np
import math

from pv_surrogate_eurocast.constants import Paths


def find_dropoffs(locations: Path, eta: float):
    results = pd.read_parquet(locations)
    ellipes_points = {}
    # for each sample search the dropoff points per bearing
    for sample_id in results['sample_id'].unique():
        print(f'working on {sample_id}')
        dropoff_set = []
        for bearing in results[results['sample_id'] == sample_id]['bearing'].unique():
            filtered = results[(results['sample_id'] == sample_id) & (results['bearing'] == bearing)]
            sorted = filtered.sort_values(by='MAPE')

            dropoff_point = sorted.iloc[0][['lon', 'lat']]
            dropoff_error = sorted.iloc[0]['MAPE']
            for current_distance in sorted['distance']:
                current_observation = sorted[sorted['distance'] == current_distance]
                current_error = current_observation['MAPE'].iloc[0]
                # calculate if distance is smaller than eta
                if abs(dropoff_error - current_error) < eta:
                    dropoff_error = current_error
                    dropoff_point = (current_observation['lon'], current_observation['lat'])
                dropoff_set.append(dropoff_point)
            ellipes_points[(sample_id, bearing)] = dropoff_set
    return ellipes_points, results['sample_id'].unique(), results['bearing'].unique()


def calculate_ellipse_area(lon_lat_points):
    # Convert lon-lat points to Point geometries
    points = [Point(lon, lat) for lon, lat in lon_lat_points]

    # Calculate the center of the points
    centroid = Point(np.mean([point.x for point in points]), np.mean([point.y for point in points]))

    # Calculate the covariance matrix
    cov = np.cov(np.array([[point.x for point in points], [point.y for point in points]]))

    # Calculate the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(cov)

    # Find the major and minor axes lengths and their orientations
    major_axis_length = 2 * math.sqrt(5.991 * eigenvalues[0])
    minor_axis_length = 2 * math.sqrt(5.991 * eigenvalues[1])
    orientation = math.degrees(math.atan2(eigenvectors[0, 1], eigenvectors[0, 0]))

    # Create the ellipse polygon
    ellipse = Point(centroid.x, centroid.y).buffer(1)
    ellipse = affinity.rotate(ellipse, -orientation, origin='centroid')
    ellipse = affinity.scale(ellipse, major_axis_length / 2, minor_axis_length / 2)

    # Calculate the area of the ellipse
    ellipse_area = ellipse.area
    ellipse_area = ellipse.area / (10**6)

    return ellipse_area

def main():
    dropoffs, sample_ids, bearings = find_dropoffs(Paths.pvgis_outward_data_dir / 'evaluation.parquet', 0.2)
    sum_area = 0
    for sample_id in sample_ids:
        for bearing in bearings:
            if (sample_id, bearing) in dropoffs and len(dropoffs[(sample_id, bearing)]) >= 5:
                try:
                    area = calculate_ellipse_area(dropoffs[(sample_id, bearing)])
                    sum_area += area / len(dropoffs[(sample_id, bearing)])
                except:
                    print(f'Skipped: {(sample_id, bearing)}')
    print(sum_area)
    print(sum_area / len(dropoffs))

    pass

if __name__ == '__main__':
    main()