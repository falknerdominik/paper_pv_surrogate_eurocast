import logging
import random
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from prefect import flow, task
from shapely.geometry import Point
from sklearn.model_selection import train_test_split

from pv_surrogate_eurocast.constants import GeoData, Paths, SystemData

logger = logging.getLogger(__name__)

# force matplotlib to not use any Xwindows backend - this crashes in non-gui environments

mpl.use("Agg")


map_mastr_orientation_to_degrees_azimuth = {
    "Nord": 180,
    "Nord-Ost": -135,
    "Nord-West": 135,
    "Ost": -90,
    # TODO
    # 'Ost-West',
    "Süd": 0,
    "Süd-Ost": -45,
    "Süd-West": 45,
    "West": 90,
}

map_mastr_orientation_to_pvoutput_orientation = {
    "Nord": "N",
    "Nord-Ost": "NE",
    "Nord-West": "NW",
    "Ost": "E",
    "Ost-West": "EW",
    "Süd": "S",
    "Süd-Ost": "SE",
    "Süd-West": "SW",
    "West": "W",
}

map_pvoutput_to_degrees_azimuth = {
    "N": 180,
    "NE": -135,
    "NW": 135,
    "E": -90,
    # Will be left out for now
    # 'Ost-West',
    "S": 0,
    "SE": -45,
    "SW": 45,
    "W": 90,
}


@task
def filter_mastr(mastr_solar_path: Path, target_path: Path) -> pd.DataFrame:
    # filters mastr data
    # - Must be a rooftop or facade installation
    # - kwP must be over 0 and below the 95% quantile
    # - Must have a valid orientation and inclination
    mastr = pd.read_parquet(mastr_solar_path)

    mastr = mastr[mastr["Lage"] == "Bauliche Anlagen (Hausdach, Gebäude und Fassade)"]

    # remove not useful categories:'Nachgeführt', 'Fassadenintegriert', None
    mastr = mastr[
        mastr["HauptausrichtungNeigungswinkel"].isin(["20 - 40 Grad", "< 20 Grad", "40 - 60 Grad", "> 60 Grad"])
    ]
    mastr = mastr[
        mastr["Hauptausrichtung"].isin(
            [
                "Nord",
                "Nord-Ost",
                "Nord-West",
                "Ost",
                "Süd",
                "Süd-Ost",
                "Süd-West",
                "West",
                # TODO: 'Ost-West' ??
            ]
        )
    ]
    mastr = mastr[
        mastr["Nutzungsbereich"].isin(
            [
                "Haushalt"
                # , 'Landwirtschaft',
                # 'Gewerbe, Handel und Dienstleistungen', nan, 'Sonstige',
                # 'Industrie', 'Öffentliches Gebäude']
            ]
        )
    ]

    # project only necessary columns
    mastr = mastr[["HauptausrichtungNeigungswinkel", "Bruttoleistung", "Hauptausrichtung"]]

    upper_quantile = mastr["Bruttoleistung"].quantile(0.99)
    lower_quantile = mastr["Bruttoleistung"].quantile(0.01)
    mastr = mastr[mastr["Bruttoleistung"] > lower_quantile]
    mastr = mastr[mastr["Bruttoleistung"] < upper_quantile]

    mastr = mastr.dropna()

    # projections and mappings
    mastr["Hauptausrichtung"] = mastr["Hauptausrichtung"].map(lambda x: map_mastr_orientation_to_degrees_azimuth[x])
    mastr.rename(
        columns={
            "Hauptausrichtung": "orientation",
            "HauptausrichtungNeigungswinkel": "coarse_array_tilt_degrees",
            "Bruttoleistung": "kwP",
        },
        inplace=True,
    )

    mastr.to_parquet(target_path)

    return target_path


def sample_tilt_from_bucket(bucket_name: str, buckets: pd.Series):
    # Filter the DataFrame for the specified bucket
    bucket_data = buckets[buckets["bucket"] == bucket_name]["array_tilt_degrees"]

    # Generate a histogram of the bucket data with a bin size of 1 degree
    counts, bin_edges = np.histogram(bucket_data, bins=range(int(bucket_data.min()), int(bucket_data.max()) + 2))

    # Create a distribution to sample from: each bin edge repeated by its count
    distribution = np.repeat(bin_edges[:-1], counts)

    # Return a random sample from the distribution
    if len(distribution) > 0:
        return np.random.choice(distribution)
    else:
        return None


def build_distribution_sampler_function(data: pd.Series) -> callable:
    counts, bin_edges = np.histogram(data, bins=range(int(data.min()), int(data.max()) + 2))

    # Create a distribution to sample from: each bin edge repeated by its count
    distribution = np.repeat(bin_edges[:-1], counts)

    def random_sampler():
        if len(distribution) > 0:
            return np.random.choice(distribution)
        else:
            return None

    return random_sampler


@task
def enrich_samples_with_pvoutput(
    mastr_samples_parquet: Path, pvoutput_samples_parquet: Path, target_parquet: Path
) -> Path:
    # enriches samples with pvoutput data
    # use the tilt distribution (pvoutput) per bucket (from mastr samples) to generate 'real' tilt values
    # bins: '< 20 Grad', '20 - 40 Grad',  '40 - 60 Grad', '> 60 Grad'
    mastr_samples = pd.read_parquet(mastr_samples_parquet)

    if mastr_samples.shape[0] == 0:
        raise ValueError("No samples found in mastr_samples_parquet. Aborting.")

    # Building the distribution sampler from pvoutput
    pvoutput_samples = pd.read_parquet(pvoutput_samples_parquet)
    pvoutput_samples["coarse_array_tilt_degrees"] = pd.cut(
        pvoutput_samples["array_tilt_degrees"],
        bins=[-np.inf, 20, 40, 60, np.inf],
        labels=["< 20 Grad", "20 - 40 Grad", "40 - 60 Grad", "> 60 Grad"],
    )
    pvoutput_samples = pvoutput_samples[pvoutput_samples["orientation"] != "EW"]
    pvoutput_samples["orientation"] = pvoutput_samples["orientation"].map(lambda x: map_pvoutput_to_degrees_azimuth[x])
    fine_tilt_distributions = pvoutput_samples.groupby(
        ["orientation", "coarse_array_tilt_degrees"], observed=True
    ).apply(lambda x: build_distribution_sampler_function(x["array_tilt_degrees"]))

    sample = mastr_samples.sample().iloc[0].to_dict()
    fine_tilt_distributions[(sample["orientation"], sample["coarse_array_tilt_degrees"])]()

    @np.vectorize()
    def mapper(orientation: int, coarse_array_tilt_degrees: str) -> int:
        if (orientation, coarse_array_tilt_degrees) in fine_tilt_distributions:
            return fine_tilt_distributions[(orientation, coarse_array_tilt_degrees)]()
        else:
            -1

    # build series for mapping the tilt to each sample
    s = pd.Series(tuple(zip(mastr_samples["orientation"], mastr_samples["coarse_array_tilt_degrees"])))
    # this is slow because mapping in python is slow (loops are bad) but it's not worth the effort to optimize for now
    # map and save in dataframe
    mastr_samples["tilt"] = s.map(lambda t: mapper(*t))

    mastr_samples = mastr_samples.dropna()
    mastr_samples["system_loss"] = 14
    mastr_samples["mounting_place"] = "building"

    mastr_samples.to_parquet(target_parquet)

    return target_parquet


def draw_random_points_from_country(
    n,
    natural_earth_data: Path,
    country_name: str,
    random_seed: int = 42,
) -> Path:
    random.seed(random_seed)
    # Filter for a specific country, e.g., Germany
    world = gpd.read_file(natural_earth_data)
    country = world[world["NAME"] == country_name]

    # Get the bounding box of the country
    minx, miny, maxx, maxy = country.total_bounds

    # Generate random points
    samples = []
    while len(samples) < n:
        # Generate a random point within the bounding box
        x = random.uniform(minx, maxx)
        y = random.uniform(miny, maxy)
        point = Point(x, y)
        # Check if the point is within the country's boundary
        if country.geometry.contains(point).any():
            samples.append(point)

    # Create a GeoDataFrame for the random points
    random_points_gdf = gpd.GeoDataFrame(geometry=samples)
    return random_points_gdf.reset_index().rename(columns={"index": "sample_id"})


@task
def generate_random_train_and_test_set(
    n: int,
    country_name: str,
    parameter_distribution_path: Path,
    train_target_path: Path,
    test_target_path: Path,
    figure_path: Path,
) -> tuple[Path, Path]:
    random_points = draw_random_points_from_country(
        n,
        GeoData.natural_earth_data,
        country_name,
    )
    parameters = pd.read_parquet(parameter_distribution_path)
    random_indices = np.random.randint(0, len(parameters), size=len(random_points))
    enriched_sample = pd.concat([random_points, parameters.iloc[random_indices].reset_index(drop=True)], axis=1).drop(
        columns=["coarse_array_tilt_degrees"]
    )

    train, test = train_test_split(enriched_sample, test_size=1 / 3, random_state=42)
    train.to_parquet(train_target_path)
    test.to_parquet(test_target_path)

    plot_compare_ground_truth_to_sample(
        parameters,
        train,
        target_path=figure_path / f"{country_name}_train_sample_comparison_limited.pdf",
        numerical_vars=["kwP", "tilt"],
        categorical_vars=["orientation"],
    )
    plot_compare_ground_truth_to_sample(
        parameters,
        test,
        target_path=figure_path / f"{country_name}_test_sample_comparison_limited.pdf",
        numerical_vars=["kwP", "tilt"],
        categorical_vars=["orientation"],
    )
    return train_target_path, test_target_path


def plot_compare_ground_truth_to_sample(
    ground_truth: pd.DataFrame,
    sample: pd.DataFrame,
    target_path: Path,
    numerical_vars: list = [],
    categorical_vars: list = [],
):
    """
    Plots and compares the distribution of variables between the ground truth and sample DataFrames.

    Args:
        ground_truth (pd.DataFrame): DataFrame containing the ground truth data.
        sample (pd.DataFrame): DataFrame containing the sample data.
        target_path (Path): Path to save the plot.
        numerical_vars (list, optional): List of numerical variables to plot. Default is an empty list.
        categorical_vars (list, optional): List of categorical variables to plot. Default is an empty list.

    Raises:
        ValueError: If there are no variables to plot or if the ground truth
        and sample DataFrames do not have the same columns.

    Returns:
        None
    """
    # Creating overlapping histograms for 'kwP' and 'tilt', and a count plot for 'orientation'
    variable_count = len(numerical_vars) + len(categorical_vars)
    maximum_columns_per_row = 3

    if variable_count == 0:
        raise ValueError("No variables to plot!")

    all_vars = numerical_vars + categorical_vars
    if not np.isin(all_vars, ground_truth.columns).all() or not np.isin(all_vars, sample.columns).all():
        raise ValueError("The ground truth and sample DataFrames do not have the same columns! Please check the input.")

    number_of_columns = min(maximum_columns_per_row, len(numerical_vars) + len(categorical_vars))
    number_of_rows = variable_count // maximum_columns_per_row + (
        1 if variable_count % maximum_columns_per_row > 0 else 0
    )
    _, axs = plt.subplots(number_of_rows, number_of_columns, figsize=(6 * number_of_columns, 6 * number_of_rows))

    # Handle the case of a single subplot (not in a list)
    if number_of_rows * number_of_columns == 1:
        axs = [axs]
    # Flatten the axs array in case of multiple rows and columns
    axs = axs.flatten()

    # Plotting histograms for 'kwP' and 'tilt'
    for i, var in enumerate(numerical_vars):
        sns.kdeplot(ground_truth[var], ax=axs[i], color="skyblue", alpha=0.5, label="Ground Truth", fill=True)
        sns.histplot(sample[var], ax=axs[i], color="orange", alpha=0.5, label="Sample", kde=False, stat="density")
        axs[i].set_title(f"Comparison of {var}")
        axs[i].legend()

    # For the categorical variable 'orientation', plotting counts
    # Adjusting the third plot for 'orientation'
    for i, var in enumerate(categorical_vars, start=len(numerical_vars)):
        sns.countplot(
            x=var, data=ground_truth, ax=axs[i], alpha=0.5, color="skyblue", label="Ground Truth", stat="percent"
        )
        sns.countplot(x=var, data=sample, ax=axs[i], alpha=0.5, color="orange", label="Sample", stat="percent")
        axs[i].set_title(f"Comparison of {var}")
        axs[i].legend()

    plt.tight_layout()
    plt.savefig(str(target_path), bbox_inches="tight", format="pdf")


@task
def draw_sampled_locations_on_map(
    country_name: str,
    natural_earth_data: Path,
    train_points: gpd.GeoDataFrame,
    test_points: gpd.GeoDataFrame,
    target_path: Path,
):
    # Filter for a specific country, e.g., Germany
    world = gpd.read_file(natural_earth_data)
    country = world[world["NAME"] == country_name]
    train_points = train_points.set_crs(country.crs).to_crs(epsg=4326)
    test_points = test_points.set_crs(country.crs).to_crs(epsg=4326)
    country = country.to_crs(epsg=4326)

    # Plot the country's boundary
    _, ax = plt.subplots(figsize=(10, 10))
    country.plot(ax=ax, color="white", edgecolor="black")

    # Plot the random points
    train_points["geometry"].plot(ax=ax, color="skyblue", markersize=2, label="Train Set")
    test_points["geometry"].plot(ax=ax, color="orange", markersize=3, label="Test Set")

    ax.set_axis_off()
    plt.legend()

    plt.savefig(str(target_path), bbox_inches="tight", format="pdf")


@flow
def main():
    # TODO: project initial data for faster loading, do this programmatically instead of manually
    # d[
    # [
    # 'Lage', 'Hauptausrichtung', 'Einspeisungsart', 'Einheittyp', 'Bruttoleistung', 'Registrierungsdatum',
    # 'Postleitzahl', 'Laengengrad', 'Breitengrad', 'Nettonennleistung', 'FernsteuerbarkeitNb', 'FernsteuerbarkeitDv',
    # 'FernsteuerbarkeitDr', 'EinheitlicheAusrichtungUndNeigungswinkel', 'GemeinsamerWechselrichterMitSpeicher',
    # 'HauptausrichtungNeigungswinkel','Nutzungsbereich'
    # ]
    # ].to_parquet(SystemData.mastr_system_data)

    # samples_target_path = filter_mastr(SystemData.mastr_system_data, SystemData.filtered_mastr_system_data)
    # distribution_path = enrich_samples_with_pvoutput(
    #     samples_target_path, SystemData.meta_german_systems, SystemData.german_system_parameter_distribution
    # )

    # CAREFUL: This will take a new random sample!
    # distribution_path = SystemData.german_system_parameter_distribution
    # generate_random_train_and_test_set(
    #     3000,
    #     "Germany",
    #     distribution_path,
    #     SystemData.german_enriched_train_distribution,
    #     SystemData.german_enriched_test_distribution,
    #     Paths.figure_dir,
    # )

    draw_sampled_locations_on_map(
        "Germany",
        GeoData.natural_earth_data,
        gpd.read_parquet(SystemData.german_enriched_train_distribution)[0:2000],
        gpd.read_parquet(SystemData.german_enriched_test_distribution)[0:1000],
        Paths.figure_dir / "sampled_locations.pdf",
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
