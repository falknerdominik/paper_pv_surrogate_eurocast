import logging
from pathlib import Path

import numpy as np
import pandas as pd
from prefect import flow, task

from pv_surrogate_eurocast.constants import SystemData

logger = logging.getLogger(__name__)


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
    # map and save in dataframe
    mastr_samples["tilt"] = s.map(lambda t: mapper(*t))

    mastr_samples = mastr_samples.dropna()
    mastr_samples["system_loss"] = 14
    mastr_samples["mounting_place"] = "building"

    mastr_samples.to_parquet(target_parquet)

    return target_parquet


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
    samples_target_path = filter_mastr(SystemData.mastr_system_data, SystemData.filtered_mastr_system_data)
    enrich_samples_with_pvoutput(
        samples_target_path, SystemData.meta_german_systems, SystemData.german_system_parameter_distribution
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
