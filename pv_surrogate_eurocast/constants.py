from pathlib import Path


class Paths:
    project_dir = Path(__file__).parents[0]

    data_dir = project_dir.parents[0] / "data"
    results = project_dir.parents[0] / "results"
    system_data_dir = data_dir / "system_data"
    figure_dir = data_dir / "figures"

    cache_dir = data_dir / "cache"

    natural_earth_data = cache_dir / "ne_10m_admin_0_countries.zip"

    @staticmethod
    def ensure_directories_exists():
        Paths.cache_dir.mkdir(parents=True, exist_ok=True)
        Paths.data_dir.mkdir(parents=True, exist_ok=True)
        Paths.system_data_dir.mkdir(parents=True, exist_ok=True)
        Paths.results.mkdir(parents=True, exist_ok=True)
        Paths.figure_dir.mkdir(parents=True, exist_ok=True)


class GeoData:
    natural_earth_data_url = "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_0_countries.zip"


class SystemData:
    austrian_systems = Paths.system_data_dir / "pvoutput_austrian_systems.parquet"
    meta_austrian_systems = Paths.system_data_dir / "pvoutput_austrian_systems_meta.parquet"
    german_systems = Paths.system_data_dir / "pvoutput_german_systems.parquet"
    meta_german_systems = Paths.system_data_dir / "pvoutput_german_systems_meta.parquet"


class ModulesNearsGeoshpereReport:
    near_1_km = Paths.results / "modules_near_geoshpere_1km.png"
    near_5_km = Paths.results / "modules_near_geoshpere_5km.png"
    near_10_km = Paths.results / "modules_near_geoshpere_10km.png"
    near_20_km = Paths.results / "modules_near_geoshpere_20km.png"


class Figures:
    system_locations = Paths.figure_dir / "system_locations.png"
