from pathlib import Path

import pandas as pd

from pv_surrogate_eurocast.typedef import NormalizedPVGISSchema


def load_static_data(target: str, static_data_path: Path):
    if target == NormalizedPVGISSchema.global_irradiance:
        return (
            pd.read_parquet(static_data_path)
            .loc[:, ["sample_id", "orientation", "tilt"]]
            .rename(columns={"sample_id": "unique_id"})
        )
    else:
        return (
            pd.read_parquet(static_data_path)
            .loc[:, ["sample_id", "kwP", "orientation", "tilt"]]
            .rename(columns={"sample_id": "unique_id"})
        )
