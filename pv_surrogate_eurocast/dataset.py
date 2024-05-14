from dataclasses import dataclass
from enum import Enum
import os
from pathlib import Path
from typing import Protocol

import pandas as pd

from pv_surrogate_eurocast.typedef import NormalizedPVGISSchema

@dataclass
class PVGISDataPackage:
    root: str = Path('./data')
    pvgis_data_path: str = root / 'pvgis'
    fixed_location_pvgis_data_path: str = root / 'pvgis_fixed_location'
    outward_pvgis_data_path: str = root / 'pvgis_outward'
    system_data_path: str = root / 'system_data'


class PVGISDataset(Protocol):
    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index: int):
        raise NotImplementedError

    def get_static_data(self, index: int) -> pd.Series:
        raise NotImplementedError

    def get_all_static_data(self) -> pd.DataFrame:
        raise NotImplementedError


def _read_module_data(path: Path, sample_id: str, column_name: str) -> pd.DataFrame:
        """
        Reads a column from the module data and transforms it into the long format, following standard
        neuralforecast conventions.
        """
        # normalize data to standard long format
        target = pd.read_parquet(path / f"{sample_id}.parquet")
        return (
            target.loc[:, [NormalizedPVGISSchema.ds, column_name]]
            .assign(unique_id=sample_id)
            .rename(columns={column_name: "y"})
        )


def _filter_samples_which_are_not_available(path: Path, metadata: pd.DataFrame) -> pd.DataFrame:
    """
    During the download it might happen that a sample is at the very edge of a country bordering the ocean. This might
    leads to pvgis not having data for this sample. This function filters out all samples which are not available 
    (are around 20 of the 15000 samples in the dataset).
    """
    files = os.listdir(path)
    available_sample_ids = set(file.split('.')[0] for file in files if file.endswith('.parquet'))
    metadata['sample_id'] = metadata['sample_id'].astype(str)

    # filter and reset index to ensure it is from 0 to len(dataframe)
    return metadata[metadata['sample_id'].isin(available_sample_ids)].reset_index(drop=True)


class IncludedData(Enum):
    train = 'train'
    test = 'test'
    all = 'all'


class PVGISGermanyDataset(PVGISDataset):
    """
    Provides access to training data for the PVGIS Germany dataset. The dataset is structured in a way that it can easily be used to train models.
    It does not load all the data into memory at once, but rather loads it on demand. This is done to save memory and to allow for large datasets.

    The dataset is already split into test and train data (ensuring the underlying distribution is followed, see paper). If you want to do your own
    split pass the `IncludedData.all` parameter and split the data yourself.
    """

    def __init__(
            self, 
            data_package_structure: PVGISDataPackage = PVGISDataPackage(),
            module_column_name: str = NormalizedPVGISSchema.power,
            included_data: IncludedData = IncludedData.all,
    ):
        self.data_package = data_package_structure
        self.data_path = data_package_structure.pvgis_data_path
        self.target_column = module_column_name

        match included_data:
            case IncludedData.train:
                self.metadata = pd.read_parquet(data_package_structure.system_data_path / "german_enriched_train_distribution.parquet")
            case IncludedData.test:
                self.metadata = pd.read_parquet(data_package_structure.system_data_path / "german_enriched_test_distribution.parquet")
            case IncludedData.all:
                self.train_metadata = pd.read_parquet(data_package_structure.system_data_path / "german_enriched_train_distribution.parquet")
                self.test_metadata = pd.read_parquet(data_package_structure.system_data_path / "german_enriched_test_distribution.parquet")
                self.metadata = pd.concat([self.train_metadata, self.test_metadata])
            case _:
                raise ValueError(f"Invalid value for included_data: {included_data}!")
        
        # filter not available samples (due to pvgis errors) - these are only a few
        self.metadata = _filter_samples_which_are_not_available(self.data_path, self.metadata)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index: int):
        row = self.metadata.iloc[index]
        target = _read_module_data(self.data_path, row['sample_id'], self.target_column)
        return target
    
    def get_static_data(self, index: int) -> pd.Series:
        return self.metadata.iloc[index]

    def get_all_static_data(self) -> pd.DataFrame:
        return self.metadata


class FixedLocationPVGISGermanyDataset(PVGISDataset):
    """
    This dataset is used to evaluate the models on a fixed location. All the time series here are sampled from a single location only 
    varying the parameters (kwP, tilt, orientation). This is only used for evaluation purposes
    """
    def __init__(
            self, 
            data_package_structure: PVGISDataPackage = PVGISDataPackage(),
            module_column_name: str = NormalizedPVGISSchema.power
    ):
        self.data_package = data_package_structure
        self.data_path = data_package_structure.fixed_location_pvgis_data_path
        self.target_column = module_column_name

        self.metadata = pd.read_parquet(data_package_structure.system_data_path  / "german_fixed_location.parquet")

        # filter not available samples (due to pvgis errors) - these are only a few
        self.metadata = _filter_samples_which_are_not_available(self.data_path, self.metadata)


    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index: int):
        row = self.metadata.iloc[index]
        target = _read_module_data(self.data_path, row['sample_id'], self.target_column)
        return target

    def get_static_data(self, index: int) -> pd.Series:
        return self.metadata.iloc[index]

    def get_all_static_data(self) -> pd.DataFrame:
        return self.metadata


class OutwardPointsPVGISGermanyDataset(PVGISDataset):
    """
    This dataset assesses a model's performance based on longitude and latitude parameters. 
    It does this by establishing a grid of points across Germany and generating additional points extending outward from these. 
    These outward points share identical module parameters but differ in their geographical locations, 
    allowing for an evaluation of the model's responsiveness to location changes.

    The outward points are created by distributing them evenly in five directions from each starting point (indicated by bearings) 
    and moving outward in increments of 5 kilometers.

    File names are expected to be constructed like: f'{meta["sample_id"]}_{meta["bearing"]}_{meta["distance"]}.parquet'
    """
    def __init__(
            self, 
            data_package_structure: PVGISDataPackage = PVGISDataPackage(),
            module_column_name: str = NormalizedPVGISSchema.power
    ):
        self.data_package = data_package_structure
        self.data_path = data_package_structure.outward_pvgis_data_path
        self.target_column = module_column_name

        self.metadata = pd.read_parquet(data_package_structure.system_data_path  / "german_outward_points.parquet")
        self.metadata['id'] = self.metadata['sample_id']
        self.metadata['sample_id'] = self.metadata['sample_id'].astype(str) + '_' + self.metadata['bearing'].astype(str) + '_' + self.metadata['distance'].astype(str)

        # filter not available samples (due to pvgis errors) - these are only a few
        self.metadata = _filter_samples_which_are_not_available(self.data_path, self.metadata)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index: int):
        # return id as string of: f'{config["sample_id"]}_{config["bearing"]}_{config["distance"]}'
        row = self.metadata.iloc[index]
        target = _read_module_data(self.data_path, row["sample_id"], self.target_column)
        return target

    def get_static_data(self, index: int) -> pd.Series:
        return self.metadata.iloc[index]

    def get_all_static_data(self) -> pd.DataFrame:
        return self.metadata


class ChallengePVGISGermanyDataset:
    """
    Add additional system degredation to the dataset. The rate is applied smoothly over the year.
    
    
    The assumed rate is given by: 
    Jordan, D. C., & Kurtz, S. R. (2013). Photovoltaic Degradation Rates-an Analytical Review. Progress in Photovoltaics, 21(1), 12–29. doi:10.1002/pip.1182
    """
    def __init__(self, dataset, yearly_degredation_rate: float = 2.5) -> None:
        self.dataset = dataset
        self.yearly_degradation_rate = yearly_degredation_rate

    def __getitem__(self, index: int):
        target = self.dataset[index]

        hourly_degredation = self.yearly_degradation_rate / 365.0 / 24.0

        # apply degredation
        target['degredation'] = hourly_degredation

        # calculate % of estimated loss
        target['deg_cumsum'] = 1 - target['degredation'].cumsum() / 100.0
        target['y'] = target['y'] * target['deg_cumsum']
        target.drop(columns=['degredation', 'deg_cumsum'], inplace=True)

        return target

    def get_static_data(self, index: int) -> pd.Series:
        return self.dataset.get_static_data(index)

    def get_all_static_data(self) -> pd.DataFrame:
        return self.dataset.get_all_static_data()