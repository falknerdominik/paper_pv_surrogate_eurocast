import pandas as pd
import pandera as pa
from pandera import DataFrameModel
from pydantic import BaseModel
from shapely.geometry.point import Point


class PVOutputSystemData(DataFrameModel):
    name: str  # user selected name
    system_DC_capacity_W: int
    address: str
    orientation: str  # orientation as a cardinal direction (N, S, W, E, SW, SE, NW, NE, ...)
    num_outputs: int  # number of recorded outputs
    last_output: str  # last received output
    panel: str  # panel type
    inverter: str  # inverter type
    distance_km: str  # distance from the search location in km
    latitude: str
    longitude: str


class SystemDataSampleSchema(pa.SchemaModel):
    sample_id: int = pa.Field()
    geometry: Point = pa.Field()  # x = longitude, y = latitude
    kwP: float = pa.Field(ge=0)  # Assuming 'kwP' represents power and should be >= 0
    orientation: float = pa.Field()
    tilt: float = pa.Field()
    system_loss: float = pa.Field()
    mounting_place: str = pa.Field()

    class Config:
        strict = True  # Ensure DataFrame only contains the columns defined in the schema


class PVGISConfigurationSchema(pa.SchemaModel):
    sample_id: int = pa.Field()
    lon: float = pa.Field()
    lat: float = pa.Field()
    peakpower: float = pa.Field(ge=0)  # Assuming 'peakpower' represents power and should be >= 0
    aspect: float = pa.Field()
    angle: float = pa.Field()
    mounting: str = pa.Field()
    loss: float = pa.Field()

    class Config:
        strict = True  # Ensure DataFrame only contains the columns defined in the schema


map_system_data_to_pvgis_configuration = {
    # geometry mapping must be done manually
    SystemDataSampleSchema.sample_id: PVGISConfigurationSchema.sample_id,
    SystemDataSampleSchema.kwP: PVGISConfigurationSchema.peakpower,
    SystemDataSampleSchema.orientation: PVGISConfigurationSchema.aspect,
    SystemDataSampleSchema.tilt: PVGISConfigurationSchema.angle,
    SystemDataSampleSchema.system_loss: PVGISConfigurationSchema.loss,
    SystemDataSampleSchema.mounting_place: PVGISConfigurationSchema.mounting,
}


class NormalizedPVGISSchema(pa.SchemaModel):
    ds: pd.Timestamp = pa.Field()
    power: float = pa.Field(ge=0)  # Assuming power should be >= 0
    global_irradiance: float = pa.Field(ge=0)  # Assuming global irradiance should be >= 0
    sun_height: float = pa.Field()
    temperature_at_2_m: float = pa.Field()
    wind_speed_at_10_m: float = pa.Field()
    is_reconstructed: int = pa.Field(ge=0, le=1)  # Assuming binary values 0 or 1

    class Config:
        strict = True  # Ensure DataFrame only contains the columns defined in the schema


map_pvgis_raw_to_normalized = {
    "time": NormalizedPVGISSchema.ds,
    "P": NormalizedPVGISSchema.power,  # "System output" in watts
    # "Global irradiance on the inclined plane (plane of the array)" in W/m2
    "G(i)": NormalizedPVGISSchema.global_irradiance,
    "H_sun": NormalizedPVGISSchema.sun_height,  # "Sun height" in degrees
    "T2m": NormalizedPVGISSchema.temperature_at_2_m,  # "2-m air temperature" in degrees Celsius
    "WS10m": NormalizedPVGISSchema.wind_speed_at_10_m,  # "10-m total wind speed" in m/s
    "Int": NormalizedPVGISSchema.is_reconstructed,  # "1 means solar radiation values are reconstructed"
}


class ConfigurationEntry(BaseModel):
    lat: float
    lon: float
    peakpower: float
    angle: float
    aspect: float
    loss: float = 14.0
    outputformat: str = "json"
    mounting: str = "building"
    startyear: int = 2005
    endyear: int = 2020
    usehorizon: int = 1
    pvcalculation: int = 1
    fixed: int = 1  # we work only with fixed modules
    # we work only with crystalline silicon modules, they make up the majority and no data is available for other types
    pvtechchoice: str = "crystSi"

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "lat": 48.368,
                    "lon": 14.513,
                    "peakpower": 3,
                    "angle": 0,
                    "aspect": 0,
                }
            ]
        }
    }
