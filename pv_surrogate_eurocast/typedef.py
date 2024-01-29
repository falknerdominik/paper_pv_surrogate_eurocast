from pandera import DataFrameModel


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
    longitud: str
