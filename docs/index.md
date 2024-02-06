# Home

Contains all the code for the eurocast 2024 paper "Surrogates for Fair-Weather Photovoltaic Module Output".

## Sampling from Distribution

## PVGIS Data Download

- Sampling using GeoPandas Grids
- Access via `pvoutput-ofc` library. Downloading metadata
- TODO: Provide Maps for grid data and maybe heatmap based on location

* **KWp**: `Bruttoleistung` in open maestr.
* **Orientation**: Estimated based on PVGIS data. Contains categorical data (Cardinal Directions). Recalculated to degrees (between -90 and 90) using the following mapping:

-90 Degrees: East
-45 Degrees: South-East
0 Degrees: South
45 Degrees: South-West
90 Degrees: West

* **Tilt**: Maybe MeSt maybe pvoutput
* **System Loss**: Hard to estimate / on-going research. Affected by multiple sources (losses in cables, power inverters, dirt (sometimes snow), ...). PVGIS proposes a default value of 14% for fixed systems. We will use this value for now.
* **PV Technology**: The performance depends on the module type (crystalline silicon cells, thin-film cells (CIS, CIGS) or Cadmium Telluride (CdTe)) which depends on the temperature and irradiance effects on the module. 
https://www.nature.com/articles/am201082

Seems CS are by far the most common and actually deployd (90+%). The rest have a smaller share (CdtTe are around 5%). Flexible Panels (thin film) maybe should not be considered (CdTe) as they are usually used for special applications (e.g. on boats, cars, ...).
BEST REF: https://www.sciencedirect.com/science/article/pii/S007964251930101X#b0005

* **Mounting Position**: Assumed fixed as for most residential systems.

Probably limit to austria / germany / (italy) / Gran Canary
* **Longitude**: 
* **Latitude**: 

## Downloaded Data Hosting

???

## Downloading Data from PVGIS

## EX01

## EX02

## EX03