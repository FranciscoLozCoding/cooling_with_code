# Overview

**Cooling with Code** aims to address the Urban Heat Island (UHI) effect, a phenomenon where urban areas experience higher temperatures than their rural counterparts due to the lack of vegetation, water bodies, and the dense presence of buildings. The goal of this project is to develop a machine learning model that can predict UHI hotspots in urban areas and understand the key factors contributing to these hotspots. The datasets specifically focus on the Bronx and Manhattan regions in New York City. Although, the aim is for the model to be used in other urban areas not only New York City

This project is part of the **2025 EY Open Science AI & Data Challenge**, which focuses on solving critical climate issues using AI and machine learning to develop sustainable solutions.

# Challenge Goal

The main objective is to develop a micro-scale machine learning model that predicts the locations and severity of the UHI effect. The model will use various datasets, including near-surface air temperatures, building footprint data, weather data, and satellite data, to identify key drivers of UHI. This model will provide insights into urban areas that are most affected by UHI, enabling urban planners and policymakers to take effective mitigation actions.

Model outputs should be in this csv format so that EY can grade our models. [Submission Link](https://challenge.ey.com/challenges/the-2025-ey-open-science-ai-and-data-challenge-cooling-urban-heat-islands-external-participants/submissions).

| Longitude  | Latitude   | UHI Index      |
|------------|-----------|---------------|
| -73.971665 | 40.78876333 | <MODEL_OUTPUT> |


# Team
- Francisco Lozano (flozano2@depaul.edu)
- Dalton Knapp (dknapp7@depaul.edu)
- Adam Zizi (azizi@depaul.edu)

## Directory setup
- **samples**: contains sample notebooks given to us by EY.
- **tools**: contains python scripts we used throughout our project.
- **data**: folder that contains the raw & preprocessed data for the project
   - **data/test**: folder that contains the test datasets. Model Ouputs using this dataset will be turned into EY so that they can grade our model.
   - **data/train**: folder that contains the train datasets. We will use these to train our models
- **images**: containes images used by notebooks for reporting

>NOTE: some of the raw datasets are managed with `git lfs`, refer to this documentation on how to work with `git lfs`: [git-lfs](https://git-lfs.com/), [Github Tutorial](https://github.com/git-lfs/git-lfs/wiki/Tutorial), [Atlassian Tutorial](https://www.atlassian.com/git/tutorials/git-lfs).

# Datasets

The following datasets will be used to develop the model:

1. **Ground-Level Air Temperature Data (UHI Index)**
   - Data collected by CAPA Strategies on July 24, 2021, between 3:00 pm and 4:00 pm, across the Bronx and Manhattan regions.
   - 11,229 data points with UHI Index values indicating the temperature difference from the city’s average temperature.

2. **Building Footprint Data**
   - Available from the Cornell University Geospatial Information Repository.
   - Useful for understanding the impact of building density on local temperatures.

3. **Satellite Data (Sentinel-2 and Landsat)**
   - **Sentinel-2**: Optical data to assess vegetation, water, and urban density.
   - **Landsat**: Provides Land Surface Temperature (LST) data to assess urban heat from the ground surface.

4. **Local Weather Data**
   - Data from New York State Mesonet, including temperature, relative humidity, wind speed, wind direction, and solar flux, collected at two weather stations in the region.

5. **Building Footprint Data with Additional Attributes**
   - Data from NYC Office of Technology and Innovation (OTI)
   - Useful for understanding the impact of buildings on local temperatures.
   - This dataset has more attributes than **Building Footprint Data** such as Building height.

6. **Automated Traffic Volume Counts**
   -  Data from New York City Department of Transportation (NYC DOT)
   - Useful for understanding the impact of traffic on local temperatures.

# Project Updates:

1. https://youtu.be/R66yFI8sWi0 

# References: 
- New York City UHI Index
   - Description: Ground temperature data over New York City on July 24, 2021 (CSV format)
   - Contributors: Climate, Adaptation, Planning, Analytics (CAPA) Strategies
   - Data Host: Center for Open Science - https://www.cos.io
   - Terms of Use: https://github.com/CenterForOpenScience/cos.io/blob/master/TERMS_OF_USE.md
   - License: Apache 2.0 > https://github.com/CenterForOpenScience/cos.io/blob/master/LICENSE
- Satellite Data (Sentinel-2 Sample Output)
   - Description: Copernicus Sentinel-2 sample data from 2021 obtained from the Microsoft Planetary Computer (TIFF format)
   - Contributors: European Space Agency (ESA), Microsoft
   - Data Host: Microsoft Planetary Computer - https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a
   - Terms of Use: https://sentinel.esa.int/documents/247904/690755/Sentinel_Data_Legal_Notice
   - License: https://creativecommons.org/licenses/by-sa/3.0/igo/
- Building Footprint Data
   - Description: Building footprint polygons over the data challenge region of interest (KML format)
   - Contributors: Open Data Team at the NYC Office of Technology and Innovation (OTI)
- New York City Open Data Project
   - Data Host: https://data.cityofnewyork.us/Housing-Development/Building-Footprints/nqwf-w8eh
   - Terms of Use: https://www.nyc.gov/html/data/terms.html and https://www.nyc.gov/home/terms-of-use.page
   - License: https://github.com/CityOfNewYork/nyc-geo-metadata#Apache-2.0-1-ov-file
- Weather Data
   - Description: Detailed weather data collected every 5 minutes at two locations (Bronx and Manhattan). Includes surface air temperature (2-meters), relative humidity, average wind speed, wind direction, and solar flux.
   - Contributors: New York State Mesonet
   - Data Host: https://nysmesonet.org/
   - Terms of Use: https://nysmesonet.org/about/data
   - License: https://nysmesonet.org/documents/NYS_Mesonet_Data_Access_Policy.pdf
- Building Footprint Data with Additional Attributes
   - Description: Building footprints represent the full perimeter outline of each building as viewed from directly above. Additional attribute information maintained for each feature includes: Building Identification Number (BIN); Borough, Block, and Lot information(BBL); ground elevation at building base; roof height above ground elevation; construction year, and feature type.
   - Data Host: [NYC Open Data](https://data.cityofnewyork.us/City-Government/Building-Footprints/5zhs-2jue)
   - repo: https://github.com/CityOfNewYork/nyc-geo-metadata/blob/main/Metadata/Metadata_BuildingFootprints.md 
   - License: https://github.com/CityOfNewYork/nyc-geo-metadata/blob/main/LICENSE
- Automated Traffic Volume Counts
   - Description: New York City Department of Transportation (NYC DOT) uses Automated Traffic Recorders (ATR) to collect traffic sample volume counts at bridge crossings and roadways.These counts do not cover the entire year, and the number of days counted per location may vary from year to year.
   - Data Host: [NYC Open Data](https://data.cityofnewyork.us/Transportation/Automated-Traffic-Volume-Counts/7ym2-wayt)
