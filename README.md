# Overview

**Cooling with Code** aims to address the Urban Heat Island (UHI) effect, a phenomenon where urban areas experience higher temperatures than their rural counterparts due to the lack of vegetation, water bodies, and the dense presence of buildings. The goal of this project is to develop a machine learning model that can predict UHI hotspots in urban areas and understand the key factors contributing to these hotspots. The datasets specifically focus on the Bronx and Manhattan regions in New York City.

This project is part of the **2025 EY Open Science AI & Data Challenge**, which focuses on solving critical climate issues using AI and machine learning to develop sustainable solutions.

# Challenge Goal

The main objective is to develop a micro-scale machine learning model that predicts the locations and severity of the UHI effect. The model will use various datasets, including near-surface air temperatures, building footprint data, weather data, and satellite data, to identify key drivers of UHI. This model will provide insights into urban areas that are most affected by UHI, enabling urban planners and policymakers to take effective mitigation actions.

# Team
- Francisco Lozano (flozano2@depaul.edu)
- Dalton Knapp (dknapp7@depaul.edu)
- Adam Zizi (azizi@depaul.edu)

## Directory setup
- samples: contains sample notebooks given to us by EY. 
- data: folder that contains the raw & preprocessed data for the project
   - data/test: folder that contains the test datasets. Model Ouputs using this dataset will be turned into EY so that they can grade our model.
   - data/train: folder that contains the train datasets. We will use these to train our models
- images: containes images used by notebooks for reporting

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

# References: 