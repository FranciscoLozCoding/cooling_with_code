"""This module contains dataset building tools."""
import numpy as np
import os
from scipy.interpolate import griddata
from geopy.geocoders import GoogleV3
import pandas as pd
import geopandas as gpd
import shapely
import shapely.vectorized
from shapely import wkt
from tqdm import tqdm
from joblib import Parallel, delayed
import pyproj
from shapely.geometry import Point
from shapely.ops import transform
from geopy.distance import geodesic
import pystac_client
import planetary_computer
from odc.stac import stac_load
from tools.environment import BUFFER_DISTANCES, TARGET_VARIABLE

def interpolate_traffic_volume(uhi_gdf, traffic_gdf, method='nearest'):
  """
  Interpolate traffic volumes at each UHI point using the specified interpolation method.

  args:
    uhi_gdf (GeoDataFrame): GeoDataFrame containing UHI points.
    traffic_gdf (GeoDataFrame): GeoDataFrame containing traffic volume points.
    method (str): Interpolation method to use. Default is 'nearest'.

  returns:
    uhi_gdf (GeoDataFrame): GeoDataFrame with interpolated traffic volume values.
  """

  # We will ensure both GeoDataFrames are in the same CRS (e.g. EPSG: 4326)
  traffic_gdf = traffic_gdf.to_crs(epsg=4326)
  uhi_gdf = uhi_gdf.to_crs(epsg=4326)

  # Extract coordinates (x, y) and the traffic volumes.
  traffic_coords = np.array(list(zip(traffic_gdf.geometry.x, traffic_gdf.geometry.y)))
  traffic_vol = traffic_gdf['avg_vol'].values

  # Get the coordinates for UHI points.
  uhi_coords = np.array(list(zip(uhi_gdf.geometry.x, uhi_gdf.geometry.y)))

  # Interpolate traffic volumes at each UHI point using 'nearest' interpolation.
  uhi_vol_interpolated = griddata(traffic_coords, traffic_vol, uhi_coords, method=method)

  # Add the interpolated traffic volume values to the UHI GeoDataFrame.
  uhi_gdf['traffic_volume'] = uhi_vol_interpolated

  return uhi_gdf

def geocode_intersection_google(row, api_key):
    """
    Geocode an intersection using the Google Maps Geocoding API.

    Parameters:
        row (pd.Series): A row from the traffic dataset.
        api_key (str): Your Google Maps Geocoding API key.

    Returns:
        pd.Series: A Series with 'lat' and 'lon' for the intersection.
    """
    geolocator = GoogleV3(api_key=api_key)

    # Construct the query string. Adjust the format as needed.
    if pd.notnull(row['fromSt']) and pd.notnull(row['street']):
        query = f"{row['street']} & {row['fromSt']}, {row['Boro']}, New York, NY"
    else:
        query = f"{row['street']}, {row['Boro']}, New York, NY"

    try:
        location = geolocator.geocode(query)
        if location:
            return pd.Series({'lat': location.latitude, 'lon': location.longitude})
    except Exception as e:
        print(f"Error geocoding query '{query}': {e}")

    return pd.Series({'lat': None, 'lon': None})

def load_building_footprints_csv(csv_file):
    """
    Load building footprint polygons from a CSV file.

    Parameters:
        csv_file (str): Path to the CSV file containing building footprints with additional attributes.

    Returns:
        GeoDataFrame: A GeoDataFrame of building footprints in EPSG:4326.
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Rename 'the_geom' to 'geometry'
    df = df.rename(columns={'the_geom': 'geometry'})

    # Convert the 'the_geom' column from WKT strings to Shapely geometries.
    df['geometry'] = df['geometry'].apply(wkt.loads)

    # Create a GeoDataFrame using the converted geometry
    buildings_gdf = gpd.GeoDataFrame(df, geometry='geometry')

    # Set the CRS to EPSG:4326
    buildings_gdf = buildings_gdf.set_crs(epsg=4326)

    return buildings_gdf

def buildings_in_buffer(buffer_geom, buildings_gdf, epsg_code_for_meters="EPSG:32618", energy_cols=[]):
    """
    Calculate building density metrics within a buffer.

    The GeoDataFrame is expected to have the following columns:
      - geometry: The building footprint geometry.
      - CNSTRCT_YR: Year the building was constructed.
      - HEIGHTROOF: The height of the roof above ground level.
      - GROUNDELEV: The ground elevation at the building site.

    Parameters:
        buffer_geom (shapely.geometry.Polygon): The buffer geometry in EPSG:4326.
        buildings_gdf (GeoDataFrame): Building footprints (with additional attributes) in EPSG:4326.
        epsg_code_for_meters (str): EPSG code for a metric CRS (default "EPSG:32618" for New York).
        energy_cols (list)(optional): List of energy-related columns to include in the output.

    Returns:
        dict: Contains metrics including:
            - "building_count": Number of buildings intersecting the buffer.
            - "total_building_area_m2": Total area (in m²) of building footprints within the buffer.
            - "building_density": Fraction of the buffer area covered by building footprints.
            - "building_height": Average roof height above ground (from HEIGHTROOF) among buildings with valid values.
            - "ground_elev": Average ground elevation (from GROUNDELEV) among buildings with valid values.
            - "construction_year": Average construction year (from CNSTRCT_YR) among buildings with valid values.
            - "energy_cols" (optional): Average values of specified energy-related columns.
    """

    # Project the building GeoDataFrame and the buffer to the specified metric CRS.
    buildings_metric = buildings_gdf.to_crs(epsg_code_for_meters)
    buffer_metric = gpd.GeoSeries([buffer_geom], crs="EPSG:4326").to_crs(epsg_code_for_meters).iloc[0]

    # Compute the area of the buffer in square meters.
    buffer_area = buffer_metric.area

    # Select buildings that intersect the buffer.
    buildings_in_buf = buildings_metric[buildings_metric.intersects(buffer_metric)]
    building_count = len(buildings_in_buf)

    # Compute the total intersection area between the building footprints and the buffer.
    intersection_area = buildings_in_buf.intersection(buffer_metric).area.sum()
    density = intersection_area / buffer_area if buffer_area > 0 else np.nan

    # Compute average building roof height using valid HEIGHTROOF values.
    valid_heights = buildings_in_buf['HEIGHTROOF'][buildings_in_buf['HEIGHTROOF'].notnull()]
    avg_height = valid_heights.mean() if not valid_heights.empty else 0

    # Compute average ground elevation using valid GROUNDELEV values.
    valid_ground = buildings_in_buf['GROUNDELEV'][buildings_in_buf['GROUNDELEV'].notnull()]
    avg_ground = valid_ground.mean() if not valid_ground.empty else 0

    # Compute average construction year using valid CNSTRCT_YR values.
    valid_years = buildings_in_buf['CNSTRCT_YR'][buildings_in_buf['CNSTRCT_YR'].notnull()]
    avg_year = valid_years.mean() if not valid_years.empty else 0

    # compute average for energy columns
    energy_data = {}
    for col in energy_cols:
        valid_energy = buildings_in_buf[col][buildings_in_buf[col].notnull()]
        avg_energy = valid_energy.mean() if not valid_energy.empty else 0
        energy_data[col] = avg_energy

    return {
        "Building_Count": building_count,
        "Total_Building_Area_m2": intersection_area,
        "Building_Density": density,
        "Building_Height": avg_height,
        "Building_Construction_Year": avg_year,
        "Ground_Elevation": avg_ground,
        **energy_data
    }

def average_band_in_buffer(buffer_geom, xarray, band_name, project_to_utm, project_to_wgs84):
    """
    Calculate the average values of a given band within a circular buffer.

    Parameters:
        buffer_geom (shapely.geometry.Polygon): The buffer in EPSG:4326.
        xarray (xarray.DataArray or xarray.Dataset): Xarray object containing the band.
            It must have 1D coordinate arrays 'latitude' and 'longitude' in EPSG:4326.
        band_name (str): Name of the band to process (e.g., 'NDVI').
        project_to_utm (pyproj.Transformer): Transformer to project points to UTM.
        project_to_wgs84 (pyproj.Transformer): Transformer to project points back to WGS84.

    Returns:
        float: The average band value within the buffer. Returns NaN if no grid cells are found.
    """

    # Extract coordinate arrays from the xarray data.
    # Assumes that the band is available in xarray[band_name]
    lons = xarray[band_name].coords['longitude'].values
    lats = xarray[band_name].coords['latitude'].values

    # Create a 2D meshgrid of longitude and latitude.
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # Create a boolean mask for grid cells that fall inside the buffer.
    mask = shapely.vectorized.contains(buffer_geom, lon_grid, lat_grid)

    # Extract the band values as a NumPy array.
    band_data = xarray[band_name].values

    # Compute the average of the band values inside the buffer.
    # np.nanmean ignores NaN values if they exist.
    if np.any(mask):
        average_val = np.nanmean(band_data[mask])
    else:
        average_val = np.nan

    return average_val

def generate_buffer_dataset(latitudes, longitudes, buffer_radius, traffic_volume, xarray, buildings_gdf, UHI=None, datetimes=None, epsg_code_for_meters="EPSG:32618", energy_cols_to_buffer=[]):
    '''
    Generate a dataset with averaged indices and building density metrics per buffer.

    Parameters:
        latitudes (list): Latitudes of center points (EPSG:4326).
        longitudes (list): Longitudes of center points (EPSG:4326).
        buffer_radius (float): Buffer radius in meters.
        traffic_volume (list): Traffic volume values.
        xarray (xarray.DataArray or Dataset): Contains the indices (e.g., 'NDVI', 'NDBI', etc.).
        buildings_gdf (GeoDataFrame): Building footprints in EPSG:4326.
        UHI (list. optional): List of UHI values.
        datetimes (list, optional): Corresponding datetimes.
        epsg_code_for_meters (str, optional): EPSG code for metric projection (e.g., "EPSG:32618" for New York).
        energy_cols_to_buffer (list, optional): List of energy-related columns in buildings_gdf to include in the output.

    Returns:
        DataFrame: A DataFrame with indices averaged per buffer plus building density metrics.
    '''
    # Set up CRS and transformation functions.
    crs_wgs84 = pyproj.CRS("EPSG:4326")
    crs_utm = pyproj.CRS(epsg_code_for_meters)

    # Create transformation functions.
    project_to_utm = pyproj.Transformer.from_crs(crs_wgs84, crs_utm, always_xy=True).transform
    project_to_wgs84 = pyproj.Transformer.from_crs(crs_utm, crs_wgs84, always_xy=True).transform

    # Helper function to process a single point.
    def process_point(lat, lon):
        # Create the buffer around the point.
        point = Point(lon, lat)
        point_utm = transform(project_to_utm, point)
        buffer_utm = point_utm.buffer(buffer_radius)
        buffer_wgs84 = transform(project_to_wgs84, buffer_utm)

        # Calculate spectral indices.
        spectral_indices = {
            "NDVI": average_band_in_buffer(buffer_wgs84, xarray, 'NDVI', project_to_utm, project_to_wgs84),
            "NDBI": average_band_in_buffer(buffer_wgs84, xarray, 'NDBI', project_to_utm, project_to_wgs84),
            "NDWI": average_band_in_buffer(buffer_wgs84, xarray, 'NDWI', project_to_utm, project_to_wgs84),
            "SI": average_band_in_buffer(buffer_wgs84, xarray, 'SI', project_to_utm, project_to_wgs84),
            "NDMI": average_band_in_buffer(buffer_wgs84, xarray, 'NDMI', project_to_utm, project_to_wgs84),
            "NPCRI": average_band_in_buffer(buffer_wgs84, xarray, 'NPCRI', project_to_utm, project_to_wgs84),
            "Coastal_Aerosol": average_band_in_buffer(buffer_wgs84, xarray, 'Coastal_Aerosol', project_to_utm, project_to_wgs84)
        }

        # Calculate building density metrics using the same buffer.
        building_metrics = buildings_in_buffer(buffer_wgs84, buildings_gdf, epsg_code_for_meters, energy_cols_to_buffer)

        # Return a dictionary
        return {**spectral_indices, **building_metrics}

    # Process all points in parallel.
    results = Parallel(n_jobs=-1)(
        delayed(process_point)(lat, lon)
        for lat, lon in tqdm(zip(latitudes, longitudes), total=len(latitudes), desc="Processing points")
    )

    # Convert list of dicts to DataFrame
    df = pd.DataFrame(results)
    df.insert(0, 'Longitude', longitudes)
    df.insert(1, 'Latitude', latitudes)
    df.insert(2, 'datetime', None if datetimes is None else pd.to_datetime(datetimes))
    df.insert(3, 'UHI', UHI)
    df.insert(4, 'Traffic_Volume', traffic_volume)

    return df

def assign_weather_station(lat, lon):
    '''
    Assign a county (Bronx or Manhattan) based on the closest weather station.

      Parameters:
        lat (float): Latitude of the point (EPSG:4326).
        lon (float): Longitude of the point (EPSG:4326).

      Returns:
        str: 'Bronx' or 'Manhattan' based on the closest weather station.
    '''
    # Define the coordinates for Bronx and Manhattan
    bronx_coords = (40.87248, -73.89352)  # Bronx: (Latitude, Longitude)
    manhattan_coords = (40.76754, -73.96449)  # Manhattan: (Latitude, Longitude)

    # Calculate distance to Bronx and Manhattan using geopy's geodesic function
    distance_bronx = geodesic((lat, lon), bronx_coords).meters
    distance_manhattan = geodesic((lat, lon), manhattan_coords).meters

    # Assign the county based on the shortest distance
    if distance_bronx < distance_manhattan:
        return 'Bronx'
    else:
        return 'Manhattan'
    
# Function to find the closest datetime in the weather data
def find_closest_datetime(row, weather_data):
    '''
    Find the closest datetime in the weather data to the given row's datetime.

    Parameters:
        row (pd.Series): A row from a DataFrame containing a 'datetime' column.
        weather_data (pd.DataFrame): A DataFrame containing the weather data with a 'Date / Time' column.

    Returns:
        pd.Series: The row from weather_data with the closest datetime.
    '''
    # Calculate the absolute time difference between the row's datetime and each weather datetime
    time_diff = abs(weather_data['Date / Time'] - row['datetime'])

    # Find the index of the minimum time difference (closest datetime)
    closest_idx = time_diff.idxmin()

    # Return the row with the closest datetime
    return weather_data.iloc[closest_idx]

# Function to assign weather data based on county and closest datetime
def assign_weather_data(row, weather_manhattan, weather_bronx):
    '''
      Assign weather data based on the county and closest datetime.

      Parameters:
        row (pd.Series): A row from a DataFrame containing 'Latitude' and 'Longitude' columns.
        weather_manhattan (pd.DataFrame): A DataFrame containing weather data for Manhattan.
        weather_bronx (pd.DataFrame): A DataFrame containing weather data for Bronx.

      Returns:
        pd.Series: A row with the closest weather data
    '''
    # Determine which county the row belongs to
    county = assign_weather_station(row['Latitude'], row['Longitude'])

    # Find the closest weather data based on county
    if county == 'Manhattan':
        closest_weather = find_closest_datetime(row, weather_manhattan)
    elif county == 'Bronx':
        closest_weather = find_closest_datetime(row, weather_bronx)
    else:
        # Handle the case where the county is not recognized
        raise ValueError(f"Unknown county: {county}")

    # Return the weather data to merge
    return pd.Series({
        'Air Temp at Surface [degC]': closest_weather['Air Temp at Surface [degC]'],
        'Relative Humidity [percent]': closest_weather['Relative Humidity [percent]'],
        'Avg Wind Speed [m/s]': closest_weather['Avg Wind Speed [m/s]'],
        'Wind Direction [degrees]': closest_weather['Wind Direction [degrees]'],
        'Solar Flux [W/m^2]': closest_weather['Solar Flux [W/m^2]']
    })

def assign_weather_data_avg(row, weather_manhattan, weather_bronx):
    '''
    Assign weather data based on the county and average the values
    from 3:00 pm to 4:00 pm on July 24, 2021.

    Parameters:
      row (pd.Series): A row from a DataFrame containing at least a 'Latitude' and 'Longitude' column.
      weather_manhattan (pd.DataFrame): Weather data for Manhattan, including a 'datetime' column.
      weather_bronx (pd.DataFrame): Weather data for Bronx, including a 'datetime' column.

    Returns:
      pd.Series: A series with the averaged weather data for the specified time period.
    '''
    import pandas as pd

    # Determine which county the row belongs to (assumes you have this helper function)
    county = assign_weather_station(row['Latitude'], row['Longitude'])

    # Select the appropriate weather dataset
    if county == 'Manhattan':
        weather = weather_manhattan
    elif county == 'Bronx':
        weather = weather_bronx
    else:
        raise ValueError(f"Unknown county: {county}")

    # Define the time window: 3:00 pm to 4:00 pm on July 24, 2021
    start_time = pd.Timestamp('2021-07-24 15:00:00')
    end_time = pd.Timestamp('2021-07-24 16:00:00')

    # Filter weather data for this time period
    time_mask = (weather['Date / Time'] >= start_time) & (weather['Date / Time'] <= end_time)
    weather_window = weather.loc[time_mask]

    # Compute the average of the selected weather parameters
    # (Make sure the weather DataFrame has the columns exactly as specified.)
    weather_avg = weather_window.mean()

    # Return the averaged weather values as a Series.
    return pd.Series({
        'Air Temp at Surface [degC]': weather_avg['Air Temp at Surface [degC]'],
        'Relative Humidity [percent]': weather_avg['Relative Humidity [percent]'],
        'Avg Wind Speed [m/s]': weather_avg['Avg Wind Speed [m/s]'],
        'Wind Direction [degrees]': weather_avg['Wind Direction [degrees]'],
        'Solar Flux [W/m^2]': weather_avg['Solar Flux [W/m^2]']
    })

def generate_median(lower_left=(40.75, -74.01), upper_right=(40.88, -73.86), time_window="2021-06-01/2021-09-01", resolution=5, degrees=111320.0):
    """
    Generate a median composite of Sentinel-2 bands and calculate spectral indices.

    Parameters:
        lower_left (tuple): Lower-left corner of the bounding box (latitude, longitude).
        upper_right (tuple): Upper-right corner of the bounding box (latitude, longitude).
        time_window (str): Time window for searching Sentinel-2 data.
        resolution (int): Pixel resolution in meters.
        degrees (float): Number of meters per degree of latitude.
    """
    # Calculate the bounds for doing an archive data search
    # bounds = (min_lon, min_lat, max_lon, max_lat)
    bounds = (lower_left[1], lower_left[0], upper_right[1], upper_right[0])

    # search for images from the Planetary Computer STAC API
    stac = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    search = stac.search(
        bbox=bounds,
        datetime=time_window,
        collections=["sentinel-2-l2a"],
        query={"eo:cloud_cover": {"lt": 20}},
    )
    items = list(search.get_items())
    print('This is the number of scenes that touch our region:',len(items))

    # Define the pixel resolution for the final product
    # Define the scale according to our selected crs, so we will use degrees
    signed_items = [planetary_computer.sign(item).to_dict() for item in items]
    scale = resolution / degrees # degrees per pixel for crs=4326

    # Load the data
    data = stac_load(
        items,
        bands=["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"],
        crs="EPSG:4326", # Latitude-Longitude
        resolution=scale, # Degrees
        chunks={"x": 2048, "y": 2048},
        dtype="uint16",
        patch_url=planetary_computer.sign,
        bbox=bounds
    )

    #compute median
    median = data.median(dim="time").compute()

    # Calculate NDVI for the median composite
    ndvi_median = (median.B08-median.B04)/(median.B08+median.B04)

    # Calculate NDBI for the median composite
    ndbi_median = (median.B11-median.B08)/(median.B11+median.B08)

    # Calculate NDWI for the median composite
    ndwi_median = (median.B03-median.B08)/(median.B03+median.B08)

    # Calculate SI for the median composite
    si_median = (median.B11 - median.B04)/(median.B11 + median.B04)

    # Calculate NDMI for the median composite
    ndmi_median = (median.B08 - median.B11)/(median.B08 + median.B11)

    # Calculate NPCRI for the median composite
    npcri_median = (median.B04 - median.B02) / (median.B04 + median.B02)

    # Add indices to the dataset
    median['NDVI'] = (['latitude', 'longitude'], ndvi_median.values)
    median['NDBI'] = (['latitude', 'longitude'], ndbi_median.values)
    median['NDWI'] = (['latitude', 'longitude'], ndwi_median.values)
    median['SI'] = (['latitude', 'longitude'], si_median.values)
    median['NDMI'] = (['latitude', 'longitude'], ndmi_median.values)
    median['NPCRI'] = (['latitude', 'longitude'], npcri_median.values)
    median['Coastal_Aerosol'] = (['latitude', 'longitude'], median.B01.values)

    return median

def generate_building_gdf(building_csv_file="data/Building_Footprints_With_Add_Attr.csv", lower_left=(40.75, -74.01), 
                          upper_right=(40.88, -73.86), padding=0.0015, 
                          drop_cols=["NAME","BIN","LSTMODDATE", "LSTSTATYPE", "DOITT_ID","FEAT_CODE", 
                                     "SHAPE_AREA", "SHAPE_LEN", "BASE_BBL", "MPLUTO_BBL", "GEOMSOURCE","GLOBALID"],
                          is_energy=False, energy_csv_file="data/nyc_energy_water_performance.csv"):
    """
    Generate a GeoDataFrame of building footprints within a bounding box.

    Parameters:
        building_csv_file (str): Path to the CSV file containing building footprints.
        lower_left (tuple): Lower-left corner of the bounding box (latitude, longitude).
        upper_right (tuple): Upper-right corner of the bounding box (latitude, longitude).
        padding (float): Padding around the bounding box to include more buildings.
        drop_cols (list): Columns to drop from the GeoDataFrame.
        is_energy (bool): Whether to include energy data.
        energy_csv_file (str): Path to the CSV file containing energy data.

    Returns:
        GeoDataFrame: A GeoDataFrame of building footprints within the bounding
                      box with the specified columns dropped.
    """
    # Load the building footprints from the CSV file
    buildings_gdf = load_building_footprints_csv(building_csv_file)

    # Compute the centroid for each building polygon.
    # Then filter the bldg GeoDataFrame based on the bounding box we are using for the City
    buildings_gdf = buildings_gdf[
        (buildings_gdf.geometry.centroid.y >= lower_left[0] + padding) &
        (buildings_gdf.geometry.centroid.y <= upper_right[0] - padding) &
        (buildings_gdf.geometry.centroid.x >= lower_left[1] + padding) &
        (buildings_gdf.geometry.centroid.x <= upper_right[1] - padding)
    ]

    if is_energy:
        energy_df = pd.read_csv(energy_csv_file)

        # List of columns to keep for UHI prediction
        keep_columns = ["NYC Borough, Block and Lot (BBL)", "NYC Building Identification Number (BIN)",
                        "Weather Normalized Site EUI (kBtu/ft²)", "Site Energy Use (kBtu)", "Fuel Oil #2 Use (kBtu)", 
                        "Fuel Oil #4 Use (kBtu)", "Diesel #2 Use (kBtu)", "District Steam Use (kBtu)",
                        "Natural Gas Use (kBtu)", "Electricity Use - Grid Purchase (kBtu)", 
                        "Electricity Use – Generated from Onsite Renewable Systems (kWh)",
                        "Direct GHG Emissions (Metric Tons CO2e)", "Percent of Electricity that is Green Power",
                        "Water Use (All Water Sources) (kgal)"]

        # Drop unnecessary columns
        energy_df = energy_df[keep_columns]

        # fill NA with 0
        energy_df.fillna(0, inplace=True)

        # Ensure both columns are strings for a proper merge
        buildings_gdf["MPLUTO_BBL"] = buildings_gdf["MPLUTO_BBL"].astype(str)
        energy_df["NYC Borough, Block and Lot (BBL)"] = energy_df["NYC Borough, Block and Lot (BBL)"].astype(str)

        # Perform a left join to keep only records from buildings_gdf
        merged_gdf = buildings_gdf.merge(energy_df, left_on="MPLUTO_BBL", right_on="NYC Borough, Block and Lot (BBL)", how="left")

        # replace Not Available with 0 in all cols
        merged_gdf.fillna(0, inplace=True)
        merged_gdf = merged_gdf.replace("Not Available", 0)

        #drop columns we only needed for merging
        merged_gdf.drop(columns=["NYC Building Identification Number (BIN)", "NYC Borough, Block and Lot (BBL)"], inplace=True)

        #new cols
        energy_cols = ["Weather Normalized Site EUI (kBtu/ft²)", "Site Energy Use (kBtu)", "Fuel Oil #2 Use (kBtu)", 
                    "Fuel Oil #4 Use (kBtu)", "Diesel #2 Use (kBtu)", "District Steam Use (kBtu)",
                    "Natural Gas Use (kBtu)", "Electricity Use - Grid Purchase (kBtu)", 
                    "Electricity Use – Generated from Onsite Renewable Systems (kWh)",
                    "Direct GHG Emissions (Metric Tons CO2e)", "Percent of Electricity that is Green Power",
                    "Water Use (All Water Sources) (kgal)"]

        # turn new cols, numeric
        merged_gdf[energy_cols] = merged_gdf[energy_cols].apply(pd.to_numeric, errors='coerce')

        #rename the energy cols
        merged_gdf = merged_gdf.rename(columns={
            "Weather Normalized Site EUI (kBtu/ft²)": "Weather_Normalized_Site_EUI_kBtu_ft2",
            "Site Energy Use (kBtu)": "Site_Energy_Use_kBtu",
            "Fuel Oil #2 Use (kBtu)": "Fuel_Oil_2_Use_kBtu",
            "Fuel Oil #4 Use (kBtu)": "Fuel_Oil_4_Use_kBtu",
            "Diesel #2 Use (kBtu)": "Diesel_2_Use_kBtu",
            "District Steam Use (kBtu)": "District_Steam_Use_kBtu",
            "Natural Gas Use (kBtu)": "Natural_Gas_Use_kBtu",
            "Electricity Use - Grid Purchase (kBtu)": "Electricity_Use_Grid_Purchase_kBtu",
            "Electricity Use – Generated from Onsite Renewable Systems (kWh)": "Electricity_Use_Generated_Onsite_Renewables_kWh",
            "Direct GHG Emissions (Metric Tons CO2e)": "Direct_GHG_Emissions_MetricTons_CO2e",
            "Percent of Electricity that is Green Power": "Percent_Electricity_Green_Power",
            "Water Use (All Water Sources) (kgal)": "Water_Use_All_Sources_kgal"
        })

        buildings_gdf = merged_gdf

    #drop cols we dont need
    buildings_gdf.drop(columns=drop_cols,inplace=True)

    return buildings_gdf

def generate_traffic(traffic_csv_file="data/Automated_Traffic_Volume_Counts.csv", lower_left=(40.75, -74.01), upper_right=(40.88, -73.86),
                     padding=0.0015, boros=['Bronx', 'Manhattan'], geodecoded_file='data/grouped_traffic.csv', 
                     drop_cols=["SegmentID", "RequestID", "WktGeom", "Direction", "toSt", 'Yr', 'M', 'D', 'HH', 'MM', 'datetime_str'],
                     uhi_csv_file="data/Training_data_uhi_index.csv", GOOGLE_GEO_API_KEY=None):
    """
    Generate a GeoDataFrame of traffic volume data within a bounding box.

    Parameters:
        traffic_csv_file (str): Path to the CSV file containing traffic volume data.
        lower_left (tuple): Lower-left corner of the bounding box (latitude, longitude).
        upper_right (tuple): Upper-right corner of the bounding box (latitude, longitude).
        padding (float): Padding around the bounding box to include more traffic locations.
        boros (list): List of boroughs to include in the traffic data.
        geodecoded_file (str): Path to the CSV file containing geocoded traffic data.
        drop_cols (list): Columns to drop from the traffic GeoDataFrame.
        uhi_csv_file (str): Path to the CSV file containing UHI data.
        GOOGLE_GEO_API_KEY (str): Your Google Maps Geocoding API key.

    Returns:
        GeoDataFrame: A GeoDataFrame of traffic volume data within the bounding
                      box combined with the UHI data.
    """
    # Load the traffic dataset
    traffic_df = pd.read_csv(traffic_csv_file)

    # Combine the columns and create a new datetime string column
    traffic_df['datetime_str'] = (
        traffic_df['Yr'].astype(str) + '-' +
        traffic_df['M'].astype(str).str.zfill(2) + '-' +
        traffic_df['D'].astype(str).str.zfill(2) + ' ' +
        traffic_df['HH'].astype(str).str.zfill(2) + ':' +
        traffic_df['MM'].astype(str).str.zfill(2) + ':00'
    )

    # Convert the datetime string column to a pandas datetime object
    traffic_df['datetime'] = pd.to_datetime(traffic_df['datetime_str'], format='%Y-%m-%d %H:%M:%S')

    # drop cols we don't need
    traffic_df.drop(columns=drop_cols, inplace=True)

    # Filter the GeoDataFrame to keep only the rows in boros
    filtered_df = traffic_df[traffic_df['Boro'].isin(boros)]

    # Now group the filtered data by 'Boro', 'street', and 'fromSt'
    # and calculate the average volume for each group.
    grouped_traffic_df = (
        filtered_df.groupby(['Boro', 'street', 'fromSt'])['Vol']
                .mean()
                .reset_index(name='avg_vol')
    )

    #check if file was created already:
    if os.path.isfile(geodecoded_file):
        print('geodecoded_file already exists, loading it...')
        grouped_traffic_df = pd.read_csv('data/grouped_traffic.csv')

        # Convert the 'geometry' column from WKT strings to Shapely geometry objects.
        grouped_traffic_df['geometry'] = grouped_traffic_df['geometry'].apply(wkt.loads)

        traffic_gdf = gpd.GeoDataFrame(grouped_traffic_df, geometry='geometry', crs='EPSG:4326')
    else:
        print('geodecoded_file does not exist, creating it...')
        # Apply the geocoding function to every row of the DataFrame
        key = GOOGLE_GEO_API_KEY
        grouped_traffic_df[['lat', 'lon']] = grouped_traffic_df.apply(lambda row: geocode_intersection_google(row, key), axis=1)

        # convert the lat/lon values to a geometry column.
        grouped_traffic_df['geometry'] = grouped_traffic_df.apply(lambda row: Point(row['lon'], row['lat']), axis=1)

        # Convert the DataFrame into a GeoDataFrame.
        traffic_gdf = gpd.GeoDataFrame(grouped_traffic_df, geometry='geometry', crs='EPSG:4326')

        # Save to csv file
        grouped_traffic_df.to_csv("data/grouped_traffic.csv", index=False)

    # filter the traffic GeoDataFrame based on the bounding box we are using for the City
    traffic_gdf = traffic_gdf[
        (traffic_gdf.geometry.y >= lower_left[0] + padding) &
        (traffic_gdf.geometry.y <= upper_right[0] - padding) &
        (traffic_gdf.geometry.x >= lower_left[1] + padding) &
        (traffic_gdf.geometry.x <= upper_right[1] - padding)
    ]

    # filter out any traffic locations that have a traffic volume of 0
    traffic_gdf = traffic_gdf[traffic_gdf['avg_vol'] != 0]

    # Load the UHI dataset
    uhi_df = pd.read_csv(uhi_csv_file)

    # Create a geometry column from the Longitude and Latitude columns
    uhi_df['geometry'] = uhi_df.apply(lambda row: Point(row['Longitude'], row['Latitude']), axis=1)

    # Convert the DataFrame into a GeoDataFrame with EPSG:4326 (WGS84)
    uhi_gdf = gpd.GeoDataFrame(uhi_df, geometry='geometry', crs='EPSG:4326')

    #apply nearest interpolation
    uhi_gdf = interpolate_traffic_volume(uhi_gdf, traffic_gdf, method='nearest')

    return uhi_gdf

def generate_weather_data(buffer_df, manhattan_weather_csv='data/NY_Mesonet_Weather_Manhattan.csv', bronx_weather_csv='data/NY_Mesonet_Weather_Bronx.csv'):
    """
    Generate weather data for the buffer points.

    Parameters:
        buffer_df (DataFrame): DataFrame containing buffer points with 'Latitude' and 'Longitude' columns.
        manhattan_weather_csv (str): Path to the CSV file containing Manhattan weather data.
        bronx_weather_csv (str): Path to the CSV file containing Bronx weather data.

    Returns:
        DataFrame: DataFrame with weather data assigned to the
                   buffer points based on the closest weather station.
    """
    # Read the weather data for both counties
    weather_manhattan = pd.read_csv(manhattan_weather_csv)
    weather_bronx = pd.read_csv(bronx_weather_csv)

    # Parse out EDT, since both datasets with data time are in the same time zone
    weather_manhattan['Date / Time'] = weather_manhattan['Date / Time'].str.replace(' EDT', '')
    weather_bronx['Date / Time'] = weather_bronx['Date / Time'].str.replace(' EDT', '')

    # Convert the "Date / Time" column to datetime
    weather_manhattan['Date / Time'] = pd.to_datetime(weather_manhattan['Date / Time'])
    weather_bronx['Date / Time'] = pd.to_datetime(weather_bronx['Date / Time'])

    #average the weather variables
    buffer_df[['Air Temp at Surface [degC]',
            'Relative Humidity [percent]',
            'Avg Wind Speed [m/s]',
            'Wind Direction [degrees]',
            'Solar Flux [W/m^2]']] = buffer_df.apply(lambda row: assign_weather_data_avg(row, weather_manhattan, weather_bronx), axis=1)
    
    return buffer_df

def combine_buffer_datasets(train_path, test_path, buffer_distances=BUFFER_DISTANCES):
    """
    Load all buffer datasets for both train and test sets, merging by row index.

    Parameters:
        train_path (str): Directory containing training buffer datasets.
        test_path (str): Directory containing test buffer datasets.
        buffer_distances (list): List of buffer distances to load.

    Returns:
        tuple: A tuple containing the combined training and test datasets.
    """

    train_combined = []
    test_combined = []
    train_shared_columns = [TARGET_VARIABLE]
    test_shared_columns = ['Longitude', 'Latitude']

    for buffer_dist in buffer_distances:
        train_file_path = f'{train_path}/{buffer_dist}m_buffer_dataset.csv'
        test_file_path = f'{test_path}/{buffer_dist}m_buffer_test_dataset.csv'

        # Process Training Data
        if os.path.exists(train_file_path):
            print(f"Loading training data: {train_file_path}")
            train_data = pd.read_csv(train_file_path)

            # Keep shared columns only once
            if not train_combined:
                train_combined.append(train_data[train_shared_columns])

            # Rename features by adding buffer prefix (excluding shared columns)
            renamed_train_data = train_data.drop(columns=train_shared_columns).add_prefix(f"{buffer_dist}m_")
            train_combined.append(renamed_train_data)
        else:
            print(f"Warning: Training file not found for {buffer_dist}m buffer.")

        # Process Test Data
        if os.path.exists(test_file_path):
            print(f"Loading test data: {test_file_path}")
            test_data = pd.read_csv(test_file_path)

            # Keep shared columns only once
            if not test_combined:
                test_combined.append(test_data[test_shared_columns])

            # Rename features by adding buffer prefix (excluding shared columns)
            renamed_test_data = test_data.drop(columns=test_shared_columns).add_prefix(f"{buffer_dist}m_")
            test_combined.append(renamed_test_data)
        else:
            print(f"Warning: Test file not found for {buffer_dist}m buffer.")

    # Concatenate all collected datasets along columns
    if train_combined:
        train_combined = pd.concat(train_combined, axis=1)
    else:
        raise FileNotFoundError("No training buffer datasets were found!")

    if test_combined:
        test_combined = pd.concat(test_combined, axis=1)
    else:
        raise FileNotFoundError("No test buffer datasets were found!")

    print(f"\nFinal Training Dataset Shape: {train_combined.shape}")
    print(f"Final Test Dataset Shape: {test_combined.shape}")

    return train_combined, test_combined
