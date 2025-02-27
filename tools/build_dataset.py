"""Dataset Building tools for the project."""
import numpy as np
from scipy.interpolate import griddata
from geopy.geocoders import GoogleV3
import pandas as pd
import geopandas as gpd
import shapely
from shapely import wkt
from tqdm import tqdm
from joblib import Parallel, delayed
import pyproj
from shapely.geometry import Point
from shapely.ops import transform
from geopy.distance import geodesic

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

def buildings_in_buffer(buffer_geom, buildings_gdf, epsg_code_for_meters="EPSG:32618"):
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

    Returns:
        dict: Contains metrics including:
            - "building_count": Number of buildings intersecting the buffer.
            - "total_building_area_m2": Total area (in m²) of building footprints within the buffer.
            - "building_density": Fraction of the buffer area covered by building footprints.
            - "building_height": Average roof height above ground (from HEIGHTROOF) among buildings with valid values.
            - "ground_elev": Average ground elevation (from GROUNDELEV) among buildings with valid values.
            - "construction_year": Average construction year (from CNSTRCT_YR) among buildings with valid values.
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

    return {
        "building_count": building_count,
        "total_building_area_m2": intersection_area,
        "building_density": density,
        "building_height": avg_height,
        "building_construction_year": avg_year,
        "ground_elev": avg_ground
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

def generate_buffer_dataset(latitudes, longitudes, buffer_radius, traffic_volume, xarray, buildings_gdf, UHI=None, datetimes=None, epsg_code_for_meters="EPSG:32618"):
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
        ndvi_value  = average_band_in_buffer(buffer_wgs84, xarray, 'NDVI', project_to_utm, project_to_wgs84)
        ndbi_value  = average_band_in_buffer(buffer_wgs84, xarray, 'NDBI', project_to_utm, project_to_wgs84)
        ndwi_value  = average_band_in_buffer(buffer_wgs84, xarray, 'NDWI', project_to_utm, project_to_wgs84)
        si_value    = average_band_in_buffer(buffer_wgs84, xarray, 'SI', project_to_utm, project_to_wgs84)
        ndmi_value  = average_band_in_buffer(buffer_wgs84, xarray, 'NDMI', project_to_utm, project_to_wgs84)
        npcri_value = average_band_in_buffer(buffer_wgs84, xarray, 'NPCRI', project_to_utm, project_to_wgs84)
        ca_value    = average_band_in_buffer(buffer_wgs84, xarray, 'Coastal_Aerosol', project_to_utm, project_to_wgs84)

        # Calculate building density metrics using the same buffer.
        building_metrics = buildings_in_buffer(buffer_wgs84, buildings_gdf, epsg_code_for_meters)

        # Return all results.
        return (ndvi_value, ndbi_value, ndwi_value, si_value, ndmi_value, npcri_value, ca_value,
                building_metrics["building_count"], building_metrics["total_building_area_m2"], building_metrics["building_density"],
                building_metrics["building_height"], building_metrics["building_construction_year"], building_metrics["ground_elev"])

    # Process all points in parallel.
    results = Parallel(n_jobs=-1)(
        delayed(process_point)(lat, lon)
        for lat, lon in tqdm(zip(latitudes, longitudes), total=len(latitudes), desc="Processing points")
    )

    # Unpack results.
    (ndvi_values, ndbi_values, ndwi_values, si_values,
     ndmi_values, npcri_values, ca_values,
     building_counts, building_areas, building_densities,
     building_heights, building_construction_years, ground_elev) = zip(*results)

    # Build the DataFrame.
    df = pd.DataFrame({
        'Longitude': longitudes,
        'Latitude': latitudes,
        'datetime': None if datetimes is None else pd.to_datetime(datetimes),
        'UHI': UHI,
        'NDVI': ndvi_values,
        'NDBI': ndbi_values,
        'NDWI': ndwi_values,
        'SI': si_values,
        'NDMI': ndmi_values,
        'NPCRI': npcri_values,
        'Coastal_Aerosol': ca_values,
        'Building_Count': building_counts,
        'Total_Building_Area_m2': building_areas,
        'Building_Density': building_densities,
        'Building_Height': building_heights,
        'Building_Construction_Year': building_construction_years,
        'Ground_Elevation': ground_elev,
        'Traffic_Volume': traffic_volume
    })
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