import gradio as gr
import shap
from model import UhiPredictor
import numpy as np
import pandas as pd
import plotly.graph_objects as go

ref_data = pd.read_parquet("UHI_explainer_ref_data.parquet")
cols = pd.read_parquet("UHI_explainer_ref_data.parquet").columns
MODEL = UhiPredictor("mixed_buffers_ResNet_model.keras", "mixed_buffers_standard_scaler.pkl", shap.DeepExplainer, ref_data, cols)

def filter_map(uhi, longitude, latitude):
    '''
    This function generates a map based on uhi prediction
    '''
    #set up custom data
    data = [[uhi, longitude, latitude]]

    # Create the plot
    fig = go.Figure(go.Scattermapbox(
        lat=[latitude],
        lon=[longitude],
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=10
        ),
        hoverinfo="text",
        hovertemplate='<b>UHI Index</b>: %{customdata[0]}<br><b>long</b>: %{customdata[1]}<br><b>lat</b>: %{customdata[2]}<br>',
        customdata=data
    ))

    fig.update_layout(
        mapbox_style="open-street-map",
        hovermode='closest',
        mapbox=dict(
            bearing=0,
            center=go.layout.mapbox.Center(
                lat=40.7638,
                lon=-74.0060  # Default to New York City for initial view
            ),
            pitch=0,
            zoom=10
        ),
    )

    return fig

def predict(
        longitude, latitude, m150_NPCRI, m100_Ground_Elevation, avg_wind_speed, 
        wind_direction_deg, traffic_volume, m150_Ground_Elevation, 
        relative_humidity, m150_NDVI, m150_NDBI, 
        m300_SI, m300_NPCRI, m300_Coastal_Aerosol, 
        m300_Total_Building_Area_m2, m300_Building_Construction_Year, m300_Ground_Elevation, 
        m300_Building_Height, m300_Building_Count, m300_NDVI,
        m300_NDBI, m300_Building_Density, solar_flux
    ):
    '''
    Predict the UHI index for the data inputed, Longitude and Latitude are used to generate a map
    and do not affect the UHI index prediction.
    '''

    # Create a dictionary with input data and dataset var names
    input_data = {
        "150m_NPCRI": m150_NPCRI,
        "100m_Ground_Elevation": m100_Ground_Elevation,
        "Avg_Wind_Speed": avg_wind_speed,
        "Wind_Direction_deg": wind_direction_deg,
        "Traffic_Volume": traffic_volume,
        "150m_Ground_Elevation": m150_Ground_Elevation,
        "Relative_Humidity": relative_humidity,
        "150m_NDVI": m150_NDVI,
        "150m_NDBI": m150_NDBI,
        "300m_SI": m300_SI,
        "300m_NPCRI": m300_NPCRI,
        "300m_Coastal_Aerosol": m300_Coastal_Aerosol,
        "300m_Total_Building_Area_m2": m300_Total_Building_Area_m2,
        "300m_Building_Construction_Year": m300_Building_Construction_Year,
        "300m_Ground_Elevation": m300_Ground_Elevation,
        "300m_Building_Height": m300_Building_Height,
        "300m_Building_Count": m300_Building_Count,
        "300m_NDVI": m300_NDVI,
        "300m_NDBI": m300_NDBI,
        "300m_Building_Density": m300_Building_Density,
        "Solar_Flux": solar_flux
    }
    
    # Convert to DataFrame
    input_df = pd.DataFrame(input_data, index=[0])

    #predict
    output = MODEL.predict(input_df)

    # generate map
    plot = filter_map(output["predicted_uhi_index"], longitude, latitude)

    return float(output["predicted_uhi_index"]) , output["uhi_status"], output["feature_contributions"], plot

def load_examples(csv_file):
    '''
    Load examples from csv file
    '''
    # Read examples from CSV file
    df = pd.read_csv(csv_file)

    # Convert DataFrame to a list of lists
    examples = df.values.tolist()

    return examples

def load_interface(): 
    '''
    Configure Gradio interface
    '''

    #set blocks
    info_page = gr.Blocks()

    with info_page:
        # set title and description
        gr.Markdown(
        """
        # ResNet model for Predicting Urban Heat Island (UHI) Index
        
        **Contributors**: Francisco Lozano, Dalton Knapp, Adam Zizi\n
        **University**: Depaul University\n

        ## Overview
        Our project focused on creating a micro-scale machine learning model that predicts the locations and severity of the UHI effect.
        The model used various datasets, including near-surface air temperatures, building footprint data, weather data, and 
        satellite data, to identify key drivers of UHI. This model provides insights into urban areas that are most affected by UHI, 
        enabling urban planners and policymakers to take effective mitigation actions. This demo only showcases the ResNet model we created, to fully
        deploy this model we would integrate it with satellite imagery and weather data to provide real-time predictions.\n

        ## How to Use
        To use the model, input the required parameters in the fields provided or select an example from the table. The model will predict the UHI index based on the inputs.
        The UHI index is a measure of the intensity of the Urban Heat Island effect, with values > 1 indicating a UHI effect.
        The model will also provide insights into the contributions of each feature to the UHI index prediction, as well as a map showing the location
        of the prediction based on the longitude and latitude inputs. The predicted UHI index and the status (Urban Heat Island or Cooler Region) will be displayed.\n
        >NOTE: The longitude and latitude inputs are used to identify the location of the prediction, but they do not affect the UHI index prediction.\n

        ## Metrics
        For a quick look of our model performance, here is its r2 score:
        ```
        Train r2 score:  0.9853446142702935
        Test r2 score:   0.9625555699901284
        ```

        ## Repository
        The code for this project is available on GitHub. It includes our Jupyter notebooks and Final Report.\n
        [Project Repo](https://github.com/FranciscoLozCoding/cooling_with_code)
        """
        )
    
    # set inputs and outputs for the model
    longitude = gr.Number(label="Longitude", precision=5, info="The Longitude of the location")
    latitude = gr.Number(label="Latitude", precision=5, info="The Latitude of the location")
    m150_NPCRI = gr.Number(label="150m NPCRI", precision=5, info="The average Normalized Difference Vegetation Index in a 150m Buffer Zone. NPCRI is a remote sensing index designed to estimate the chlorophyll content in vegetation.")
    m100_Ground_Elevation = gr.Number(label="100m Ground Elevation", precision=5, info="The average Ground Elevation in a 100m Buffer Zone")
    avg_wind_speed = gr.Number(label="Avg Wind Speed [m/s]", precision=5, info="The average Wind Speed [m/s] at the location")
    wind_direction = gr.Number(label="Wind Direction [degrees]", precision=5, info="The average Wind Direction [degrees] at the location")
    traffic_volume = gr.Number(label="Traffic Volume", precision=5, info="The Traffic Volume at the location")
    m150_Ground_Elevation = gr.Number(label="150m Ground Elevation", precision=5, info="The average Ground Elevation in a 150m Buffer Zone")
    relative_humidity = gr.Number(label="Relative Humidity [percent]", precision=5, info="The average Relative Humidity [percent] at the location")
    m150_NDVI = gr.Number(label="150m NDVI", precision=5, info="The average Normalized Difference Vegetation Index in a 150m Buffer Zone. NDVI is used to measure the greenness of vegetation.")
    m150_NDBI = gr.Number(label="150m NDBI", precision=5, info="The average Normalized Difference Built-up Index in a 150m Buffer Zone. NDBI is a ratio-based index to highlight built-up areas or areas of urbanization")
    m300_SI = gr.Number(label="300m SI", precision=5, info="The average Shadow Index in a 300m Buffer Zone. SI helps in identifying areas where shadows occur.")
    m300_NPCRI = gr.Number(label="300m NPCRI", precision=5, info="The average Normalized Pigment Chlorophyll Ratio Index in a 300m Buffer Zone. NPCRI is a remote sensing index designed to estimate the chlorophyll content in vegetation.")
    m300_Coastal_Aerosol = gr.Number(label="300m Coastal Aerosol", precision=5, info="The average Coastal Aerosol in a 300m Buffer Zone. Coastal aerosol refers to aerosol particles (tiny solid or liquid particles suspended in the atmosphere) that are found in or around coastal regions.")
    m300_Total_Building_Area_m2 = gr.Number(label="300m Total Building Area(m2)", precision=5, info="The Total Building Area in a 300m Buffer Zone")
    m300_Building_Construction_Year = gr.Number(label="300m Building Construction Year", precision=5, info="The average Building Construction Year in a 300m Buffer Zone")
    m300_Ground_Elevation = gr.Number(label="300m Ground Elevation", precision=5, info="The average Ground Elevation in a 300m Buffer Zone")
    m300_Building_Height = gr.Number(label="300m Building Height", precision=5, info="The average Building Height in a 300m Buffer Zone")
    m300_Building_Count = gr.Number(label="300m Building Count", precision=5, info="The average Building Count in a 300m Buffer Zone")
    m300_NDVI = gr.Number(label="300m NDVI", precision=5, info="The average Normalized Difference Vegetation Index in a 300m Buffer Zone. NDVI is used to measure the greenness of vegetation.")
    m300_NDBI = gr.Number(label="300m NDBI", precision=5, info="The average Normalized Difference Built-up Index in a 300m Buffer Zone. NDBI is a ratio-based index to highlight built-up areas or areas of urbanization")
    m300_Building_Density = gr.Number(label="300m Building Density", precision=5, info="The average Building Density in a 300m Buffer Zone")
    solar_flux = gr.Number(label="Solar Flux [W/m^2]", precision=5, info="The average Solar Flux [W/m^2] at the location")
    inputs = [longitude, latitude, m150_NPCRI, m100_Ground_Elevation, avg_wind_speed, wind_direction, 
              traffic_volume, m150_Ground_Elevation, relative_humidity, m150_NDVI, 
              m150_NDBI, m300_SI, m300_NPCRI, m300_Coastal_Aerosol, m300_Total_Building_Area_m2,
              m300_Building_Construction_Year, m300_Ground_Elevation, m300_Building_Height, m300_Building_Count,
              m300_NDVI, m300_NDBI, m300_Building_Density, solar_flux]
    uhi = gr.Number(label="Predicted UHI Index", precision=5)

    # set model explainer outputs
    uhi_label = gr.Label(label="Predicted Status based on UHI Index")
    feature_contributions = gr.JSON(label="Feature Contributions")

    # Urban Location
    plot = gr.Plot(label="Urban Location")

    model_page = gr.Interface(
        predict, 
        inputs=inputs, 
        outputs=[uhi, uhi_label, feature_contributions, plot],
        examples=load_examples("examples.csv"),
        cache_examples=True,
        cache_mode='lazy',
        title="Interact with The ResNet UHI Model",
        description="This model predicts the Urban Heat Island (UHI) index based on various environmental and urban factors. Adjust the inputs to see how they affect the UHI index prediction.",
    )

    iface = gr.TabbedInterface(
        [info_page, model_page],
        ["Information", "UHI Model"]
    )
    
    iface.launch(server_name="0.0.0.0", server_port=7860, allowed_paths=["/"], share=True)

if __name__ == "__main__":
    load_interface()
