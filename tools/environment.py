'''This module is for defining global variables for the project.'''

# used for the defining the random state in models
RANDOM_STATE=123

# used for defining the validation size in train_test_split
VALID_SPLIT=0.3

# defines the different buffer distances we used for the project
BUFFER_DISTANCES=[50, 100, 150, 300]

# defines the target variable for the project
TARGET_VARIABLE='UHI'

# defines the selected features for Mixed Buffers ResNet model
MixBuf_ResNet_SelectAttr=['50m_1NPCRI', '100m_Elevation_Wind_X', '150m_Traffic_Volume',
                        '150m_Elevation_Wind_Y', '150m_Humidity_NDVI', '150m_Traffic_NDBI',
                        '300m_SI', '300m_NPCRI', '300m_Coastal_Aerosol', '300m_Total_Building_Area_m2',
                        '300m_Building_Construction_Year', '300m_Ground_Elevation', '300m_Building_Wind_X',
                        '300m_Building_Wind_Y', '300m_Elevation_Wind_Y', '300m_BldgHeight_Count',
                        '300m_TotalBuildingArea_NDVI', '300m_Traffic_NDVI', '300m_Traffic_NDBI',
                        '300m_Building_Aspect_Ratio', '300m_Sky_View_Factor', '300m_Canopy_Cover_Ratio',
                        '300m_GHG_Proxy']