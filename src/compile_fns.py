import os
import json
import gc
import difflib
import string
import nltk
import requests
import numpy as np
import pandas as pd
import ipywidgets as widgets
import matplotlib.pyplot as plt
import geopandas as gpd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from pykrige.ok import OrdinaryKriging
from shapely.geometry import LineString, Polygon, MultiLineString, Point
from shapely.vectorized import contains
from pyproj import Geod

feature_dict = {
    "A1-A25": [[0, 3, 6], ["Total", "Rural", "Urban"]],
    "B.1": [[1, 4, 7], ["Total", "Male", "Female"]],
    "B.2-B.13": [[0, 4], ["Primary", "Upper Primary"]],
    "C.1-C.2": [[0, 3, 6], ["Total", "Rural", "Urban"]],
    "C.3-C.5": [
        [0, 1, 3, 4, 6, 7],
        ["pct_Total", "cnt_Total", "pct_Rural", "cnt_Rural", "pct_Urban", "cnt_Urban"],
    ],
    "D.1-D.2": [[0, 3, 6], ["Total", "Rural", "Urban"]],
    "D.3-D.5": [
        [0, 1, 3, 4, 6, 7],
        ["pct_Total", "cnt_Total", "pct_Rural", "cnt_Rural", "pct_Urban", "cnt_Urban"],
    ],
    "E.1-E.3": [[0, 3, 6], ["Total", "Rural", "Urban"]],
}

invalid_districts = ["Mirpur", "Muzaffarabad"]

state_mapping = {
    "Telangana": ["Telangana"],
    "Uttar Pradesh": ["Uttar Pradesh"],
    "Maharashtra": ["Maharashtra"],
    "Mizoram": ["Mizoram"],
    "Rajasthan": ["Rajasthan"],
    "Kerala": ["Kerala"],
    "West Bengal": ["West Bengal"],
    "Uttarakhand": ["Uttarakhand"],
    "Haryana": ["Haryana"],
    "Punjab": ["Punjab"],
    "Andhra Pradesh": ["Andhra Pradesh"],
    "Jammu & Kashmir": ["Jammu & Kashmir", "Ladakh"],
    "Bihar": ["Bihar"],
    "Tamilnadu": ["Tamil Nadu"],
    "Karnataka": ["Karnataka"],
    "Assam": ["Assam"],
    "Chhattisgarh": ["Chhatisgarh"],
    "Himachal Pradesh": ["Himachal Pradesh"],
    "Manipur": ["Manipur"],
    "Jharkhand": ["Jharkhand"],
    "Nct Of Delhi": ["NCT of Delhi"],
    "Tripura": ["Tripura"],
    "Nagland": ["Nagaland"],
    "Meghalaya": ["Meghalaya"],
    "Sikkim": ["Sikkim"],
    "Puducherry": ["Pondicherry"],
    "Andaman And Nicobar Islands": ["Andaman & Nicobar Islands"],
    "Odisha": ["Orissa"],
    "Madhya Pradesh": ["Madhya Pradesh"],
    "Gujarat": ["Gujarat"],
    "Arunachal Pradesh": ["Arunachal Pradesh"],
    "Chandigarh": ["Chandigarh"],
    "Dadra And Nagar Haveli": ["Dadra & Nagar Haveli"],
    "Daman And Diu": ["Daman & Diu"],
    "Lakshadweep": ["Lakshadweep"],
    "Goa": ["Goa"],
}
state_mapping = {key: state_mapping[key] for key in sorted(state_mapping.keys())}

district_splits = {
    "Jalpaiguri": ["Jalpaiguri, Alipurduar"],
    "North East Delhi": ["North East Delhi", "Shahdara"],
    "Shaja": ["Shajapur, AgarMalwa"],
    "Tirap": ["Tirap, Longding"],
}

indicators_dict = {
    "hazard": {
        "Severe cyclones": [
            "severe_cyclones",
            0,
            -1,
            "H",
        ],
        # "Total cyclones": ["total_cyclones", 0, -1, "H"],
        "Wind speed (knots)": ["wind_speed_knots", 0, -1, "H"],
        "Maximum storm surge (m)": ["pmss_m", 0, -1, "H"],
        "Maximum precipitation (cm)": ["pmp_cm", 0, -1, "H"],
    },
   
    "critical_infrastructure": {
        # "Road length (km)": ["roadlen_km", 0, -1, "R"],
        "Highway density (km per 100 sqkm)": ["highway_per_100sqkm", 0, -1, "R"],
        "Road density (km per 100 sqkm)": ["roads_100sqkm", 0, -1, "R"],
        # "Power line length (km)": ["powerlen_km", 0, -1, "R"],
        "Transmission line density (km per 100 sqkm)": [
            "transmission_100sqkm",
            0,
            -1,
            "R",
        ],
        "Percentage groundwater extraction (%)": [
            "perc_extraction",
            0,
            -1,
            "V",
        ],
    },
    "infrastructure_accessibility": {
        "Households with electricity (%)": ["perc_hh_electricity", 0, 100, "R"],
        "Households with improved sanitation facility (%)": [
            "perc_hh_sanitation",
            0,
            100,
            "R",
        ],
        "Households with improved drinking water source (%)": [
            "perc_hh_drinkingwater",
            0,
            100,
            "R",
        ],
        "Households with clean fuel for cooking (%)": [
            "perc_hh_cleanfuel",
            0,
            100,
            "R",
        ],
    },
    "social_infrastructure": {
        "Police stations (nos. per 1,00,000 pop)": [
            "police_density",
            0,
            -1,
            "R",
        ],
        "Schools (nos. per 1,00,000 pop)": [
            "schools_density",
            0,
            -1,
            "R",
        ],
        "Primary healthcare (nos. per 1,00,000 pop)": [
            "pri_health_density",
            0,
            -1,
            "R",
        ],
        "Secondary & tertiary healthcare (nos. per 1,00,000 pop)": [
            "sec_health_density",
            0,
            -1,
            "R",
        ],
        # "Households with health insurance/financial schemes (%)": [
        #     "perc_social_security",
        #     0,
        #     100,
        #     "R",
        # ],
    },
    
     "community": {
        "Per capita income (INR '000)": ["per_capita_income", 0, -1, "R"],
        # "Population <15 yrs (%)": ["perc_minors", 0, 100, "V"],
        "Women (age 15-49) who are literate (%)": ["perc_women_literate", 0, 100, "R"],
        # "Fully-vaccinated children aged 12-23 months (%)": [
        #     "perc_child_vaccinated",
        #     0,
        #     100,
        #     "R",
        # ],
        "Underweight children <5 yrs (%)": [
            "perc_child_uweight",
            0,
            100,
            "V",
        ],
        "Women (age 15-49) with Low BMI (%)": [
            "perc_women_uweight",
            0,
            100,
            "V",
        ],
        "Women (age 15-49) who are anaemic (%)": [
            "perc_women_aneamic",
            0,
            100,
            "V",
        ],
        # "Men age 15 years and above with Moderately or severely elevated blood pressure (Systolic ≥160 mm of Hg and/or Diastolic ≥100 mm of Hg) (%)": [
        #     "perc_men_highbp",
        #     0,
        #     100,
        #     "V",
        # ],
    },
    # "state_capacity": {
    #     "Citizen centric governance": ["citizen_governance", 0, 1, "R"],
    #     "Judiciary and public security": ["public_security", 0, 1, "R"],
    #     "Commerce and industry": ["commerce_industry", 0, 1, "R"],
    #     "Human resource development": ["hr_devt", 0, 1, "R"],
    #     "Public infrastructure and utilities": ["public_infra", 0, 1, "R"],
    #     "Public health": ["public_health", 0, 1, "R"],
    #     "Economic governance": ["economic_governance", 0, 1, "R"],
    #     "Social welfare": ["social_welfare", 0, 1, "R"],
    #     "Agriculture and allied activities": ["agriculture_allied", 0, 1, "R"],
    #     "Environment governance": ["env_governance", 0, 1, "R"],
    #     # "Composite score for governance": ["governance_composite", 0, 10, "R"],
    # },
    # "state_capacity_social": {
    #     "Citizen centric governance": ["citizen_governance", 0, 1, "R"],
    #     "Judiciary and public security": ["public_security", 0, 1, "R"],
    #     "Human resource development": ["hr_devt", 0, 1, "R"],
    #     "Public health": ["public_health", 0, 1, "R"],
    #     "Social welfare": ["social_welfare", 0, 1, "R"],
    # },
    # "state_capacity_economic": {
    #     "Commerce and industry": ["commerce_industry", 0, 1, "R"],
    #     "Public infrastructure and utilities": ["public_infra", 0, 1, "R"],
    #     "Economic governance": ["economic_governance", 0, 1, "R"],
    #     "Agriculture and allied activities": ["agriculture_allied", 0, 1, "R"],
    # },
}

indicators_key_dict = {
    "hazard": "Hazard",
    "critical_infrastructure": "Critical infrastructure\navailability",
    
    "infrastructure_accessibility": "Household essential\nutility access",
    "social_infrastructure": "Social infrastructure\ndensity",
    "community": "Community\nvulnerability",
    # "state_capacity": "State capacity",
    # "state_capacity_social": "State capacity (social)",
    # "state_capacity_economic": "State capacity (economic)",
}


def combine_district_df_with_gdf(
    df,
    gdf,
    df_state_field,
    gdf_state_field,
    df_dist_field,
    gdf_dist_field,
    alternate_names_df=None,
):
    merged_data, unmatched_dists, states_included, only_complete_data = [], [], [], []

    for state in sorted(gdf[gdf_state_field].unique()):
        state_in_df = None
        if alternate_names_df is not None:
            if state in alternate_names_df["gdf_state"].unique():
                alternate_state_df = alternate_names_df[
                    alternate_names_df["gdf_state"] == state
                ]
                if not alternate_state_df["df_state"].empty:
                    state_in_df = alternate_state_df["df_state"].iloc[0]
        else:
            state_in_df = find_matching_entity(
                state, df[df_state_field].unique(), cutoff=0.85
            )
        if state_in_df:
            states_included.append(state_in_df)
            state_df = df[df[df_state_field] == state_in_df]
            state_gpd = gdf[gdf[gdf_state_field] == state]

            state_df_dists = state_df[df_dist_field].tolist()

            for _, gpd_row in state_gpd.iterrows():
                # print(gpd_row[gdf_dist_field])
                matching_dist = None
                if alternate_names_df is not None:
                    if state in alternate_names_df["gdf_state"].unique():
                        alternate_dist_df = alternate_names_df[
                            alternate_names_df["gdf_state"] == state
                        ]
                        matching_rows = alternate_dist_df[
                            alternate_dist_df["gdf_district"] == gpd_row[gdf_dist_field]
                        ]
                        if not matching_rows.empty:
                            if matching_rows["df_district"].iloc[0] is not np.nan:
                                matching_dist = (
                                    matching_rows["df_district"].iloc[0].split(",")[0]
                                )

                if matching_dist is None:
                    matching_dist = find_matching_entity(
                        gpd_row[gdf_dist_field], state_df_dists, cutoff=0.85
                    )

                if matching_dist:
                    print(
                        f"{state},{state_in_df},{gpd_row[gdf_dist_field]},{matching_dist}"
                    )
                    df_row = state_df[state_df[df_dist_field] == matching_dist].iloc[0]
                    merged_row = gpd_row.to_dict()
                    if gpd_row[gdf_dist_field] not in invalid_districts:
                        for key, value in df_row.items():
                            if key not in merged_row:
                                merged_row[key] = value

                    only_complete_data.append(merged_row.copy())

                else:
                    print(f"{state},{state_in_df},{gpd_row[gdf_dist_field]},")
                    unmatched_dists.append((gpd_row[gdf_dist_field], state))
                    merged_row = gpd_row.to_dict()

                merged_data.append(merged_row)
        else:
            state_gpd = gdf[gdf[gdf_state_field] == state]

            for _, gpd_row in state_gpd.iterrows():
                # print(f"{state},{state_in_df},{gpd_row[gdf_dist_field]},")
                merged_data.append(gpd_row.to_dict())

    # Create a new GeoDataFrame with the merged data
    if isinstance(gdf, gpd.GeoDataFrame):
        gdf_result = gpd.GeoDataFrame(merged_data, geometry="geometry", crs=gdf.crs)
        only_complete_gdf = gpd.GeoDataFrame(
            only_complete_data, geometry="geometry", crs=gdf.crs
        )
    else:
        gdf_result = pd.DataFrame(merged_data)
        only_complete_gdf = pd.DataFrame(only_complete_data)

    return gdf_result, only_complete_gdf, unmatched_dists


def combine_state_df_with_gdf(df, gdf, df_state_field, gdf_state_field, gdf_dist_field):
    merged_data, states_included, unmatched_states, only_complete_data = [], [], [], []

    for state in gdf[gdf_state_field].unique():
        state_in_df = find_matching_entity(
            state, df[df_state_field].unique(), cutoff=0.85
        )
        if state_in_df:
            states_included.append(state_in_df)
            print(f"Processing {state}...{state_in_df}")
            state_df = df[df[df_state_field] == state_in_df].iloc[0]
            state_gpd = gdf[gdf[gdf_state_field] == state]

            for _, gpd_row in state_gpd.iterrows():
                if gpd_row[gdf_dist_field] not in invalid_districts:
                    merged_row = gpd_row.to_dict()
                    for key, value in state_df.items():
                        if key not in merged_row:
                            merged_row[key] = value
                    only_complete_data.append(merged_row.copy())
                    merged_data.append(merged_row)
                else:
                    merged_row = gpd_row.to_dict()
                    merged_data.append(merged_row)

        else:
            print(f"Warning: {state} not found in DataFrame")
            unmatched_states.append(state)
            state_gpd = gdf[gdf[gdf_state_field] == state]

            for _, gpd_row in state_gpd.iterrows():
                merged_data.append(gpd_row.to_dict())

    # Create a new GeoDataFrame with the merged data
    gdf = gpd.GeoDataFrame(merged_data, geometry="geometry", crs=gdf.crs)
    only_complete_gdf = gpd.GeoDataFrame(
        only_complete_data, geometry="geometry", crs=gdf.crs
    )

    return gdf, only_complete_gdf, unmatched_states


def combine_pointgdf_with_gdf(gdf, point_gdf, new_field_name):
    print(f"Creating {new_field_name} column...")
    point_gdf = point_gdf.to_crs(gdf.crs)
    district_sindex = gdf.sindex
    point_coords = np.column_stack((point_gdf.geometry.x, point_gdf.geometry.y))

    district_indices = np.zeros(len(point_gdf), dtype=int)

    for idx, district in enumerate(gdf.geometry):
        possible_matches_index = list(district_sindex.intersection(district.bounds))

        if possible_matches_index:
            mask = contains(district, point_coords[:, 0], point_coords[:, 1])
            district_indices[mask] = idx

    point_gdf["district_index"] = district_indices

    counts = point_gdf.groupby("district_index").size()

    all_districts = pd.Series(0, index=range(len(gdf)))
    counts = all_districts.add(counts, fill_value=0)

    gdf[new_field_name] = counts.values

    return gdf


def create_map_text_dict(nitiayog_meta, district_profiles):

    map_text_dict = {}

    for _, feature in enumerate(district_profiles.columns[2:]):
        qn, level = feature.split("|")
        level_details = level.split("_")
        if len(level_details) == 1:
            num_type, data_type = "NA", level_details[0]
            num_text = "NA"
        else:
            num_type, data_type = level_details

        if num_type == "pct":
            num_text = "Percentage"
        elif num_type == "cnt":
            num_text = "Count"

        category = nitiayog_meta[nitiayog_meta["Qn"] == qn]["Category"].values[0]
        title = nitiayog_meta[nitiayog_meta["Qn"] == qn]["Title"].values[0]
        source = nitiayog_meta[nitiayog_meta["Qn"] == qn]["Source"].values[0]
        frequency = nitiayog_meta[nitiayog_meta["Qn"] == qn]["Frequency"].values[0]

        map_text_dict[feature] = {}
        map_text_dict[feature]["num_text"] = num_text
        map_text_dict[feature]["data_type"] = data_type
        map_text_dict[feature]["category"] = category
        map_text_dict[feature]["title"] = title
        map_text_dict[feature]["source"] = source
        map_text_dict[feature]["frequency"] = frequency

    map_text_df = pd.DataFrame(map_text_dict).T
    map_text_df.reset_index(inplace=True)
    return map_text_df


def find_matching_entity(target, candidates, cutoff=0.6):
    target_parts = target.split("/")
    matches = []

    for candidate in candidates:
        candidate_parts = candidate.split("/")

        # Check similarity between all combinations of target and candidate parts
        for target_part in target_parts:
            for candidate_part in candidate_parts:
                similarity = difflib.SequenceMatcher(
                    None,
                    target_part.replace("&", "and").strip().lower(),
                    candidate_part.replace("&", "and").strip().lower(),
                ).ratio()
                if similarity >= cutoff:
                    matches.append(candidate)
                    break
            if matches:
                break

    if matches:
        # Return the closest match from the candidates list
        closest_match = difflib.get_close_matches(target, matches, n=1, cutoff=cutoff)
        if closest_match:
            return closest_match[0]
        else:
            # If no close match is found, return the first matching candidate
            return matches[0]
    else:
        # Check if the target is a substring of any candidate
        for candidate in candidates:
            if (
                target.lower() in candidate.lower()
                or candidate.lower() in target.lower()
            ):
                return candidate

    return None


def kriging_interpolation_polygons(gdf, column_to_interpolate):
    # Separate the data into known and unknown polygons
    known = gdf[gdf[column_to_interpolate].notna()]
    known = known.to_crs(epsg=4326)
    unknown = gdf[gdf[column_to_interpolate].isna()]
    unknown = unknown.to_crs(epsg=4326)

    # Use centroids for kriging
    x_known = known.geometry.centroid.x.values
    y_known = known.geometry.centroid.y.values
    z_known = known[column_to_interpolate].values

    x_unknown = unknown.geometry.centroid.x.values
    y_unknown = unknown.geometry.centroid.y.values

    # Perform ordinary kriging
    ok = OrdinaryKriging(
        x_known,
        y_known,
        z_known,
        variogram_model="spherical",
        verbose=False,
        enable_plotting=False,
    )

    z_interpolated, _ = ok.execute("points", x_unknown, y_unknown)

    # Update the original dataframe
    gdf.loc[gdf[column_to_interpolate].isna(), column_to_interpolate] = z_interpolated

    return gdf


def calculate_geodesic_length(geometry):
    geod = Geod(ellps="WGS84")
    if geometry.is_empty or geometry is None:
        return 0.0
    elif isinstance(geometry, LineString):
        return geod.geometry_length(geometry) / 1000
    elif isinstance(geometry, MultiLineString):
        return sum(geod.geometry_length(line) for line in geometry.geoms) / 1000
    else:
        return 0.0


def add_length_field(lines_gdf, polygon_gdf):
    # Ensure both GeoDataFrames are in the same CRS
    lines_gdf = lines_gdf.to_crs(polygon_gdf.crs)

    # Create spatial index for lines
    lines_sindex = lines_gdf.sindex

    # Vectorize the calculation of line lengths
    lines_gdf["length"] = lines_gdf.geometry.length

    total_lengths = []

    for polygon in polygon_gdf.geometry:
        # Get potential matches using spatial index
        possible_matches_index = list(lines_sindex.intersection(polygon.bounds))
        possible_matches = lines_gdf.iloc[possible_matches_index]

        # Use vectorized operations to check for intersections
        mask = contains(
            polygon,
            possible_matches.geometry.centroid.x,
            possible_matches.geometry.centroid.y,
        )
        intersecting_lines = possible_matches[mask]

        # Calculate total length
        total_length = intersecting_lines["length"].sum()
        total_lengths.append(total_length)

    return total_lengths


def calculate_geodesic_area(geometry):
    geod = Geod(ellps="WGS84")
    if geometry.geom_type == "Polygon":
        return abs(geod.geometry_area_perimeter(geometry)[0]) / 10**6
    elif geometry.geom_type == "MultiPolygon":
        return (
            sum(abs(geod.geometry_area_perimeter(poly)[0]) for poly in geometry.geoms)
            / 10**6
        )
    else:
        return None


# def process_udise_data(school_udise_folder):
#     school_udise_files = os.listdir(school_udise_folder)
#     school_df = pd.DataFrame(columns=["state", "district", "total_schools"])

#     for state in school_udise_files:
#         print("Processing ", state)
#         state_df = pd.read_excel(
#             os.path.join(school_udise_folder, state),
#             sheet_name="ag-grid",
#             skiprows=[0, 1, 2],
#         )
#         state_df = state_df[state_df["Location"] != "Total"]
#         state_df = state_df[state_df["School Management"] == "Overall"]
#         for _, row in state_df.iterrows():
#             if row["Location"] != "Total":
#                 print(state.split(".xlsx")[0], row["Location"].title(), row["Total"])
#                 school_df.loc[len(school_df)] = [
#                     state.split(".xlsx")[0],
#                     row["Location"].title(),
#                     row["Total"],
#                 ]

#     return school_df
