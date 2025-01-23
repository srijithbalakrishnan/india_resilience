import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import numpy as np


# dictionaries

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
        # "Highway density (km per 100 sqkm)": ["highway_per_100sqkm", 0, -1, "R"],
        "Road density (km per 100 sqkm)": ["roads_100sqkm", 0, -1, "R"],
        # "Power line length (km)": ["powerlen_km", 0, -1, "R"],
        "Transmission line density (km per 100 sqkm)": [
            "transmission_100sqkm",
            0,
            -1,
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
    },
    "community": {
        "Per capita income (INR '000)": ["per_capita_income", 0, -1, "R"],
        "Population <15 yrs (%)": ["perc_minors", 0, 100, "V"],
        "Women (age 15-49) who are literate (%)": ["perc_women_literate", 0, 100, "R"],
        "Fully-vaccinated children aged 12-23 months (%)": [
            "perc_child_vaccinated",
            0,
            100,
            "R",
        ],
        "Underweight Children <5 yrs (%)": [
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
        "Households with health insurance/financial schemes (%)": [
            "perc_social_security",
            0,
            100,
            "R",
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
    "state_capacity": {
        "Citizen centric governance": ["citizen_governance", 0, 1, "R"],
        "Judiciary and public security": ["public_security", 0, 1, "R"],
        "Commerce and industry": ["commerce_industry", 0, 1, "R"],
        "Human resource development": ["hr_devt", 0, 1, "R"],
        "Public infrastructure and utilities": ["public_infra", 0, 1, "R"],
        "Public health": ["public_health", 0, 1, "R"],
        "Economic governance": ["economic_governance", 0, 1, "R"],
        "Social welfare": ["social_welfare", 0, 1, "R"],
        "Agriculture and allied activities": ["agriculture_allied", 0, 1, "R"],
        "Environment governance": ["env_governance", 0, 1, "R"],
        # "Composite score for governance": ["governance_composite", 0, 10, "R"],
    },
}

indicators_key_dict = {
    "hazard": "Hazard",
    "community": "Community",
    "infrastructure_accessibility": "Infrastructure service accessibility",
    "social_infrastructure": "Social infrastructure",
    "critical_infrastructure": "Critical infrastructure",
    "state_capacity": "State capacity",
}


# Principal component analysis


def scale_data(data):
    """Standardize the data."""
    scaler = StandardScaler()
    return scaler.fit_transform(data)


def find_optimal_components(data, variance_threshold=0.95):
    """Find the optimal number of PCA components to explain the desired variance."""
    data_scaled = scale_data(data)
    pca = PCA()
    pca.fit(data_scaled)
    cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumulative_variance_ratio >= variance_threshold) + 1
    return n_components


def calculate_pca_weights(data_scaled, n_components):
    """Perform PCA and calculate variable contributions."""
    pca = PCA(n_components=n_components)
    pca.fit(data_scaled)
    explained_variance = pca.explained_variance_ratio_

    # Calculate variable contributions
    components = np.abs(pca.components_)
    variable_contributions = np.zeros(components.shape[1])
    for k in range(n_components):
        variable_contributions += components[k, :] * explained_variance[k]

    # Normalize contributions
    weights = variable_contributions / variable_contributions.sum()
    return weights


def create_composite_index(data_scaled, weights, columns):
    """Calculate the composite index based on variable weights."""
    composite_index = np.zeros(len(data_scaled))
    for i, weight in enumerate(weights):
        composite_index += weight * data_scaled[:, i]
    return composite_index


# def scale_to_range(data, min_val=0, max_val=100):
#     """Scale data to a specified range (default 0-100)."""
#     return min_val + (data - data.min()) * (max_val - min_val) / (
#         data.max() - data.min()
#     )


def process_cluster(data_resilience_gdf, cluster_type):
    """
    Process a single cluster type: calculate PCA, variable weights,
    and generate a scaled composite index.
    """

    indicators_dict_new = {
        cluster_type: {
            values[0]: key for key, values in indicators_dict[cluster_type].items()
        }
        for cluster_type in indicators_dict.keys()
    }
    # Extract relevant columns and handle missing values
    columns = [values[0] for values in indicators_dict[cluster_type].values()]
    dat = data_resilience_gdf[columns].copy()
    dat.replace(np.nan, 0, inplace=True)

    # Scale the data
    dat_scaled = scale_data(dat)

    # Find optimal PCA components
    n_compon = find_optimal_components(dat_scaled)

    # Calculate PCA weights
    weights = calculate_pca_weights(dat_scaled, n_compon)
    # variable_weights = dict(zip(columns, weights))

    # Print variable contributions
    # for var, weight in variable_weights.items():
    # print(indicators_dict_new[cluster_type][var], "@", weight)

    # Create composite index
    composite_index = create_composite_index(dat_scaled, weights, columns)

    return composite_index
