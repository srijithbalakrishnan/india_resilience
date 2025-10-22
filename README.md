# India District-level Infrastructure and Community Vulnerability Data

This repository contains the open-source data, methodology, and results from the study on district-level infrastructure access, availability, and community vulnerability in India. The study develops composite indicators for critical, essential, and social infrastructure systems, and examines their spatial associations with community vulnerability across 733 districts using spatial regression and clustering analysis.  

The analysis identifies distinct infrastructure profiles that explain spatial inequalities in vulnerability and provides insights for equitable and targeted infrastructure investments.

This work is a collaboration between researchers at Delft University of Technology and the Indian Institute of Technology Kanpur. Researchers are encouraged to use the data published in this repository for research and policy analysis.

If you use this repository in your research or projects, please cite the original study as follows:  
> Balakrishnan, S., S., Shivam and Kothari, C., *Empirical Evidence for Synergistic Influence of Regional Infrastructure Availability and  Access on Community Vulnerability*. Available at SSRN: https://ssrn.com/abstract=5475572 (Preprint).

## Results and Data

### 1. Composite Indicators

  Composite indicators for district-level infrastructure characteristics and community vulnerability are calculated across four dimensions:  
       - **Critical Infrastructure ($CI_{crit}$):** Availability of key physical systems such as road networks, energy grids, and transport connectivity.  
       - **Essential Utility Access ($CI_{ess}$):** Household-level access to electricity, drinking water, sanitation, and clean cooking fuel.  
       - **Social Infrastructure ($CI_{soc}$):** Density of educational, health, and safety facilities such as schools, hospitals, and police stations.  
       - **Community Vulnerability ($CI_{comm}$):** Composite index representing socioeconomic sensitivity, exposure, and adaptive capacity indicators.  
   
   - #### Composite indicator maps
     All composite indicators are derived by applying Principal Component Analysis (PCA) on normalized infrastructure and community features (using standard scaling). A positive value indicates capabilities above the national average, while a negative value indicates a score below the national average.

     <img src="graphics/InfraIndex.png" alt="Composite infrastrucutre indicators" width="750">
     <img src="graphics/CommunityIndex.png" alt="Composite community indicators" width="255">

   - #### Data for download:  
       - [District-level composite infrastructure and vulnerability indices](data/composite_resilience_indices.parquet): District-level composite index values for all infrastructure dimensions and vulnerability (use geopandas).

       > import geopandas as gpd
       > gdf = gpd.read_parquet(<parquet_file>.parquet)

### 2. Infrastructure and Vulnerability Data

  This section includes the raw and processed datasets used to compute infrastructure and vulnerability indicators. These datasets form the basis for reproducing the spatial analysis and include:  
  - Infrastructure indicators (e.g., road density, hospital availability, utility service coverage).  
  - Vulnerability indicators (e.g., literacy, household composition, and financial resilience).  

  #### Data Sources

  The district-level infrastructure and vulnerability data were compiled and derived from multiple national and open datasets, including OpenStreetMap, ESRI India, UDISE+ (Ministry of Education, Government of India), State economic censuses and budget documents, and National Family Health Survey 5 (Ministry of Health and Family Welfare, Government of India). Please refer to the data dictionary for complete metadata and feature definitions: [data/resilience_data_dictionary.pdf](data/resilience_data_dictionary.pdf).
   
  #### District-level feature maps

  <img src="graphics/combined_plot_pillars1.png" alt="Infrastructure Data1" width="750">
  <img src="graphics/combined_plot_pillars2.png" alt="Infrastructure Data2" width="750">

  #### Correlation among vulnerability and resilience characteristics

  Correlation analysis was done using Kendall's rank correlation ($\tau$).
  <img src="graphics/indicators_corr.png" alt="Infrastructure Data2" width="800">

  #### Download data (pre-processed)

  - [District-level infrastructure and vulnerability features](data/resilience_data.parquet): District-level dataset for all indicators (use geopandas).  
  - [Data dictionary](data/resilience_data_dictionary.pdf): Detailed description of all indicators and data sources.

### 3. Cluster Analysis: Infrastructure and Vulnerability Profiles
 
  Districts are grouped into six distinct clusters based on shared infrastructure and vulnerability characteristics. The clustering integrates spatial regression outputs and composite indicators to reveal inter-district similarities and disparities in access, availability, and social infrastructure.  
  - Clustering method: Agglomerative Hierarchical Clustering algorithm (validated with silhouette and Calinski–Harabasz indices).  
  - Outputs include spatial cluster maps and average infrastructure profiles per cluster to guide equitable infrastructure planning.  

    #### Spatial infrastructure–vulnerability profiles
     The staitsital distributions of infrastructure characcteristics were obtained and mapped to understand the unique infrastructure profiles and spatial disparities.

     <img src="graphics/combined_cluster_plots.png" alt="Cluster Profiles" width="900">


## Methods

The methodological framework integrates spatial econometric modeling with composite indicator analysis to capture multidimensional infrastructure–vulnerability linkages.  

Four main stages are followed:
1. **Indicator development:** High-resolution geospatial, demographic, and infrastructural datasets are processed and standardized to construct composite indicators representing critical, essential (utility-based), and social infrastructure dimensions, as well as community vulnerability.  
2. **Spatial analysis:** A spatial lag model (SLM) is used to quantify both direct and spillover associations between infrastructure characteristics and community vulnerability, explicitly accounting for spatial dependence across neighboring districts.  
3. **Cluster analysis:** The resulting marginal effects are aggregated and analyzed through clustering to identify typologies of infrastructure access and availability, highlighting spatial inequalities and informing resource allocation priorities. 
4. **Statistical modelling:** The association between community vulnerability and infrastructure dimensions were statistically emanied by applying linear regression on the composite indicators. 

  <img src="graphics/Resilience Disaggregation.jpeg" alt="Methodology" width="600">


### Contact

   - For more details, contact Srijith Balakrishnan. Email: s.balakrishnan@tudelft.nl
