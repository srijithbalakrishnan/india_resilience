# India District-level Resilience Data

This repository contains the open-source data, methodology, and results from the study on district-level resilience patterns in India. The study develops composite indices and resilience profiles based on a five-pillar framework, leveraging geospatial tools and clustering algorithms to identify resilience patterns and inform policy interventions. All 733 districts in India are covered.

This work is led by researchers at the Delft University of Technology and the Indian Insititue of Technology Kanpur. Researchers are free to use the data published in the repository for their work.

This is an ongoing work and more data and analyses will be added.

If you use data from this repository in your research or projects, please cite the original study as follows:  
> Srijith Balakrishnan, Shivam Srivastava, Chirag Kothari. "Decoding Territorial Resilience Patterns in India", Working Paper, in preparation. *(2025)*.

## Results and Data

Complete analysis can be found [here](india_resilience_analysis.ipynb) (notebook file). For queries and suggestions on raw data, processing, and further analysis, contact the author.

### 1. **Composite Resilience Indices**
   - ### *Description:*  
     Composite indices for district-level territorial resilience are calculated across five key pillars:  
       - **Critical Infrastructure:** Infrastructure density and robustness (e.g., road and power transmission density).  
       - **Social Infrastructure:** Availability of facilities such as schools, healthcare centers, and police stations.  
       - **Community Resilience:** Socioeconomic characteristics, literacy, vaccination, and household-level financial resilience.  
       - **Infrastructure Service Access:** Access to electricity, sanitation, clean water, and clean cooking fuel.  
       - **State Capacity:** Governance indicators such as public safety, economic governance, and public health.  
   
   - ### *Composite index maps*

     <img src="graphics/elsevier/CompositeIndex_all_districts.png" alt="Resilience Map" width="700">

   - ### **Download data:**  
       - [District-level compisite resilience indices](data/composite_resilience_indices.parquet): District-level composite index values for all resilience pillars (use geopandas).

### 2. **Resilience and Vulnerability Data**

   - ### *Description:** 
     This section includes the raw and processed datasets used to compute resilience and vulnerability indicators. These datasets are critical for replicating the analysis and are categorized as follows:  
       - Resilience indicators (e.g., literacy rates, health infrastructure density).  
       - Vulnerability indicators (e.g., underweight children, low BMI). 

   - ### *Data Sources*   

     The district-level resilience and vulnerability data have been collated/derived from various datasets including OpenStreetMap, ESRI India, UDISE+ (Ministry of Education, Government of India), State economic censuses and budget documents, National Family Health Survey 5 (Ministry of Heath and Family Welfare, Government of India), and Good Governance Index (Department of Administrative Reforms and Public Grievances, Government of India). Please find the data dictionary along with the sources [here](data/resilience_data_dictionary.pdf).
   

   - ### *District-level feature maps*

     <img src="graphics/combined_plot_pillars1.png" alt="Resilience Data1" width="750">
     <img src="graphics/combined_plot_pillars2.png" alt="Resilience Data2" width="750">

   - ### *Download data (pre-processed)*

       - [District-level resilience features](data/resilience_data.parquet): District-level data for all resilience indicators (use geopandas).  
       - [District-level indicators (using Standard scaler function)](data/resilience_indicators.parquet): District-level indicators (standardized features; use geopandas).
       - [Data dictionary](data/resilience_data_dictionary.pdf): Detailed description of all features and their sources.

### 3. **Case Study: Resilience Patterns in Cyclone-Prone Districts**

   - ### *Description:* 

     A focused analysis on cyclone-prone districts along India's eastern and western coasts. This case study integrates the resilience framework with cyclone hazard data to classify districts into clusters based on their resilience profiles.  
       - Clustering algorithm: Agglomerative clustering  
       - Key outputs include resilience patterns and spatial distributions of clusters.  

   - ### *Clustering of cyclone-prone districts based on resilience capabilities*

     The northern parts of the eastern coast contain several districts with high cyclone risks, coinciding with low resilience capacities.

     <img src="graphics/combined_cluster_plots.png" alt="Resilience Data2" width="950">


## Methods

This repository presents a framework for assessing district-level hazard-resilience profiles by integrating resilience indicators and hazard characteristics.

- **Composite resilience indicators:**

    Relevant features were identified from government datasets, surveys, and open-source databases.
    Features were standardized and combined into composite resilience indicators using Principal Component Analysis (PCA) to reduce dimensionality and uncover patterns.

 - **Hazard risk analysis**:

    Spatial analysis was conducted to create a district-level composite hazard index, focusing on hazard characteristics.

 - **Integration and Clustering**:

    Resilience indicators and the hazard index were integrated to develop district-level hazard-resilience profiles.
    Distinct resilience patterns for hazard-prone regions were identified using clustering algorithms.
    This comprehensive framework supports spatially informed resilience planning for hazard-prone areas.

  <img src="graphics/Resilience Disaggregation.png" alt="Methodology" width="1050">


## Contact

   - For more details, contact Srijith Balakrishnan. Email: s.balakrishnan@tudelft.nl