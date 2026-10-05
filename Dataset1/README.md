# Data from: Integration of physiological and remote sensing traits for improved genomic prediction of wheat yield

Dataset DOI: [10.5061/dryad.rbnzs7hqq](10.5061/dryad.rbnzs7hqq)

## Description of the data and file structure

Dataset used in the publication "Integration of physiological and remote sensing traits for improved genomic prediction of wheat yield". 

The data was collected at the Campo Experimental Norman E. Borlaug (CENEB) Field station from the International Maize and Wheat Improvement Center (CIMMYT), Ciudad Obregon, Sonora, Mexico in the growth season 2022-2023. We collected agronomic, physiological and remote sensing traits to predict wheat yield using genomic prediction in three environments: irrigation, drought stress and terminal heat stress. Raw data and BLUEs for grain yield, grain filling percentage, grain filling rate, plant height, days to heading, stomatal conductance, transpiration, quantum yield of photosystem II and high-throughput traits in the MOLPAN wheat population.

### Files and variables

#### File: Garcia-Barrios_et_al_2025_TPG_raw_BLUEs_datasets.xlsx

**Description:** 

##### Variables

* Environment: irrigation, drought, terminal heat
* Entry: genotype number
* Rep: replicate 
* CIMMYT ID: ID used for internal CIMMYT identification control
* Cross Name
* Grain Yield_gm2: Grain yield harvested at physiological maturity in grams per square meter
* GFP_%: Grain filling percentage (%) in days, this means how many days each genotype was in the grain-filling period
* GFR_gm2: Grain filling rate in grams per square meter
* Height_cm: Plant height at physiological maturity
* Heading_days: Days passed from emergence to heading
* gs_molm2s1: Leaf stomatal conductance
* E_mmolm2s1: Leaf transpiration rate
* PhiPS2: Quantum yield of photosystem II
* R_Blue_vg: Reflectance value of the blue band at the vegetative stage
* R_Green_vg: Reflectance value of the green band at the vegetative stage
* R_Red_vg: Reflectance value of the red band at the vegetative stage
* R_RedEdge_vg: Reflectance value of the red edge band at the vegetative stage
* R_NIR_vg: Reflectance value of the near infrared band at the vegetative stage
* Cired_edge_vg: Chlorophyll index calculated with the red edge band at the vegetative stage
* NDVI_vg: Normalized Differenced Vegetation Index at the vegetative stage
* LAI_PROSAIL_vg: Leaf area index derived from the inversion of the radiative transfer model PROSAIL at the vegetative stage
* Fcover_PROSAIL_vg: Fraction of vegetation cover derived from the inversion of the radiative transfer model PROSAIL at the vegetative stage
* ALA_PROSAIL_vg: Average leaf angle derived from the inversion of the radiative transfer model PROSAIL at the vegetative stage
* SR_vg: Simple ratio index at the vegetative stage
* FIPAR_PROSAIL_vg: Fraction of Intercepted Photosynthetically Active Radiation derived from the inversion of the radiative transfer model PROSAIL at the vegetative stage
* Cigreen_vg: Chlorophyll index calculated with the green band at the vegetative stage
* Cab_PROSAIL_vg: Chlorophyll a+b content derived from the inversion of the radiative transfer model PROSAIL at the vegetative stage
* LAICab_PROSAIL_vg: Leaf area index based on chlorophyll content derived from the inversion of the radiative transfer model PROSAIL at the vegetative stage
* MCARI_vg: Modified Chlorophyll Absorbed Reflectance Index at the vegetative stage
* R_Blue_gf: Reflectance value of the blue band at the grain-filling stage
* R_Green_gf: Reflectance value of the green band at the grain-filling stage
* R_Red_gf: Reflectance value of the red band at the grain-filling stage
* R_RedEdge_gf: Reflectance value of the red edge band at the grain-filling stage
* R_NIR_gf: Reflectance value of the near infrared band at the grain-filling stage
* Cired_edge_gf: Chlorophyll index calculated with the red edge band at the grain-filling stage
* NDVI_gf: Normalized Differenced Vegetation Index at the grain-filling stage
* LAI_PROSAIL_gf: Leaf area index derived from the inversion of the radiative transfer model PROSAIL at the grain-filling stage
* Fcover_PROSAIL_gf: Fraction of vegetation cover derived from the inversion of the radiative transfer model PROSAIL at the grain-filling stage
* ALA_PROSAIL_gf: Average leaf angle derived from the inversion of the radiative transfer model PROSAIL at the grain-filling stage
* SR_gf: Simple ratio index at the grain-filling stage
* FIPAR_PROSAIL_gf: Fraction of Intercepted Photosynthetically Active Radiation derived from the inversion of the radiative transfer model PROSAIL at the grain-filling stage
* Cigreen_gf: Chlorophyll index calculated with the green band at the grain-filling stage
* Cab_PROSAIL_gf: Chlorophyll a+b content derived from the inversion of the radiative transfer model PROSAIL at the grain-filling stage
* LAICab_PROSAIL_gf: Leaf area index based on chlorophyll content derived from the inversion of the radiative transfer model PROSAIL at the grain-filling stage
* MCARI_gf: Modified Chlorophyll Absorbed Reflectance Index at the grain-filling stage

## Code/software

To analyze this dataset we used Rstudio with the packages lme4 and prosail. 

The package lme4 was further implemented using the graphic user interface META-R (v6.04)

## Access information

Other publicly accessible locations of the data:

* None

Data was derived from the following sources:

* None

