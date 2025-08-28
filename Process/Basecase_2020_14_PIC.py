# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 14:00:41 2025

@author: ajh287
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 15:43:16 2025

@author: ajh287
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 16:05:12 2025

@author: ajh287
"""



# %% [markdown]
# (tutorial_101_label)=
#
# # Tutorial 101 - Converters, sources and sinks
#
# <div style="text-align: center;">
#
# ![Model overview for tutorial 101](../../img/REMix_tutorial101.svg "Model overview for tutorial 101")
#
# Model overview of tutorial 101
#
# </div>
#
# ## Part a: setting up the model
#
# This is the first tutorial to introduce a way to set up a model in REMix. It presents a basic model with four regions
# including renewable energy sources, conventional power plant technologies, an electrical demand and accounting for
# carbon emissions.
#
# For the general structure of REMix tutorials have a look at the README.
#
# We build a first base model to be used in later tutorials to build up on and include other energy system
# components (like energy storage and transfer) as well as technologies (e.g. electric vehicles) and concepts
# (e.g. demand response).

# %% [markdown]
# ### Setting up Python
#
# In this first section, we are importing the Python packages needed to run the model and later exemplary evaluation.
# There are also directories defined where the model data and optimization results will be stored.

# %%
# importing dependencies
import numpy as np
import pandas as pd

from remix.framework import Instance

# define often-used shortcut
idx = pd.IndexSlice
# %% [markdown]
# ### General introduction to building models in REMix
#
# For the setup of a model in REMix, preprocessing of data is necessary.
# To do that, the tutorials make use of Pandas DataFrames.
# These are separately set up and collected in lists, before these are being
# written to files that are used as input to the solver.
#
# For the creation of Pandas (pd) DataFrames, we will typically use the
# pd.DataFrame class.
# In addition, we use the pd.MultiIndex.from_product() method to generate a
# multi-index (e.g. three index layers with the first describing the indicator,
# the second describing the indicator used to derive the first indicator and
# third the years).
#
# In the following section, the lists to collect the Pandas DataFrames in are
# initialized in the Instance `m` (as in "model").
# This object is a container in which we will collect all necessary model data.
#
# Not all of the lists initialized with the Instance `m` will be filled in this
# first tutorial.
# This is especially true for storage technologies and energy transfer.
# These two concepts (and more) will be introduced in later tutorials.
#
# One more note: if you do not provide a feature (i.e. fill an empty list),
# REMix will run anyway without that feature but with the other available
# files/features, unless that feature is strictly necessary, like a regional
# mapping.
#
# If you are not yet familiar with the basic functions of Pandas, you can check
# out the 10-minute tutorial in the Pandas documentation:
# https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html

# %%
# initialize model structure of REMix
m = Instance()

# setting the directory the model data should be written to
# a folder "./data" in the project directory is the default in REMix
m.datadir = "./data"
# %% [markdown]
#
# When printing `m`, you will see all REMix features it includes.
#
# For the purpose of the REMix tutorials, we have prepared some dummy data with
# time profiles that are loaded here.
# %%
# load input data
profiles = pd.read_csv("../_input/Basecase_2020_14_PIC.csv", index_col=0)
# %% [markdown]
# ### Defining the model scope
#
# Here is where the model building starts. First of all, we define the model scope.
#
# The model scope describes the fundamental dimensions of the model, e.g. which
# distinct regions and years are modeled.
#
# #### Spatial scope
#
# - `set.nodesdata` : describes the regions for which input data is provided
# such as profiles and capacities for
# power plants
# - `set.nodesmodel` : describes the model regions which can be the same as the
# data regions if the optimization should be done in full resolution.
# - `map.aggregatenodesmodel` : describes the aggregation mapping for data to
# model regions. This can be a 1:1 mapping (like `R3_data` to `R3_model`) or a
# n:1 mapping (like e.g. "R1_North_data" and "R1_South_data" to `R1_model`) if
# multiple data regions should be summed up to a model region.
#
# #### Temporal scope
#
# - `set.years` : the individual years which can be modeled for historical and
# new power plants
# - `set.yearssel` : the years which should be optimized during the run. For
# now, we only use a single year to be optimized.
#
# Our model will comprise four regions, also referred to as "nodes", whose names
# can be arbitrarily chosen. Here, they are called `R3_model`, `R1_model`,
# `R2_model` and `R4_model` (although having nothing to do with the actual
# energy systems of the countries these abbreviations hint at).
# In the first two tutorials, we will only use one node, which is
# `R1_data`/`R1_model`, so the other nodes are not needed until tutorial 103.

# %%
# "map_aggregateNodesModel"
# DataFrame for aggregation from data to model regions
df = pd.DataFrame(
    [
        ["CI_data", "CI_model", 1],
        ["FJ_data", "FJ_model", 1],  # not strictly necessary for tutorial 1 and 2
        ["FSM_data", "FSM_model", 1],  # not strictly necessary for tutorial 1 and 2
        ["KB_data", "KB_model", 1],
        ["MI_data", "MI_model", 1],
        ["NU_data", "NU_model", 1],
        ["NE_data", "NE_model", 1],
        ["PU_data", "PU_model", 1],
        ["PNG_data", "PNG_model", 1],
        ["SA_data", "SA_model", 1],
        ["SI_data", "SI_model", 1],
        ["TA_data", "TA_model", 1],
        ["TU_data", "TU_model", 1],
        ["VU_data", "VU_model", 1],# not strictly necessary for tutorial 1 and 2
    ]
)
df.columns = ["nodesData", "nodesModel", "aggregate"]
df = df.set_index(["nodesData", "nodesModel"])
df["aggregate"] = ""
df.columns = [""]

m.map.add(df, "aggregatenodesmodel")

# Get the data and model regions based on the mapping
# "set_nodesData"
m.set.add(
    list(sorted(set(m.map.aggregatenodesmodel.index.get_level_values(0)))), "nodesdata"
)
# "set_nodesModel" & "set_nodesModelSel"
m.set.add(
    list(sorted(set(m.map.aggregatenodesmodel.index.get_level_values(1)))), "nodesmodel"
)

# Set the years to be considered in the model and the years to be optimized
# "set_years"
m.set.add(
    ["2030"], "years"
)  # must include all years that data is provided for in the model
# "set_yearsSel"
m.set.add(["2030"], "yearssel")  # years to be optimised
# %% [markdown]
# ### Setting the objective function and indicator bounds
#
# Models in REMix are usually optimized based on a cost-minimization approach.
# The framework theoretically also allows other approaches.
#
# We will use different types of commodities - electricity, methane, carbon
# dioxide - and system costs as indicator.
# We will use the following units for these:
#
# - Elec : electricity in GWh_el
# - CH4 : methane in GWh_ch
# - CO2 : carbon dioxide emissions in tsd. t or kt
# - Cost (Invest, OMVar, OMFix, CarbonCost, FuelCost) : cost values in million EUR or MEUR
#
# In the first DataFrame we define a value for the indicator `SystemCost` and
# column `obj` to -1 to communicate that we want to minimize this indicator.
# Similarly, a value of 1 would indicate a maximization.
# The first field is used for the regional and year dimensions.
# The value `global` uses all the regions in the system (in this example
# R1_model, R2_model, R3_model, R4_model), whereas the value `horizon` takes
# into account all years in the set `set.yearssel` (here only 2030).
#
# We set a social discount rate in the same DataFrame, which will be the default
# value throughout the model, but can be overwritten for certain technologies or
# model regions if wanted.

# %%
# "accounting_indicatorBounds"
# setting the objective function and indicator bounds
accounting_indicatorBounds = pd.DataFrame(
    index=pd.MultiIndex.from_product([["global"], ["horizon"], ["SystemCost"]])
)
accounting_indicatorBounds["obj"] = -1  # minimization of system costs
accounting_indicatorBounds["discount"] = 0.08  # social discount rate for the indicators

m.parameter.add(accounting_indicatorBounds, "accounting_indicatorbounds")
accounting_indicatorBounds
# %% [markdown]
# We are also setting up the indicators we want to account for as `SystemCost`
# in the model.
#
# Indicators are used for general accounting inside the energy system. For this
# purpose we introduce an indicator `SystemCost` to reflect the overall costs of
# the system.
# This indicator is calculated by summing up the following individual cost
# indicators with an equal weighting of 1 in the `accounting_perIndicator`
# DataFrame.
#
# - `Invest` : investment cost for a technology unit (in MEUR/MW)
# - `OMVar` : variable operation and maintenance cost  (in MEUR/MWh) (not set in this tutorial)
# - `OMFix` : fix operation and maintenance costs (in MEUR/MW/year)
# - `FuelCost` : costs for imports of methane into the model regions (in MEUR/MWh)

# %%
# "accounting_perIndicator"
# set up accounting per indicator for all years to calculate
accounting_perIndicator = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [
            ["SystemCost"],
            [
                "Invest",
                "OMFix",
                "FuelCost",
            ],
            ["global"],
            m.set.yearssel,  # accounting for all optimization years
        ]
    )
)
accounting_perIndicator["perIndicator"] = 1

m.parameter.add(accounting_perIndicator, "accounting_perindicator")
accounting_perIndicator
# %% [markdown]
# ### Converter technologies
#
# #### Adding converter technologies
#
# In this section, the basic structure of including different converter
# technologies in REMix is introduced.
#
# In this basic model, we introduce the possibility for the model to build
# methane-fired combined-cycle gas turbines ("CCGT"), solar power plants ("PV")
# and onshore wind turbine ("WindOnshore").
#
# The names chosen for the technologies are completely arbitrary.
# We are trying to use the same ones throughout the tutorials, however.

# %%
# "converter_techParam"
# setting technology parameters
# Define tech groups
# Define tech groups with lifetime and availability
tech_specs = {
    "DG": {"lifeTime": 25, "activityUpperLimit": 1},
    "NG_plant": {"lifeTime": 25, "activityUpperLimit": 1},
    "BG_B": {"lifeTime": 25, "activityUpperLimit": 0},   # No feed-in
    "PV_B": {"lifeTime": 25, "activityUpperLimit": 0},  # Feed-in
    "WindOnshore_B": {"lifeTime": 25, "activityUpperLimit": 0},
    "Hydro_B": {"lifeTime": 25, "activityUpperLimit": 0},
    "Geothermal_B": {"lifeTime": 25, "activityUpperLimit": 0},
    "MDV": {"lifeTime": 25, "activityUpperLimit": 1},
    "HDV": {"lifeTime": 25, "activityUpperLimit": 1},
    "LDV": {"lifeTime": 25, "activityUpperLimit": 1},
    "Bus": {"lifeTime": 25, "activityUpperLimit": 1},
    "Two_wheel": {"lifeTime": 25, "activityUpperLimit": 1},
    "Aviation": {"lifeTime": 25, "activityUpperLimit": 1},
    "Marine": {"lifeTime": 25, "activityUpperLimit": 1},
    "cook": {"lifeTime": 25, "activityUpperLimit": 1},
    "Industry": {"lifeTime": 25, "activityUpperLimit": 1},
    "DW_LPG_converter": {"lifeTime": 25, "activityUpperLimit": 1},
    "DW_Electric_converter": {"lifeTime": 25, "activityUpperLimit": 1}

}

# Create DataFrame
converter_techParam = pd.DataFrame(
    index=pd.MultiIndex.from_product([list(tech_specs.keys()), m.set.yearssel])
)

# Assign values from dictionary
for tech, specs in tech_specs.items():
    converter_techParam.loc[idx[tech], "lifeTime"] = specs["lifeTime"]
    converter_techParam.loc[idx[tech], "activityUpperLimit"] = specs["activityUpperLimit"]

# Add to model
m.parameter.add(converter_techParam, "converter_techparam")
converter_techParam


# %%
# "converter_capacityParam"
# defining upper and/or lower limits for converter technologies
# Example user inputs for each node and tech
# Keys: node -> tech -> (lower_limit, upper_limit) in GW
#biomass_limits = [12, 2380, 168, 221, 22,5,4,11330, 1, 295, 1507, 211, 9, 671] 
capacity_limits = {
    "CI_data": {
        "DG": (0.018, 0.018),
        "PV_B": (0.0052, 0.0052),
        "MDV": (0, 1000),
        "HDV": (0, 1000),
        "LDV": (0, 1000),
        "Bus": (0, 1000),
        "Two_wheel": (0, 1000),
        "Aviation": (0, 1000),
        "Marine": (0, 1000),
        "cook": (0, 1000),
        "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000),
        "DW_Electric_converter": (0, 1000)
    },
    "FJ_data": {
        "DG": (0.172, 0.172),
        "BG_B": (0.0580, 0.0580),
        "PV_B": (0.0090, 0.0090),
        "Hydro_B": (0.0625, 0.0625),
        "MDV": (0, 1000),
        "HDV": (0, 1000),
        "LDV": (0, 1000),
        "Bus": (0, 1000),
        "Two_wheel": (0, 1000),
        "Aviation": (0, 1000),
        "Marine": (0, 1000),
        "cook": (0, 1000),
        "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000),
        "DW_Electric_converter": (0, 1000)# hydro adjusted
    },
    "FSM_data": {
        "DG": (0.0388, 0.0388),
        "PV_B": (0.0028, 0.0028),
        "WindOnshore_B": (0.0009, 0.0009),
        "Hydro_B": (0.000225, 0.000225),
        "MDV": (0, 1000),
        "HDV": (0, 1000),
        "LDV": (0, 1000),
        "Bus": (0, 1000),
        "Two_wheel": (0, 1000),
        "Aviation": (0, 1000),
        "Marine": (0, 1000),
        "cook": (0, 1000),
        "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000),
        "DW_Electric_converter": (0, 1000)# hydro adjusted
    },
    "KB_data": {
        "DG": (0.0066, 0.0066),
        "PV_B": (0.0030, 0.0030),
        "MDV": (0, 1000),
        "HDV": (0, 1000),
        "LDV": (0, 1000),
        "Bus": (0, 1000),
        "Two_wheel": (0, 1000),
        "Aviation": (0, 1000),
        "Marine": (0, 1000),
        "cook": (0, 1000),
        "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000),
        "DW_Electric_converter": (0, 1000)
    },
    "MI_data": {
        "DG": (0.0287, 0.0287),
        "PV_B": (0.0017, 0.0017),
        "MDV": (0, 1000),
        "HDV": (0, 1000),
        "LDV": (0, 1000),
        "Bus": (0, 1000),
        "Two_wheel": (0, 1000),
        "Aviation": (0, 1000),
        "Marine": (0, 1000),
        "cook": (0, 1000),
        "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000),
        "DW_Electric_converter": (0, 1000)
    },
    "NU_data": {
        "DG": (0.0245, 0.0245),
        "PV_B": (0.0028, 0.0028),
        "MDV": (0, 1000),
        "HDV": (0, 1000),
        "LDV": (0, 1000),
        "Bus": (0, 1000),
        "Two_wheel": (0, 1000),
        "Aviation": (0, 1000),
        "Marine": (0, 1000),
        "cook": (0, 1000),
        "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000),
        "DW_Electric_converter": (0, 1000)
    },
    "NE_data": {
        "DG": (0.0021, 0.0021),
        "PV_B": (0.0010, 0.0011),
        "MDV": (0, 1000),
        "HDV": (0, 1000),
        "LDV": (0, 1000),
        "Bus": (0, 1000),
        "Two_wheel": (0, 1000),
        "Aviation": (0, 1000),
        "Marine": (0, 1000),
        "cook": (0, 1000),
        "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000),
        "DW_Electric_converter": (0, 1000)
    },
    "PU_data": {
        "DG": (0.0303, 0.0303),
        "PV_B": (0.0030, 0.0032),
        "MDV": (0, 1000),
        "HDV": (0, 1000),
        "LDV": (0, 1000),
        "Bus": (0, 1000),
        "Two_wheel": (0, 1000),
        "Aviation": (0, 1000),
        "Marine": (0, 1000),
        "cook": (0, 1000),
        "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000),
        "DW_Electric_converter": (0, 1000)
    },
    "PNG_data": {
        "DG": (0.280, .300),
        "NG_plant": (0.082, 0.082),
        "BG_B": (0.0182, 0.0182),
        "PV_B": (0.0031, 0.0031),
        "Hydro_B": (0.115, 0.115),
        "Geothermal_B": (0.011, 0.011),
        "MDV": (0, 1000),
        "HDV": (0, 1000),
        "LDV": (0, 1000),
        "Bus": (0, 1000),
        "Two_wheel": (0, 1000),
        "Aviation": (0, 1000),
        "Marine": (0, 1000),
        "cook": (0, 1000),
        "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000),
        "DW_Electric_converter": (0, 1000)
        },
    "SA_data": {
        "DG": (0.0315, 0.0315),
        "BG_B": (0.0011, 0.0011),
        "PV_B": (0.0138, 0.0138),
        "WindOnshore_B": (0.0005, 0.0005),
        "Hydro_B": (.0063, .0063),
        "MDV": (0, 1000),
        "HDV": (0, 1000),
        "LDV": (0, 1000),
        "Bus": (0, 1000),
        "Two_wheel": (0, 1000),
        "Aviation": (0, 1000),
        "Marine": (0, 1000),
        "cook": (0, 1000),
        "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000),
        "DW_Electric_converter": (0, 1000)# hydro adjusted
    },
    "SI_data": {
        "DG": (0.0527, 0.0527),
        "BG_B": (0.0008, 0.0008),
        "PV_B": (0.0023, 0.0023),
        "Hydro_B": (.00018, .00018),
        "MDV": (0, 1000),
        "HDV": (0, 1000),
        "LDV": (0, 1000),
        "Bus": (0, 1000),
        "Two_wheel": (0, 1000),
        "Aviation": (0, 1000),
        "Marine": (0, 1000),
        "cook": (0, 1000),
        "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000),
        "DW_Electric_converter": (0, 1000)# hydro adjusted
    },
    "TA_data": {
        "DG": (0.0167, 0.0167),
        "PV_B": (0.0071, 0.0071),
        "WindOnshore_B": (0.00151, 0.00151),
        "MDV": (0, 1000),
        "HDV": (0, 1000),
        "LDV": (0, 1000),
        "Bus": (0, 1000),
        "Two_wheel": (0, 1000),
        "Aviation": (0, 1000),
        "Marine": (0, 1000),
        "cook": (0, 1),
        "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000),
        "DW_Electric_converter": (0, 1000)
    },
    "TU_data": {
        "DG": (0.003, 0.003),
        "PV_B": (0.0029, 0.0029),
        "MDV": (0, 1000),
        "HDV": (0, 1000),
        "LDV": (0, 1000),
        "Bus": (0, 1000),
        "Two_wheel": (0, 1000),
        "Aviation": (0, 1000),
        "Marine": (0, 1000),
        "cook": (0, 1),
        "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000),
        "DW_Electric_converter": (0, 1000)
    },
    "VU_data": {
        "DG": (0.0232, 0.0232),
        "PV_B": (0.0044, 0.0044),
        "WindOnshore_B": (0.0032, 0.0032),
        "Hydro_B": (.00054, .00054),
        "MDV": (0, 1000),
        "HDV": (0, 1000),
        "LDV": (0, 1000),
        "Bus": (0, 1000),
        "Two_wheel": (0, 1000),
        "Aviation": (0, 1000),
        "Marine": (0, 1000),
        "cook": (0, 1),
        "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000),
        "DW_Electric_converter": (0, 1000)# hydro adjusted
    }
}


# Build DataFrame index
all_techs = list({tech for node in capacity_limits for tech in capacity_limits[node]})
converter_capacityParam = pd.DataFrame(
    index=pd.MultiIndex.from_product([m.set.nodesdata, m.set.yearssel, all_techs])
)

# Fill from user input
for node, techs in capacity_limits.items():
    for tech, (lower, upper) in techs.items():
        converter_capacityParam.loc[idx[node, :, tech], "unitsLowerLimit"] = lower
        converter_capacityParam.loc[idx[node, :, tech], "unitsUpperLimit"] = upper

# Drop empty rows
converter_capacityParam = converter_capacityParam.dropna(how="all")

# Add to model
m.parameter.add(converter_capacityParam, "converter_capacityparam")

converter_capacityParam

# %% [markdown]
# Activities in REMix are the conversion processes a technology can perform.
# For this example we define an activity "Powergen" (as in power generation).
#
# For the CCGT technology this means burning methane in order to get electricity
# and carbon dioxide as a by-product of the combustion process.
#
# For the renewable energy sources wind and PV we model the activity `Powergen`
# by setting a value of 1, which is arbitrary in this case, however, since the
# actual potential for wind and solar energy is modeled as "activityProfile"
# below, which overwrites this value.

# %%
# "converter_coefficient"
converter_coefficient = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [
            ["DG", "BG_B", "PV_B", "WindOnshore_B", "Hydro_B", "Geothermal_B","MDV","HDV", "LDV", "Bus", "Two_wheel", "Aviation", "Marine","cook", "Industry", "DW_LPG_converter", "DW_Electric_converter", "NG_plant"],
            m.set.yearssel,
            ["Powergen"],
            ["Biomass", "Elec", "CO2", "Diesel", "Gasoline", "JetA1", "MDO", "T_MDV", "T_HDV","T_LDV","T_Bus","T_Two_wheel","T_Aviation","T_Marine", "Heat_cooking", "Heat_industry", "LPG", "DHW_LPG", "DHW_Elec", "NG"],
        ]
    )
)
converter_coefficient.loc[idx["DG", :, :, "Elec"], "coefficient"] = 1  # GWh_el
converter_coefficient.loc[idx["DG", :, :, "Diesel"], "coefficient"] = -2.85  # GWh_ch
converter_coefficient.loc[idx["DG", :, :, "CO2"], "coefficient"] = 0.02

converter_coefficient.loc[idx["NG_plant", :, :, "Elec"], "coefficient"] = 1  # GWh_el
converter_coefficient.loc[idx["NG_plant", :, :, "NG"], "coefficient"] = -2  # GWh_ch
converter_coefficient.loc[idx["NG_plant", :, :, "CO2"], "coefficient"] = 0.02

converter_coefficient.loc[idx["BG_B", :, :, "Elec"], "coefficient"] = 1  # GWh_el
converter_coefficient.loc[idx["BG_B", :, :, "Biomass"], "coefficient"] = -2.85  # GWh_ch
converter_coefficient.loc[idx["BG_B", :, :, "CO2"], "coefficient"] = 0.02

converter_coefficient.loc[idx["PV_B", :, :, "Elec"], "coefficient"] = 1  # GWh_el

converter_coefficient.loc[idx["WindOnshore_B", :, :, "Elec"], "coefficient"] = 1  # GWh_el

converter_coefficient.loc[idx["Hydro_B", :, :, "Elec"], "coefficient"] = 1 
converter_coefficient.loc[idx["Geothermal_B", :, :, "Elec"], "coefficient"] = 1 

converter_coefficient.loc[idx["cook",:,:,"Heat_cooking"], "coefficient"] = 1  # GWh_el # GWh_ch
converter_coefficient.loc[idx["cook",:, :, "Biomass"], "coefficient"] = -1
converter_coefficient.loc[idx["cook",:, :, "CO2"], "coefficient"] = 0.02

converter_coefficient.loc[idx["Industry",:,:,"Heat_industry"], "coefficient"] = 1  # GWh_el # GWh_ch
converter_coefficient.loc[idx["Industry",:, :, "Diesel"], "coefficient"] = -1.17
converter_coefficient.loc[idx["Industry",:, :, "CO2"], "coefficient"] = 0.02

converter_coefficient.loc[idx["DW_LPG_converter",:,:,"DHW_LPG"], "coefficient"] = 1  # GWh_el # GWh_ch
converter_coefficient.loc[idx["DW_LPG_converter",:, :, "LPG"], "coefficient"] = -1.17
converter_coefficient.loc[idx["DW_LPG_converter",:, :, "CO2"], "coefficient"] = 0.02

converter_coefficient.loc[idx["DW_Electric_converter",:,:,"DHW_Elec"], "coefficient"] = 1  # GWh_el # GWh_ch
converter_coefficient.loc[idx["DW_Electric_converter",:, :, "Elec"], "coefficient"] = -1.17
converter_coefficient.loc[idx["DW_Electric_converter",:, :, "CO2"], "coefficient"] = 0.02


converter_coefficient.loc[idx["MDV", :, :, "T_MDV"], "coefficient"] = 1  # GWh_el # GWh_ch
converter_coefficient.loc[idx["MDV",:, :, "Diesel"], "coefficient"] = -1
converter_coefficient.loc[idx["MDV",:, :, "CO2"], "coefficient"] = 0.02

converter_coefficient.loc[idx["HDV",:, :, "T_HDV"], "coefficient"] = 1  # GWh_el # GWh_ch
converter_coefficient.loc[idx["HDV",:,:,"Diesel"], "coefficient"] = -1
converter_coefficient.loc[idx["HDV",:,:,"CO2"], "coefficient"] = 0.02

converter_coefficient.loc[idx["LDV",:, :, "T_LDV"], "coefficient"] = 1  # GWh_el # GWh_ch
converter_coefficient.loc[idx["LDV",:,:,"Gasoline"], "coefficient"] = -1
converter_coefficient.loc[idx["LDV",:,:,"CO2"], "coefficient"] = 0.02

converter_coefficient.loc[idx["Bus",:, :, "T_Bus"], "coefficient"] = 1  # GWh_el # GWh_ch
converter_coefficient.loc[idx["Bus",:,:,"Diesel"], "coefficient"] = -1
converter_coefficient.loc[idx["Bus",:,:,"CO2"], "coefficient"] = 0.02

converter_coefficient.loc[idx["Two_wheel",:, :, "T_Two_wheel"], "coefficient"] = 1  # GWh_el # GWh_ch
converter_coefficient.loc[idx["Two_wheel",:,:,"Gasoline"], "coefficient"] = -1
converter_coefficient.loc[idx["Two_wheel",:,:,"CO2"], "coefficient"] = 0.02

converter_coefficient.loc[idx["Aviation",:,:,"T_Aviation"], "coefficient"] = 1  # GWh_el # GWh_ch
converter_coefficient.loc[idx["Aviation",:, :, "JetA1"], "coefficient"] = -1
converter_coefficient.loc[idx["Aviation",:, :, "CO2"], "coefficient"] = 0.02

converter_coefficient.loc[idx["Marine",:,:,"T_Marine"], "coefficient"] = 1  # GWh_el # GWh_ch
converter_coefficient.loc[idx["Marine",:, :, "MDO"], "coefficient"] = -1
converter_coefficient.loc[idx["Marine",:, :, "CO2"], "coefficient"] = 0.02


converter_coefficient = converter_coefficient.dropna(how="all")

m.parameter.add(converter_coefficient, "converter_coefficient")
converter_coefficient
# %% [markdown]
# Since we now introduced a conversion unit that runs on variable renewable
# energy, we need to limit the profile for the activity on the potential
# feed-in.
# We can do this in a similar way to adding the electrical demand profile.
#
# The values in the `profiles.csv` are given in mega watt (MW) of electrical
# feed-in.
# We need to normalize them to values between 0 and 1.
# This normalized profile describes the maximum activity per unit of power plant.
#
# Example: 10 PV units with 1 GW rated capacity each (as specified by the
# activity parameter) with an activity profile of 0.24 in hour 11 could produce
# up to 10 * 1 GWh/h * 0.24 = 2.4 GWh/h.

# %%
# "converter_activityProfile"


# load the profiles DataFrame, select its PV and WindOnshore columns
# load the profiles DataFrame, select its PV and WindOnshore columns
for data_node in ["CI_data","FJ_data","FSM_data", "KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"]:
    
    region_code = data_node.split("_")[0]  # "R1" or "R2"
    
    techs = ["PV_B", "WindOnshore_B", "Hydro_B", "Geothermal_B", "BG_B"]
    techs_region = [f"{t}_{region_code}" for t in techs]  # add R1 or R2 suffix

    # Select, convert MW→GW, transpose
    converter_activityProfile = profiles[techs_region].div(1e3).T

    # Rename back to original tech names
    converter_activityProfile.index = techs

    # Normalize
    converter_activityProfile = converter_activityProfile.div(
        converter_activityProfile.max(axis=1), axis=0
    )
    converter_activityProfile.index.names = ["techs"]

    # Add index columns
    converter_activityProfile["region"] = data_node
    converter_activityProfile["years"] = "2030"
    converter_activityProfile["type"] = "upper"

    converter_activityProfile = converter_activityProfile.reset_index().set_index(
        ["region", "years", "techs", "type"]
    )

    m.profile.add(converter_activityProfile, "converter_activityprofile")

converter_activityProfile.iloc[:, 0:8]


# %%
# "accounting_converterUnits"
# setting the costs of technologies
accounting_converterUnits = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [
            ["Invest", "OMFix"],
            ["global"],
            ["horizon"],
            ["DG", "BG_B", "PV_B", "WindOnshore_B", "Hydro_B", "Geothermal_B", "MDV","HDV", "LDV", "Bus", "Two_wheel", "Aviation", "Marine", "cook", "Industry", "DW_LPG_converter", "DW_Electric_converter"],
            m.set.yearssel,
        ]
    )
).sort_index()

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "DG", "2030"], "perUnitBuild"
] = 400  # Mio EUR per unit
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "DG", "2030"], "useAnnuity"
] = 1  # binary yes/no
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "DG", "2030"], "amorTime"
] = 2  # years
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "DG", "2030"], "interest"
] = 0.06  # percent/100
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "DG", "2030"], "perUnitTotal"
] = 160

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "NG_plant", "2030"], "perUnitBuild"
] = 0  # Mio EUR per unit
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "NG_plant", "2030"], "useAnnuity"
] = 1  # binary yes/no
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "NG_plant", "2030"], "amorTime"
] = 2  # years
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "NG_plant", "2030"], "interest"
] = 0.06  # percent/100
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "NG_plant", "2030"], "perUnitTotal"
] = 87.6

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "BG_B", "2030"], "perUnitBuild"
] = 0  # Mio EUR per unit
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "BG_B", "2030"], "useAnnuity"
] = 1  # binary yes/no
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "BG_B", "2030"], "amorTime"
] = 25  # years
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "BG_B", "2030"], "interest"
] = 0.06  # percent/100
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "BG_B", "2030"], "perUnitTotal"
] = 78


accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "PV_B", "2030"], "perUnitBuild"
] = 0  # Mio EUR per unit
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "PV_B", "2030"], "useAnnuity"
] = 1  # binary yes/no
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "PV_B", "2030"], "amorTime"
] = 25  # years
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "PV_B", "2030"], "interest"
] = 0.06  # percent/100
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "PV_B", "2030"], "perUnitTotal"
] = 14

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "WindOnshore_B", "2030"], "perUnitBuild"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "WindOnshore_B", "2030"], "useAnnuity"
] = 1
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "WindOnshore_B", "2030"], "amorTime"
] = 25
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "WindOnshore_B", "2030"], "interest"
] = 0.06
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "WindOnshore_B", "2030"], "perUnitTotal"
] = 22



accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Hydro_B", "2030"], "perUnitBuild"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Hydro_B", "2030"], "useAnnuity"
] = 1
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Hydro_B", "2030"], "amorTime"
] = 25
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Hydro_B", "2030"], "interest"
] = 0.06
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "Hydro_B", "2030"], "perUnitTotal"
] = 168 * 2.22 ## to balance our reduction of capacity by 55%, capacity *.45

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Geothermal_B", "2030"], "perUnitBuild"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Geothermal_B", "2030"], "useAnnuity"
] = 1
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Geothermal_B", "2030"], "amorTime"
] = 25
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Geothermal_B", "2030"], "interest"
] = 0.06
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "Geothermal_B", "2030"], "perUnitTotal"
] = 118 * 4.54


accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "MDV", "2030"], "perUnitBuild"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "MDV", "2030"], "useAnnuity"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "MDV", "2030"], "amorTime"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "MDV", "2030"], "interest"
] = 0
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "MDV", "2030"], "perUnitTotal"
] = 0

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "HDV", "2030"], "perUnitBuild"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "HDV", "2030"], "useAnnuity"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "HDV", "2030"], "amorTime"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "HDV", "2030"], "interest"
] = 0
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "HDV", "2030"], "perUnitTotal"
] = 0


accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "LDV", "2030"], "perUnitBuild"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "LDV", "2030"], "useAnnuity"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "LDV", "2030"], "amorTime"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "LDV", "2030"], "interest"
] = 0
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "LDV", "2030"], "perUnitTotal"
] = 0


accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Bus", "2030"], "perUnitBuild"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Bus", "2030"], "useAnnuity"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Bus", "2030"], "amorTime"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Bus", "2030"], "interest"
] = 0
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "Bus", "2030"], "perUnitTotal"
] = 0

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Two_wheel", "2030"], "perUnitBuild"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Two_wheel", "2030"], "useAnnuity"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Two_wheel", "2030"], "amorTime"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Two_wheel", "2030"], "interest"
] = 0
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "Two_wheel", "2030"], "perUnitTotal"
] = 0

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Aviation", "2030"], "perUnitBuild"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Aviation", "2030"], "useAnnuity"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Aviation", "2030"], "amorTime"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Aviation", "2030"], "interest"
] = 0
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "Aviation", "2030"], "perUnitTotal"
] = 0

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Marine", "2030"], "perUnitBuild"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Marine", "2030"], "useAnnuity"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Marine", "2030"], "amorTime"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Marine", "2030"], "interest"
] = 0
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "Marine", "2030"], "perUnitTotal"
] = 0


accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "cook", "2030"], "perUnitBuild"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "cook", "2030"], "useAnnuity"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "cook", "2030"], "amorTime"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "cook", "2030"], "interest"
] = 0
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "cook", "2030"], "perUnitTotal"
] = 0

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Industry", "2030"], "perUnitBuild"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Industry", "2030"], "useAnnuity"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Industry", "2030"], "amorTime"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Industry", "2030"], "interest"
] = 0
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "Industry", "2030"], "perUnitTotal"
] = 0

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "DW_LPG_converter", "2030"], "perUnitBuild"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "DW_LPG_converter", "2030"], "useAnnuity"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "DW_LPG_converter", "2030"], "amorTime"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "DW_LPG_converter", "2030"], "interest"
] = 0
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "DW_LPG_converter", "2030"], "perUnitTotal"
] = 0

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "DW_Electric_converter", "2030"], "perUnitBuild"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "DW_Electric_converter", "2030"], "useAnnuity"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "DW_Electric_converter", "2030"], "amorTime"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "DW_Electric_converter", "2030"], "interest"
] = 0
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "DW_Electric_converter", "2030"], "perUnitTotal"
] = 0
accounting_converterUnits = accounting_converterUnits.fillna(0)

m.parameter.add(accounting_converterUnits, "accounting_converterunits")
accounting_converterUnits
# %% [markdown]
# ### Sources and sinks
#
# #### Adding a demand profile as sink
#
# In this part, we set a demand for the data node `R1_data` (which is aggregated
# to the model node `R1_model`) only.
# The region name and year have to be included in the `map.aggregatenodesmodel`
# and `set.years` defined in the beginning.
# The name for the source-sink technology (here: `Demand`) can be freely chosen.
#
# We need to specify that the demand is applied to the electrical commodity and
# that this profile needs to be matched exactly on an hour-by-hour level.

# %%

# "sourcesink_profile"
demand_R4_R2_CH = profiles[["demand_CI", "demand_FJ","demand_FSM", "demand_KB", "demand_MI","demand_NU","demand_NE","demand_PU","demand_PNG","demand_SA","demand_SI","demand_TA","demand_TU","demand_VU"]]

demand_R4_R2_CH = demand_R4_R2_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R2_CH = demand_R4_R2_CH.T

demand_R4_R2_CH = demand_R4_R2_CH.rename(
    index={"demand_CI": "CI_data", "demand_FJ": "FJ_data", "demand_FSM": "FSM_data", "demand_KB": "KB_data", "demand_MI": "MI_data","demand_NU": "NU_data","demand_NE": "NE_data","demand_PU": "PU_data","demand_PNG": "PNG_data","demand_SA": "SA_data","demand_SI": "SI_data","demand_TA": "TA_data","demand_TU": "TU_data","demand_VU": "VU_data"}
)

# add columns and set them as index
demand_R4_R2_CH["years"] = "2030"
demand_R4_R2_CH["techs"] = "Demand"
demand_R4_R2_CH["commodity"] = "Elec"
demand_R4_R2_CH["type"] = "fixed"
demand_R4_R2_CH = demand_R4_R2_CH.set_index(
    ["years", "techs", "commodity", "type"], append=True
)

m.profile.add(demand_R4_R2_CH, "sourcesink_profile")
demand_R4_R2_CH.iloc[:, 0:8]

# load the profiles DataFrame, select the demand column
# %% [markdown]
# Now that we have created the profile, we need to create a config with the
# information that the created profile is going to be integrated into the model
# as fixed profile.

# %%
# "sourcesink_config" (demand configuration)
sourcesink_config = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [m.set.nodesdata, m.set.yearssel, ["Demand"], ["Elec"]]
    )
)
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesFixedProfile"] = 1
sourcesink_config = sourcesink_config.dropna()

m.parameter.add(sourcesink_config, "sourcesink_config")
sourcesink_config

####################
demand_R4_R3_CH = profiles[["MDV_CI","MDV_FJ","MDV_FSM","MDV_KB","MDV_MI","MDV_NU","MDV_NE","MDV_PU","MDV_PNG","MDV_SA","MDV_SI","MDV_TA","MDV_TU","MDV_VU"]]

demand_R4_R3_CH = demand_R4_R3_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R3_CH = demand_R4_R3_CH.T

demand_R4_R3_CH = demand_R4_R3_CH.rename(
    index={"MDV_CI": "CI_data", "MDV_FJ": "FJ_data", "MDV_FSM": "FSM_data", "MDV_KB": "KB_data", "MDV_MI": "MI_data","MDV_NU": "NU_data","MDV_NE": "NE_data","MDV_PU": "PU_data","MDV_PNG": "PNG_data","MDV_SA": "SA_data","MDV_SI": "SI_data","MDV_TA": "TA_data","MDV_TU": "TU_data","MDV_VU": "VU_data"}
)

# add columns and set them as index
demand_R4_R3_CH["years"] = "2030"
demand_R4_R3_CH["techs"] = "Demand"
demand_R4_R3_CH["commodity"] = "T_MDV"
demand_R4_R3_CH["type"] = "fixed"
demand_R4_R3_CH = demand_R4_R3_CH.set_index(
    ["years", "techs", "commodity", "type"], append=True
)

m.profile.add(demand_R4_R3_CH, "sourcesink_profile")
demand_R4_R3_CH.iloc[:, 0:8]

# load the profiles DataFrame, select the demand column
# %% [markdown]
# Now that we have created the profile, we need to create a config with the
# information that the created profile is going to be integrated into the model
# as fixed profile.

# %%
# "sourcesink_config" (demand configuration)
sourcesink_config = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [m.set.nodesdata, m.set.yearssel, ["Demand"], ["T_MDV"]]
    )
)
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesFixedProfile"] = 1
sourcesink_config = sourcesink_config.dropna()

m.parameter.add(sourcesink_config, "sourcesink_config")
sourcesink_config
########################################################

demand_R4_R4_CH = profiles[["HDV_CI", "HDV_FJ","HDV_FSM", "HDV_KB", "HDV_MI","HDV_NU","HDV_NE","HDV_PU","HDV_PNG","HDV_SA","HDV_SI","HDV_TA","HDV_TU","HDV_VU"]]

demand_R4_R4_CH = demand_R4_R4_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R4_CH = demand_R4_R4_CH.T

demand_R4_R4_CH = demand_R4_R4_CH.rename(
    index={"HDV_CI": "CI_data", "HDV_FJ": "FJ_data", "HDV_FSM": "FSM_data", "HDV_KB": "KB_data", "HDV_MI": "MI_data","HDV_NU": "NU_data","HDV_NE": "NE_data","HDV_PU": "PU_data","HDV_PNG": "PNG_data","HDV_SA": "SA_data","HDV_SI": "SI_data","HDV_TA": "TA_data","HDV_TU": "TU_data","HDV_VU": "VU_data"}
)

# add columns and set them as index
demand_R4_R4_CH["years"] = "2030"
demand_R4_R4_CH["techs"] = "Demand"
demand_R4_R4_CH["commodity"] = "T_HDV"
demand_R4_R4_CH["type"] = "fixed"
demand_R4_R4_CH = demand_R4_R4_CH.set_index(
    ["years", "techs", "commodity", "type"], append=True
)

m.profile.add(demand_R4_R4_CH, "sourcesink_profile")
demand_R4_R4_CH.iloc[:, 0:8]

# load the profiles DataFrame, select the demand column
# %% [markdown]
# Now that we have created the profile, we need to create a config with the
# information that the created profile is going to be integrated into the model
# as fixed profile.

# %%
# "sourcesink_config" (demand configuration)
sourcesink_config = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [m.set.nodesdata, m.set.yearssel, ["Demand"], ["T_HDV"]]
    )
)
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesFixedProfile"] = 1
sourcesink_config = sourcesink_config.dropna()

m.parameter.add(sourcesink_config, "sourcesink_config")
sourcesink_config
#############################################################

demand_R4_R5_CH = profiles[["LDV_CI", "LDV_FJ","LDV_FSM", "LDV_KB", "LDV_MI","LDV_NU","LDV_NE","LDV_PU","LDV_PNG","LDV_SA","LDV_SI","LDV_TA","LDV_TU","LDV_VU"]]

demand_R4_R5_CH = demand_R4_R5_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R5_CH = demand_R4_R5_CH.T

demand_R4_R5_CH = demand_R4_R5_CH.rename(
    index={"LDV_CI": "CI_data", "LDV_FJ": "FJ_data", "LDV_FSM": "FSM_data", "LDV_KB": "KB_data", "LDV_MI": "MI_data","LDV_NU": "NU_data","LDV_NE": "NE_data","LDV_PU": "PU_data","LDV_PNG": "PNG_data","LDV_SA": "SA_data","LDV_SI": "SI_data","LDV_TA": "TA_data","LDV_TU": "TU_data","LDV_VU": "VU_data"}
)

# add columns and set them as index
demand_R4_R5_CH["years"] = "2030"
demand_R4_R5_CH["techs"] = "Demand"
demand_R4_R5_CH["commodity"] = "T_LDV"
demand_R4_R5_CH["type"] = "fixed"
demand_R4_R5_CH = demand_R4_R5_CH.set_index(
    ["years", "techs", "commodity", "type"], append=True
)

m.profile.add(demand_R4_R5_CH, "sourcesink_profile")
demand_R4_R5_CH.iloc[:, 0:8]

# load the profiles DataFrame, select the demand column
# %% [markdown]
# Now that we have created the profile, we need to create a config with the
# information that the created profile is going to be integrated into the model
# as fixed profile.

# %%
# "sourcesink_config" (demand configuration)
sourcesink_config = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [m.set.nodesdata, m.set.yearssel, ["Demand"], ["T_LDV"]]
    )
)
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesFixedProfile"] = 1
sourcesink_config = sourcesink_config.dropna()

m.parameter.add(sourcesink_config, "sourcesink_config")
sourcesink_config
############################################################
demand_R4_R6_CH = profiles[["Bus_CI", "Bus_FJ","Bus_FSM", "Bus_KB", "Bus_MI","Bus_NU","Bus_NE","Bus_PU","Bus_PNG","Bus_SA","Bus_SI","Bus_TA","Bus_TU","Bus_VU"]]

demand_R4_R6_CH = demand_R4_R6_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R6_CH = demand_R4_R6_CH.T

demand_R4_R6_CH = demand_R4_R6_CH.rename(
    index={"Bus_CI": "CI_data", "Bus_FJ": "FJ_data", "Bus_FSM": "FSM_data", "Bus_KB": "KB_data", "Bus_MI": "MI_data","Bus_NU": "NU_data","Bus_NE": "NE_data","Bus_PU": "PU_data","Bus_PNG": "PNG_data","Bus_SA": "SA_data","Bus_SI": "SI_data","Bus_TA": "TA_data","Bus_TU": "TU_data","Bus_VU": "VU_data"}
)

# add columns and set them as index
demand_R4_R6_CH["years"] = "2030"
demand_R4_R6_CH["techs"] = "Demand"
demand_R4_R6_CH["commodity"] = "T_Bus"
demand_R4_R6_CH["type"] = "fixed"
demand_R4_R6_CH = demand_R4_R6_CH.set_index(
    ["years", "techs", "commodity", "type"], append=True
)

m.profile.add(demand_R4_R6_CH, "sourcesink_profile")
demand_R4_R6_CH.iloc[:, 0:8]

# load the profiles DataFrame, select the demand column
# %% [markdown]
# Now that we have created the profile, we need to create a config with the
# information that the created profile is going to be integrated into the model
# as fixed profile.

# %%
# "sourcesink_config" (demand configuration)
sourcesink_config = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [m.set.nodesdata, m.set.yearssel, ["Demand"], ["T_Bus"]]
    )
)
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesFixedProfile"] = 1
sourcesink_config = sourcesink_config.dropna()

m.parameter.add(sourcesink_config, "sourcesink_config")
sourcesink_config
############################################################
demand_R4_R7_CH = profiles[["Two_wheel_CI", "Two_wheel_FJ","Two_wheel_FSM", "Two_wheel_KB", "Two_wheel_MI","Two_wheel_NU","Two_wheel_NE","Two_wheel_PU","Two_wheel_PNG","Two_wheel_SA","Two_wheel_SI","Two_wheel_TA","Two_wheel_TU","Two_wheel_VU"]]

demand_R4_R7_CH = demand_R4_R7_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R7_CH = demand_R4_R7_CH.T

demand_R4_R7_CH = demand_R4_R7_CH.rename(
    index={"Two_wheel_CI": "CI_data", "Two_wheel_FJ": "FJ_data", "Two_wheel_FSM": "FSM_data", "Two_wheel_KB": "KB_data", "Two_wheel_MI": "MI_data","Two_wheel_NU": "NU_data","Two_wheel_NE": "NE_data","Two_wheel_PU": "PU_data","Two_wheel_PNG": "PNG_data","Two_wheel_SA": "SA_data","Two_wheel_SI": "SI_data","Two_wheel_TA": "TA_data","Two_wheel_TU": "TU_data","Two_wheel_VU": "VU_data"}
)

# add columns and set them as index
demand_R4_R7_CH["years"] = "2030"
demand_R4_R7_CH["techs"] = "Demand"
demand_R4_R7_CH["commodity"] = "T_Two_wheel"
demand_R4_R7_CH["type"] = "fixed"
demand_R4_R7_CH = demand_R4_R7_CH.set_index(
    ["years", "techs", "commodity", "type"], append=True
)

m.profile.add(demand_R4_R7_CH, "sourcesink_profile")
demand_R4_R7_CH.iloc[:, 0:8]

# load the profiles DataFrame, select the demand column
# %% [markdown]
# Now that we have created the profile, we need to create a config with the
# information that the created profile is going to be integrated into the model
# as fixed profile.

# %%
# "sourcesink_config" (demand configuration)
sourcesink_config = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [m.set.nodesdata, m.set.yearssel, ["Demand"], ["T_Two_wheel"]]
    )
)
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesFixedProfile"] = 1
sourcesink_config = sourcesink_config.dropna()

m.parameter.add(sourcesink_config, "sourcesink_config")
sourcesink_config
############################################################
demand_R4_R8_CH = profiles[["Marine_CI", "Marine_FJ","Marine_FSM", "Marine_KB", "Marine_MI","Marine_NU","Marine_NE","Marine_PU","Marine_PNG","Marine_SA","Marine_SI","Marine_TA","Marine_TU","Marine_VU"]]

demand_R4_R8_CH = demand_R4_R8_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R8_CH = demand_R4_R8_CH.T

demand_R4_R8_CH = demand_R4_R8_CH.rename(
    index={"Marine_CI": "CI_data", "Marine_FJ": "FJ_data", "Marine_FSM": "FSM_data", "Marine_KB": "KB_data", "Marine_MI": "MI_data","Marine_NU": "NU_data","Marine_NE": "NE_data","Marine_PU": "PU_data","Marine_PNG": "PNG_data","Marine_SA": "SA_data","Marine_SI": "SI_data","Marine_TA": "TA_data","Marine_TU": "TU_data","Marine_VU": "VU_data"}
)

# add columns and set them as index
demand_R4_R8_CH["years"] = "2030"
demand_R4_R8_CH["techs"] = "Demand"
demand_R4_R8_CH["commodity"] = "T_Marine"
demand_R4_R8_CH["type"] = "fixed"
demand_R4_R8_CH = demand_R4_R8_CH.set_index(
    ["years", "techs", "commodity", "type"], append=True
)

m.profile.add(demand_R4_R8_CH, "sourcesink_profile")
demand_R4_R8_CH.iloc[:, 0:8]

# load the profiles DataFrame, select the demand column
# %% [markdown]
# Now that we have created the profile, we need to create a config with the
# information that the created profile is going to be integrated into the model
# as fixed profile.

# %%
# "sourcesink_config" (demand configuration)
sourcesink_config = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [m.set.nodesdata, m.set.yearssel, ["Demand"], ["T_Marine"]]
    )
)
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesFixedProfile"] = 1
sourcesink_config = sourcesink_config.dropna()

m.parameter.add(sourcesink_config, "sourcesink_config")
sourcesink_config
#############################################################
demand_R4_R9_CH = profiles[["Aviation_CI", "Aviation_FJ","Aviation_FSM", "Aviation_KB", "Aviation_MI","Aviation_NU","Aviation_NE","Aviation_PU","Aviation_PNG","Aviation_SA","Aviation_SI","Aviation_TA","Aviation_TU","Aviation_VU"]]

demand_R4_R9_CH = demand_R4_R9_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R9_CH = demand_R4_R9_CH.T

demand_R4_R9_CH = demand_R4_R9_CH.rename(
    index={"Aviation_CI": "CI_data", "Aviation_FJ": "FJ_data", "Aviation_FSM": "FSM_data", "Aviation_KB": "KB_data", "Aviation_MI": "MI_data","Aviation_NU": "NU_data","Aviation_NE": "NE_data","Aviation_PU": "PU_data","Aviation_PNG": "PNG_data","Aviation_SA": "SA_data","Aviation_SI": "SI_data","Aviation_TA": "TA_data","Aviation_TU": "TU_data","Aviation_VU": "VU_data"}
)

# add columns and set them as index
demand_R4_R9_CH["years"] = "2030"
demand_R4_R9_CH["techs"] = "Demand"
demand_R4_R9_CH["commodity"] = "T_Aviation"
demand_R4_R9_CH["type"] = "fixed"
demand_R4_R9_CH = demand_R4_R9_CH.set_index(
    ["years", "techs", "commodity", "type"], append=True
)

m.profile.add(demand_R4_R9_CH, "sourcesink_profile")
demand_R4_R9_CH.iloc[:, 0:8]

# load the profiles DataFrame, select the demand column
# %% [markdown]
# Now that we have created the profile, we need to create a config with the
# information that the created profile is going to be integrated into the model
# as fixed profile.

# %%
# "sourcesink_config" (demand configuration)
sourcesink_config = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [m.set.nodesdata, m.set.yearssel, ["Demand"], ["T_Aviation"]]
    )
)
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesFixedProfile"] = 1
sourcesink_config = sourcesink_config.dropna()

m.parameter.add(sourcesink_config, "sourcesink_config")
sourcesink_config
###############################################################################
demand_R4_R10_CH = profiles[["HC_CI", "HC_FJ","HC_FSM", "HC_KB", "HC_MI","HC_NU","HC_NE","HC_PU","HC_PNG","HC_SA","HC_SI","HC_TA","HC_TU","HC_VU"]]

demand_R4_R10_CH = demand_R4_R10_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R10_CH = demand_R4_R10_CH.T

demand_R4_R10_CH = demand_R4_R10_CH.rename(
    index={"HC_CI": "CI_data", "HC_FJ": "FJ_data", "HC_FSM": "FSM_data", "HC_KB": "KB_data", "HC_MI": "MI_data","HC_NU": "NU_data","HC_NE": "NE_data","HC_PU": "PU_data","HC_PNG": "PNG_data","HC_SA": "SA_data","HC_SI": "SI_data","HC_TA": "TA_data","HC_TU": "TU_data","HC_VU": "VU_data"}
)

# add columns and set them as index
demand_R4_R10_CH["years"] = "2030"
demand_R4_R10_CH["techs"] = "Demand"
demand_R4_R10_CH["commodity"] = "Heat_cooking"
demand_R4_R10_CH["type"] = "fixed"
demand_R4_R10_CH = demand_R4_R10_CH.set_index(
    ["years", "techs", "commodity", "type"], append=True
)

m.profile.add(demand_R4_R10_CH, "sourcesink_profile")
demand_R4_R10_CH.iloc[:, 0:8]

# load the profiles DataFrame, select the demand column
# %% [markdown]
# Now that we have created the profile, we need to create a config with the
# information that the created profile is going to be integrated into the model
# as fixed profile.

# %%
# "sourcesink_config" (demand configuration)
sourcesink_config = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [m.set.nodesdata, m.set.yearssel, ["Demand"], ["Heat_cooking"]]
    )
)
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesFixedProfile"] = 1
sourcesink_config = sourcesink_config.dropna()

m.parameter.add(sourcesink_config, "sourcesink_config")
sourcesink_config
###########################################################
demand_R4_R10_CH = profiles[["HI_CI", "HI_FJ","HI_FSM", "HI_KB", "HI_MI","HI_NU","HI_NE","HI_PU","HI_PNG","HI_SA","HI_SI","HI_TA","HI_TU","HI_VU"]]

demand_R4_R10_CH = demand_R4_R10_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R10_CH = demand_R4_R10_CH.T

demand_R4_R10_CH = demand_R4_R10_CH.rename(
    index={"HI_CI": "CI_data", "HI_FJ": "FJ_data", "HI_FSM": "FSM_data", "HI_KB": "KB_data", "HI_MI": "MI_data","HI_NU": "NU_data","HI_NE": "NE_data","HI_PU": "PU_data","HI_PNG": "PNG_data","HI_SA": "SA_data","HI_SI": "SI_data","HI_TA": "TA_data","HI_TU": "TU_data","HI_VU": "VU_data"}
)

# add columns and set them as index
demand_R4_R10_CH["years"] = "2030"
demand_R4_R10_CH["techs"] = "Demand"
demand_R4_R10_CH["commodity"] = "Heat_industry"
demand_R4_R10_CH["type"] = "fixed"
demand_R4_R10_CH = demand_R4_R10_CH.set_index(
    ["years", "techs", "commodity", "type"], append=True
)

m.profile.add(demand_R4_R10_CH, "sourcesink_profile")
demand_R4_R10_CH.iloc[:, 0:8]

# load the profiles DataFrame, select the demand column
# %% [markdown]
# Now that we have created the profile, we need to create a config with the
# information that the created profile is going to be integrated into the model
# as fixed profile.

# %%
# "sourcesink_config" (demand configuration)
sourcesink_config = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [m.set.nodesdata, m.set.yearssel, ["Demand"], ["Heat_industry"]]
    )
)
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesFixedProfile"] = 1
sourcesink_config = sourcesink_config.dropna()

m.parameter.add(sourcesink_config, "sourcesink_config")
sourcesink_config
################################################################
demand_R4_R10_CH = profiles[["DHWE_CI", "DHWE_FJ","DHWE_FSM", "DHWE_KB", "DHWE_MI","DHWE_NU","DHWE_NE","DHWE_PU","DHWE_PNG","DHWE_SA","DHWE_SI","DHWE_TA","DHWE_TU","DHWE_VU"]]

demand_R4_R10_CH = demand_R4_R10_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R10_CH = demand_R4_R10_CH.T

demand_R4_R10_CH = demand_R4_R10_CH.rename(
    index={"DHWE_CI": "CI_data", "DHWE_FJ": "FJ_data", "DHWE_FSM": "FSM_data", "DHWE_KB": "KB_data", "DHWE_MI": "MI_data","DHWE_NU": "NU_data","DHWE_NE": "NE_data","DHWE_PU": "PU_data","DHWE_PNG": "PNG_data","DHWE_SA": "SA_data","DHWE_SI": "SI_data","DHWE_TA": "TA_data","DHWE_TU": "TU_data","DHWE_VU": "VU_data"}
)

# add columns and set them as index
demand_R4_R10_CH["years"] = "2030"
demand_R4_R10_CH["techs"] = "Demand"
demand_R4_R10_CH["commodity"] = "DHW_Elec"
demand_R4_R10_CH["type"] = "fixed"
demand_R4_R10_CH = demand_R4_R10_CH.set_index(
    ["years", "techs", "commodity", "type"], append=True
)

m.profile.add(demand_R4_R10_CH, "sourcesink_profile")
demand_R4_R10_CH.iloc[:, 0:8]

# load the profiles DataFrame, select the demand column
# %% [markdown]
# Now that we have created the profile, we need to create a config with the
# information that the created profile is going to be integrated into the model
# as fixed profile.

# %%
# "sourcesink_config" (demand configuration)
sourcesink_config = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [m.set.nodesdata, m.set.yearssel, ["Demand"], ["DHW_Elec"]]
    )
)
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesFixedProfile"] = 1
sourcesink_config = sourcesink_config.dropna()

m.parameter.add(sourcesink_config, "sourcesink_config")
sourcesink_config
#############################################################
demand_R4_R10_CH = profiles[["DHWL_CI", "DHWL_FJ","DHWL_FSM", "DHWL_KB", "DHWL_MI","DHWL_NU","DHWL_NE","DHWL_PU","DHWL_PNG","DHWL_SA","DHWL_SI","DHWL_TA","DHWL_TU","DHWL_VU"]]

demand_R4_R10_CH = demand_R4_R10_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R10_CH = demand_R4_R10_CH.T

demand_R4_R10_CH = demand_R4_R10_CH.rename(
    index={"DHWL_CI": "CI_data", "DHWL_FJ": "FJ_data", "DHWL_FSM": "FSM_data", "DHWL_KB": "KB_data", "DHWL_MI": "MI_data","DHWL_NU": "NU_data","DHWL_NE": "NE_data","DHWL_PU": "PU_data","DHWL_PNG": "PNG_data","DHWL_SA": "SA_data","DHWL_SI": "SI_data","DHWL_TA": "TA_data","DHWL_TU": "TU_data","DHWL_VU": "VU_data"}
)

# add columns and set them as index
demand_R4_R10_CH["years"] = "2030"
demand_R4_R10_CH["techs"] = "Demand"
demand_R4_R10_CH["commodity"] = "DHW_LPG"
demand_R4_R10_CH["type"] = "fixed"
demand_R4_R10_CH = demand_R4_R10_CH.set_index(
    ["years", "techs", "commodity", "type"], append=True
)

m.profile.add(demand_R4_R10_CH, "sourcesink_profile")
demand_R4_R10_CH.iloc[:, 0:8]

# load the profiles DataFrame, select the demand column
# %% [markdown]
# Now that we have created the profile, we need to create a config with the
# information that the created profile is going to be integrated into the model
# as fixed profile.

# %%
# "sourcesink_config" (demand configuration)
sourcesink_config = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [m.set.nodesdata, m.set.yearssel, ["Demand"], ["DHW_LPG"]]
    )
)
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesFixedProfile"] = 1
sourcesink_config = sourcesink_config.dropna()

m.parameter.add(sourcesink_config, "sourcesink_config")
sourcesink_config
# %% [markdown]
# #### Add sources for fuels and sinks for carbon emissions
#
# Since CCGT uses CH4 as a fuel, we need to allow import of CH4 for the model
# region `R1_model` (since the technology is only installed there).
# This is very similar to the source-sink technology we used for the electrical
# demand.
# However, in this case we want to be able to import an unlimited amount of fuel
# at a fixed price of 0.0306 million EUR/GWh_ch.
# By adding a lower profile of 0, we ensure the model cannot export fuel to make
# money.

# %%
# "sourcesink_annualSum"
# limiting the annual sum of fuel imports into a model region

# %%
# "sourcesink_config" (import configuration)


# User inputs upper limits for Biomass for each node (order matches m.set.nodesdata)
#biomass_limits = [10000, 10000, 10000, 10000, 10000,10000,10000,10000, 10000,10000, 10000, 10000,10000, 10000] 
biomass_limits = [12, 2380, 168, 221, 22,5,4,11330, 1, 295, 1507, 211, 9, 671] 
#Dictionary####dic - pd df 
# GW or other units for R1_data, R2_data
lower_limit = 0  # same for all in this example

sourcesink_annualSum = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [m.set.nodesdata, m.set.yearssel, ["FuelImport"], ["Biomass"]]
    )
)

for node, limit in zip(m.set.nodesdata, biomass_limits):
    sourcesink_annualSum.loc[idx[node, :, :, :], "upper"] = limit
    sourcesink_annualSum.loc[idx[node, :, :, :], "lower"] = lower_limit

sourcesink_annualSum = sourcesink_annualSum.dropna()

m.parameter.add(sourcesink_annualSum, "sourcesink_annualsum")
###############################################################

NG_limits = [1000000, 1000000, 1000000, 1000000, 1000000,1000000,100000,1000000, 1000000,1000000, 1000000, 1000000,1000000, 1000000] 
#Dictionary####dic - pd df 
# GW or other units for R1_data, R2_data
lower_limit = 0  # same for all in this example

sourcesink_annualSum = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [m.set.nodesdata, m.set.yearssel, ["FuelImport"], ["NG"]]
    )
)

for node, limit in zip(m.set.nodesdata, NG_limits):
    sourcesink_annualSum.loc[idx[node, :, :, :], "upper"] = limit
    sourcesink_annualSum.loc[idx[node, :, :, :], "lower"] = lower_limit

sourcesink_annualSum = sourcesink_annualSum.dropna()

m.parameter.add(sourcesink_annualSum, "sourcesink_annualsum")
###########################################################
Diesel_limits = [1000000, 1000000, 1000000, 1000000, 1000000,1000000,100000,1000000, 1000000,1000000, 1000000, 1000000,1000000, 1000000] 
#Dictionary####dic - pd df 
# GW or other units for R1_data, R2_data
lower_limit = 0  # same for all in this example

sourcesink_annualSum = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [m.set.nodesdata, m.set.yearssel, ["FuelImport"], ["Diesel"]]
    )
)

for node, limit in zip(m.set.nodesdata, Diesel_limits):
    sourcesink_annualSum.loc[idx[node, :, :, :], "upper"] = limit
    sourcesink_annualSum.loc[idx[node, :, :, :], "lower"] = lower_limit

sourcesink_annualSum = sourcesink_annualSum.dropna()

m.parameter.add(sourcesink_annualSum, "sourcesink_annualsum")
#############################################################
LPG_limits = [1000000, 1000000, 1000000, 1000000, 1000000,1000000,100000,1000000, 1000000,1000000, 1000000, 1000000,1000000, 1000000] 
#Dictionary####dic - pd df 
# GW or other units for R1_data, R2_data
lower_limit = 0  # same for all in this example

sourcesink_annualSum = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [m.set.nodesdata, m.set.yearssel, ["FuelImport"], ["LPG"]]
    )
)

for node, limit in zip(m.set.nodesdata, Diesel_limits):
    sourcesink_annualSum.loc[idx[node, :, :, :], "upper"] = limit
    sourcesink_annualSum.loc[idx[node, :, :, :], "lower"] = lower_limit

sourcesink_annualSum = sourcesink_annualSum.dropna()

m.parameter.add(sourcesink_annualSum, "sourcesink_annualsum")
############################################################
Gasoline_limits = [1000000, 1000000, 1000000, 1000000, 1000000,1000000,100000,1000000, 1000000,1000000, 1000000, 1000000,1000000, 1000000]
#Dictionary####dic - pd df 
# GW or other units for R1_data, R2_data
lower_limit = 0  # same for all in this example

sourcesink_annualSum = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [m.set.nodesdata, m.set.yearssel, ["FuelImport"], ["Gasoline"]]
    )
)

for node, limit in zip(m.set.nodesdata, Gasoline_limits):
    sourcesink_annualSum.loc[idx[node, :, :, :], "upper"] = limit
    sourcesink_annualSum.loc[idx[node, :, :, :], "lower"] = lower_limit

sourcesink_annualSum = sourcesink_annualSum.dropna()

m.parameter.add(sourcesink_annualSum, "sourcesink_annualsum")
#############################################################
JetA1_limits = [1000000, 1000000, 1000000, 1000000, 1000000,1000000,100000,1000000, 1000000,1000000, 1000000, 1000000,1000000, 1000000] 
#Dictionary####dic - pd df 
# GW or other units for R1_data, R2_data
lower_limit = 0  # same for all in this example

sourcesink_annualSum = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [m.set.nodesdata, m.set.yearssel, ["FuelImport"], ["JetA1"]]
    )
)

for node, limit in zip(m.set.nodesdata, JetA1_limits):
    sourcesink_annualSum.loc[idx[node, :, :, :], "upper"] = limit
    sourcesink_annualSum.loc[idx[node, :, :, :], "lower"] = lower_limit

sourcesink_annualSum = sourcesink_annualSum.dropna()

m.parameter.add(sourcesink_annualSum, "sourcesink_annualsum")
############################################################
MDO_limits = [1000000, 1000000, 1000000, 1000000, 1000000,1000000,100000,1000000, 1000000,1000000, 1000000, 1000000,1000000, 1000000] 
#Dictionary####dic - pd df 
# GW or other units for R1_data, R2_data
lower_limit = 0  # same for all in this example

sourcesink_annualSum = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [m.set.nodesdata, m.set.yearssel, ["FuelImport"], ["MDO"]]
    )
)

for node, limit in zip(m.set.nodesdata, MDO_limits):
    sourcesink_annualSum.loc[idx[node, :, :, :], "upper"] = limit
    sourcesink_annualSum.loc[idx[node, :, :, :], "lower"] = lower_limit

sourcesink_annualSum = sourcesink_annualSum.dropna()

m.parameter.add(sourcesink_annualSum, "sourcesink_annualsum")
# %%
# "sourcesink_config" (import configuration)
sourcesink_config = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [m.set.nodesdata, m.set.yearssel, ["FuelImport"], ["Biomass"]]
    )
)
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesUpperSum"] = 1
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesLowerProfile"] = 1
sourcesink_config = sourcesink_config.dropna()

m.parameter.add(sourcesink_config, "sourcesink_config")
sourcesink_config
################################################################################
sourcesink_config = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [m.set.nodesdata, m.set.yearssel, ["FuelImport"], ["NG"]]
    )
)
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesUpperSum"] = 1
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesLowerProfile"] = 1
sourcesink_config = sourcesink_config.dropna()

m.parameter.add(sourcesink_config, "sourcesink_config")
sourcesink_config
#################################################################################
sourcesink_config = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [m.set.nodesdata, m.set.yearssel, ["FuelImport"], ["Diesel"]]
    )
)
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesUpperSum"] = 1
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesLowerProfile"] = 1
sourcesink_config = sourcesink_config.dropna()

m.parameter.add(sourcesink_config, "sourcesink_config")
sourcesink_config
###############################################################################
sourcesink_config = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [m.set.nodesdata, m.set.yearssel, ["FuelImport"], ["LPG"]]
    )
)
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesUpperSum"] = 1
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesLowerProfile"] = 1
sourcesink_config = sourcesink_config.dropna()

m.parameter.add(sourcesink_config, "sourcesink_config")
sourcesink_config
###############################################################################
sourcesink_config = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [m.set.nodesdata, m.set.yearssel, ["FuelImport"], ["Gasoline"]]
    )
)
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesUpperSum"] = 1
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesLowerProfile"] = 1
sourcesink_config = sourcesink_config.dropna()

m.parameter.add(sourcesink_config, "sourcesink_config")
sourcesink_config
###############################################################################
sourcesink_config = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [m.set.nodesdata, m.set.yearssel, ["FuelImport"],["JetA1"]]
    )
)
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesUpperSum"] = 1
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesLowerProfile"] = 1
sourcesink_config = sourcesink_config.dropna()

m.parameter.add(sourcesink_config, "sourcesink_config")
sourcesink_config
###############################################################################
sourcesink_config = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [m.set.nodesdata, m.set.yearssel, ["FuelImport"], ["MDO"]]
    )
)
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesUpperSum"] = 1
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesLowerProfile"] = 1
sourcesink_config = sourcesink_config.dropna()

m.parameter.add(sourcesink_config, "sourcesink_config")
sourcesink_config
##################################################################################
# %%
# "accounting_sourcesinkFlow"
# setting a cost for methane imports
# User inputs perFlow prices for Biomass for each node
biomass_prices = [0.032, 0.032, 0.032, 0.032,0.032, 0.032, 0.032, 0.032, 0.032, 0.032, 0.032, 0.032,0.032, 0.032]  # Mio EUR per GWh_ch CH4 for R1_data, R2_data

accounting_sourcesinkFlow = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [["FuelCost"], m.set.nodesdata, m.set.yearssel, ["FuelImport"], ["Biomass"]]
    )
)

for node, price in zip(m.set.nodesdata, biomass_prices):
    accounting_sourcesinkFlow.loc[idx["FuelCost", node, :, :, :], "perFlow"] = price

accounting_sourcesinkFlow = accounting_sourcesinkFlow.dropna()

m.parameter.add(accounting_sourcesinkFlow, "accounting_sourcesinkflow")
########################################################################
NG_prices = [0.027, 0.027, 0.027, 0.027,0.027, 0.027, 0.027, 0.027, 0.027, 0.027, 0.027, 0.027,0.027, 0.027]  # Mio EUR per GWh_ch CH4 for R1_data, R2_data

accounting_sourcesinkFlow = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [["FuelCost"], m.set.nodesdata, m.set.yearssel, ["FuelImport"], ["NG"]]
    )
)

for node, price in zip(m.set.nodesdata, NG_prices):
    accounting_sourcesinkFlow.loc[idx["FuelCost", node, :, :, :], "perFlow"] = price

accounting_sourcesinkFlow = accounting_sourcesinkFlow.dropna()

m.parameter.add(accounting_sourcesinkFlow, "accounting_sourcesinkflow")
########################################################################
Diesel_prices = [0.095, 0.095, 0.095, 0.095,0.095, 0.095, 0.095, 0.095, 0.095, 0.095, 0.095, 0.095,0.095, 0.095]  # Mio EUR per GWh_ch CH4 for R1_data, R2_data

accounting_sourcesinkFlow = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [["FuelCost"], m.set.nodesdata, m.set.yearssel, ["FuelImport"], ["Diesel"]]
    )
)

for node, price in zip(m.set.nodesdata, Diesel_prices):
    accounting_sourcesinkFlow.loc[idx["FuelCost", node, :, :, :], "perFlow"] = price

accounting_sourcesinkFlow = accounting_sourcesinkFlow.dropna()

m.parameter.add(accounting_sourcesinkFlow, "accounting_sourcesinkflow")
########################################################################
LPG_prices = [0.260, 0.260, 0.260, 0.260,0.260, 0.260, 0.260, 0.260, 0.260, 0.260, 0.260, 0.260,0.260, 0.260] # Mio EUR per GWh_ch CH4 for R1_data, R2_data

accounting_sourcesinkFlow = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [["FuelCost"], m.set.nodesdata, m.set.yearssel, ["FuelImport"], ["LPG"]]
    )
)

for node, price in zip(m.set.nodesdata, LPG_prices):
    accounting_sourcesinkFlow.loc[idx["FuelCost", node, :, :, :], "perFlow"] = price

accounting_sourcesinkFlow = accounting_sourcesinkFlow.dropna()

m.parameter.add(accounting_sourcesinkFlow, "accounting_sourcesinkflow")
#########################################################################
Gasoline_prices = [0.105, 0.105, 0.105, 0.105,0.105, 0.105, 0.105, 0.105, 0.105, 0.105, 0.105, 0.105,0.105, 0.105]  # Mio EUR per GWh_ch CH4 for R1_data, R2_data

accounting_sourcesinkFlow = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [["FuelCost"], m.set.nodesdata, m.set.yearssel, ["FuelImport"], ["Gasoline"]]
    )
)

for node, price in zip(m.set.nodesdata, Gasoline_prices):
    accounting_sourcesinkFlow.loc[idx["FuelCost", node, :, :, :], "perFlow"] = price

accounting_sourcesinkFlow = accounting_sourcesinkFlow.dropna()

m.parameter.add(accounting_sourcesinkFlow, "accounting_sourcesinkflow")
#########################################################################
JetA1_prices = [0.05, 0.05, 0.05, 0.05,0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,0.05, 0.05]  # Mio EUR per GWh_ch CH4 for R1_data, R2_data

accounting_sourcesinkFlow = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [["FuelCost"], m.set.nodesdata, m.set.yearssel, ["FuelImport"], ["JetA1"]]
    )
)

for node, price in zip(m.set.nodesdata, JetA1_prices):
    accounting_sourcesinkFlow.loc[idx["FuelCost", node, :, :, :], "perFlow"] = price

accounting_sourcesinkFlow = accounting_sourcesinkFlow.dropna()

m.parameter.add(accounting_sourcesinkFlow, "accounting_sourcesinkflow")
#########################################################################
MDO_prices = [0.045, 0.045, 0.045, 0.045,0.045, 0.045, 0.045, 0.045, 0.045, 0.045, 0.045, 0.045,0.045, 0.045]  # Mio EUR per GWh_ch CH4 for R1_data, R2_data

accounting_sourcesinkFlow = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [["FuelCost"], m.set.nodesdata, m.set.yearssel, ["FuelImport"], ["MDO"]]
    )
)

for node, price in zip(m.set.nodesdata, MDO_prices):
    accounting_sourcesinkFlow.loc[idx["FuelCost", node, :, :, :], "perFlow"] = price

accounting_sourcesinkFlow = accounting_sourcesinkFlow.dropna()

m.parameter.add(accounting_sourcesinkFlow, "accounting_sourcesinkflow")
# %% [markdown]
# Similar to the fuel source we need to specify a sink for our carbon emissions.
# In this case we need to use negative values since the carbon is leaving our
# frame of accounting. So we specify a lower sum of -infinity and an upper
# profile of 0 (meaning we are not allowed to extract carbon out of the
# atmosphere).
# By changing the condition from -infinity to -100, we could also impose a
# carbon limit of 100 kilotonnes of CO2.
# Or we could add a new indicator "CarbonCost" (at the top) which accounts for
# the carbon flow out of the system and imposes an associated cost.

# %%
# "sourcesink_annualSum"
# limiting annual sum of carbon emissions
sourcesink_annualSum = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [m.set.nodesdata, m.set.yearssel, ["Emission"], ["CO2"]]
    )
)
sourcesink_annualSum.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"],  :, :, :], "lower"] = -np.inf
sourcesink_annualSum = sourcesink_annualSum.dropna()

m.parameter.add(sourcesink_annualSum, "sourcesink_annualsum")
sourcesink_annualSum
# %%
# "sourcesink_config" (emission configuration)
sourcesink_config = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [m.set.nodesdata, m.set.yearssel, ["Emission"], ["CO2"]]
    )
)
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesLowerSum"] = 1
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesUpperProfile"] = 1
sourcesink_config = sourcesink_config.dropna()

m.parameter.add(sourcesink_config, "sourcesink_config")
sourcesink_config
# %% [markdown]
# ### Writing DataFrames to files
#
# In this section we collect the DataFrames from the previous sections and
# convert them to files inside the folder which was specified in the
# beginning or the `data/` directory by default. Writing to `*.csv` files will
# work similarly.

# importing dependencies
from remix.framework import Instance
import pandas as pd
import pathlib as pt


# %% [markdown]
# ### Adding a storage technology
#
# After loading the model and dependencies from our base model (i.e.
# `tutorial_101a_build.py`), we can now simply add the components of the
# storage.
#
# Storage technologies are typically comprised of two parts:
# (1) the energy storage itself;
# (2) the component for charging and discharging the storage.
#
# Similarly, in REMix the storages are also built on top of two different
# components.
# A storage converter for charging and discharging a storage reservoir and that
# reservoir itself that contains the chosen commodity (in this case
# electricity).
#
# #### The charging/discharging unit (=converter)
#
# First, we will define the storage converter, i.e. the charging/discharging
# unit.
#
# We can use the same features we used for the converters of conventional power
# plants.
# The difference is that a storage by definition converts one commodity into the
# same commodity (e.g. electricity to electricity).
#
# As an example, we introduce a lithium-ion battery as electricity storage.

# %%
# "converter_techParam"

# %% [markdown]
# In contrast to the previous modeling of converter units for conventional power plants, we now need to define a
# reversible activity. In this example, we can both charge and discharge our lithium-ion battery with the same power
# unit. Therefore, we add both activities---`Charge` and `Discharge`---and use the coefficients to model the
# corresponding losses.
#
# We can also use two different converters for charging and discharging. This is necessary when wanting to better
# represent the real-world difference between the turbine and optional pumps in hydroelectric power plants for example.
# These can then also have different rated powers.
#
# A storage in REMix per definition has the same input and output commodity. To be able to account for storage losses,
# it is necessary to define a dummy commodity (here called `Elec_LiIon`), which is only used inside that one technology.
#
# In this tutorial, we fill the two activities of our single converter unit for charging and discharging so that each
# process has an efficiency of 95 %.

# %%
# "converter_coefficient"

# %% [markdown]
# #### The storage reservoir
#
# The storage features are always connected to a node and commodity combination
# and allow storing the connected commodity freely up to the rated capacity of
# the storage reservoir.
# We account for storage units in the same manner as for converter units and use
# a rated capacity to connect the units to a commodity and size.
# Storage technologies and converter technologies have the same name to make it
# easier to represent them as the same technology.

# %%

# %% [markdown]
# Now we can set the storage reservoir upper limit to 30 units for a specific
# model region, therefore the model can build up to 240 GWh_ch of storage
# reservoir (8 GWh_ch / unit * 30 units = 240 GWh_ch).

# %%
# "storage_reservoirParam"
# installed storage reservoir units

# %%
# write all files to `data/` directory
m.write(fileformat="dat")
# %% [markdown]
# That's it. We have successfully added a lithium-ion battery as storage
# technology to our model. We can now start a GAMS optimization run (part b).


#########################################
# %% [markdown]
# (tutorial_102_label)=
#
# # Tutorial 102 - Storage technologies
#
# <div style="text-align: center;">
#
# ![Model overview for tutorial 102](../../img/REMix_tutorial102.svg "Model overview for tutorial 102")
#
# Model overview of tutorial 102
#
# </div>
#
# ## Part a: setting up the model
#
# In this tutorial we have a closer look at **storage technologies**.
# In the first tutorial we had renewable energies in the system and checked two
# weeks with the highest and lowest renewable generation.
# The feed-in from renewable energies was mainly limited by the feed-in
# profiles.
# As a next step, in this tutorial we include technologies to store the
# electrical energy from the volatile renewable sources and thus add a
# flexibility.
#
# As mentioned during tutorial_101, we will use it as a base model here by
# reading its files into an Instance object `m` and adding a storage technology
# to it.

# %%

# %% [markdown]
# That's it. We have successfully added a lithium-ion battery as storage
# technology to our model. We can now start a GAMS optimization run (part b).
m.run(
    resultfile="PIC_Basecase_2020",
    lo=3,
    postcalc=1,
    roundts=1,) 