# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 10:33:50 2025

@author: ajh287
"""

# -*- coding: utf-8 -*-
"""

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
profiles = pd.read_csv("../_input/IP_2040_2050_14_PIC.csv", index_col=0)
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
    ["2020", "2030", "2040", "2050"], "years"
)  # must include all years that data is provided for in the model
# "set_yearsSel"
m.set.add(["2020", "2030", "2040", "2050"], "yearssel") 





 # years to be optimised
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
# into account all years in the set `set.yearssel` (here only 2020).
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
                "FuelCost","OMVar"
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
    "DG": {"lifeTime": 25, "activityUpperLimit": 1}, # No feed-in
    "NG_plant": {"lifeTime": 25, "activityUpperLimit": 1},
    "BG_B": {"lifeTime": 25, "activityUpperLimit": 0},  
    "PV_B": {"lifeTime": 25, "activityUpperLimit": 0},  # Feed-in
    "WindOnshore_B": {"lifeTime": 25, "activityUpperLimit": 0},
    "Hydro_B": {"lifeTime": 50, "activityUpperLimit": 0},
    "Geothermal_B": {"lifeTime": 50, "activityUpperLimit": 0},
    "MDV": {"lifeTime": 35, "activityUpperLimit": 1},
    "HDV": {"lifeTime": 35, "activityUpperLimit": 1},
    "LDV": {"lifeTime": 35, "activityUpperLimit": 1},
    "Bus": {"lifeTime": 35, "activityUpperLimit": 1},
    "Two_wheel": {"lifeTime": 35, "activityUpperLimit": 1},
    "Aviation": {"lifeTime": 35, "activityUpperLimit": 1},
    "Marine": {"lifeTime": 35, "activityUpperLimit": 1},
    "cook_b": {"lifeTime": 35, "activityUpperLimit": 1},
    "Industry": {"lifeTime": 35, "activityUpperLimit": 1},
    "DW_LPG_converter": {"lifeTime": 35, "activityUpperLimit": 1},
    "DW_Electric_converter": {"lifeTime": 35, "activityUpperLimit": 1},
    "HFO": {"lifeTime": 35, "activityUpperLimit": 1},
}



# Create DataFrame
converter_techParam = pd.DataFrame(
    index=pd.MultiIndex.from_product([list(tech_specs.keys()), ['2020']])
)

# Assign values from dictionary
for tech, specs in tech_specs.items():
    converter_techParam.loc[idx[tech], "lifeTime"] = specs["lifeTime"]
    converter_techParam.loc[idx[tech], "activityUpperLimit"] = specs["activityUpperLimit"]

# Add to model
m.parameter.add(converter_techParam, "converter_techparam")
converter_techParam


tech_specss = {
    "BG_N": {"lifeTime": 25, "activityUpperLimit": 0},  # No feed-in
    "PV_N": {"lifeTime": 25, "activityUpperLimit": 0},
    "WindOnshore_N": {"lifeTime": 25, "activityUpperLimit": 0},
    "Hydro_N": {"lifeTime": 50, "activityUpperLimit": 0},
    "Wave_N": {"lifeTime": 25, "activityUpperLimit": 0},
#    "ST_N": {"lifeTime": 25, "activityUpperLimit": 0, "mipUnits": 0},
    "DW_Electric_converter_D": {"lifeTime": 25, "activityUpperLimit": 1},
#    "LDV_BF": {"lifeTime": 25, "activityUpperLimit": 1, "mipUnits": 0},
#    "RO": {"lifeTime": 25, "activityUpperLimit": 1, "mipUnits": 0},
#    "AEL": {"lifeTime": 25, "activityUpperLimit": 1, "mipUnits": 0},
#    "AEL_10": {"lifeTime": 25,"activityUpperLimit": 1,"mipUnits": 0},
#    "AEL_100": {"lifeTime": 25, "activityUpperLimit": 1,"mipUnits": 0},
#    "Ammonia_synthesis": {"lifeTime": 25, "activityUpperLimit": 1, "mipUnits": 0},   
#    "DAC": {"lifeTime": 25, "activityUpperLimit": 1, "mipUnits": 0}, 
#    "Methanol_synthesis": {"lifeTime": 25, "activityUpperLimit": 1, "mipUnits": 0},
#    "HP": {"lifeTime": 25, "activityUpperLimit": 1, "mipUnits": 0},
#    "FTL": {"lifeTime": 25, "activityUpperLimit": 1, "mipUnits": 0},
    "WindOffshore_N": {"lifeTime": 25, "activityUpperLimit": 0}

}

# Create DataFrame
converter_techParam = pd.DataFrame(
    index=pd.MultiIndex.from_product([list(tech_specss.keys()), ['2030']])
)

# Assign values from dictionary
for tech, specs in tech_specss.items():
    converter_techParam.loc[idx[tech], "lifeTime"] = specs["lifeTime"]
    converter_techParam.loc[idx[tech], "activityUpperLimit"] = specs["activityUpperLimit"]


# Add to model
m.parameter.add(converter_techParam, "converter_techparam")
converter_techParam
################################################################################

tech_specss = {
    "BG_N": {"lifeTime": 25, "activityUpperLimit": 0},  # No feed-in
    "PV_N": {"lifeTime": 25, "activityUpperLimit": 0},
    "WindOnshore_N": {"lifeTime": 25, "activityUpperLimit": 0},
    "Hydro_N": {"lifeTime": 50, "activityUpperLimit": 0},
    "Wave_N": {"lifeTime": 25, "activityUpperLimit": 0},
    "ST_N": {"lifeTime": 25, "activityUpperLimit": 0},
    "DW_Electric_converter_2": {"lifeTime": 25, "activityUpperLimit": 1},
    "LDV_BF": {"lifeTime": 25, "activityUpperLimit": 1},
    "RO": {"lifeTime": 25, "activityUpperLimit": 1},
#    "AEL": {"lifeTime": 25, "activityUpperLimit": 1},
#    "AEL_10": {"lifeTime": 25,"activityUpperLimit": 1},
    "AEL_100": {"lifeTime": 25, "activityUpperLimit": 1},
    "Ammonia_synthesis": {"lifeTime": 25, "activityUpperLimit": 1},   
    "DAC": {"lifeTime": 25, "activityUpperLimit": 1}, 
    "Methanol_synthesis": {"lifeTime": 25, "activityUpperLimit": 1},
    "HP": {"lifeTime": 25, "activityUpperLimit": 1},
    "FTL": {"lifeTime": 25, "activityUpperLimit": 1},
    "LDV_el": {"lifeTime": 25, "activityUpperLimit": 1},
    "HDV_el": {"lifeTime": 25, "activityUpperLimit": 1},
    "HDV_BF": {"lifeTime": 25, "activityUpperLimit": 1},
    "MDV_el": {"lifeTime": 25, "activityUpperLimit": 1},
    "MDV_BF": {"lifeTime": 25, "activityUpperLimit": 1},
    "Two_wheel_el": {"lifeTime": 25, "activityUpperLimit": 1},
    "Bus_el": {"lifeTime": 25, "activityUpperLimit": 1},
    "Marine_e": {"lifeTime": 25, "activityUpperLimit": 1},
    "Aviation_el": {"lifeTime": 25, "activityUpperLimit": 1},
    "Aviation_e": {"lifeTime": 25, "activityUpperLimit": 1},
    "cook_el": {"lifeTime": 25, "activityUpperLimit": 1},
    "cook_LPG": {"lifeTime": 25, "activityUpperLimit": 1},
    "Industry_EH": {"lifeTime": 25, "activityUpperLimit": 1},
    "ST_N_DW": {"lifeTime": 25, "activityUpperLimit": 0},
    "DW_heat": {"lifeTime": 25, "activityUpperLimit": 1},
    "Dummy_Ammonia": {"lifeTime": 25, "activityUpperLimit": 1},
    "Dummy_Methanol": {"lifeTime": 25, "activityUpperLimit": 1},
    "Industry_EL": {"lifeTime": 25, "activityUpperLimit": 1},
    "WindOffshore_N": {"lifeTime": 25, "activityUpperLimit": 0}

}

# Create DataFrame
converter_techParam = pd.DataFrame(
    index=pd.MultiIndex.from_product([list(tech_specss.keys()), ['2040']])
)

# Assign values from dictionary
for tech, specs in tech_specss.items():
    converter_techParam.loc[idx[tech], "lifeTime"] = specs["lifeTime"]
    converter_techParam.loc[idx[tech], "activityUpperLimit"] = specs["activityUpperLimit"]


# Add to model
m.parameter.add(converter_techParam, "converter_techparam")
converter_techParam
###############################################################################
tech_specss = {
    "BG_N": {"lifeTime": 25, "activityUpperLimit": 0},  # No feed-in
    "PV_N": {"lifeTime": 25, "activityUpperLimit": 0},
    "WindOnshore_N": {"lifeTime": 25, "activityUpperLimit": 0},
    "Hydro_N": {"lifeTime": 50, "activityUpperLimit": 0},
    "Wave_N": {"lifeTime": 25, "activityUpperLimit": 0},
    "ST_N": {"lifeTime": 25, "activityUpperLimit": 0},
    "ST_N_DW": {"lifeTime": 25, "activityUpperLimit": 0},
    "DW_Electric_converter_2": {"lifeTime": 25, "activityUpperLimit": 1},
    "LDV_BF": {"lifeTime": 25, "activityUpperLimit": 1},
    "RO": {"lifeTime": 25, "activityUpperLimit": 1},
#    "AEL": {"lifeTime": 25, "activityUpperLimit": 1},
#    "AEL_10": {"lifeTime": 25,"activityUpperLimit": 1},
    "AEL_100": {"lifeTime": 25, "activityUpperLimit": 1},
    "Ammonia_synthesis": {"lifeTime": 25, "activityUpperLimit": 1},   
    "DAC": {"lifeTime": 25, "activityUpperLimit": 1}, 
    "Methanol_synthesis": {"lifeTime": 25, "activityUpperLimit": 1},
    "HP": {"lifeTime": 25, "activityUpperLimit": 1},
    "FTL": {"lifeTime": 25, "activityUpperLimit": 1},
    "LDV_el": {"lifeTime": 25, "activityUpperLimit": 1},
    "HDV_el": {"lifeTime": 25, "activityUpperLimit": 1},
    "HDV_BF": {"lifeTime": 25, "activityUpperLimit": 1},
    "MDV_el": {"lifeTime": 25, "activityUpperLimit": 1},
    "MDV_BF": {"lifeTime": 25, "activityUpperLimit": 1},
    "Two_wheel_el": {"lifeTime": 25, "activityUpperLimit": 1},
    "Bus_el": {"lifeTime": 25, "activityUpperLimit": 1},
    "Marine_e": {"lifeTime": 25, "activityUpperLimit": 1},
    "Aviation_el": {"lifeTime": 25, "activityUpperLimit": 1},
    "Aviation_e": {"lifeTime": 25, "activityUpperLimit": 1},
    "cook_el": {"lifeTime": 25, "activityUpperLimit": 1},
    "cook_LPG": {"lifeTime": 25, "activityUpperLimit": 1},
    "Industry_EH": {"lifeTime": 25, "activityUpperLimit": 1},
    "DW_heat": {"lifeTime": 25, "activityUpperLimit": 1},
    "Dummy_Ammonia": {"lifeTime": 50, "activityUpperLimit": 1},
    "Dummy_Methanol": {"lifeTime": 50, "activityUpperLimit": 1},
    "Industry_EL": {"lifeTime": 25, "activityUpperLimit": 1},
    "WindOffshore_N": {"lifeTime": 25, "activityUpperLimit": 0}
}

# Create DataFrame
converter_techParam = pd.DataFrame(
    index=pd.MultiIndex.from_product([list(tech_specss.keys()), ['2050']])
)

# Assign values from dictionary
for tech, specs in tech_specss.items():
    converter_techParam.loc[idx[tech], "lifeTime"] = specs["lifeTime"]
    converter_techParam.loc[idx[tech], "activityUpperLimit"] = specs["activityUpperLimit"]
#    converter_techParam.loc[idx[tech], "mipUnits"] = specs["mipUnits"]

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
        "cook_b": (0, 1000),
        "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000),
        "DW_Electric_converter": (0, 1000),
        "HFO": (0, 1000)
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
        "cook_b": (0, 1000),
        "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000),
        "DW_Electric_converter": (0, 1000),
        "HFO": (0, 1000)# hydro adjusted
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
        "cook_b": (0, 1000),
        "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000),
        "DW_Electric_converter": (0, 1000),
        "HFO": (0, 1000)# hydro adjusted
    },
    "KB_data": { 
        "DG": (0.0066, 0.0067),
        "PV_B": (0.0030, 0.0030),
        "MDV": (0, 1000),
        "HDV": (0, 1000),
        "LDV": (0, 1000),
        "Bus": (0, 1000),
        "Two_wheel": (0, 1000),
        "Aviation": (0, 1000),
        "Marine": (0, 1000),
        "cook_b": (0, 1000),
        "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000),
        "DW_Electric_converter": (0, 1000),
        "HFO": (0, 1000)
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
        "cook_b": (0, 1000),
        "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000),
        "DW_Electric_converter": (0, 1000),
        "HFO": (0, 1000)
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
        "cook_b": (0, 1000),
        "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000),
        "DW_Electric_converter": (0, 1000),
        "HFO": (0, 1000)
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
        "cook_b": (0, 1000),
        "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000),
        "DW_Electric_converter": (0, 1000),
        "HFO": (0, 1000)
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
        "cook_b": (0, 1000),
        "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000),
        "DW_Electric_converter": (0, 1000),
        "HFO": (0, 1000)
    },
    "PNG_data": {  
        "DG": (0.280, .350),
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
        "cook_b": (0, 1000),
        "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000),
        "DW_Electric_converter": (0, 1000),
        "HFO": (0, 1000)
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
        "cook_b": (0, 1000),
        "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000),
        "DW_Electric_converter": (0, 1000),
        "HFO": (0, 1000)# hydro adjusted
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
        "cook_b": (0, 1000),
        "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000),
        "DW_Electric_converter": (0, 1000),
        "HFO": (0, 1000)# hydro adjusted
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
        "cook_b": (0, 1),
        "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000),
        "DW_Electric_converter": (0, 1000),
        "HFO": (0, 1000)
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
        "cook_b": (0, 1),
        "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000),
        "DW_Electric_converter": (0, 1000),
        "HFO": (0, 1000)
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
        "cook_b": (0, 1),
        "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000),
        "DW_Electric_converter": (0, 1000),
        "HFO": (0, 1000)# hydro adjusted
    }
}


# Build DataFrame index
all_techs = list({tech for node in capacity_limits for tech in capacity_limits[node]})
converter_capacityParam = pd.DataFrame(
    index=pd.MultiIndex.from_product([m.set.nodesdata, ['2020'], all_techs])
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
################################################################################

capacity_limits = {
    "CI_data": {
        # "DG": (0, 0.018),
        "PV_B": (0.0052, 0.0052),
        "MDV": (0, 1000),
        "HDV": (0, 1000),
        "LDV": (0, 1000),
        "LDV_BF": (0, 1000),
        "Bus": (0, 1000),
        "Two_wheel": (0, 1000),
        "Aviation": (0, 1000),
        "Marine": (0, 1000),
        "cook_b": (0, 1000),
        "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000),
        "DW_Electric_converter": (0, 1000),
        "DW_Electric_converter_D": (0, 1000),
        "HFO": (0, 1000),
        "BG_N": (0, 0.011),
        "PV_N": (0, 20),         
        "WindOnshore_N": (0, 20)
    },
    "FJ_data": {
        # "DG": (0, 0.172),
        "BG_B": (0.0580, 0.0580),
        "PV_B": (0.0090, 0.0090),
        "Hydro_B": (0.0625, 0.0625),
        "MDV": (0, 1000),
        "HDV": (0, 1000),
        "LDV": (0, 1000),
        "LDV_BF": (0, 1000),
        "Bus": (0, 1000),
        "Two_wheel": (0, 1000),
        "Aviation": (0, 1000),
        "Marine": (0, 1000),
        "cook_b": (0, 1000),
        "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000),
        "DW_Electric_converter": (0, 1000),
        "DW_Electric_converter_D": (0, 1000),
        "HFO": (0, 1000),
        "BG_N": (0, 1),
        "PV_N": (0, 1),
        "WindOnshore_N": (0,1),
        "Wave_N": (0, 1),  
        "WindOffshore_N": (0, 1)# hydro adjusted
    },
    "FSM_data": { 
        # "DG": (0, 0.0388),
        "PV_B": (0.0028, 0.0028),
        "WindOnshore_B": (0.0009, 0.0009),
        "Hydro_B": (0.000225, 0.000225),
        "MDV": (0, 1000),
        "HDV": (0, 1000),
        "LDV": (0, 1000),
        "LDV_BF": (0, 1000),
        "Bus": (0, 1000),
        "Two_wheel": (0, 1000),
        "Aviation": (0, 1000),
        "Marine": (0, 1000),
        "cook_b": (0, 1000),
        "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000),
        "DW_Electric_converter": (0, 1000),
        "DW_Electric_converter_D": (0, 1000),
        "HFO": (0, 1000),
        "BG_N": (0, 0.160),
        "PV_N": (0, 20),
        "WindOnshore_N": (0, 20),
        "Wave_N": (0, 20),
        "WindOffshore_N": (0, 20)# hydro adjusted
    },
    "KB_data": { 
        # "DG": (0, 0.0067),
        "PV_B": (0.0030, 0.0030),
        "MDV": (0, 1000),
        "HDV": (0, 1000),
        "LDV": (0, 1000),
        "LDV_BF": (0, 1000),
        "Bus": (0, 1000),
        "Two_wheel": (0, 1000),
        "Aviation": (0, 1000),
        "Marine": (0, 1000),
        "cook_b": (0, 1000),
        "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000),
        "DW_Electric_converter": (0, 1000),
        "DW_Electric_converter_D": (0, 1000),
        "HFO": (0, 1000),
        "BG_N": (0, 0.211),
        "PV_N": (0, 20), 
        "WindOnshore_N": (0, 20),

    },
    "MI_data": { 
        # "DG": (0, 0.0287),
        "PV_B": (0.0017, 0.0017),
        "MDV": (0, 1000),
        "HDV": (0, 1000),
        "LDV": (0, 1000),
        "LDV_BF": (0, 1000),
        "Bus": (0, 1000),
        "Two_wheel": (0, 1000),
        "Aviation": (0, 1000),
        "Marine": (0, 1000),
        "cook_b": (0, 1000),
        "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000),
        "DW_Electric_converter": (0, 1000),
        "DW_Electric_converter_D": (0, 1000),
        "HFO": (0, 1000),
        "BG_N": (0, 0.02),
        "PV_N": (0, 20),
        "WindOnshore_N": (0, 20),
        "Wave_N": (0, 20), 
        "WindOffshore_N": (0, 20)
    },
    "NU_data": { 
        # "DG": (0, 0.0245),
        "PV_B": (0.0028, 0.0028),
        "MDV": (0, 1000),
        "HDV": (0, 1000),
        "LDV": (0, 1000),
        "LDV_BF": (0, 1000),
        "Bus": (0, 1000),
        "Two_wheel": (0, 1000),
        "Aviation": (0, 1000),
        "Marine": (0, 1000),
        "cook_b": (0, 1000),
        "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000),
        "DW_Electric_converter": (0, 1000),
        "DW_Electric_converter_D": (0, 1000),
        "HFO": (0, 1000),
        "BG_N": (0, 0.0038),
        "PV_N": (0, 20),  
        "WindOnshore_N": (0, 20)
    },
    "NE_data": {
        # "DG": (0, 0.0021),
        "PV_B": (0.0010, 0.0011),
        "MDV": (0, 1000),
        "HDV": (0, 1000),
        "LDV": (0, 1000),
        "LDV_BF": (0, 1000),
        "Bus": (0, 1000),
        "Two_wheel": (0, 1000),
        "Aviation": (0, 1000),
        "Marine": (0, 1000),
        "cook_b": (0, 1000),
        "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000),
        "DW_Electric_converter": (0, 1000),
        "DW_Electric_converter_D": (0, 1000),
        "HFO": (0, 1000),
        "BG_N": (0, 0.0047),
        "PV_N": (0, 20), 
        "WindOnshore_N": (0, 20)
    },
    "PU_data": {
        # "DG": (0, 0.0303),
        "PV_B": (0.0030, 0.0032),
        "MDV": (0, 1000),
        "HDV": (0, 1000),
        "LDV": (0, 1000),
        "LDV_BF": (0, 1000),
        "Bus": (0, 1000),
        "Two_wheel": (0, 1000),
        "Aviation": (0, 1000),
        "Marine": (0, 1000),
        "cook_b": (0, 1000),
        "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000),
        "DW_Electric_converter": (0, 1000),
        "DW_Electric_converter_D": (0, 1000),
        "HFO": (0, 1000),
        "BG_N": (0, 0.0009),
        "PV_N": (0, 20),
        "WindOnshore_N": (0, 20), 
        "WindOffshore_N": (0, 20)
    },
    "PNG_data": {  
        # "DG": (0, .350),
        # "NG_plant": (0.082, 0.082),
        "BG_B": (0.0182, 0.0182),
        "PV_B": (0.0031, 0.0031),
        "Hydro_B": (0.115, 0.115),
        "Geothermal_B": (0.011, 0.011),
        "MDV": (0, 1000),
        "HDV": (0, 1000),
        "LDV": (0, 1000),
        "LDV_BF": (0, 1000),
        "Bus": (0, 1000),
        "Two_wheel": (0, 1000),
        "Aviation": (0, 1000),
        "Marine": (0, 1000),
        "cook_b": (0, 1000),
        "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000),
        "DW_Electric_converter": (0, 1000),
        "DW_Electric_converter_D": (0, 1000),
        "HFO": (0, 1000),
        "BG_N": (0, 0.452),
        "PV_N": (0, 20),
        "WindOnshore_N": (0, 20),
        "Wave_N": (0, 20),  
        "WindOffshore_N": (0, 20)
        },
    "SA_data": { 
        # "DG": (0, 0.0315),
        "BG_B": (0.0011, 0.0011),
        "PV_B": (0.0138, 0.0138),
        "WindOnshore_B": (0.0005, 0.0005),
        "Hydro_B": (.0063, .0063),
        "MDV": (0, 1000),
        "HDV": (0, 1000),
        "LDV": (0, 1000),
        "LDV_BF": (0, 1000),
        "Bus": (0, 1000),
        "Two_wheel": (0, 1000),
        "Aviation": (0, 1000),
        "Marine": (0, 1000),
        "cook_b": (0, 1000),
        "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000),
        "DW_Electric_converter": (0, 1000),
        "DW_Electric_converter_D": (0, 1000),
        "HFO": (0, 1000),
        "BG_N": (0, 0.28),
        "PV_N": (0, 20),
        "WindOnshore_N": (0, 20),
        "Wave_N": (0, 20), 
        "WindOffshore_N": (0, 20)# hydro adjusted
    },
    "SI_data": {   
        # "DG": (0, 0.0527),
        "BG_B": (0.0008, 0.0008),
        "PV_B": (0.0023, 0.0023),
        "Hydro_B": (.00018, .00018),
        "MDV": (0, 1000),
        "HDV": (0, 1000),
        "LDV": (0, 1000),
        "LDV_BF": (0, 1000),
        "Bus": (0, 1000),
        "Two_wheel": (0, 1000),
        "Aviation": (0, 1000),
        "Marine": (0, 1000),
        "cook_b": (0, 1000),
        "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000),
        "DW_Electric_converter": (0, 1000),
        "DW_Electric_converter_D": (0, 1000),
        "HFO": (0, 1000),
        "BG_N": (0, 1.44),
        "PV_N": (0, 20),
        "WindOnshore_N": (0, 20),
        "Wave_N": (0, 20),
        "WindOffshore_N": (0, 20), 
        "Hydro_N": (.00675, .00675),
        # hydro adjusted
    },
    "TA_data": {  
        # "DG": (0, 0.0167),
        "PV_B": (0.0071, 0.0071),
        "WindOnshore_B": (0.00151, 0.00151),
        "MDV": (0, 1000),
        "HDV": (0, 1000),
        "LDV": (0, 1000),
        "LDV_BF": (0, 1000),
        "Bus": (0, 1000),
        "Two_wheel": (0, 1000),
        "Aviation": (0, 1000),
        "Marine": (0, 1000),
        "cook_b": (0, 1),
        "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000),
        "DW_Electric_converter": (0, 1000),
        "DW_Electric_converter_D": (0, 1000),
        "HFO": (0, 1000),
        "BG_N": (0, 0.20),
        "PV_N": (0, 20),
        "WindOnshore_N": (0, 20),
        "Wave_N": (0, 20),  
        "WindOffshore_N": (0, 20)
    },
    "TU_data": {  
        # "DG": (0, 0.003),
        "PV_B": (0.0029, 0.0029),
        "MDV": (0, 1000),
        "HDV": (0, 1000),
        "LDV": (0, 1000),
        "LDV_BF": (0, 1000),
        "Bus": (0, 1000),
        "Two_wheel": (0, 1000),
        "Aviation": (0, 1000),
        "Marine": (0, 1000),
        "cook_b": (0, 1),
        "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000),
        "DW_Electric_converter": (0, 1000),
        "DW_Electric_converter_D": (0, 1000),
        "HFO": (0, 1000),
        "BG_N": (0, 0.0084),
        "PV_N": (0, 20),
        "WindOnshore_N": (0, 20), 
        "Wave_N": (0, 20)
    },
    "VU_data": {  
        # "DG": (0, 0.0232),
        "PV_B": (0.0044, 0.0044),
        "WindOnshore_B": (0.0032, 0.0032),
        "Hydro_B": (.00054, .00054),
        "MDV": (0, 1000),
        "HDV": (0, 1000),
        "LDV": (0, 1000),
        "LDV_BF": (0, 1000),
        "Bus": (0, 1000),
        "Two_wheel": (0, 1000),
        "Aviation": (0, 1000),
        "Marine": (0, 1000),
        "cook_b": (0, 1),
        "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000),
        "DW_Electric_converter": (0, 1000),
        "DW_Electric_converter_D": (0, 1000),
        "HFO": (0, 1000),
        "BG_N": (0, .062),
        "PV_N": (0, 10),
        "WindOnshore_N": (0, 10),
        "Wave_N": (0, 10),
        "WindOffshore_N": (0, 10)# hydro adjusted
    }
}

# Build DataFrame index
all_techs = list({tech for node in capacity_limits for tech in capacity_limits[node]})
converter_capacityParam = pd.DataFrame(
    index=pd.MultiIndex.from_product([m.set.nodesdata, ['2030'], all_techs])
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
###################################################################################
capacity_limits = {
    "CI_data": {
        # "DG": (0, 0.018),
#        "PV_B": (0.0052, 0.0052),
        "MDV": (0, 1000),
        "HDV": (0, 1000),
        "LDV": (0, 1000),
        "LDV_BF": (0, 1000),
        "Bus": (0, 1000),
        "Two_wheel": (0, 1000),
        "Aviation": (0, 1000),
        "Marine": (0, 1000),
        "cook_b": (0, 1000),
        "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000),
        "DW_Electric_converter": (0, 1000),
        "DW_Electric_converter_2": (0, 1000),
        "ST_N": (0, 1000000),
        "HFO": (0, 1000),
        "BG_N": (0, 0.011),
        "PV_N": (0, 20),
        "RO": (0, 1000),
        "AEL_100": (0, 1000),
        "Ammonia_synthesis": (0, 1000), 
        "Methanol_synthesis": (0, 1000), 
        "HP": (0, 1000),
        "DAC": (0, 1000), 
        "FTL": (0, 1000), 
        "LDV_el": (0, 1000),
        "HDV_el": (0, 1000),
        "HDV_BF": (0, 1000),
        "MDV_el": (0, 1000),
        "MDV_BF": (0, 1000),
        "Two_wheel_el": (0, 1000),
        "Bus_el": (0, 1000),
        "Marine_e": (0, 1000),
        "Aviation_el": (0, 1000),
        "Aviation_e": (0, 1000),
        "cook_el": (0, 1000),
        "cook_LPG": (0, 1000),
        "Industry_EH": (0, 1000),
        "DW_heat": (0, 1000), 
        "Dummy_Ammonia": (0, 1000),
        "Dummy_Methanol": (0, 1000),
        "Industry_EL": (0, 1000),
        "ST_N_DW": (0, 1000000),
        "WindOnshore_N": (0, 20)
    },
    "FJ_data": {
        # "DG": (0, 0.172),
#        "BG_B": (0.0580, 0.0580),
#        "PV_B": (0.0090, 0.0090),
        "Hydro_B": (0.0625, 0.0625),
        "MDV": (0, 1000),
        "HDV": (0, 1000),
        "LDV": (0, 1000),
        "LDV_BF": (0, 1000),
        "Bus": (0, 1000),
        "Two_wheel": (0, 1000),
        "Aviation": (0, 1000),
        "Marine": (0, 1000),
        "cook_b": (0, 1000),
        "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000),
        "DW_Electric_converter": (0, 1000),
        "DW_Electric_converter_2": (0, 1000),
        "ST_N": (0, 1000000),
        "HFO": (0, 1000),
        "BG_N": (0, 20),
        "PV_N": (0, 20),
        "WindOnshore_N": (0, 20),
        "Wave_N": (0, 20),
        "RO": (0, 1000),
        "AEL_100": (0, 1000),
        "Ammonia_synthesis": (0, 1000), 
        "Methanol_synthesis": (0, 1000), 
        "HP": (0, 1000),
        "DAC": (0, 1000), 
        "FTL": (0, 1000),
        "LDV_el": (0, 1000),
        "HDV_el": (0, 1000),
        "HDV_BF": (0, 1000),
        "MDV_el": (0, 1000),
        "MDV_BF": (0, 1000),
        "Two_wheel_el": (0, 1000),
        "Bus_el": (0, 1000),
        "Marine_e": (0, 1000),
        "Aviation_el": (0, 1000),
        "Aviation_e": (0, 1000),
        "cook_el": (0, 1000),
        "cook_LPG": (0, 1000),
        "Industry_EH": (0, 1000),
        "DW_heat": (0, 1000),
        "Dummy_Ammonia": (0, 1000),
        "Dummy_Methanol": (0, 1000),
        "Industry_EL": (0, 1000),
        "ST_N_DW": (0, 1000000),
        "WindOffshore_N": (0, 20)# hydro adjusted
    },
    "FSM_data": { 
        # "DG": (0, 0.0388),
 #       "PV_B": (0.0028, 0.0028),
 #       "WindOnshore_B": (0.0009, 0.0009),
        "Hydro_B": (0.000225, 0.000225),
        "MDV": (0, 1000),
        "HDV": (0, 1000),
        "LDV": (0, 1000),
        "LDV_BF": (0, 1000),
        "Bus": (0, 1000),
        "Two_wheel": (0, 1000),
        "Aviation": (0, 1000),
        "Marine": (0, 1000),
        "cook_b": (0, 1000),
        "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000),
        "DW_Electric_converter": (0, 1000),
        "DW_Electric_converter_2": (0, 1000),
        "ST_N": (0, 1000000),
        "HFO": (0, 1000),
        "BG_N": (0, 0.160),
        "PV_N": (0, 20),
        "WindOnshore_N": (0, 20),
        "Wave_N": (0, 20),
        "RO": (0, 1000),
        "AEL_100": (0, 1000),
        "Ammonia_synthesis": (0, 1000),
        "Methanol_synthesis": (0, 1000), 
        "HP": (0, 1000),
        "DAC": (0, 1000), 
        "FTL": (0, 1000), 
        "LDV_el": (0, 1000),
        "HDV_el": (0, 1000),
        "HDV_BF": (0, 1000),
        "MDV_el": (0, 1000),
        "MDV_BF": (0, 1000),
        "Two_wheel_el": (0, 1000),
        "Bus_el": (0, 1000),
        "Marine_e": (0, 1000),
        "Aviation_el": (0, 1000),
        "Aviation_e": (0, 1000),
        "cook_el": (0, 1000),
        "cook_LPG": (0, 1000),
        "Industry_EH": (0, 1000),
        "DW_heat": (0, 1000),
        "Dummy_Ammonia": (0, 1000),
        "Dummy_Methanol": (0, 1000),
        "Industry_EL": (0, 1000),
        "ST_N_DW": (0, 1000000),
        "WindOffshore_N": (0, 20)# hydro adjusted
    },
    "KB_data": { 
        # "DG": (0, 0.0067),
#        "PV_B": (0.0030, 0.0030),
        "MDV": (0, 1000),
        "HDV": (0, 1000),
        "LDV": (0, 1000),
        "LDV_BF": (0, 1000),
        "Bus": (0, 1000),
        "Two_wheel": (0, 1000),
        "Aviation": (0, 1000),
        "Marine": (0, 1000),
        "cook_b": (0, 1000),
        "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000),
        "DW_Electric_converter": (0, 1000),
        "DW_Electric_converter_2": (0, 1000),
        "ST_N": (0, 1000000),
        "HFO": (0, 1000),
        "BG_N": (0, 0.211),
        "PV_N": (0, 20),
        "RO": (0, 1000),
        "AEL_100": (0, 1000),
        "Ammonia_synthesis": (0, 1000), 
        "Methanol_synthesis": (0, 1000), 
        "HP": (0, 1000),
        "DAC": (0, 1000), 
        "FTL": (0, 1000), 
        "LDV_el": (0, 1000),
        "HDV_el": (0, 1000),
        "HDV_BF": (0, 1000),
        "MDV_el": (0, 1000),
        "MDV_BF": (0, 1000),
        "Two_wheel_el": (0, 1000),
        "Bus_el": (0, 1000),
        "Marine_e": (0, 1000),
        "Aviation_el": (0, 1000),
        "Aviation_e": (0, 1000),
        "cook_el": (0, 1000),
        "cook_LPG": (0, 1000),
        "Industry_EH": (0, 1000),
        "DW_heat": (0, 1000),
        "Dummy_Ammonia": (0, 1000),
        "Dummy_Methanol": (0, 1000),
        "Industry_EL": (0, 1000),
        "ST_N_DW": (0, 1000000),
        "WindOnshore_N": (0, 20),

    },
    "MI_data": { 
        # "DG": (0, 0.0287),
#        "PV_B": (0.0017, 0.0017),
        "MDV": (0, 1000),
        "HDV": (0, 1000),
        "LDV": (0, 1000),
        "LDV_BF": (0, 1000),
        "Bus": (0, 1000),
        "Two_wheel": (0, 1000),
        "Aviation": (0, 1000),
        "Marine": (0, 1000),
        "cook_b": (0, 1000),
        "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000),
        "DW_Electric_converter": (0, 1000),
        "DW_Electric_converter_2": (0, 1000),
        "ST_N": (0, 1000000),
        "HFO": (0, 1000),
        "BG_N": (0, 0.02),
        "PV_N": (0, 20),
        "WindOnshore_N": (0, 20),
        "Wave_N": (0, 20),
        "RO": (0, 1000),
        "AEL_100": (0, 1000),
        "Ammonia_synthesis": (0, 1000), 
        "Methanol_synthesis": (0, 1000), 
        "HP": (0, 1000),
        "DAC": (0, 1000), 
        "FTL": (0, 1000),
        "LDV_el": (0, 1000),
        "HDV_el": (0, 1000),
        "HDV_BF": (0, 1000),
        "MDV_el": (0, 1000),
        "MDV_BF": (0, 1000),
        "Two_wheel_el": (0, 1000),
        "Bus_el": (0, 1000),
        "Marine_e": (0, 1000),
        "Aviation_el": (0, 1000),
        "Aviation_e": (0, 1000),
        "cook_el": (0, 1000),
        "cook_LPG": (0, 1000),
        "Industry_EH": (0, 1000),
        "DW_heat": (0, 1000),
        "Dummy_Ammonia": (0, 1000),
        "Dummy_Methanol": (0, 1000),
        "Industry_EL": (0, 1000),
        "ST_N_DW": (0, 1000000),
        "WindOffshore_N": (0, 20)
    },
    "NU_data": { 
        # "DG": (0, 0.0245),
#        "PV_B": (0.0028, 0.0028),
        "MDV": (0, 1000),
        "HDV": (0, 1000),
        "LDV": (0, 1000),
        "LDV_BF": (0, 1000),
        "Bus": (0, 1000),
        "Two_wheel": (0, 1000),
        "Aviation": (0, 1000),
        "Marine": (0, 1000),
        "cook_b": (0, 1000),
        "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000),
        "DW_Electric_converter": (0, 1000),
        "DW_Electric_converter_2": (0, 1000),
        "ST_N": (0, 1000000),
        "HFO": (0, 1000),
        "BG_N": (0, 0.0038),
        "PV_N": (0, 20),
        "RO": (0, 1000),
        "AEL_100": (0, 1000),
        "Ammonia_synthesis": (0, 1000), 
        "Methanol_synthesis": (0, 1000), 
        "HP": (0, 1000),
        "DAC": (0, 1000), 
        "FTL": (0, 1000), 
        "LDV_el": (0, 1000),
        "HDV_el": (0, 1000),
        "HDV_BF": (0, 1000),
        "MDV_el": (0, 1000),
        "MDV_BF": (0, 1000),
        "Two_wheel_el": (0, 1000),
        "Bus_el": (0, 1000),
        "Marine_e": (0, 1000),
        "Aviation_el": (0, 1000),
        "Aviation_e": (0, 1000),
        "cook_el": (0, 1000),
        "cook_LPG": (0, 1000),
        "Industry_EH": (0, 1000),
        "DW_heat": (0, 1000),
        "Dummy_Ammonia": (0, 1000),
        "Dummy_Methanol": (0, 1000),
        "Industry_EL": (0, 1000),
        "ST_N_DW": (0, 1000000),
        "WindOnshore_N": (0, 20)
    },
    "NE_data": {
        # "DG": (0, 0.0021),
 #       "PV_B": (0.0010, 0.0011),
        "MDV": (0, 1000),
        "HDV": (0, 1000),
        "LDV": (0, 1000),
        "LDV_BF": (0, 1000),
        "Bus": (0, 1000),
        "Two_wheel": (0, 1000),
        "Aviation": (0, 1000),
        "Marine": (0, 1000),
        "cook_b": (0, 1000),
        "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000),
        "DW_Electric_converter": (0, 1000),
        "DW_Electric_converter_2": (0, 1000),
        "ST_N": (0, 1000000),
        "HFO": (0, 1000),
        "BG_N": (0, 0.0047),
        "PV_N": (0, 20),
        "RO": (0, 1000),
        "AEL_100": (0, 1000),
        "Ammonia_synthesis": (0, 1000), 
        "Methanol_synthesis": (0, 1000), 
        "HP": (0, 1000),
        "DAC": (0, 1000), 
        "FTL": (0, 1000), 
        "LDV_el": (0, 1000),
        "HDV_el": (0, 1000),
        "HDV_BF": (0, 1000),
        "MDV_el": (0, 1000),
        "MDV_BF": (0, 1000),
        "Two_wheel_el": (0, 1000),
        "Bus_el": (0, 1000),
        "Marine_e": (0, 1000),
        "Aviation_el": (0, 1000),
        "Aviation_e": (0, 1000),
        "cook_el": (0, 1000),
        "cook_LPG": (0, 1000),
        "Industry_EH": (0, 1000),
        "DW_heat": (0, 1000),
        "Dummy_Ammonia": (0, 1000),
        "Dummy_Methanol": (0, 1000),
        "Industry_EL": (0, 1000),
        "ST_N_DW": (0, 1000000),
        "WindOnshore_N": (0, 20)
    },
    "PU_data": {
        # "DG": (0, 0.0303),
#        "PV_B": (0.0030, 0.0032),
        "MDV": (0, 1000),
        "HDV": (0, 1000),
        "LDV": (0, 1000),
        "LDV_BF": (0, 1000),
        "Bus": (0, 1000),
        "Two_wheel": (0, 1000),
        "Aviation": (0, 1000),
        "Marine": (0, 1000),
        "cook_b": (0, 1000),
        "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000),
        "DW_Electric_converter": (0, 1000),
        "DW_Electric_converter_2": (0, 1000),
        "ST_N": (0, 1000000),
        "HFO": (0, 1000),
        "BG_N": (0, 0.0009),
        "PV_N": (0, 20),
        "WindOnshore_N": (0, 20),
        "RO": (0, 1000),
        "AEL_100": (0, 1000),
        "Ammonia_synthesis": (0, 1000), 
        "Methanol_synthesis": (0, 1000),
        "HP": (0, 1000),
        "DAC": (0, 1000),
        "FTL": (0, 1000),  
        "LDV_el": (0, 1000),
        "HDV_el": (0, 1000),
        "HDV_BF": (0, 1000),
        "MDV_el": (0, 1000),
        "MDV_BF": (0, 1000),
        "Two_wheel_el": (0, 1000),
        "Bus_el": (0, 1000),
        "Marine_e": (0, 1000),
        "Aviation_el": (0, 1000),
        "Aviation_e": (0, 1000),
        "cook_el": (0, 1000),
        "cook_LPG": (0, 1000),
        "Industry_EH": (0, 1000),
        "DW_heat": (0, 1000),
        "Dummy_Ammonia": (0, 1000),
        "Dummy_Methanol": (0, 1000),
        "Industry_EL": (0, 1000),
        "ST_N_DW": (0, 1000000),
        "WindOffshore_N": (0, 20)
    },
    "PNG_data": {  
        # "DG": (0, .350),
        # "NG_plant": (0.082, 0.082),
#        "BG_B": (0.0182, 0.0182),
#        "PV_B": (0.0031, 0.0031),
        "Hydro_B": (0.115, 0.115),
        "Geothermal_B": (0.011, 0.011),
        "MDV": (0, 1000),
        "HDV": (0, 1000),
        "LDV": (0, 1000),
        "LDV_BF": (0, 1000),
        "Bus": (0, 1000),
        "Two_wheel": (0, 1000),
        "Aviation": (0, 1000),
        "Marine": (0, 1000),
        "cook_b": (0, 1000),
        "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000),
        "DW_Electric_converter": (0, 1000),
        "DW_Electric_converter_2": (0, 1000),
        "ST_N": (0, 1000000),
        "HFO": (0, 1000),
        "BG_N": (0, 0.452),
        "PV_N": (0, 20),
        "WindOnshore_N": (0, 20),
        "Wave_N": (0, 20),
        "RO": (0, 1000),
        "AEL_100": (0, 1000),
        "Ammonia_synthesis": (0, 1000), 
        "Methanol_synthesis": (0, 1000), 
        "HP": (0, 1000),
        "DAC": (0, 1000), 
        "FTL": (0, 1000), 
        "LDV_el": (0, 1000),
        "HDV_el": (0, 1000),
        "HDV_BF": (0, 1000),
        "MDV_el": (0, 1000),
        "MDV_BF": (0, 1000),
        "Two_wheel_el": (0, 1000),
        "Bus_el": (0, 1000),
        "Marine_e": (0, 1000),
        "Aviation_el": (0, 1000),
        "Aviation_e": (0, 1000),
        "cook_el": (0, 1000),
        "cook_LPG": (0, 1000),
        "Industry_EH": (0, 1000),
        "DW_heat": (0, 1000),
        "Dummy_Ammonia": (0, 1000),
        "Dummy_Methanol": (0, 1000),
        "Industry_EL": (0, 1000),
        "ST_N_DW": (0, 1000000),
        "WindOffshore_N": (0, 20)
        },
    "SA_data": { 
        # "DG": (0, 0.0315),
#        "BG_B": (0.0011, 0.0011),
#        "PV_B": (0.0138, 0.0138),
#        "WindOnshore_B": (0.0005, 0.0005),
        "Hydro_B": (.0063, .0063),
        "MDV": (0, 1000),
        "HDV": (0, 1000),
        "LDV": (0, 1000),
        "LDV_BF": (0, 1000),
        "Bus": (0, 1000),
        "Two_wheel": (0, 1000),
        "Aviation": (0, 1000),
        "Marine": (0, 1000),
        "cook_b": (0, 1000),
        "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000),
        "DW_Electric_converter": (0, 1000),
        "DW_Electric_converter_2": (0, 1000),
        "ST_N": (0, 1000000),
        "HFO": (0, 1000),
        "BG_N": (0, 0.28),
        "PV_N": (0, 20),
        "WindOnshore_N": (0, 20),
        "Wave_N": (0, 20),
        "RO": (0, 1000),
        "AEL_100": (0, 1000),
        "Ammonia_synthesis": (0, 1000), 
        "Methanol_synthesis": (0, 1000),
        "HP": (0, 1000),
        "DAC": (0, 1000),
        "FTL": (0, 1000), 
        "LDV_el": (0, 1000),
        "HDV_el": (0, 1000),
        "HDV_BF": (0, 1000),
        "MDV_el": (0, 1000),
        "MDV_BF": (0, 1000),
        "Two_wheel_el": (0, 1000),
        "Bus_el": (0, 1000),
        "Marine_e": (0, 1000),
        "Aviation_el": (0, 1000),
        "Aviation_e": (0, 1000),
        "cook_el": (0, 1000),
        "cook_LPG": (0, 1000),
        "Industry_EH": (0, 1000),
        "DW_heat": (0, 1000),
        "Dummy_Ammonia": (0, 1000),
        "Dummy_Methanol": (0, 1000),
        "Industry_EL": (0, 1000),
        "ST_N_DW": (0, 1000000),
        "WindOffshore_N": (0, 20)# hydro adjusted
    },
    "SI_data": {   
        # "DG": (0, 0.0527),
 #       "BG_B": (0.0008, 0.0008),
 #       "PV_B": (0.0023, 0.0023),
        "Hydro_B": (.00018, .00018),
        "MDV": (0, 1000),
        "HDV": (0, 1000),
        "LDV": (0, 1000),
        "LDV_BF": (0, 1000),
        "Bus": (0, 1000),
        "Two_wheel": (0, 1000),
        "Aviation": (0, 1000),
        "Marine": (0, 1000),
        "cook_b": (0, 1000),
        "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000),
        "DW_Electric_converter": (0, 1000),
        "DW_Electric_converter_2": (0, 1000),
        "ST_N": (0, 1000000),
        "HFO": (0, 1000),
        "BG_N": (0, 1.44),
        "PV_N": (0, 20),
        "WindOnshore_N": (0, 20),
        "Wave_N": (0, 20),
        "WindOffshore_N": (0, 20),
        "RO": (0, 1000),
        "AEL_100": (0, 1000),
        "Ammonia_synthesis": (0, 1000), 
        "Methanol_synthesis": (0, 1000), 
        "HP": (0, 1000),
        "DAC": (0, 1000), 
        "FTL": (0, 1000), 
        "LDV_el": (0, 1000),
        "HDV_el": (0, 1000),
        "HDV_BF": (0, 1000),
        "MDV_el": (0, 1000),
        "MDV_BF": (0, 1000),
        "Two_wheel_el": (0, 1000),
        "Bus_el": (0, 1000),
        "Marine_e": (0, 1000),
        "Aviation_el": (0, 1000),
        "Aviation_e": (0, 1000),
        "cook_el": (0, 1000),
        "cook_LPG": (0, 1000),
        "Industry_EH": (0, 1000),
        "DW_heat": (0, 1000),
        "Dummy_Ammonia": (0, 1000),
        "Dummy_Methanol": (0, 1000),
        "Industry_EL": (0, 1000),
        "ST_N_DW": (0, 1000000),
        "Hydro_N": (.00675, .00675),
        # hydro adjusted
    },
    "TA_data": {  
        # "DG": (0, 0.0167),
#        "PV_B": (0.0071, 0.0071),
#        "WindOnshore_B": (0.00151, 0.00151),
        "MDV": (0, 1000),
        "HDV": (0, 1000),
        "LDV": (0, 1000),
        "LDV_BF": (0, 1000),
        "Bus": (0, 1000),
        "Two_wheel": (0, 1000),
        "Aviation": (0, 1000),
        "Marine": (0, 1000),
        "cook_b": (0, 1),
        "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000),
        "DW_Electric_converter": (0, 1000),
        "DW_Electric_converter_2": (0, 1000),
        "ST_N": (0, 1000000),
        "HFO": (0, 1000),
        "BG_N": (0, 0.20),
        "PV_N": (0, 20),
        "WindOnshore_N": (0, 20),
        "Wave_N": (0, 20),
        "RO": (0, 1000),
        "AEL_100": (0, 1000),
        "Ammonia_synthesis": (0, 1000), 
        "Methanol_synthesis": (0, 1000), 
        "HP": (0, 1000),
        "DAC": (0, 1000), 
        "FTL": (0, 1000), 
        "LDV_el": (0, 1000),
        "HDV_el": (0, 1000),
        "HDV_BF": (0, 1000),
        "MDV_el": (0, 1000),
        "MDV_BF": (0, 1000),
        "Two_wheel_el": (0, 1000),
        "Bus_el": (0, 1000),
        "Marine_e": (0, 1000),
        "Aviation_el": (0, 1000),
        "Aviation_e": (0, 1000),
        "cook_el": (0, 1000),
        "cook_LPG": (0, 1000),
        "Industry_EH": (0, 1000),
        "DW_heat": (0, 1000),
        "Dummy_Ammonia": (0, 1000),
        "Dummy_Methanol": (0, 1000),
        "Industry_EL": (0, 1000),
        "ST_N_DW": (0, 1000000),
        "WindOffshore_N": (0, 20)
    },
    "TU_data": {  
        # "DG": (0, 0.003),
#        "PV_B": (0.0029, 0.0029),
        "MDV": (0, 1000),
        "HDV": (0, 1000),
        "LDV": (0, 1000),
        "LDV_BF": (0, 1000),
        "Bus": (0, 1000),
        "Two_wheel": (0, 1000),
        "Aviation": (0, 1000),
        "Marine": (0, 1000),
        "cook_b": (0, 1),
        "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000),
        "DW_Electric_converter": (0, 1000),
        "DW_Electric_converter_2": (0, 1000),
        "ST_N": (0, 1000000),
        "HFO": (0, 1000),
        "BG_N": (0, 0.0084),
        "PV_N": (0, 20),
        "WindOnshore_N": (0, 20),
        "RO": (0, 1000),
        "AEL_100": (0, 1000),
        "Ammonia_synthesis": (0, 1000), 
        "Methanol_synthesis": (0, 1000), 
        "HP": (0, 1000),
        "DAC": (0, 1000), 
        "FTL": (0, 1000), 
        "LDV_el": (0, 1000),
        "HDV_el": (0, 1000),
        "HDV_BF": (0, 1000),
        "MDV_el": (0, 1000),
        "MDV_BF": (0, 1000),
        "Two_wheel_el": (0, 1000),
        "Bus_el": (0, 1000),
        "Marine_e": (0, 1000),
        "Aviation_el": (0, 1000),
        "Aviation_e": (0, 1000),
        "cook_el": (0, 1000),
        "cook_LPG": (0, 1000),
        "Industry_EH": (0, 1000),
        "DW_heat": (0, 1000),
        "Dummy_Ammonia": (0, 1000),
        "Dummy_Methanol": (0, 1000),
        "Industry_EL": (0, 1000),
        "ST_N_DW": (0, 1000000),
        "Wave_N": (0, 20)
    },
    "VU_data": {  
        # "DG": (0, 0.0232),
#        "PV_B": (0.0044, 0.0044),
#        "WindOnshore_B": (0.0032, 0.0032),
        "Hydro_B": (.00054, .00054),
        "MDV": (0, 1000),
        "HDV": (0, 1000),
        "LDV": (0, 1000),
        "LDV_BF": (0, 1000),
        "Bus": (0, 1000),
        "Two_wheel": (0, 1000),
        "Aviation": (0, 1000),
        "Marine": (0, 1000),
        "cook_b": (0, 1),
        "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000),
        "DW_Electric_converter": (0, 1000),
        "DW_Electric_converter_2": (0, 1000),
        "ST_N": (0, 1000000),
        "HFO": (0, 1000),
        "BG_N": (0, .062),
        "PV_N": (0, 10),
        "WindOnshore_N": (0, 10),
        "Wave_N": (0, 10),
        "RO": (0, 1000),
        "AEL_100": (0, 1000),
        "Ammonia_synthesis": (0, 1000),  
        "Methanol_synthesis": (0, 1000), 
        "DAC": (0, 1000), 
        "FTL": (0, 1000),  
        "LDV_el": (0, 1000),
        "HDV_el": (0, 1000),
        "HDV_BF": (0, 1000),
        "MDV_el": (0, 1000),
        "MDV_BF": (0, 1000),
        "Two_wheel_el": (0, 1000),
        "Bus_el": (0, 1000),
        "Marine_e": (0, 1000),
        "Aviation_el": (0, 1000),
        "Aviation_e": (0, 1000),
        "cook_el": (0, 1000),
        "cook_LPG": (0, 1000),
        "Industry_EH": (0, 1000),
        "DW_heat": (0, 1000),
        "Dummy_Ammonia": (0, 1000),
        "Dummy_Methanol": (0, 1000),
        "Industry_EL": (0, 1000),
        "ST_N_DW": (0, 1000000),
        "WindOffshore_N": (0, 10)# hydro adjusted
    }
}

# Build DataFrame index
all_techs = list({tech for node in capacity_limits for tech in capacity_limits[node]})
converter_capacityParam = pd.DataFrame(
    index=pd.MultiIndex.from_product([m.set.nodesdata, ['2040'], all_techs])
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
################################################################################
capacity_limits = {
    "CI_data": {
        # "DG": (0, 0.018),
#        "PV_B": (0.0052, 0.0052),
        "MDV": (0, 1000),
        "HDV": (0, 1000),
        "LDV": (0, 1000),
        "LDV_BF": (0, 1000),
        "Bus": (0, 1000),
        "Two_wheel": (0, 1000),
        "Aviation": (0, 1000),
        "Marine": (0, 1000),
        "cook_b": (0, 1000),
        "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000),
        "DW_Electric_converter": (0, 1000),
        "DW_Electric_converter_2": (0, 1000),
        "ST_N": (0, 1000000),
        "HFO": (0, 1000),
        "BG_N": (0, 0.011),
        "PV_N": (0, 20),
        "RO": (0, 1000),
        "AEL_100": (0, 1000),
        "Ammonia_synthesis": (0, 1000), 
        "Methanol_synthesis": (0, 1000), 
        "HP": (0, 1000),
        "DAC": (0, 1000), 
        "FTL": (0, 1000), 
        "LDV_el": (0, 1000),
        "HDV_el": (0, 1000),
        "HDV_BF": (0, 1000),
        "MDV_el": (0, 1000),
        "MDV_BF": (0, 1000),
        "Two_wheel_el": (0, 1000),
        "Bus_el": (0, 1000),
        "Marine_e": (0, 1000),
        "Aviation_el": (0, 1000),
        "Aviation_e": (0, 1000),
        "cook_el": (0, 1000),
        "cook_LPG": (0, 1000),
        "Industry_EH": (0, 1000),
        "DW_heat": (0, 1000), 
        "Dummy_Ammonia": (0, 1000),
        "Dummy_Methanol": (0, 1000),
        "Industry_EL": (0, 1000),
        "ST_N_DW": (0, 1000000),
        "WindOnshore_N": (0, 20)
    },
    "FJ_data": {
        # "DG": (0, 0.172),
#        "BG_B": (0.0580, 0.0580),
#        "PV_B": (0.0090, 0.0090),
        "Hydro_B": (0.0625, 0.0625),
        "MDV": (0, 1000),
        "HDV": (0, 1000),
        "LDV": (0, 1000),
        "LDV_BF": (0, 1000),
        "Bus": (0, 1000),
        "Two_wheel": (0, 1000),
        "Aviation": (0, 1000),
        "Marine": (0, 1000),
        "cook_b": (0, 1000),
        "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000),
        "DW_Electric_converter": (0, 1000),
        "DW_Electric_converter_2": (0, 1000),
        "ST_N": (0, 1000000),
        "HFO": (0, 1000),
        "BG_N": (0, 20),
        "PV_N": (0, 20),
        "WindOnshore_N": (0, 20),
        "Wave_N": (0, 20),
        "RO": (0, 1000),
        "AEL_100": (0, 1000),
        "Ammonia_synthesis": (0, 1000), 
        "Methanol_synthesis": (0, 1000), 
        "HP": (0, 1000),
        "DAC": (0, 1000), 
        "FTL": (0, 1000),
        "LDV_el": (0, 1000),
        "HDV_el": (0, 1000),
        "HDV_BF": (0, 1000),
        "MDV_el": (0, 1000),
        "MDV_BF": (0, 1000),
        "Two_wheel_el": (0, 1000),
        "Bus_el": (0, 1000),
        "Marine_e": (0, 1000),
        "Aviation_el": (0, 1000),
        "Aviation_e": (0, 1000),
        "cook_el": (0, 1000),
        "cook_LPG": (0, 1000),
        "Industry_EH": (0, 1000),
        "DW_heat": (0, 1000),
        "Dummy_Ammonia": (0, 1000),
        "Dummy_Methanol": (0, 1000),
        "Industry_EL": (0, 1000),
        "ST_N_DW": (0, 1000000),
        "WindOffshore_N": (0, 20)# hydro adjusted
    },
    "FSM_data": { 
        # "DG": (0, 0.0388),
 #       "PV_B": (0.0028, 0.0028),
 #       "WindOnshore_B": (0.0009, 0.0009),
        "Hydro_B": (0.000225, 0.000225),
        "MDV": (0, 1000),
        "HDV": (0, 1000),
        "LDV": (0, 1000),
        "LDV_BF": (0, 1000),
        "Bus": (0, 1000),
        "Two_wheel": (0, 1000),
        "Aviation": (0, 1000),
        "Marine": (0, 1000),
        "cook_b": (0, 1000),
        "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000),
        "DW_Electric_converter": (0, 1000),
        "DW_Electric_converter_2": (0, 1000),
        "ST_N": (0, 1000000),
        "HFO": (0, 1000),
        "BG_N": (0, 0.160),
        "PV_N": (0, 20),
        "WindOnshore_N": (0, 20),
        "Wave_N": (0, 20),
        "RO": (0, 1000),
        "AEL_100": (0, 1000),
        "Ammonia_synthesis": (0, 1000),
        "Methanol_synthesis": (0, 1000), 
        "HP": (0, 1000),
        "DAC": (0, 1000), 
        "FTL": (0, 1000), 
        "LDV_el": (0, 1000),
        "HDV_el": (0, 1000),
        "HDV_BF": (0, 1000),
        "MDV_el": (0, 1000),
        "MDV_BF": (0, 1000),
        "Two_wheel_el": (0, 1000),
        "Bus_el": (0, 1000),
        "Marine_e": (0, 1000),
        "Aviation_el": (0, 1000),
        "Aviation_e": (0, 1000),
        "cook_el": (0, 1000),
        "cook_LPG": (0, 1000),
        "Industry_EH": (0, 1000),
        "DW_heat": (0, 1000),
        "Dummy_Ammonia": (0, 1000),
        "Dummy_Methanol": (0, 1000),
        "Industry_EL": (0, 1000),
        "ST_N_DW": (0, 1000000),
        "WindOffshore_N": (0, 20)# hydro adjusted
    },
    "KB_data": { 
        # "DG": (0, 0.0067),
#        "PV_B": (0.0030, 0.0030),
        "MDV": (0, 1000),
        "HDV": (0, 1000),
        "LDV": (0, 1000),
        "LDV_BF": (0, 1000),
        "Bus": (0, 1000),
        "Two_wheel": (0, 1000),
        "Aviation": (0, 1000),
        "Marine": (0, 1000),
        "cook_b": (0, 1000),
        "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000),
        "DW_Electric_converter": (0, 1000),
        "DW_Electric_converter_2": (0, 1000),
        "ST_N": (0, 1000000),
        "HFO": (0, 1000),
        "BG_N": (0, 0.211),
        "PV_N": (0, 20),
        "RO": (0, 1000),
        "AEL_100": (0, 1000),
        "Ammonia_synthesis": (0, 1000), 
        "Methanol_synthesis": (0, 1000), 
        "HP": (0, 1000),
        "DAC": (0, 1000), 
        "FTL": (0, 1000), 
        "LDV_el": (0, 1000),
        "HDV_el": (0, 1000),
        "HDV_BF": (0, 1000),
        "MDV_el": (0, 1000),
        "MDV_BF": (0, 1000),
        "Two_wheel_el": (0, 1000),
        "Bus_el": (0, 1000),
        "Marine_e": (0, 1000),
        "Aviation_el": (0, 1000),
        "Aviation_e": (0, 1000),
        "cook_el": (0, 1000),
        "cook_LPG": (0, 1000),
        "Industry_EH": (0, 1000),
        "DW_heat": (0, 1000),
        "Dummy_Ammonia": (0, 1000),
        "Dummy_Methanol": (0, 1000),
        "Industry_EL": (0, 1000),
        "ST_N_DW": (0, 1000000),
        "WindOnshore_N": (0, 20),

    },
    "MI_data": { 
        # "DG": (0, 0.0287),
#        "PV_B": (0.0017, 0.0017),
        "MDV": (0, 1000),
        "HDV": (0, 1000),
        "LDV": (0, 1000),
        "LDV_BF": (0, 1000),
        "Bus": (0, 1000),
        "Two_wheel": (0, 1000),
        "Aviation": (0, 1000),
        "Marine": (0, 1000),
        "cook_b": (0, 1000),
        "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000),
        "DW_Electric_converter": (0, 1000),
        "DW_Electric_converter_2": (0, 1000),
        "ST_N": (0, 1000000),
        "HFO": (0, 1000),
        "BG_N": (0, 0.02),
        "PV_N": (0, 20),
        "WindOnshore_N": (0, 20),
        "Wave_N": (0, 20),
        "RO": (0, 1000),
        "AEL_100": (0, 1000),
        "Ammonia_synthesis": (0, 1000), 
        "Methanol_synthesis": (0, 1000), 
        "HP": (0, 1000),
        "DAC": (0, 1000), 
        "FTL": (0, 1000),
        "LDV_el": (0, 1000),
        "HDV_el": (0, 1000),
        "HDV_BF": (0, 1000),
        "MDV_el": (0, 1000),
        "MDV_BF": (0, 1000),
        "Two_wheel_el": (0, 1000),
        "Bus_el": (0, 1000),
        "Marine_e": (0, 1000),
        "Aviation_el": (0, 1000),
        "Aviation_e": (0, 1000),
        "cook_el": (0, 1000),
        "cook_LPG": (0, 1000),
        "Industry_EH": (0, 1000),
        "DW_heat": (0, 1000),
        "Dummy_Ammonia": (0, 1000),
        "Dummy_Methanol": (0, 1000),
        "Industry_EL": (0, 1000),
        "ST_N_DW": (0, 1000000),
        "WindOffshore_N": (0, 20)
    },
    "NU_data": { 
        # "DG": (0, 0.0245),
#        "PV_B": (0.0028, 0.0028),
        "MDV": (0, 1000),
        "HDV": (0, 1000),
        "LDV": (0, 1000),
        "LDV_BF": (0, 1000),
        "Bus": (0, 1000),
        "Two_wheel": (0, 1000),
        "Aviation": (0, 1000),
        "Marine": (0, 1000),
        "cook_b": (0, 1000),
        "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000),
        "DW_Electric_converter": (0, 1000),
        "DW_Electric_converter_2": (0, 1000),
        "ST_N": (0, 1000000),
        "HFO": (0, 1000),
        "BG_N": (0, 0.0038),
        "PV_N": (0, 20),
        "RO": (0, 1000),
        "AEL_100": (0, 1000),
        "Ammonia_synthesis": (0, 1000), 
        "Methanol_synthesis": (0, 1000), 
        "HP": (0, 1000),
        "DAC": (0, 1000), 
        "FTL": (0, 1000), 
        "LDV_el": (0, 1000),
        "HDV_el": (0, 1000),
        "HDV_BF": (0, 1000),
        "MDV_el": (0, 1000),
        "MDV_BF": (0, 1000),
        "Two_wheel_el": (0, 1000),
        "Bus_el": (0, 1000),
        "Marine_e": (0, 1000),
        "Aviation_el": (0, 1000),
        "Aviation_e": (0, 1000),
        "cook_el": (0, 1000),
        "cook_LPG": (0, 1000),
        "Industry_EH": (0, 1000),
        "DW_heat": (0, 1000),
        "Dummy_Ammonia": (0, 1000),
        "Dummy_Methanol": (0, 1000),
        "Industry_EL": (0, 1000),
        "ST_N_DW": (0, 1000000),
        "WindOnshore_N": (0, 20)
    },
    "NE_data": {
        # "DG": (0, 0.0021),
 #       "PV_B": (0.0010, 0.0011),
        "MDV": (0, 1000),
        "HDV": (0, 1000),
        "LDV": (0, 1000),
        "LDV_BF": (0, 1000),
        "Bus": (0, 1000),
        "Two_wheel": (0, 1000),
        "Aviation": (0, 1000),
        "Marine": (0, 1000),
        "cook_b": (0, 1000),
        "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000),
        "DW_Electric_converter": (0, 1000),
        "DW_Electric_converter_2": (0, 1000),
        "ST_N": (0, 1000000),
        "HFO": (0, 1000),
        "BG_N": (0, 0.0047),
        "PV_N": (0, 20),
        "RO": (0, 1000),
        "AEL_100": (0, 1000),
        "Ammonia_synthesis": (0, 1000), 
        "Methanol_synthesis": (0, 1000), 
        "HP": (0, 1000),
        "DAC": (0, 1000), 
        "FTL": (0, 1000), 
        "LDV_el": (0, 1000),
        "HDV_el": (0, 1000),
        "HDV_BF": (0, 1000),
        "MDV_el": (0, 1000),
        "MDV_BF": (0, 1000),
        "Two_wheel_el": (0, 1000),
        "Bus_el": (0, 1000),
        "Marine_e": (0, 1000),
        "Aviation_el": (0, 1000),
        "Aviation_e": (0, 1000),
        "cook_el": (0, 1000),
        "cook_LPG": (0, 1000),
        "Industry_EH": (0, 1000),
        "DW_heat": (0, 1000),
        "Dummy_Ammonia": (0, 1000),
        "Dummy_Methanol": (0, 1000),
        "Industry_EL": (0, 1000),
        "ST_N_DW": (0, 1000000),
        "WindOnshore_N": (0, 20)
    },
    "PU_data": {
        # "DG": (0, 0.0303),
#        "PV_B": (0.0030, 0.0032),
        "MDV": (0, 1000),
        "HDV": (0, 1000),
        "LDV": (0, 1000),
        "LDV_BF": (0, 1000),
        "Bus": (0, 1000),
        "Two_wheel": (0, 1000),
        "Aviation": (0, 1000),
        "Marine": (0, 1000),
        "cook_b": (0, 1000),
        "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000),
        "DW_Electric_converter": (0, 1000),
        "DW_Electric_converter_2": (0, 1000),
        "ST_N": (0, 1000000),
        "HFO": (0, 1000),
        "BG_N": (0, 0.0009),
        "PV_N": (0, 20),
        "WindOnshore_N": (0, 20),
        "RO": (0, 1000),
        "AEL_100": (0, 1000),
        "Ammonia_synthesis": (0, 1000), 
        "Methanol_synthesis": (0, 1000),
        "HP": (0, 1000),
        "DAC": (0, 1000),
        "FTL": (0, 1000),  
        "LDV_el": (0, 1000),
        "HDV_el": (0, 1000),
        "HDV_BF": (0, 1000),
        "MDV_el": (0, 1000),
        "MDV_BF": (0, 1000),
        "Two_wheel_el": (0, 1000),
        "Bus_el": (0, 1000),
        "Marine_e": (0, 1000),
        "Aviation_el": (0, 1000),
        "Aviation_e": (0, 1000),
        "cook_el": (0, 1000),
        "cook_LPG": (0, 1000),
        "Industry_EH": (0, 1000),
        "DW_heat": (0, 1000),
        "Dummy_Ammonia": (0, 1000),
        "Dummy_Methanol": (0, 1000),
        "Industry_EL": (0, 1000),
        "ST_N_DW": (0, 1000000),
        "WindOffshore_N": (0, 20)
    },
    "PNG_data": {  
        # "DG": (0, .350),
        # "NG_plant": (0.082, 0.082),
#        "BG_B": (0.0182, 0.0182),
#        "PV_B": (0.0031, 0.0031),
        "Hydro_B": (0.115, 0.115),
        "Geothermal_B": (0.011, 0.011),
        "MDV": (0, 1000),
        "HDV": (0, 1000),
        "LDV": (0, 1000),
        "LDV_BF": (0, 1000),
        "Bus": (0, 1000),
        "Two_wheel": (0, 1000),
        "Aviation": (0, 1000),
        "Marine": (0, 1000),
        "cook_b": (0, 1000),
        "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000),
        "DW_Electric_converter": (0, 1000),
        "DW_Electric_converter_2": (0, 1000),
        "ST_N": (0, 1000000),
        "HFO": (0, 1000),
        "BG_N": (0, 0.452),
        "PV_N": (0, 20),
        "WindOnshore_N": (0, 20),
        "Wave_N": (0, 20),
        "RO": (0, 1000),
        "AEL_100": (0, 1000),
        "Ammonia_synthesis": (0, 1000), 
        "Methanol_synthesis": (0, 1000), 
        "HP": (0, 1000),
        "DAC": (0, 1000), 
        "FTL": (0, 1000), 
        "LDV_el": (0, 1000),
        "HDV_el": (0, 1000),
        "HDV_BF": (0, 1000),
        "MDV_el": (0, 1000),
        "MDV_BF": (0, 1000),
        "Two_wheel_el": (0, 1000),
        "Bus_el": (0, 1000),
        "Marine_e": (0, 1000),
        "Aviation_el": (0, 1000),
        "Aviation_e": (0, 1000),
        "cook_el": (0, 1000),
        "cook_LPG": (0, 1000),
        "Industry_EH": (0, 1000),
        "DW_heat": (0, 1000),
        "Dummy_Ammonia": (0, 1000),
        "Dummy_Methanol": (0, 1000),
        "Industry_EL": (0, 1000),
        "ST_N_DW": (0, 1000000),
        "WindOffshore_N": (0, 20)
        },
    "SA_data": { 
        # "DG": (0, 0.0315),
#        "BG_B": (0.0011, 0.0011),
#        "PV_B": (0.0138, 0.0138),
#        "WindOnshore_B": (0.0005, 0.0005),
        "Hydro_B": (.0063, .0063),
        "MDV": (0, 1000),
        "HDV": (0, 1000),
        "LDV": (0, 1000),
        "LDV_BF": (0, 1000),
        "Bus": (0, 1000),
        "Two_wheel": (0, 1000),
        "Aviation": (0, 1000),
        "Marine": (0, 1000),
        "cook_b": (0, 1000),
        "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000),
        "DW_Electric_converter": (0, 1000),
        "DW_Electric_converter_2": (0, 1000),
        "ST_N": (0, 1000000),
        "HFO": (0, 1000),
        "BG_N": (0, 0.28),
        "PV_N": (0, 20),
        "WindOnshore_N": (0, 20),
        "Wave_N": (0, 20),
        "RO": (0, 1000),
        "AEL_100": (0, 1000),
        "Ammonia_synthesis": (0, 1000), 
        "Methanol_synthesis": (0, 1000),
        "HP": (0, 1000),
        "DAC": (0, 1000),
        "FTL": (0, 1000), 
        "LDV_el": (0, 1000),
        "HDV_el": (0, 1000),
        "HDV_BF": (0, 1000),
        "MDV_el": (0, 1000),
        "MDV_BF": (0, 1000),
        "Two_wheel_el": (0, 1000),
        "Bus_el": (0, 1000),
        "Marine_e": (0, 1000),
        "Aviation_el": (0, 1000),
        "Aviation_e": (0, 1000),
        "cook_el": (0, 1000),
        "cook_LPG": (0, 1000),
        "Industry_EH": (0, 1000),
        "DW_heat": (0, 1000),
        "Dummy_Ammonia": (0, 1000),
        "Dummy_Methanol": (0, 1000),
        "Industry_EL": (0, 1000),
        "ST_N_DW": (0, 1000000),
        "WindOffshore_N": (0, 20)# hydro adjusted
    },
    "SI_data": {   
        # "DG": (0, 0.0527),
 #       "BG_B": (0.0008, 0.0008),
 #       "PV_B": (0.0023, 0.0023),
        "Hydro_B": (.00018, .00018),
        "MDV": (0, 1000),
        "HDV": (0, 1000),
        "LDV": (0, 1000),
        "LDV_BF": (0, 1000),
        "Bus": (0, 1000),
        "Two_wheel": (0, 1000),
        "Aviation": (0, 1000),
        "Marine": (0, 1000),
        "cook_b": (0, 1000),
        "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000),
        "DW_Electric_converter": (0, 1000),
        "DW_Electric_converter_2": (0, 1000),
        "ST_N": (0, 1000000),
        "HFO": (0, 1000),
        "BG_N": (0, 1.44),
        "PV_N": (0, 20),
        "WindOnshore_N": (0, 20),
        "Wave_N": (0, 20),
        "WindOffshore_N": (0, 20),
        "RO": (0, 1000),
        "AEL_100": (0, 1000),
        "Ammonia_synthesis": (0, 1000), 
        "Methanol_synthesis": (0, 1000), 
        "HP": (0, 1000),
        "DAC": (0, 1000), 
        "FTL": (0, 1000), 
        "LDV_el": (0, 1000),
        "HDV_el": (0, 1000),
        "HDV_BF": (0, 1000),
        "MDV_el": (0, 1000),
        "MDV_BF": (0, 1000),
        "Two_wheel_el": (0, 1000),
        "Bus_el": (0, 1000),
        "Marine_e": (0, 1000),
        "Aviation_el": (0, 1000),
        "Aviation_e": (0, 1000),
        "cook_el": (0, 1000),
        "cook_LPG": (0, 1000),
        "Industry_EH": (0, 1000),
        "DW_heat": (0, 1000),
        "Dummy_Ammonia": (0, 1000),
        "Dummy_Methanol": (0, 1000),
        "Industry_EL": (0, 1000),
        "ST_N_DW": (0, 1000000),
        "Hydro_N": (.00675, .00675),
        # hydro adjusted
    },
    "TA_data": {  
        # "DG": (0, 0.0167),
#        "PV_B": (0.0071, 0.0071),
#        "WindOnshore_B": (0.00151, 0.00151),
        "MDV": (0, 1000),
        "HDV": (0, 1000),
        "LDV": (0, 1000),
        "LDV_BF": (0, 1000),
        "Bus": (0, 1000),
        "Two_wheel": (0, 1000),
        "Aviation": (0, 1000),
        "Marine": (0, 1000),
        "cook_b": (0, 1),
        "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000),
        "DW_Electric_converter": (0, 1000),
        "DW_Electric_converter_2": (0, 1000),
        "ST_N": (0, 1000000),
        "HFO": (0, 1000),
        "BG_N": (0, 0.20),
        "PV_N": (0, 20),
        "WindOnshore_N": (0, 20),
        "Wave_N": (0, 20),
        "RO": (0, 1000),
        "AEL_100": (0, 1000),
        "Ammonia_synthesis": (0, 1000), 
        "Methanol_synthesis": (0, 1000), 
        "HP": (0, 1000),
        "DAC": (0, 1000), 
        "FTL": (0, 1000), 
        "LDV_el": (0, 1000),
        "HDV_el": (0, 1000),
        "HDV_BF": (0, 1000),
        "MDV_el": (0, 1000),
        "MDV_BF": (0, 1000),
        "Two_wheel_el": (0, 1000),
        "Bus_el": (0, 1000),
        "Marine_e": (0, 1000),
        "Aviation_el": (0, 1000),
        "Aviation_e": (0, 1000),
        "cook_el": (0, 1000),
        "cook_LPG": (0, 1000),
        "Industry_EH": (0, 1000),
        "DW_heat": (0, 1000),
        "Dummy_Ammonia": (0, 1000),
        "Dummy_Methanol": (0, 1000),
        "Industry_EL": (0, 1000),
        "ST_N_DW": (0, 1000000),
        "WindOffshore_N": (0, 20)
    },
    "TU_data": {  
        # "DG": (0, 0.003),
#        "PV_B": (0.0029, 0.0029),
        "MDV": (0, 1000),
        "HDV": (0, 1000),
        "LDV": (0, 1000),
        "LDV_BF": (0, 1000),
        "Bus": (0, 1000),
        "Two_wheel": (0, 1000),
        "Aviation": (0, 1000),
        "Marine": (0, 1000),
        "cook_b": (0, 1),
        "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000),
        "DW_Electric_converter": (0, 1000),
        "DW_Electric_converter_2": (0, 1000),
        "ST_N": (0, 1000000),
        "HFO": (0, 1000),
        "BG_N": (0, 0.0084),
        "PV_N": (0, 20),
        "WindOnshore_N": (0, 20),
        "RO": (0, 1000),
        "AEL_100": (0, 1000),
        "Ammonia_synthesis": (0, 1000), 
        "Methanol_synthesis": (0, 1000), 
        "HP": (0, 1000),
        "DAC": (0, 1000), 
        "FTL": (0, 1000), 
        "LDV_el": (0, 1000),
        "HDV_el": (0, 1000),
        "HDV_BF": (0, 1000),
        "MDV_el": (0, 1000),
        "MDV_BF": (0, 1000),
        "Two_wheel_el": (0, 1000),
        "Bus_el": (0, 1000),
        "Marine_e": (0, 1000),
        "Aviation_el": (0, 1000),
        "Aviation_e": (0, 1000),
        "cook_el": (0, 1000),
        "cook_LPG": (0, 1000),
        "Industry_EH": (0, 1000),
        "DW_heat": (0, 1000),
        "Dummy_Ammonia": (0, 1000),
        "Dummy_Methanol": (0, 1000),
        "Industry_EL": (0, 1000),
        "ST_N_DW": (0, 1000000),
        "Wave_N": (0, 20)
    },
    "VU_data": {  
        # "DG": (0, 0.0232),
#        "PV_B": (0.0044, 0.0044),
#        "WindOnshore_B": (0.0032, 0.0032),
        "Hydro_B": (.00054, .00054),
        "MDV": (0, 1000),
        "HDV": (0, 1000),
        "LDV": (0, 1000),
        "LDV_BF": (0, 1000),
        "Bus": (0, 1000),
        "Two_wheel": (0, 1000),
        "Aviation": (0, 1000),
        "Marine": (0, 1000),
        "cook_b": (0, 1),
        "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000),
        "DW_Electric_converter": (0, 1000),
        "DW_Electric_converter_2": (0, 1000),
        "ST_N": (0, 1000000),
        "ST_N_DW": (0, 1000000),
        "HFO": (0, 1000),
        "BG_N": (0, .062),
        "PV_N": (0, 10),
        "WindOnshore_N": (0, 10),
        "Wave_N": (0, 10),
        "RO": (0, 1000),
        "AEL_100": (0, 1000),
        "Ammonia_synthesis": (0, 1000),  
        "Methanol_synthesis": (0, 1000), 
        "DAC": (0, 1000), 
        "FTL": (0, 1000),  
        "LDV_el": (0, 1000),
        "HDV_el": (0, 1000),
        "HDV_BF": (0, 1000),
        "MDV_el": (0, 1000),
        "MDV_BF": (0, 1000),
        "Two_wheel_el": (0, 1000),
        "Bus_el": (0, 1000),
        "Marine_e": (0, 1000),
        "Aviation_el": (0, 1000),
        "Aviation_e": (0, 1000),
        "cook_el": (0, 1000),
        "cook_LPG": (0, 1000),
        "Industry_EH": (0, 1000),
        "DW_heat": (0, 1000),
        "Dummy_Ammonia": (0, 1000),
        "Dummy_Methanol": (0, 1000),
        "Industry_EL": (0, 1000),
        "WindOffshore_N": (0, 10)# hydro adjusted
    }
}
# Build DataFrame index
all_techs = list({tech for node in capacity_limits for tech in capacity_limits[node]})
converter_capacityParam = pd.DataFrame(
    index=pd.MultiIndex.from_product([m.set.nodesdata, ['2050'], all_techs])
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
            ["DG", "BG_B", "PV_B", "WindOnshore_B", "Hydro_B", "Geothermal_B","MDV","HDV", "LDV", "Bus", "Two_wheel", "Aviation", "Marine","cook_b", "Industry", "DW_LPG_converter", "DW_Electric_converter", "NG_plant", "HFO"],
            ['2020'],
            ["Powergen"],
            ["Biomass", "Elec", "CO2", "Diesel", "Gasoline", "JetA1", "MDO", "T_MDV_th", "T_HDV_th","T_LDV_th","T_Bus_th","T_Two_wheel_th","T_Aviation_th","T_Marine_th", "T_Marine_f_th","Heat_cooking", "Heat_industry", "LPG", "DHW_LPG", "DHW_el","NG", "HFOO"],
        ]
    )
)
converter_coefficient.loc[idx["DG", :, :, "Elec"], "coefficient"] = 1  # GWh_el
converter_coefficient.loc[idx["DG", :, :, "Diesel"], "coefficient"] = -2.85  # GWh_ch
converter_coefficient.loc[idx["DG", :, :, "CO2"], "coefficient"] = 0.76

converter_coefficient.loc[idx["NG_plant", :, :, "Elec"], "coefficient"] = 1  # GWh_el
converter_coefficient.loc[idx["NG_plant", :, :, "NG"], "coefficient"] = -2  # GWh_ch
converter_coefficient.loc[idx["NG_plant", :, :, "CO2"], "coefficient"] = 0.40

converter_coefficient.loc[idx["BG_B", :, :, "Elec"], "coefficient"] = 1  # GWh_el
converter_coefficient.loc[idx["BG_B", :, :, "Biomass"], "coefficient"] = -2.85  # GWh_ch
converter_coefficient.loc[idx["BG_B", :, :, "CO2"], "coefficient"] = 0

converter_coefficient.loc[idx["PV_B", :, :, "Elec"], "coefficient"] = 1  # GWh_el

converter_coefficient.loc[idx["WindOnshore_B", :, :, "Elec"], "coefficient"] = 1  # GWh_el

converter_coefficient.loc[idx["Hydro_B", :, :, "Elec"], "coefficient"] = 1 
converter_coefficient.loc[idx["Geothermal_B", :, :, "Elec"], "coefficient"] = 1 

converter_coefficient.loc[idx["cook_b",:,:,"Heat_cooking"], "coefficient"] = 1  # GWh_el # GWh_ch
converter_coefficient.loc[idx["cook_b",:, :, "Biomass"], "coefficient"] = -1
converter_coefficient.loc[idx["cook_b",:, :, "CO2"], "coefficient"] = 0

converter_coefficient.loc[idx["Industry",:,:,"Heat_industry"], "coefficient"] = 1  # GWh_el # GWh_ch
converter_coefficient.loc[idx["Industry",:, :, "Diesel"], "coefficient"] = -1.17
converter_coefficient.loc[idx["Industry",:, :, "CO2"], "coefficient"] = .31

converter_coefficient.loc[idx["DW_LPG_converter",:,:,"DHW_LPG"], "coefficient"] = 1  # GWh_el # GWh_ch
converter_coefficient.loc[idx["DW_LPG_converter",:, :, "LPG"], "coefficient"] = -1.17
converter_coefficient.loc[idx["DW_LPG_converter",:, :, "CO2"], "coefficient"] = .27

converter_coefficient.loc[idx["DW_Electric_converter",:,:,"DHW_el"], "coefficient"] = 1  # GWh_el # GWh_ch
converter_coefficient.loc[idx["DW_Electric_converter",:, :, "Elec"], "coefficient"] = -1.05
converter_coefficient.loc[idx["DW_Electric_converter",:, :, "CO2"], "coefficient"] = 0

converter_coefficient.loc[idx["MDV", :, :, "T_MDV_th"], "coefficient"] = 1  # GWh_el # GWh_ch
converter_coefficient.loc[idx["MDV",:, :, "Diesel"], "coefficient"] = -1
converter_coefficient.loc[idx["MDV",:, :, "CO2"], "coefficient"] = 0.26

converter_coefficient.loc[idx["HDV",:, :, "T_HDV_th"], "coefficient"] = 1  # GWh_el # GWh_ch
converter_coefficient.loc[idx["HDV",:,:,"Diesel"], "coefficient"] = -1
converter_coefficient.loc[idx["HDV",:,:,"CO2"], "coefficient"] = 0.26

converter_coefficient.loc[idx["LDV",:, :, "T_LDV_th"], "coefficient"] = 1  # GWh_el # GWh_ch
converter_coefficient.loc[idx["LDV",:,:,"Gasoline"], "coefficient"] = -1
converter_coefficient.loc[idx["LDV",:,:,"CO2"], "coefficient"] = 0.25

converter_coefficient.loc[idx["Bus",:, :, "T_Bus_th"], "coefficient"] = 1  # GWh_el # GWh_ch
converter_coefficient.loc[idx["Bus",:,:,"Diesel"], "coefficient"] = -1
converter_coefficient.loc[idx["Bus",:,:,"CO2"], "coefficient"] = 0.26

converter_coefficient.loc[idx["Two_wheel",:, :, "T_Two_wheel_th"], "coefficient"] = 1  # GWh_el # GWh_ch
converter_coefficient.loc[idx["Two_wheel",:,:,"Gasoline"], "coefficient"] = -1
converter_coefficient.loc[idx["Two_wheel",:,:,"CO2"], "coefficient"] = 0.25

converter_coefficient.loc[idx["Aviation",:,:,"T_Aviation_th"], "coefficient"] = 1  # GWh_el # GWh_ch
converter_coefficient.loc[idx["Aviation",:, :, "JetA1"], "coefficient"] = -1
converter_coefficient.loc[idx["Aviation",:, :, "CO2"], "coefficient"] = 0.26

converter_coefficient.loc[idx["Marine",:,:,"T_Marine_th"], "coefficient"] = 1  # GWh_el # GWh_ch
converter_coefficient.loc[idx["Marine",:, :, "MDO"], "coefficient"] = -1
converter_coefficient.loc[idx["Marine",:, :, "CO2"], "coefficient"] = 0.27

converter_coefficient.loc[idx["HFO",:,:,"T_Marine_f_th"], "coefficient"] = 1  # GWh_el # GWh_ch
converter_coefficient.loc[idx["HFO",:, :, "HFOO"], "coefficient"] = -1
converter_coefficient.loc[idx["HFO",:, :, "CO2"], "coefficient"] = 0.29


converter_coefficient = converter_coefficient.dropna(how="all")

m.parameter.add(converter_coefficient, "converter_coefficient")
converter_coefficient
###############################################################################
converter_coefficient = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [
            ["BG_N", "PV_N", "WindOnshore_N", "Wave_N","WindOffshore_N", "Hydro_N"],
            ['2030'],
            ["Powergen"],
            ["Biomass", "Elec", "CO2"],
        ]
    )
)
converter_coefficient.loc[idx["BG_N", :, :, "Elec"], "coefficient"] = 1  # GWh_el
converter_coefficient.loc[idx["BG_N", :, :, "Biomass"], "coefficient"] = -2.85  # GWh_ch
converter_coefficient.loc[idx["BG_N", :, :, "CO2"], "coefficient"] = 0.02 #kt co2

converter_coefficient.loc[idx["PV_N", :, :, "Elec"], "coefficient"] = 1  # GWh_el

converter_coefficient.loc[idx["WindOnshore_N", :, :, "Elec"], "coefficient"] = 1

converter_coefficient.loc[idx["Wave_N", :, :, "Elec"], "coefficient"] = 1  
converter_coefficient.loc[idx["WindOffshore_N", :, :, "Elec"], "coefficient"] = 1 
converter_coefficient.loc[idx["Hydro_N", :, :, "Elec"], "coefficient"] = 1 





converter_coefficient = converter_coefficient.dropna(how="all")

m.parameter.add(converter_coefficient, "converter_coefficient")
converter_coefficient
#################################################################################
converter_coefficient = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [
            ["BG_N","PV_N","WindOnshore_N", "Wave_N","WindOffshore_N", "ST_N","Industry_EL", "LDV_BF", "RO","Ammonia_synthesis", "DAC", "Methanol_synthesis", "HP", "FTL","AEL_100", "LDV_el", "HDV_el", "HDV_BF", "MDV_el", "MDV_BF", "Two_wheel_el", "Bus_el", "Marine_e", "Aviation_el", "Aviation_e", "cook_el", "cook_LPG", "Industry_EH", "DW_heat", "Dummy_Ammonia", "Dummy_Methanol", "ST_N_DW", "DW_Electric_converter_2"],
            ['2040', "2050"],
            ["Powergen"],
            ["Biomass", "Elec", "LPG","CO2", "T_LDV_BF","Pure_water", "Hydrogen", "Heat","ST_Heat", "Ammonia", "co", "Methanol", "eKerosene", "T_LDV_el","T_HDV_el" , "T_HDV_BF", "T_MDV_el", "T_MDV_BF", "T_Two_wheel_el", "T_Bus_el","T_Marine_e", "T_Aviation_el", "T_Aviation_e", "DHW_ST", "T_cook_el", "T_cook_LPG", "T_Industry_EH", "T_DHW_heat", "Dummy_EL", "DHW_el"],
        ]
    )
)

# converter_coefficient.loc[idx["DW_Electric_converter_2",:,:,"DHW_el"], "coefficient"] = 1  # GWh_el # GWh_ch
# converter_coefficient.loc[idx["DW_Electric_converter_2",:, :, "Elec"], "coefficient"] = -1.05
# converter_coefficient.loc[idx["DW_Electric_converter_2",:, :, "CO2"], "coefficient"] = 0
converter_coefficient.loc[idx["BG_N", :, :, "Elec"], "coefficient"] = 1  # GWh_el
converter_coefficient.loc[idx["BG_N", :, :, "Biomass"], "coefficient"] = -2.85  # GWh_ch
converter_coefficient.loc[idx["BG_N", :, :, "CO2"], "coefficient"] = 0.02 #kt co2

converter_coefficient.loc[idx["PV_N", :, :, "Elec"], "coefficient"] = 1  # GWh_el

converter_coefficient.loc[idx["WindOnshore_N", :, :, "Elec"], "coefficient"] = 1

converter_coefficient.loc[idx["Wave_N", :, :, "Elec"], "coefficient"] = 1  
converter_coefficient.loc[idx["WindOffshore_N", :, :, "Elec"], "coefficient"] = 1 


converter_coefficient.loc[idx["ST_N",:,:,"Heat"], "coefficient"] = 1

converter_coefficient.loc[idx["ST_N_DW",:,:,"DHW_el"], "coefficient"] = 1
converter_coefficient.loc[idx["ST_N_DW",:,:,"ST_Heat"], "coefficient"] = -1  # GWh_el # GWh_ch

converter_coefficient.loc[idx["LDV_BF",:, :, "T_LDV_BF"], "coefficient"] = 1  # GWh_el # GWh_ch
converter_coefficient.loc[idx["LDV_BF",:,:,"Biomass"], "coefficient"] = -2.85
converter_coefficient.loc[idx["LDV_BF",:,:,"CO2"], "coefficient"] = 0.25

converter_coefficient.loc[idx["LDV_el",:, :, "T_LDV_el"], "coefficient"] = 1  # GWh_el # GWh_ch
converter_coefficient.loc[idx["LDV_el",:,:,"Elec"], "coefficient"] = -1

converter_coefficient.loc[idx["HDV_el",:, :, "T_HDV_el"], "coefficient"] = 1  # GWh_el # GWh_ch
converter_coefficient.loc[idx["HDV_el",:,:,"Elec"], "coefficient"] = -1

converter_coefficient.loc[idx["HDV_BF",:, :, "T_HDV_BF"], "coefficient"] = 1  # GWh_el # GWh_ch
converter_coefficient.loc[idx["HDV_BF",:,:,"Biomass"], "coefficient"] = -2.85
converter_coefficient.loc[idx["HDV_BF",:,:,"CO2"], "coefficient"] = 0.25

converter_coefficient.loc[idx["MDV_BF",:, :, "T_MDV_BF"], "coefficient"] = 1  # GWh_el # GWh_ch
converter_coefficient.loc[idx["MDV_BF",:,:,"Biomass"], "coefficient"] = -2.85
converter_coefficient.loc[idx["MDV_BF",:,:,"CO2"], "coefficient"] = 0.25

converter_coefficient.loc[idx["MDV_el",:, :, "T_MDV_el"], "coefficient"] = 1  # GWh_el # GWh_ch
converter_coefficient.loc[idx["MDV_el",:,:,"Elec"], "coefficient"] = -1

converter_coefficient.loc[idx["Bus_el",:, :, "T_Bus_el"], "coefficient"] = 1  # GWh_el # GWh_ch
converter_coefficient.loc[idx["Bus_el",:,:,"Elec"], "coefficient"] = -1

converter_coefficient.loc[idx["Two_wheel_el",:, :, "T_Two_wheel_el"], "coefficient"] = 1  # GWh_el # GWh_ch
converter_coefficient.loc[idx["Two_wheel_el",:,:,"Elec"], "coefficient"] = -1

converter_coefficient.loc[idx["Bus_el",:, :, "T_Bus_el"], "coefficient"] = 1  # GWh_el # GWh_ch
converter_coefficient.loc[idx["Bus_el",:,:,"Elec"], "coefficient"] = -1

converter_coefficient.loc[idx["Aviation_el",:, :, "T_Aviation_el"], "coefficient"] = 1  # GWh_el # GWh_ch
converter_coefficient.loc[idx["Aviation_el",:,:,"Elec"], "coefficient"] = -1

converter_coefficient.loc[idx["RO",:, :, "Pure_water"], "coefficient"] = 1  # (1000 * m3) (can be 1 * 1000 m3 here and corresponding elec as input)
converter_coefficient.loc[idx["RO",:,:,"Elec"], "coefficient"] = -.00315# adding free sea water just consume computation

converter_coefficient.loc[idx["Ammonia_synthesis",:, :, "Ammonia"], "coefficient"] = 1 #Gwh ( 177 kg of H2 and 823 kg of N2 are theoretically necessary to produce1 ton of ammonia)
converter_coefficient.loc[idx["Ammonia_synthesis",:,:,"Elec"], "coefficient"] = -0.02 #(8% elec is supplied by other sources: https://www.sciencedirect.com/science/article/pii/S2666955223000205)
converter_coefficient.loc[idx["Ammonia_synthesis",:,:,"Hydrogen"], "coefficient"] = -1.14 

converter_coefficient.loc[idx["DAC",:, :, "co"], "coefficient"] = 1 #kt
converter_coefficient.loc[idx["DAC",:,:,"Elec"], "coefficient"] = -0.25 
converter_coefficient.loc[idx["DAC",:,:,"Heat"], "coefficient"] = -1.7 

converter_coefficient.loc[idx["Methanol_synthesis",:, :, "Methanol"], "coefficient"] = 1 #gwh
converter_coefficient.loc[idx["Methanol_synthesis",:,:,"Hydrogen"], "coefficient"] = -1.127 
converter_coefficient.loc[idx["Methanol_synthesis",:,:,"co"], "coefficient"] = -0.2485 #kt 

converter_coefficient.loc[idx["HP",:, :,"Heat"], "coefficient"] = 1
converter_coefficient.loc[idx["HP",:,:,"Elec"], "coefficient"] = -.285

converter_coefficient.loc[idx["FTL",:, :, "eKerosene"], "coefficient"] = 1 #gwh
converter_coefficient.loc[idx["FTL",:,:,"Hydrogen"], "coefficient"] = -1.2 # (https://ehb.eu/files/downloads/EHB-Analysing-the-future-demand-supply-and-transport-of-hydrogen-June-2021-v3.pdf) (1.2 TWH of h2 for 1 TWH of syn kerosene)
converter_coefficient.loc[idx["FTL",:,:,"co"], "coefficient"] = -0.305 #(CBR) kt
#converter_coefficient.loc[idx["FTL",:,:,"Heat"], "coefficient"] = 0.13 #:50% of .27 GWh have references for this) (.27 GWh_heat is released based on 165 KJ/mol of CO: https://en.wikipedia.org/wiki/Fischer%E2%80%93Tropsch_process#:~:text=The%20reaction%20is%20a%20highly,O%2C%20several%20reactions%20are%20necessary:)

# converter_coefficient.loc[idx["AEL",:,:,"Hydrogen"], "coefficient"] = .001 #Gwh
# converter_coefficient.loc[idx["AEL",:,:,"Pure_water"], "coefficient"] = -0.000450  # (1000 * m3) (Based on 15L/KG H2 from IRENA data and 70% eff)
# converter_coefficient.loc[idx["AEL",:,:,"Elec"], "coefficient"] = -.00142 
# #converter_coefficient.loc[idx["AEL",:,:,"Heat"], "coefficient"] = 0.38  # 30% of energy is heat and 90% is recoverable

# converter_coefficient.loc[idx["AEL_10",:,:,"Hydrogen"], "coefficient"] = .01 #Gwh
# converter_coefficient.loc[idx["AEL_10",:,:,"Pure_water"], "coefficient"] = -0.00450  # (1000 * m3) (Based on 15L/KG H2 from IRENA data and 70% eff)
# converter_coefficient.loc[idx["AEL_10",:,:,"Elec"], "coefficient"] = -.0142 
# #converter_coefficient.loc[idx["AEL_10",:,:,"Heat"], "coefficient"] = 0.38 


converter_coefficient.loc[idx["AEL_100",:,:,"Hydrogen"], "coefficient"] = 1 #Gwh
converter_coefficient.loc[idx["AEL_100",:,:,"Pure_water"], "coefficient"] = -0.450  # (1000 * m3) (Based on 15L/KG H2 from IRENA data and 70% eff)
converter_coefficient.loc[idx["AEL_100",:,:,"Elec"], "coefficient"] = -1.42 
#converter_coefficient.loc[idx["AEL_100",:,:,"Heat"], "coefficient"] = 0.38 

converter_coefficient.loc[idx["cook_el",:,:,"T_cook_el"], "coefficient"] = 1  # GWh_el # GWh_ch
converter_coefficient.loc[idx["cook_el",:, :, "Elec"], "coefficient"] = -1


converter_coefficient.loc[idx["cook_LPG",:,:,"T_cook_LPG"], "coefficient"] = 1  # GWh_el # GWh_ch
converter_coefficient.loc[idx["cook_LPG",:, :, "LPG"], "coefficient"] = -1.17
converter_coefficient.loc[idx["cook_LPG",:, :, "CO2"], "coefficient"] = .27

converter_coefficient.loc[idx["Industry_EH",:,:,"T_Industry_EH"], "coefficient"] = 1  # GWh_el # GWh_ch
converter_coefficient.loc[idx["Industry_EH",:, :, "Heat"], "coefficient"] = -1

converter_coefficient.loc[idx["Industry_EL",:,:,"T_Industry_EH"], "coefficient"] = 1  # GWh_el # GWh_ch
converter_coefficient.loc[idx["Industry_EL",:, :, "Elec"], "coefficient"] = -1.05

# converter_coefficient.loc[idx["DW_heat",:,:,"T_DHW_heat"], "coefficient"] = 1  # GWh_el # GWh_ch
# converter_coefficient.loc[idx["DW_heat",:, :, "Heat"], "coefficient"] = -1

converter_coefficient.loc[idx["Dummy_Ammonia",:,:,"Dummy_EL"], "coefficient"] = 1  # GWh_el # GWh_ch
converter_coefficient.loc[idx["Dummy_Ammonia",:, :, "Ammonia"], "coefficient"] = -1

converter_coefficient.loc[idx["Dummy_Methanol",:,:,"Dummy_EL"], "coefficient"] = 1  # GWh_el # GWh_ch
converter_coefficient.loc[idx["Dummy_Methanol",:, :, "Methanol"], "coefficient"] = -1

converter_coefficient = converter_coefficient.dropna(how="all")

m.parameter.add(converter_coefficient, "converter_coefficient")
converter_coefficient
######################################################################################################
# converter_coefficient = pd.DataFrame(
#     index=pd.MultiIndex.from_product(
#         [
#             ["ST_N", "LDV_BF", "RO","Ammonia_synthesis", "DAC", "Methanol_synthesis", "HP", "FTL","AEL_100", "LDV_el", "HDV_el", "HDV_BF", "MDV_el", "MDV_BF", "Two_wheel_el", "Bus_el", "Marine_e", "Aviation_el", "Aviation_e", "cook_el", "cook_LPG", "Industry_EH", "DW_heat"],
#             ['2050'],
#             ["Powergen"],
#             ["Biomass", "Elec", "LPG","CO2", "T_LDV_BF","Pure_water", "Hydrogen", "Heat", "Ammonia", "co", "Methanol", "eKerosene", "DHW_Elec", "T_LDV_el","T_HDV_el" , "T_HDV_BF", "T_MDV_el", "T_MDV_BF", "T_Two_wheel_el", "T_Bus_el","T_Marine_e", "T_Aviation_el", "T_Aviation_e", "DHW_ST", "T_cook_el", "T_cook_LPG", "T_Industry_EH", "T_DHW_heat"],
#         ]
#     )
# )


# converter_coefficient.loc[idx["ST_N",:,:,"Heat"], "coefficient"] = 1  # GWh_el # GWh_ch

# converter_coefficient.loc[idx["LDV_BF",:, :, "T_LDV_BF"], "coefficient"] = 1  # GWh_el # GWh_ch
# converter_coefficient.loc[idx["LDV_BF",:,:,"Biomass"], "coefficient"] = -2.85
# converter_coefficient.loc[idx["LDV_BF",:,:,"CO2"], "coefficient"] = 0.25

# converter_coefficient.loc[idx["LDV_el",:, :, "T_LDV_el"], "coefficient"] = 1  # GWh_el # GWh_ch
# converter_coefficient.loc[idx["LDV_el",:,:,"Elec"], "coefficient"] = -1

# converter_coefficient.loc[idx["HDV_el",:, :, "T_HDV_el"], "coefficient"] = 1  # GWh_el # GWh_ch
# converter_coefficient.loc[idx["HDV_el",:,:,"Elec"], "coefficient"] = -1

# converter_coefficient.loc[idx["HDV_BF",:, :, "T_HDV_BF"], "coefficient"] = 1  # GWh_el # GWh_ch
# converter_coefficient.loc[idx["HDV_BF",:,:,"Biomass"], "coefficient"] = -2.85
# converter_coefficient.loc[idx["HDV_BF",:,:,"CO2"], "coefficient"] = 0.25

# converter_coefficient.loc[idx["MDV_BF",:, :, "T_MDV_BF"], "coefficient"] = 1  # GWh_el # GWh_ch
# converter_coefficient.loc[idx["MDV_BF",:,:,"Biomass"], "coefficient"] = -2.85
# converter_coefficient.loc[idx["MDV_BF",:,:,"CO2"], "coefficient"] = 0.25

# converter_coefficient.loc[idx["MDV_el",:, :, "T_MDV_el"], "coefficient"] = 1  # GWh_el # GWh_ch
# converter_coefficient.loc[idx["MDV_el",:,:,"Elec"], "coefficient"] = -1

# converter_coefficient.loc[idx["Bus_el",:, :, "T_Bus_el"], "coefficient"] = 1  # GWh_el # GWh_ch
# converter_coefficient.loc[idx["Bus_el",:,:,"Elec"], "coefficient"] = -1

# converter_coefficient.loc[idx["Two_wheel_el",:, :, "T_Two_wheel_el"], "coefficient"] = 1  # GWh_el # GWh_ch
# converter_coefficient.loc[idx["Two_wheel_el",:,:,"Elec"], "coefficient"] = -1

# converter_coefficient.loc[idx["Bus_el",:, :, "T_Bus_el"], "coefficient"] = 1  # GWh_el # GWh_ch
# converter_coefficient.loc[idx["Bus_el",:,:,"Elec"], "coefficient"] = -1

# converter_coefficient.loc[idx["Aviation_el",:, :, "T_Aviation_el"], "coefficient"] = 1  # GWh_el # GWh_ch
# converter_coefficient.loc[idx["Aviation_el",:,:,"Elec"], "coefficient"] = -1

# converter_coefficient.loc[idx["RO",:, :, "Pure_water"], "coefficient"] = 285.7  # (1000 * m3)
# converter_coefficient.loc[idx["RO",:,:,"Elec"], "coefficient"] = -1# adding free sea water just consume computation

# converter_coefficient.loc[idx["Ammonia_synthesis",:, :, "Ammonia"], "coefficient"] = 1 #Gwh ( 177 kg of H2 and 823 kg of N2 are theoretically necessary to produce1 ton of ammonia)
# converter_coefficient.loc[idx["Ammonia_synthesis",:,:,"Elec"], "coefficient"] = -0.02 #(8% elec is supplied by other sources: https://www.sciencedirect.com/science/article/pii/S2666955223000205)
# converter_coefficient.loc[idx["Ammonia_synthesis",:,:,"Hydrogen"], "coefficient"] = -1.14 

# converter_coefficient.loc[idx["DAC",:, :, "co"], "coefficient"] = 1 #kt
# converter_coefficient.loc[idx["DAC",:,:,"Elec"], "coefficient"] = -0.25 
# converter_coefficient.loc[idx["DAC",:,:,"Heat"], "coefficient"] = -1.7 

# converter_coefficient.loc[idx["Methanol_synthesis",:, :, "Methanol"], "coefficient"] = 1 #gwh
# converter_coefficient.loc[idx["Methanol_synthesis",:,:,"Hydrogen"], "coefficient"] = -1.127 
# converter_coefficient.loc[idx["Methanol_synthesis",:,:,"co"], "coefficient"] = -0.2485 #kt 

# converter_coefficient.loc[idx["HP",:, :,"Heat"], "coefficient"] = 1
# converter_coefficient.loc[idx["HP",:,:,"Elec"], "coefficient"] = -.33 

# converter_coefficient.loc[idx["FTL",:, :, "eKerosene"], "coefficient"] = 1 #gwh
# converter_coefficient.loc[idx["FTL",:,:,"Hydrogen"], "coefficient"] = -1.2 # (https://ehb.eu/files/downloads/EHB-Analysing-the-future-demand-supply-and-transport-of-hydrogen-June-2021-v3.pdf) (1.2 TWH of h2 for 1 TWH of syn kerosene)
# converter_coefficient.loc[idx["FTL",:,:,"co"], "coefficient"] = -0.305 #(CBR) kt
# #converter_coefficient.loc[idx["FTL",:,:,"Heat"], "coefficient"] = 0.13 #:50% of .27 GWh have references for this) (.27 GWh_heat is released based on 165 KJ/mol of CO: https://en.wikipedia.org/wiki/Fischer%E2%80%93Tropsch_process#:~:text=The%20reaction%20is%20a%20highly,O%2C%20several%20reactions%20are%20necessary:)

# # converter_coefficient.loc[idx["AEL",:,:,"Hydrogen"], "coefficient"] = .001 #Gwh
# # converter_coefficient.loc[idx["AEL",:,:,"Pure_water"], "coefficient"] = -0.000450  # (1000 * m3) (Based on 15L/KG H2 from IRENA data and 70% eff)
# # converter_coefficient.loc[idx["AEL",:,:,"Elec"], "coefficient"] = -.00142 
# # #converter_coefficient.loc[idx["AEL",:,:,"Heat"], "coefficient"] = 0.38  # 30% of energy is heat and 90% is recoverable

# # converter_coefficient.loc[idx["AEL_10",:,:,"Hydrogen"], "coefficient"] = .01 #Gwh
# # converter_coefficient.loc[idx["AEL_10",:,:,"Pure_water"], "coefficient"] = -0.00450  # (1000 * m3) (Based on 15L/KG H2 from IRENA data and 70% eff)
# # converter_coefficient.loc[idx["AEL_10",:,:,"Elec"], "coefficient"] = -.0142 
# # #converter_coefficient.loc[idx["AEL_10",:,:,"Heat"], "coefficient"] = 0.38 


# converter_coefficient.loc[idx["AEL_100",:,:,"Hydrogen"], "coefficient"] = 1 #Gwh
# converter_coefficient.loc[idx["AEL_100",:,:,"Pure_water"], "coefficient"] = -0.450  # (1000 * m3) (Based on 15L/KG H2 from IRENA data and 70% eff)
# converter_coefficient.loc[idx["AEL_100",:,:,"Elec"], "coefficient"] = -1.42 
# #converter_coefficient.loc[idx["AEL_100",:,:,"Heat"], "coefficient"] = 0.38 

# converter_coefficient.loc[idx["cook_el",:,:,"T_cook_el"], "coefficient"] = 1  # GWh_el # GWh_ch
# converter_coefficient.loc[idx["cook_el",:, :, "Elec"], "coefficient"] = -1


# converter_coefficient.loc[idx["cook_LPG",:,:,"T_cook_LPG"], "coefficient"] = 1  # GWh_el # GWh_ch
# converter_coefficient.loc[idx["cook_LPG",:, :, "LPG"], "coefficient"] = -1.17
# converter_coefficient.loc[idx["cook_LPG",:, :, "CO2"], "coefficient"] = .27

# converter_coefficient.loc[idx["Industry_EH",:,:,"T_Industry_EH"], "coefficient"] = 1  # GWh_el # GWh_ch
# converter_coefficient.loc[idx["Industry_EH",:, :, "Heat"], "coefficient"] = -1

# converter_coefficient.loc[idx["DW_heat",:,:,"T_DHW_heat"], "coefficient"] = 1  # GWh_el # GWh_ch
# converter_coefficient.loc[idx["DW_heat",:, :, "Heat"], "coefficient"] = -1


# converter_coefficient = converter_coefficient.dropna(how="all")

# m.parameter.add(converter_coefficient, "converter_coefficient")
# converter_coefficient
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
for data_node in ["CI_data","FJ_data","FSM_data", "KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"]:
    
    region_code = data_node.split("_")[0]  # "R1" or "R2"
    
    techs = ["PV_B", "WindOnshore_B", "Hydro_B", "Geothermal_B", "BG_B","PV_N", "WindOnshore_N", "Wave_N", "WindOffshore_N", "BG_N", "Hydro_N"]
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
##########################################2030###################################
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
    converter_activityProfile["years"] = "2020"
    converter_activityProfile["type"] = "upper"

    converter_activityProfile = converter_activityProfile.reset_index().set_index(
        ["region", "years", "techs", "type"]
    )

    m.profile.add(converter_activityProfile, "converter_activityprofile")

converter_activityProfile.iloc[:, 0:8]
#################################################################################
for data_node in ["CI_data","FJ_data","FSM_data", "KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"]:
    
    region_code = data_node.split("_")[0]  # "R1" or "R2"
    
    techs = ["PV_B", "WindOnshore_B", "Hydro_B", "Geothermal_B", "BG_B","PV_N", "WindOnshore_N", "Wave_N", "WindOffshore_N", "BG_N", "Hydro_N","ST_N"]
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
    converter_activityProfile["years"] = "2040"
    converter_activityProfile["type"] = "upper"

    converter_activityProfile = converter_activityProfile.reset_index().set_index(
        ["region", "years", "techs", "type"]
    )

    m.profile.add(converter_activityProfile, "converter_activityprofile")

converter_activityProfile.iloc[:, 0:8]
################################################################################
for data_node in ["CI_data","FJ_data","FSM_data", "KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"]:
    
    region_code = data_node.split("_")[0]  # "R1" or "R2"
    
    techs = ["PV_B", "WindOnshore_B", "Hydro_B", "Geothermal_B", "BG_B","PV_N", "WindOnshore_N", "Wave_N", "WindOffshore_N", "BG_N", "Hydro_N","ST_N"]
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
    converter_activityProfile["years"] = "2050"
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
            ["DG", "BG_B", "PV_B", "WindOnshore_B", "Hydro_B", "Geothermal_B", "MDV","HDV", "LDV", "Bus", "Two_wheel", "Aviation", "Marine", "cook_b", "Industry", "DW_LPG_converter", "DW_Electric_converter", "HFOO"],
            ['2020'],
        ]
    )
).sort_index()

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "DG", "2020"], "perUnitBuild"
] = 0  # Mio EUR per unit
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "DG", "2020"], "useAnnuity"
] = 1  # binary yes/no
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "DG", "2020"], "amorTime"
] = 2  # years
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "DG", "2020"], "interest"
] = 0.06  # percent/100
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "DG", "2020"], "perUnitTotal"
] = 160

# accounting_converterUnits.loc[
#     idx["Invest", "global", "horizon", "DG", "2030"], "perUnitBuild"
# ] = 400  # Mio EUR per unit
# accounting_converterUnits.loc[
#     idx["Invest", "global", "horizon", "DG", "2030"], "useAnnuity"
# ] = 1  # binary yes/no
# accounting_converterUnits.loc[
#     idx["Invest", "global", "horizon", "DG", "2030"], "amorTime"
# ] = 25  # years
# accounting_converterUnits.loc[
#     idx["Invest", "global", "horizon", "DG", "2030"], "interest"
# ] = 0.06  # percent/100
# accounting_converterUnits.loc[
#     idx["OMFix", "global", "horizon", "DG", "2030"], "perUnitTotal"
# ] = 160

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "NG_plant", "2020"], "perUnitBuild"
] = 0  # Mio EUR per unit
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "NG_plant", "2020"], "useAnnuity"
] = 1  # binary yes/no
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "NG_plant", "2020"], "amorTime"
] = 25  # years
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "NG_plant", "2020"], "interest"
] = 0.06  # percent/100
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "NG_plant", "2020"], "perUnitTotal"
] = 87.6

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "BG_B", "2020"], "perUnitBuild"
] = 0  # Mio EUR per unit
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "BG_B", "2020"], "useAnnuity"
] = 1  # binary yes/no
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "BG_B", "2020"], "amorTime"
] = 25  # years
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "BG_B", "2020"], "interest"
] = 0.06  # percent/100
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "BG_B", "2020"], "perUnitTotal"
] = 78


accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "PV_B", "2020"], "perUnitBuild"
] = 0  # Mio EUR per unit
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "PV_B", "2020"], "useAnnuity"
] = 1  # binary yes/no
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "PV_B", "2020"], "amorTime"
] = 25  # years
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "PV_B", "2020"], "interest"
] = 0.06  # percent/100
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "PV_B", "2020"], "perUnitTotal"
] = 14

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "WindOnshore_B", "2020"], "perUnitBuild"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "WindOnshore_B", "2020"], "useAnnuity"
] = 1
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "WindOnshore_B", "2020"], "amorTime"
] = 25
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "WindOnshore_B", "2020"], "interest"
] = 0.06
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "WindOnshore_B", "2020"], "perUnitTotal"
] = 22



accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Hydro_B", "2020"], "perUnitBuild"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Hydro_B", "2020"], "useAnnuity"
] = 1
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Hydro_B", "2020"], "amorTime"
] = 25
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Hydro_B", "2020"], "interest"
] = 0.06
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "Hydro_B", "2020"], "perUnitTotal"
] = 168 * 2.22 ## to balance our reduction of capacity by 55%, capacity *.45

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Geothermal_B", "2020"], "perUnitBuild"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Geothermal_B", "2020"], "useAnnuity"
] = 1
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Geothermal_B", "2020"], "amorTime"
] = 25
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Geothermal_B", "2020"], "interest"
] = 0.06
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "Geothermal_B", "2020"], "perUnitTotal"
] = 118 * 4.54


accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "MDV", "2020"], "perUnitBuild"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "MDV", "2020"], "useAnnuity"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "MDV", "2020"], "amorTime"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "MDV", "2020"], "interest"
] = 0
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "MDV", "2020"], "perUnitTotal"
] = 0

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "HDV", "2020"], "perUnitBuild"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "HDV", "2020"], "useAnnuity"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "HDV", "2020"], "amorTime"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "HDV", "2020"], "interest"
] = 0
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "HDV", "2020"], "perUnitTotal"
] = 0


accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "LDV", "2020"], "perUnitBuild"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "LDV", "2020"], "useAnnuity"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "LDV", "2020"], "amorTime"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "LDV", "2020"], "interest"
] = 0
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "LDV", "2020"], "perUnitTotal"
] = 0


accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Bus", "2020"], "perUnitBuild"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Bus", "2020"], "useAnnuity"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Bus", "2020"], "amorTime"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Bus", "2020"], "interest"
] = 0
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "Bus", "2020"], "perUnitTotal"
] = 0

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Two_wheel", "2020"], "perUnitBuild"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Two_wheel", "2020"], "useAnnuity"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Two_wheel", "2020"], "amorTime"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Two_wheel", "2020"], "interest"
] = 0
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "Two_wheel", "2020"], "perUnitTotal"
] = 0

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Aviation", "2020"], "perUnitBuild"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Aviation", "2020"], "useAnnuity"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Aviation", "2020"], "amorTime"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Aviation", "2020"], "interest"
] = 0
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "Aviation", "2020"], "perUnitTotal"
] = 0

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Marine", "2020"], "perUnitBuild"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Marine", "2020"], "useAnnuity"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Marine", "2020"], "amorTime"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Marine", "2020"], "interest"
] = 0
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "Marine", "2020"], "perUnitTotal"
] = 0


accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "cook_b", "2020"], "perUnitBuild"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "cook_b", "2020"], "useAnnuity"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "cook_b", "2020"], "amorTime"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "cook_b", "2020"], "interest"
] = 0
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "cook_b", "2020"], "perUnitTotal"
] = 0

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Industry", "2020"], "perUnitBuild"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Industry", "2020"], "useAnnuity"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Industry", "2020"], "amorTime"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Industry", "2020"], "interest"
] = 0
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "Industry", "2020"], "perUnitTotal"
] = 0

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "DW_LPG_converter", "2020"], "perUnitBuild"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "DW_LPG_converter", "2020"], "useAnnuity"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "DW_LPG_converter", "2020"], "amorTime"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "DW_LPG_converter", "2020"], "interest"
] = 0
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "DW_LPG_converter", "2020"], "perUnitTotal"
] = 0

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "DW_Electric_converter", "2020"], "perUnitBuild"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "DW_Electric_converter", "2020"], "useAnnuity"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "DW_Electric_converter", "2020"], "amorTime"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "DW_Electric_converter", "2020"], "interest"
] = 0
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "DW_Electric_converter", "2020"], "perUnitTotal"
] = 0

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "HFO", "2020"], "perUnitBuild"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "HFO", "2020"], "useAnnuity"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "HFO", "2020"], "amorTime"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "HFO", "2020"], "interest"
] = 0
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "HFO", "2020"], "perUnitTotal"
] = 0


accounting_converterUnits = accounting_converterUnits.fillna(0)

m.parameter.add(accounting_converterUnits, "accounting_converterunits")
accounting_converterUnits
#################################################################################
#################################################################################
accounting_converterUnits = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [
            ["Invest", "OMFix"],
            ["global"],
            ["horizon"],
            ["PV_N", "WindOnshore_N", "Wave_N", "WindOffshore_N", "BG_N", "Hydro_N"],
            ['2030'],
        ]
    )
).sort_index()



accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "BG_N", "2030"], "perUnitBuild"
] = 2600.0  # Mio EUR per unit
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "BG_N", "2030"], "useAnnuity"
] = 1  # binary yes/no
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "BG_N", "2030"], "amorTime"
] = 25  # years
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "BG_N", "2030"], "interest"
] = 0.06  # percent/100
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "BG_N", "2030"], "perUnitTotal"
] = 78

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "PV_N", "2030"], "perUnitBuild"
] = 331
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "PV_N", "2030"], "useAnnuity"
] = 1
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "PV_N", "2030"], "amorTime"
] = 25
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "PV_N", "2030"], "interest"
] = 0.06
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "PV_N", "2030"], "perUnitTotal"
] = 7

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "WindOnshore_N", "2030"], "perUnitBuild"
] = 1100
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "WindOnshore_N", "2030"], "useAnnuity"
] = 1
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "WindOnshore_N", "2030"], "amorTime"
] = 25
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "WindOnshore_N", "2030"], "interest"
] = 0.06
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "WindOnshore_N", "2030"], "perUnitTotal"
] = 30

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Wave_N", "2030"], "perUnitBuild"
] = 3030
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Wave_N", "2030"], "useAnnuity"
] = 1
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Wave_N", "2030"], "amorTime"
] = 25
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Wave_N", "2030"], "interest"
] = 0.06
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "Wave_N", "2030"], "perUnitTotal"
] = 83

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "WindOffshore_N", "2030"], "perUnitBuild"
] = 2660
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "WindOffshore_N", "2030"], "useAnnuity"
] = 1
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "WindOffshore_N", "2030"], "amorTime"
] = 25
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "WindOffshore_N", "2030"], "interest"
] = 0.06
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "WindOffshore_N", "2030"], "perUnitTotal"
] = 75

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Hydro_N", "2030"], "perUnitBuild"
] = 12400 * 2.22
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Hydro_N", "2030"], "useAnnuity"
] = 1
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Hydro_N", "2030"], "amorTime"
] = 50
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Hydro_N", "2030"], "interest"
] = 0.06
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "Hydro_N", "2030"], "perUnitTotal"
] = 2.22 * 490

accounting_converterUnits = accounting_converterUnits.fillna(0)

m.parameter.add(accounting_converterUnits, "accounting_converterunits")
accounting_converterUnits


################################################################################
accounting_converterUnits = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [
            ["Invest", "OMFix"],
            ["global"],
            ["horizon"],
            ["PV_N", "WindOnshore_N", "Wave_N", "WindOffshore_N", "BG_N", "Hydro_N", "ST_N", "LDV_BF", "RO","AEL_100","Ammonia_synthesis", "DAC", "Methanol_synthesis", "HP", "FTL", "LDV_el", "HDV_el", "HDV_BF", "MDV_el", "MDV_BF", "Two_wheel_el", "Bus_el", "Marine_e", "Aviation_el", "Aviation_e", "cook_el", "cook_LPG", "Industry_EH", "DW_heat", "Dummy_Ammonia", "Dummy_Methanol", "Industry_EL", "ST_N_DW", "DW_Electric_converter_2"],
            ['2040'],
        ]
    )
).sort_index()


accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "DW_Electric_converter_2", "2040"], "perUnitBuild"
] = 0 # Mio EUR per unit
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "DW_Electric_converter_2", "2040"], "useAnnuity"
] = 0  # binary yes/no
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "DW_Electric_converter_2", "2040"], "amorTime"
] = 0  # years
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "DW_Electric_converter_2", "2040"], "interest"
] = 0  # percent/100
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "DW_Electric_converter_2", "2040"], "perUnitTotal"
] = 0 


accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "ST_N_DW", "2040"], "perUnitBuild"
] = 0 # Mio EUR per unit
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "ST_N_DW", "2040"], "useAnnuity"
] = 0  # binary yes/no
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "ST_N_DW", "2040"], "amorTime"
] = 0  # years
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "ST_N_DW", "2040"], "interest"
] = 0  # percent/100
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "ST_N_DW", "2040"], "perUnitTotal"
] = 0 

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Dummy_Ammonia", "2040"], "perUnitBuild"
] = 0 # Mio EUR per unit
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Dummy_Ammonia", "2040"], "useAnnuity"
] = 0  # binary yes/no
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Dummy_Ammonia", "2040"], "amorTime"
] = 0  # years
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Dummy_Ammonia", "2040"], "interest"
] = 0  # percent/100
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "Dummy_Ammonia", "2040"], "perUnitTotal"
] = 0



accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Industry_EL", "2040"], "perUnitBuild"
] = 140 # Mio EUR per unit
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Industry_EL", "2040"], "useAnnuity"
] = 1  # binary yes/no
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Industry_EL", "2040"], "amorTime"
] = 20  # years
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Industry_EL", "2040"], "interest"
] = 0.06  # percent/100
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "Industry_EL", "2040"], "perUnitTotal"
] = 0.5


accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Dummy_Methanol", "2040"], "perUnitBuild"
] = 0  # Mio EUR per unit
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Dummy_Methanol", "2040"], "useAnnuity"
] = 0  # binary yes/no
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Dummy_Methanol", "2040"], "amorTime"
] = 0  # years
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Dummy_Methanol", "2040"], "interest"
] = 0  # percent/100
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "Dummy_Methanol", "2040"], "perUnitTotal"
] = 0

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "BG_N", "2040"], "perUnitBuild"
] = 2600.0  # Mio EUR per unit
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "BG_N", "2040"], "useAnnuity"
] = 1  # binary yes/no
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "BG_N", "2040"], "amorTime"
] = 25  # years
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "BG_N", "2040"], "interest"
] = 0.06  # percent/100
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "BG_N", "2040"], "perUnitTotal"
] = 78


accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "PV_N", "2040"], "perUnitBuild"
] = 240
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "PV_N", "2040"], "useAnnuity"
] = 1
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "PV_N", "2040"], "amorTime"
] = 35
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "PV_N", "2040"], "interest"
] = 0.06
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "PV_N", "2040"], "perUnitTotal"
] = 5.1

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "WindOnshore_N", "2040"], "perUnitBuild"
] = 1080
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "WindOnshore_N", "2040"], "useAnnuity"
] = 1
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "WindOnshore_N", "2040"], "amorTime"
] = 25
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "WindOnshore_N", "2040"], "interest"
] = 0.06
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "WindOnshore_N", "2040"], "perUnitTotal"
] = 21

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Wave_N", "2040"], "perUnitBuild"
] = 2300
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Wave_N", "2040"], "useAnnuity"
] = 1
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Wave_N", "2040"], "amorTime"
] = 30
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Wave_N", "2040"], "interest"
] = 0.06
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "Wave_N", "2040"], "perUnitTotal"
] = 58

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "WindOffshore_N", "2040"], "perUnitBuild"
] = 2520
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "WindOffshore_N", "2040"], "useAnnuity"
] = 1
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "WindOffshore_N", "2040"], "amorTime"
] = 25
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "WindOffshore_N", "2040"], "interest"
] = 0.06
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "WindOffshore_N", "2040"], "perUnitTotal"
] = 72

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Hydro_N", "2040"], "perUnitBuild"
] = 12400 * 2.22
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Hydro_N", "2040"], "useAnnuity"
] = 1
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Hydro_N", "2040"], "amorTime"
] = 50
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Hydro_N", "2040"], "interest"
] = 0.06
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "Hydro_N", "2040"], "perUnitTotal"
] = 2.22 * 490


accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "ST_N", "2040"], "perUnitBuild"
] = 530
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "ST_N", "2040"], "useAnnuity"
] = 1
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "ST_N", "2040"], "amorTime"
] = 30
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "ST_N", "2040"], "interest"
] = 0.06
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "ST_N", "2040"], "perUnitTotal"
] = 1.5

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "LDV_BF", "2040"], "perUnitBuild"
] = 448
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "LDV_BF", "2040"], "useAnnuity"
] = 1
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "LDV_BF", "2040"], "amorTime"
] = 30
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "LDV_BF", "2040"], "interest"
] = 0.06
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "LDV_BF", "2040"], "perUnitTotal"
] = 40

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "RO", "2040"], "perUnitBuild"
] = 28
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "RO", "2040"], "useAnnuity"
] = 1
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "RO", "2040"], "amorTime"
] = 30
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "RO", "2040"], "interest"
] = 0.06
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "RO", "2040"], "perUnitTotal"
] = 1.12

# accounting_converterUnits.loc[
#     idx["Invest", "global", "horizon", "AEL", "2040"], "perUnitBuild"
# ] = 400
# accounting_converterUnits.loc[
#     idx["Invest", "global", "horizon", "AEL", "2040"], "useAnnuity"
# ] = 1
# accounting_converterUnits.loc[
#     idx["Invest", "global", "horizon", "AEL", "2040"], "amorTime"
# ] = 25
# accounting_converterUnits.loc[
#     idx["Invest", "global", "horizon", "AEL", "2040"], "interest"
# ] = 0.06
# accounting_converterUnits.loc[
#     idx["OMFix", "global", "horizon", "AEL", "2040"], "perUnitTotal"
# ] = 0

# accounting_converterUnits.loc[
#     idx["Invest", "global", "horizon", "AEL_10", "2040"], "perUnitBuild"
# ] = 200
# accounting_converterUnits.loc[
#     idx["Invest", "global", "horizon", "AEL_10", "2040"], "useAnnuity"
# ] = 1
# accounting_converterUnits.loc[
#     idx["Invest", "global", "horizon", "AEL_10", "2040"], "amorTime"
# ] = 25
# accounting_converterUnits.loc[
#     idx["Invest", "global", "horizon", "AEL_10", "2040"], "interest"
# ] = 0.06
# accounting_converterUnits.loc[
#     idx["OMFix", "global", "horizon", "AEL_10", "2040"], "perUnitTotal"
# ] = 0

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "AEL_100", "2040"], "perUnitBuild"
] = 300
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "AEL_100", "2040"], "useAnnuity"
] = 1
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "AEL_100", "2040"], "amorTime"
] = 25
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "AEL_100", "2040"], "interest"
] = 0.06
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "AEL_100", "2040"], "perUnitTotal"
] = 14.08


accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Ammonia_synthesis", "2040"], "perUnitBuild"
] = 1348
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Ammonia_synthesis", "2040"], "useAnnuity"
] = 1
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Ammonia_synthesis", "2040"], "amorTime"
] = 30
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Ammonia_synthesis", "2040"], "interest"
] = 0.06
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "Ammonia_synthesis", "2040"], "perUnitTotal"
] = 64.3

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "DAC", "2040"], "perUnitBuild"
] = 2418
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "DAC", "2040"], "useAnnuity"
] = 1
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "DAC", "2040"], "amorTime"
] = 30
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "DAC", "2040"], "interest"
] = 0.06
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "DAC", "2040"], "perUnitTotal"
] = 97

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Methanol_synthesis", "2040"], "perUnitBuild"
] = 971
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Methanol_synthesis", "2040"], "useAnnuity"
] = 1
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Methanol_synthesis", "2040"], "amorTime"
] = 30
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Methanol_synthesis", "2040"], "interest"
] = 0.06
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "Methanol_synthesis", "2040"], "perUnitTotal"
] = 39

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "HP", "2040"], "perUnitBuild"
] = 650
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "HP", "2040"], "useAnnuity"
] = 1
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "HP", "2040"], "amorTime"
] = 30
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "HP", "2040"], "interest"
] = 0.06
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "HP", "2040"], "perUnitTotal"
] = 2.8

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "FTL", "2040"], "perUnitBuild"
] = 1065
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "FTL", "2040"], "useAnnuity"
] = 1
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "FTL", "2040"], "amorTime"
] = 30
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "FTL", "2040"], "interest"
] = 0.06
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "FTL", "2040"], "perUnitTotal"
] = 31

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "LDV_el", "2040"], "perUnitBuild"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "LDV_el", "2040"], "useAnnuity"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "LDV_el", "2040"], "amorTime"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "LDV_el", "2040"], "interest"
] = 0
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "LDV_el", "2040"], "perUnitTotal"
] = 0

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "HDV_el", "2040"], "perUnitBuild"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "HDV_el", "2040"], "useAnnuity"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "HDV_el", "2040"], "amorTime"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "HDV_el", "2040"], "interest"
] = 0
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "HDV_el", "2040"], "perUnitTotal"
] = 0

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "HDV_BF", "2040"], "perUnitBuild"
] = 448
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "HDV_BF", "2040"], "useAnnuity"
] = 1
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "HDV_BF", "2040"], "amorTime"
] = 30
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "HDV_BF", "2040"], "interest"
] = 0.06
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "HDV_BF", "2040"], "perUnitTotal"
] = 40

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "MDV_el", "2040"], "perUnitBuild"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "MDV_el", "2040"], "useAnnuity"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "MDV_el", "2040"], "amorTime"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "MDV_el", "2040"], "interest"
] = 0
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "MDV_el", "2040"], "perUnitTotal"
] = 0

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "MDV_BF", "2040"], "perUnitBuild"
] = 448
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "MDV_BF", "2040"], "useAnnuity"
] = 1
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "MDV_BF", "2040"], "amorTime"
] = 30
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "MDV_BF", "2040"], "interest"
] = 0.06
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "MDV_BF", "2040"], "perUnitTotal"
] = 40

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Two_wheel_el", "2040"], "perUnitBuild"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Two_wheel_el", "2040"], "useAnnuity"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Two_wheel_el", "2040"], "amorTime"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Two_wheel_el", "2040"], "interest"
] = 0
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "Two_wheel_el", "2040"], "perUnitTotal"
] = 0

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Bus_el", "2040"], "perUnitBuild"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Bus_el", "2040"], "useAnnuity"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Bus_el", "2040"], "amorTime"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Bus_el", "2040"], "interest"
] = 0
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "Bus_el", "2040"], "perUnitTotal"
] = 0

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Marine_e", "2040"], "perUnitBuild"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Marine_e", "2040"], "useAnnuity"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Marine_e", "2040"], "amorTime"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Marine_e", "2040"], "interest"
] = 0
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "Marine_e", "2040"], "perUnitTotal"
] = 0

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Aviation_el", "2040"], "perUnitBuild"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Aviation_el", "2040"], "useAnnuity"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Aviation_el", "2040"], "amorTime"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Aviation_el", "2040"], "interest"
] = 0
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "Aviation_el", "2040"], "perUnitTotal"
] = 0

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Aviation_e", "2040"], "perUnitBuild"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Aviation_e", "2040"], "useAnnuity"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Aviation_e", "2040"], "amorTime"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Aviation_e", "2040"], "interest"
] = 0
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "Aviation_e", "2040"], "perUnitTotal"
] = 0

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "cook_el", "2040"], "perUnitBuild"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "cook_el", "2040"], "useAnnuity"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "cook_el", "2040"], "amorTime"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "cook_el", "2040"], "interest"
] = 0
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "cook_el", "2040"], "perUnitTotal"
] = 0

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "cook_LPG", "2040"], "perUnitBuild"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "cook_LPG", "2040"], "useAnnuity"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "cook_LPG", "2040"], "amorTime"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "cook_LPG", "2040"], "interest"
] = 0
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "cook_LPG", "2040"], "perUnitTotal"
] = 0

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Industry_EH", "2040"], "perUnitBuild"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Industry_EH", "2040"], "useAnnuity"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Industry_EH", "2040"], "amorTime"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Industry_EH", "2040"], "interest"
] = 0
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "Industry_EH", "2040"], "perUnitTotal"
] = 0

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "DW_heat", "2040"], "perUnitBuild"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "DW_heat", "2040"], "useAnnuity"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "DW_heat", "2040"], "amorTime"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "DW_heat", "2040"], "interest"
] = 0
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "DW_heat", "2040"], "perUnitTotal"
] = 0

accounting_converterUnits = accounting_converterUnits.fillna(0)

m.parameter.add(accounting_converterUnits, "accounting_converterunits")
accounting_converterUnits
###########################################################################
accounting_converteractivity = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [["OMVar"], ["global"], ["horizon"], ["AEL_100"], ["2040"], ['Powergen']]
 )
).sort_index()

accounting_converteractivity.loc[
    idx["OMVar", "global", "horizon", "AEL_100", "2040", "Powergen"], "perActivity"
] = 0.0016

accounting_converteractivity = accounting_converteractivity.fillna(0)

m.parameter.add(accounting_converteractivity, "accounting_converteractivity")
accounting_converteractivity

accounting_converteractivity = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [["OMVar"], ["global"], ["horizon"], ["Industry_EL"], ["2040"], ['Powergen']]
 )
).sort_index()

accounting_converteractivity.loc[
    idx["OMVar", "global", "horizon", "Industry_EL", "2040", "Powergen"], "perActivity"
] = 0.002

accounting_converteractivity = accounting_converteractivity.fillna(0)

m.parameter.add(accounting_converteractivity, "accounting_converteractivity")
accounting_converteractivity
#########################################################################
accounting_converterUnits = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [
            ["Invest", "OMFix"],
            ["global"],
            ["horizon"],
            ["PV_N", "WindOnshore_N", "Wave_N", "WindOffshore_N", "BG_N", "Hydro_N", "ST_N", "LDV_BF", "RO","AEL_100","Ammonia_synthesis", "DAC", "Methanol_synthesis", "HP", "FTL", "LDV_el", "HDV_el", "HDV_BF", "MDV_el", "MDV_BF", "Two_wheel_el", "Bus_el", "Marine_e", "Aviation_el", "Aviation_e", "cook_el", "cook_LPG", "Industry_EH", "DW_heat", "Dummy_Ammonia", "Dummy_Methanol", "ST_N_DW", "Industry_EL"],
            ['2050'],
        ]
    )
).sort_index()

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "ST_N_DW", "2050"], "perUnitBuild"
] = 0  # Mio EUR per unit
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "ST_N_DW", "2050"], "useAnnuity"
] = 0  # binary yes/no
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "ST_N_DW", "2050"], "amorTime"
] = 0  # years
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "ST_N_DW", "2050"], "interest"
] = 0  # percent/100
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "ST_N_DW", "2050"], "perUnitTotal"
] = 0

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "BG_N", "2050"], "perUnitBuild"
] = 2600.0  # Mio EUR per unit
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "BG_N", "2050"], "useAnnuity"
] = 1  # binary yes/no
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "BG_N", "2050"], "amorTime"
] = 25  # years
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "BG_N", "2050"], "interest"
] = 0.06  # percent/100
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "BG_N", "2050"], "perUnitTotal"
] = 78

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Industry_EL", "2050"], "perUnitBuild"
] = 140 # Mio EUR per unit
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Industry_EL", "2050"], "useAnnuity"
] = 1  # binary yes/no
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Industry_EL", "2050"], "amorTime"
] = 20  # years
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Industry_EL", "2050"], "interest"
] = 0.06  # percent/100
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "Industry_EL", "2050"], "perUnitTotal"
] = 0.5


accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Dummy_Ammonia", "2050"], "perUnitBuild"
] = 0  # Mio EUR per unit
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Dummy_Ammonia", "2050"], "useAnnuity"
] = 0  # binary yes/no
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Dummy_Ammonia", "2050"], "amorTime"
] = 0  # years
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Dummy_Ammonia", "2050"], "interest"
] = 0  # percent/100
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "Dummy_Ammonia", "2050"], "perUnitTotal"
] = 0

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Dummy_Methanol", "2050"], "perUnitBuild"
] = 0  # Mio EUR per unit
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Dummy_Methanol", "2050"], "useAnnuity"
] = 0  # binary yes/no
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Dummy_Methanol", "2050"], "amorTime"
] = 0  # years
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Dummy_Methanol", "2050"], "interest"
] = 0  # percent/100
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "Dummy_Methanol", "2050"], "perUnitTotal"
] = 0

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "PV_N", "2050"], "perUnitBuild"
] = 213
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "PV_N", "2050"], "useAnnuity"
] = 1
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "PV_N", "2050"], "amorTime"
] = 35
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "PV_N", "2050"], "interest"
] = 0.06
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "PV_N", "2050"], "perUnitTotal"
] = 4.7

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "WindOnshore_N", "2050"], "perUnitBuild"
] = 1040
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "WindOnshore_N", "2050"], "useAnnuity"
] = 1
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "WindOnshore_N", "2050"], "amorTime"
] = 25
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "WindOnshore_N", "2050"], "interest"
] = 0.06
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "WindOnshore_N", "2050"], "perUnitTotal"
] = 21

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Wave_N", "2050"], "perUnitBuild"
] = 2015
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Wave_N", "2050"], "useAnnuity"
] = 1
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Wave_N", "2050"], "amorTime"
] = 30
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Wave_N", "2050"], "interest"
] = 0.06
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "Wave_N", "2050"], "perUnitTotal"
] = 49

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "WindOffshore_N", "2050"], "perUnitBuild"
] = 2480
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "WindOffshore_N", "2050"], "useAnnuity"
] = 1
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "WindOffshore_N", "2050"], "amorTime"
] = 25
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "WindOffshore_N", "2050"], "interest"
] = 0.06
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "WindOffshore_N", "2050"], "perUnitTotal"
] = 70

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Hydro_N", "2050"], "perUnitBuild"
] = 12400 * 2.22
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Hydro_N", "2050"], "useAnnuity"
] = 1
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Hydro_N", "2050"], "amorTime"
] = 50
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Hydro_N", "2050"], "interest"
] = 0.06
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "Hydro_N", "2050"], "perUnitTotal"
] = 2.22 * 490


accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "ST_N", "2050"], "perUnitBuild"
] = 506
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "ST_N", "2050"], "useAnnuity"
] = 1
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "ST_N", "2050"], "amorTime"
] = 30
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "ST_N", "2050"], "interest"
] = 0.06
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "ST_N", "2050"], "perUnitTotal"
] = 1.5

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "LDV_BF", "2050"], "perUnitBuild"
] = 448
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "LDV_BF", "2050"], "useAnnuity"
] = 1
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "LDV_BF", "2050"], "amorTime"
] = 30
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "LDV_BF", "2050"], "interest"
] = 0.06
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "LDV_BF", "2050"], "perUnitTotal"
] = 40

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "RO", "2050"], "perUnitBuild"
] = 11.9
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "RO", "2050"], "useAnnuity"
] = 1
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "RO", "2050"], "amorTime"
] = 25
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "RO", "2050"], "interest"
] = 0.06
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "RO", "2050"], "perUnitTotal"
] = 0.48

# accounting_converterUnits.loc[
#     idx["Invest", "global", "horizon", "AEL", "2040"], "perUnitBuild"
# ] = 400
# accounting_converterUnits.loc[
#     idx["Invest", "global", "horizon", "AEL", "2040"], "useAnnuity"
# ] = 1
# accounting_converterUnits.loc[
#     idx["Invest", "global", "horizon", "AEL", "2040"], "amorTime"
# ] = 25
# accounting_converterUnits.loc[
#     idx["Invest", "global", "horizon", "AEL", "2040"], "interest"
# ] = 0.06
# accounting_converterUnits.loc[
#     idx["OMFix", "global", "horizon", "AEL", "2040"], "perUnitTotal"
# ] = 0

# accounting_converterUnits.loc[
#     idx["Invest", "global", "horizon", "AEL_10", "2040"], "perUnitBuild"
# ] = 200
# accounting_converterUnits.loc[
#     idx["Invest", "global", "horizon", "AEL_10", "2040"], "useAnnuity"
# ] = 1
# accounting_converterUnits.loc[
#     idx["Invest", "global", "horizon", "AEL_10", "2040"], "amorTime"
# ] = 25
# accounting_converterUnits.loc[
#     idx["Invest", "global", "horizon", "AEL_10", "2040"], "interest"
# ] = 0.06
# accounting_converterUnits.loc[
#     idx["OMFix", "global", "horizon", "AEL_10", "2040"], "perUnitTotal"
# ] = 0

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "AEL_100", "2050"], "perUnitBuild"
] = 200
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "AEL_100", "2050"], "useAnnuity"
] = 1
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "AEL_100", "2050"], "amorTime"
] = 30
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "AEL_100", "2050"], "interest"
] = 0.06
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "AEL_100", "2050"], "perUnitTotal"
] = 11.9


accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Ammonia_synthesis", "2050"], "perUnitBuild"
] = 1348
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Ammonia_synthesis", "2050"], "useAnnuity"
] = 1
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Ammonia_synthesis", "2050"], "amorTime"
] = 30
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Ammonia_synthesis", "2050"], "interest"
] = 0.06
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "Ammonia_synthesis", "2050"], "perUnitTotal"
] = 64.3

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "DAC", "2050"], "perUnitBuild"
] = 2024
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "DAC", "2050"], "useAnnuity"
] = 1
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "DAC", "2050"], "amorTime"
] = 30
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "DAC", "2050"], "interest"
] = 0.06
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "DAC", "2050"], "perUnitTotal"
] = 81.4

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Methanol_synthesis", "2050"], "perUnitBuild"
] = 971
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Methanol_synthesis", "2050"], "useAnnuity"
] = 1
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Methanol_synthesis", "2050"], "amorTime"
] = 30
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Methanol_synthesis", "2050"], "interest"
] = 0.06
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "Methanol_synthesis", "2050"], "perUnitTotal"
] = 39

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "HP", "2050"], "perUnitBuild"
] = 630
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "HP", "2050"], "useAnnuity"
] = 1
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "HP", "2050"], "amorTime"
] = 30
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "HP", "2050"], "interest"
] = 0.06
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "HP", "2050"], "perUnitTotal"
] = 2.6

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "FTL", "2050"], "perUnitBuild"
] = 1065
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "FTL", "2050"], "useAnnuity"
] = 1
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "FTL", "2050"], "amorTime"
] = 30
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "FTL", "2050"], "interest"
] = 0.06
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "FTL", "2050"], "perUnitTotal"
] = 31

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "LDV_el", "2050"], "perUnitBuild"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "LDV_el", "2050"], "useAnnuity"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "LDV_el", "2050"], "amorTime"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "LDV_el", "2050"], "interest"
] = 0
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "LDV_el", "2050"], "perUnitTotal"
] = 0

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "HDV_el", "2050"], "perUnitBuild"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "HDV_el", "2050"], "useAnnuity"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "HDV_el", "2050"], "amorTime"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "HDV_el", "2050"], "interest"
] = 0
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "HDV_el", "2050"], "perUnitTotal"
] = 0

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "HDV_BF", "2050"], "perUnitBuild"
] = 448
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "HDV_BF", "2050"], "useAnnuity"
] = 1
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "HDV_BF", "2050"], "amorTime"
] = 30
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "HDV_BF", "2050"], "interest"
] = 0.06
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "HDV_BF", "2050"], "perUnitTotal"
] = 40

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "MDV_el", "2050"], "perUnitBuild"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "MDV_el", "2050"], "useAnnuity"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "MDV_el", "2050"], "amorTime"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "MDV_el", "2050"], "interest"
] = 0
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "MDV_el", "2050"], "perUnitTotal"
] = 0

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "MDV_BF", "2050"], "perUnitBuild"
] = 448
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "MDV_BF", "2050"], "useAnnuity"
] = 1
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "MDV_BF", "2050"], "amorTime"
] = 30
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "MDV_BF", "2050"], "interest"
] = 0.06
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "MDV_BF", "2050"], "perUnitTotal"
] = 40

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Two_wheel_el", "2050"], "perUnitBuild"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Two_wheel_el", "2050"], "useAnnuity"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Two_wheel_el", "2050"], "amorTime"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Two_wheel_el", "2050"], "interest"
] = 0
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "Two_wheel_el", "2050"], "perUnitTotal"
] = 0

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Bus_el", "2050"], "perUnitBuild"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Bus_el", "2050"], "useAnnuity"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Bus_el", "2050"], "amorTime"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Bus_el", "2050"], "interest"
] = 0
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "Bus_el", "2050"], "perUnitTotal"
] = 0

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Marine_e", "2050"], "perUnitBuild"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Marine_e", "2050"], "useAnnuity"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Marine_e", "2050"], "amorTime"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Marine_e", "2050"], "interest"
] = 0
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "Marine_e", "2050"], "perUnitTotal"
] = 0

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Aviation_el", "2050"], "perUnitBuild"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Aviation_el", "2050"], "useAnnuity"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Aviation_el", "2050"], "amorTime"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Aviation_el", "2050"], "interest"
] = 0
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "Aviation_el", "2050"], "perUnitTotal"
] = 0

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Aviation_e", "2050"], "perUnitBuild"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Aviation_e", "2050"], "useAnnuity"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Aviation_e", "2050"], "amorTime"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Aviation_e", "2050"], "interest"
] = 0
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "Aviation_e", "2050"], "perUnitTotal"
] = 0

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "cook_el", "2050"], "perUnitBuild"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "cook_el", "2050"], "useAnnuity"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "cook_el", "2050"], "amorTime"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "cook_el", "2050"], "interest"
] = 0
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "cook_el", "2050"], "perUnitTotal"
] = 0

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "cook_LPG", "2050"], "perUnitBuild"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "cook_LPG", "2050"], "useAnnuity"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "cook_LPG", "2050"], "amorTime"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "cook_LPG", "2050"], "interest"
] = 0
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "cook_LPG", "2050"], "perUnitTotal"
] = 0

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Industry_EH", "2050"], "perUnitBuild"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Industry_EH", "2050"], "useAnnuity"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Industry_EH", "2050"], "amorTime"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Industry_EH", "2050"], "interest"
] = 0
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "Industry_EH", "2050"], "perUnitTotal"
] = 0

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "DW_heat", "2050"], "perUnitBuild"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "DW_heat", "2050"], "useAnnuity"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "DW_heat", "2050"], "amorTime"
] = 0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "DW_heat", "2050"], "interest"
] = 0
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "DW_heat", "2050"], "perUnitTotal"
] = 0


accounting_converterUnits = accounting_converterUnits.fillna(0)

m.parameter.add(accounting_converterUnits, "accounting_converterunits")
accounting_converterUnits

##########################################################################
accounting_converteractivity = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [["OMVar"], ["global"], ["horizon"], ["AEL_100"], ["2050"], ['Powergen']]
 )
).sort_index()

accounting_converteractivity.loc[
    idx["OMVar", "global", "horizon", "AEL_100", "2050", "Powergen"], "perActivity"
] = 0.0016

accounting_converteractivity = accounting_converteractivity.fillna(0)

m.parameter.add(accounting_converteractivity, "accounting_converteractivity")
accounting_converteractivity

accounting_converteractivity = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [["OMVar"], ["global"], ["horizon"], ["Industry_EL"], ["2050"], ['Powergen']]
 )
).sort_index()

accounting_converteractivity.loc[
    idx["OMVar", "global", "horizon", "Industry_EL", "2050", "Powergen"], "perActivity"
] = 0.002

accounting_converteractivity = accounting_converteractivity.fillna(0)

m.parameter.add(accounting_converteractivity, "accounting_converteractivity")
accounting_converteractivity
###########################################################################
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
###DEmand data#############################################################
# "sourcesink_profile"
demand_R4_R2_CH = profiles[["demand_CI_2020", "demand_FJ_2020","demand_FSM_2020", "demand_KB_2020", "demand_MI_2020","demand_NU_2020","demand_NE_2020","demand_PU_2020","demand_PNG_2020","demand_SA_2020","demand_SI_2020","demand_TA_2020","demand_TU_2020","demand_VU_2020"]]

demand_R4_R2_CH = demand_R4_R2_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R2_CH = demand_R4_R2_CH.T

demand_R4_R2_CH = demand_R4_R2_CH.rename(
    index={"demand_CI_2020": "CI_data", "demand_FJ_2020": "FJ_data", "demand_FSM_2020": "FSM_data", "demand_KB_2020": "KB_data", "demand_MI_2020": "MI_data","demand_NU_2020": "NU_data","demand_NE_2020": "NE_data","demand_PU_2020": "PU_data","demand_PNG_2020": "PNG_data","demand_SA_2020": "SA_data","demand_SI_2020": "SI_data","demand_TA_2020": "TA_data","demand_TU_2020": "TU_data","demand_VU_2020": "VU_data"}
)

# add columns and set them as index
demand_R4_R2_CH["years"] = "2020"
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

##############################################################################
demand_R4_R3_CH = profiles[["MDV_CI","MDV_FJ","MDV_FSM","MDV_KB","MDV_MI","MDV_NU","MDV_NE","MDV_PU","MDV_PNG","MDV_SA","MDV_SI","MDV_TA","MDV_TU","MDV_VU"]]

demand_R4_R3_CH = demand_R4_R3_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R3_CH = demand_R4_R3_CH.T

demand_R4_R3_CH = demand_R4_R3_CH.rename(
    index={"MDV_CI": "CI_data", "MDV_FJ": "FJ_data", "MDV_FSM": "FSM_data", "MDV_KB": "KB_data", "MDV_MI": "MI_data","MDV_NU": "NU_data","MDV_NE": "NE_data","MDV_PU": "PU_data","MDV_PNG": "PNG_data","MDV_SA": "SA_data","MDV_SI": "SI_data","MDV_TA": "TA_data","MDV_TU": "TU_data","MDV_VU": "VU_data"}
)

# add columns and set them as index
demand_R4_R3_CH["years"] = "2020"
demand_R4_R3_CH["techs"] = "Demand"
demand_R4_R3_CH["commodity"] = "T_MDV_th"
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
        [m.set.nodesdata, m.set.yearssel, ["Demand"], ["T_MDV_th"]]
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
demand_R4_R4_CH["years"] = "2020"
demand_R4_R4_CH["techs"] = "Demand"
demand_R4_R4_CH["commodity"] = "T_HDV_th"
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
        [m.set.nodesdata, m.set.yearssel, ["Demand"], ["T_HDV_th"]]
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
demand_R4_R5_CH["years"] = "2020"
demand_R4_R5_CH["techs"] = "Demand"
demand_R4_R5_CH["commodity"] = "T_LDV_th"
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
        [m.set.nodesdata, m.set.yearssel, ["Demand"], ["T_LDV_th"]]
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
demand_R4_R6_CH["years"] = "2020"
demand_R4_R6_CH["techs"] = "Demand"
demand_R4_R6_CH["commodity"] = "T_Bus_th"
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
        [m.set.nodesdata, m.set.yearssel, ["Demand"], ["T_Bus_th"]]
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
demand_R4_R7_CH["years"] = "2020"
demand_R4_R7_CH["techs"] = "Demand"
demand_R4_R7_CH["commodity"] = "T_Two_wheel_th"
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
        [m.set.nodesdata, m.set.yearssel, ["Demand"], ["T_Two_wheel_th"]]
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
demand_R4_R8_CH["years"] = "2020"
demand_R4_R8_CH["techs"] = "Demand"
demand_R4_R8_CH["commodity"] = "T_Marine_th"
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
        [m.set.nodesdata, m.set.yearssel, ["Demand"], ["T_Marine_th"]]
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
demand_R4_R9_CH["years"] = "2020"
demand_R4_R9_CH["techs"] = "Demand"
demand_R4_R9_CH["commodity"] = "T_Aviation_th"
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
        [m.set.nodesdata, m.set.yearssel, ["Demand"], ["T_Aviation_th"]]
    )
)
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesFixedProfile"] = 1
sourcesink_config = sourcesink_config.dropna()

m.parameter.add(sourcesink_config, "sourcesink_config")
sourcesink_config
#################################################################################
demand_R4_R9_CH = profiles[["Marinef_CI", "Marinef_FJ","Marinef_FSM", "Marinef_KB", "Marinef_MI","Marinef_NU","Marinef_NE","Marinef_PU","Marinef_PNG","Marinef_SA","Marinef_SI","Marinef_TA","Marinef_TU","Marinef_VU"]]

demand_R4_R9_CH = demand_R4_R9_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R9_CH = demand_R4_R9_CH.T

demand_R4_R9_CH = demand_R4_R9_CH.rename(
    index={"Marinef_CI": "CI_data", "Marinef_FJ": "FJ_data", "Marinef_FSM": "FSM_data", "Marinef_KB": "KB_data", "Marinef_MI": "MI_data","Marinef_NU": "NU_data","Marinef_NE": "NE_data","Marinef_PU": "PU_data","Marinef_PNG": "PNG_data","Marinef_SA": "SA_data","Marinef_SI": "SI_data","Marinef_TA": "TA_data","Marinef_TU": "TU_data","Marinef_VU": "VU_data"}
)

# add columns and set them as index
demand_R4_R9_CH["years"] = "2020"
demand_R4_R9_CH["techs"] = "Demand"
demand_R4_R9_CH["commodity"] = "T_Marine_f_th"
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
        [m.set.nodesdata, m.set.yearssel, ["Demand"], ["T_Marine_f_th"]]
    )
)
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesFixedProfile"] = 1
sourcesink_config = sourcesink_config.dropna()

m.parameter.add(sourcesink_config, "sourcesink_config")
sourcesink_config
#################################################################################
demand_R4_R10_CH = profiles[["HC_CI", "HC_FJ","HC_FSM", "HC_KB", "HC_MI","HC_NU","HC_NE","HC_PU","HC_PNG","HC_SA","HC_SI","HC_TA","HC_TU","HC_VU"]]

demand_R4_R10_CH = demand_R4_R10_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R10_CH = demand_R4_R10_CH.T

demand_R4_R10_CH = demand_R4_R10_CH.rename(
    index={"HC_CI": "CI_data", "HC_FJ": "FJ_data", "HC_FSM": "FSM_data", "HC_KB": "KB_data", "HC_MI": "MI_data","HC_NU": "NU_data","HC_NE": "NE_data","HC_PU": "PU_data","HC_PNG": "PNG_data","HC_SA": "SA_data","HC_SI": "SI_data","HC_TA": "TA_data","HC_TU": "TU_data","HC_VU": "VU_data"}
)

# add columns and set them as index
demand_R4_R10_CH["years"] = "2020"
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
demand_R4_R10_CH["years"] = "2020"
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
demand_R4_R10_CH["years"] = "2020"
demand_R4_R10_CH["techs"] = "Demand"
demand_R4_R10_CH["commodity"] = "DHW_el"
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
        [m.set.nodesdata, m.set.yearssel, ["Demand"], ["DHW_el"]]
    )
)
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesFixedProfile"] = 1
sourcesink_config = sourcesink_config.dropna()

m.parameter.add(sourcesink_config, "sourcesink_config")
sourcesink_config
#############################################################

###########################################################
demand_R4_R10_CH = profiles[["DHWL_CI", "DHWL_FJ","DHWL_FSM", "DHWL_KB", "DHWL_MI","DHWL_NU","DHWL_NE","DHWL_PU","DHWL_PNG","DHWL_SA","DHWL_SI","DHWL_TA","DHWL_TU","DHWL_VU"]]

demand_R4_R10_CH = demand_R4_R10_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R10_CH = demand_R4_R10_CH.T

demand_R4_R10_CH = demand_R4_R10_CH.rename(
    index={"DHWL_CI": "CI_data", "DHWL_FJ": "FJ_data", "DHWL_FSM": "FSM_data", "DHWL_KB": "KB_data", "DHWL_MI": "MI_data","DHWL_NU": "NU_data","DHWL_NE": "NE_data","DHWL_PU": "PU_data","DHWL_PNG": "PNG_data","DHWL_SA": "SA_data","DHWL_SI": "SI_data","DHWL_TA": "TA_data","DHWL_TU": "TU_data","DHWL_VU": "VU_data"}
)

# add columns and set them as index
demand_R4_R10_CH["years"] = "2020"
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
##################################2030##########################################
demand_R4_R2_CH = profiles[["demand_CI_2030", "demand_FJ_2030","demand_FSM_2030", "demand_KB_2030", "demand_MI_2030","demand_NU_2030","demand_NE_2030","demand_PU_2030","demand_PNG_2030","demand_SA_2030","demand_SI_2030","demand_TA_2030","demand_TU_2030","demand_VU_2030"]]

demand_R4_R2_CH = demand_R4_R2_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R2_CH = demand_R4_R2_CH.T

demand_R4_R2_CH = demand_R4_R2_CH.rename(
    index={"demand_CI_2030": "CI_data", "demand_FJ_2030": "FJ_data", "demand_FSM_2030": "FSM_data", "demand_KB_2030": "KB_data", "demand_MI_2030": "MI_data","demand_NU_2030": "NU_data","demand_NE_2030": "NE_data","demand_PU_2030": "PU_data","demand_PNG_2030": "PNG_data","demand_SA_2030": "SA_data","demand_SI_2030": "SI_data","demand_TA_2030": "TA_data","demand_TU_2030": "TU_data","demand_VU_2030": "VU_data"}
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

#######################################################################################################################

########################################################################################

#############################################################################################
demand_R4_R3_CH = profiles[["MDV_CI_2030","MDV_FJ_2030","MDV_FSM_2030","MDV_KB_2030","MDV_MI_2030","MDV_NU_2030","MDV_NE_2030","MDV_PU_2030","MDV_PNG_2030","MDV_SA_2030","MDV_SI_2030","MDV_TA_2030","MDV_TU_2030","MDV_VU_2030"]]

demand_R4_R3_CH = demand_R4_R3_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R3_CH = demand_R4_R3_CH.T

demand_R4_R3_CH = demand_R4_R3_CH.rename(
    index={"MDV_CI_2030": "CI_data", "MDV_FJ_2030": "FJ_data", "MDV_FSM_2030": "FSM_data", "MDV_KB_2030": "KB_data", "MDV_MI_2030": "MI_data","MDV_NU_2030": "NU_data","MDV_NE_2030": "NE_data","MDV_PU_2030": "PU_data","MDV_PNG_2030": "PNG_data","MDV_SA_2030": "SA_data","MDV_SI_2030": "SI_data","MDV_TA_2030": "TA_data","MDV_TU_2030": "TU_data","MDV_VU_2030": "VU_data"}
)

# add columns and set them as index
demand_R4_R3_CH["years"] = "2030"
demand_R4_R3_CH["techs"] = "Demand"
demand_R4_R3_CH["commodity"] = "T_MDV_th"
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
        [m.set.nodesdata, m.set.yearssel, ["Demand"], ["T_MDV_th"]]
    )
)
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesFixedProfile"] = 1
sourcesink_config = sourcesink_config.dropna()

m.parameter.add(sourcesink_config, "sourcesink_config")
sourcesink_config
########################################################

demand_R4_R4_CH = profiles[["HDV_CI_2030", "HDV_FJ_2030","HDV_FSM_2030", "HDV_KB_2030", "HDV_MI_2030","HDV_NU_2030","HDV_NE_2030","HDV_PU_2030","HDV_PNG_2030","HDV_SA_2030","HDV_SI_2030","HDV_TA_2030","HDV_TU_2030","HDV_VU_2030"]]

demand_R4_R4_CH = demand_R4_R4_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R4_CH = demand_R4_R4_CH.T

demand_R4_R4_CH = demand_R4_R4_CH.rename(
    index={"HDV_CI_2030": "CI_data", "HDV_FJ_2030": "FJ_data", "HDV_FSM_2030": "FSM_data", "HDV_KB_2030": "KB_data", "HDV_MI_2030": "MI_data","HDV_NU_2030": "NU_data","HDV_NE_2030": "NE_data","HDV_PU_2030": "PU_data","HDV_PNG_2030": "PNG_data","HDV_SA_2030": "SA_data","HDV_SI_2030": "SI_data","HDV_TA_2030": "TA_data","HDV_TU_2030": "TU_data","HDV_VU_2030": "VU_data"}
)

# add columns and set them as index
demand_R4_R4_CH["years"] = "2030"
demand_R4_R4_CH["techs"] = "Demand"
demand_R4_R4_CH["commodity"] = "T_HDV_th"
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
        [m.set.nodesdata, m.set.yearssel, ["Demand"], ["T_HDV_th"]]
    )
)
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesFixedProfile"] = 1
sourcesink_config = sourcesink_config.dropna()

m.parameter.add(sourcesink_config, "sourcesink_config")
sourcesink_config
#############################################################

demand_R4_R5_CH = profiles[["LDV_CI_2030", "LDV_FJ_2030","LDV_FSM_2030", "LDV_KB_2030", "LDV_MI_2030","LDV_NU_2030","LDV_NE_2030","LDV_PU_2030","LDV_PNG_2030","LDV_SA_2030","LDV_SI_2030","LDV_TA_2030","LDV_TU_2030","LDV_VU_2030"]]

demand_R4_R5_CH = demand_R4_R5_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R5_CH = demand_R4_R5_CH.T

demand_R4_R5_CH = demand_R4_R5_CH.rename(
    index={"LDV_CI_2030": "CI_data", "LDV_FJ_2030": "FJ_data", "LDV_FSM_2030": "FSM_data", "LDV_KB_2030": "KB_data", "LDV_MI_2030": "MI_data","LDV_NU_2030": "NU_data","LDV_NE_2030": "NE_data","LDV_PU_2030": "PU_data","LDV_PNG_2030": "PNG_data","LDV_SA_2030": "SA_data","LDV_SI_2030": "SI_data","LDV_TA_2030": "TA_data","LDV_TU_2030": "TU_data","LDV_VU_2030": "VU_data"}
)

# add columns and set them as index
demand_R4_R5_CH["years"] = "2030"
demand_R4_R5_CH["techs"] = "Demand"
demand_R4_R5_CH["commodity"] = "T_LDV_th"
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
        [m.set.nodesdata, m.set.yearssel, ["Demand"], ["T_LDV_th"]]
    )
)
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesFixedProfile"] = 1
sourcesink_config = sourcesink_config.dropna()

m.parameter.add(sourcesink_config, "sourcesink_config")
sourcesink_config
############################################################

############################################################
demand_R4_R6_CH = profiles[["Bus_CI_2030", "Bus_FJ_2030","Bus_FSM_2030", "Bus_KB_2030", "Bus_MI_2030","Bus_NU_2030","Bus_NE_2030","Bus_PU_2030","Bus_PNG_2030","Bus_SA_2030","Bus_SI_2030","Bus_TA_2030","Bus_TU_2030","Bus_VU_2030"]]

demand_R4_R6_CH = demand_R4_R6_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R6_CH = demand_R4_R6_CH.T

demand_R4_R6_CH = demand_R4_R6_CH.rename(
    index={"Bus_CI_2030": "CI_data", "Bus_FJ_2030": "FJ_data", "Bus_FSM_2030": "FSM_data", "Bus_KB_2030": "KB_data", "Bus_MI_2030": "MI_data","Bus_NU_2030": "NU_data","Bus_NE_2030": "NE_data","Bus_PU_2030": "PU_data","Bus_PNG_2030": "PNG_data","Bus_SA_2030": "SA_data","Bus_SI_2030": "SI_data","Bus_TA_2030": "TA_data","Bus_TU_2030": "TU_data","Bus_VU_2030": "VU_data"}
)

# add columns and set them as index
demand_R4_R6_CH["years"] = "2030"
demand_R4_R6_CH["techs"] = "Demand"
demand_R4_R6_CH["commodity"] = "T_Bus_th"
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
        [m.set.nodesdata, m.set.yearssel, ["Demand"], ["T_Bus_th"]]
    )
)
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesFixedProfile"] = 1
sourcesink_config = sourcesink_config.dropna()

m.parameter.add(sourcesink_config, "sourcesink_config")
sourcesink_config
############################################################
demand_R4_R7_CH = profiles[["Two_wheel_CI_2030", "Two_wheel_FJ_2030","Two_wheel_FSM_2030", "Two_wheel_KB_2030", "Two_wheel_MI_2030","Two_wheel_NU_2030","Two_wheel_NE_2030","Two_wheel_PU_2030","Two_wheel_PNG_2030","Two_wheel_SA_2030","Two_wheel_SI_2030","Two_wheel_TA_2030","Two_wheel_TU_2030","Two_wheel_VU_2030"]]

demand_R4_R7_CH = demand_R4_R7_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R7_CH = demand_R4_R7_CH.T

demand_R4_R7_CH = demand_R4_R7_CH.rename(
    index={"Two_wheel_CI_2030": "CI_data", "Two_wheel_FJ_2030": "FJ_data", "Two_wheel_FSM_2030": "FSM_data", "Two_wheel_KB_2030": "KB_data", "Two_wheel_MI_2030": "MI_data","Two_wheel_NU_2030": "NU_data","Two_wheel_NE_2030": "NE_data","Two_wheel_PU_2030": "PU_data","Two_wheel_PNG_2030": "PNG_data","Two_wheel_SA_2030": "SA_data","Two_wheel_SI_2030": "SI_data","Two_wheel_TA_2030": "TA_data","Two_wheel_TU_2030": "TU_data","Two_wheel_VU_2030": "VU_data"}
)

# add columns and set them as index
demand_R4_R7_CH["years"] = "2030"
demand_R4_R7_CH["techs"] = "Demand"
demand_R4_R7_CH["commodity"] = "T_Two_wheel_th"
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
        [m.set.nodesdata, m.set.yearssel, ["Demand"], ["T_Two_wheel_th"]]
    )
)
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesFixedProfile"] = 1
sourcesink_config = sourcesink_config.dropna()

m.parameter.add(sourcesink_config, "sourcesink_config")
sourcesink_config
############################################################
demand_R4_R8_CH = profiles[["Marine_CI_2030", "Marine_FJ_2030","Marine_FSM_2030", "Marine_KB_2030", "Marine_MI_2030","Marine_NU_2030","Marine_NE_2030","Marine_PU_2030","Marine_PNG_2030","Marine_SA_2030","Marine_SI_2030","Marine_TA_2030","Marine_TU_2030","Marine_VU_2030"]]

demand_R4_R8_CH = demand_R4_R8_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R8_CH = demand_R4_R8_CH.T

demand_R4_R8_CH = demand_R4_R8_CH.rename(
    index={"Marine_CI_2030": "CI_data", "Marine_FJ_2030": "FJ_data", "Marine_FSM_2030": "FSM_data", "Marine_KB_2030": "KB_data", "Marine_MI_2030": "MI_data","Marine_NU_2030": "NU_data","Marine_NE_2030": "NE_data","Marine_PU_2030": "PU_data","Marine_PNG_2030": "PNG_data","Marine_SA_2030": "SA_data","Marine_SI_2030": "SI_data","Marine_TA_2030": "TA_data","Marine_TU_2030": "TU_data","Marine_VU_2030": "VU_data"}
)

# add columns and set them as index
demand_R4_R8_CH["years"] = "2030"
demand_R4_R8_CH["techs"] = "Demand"
demand_R4_R8_CH["commodity"] = "T_Marine_th"
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
        [m.set.nodesdata, m.set.yearssel, ["Demand"], ["T_Marine_th"]]
    )
)
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesFixedProfile"] = 1
sourcesink_config = sourcesink_config.dropna()

m.parameter.add(sourcesink_config, "sourcesink_config")
sourcesink_config
#############################################################
demand_R4_R9_CH = profiles[["Aviation_CI_2030", "Aviation_FJ_2030","Aviation_FSM_2030", "Aviation_KB_2030", "Aviation_MI_2030","Aviation_NU_2030","Aviation_NE_2030","Aviation_PU_2030","Aviation_PNG_2030","Aviation_SA_2030","Aviation_SI_2030","Aviation_TA_2030","Aviation_TU_2030","Aviation_VU_2030"]]

demand_R4_R9_CH = demand_R4_R9_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R9_CH = demand_R4_R9_CH.T

demand_R4_R9_CH = demand_R4_R9_CH.rename(
    index={"Aviation_CI_2030": "CI_data", "Aviation_FJ_2030": "FJ_data", "Aviation_FSM_2030": "FSM_data", "Aviation_KB_2030": "KB_data", "Aviation_MI_2030": "MI_data","Aviation_NU_2030": "NU_data","Aviation_NE_2030": "NE_data","Aviation_PU_2030": "PU_data","Aviation_PNG_2030": "PNG_data","Aviation_SA_2030": "SA_data","Aviation_SI_2030": "SI_data","Aviation_TA_2030": "TA_data","Aviation_TU_2030": "TU_data","Aviation_VU_2030": "VU_data"}
)

# add columns and set them as index
demand_R4_R9_CH["years"] = "2030"
demand_R4_R9_CH["techs"] = "Demand"
demand_R4_R9_CH["commodity"] = "T_Aviation_th"
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
        [m.set.nodesdata, m.set.yearssel, ["Demand"], ["T_Aviation_th"]]
    )
)
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesFixedProfile"] = 1
sourcesink_config = sourcesink_config.dropna()

m.parameter.add(sourcesink_config, "sourcesink_config")
sourcesink_config
#################################################################################
demand_R4_R9_CH = profiles[["Marinef_CI_2030", "Marinef_FJ_2030","Marinef_FSM_2030", "Marinef_KB_2030", "Marinef_MI_2030","Marinef_NU_2030","Marinef_NE_2030","Marinef_PU_2030","Marinef_PNG_2030","Marinef_SA_2030","Marinef_SI_2030","Marinef_TA_2030","Marinef_TU_2030","Marinef_VU_2030"]]

demand_R4_R9_CH = demand_R4_R9_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R9_CH = demand_R4_R9_CH.T

demand_R4_R9_CH = demand_R4_R9_CH.rename(
    index={"Marinef_CI_2030": "CI_data", "Marinef_FJ_2030": "FJ_data", "Marinef_FSM_2030": "FSM_data", "Marinef_KB_2030": "KB_data", "Marinef_MI_2030": "MI_data","Marinef_NU_2030": "NU_data","Marinef_NE_2030": "NE_data","Marinef_PU_2030": "PU_data","Marinef_PNG_2030": "PNG_data","Marinef_SA_2030": "SA_data","Marinef_SI_2030": "SI_data","Marinef_TA_2030": "TA_data","Marinef_TU_2030": "TU_data","Marinef_VU_2030": "VU_data"}
)

# add columns and set them as index
demand_R4_R9_CH["years"] = "2030"
demand_R4_R9_CH["techs"] = "Demand"
demand_R4_R9_CH["commodity"] = "T_Marine_f_th"
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
        [m.set.nodesdata, m.set.yearssel, ["Demand"], ["T_Marine_f_th"]]
    )
)
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesFixedProfile"] = 1
sourcesink_config = sourcesink_config.dropna()

m.parameter.add(sourcesink_config, "sourcesink_config")
sourcesink_config
#################################################################################
demand_R4_R10_CH = profiles[["HC_CI_2030", "HC_FJ_2030","HC_FSM_2030", "HC_KB_2030", "HC_MI_2030","HC_NU_2030","HC_NE_2030","HC_PU_2030","HC_PNG_2030","HC_SA_2030","HC_SI_2030","HC_TA_2030","HC_TU_2030","HC_VU_2030"]]

demand_R4_R10_CH = demand_R4_R10_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R10_CH = demand_R4_R10_CH.T

demand_R4_R10_CH = demand_R4_R10_CH.rename(
    index={"HC_CI_2030": "CI_data", "HC_FJ_2030": "FJ_data", "HC_FSM_2030": "FSM_data", "HC_KB_2030": "KB_data", "HC_MI_2030": "MI_data","HC_NU_2030": "NU_data","HC_NE_2030": "NE_data","HC_PU_2030": "PU_data","HC_PNG_2030": "PNG_data","HC_SA_2030": "SA_data","HC_SI_2030": "SI_data","HC_TA_2030": "TA_data","HC_TU_2030": "TU_data","HC_VU_2030": "VU_data"}
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
demand_R4_R10_CH = profiles[["HI_CI_2030", "HI_FJ_2030","HI_FSM_2030", "HI_KB_2030", "HI_MI_2030","HI_NU_2030","HI_NE_2030","HI_PU_2030","HI_PNG_2030","HI_SA_2030","HI_SI_2030","HI_TA_2030","HI_TU_2030","HI_VU_2030"]]

demand_R4_R10_CH = demand_R4_R10_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R10_CH = demand_R4_R10_CH.T

demand_R4_R10_CH = demand_R4_R10_CH.rename(
    index={"HI_CI_2030": "CI_data", "HI_FJ_2030": "FJ_data", "HI_FSM_2030": "FSM_data", "HI_KB_2030": "KB_data", "HI_MI_2030": "MI_data","HI_NU_2030": "NU_data","HI_NE_2030": "NE_data","HI_PU_2030": "PU_data","HI_PNG_2030": "PNG_data","HI_SA_2030": "SA_data","HI_SI_2030": "SI_data","HI_TA_2030": "TA_data","HI_TU_2030": "TU_data","HI_VU_2030": "VU_data"}
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
demand_R4_R10_CH = profiles[["DHWE_CI_2030", "DHWE_FJ_2030","DHWE_FSM_2030", "DHWE_KB_2030", "DHWE_MI_2030","DHWE_NU_2030","DHWE_NE_2030","DHWE_PU_2030","DHWE_PNG_2030","DHWE_SA_2030","DHWE_SI_2030","DHWE_TA_2030","DHWE_TU_2030","DHWE_VU_2030"]]

demand_R4_R10_CH = demand_R4_R10_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R10_CH = demand_R4_R10_CH.T

demand_R4_R10_CH = demand_R4_R10_CH.rename(
    index={"DHWE_CI_2030": "CI_data", "DHWE_FJ_2030": "FJ_data", "DHWE_FSM_2030": "FSM_data", "DHWE_KB_2030": "KB_data", "DHWE_MI_2030": "MI_data","DHWE_NU_2030": "NU_data","DHWE_NE_2030": "NE_data","DHWE_PU_2030": "PU_data","DHWE_PNG_2030": "PNG_data","DHWE_SA_2030": "SA_data","DHWE_SI_2030": "SI_data","DHWE_TA_2030": "TA_data","DHWE_TU_2030": "TU_data","DHWE_VU_2030": "VU_data"}
)

# add columns and set them as index
demand_R4_R10_CH["years"] = "2030"
demand_R4_R10_CH["techs"] = "Demand"
demand_R4_R10_CH["commodity"] = "DHW_el"
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
        [m.set.nodesdata, m.set.yearssel, ["Demand"], ["DHW_el"]]
    )
)
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesFixedProfile"] = 1
sourcesink_config = sourcesink_config.dropna()

m.parameter.add(sourcesink_config, "sourcesink_config")
sourcesink_config
#############################################################

####################################################################################
demand_R4_R10_CH = profiles[["DHWL_CI_2030", "DHWL_FJ_2030","DHWL_FSM_2030", "DHWL_KB_2030", "DHWL_MI_2030","DHWL_NU_2030","DHWL_NE_2030","DHWL_PU_2030","DHWL_PNG_2030","DHWL_SA_2030","DHWL_SI_2030","DHWL_TA_2030","DHWL_TU_2030","DHWL_VU_2030"]]

demand_R4_R10_CH = demand_R4_R10_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R10_CH = demand_R4_R10_CH.T

demand_R4_R10_CH = demand_R4_R10_CH.rename(
    index={"DHWL_CI_2030": "CI_data", "DHWL_FJ_2030": "FJ_data", "DHWL_FSM_2030": "FSM_data", "DHWL_KB_2030": "KB_data", "DHWL_MI_2030": "MI_data","DHWL_NU_2030": "NU_data","DHWL_NE_2030": "NE_data","DHWL_PU_2030": "PU_data","DHWL_PNG_2030": "PNG_data","DHWL_SA_2030": "SA_data","DHWL_SI_2030": "SI_data","DHWL_TA_2030": "TA_data","DHWL_TU_2030": "TU_data","DHWL_VU_2030": "VU_data"}
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
##################################################################################2040 demand####################################################################################################
demand_R4_R20_CH = profiles[["demand_CI_2040", "demand_FJ_2040","demand_FSM_2040", "demand_KB_2040", "demand_MI_2040","demand_NU_2040","demand_NE_2040","demand_PU_2040","demand_PNG_2040","demand_SA_2040","demand_SI_2040","demand_TA_2040","demand_TU_2040","demand_VU_2040"]]

demand_R4_R20_CH = demand_R4_R20_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R20_CH = demand_R4_R20_CH.T

demand_R4_R20_CH = demand_R4_R20_CH.rename(
    index={"demand_CI_2040": "CI_data", "demand_FJ_2040": "FJ_data", "demand_FSM_2040": "FSM_data", "demand_KB_2040": "KB_data", "demand_MI_2040": "MI_data","demand_NU_2040": "NU_data","demand_NE_2040": "NE_data","demand_PU_2040": "PU_data","demand_PNG_2040": "PNG_data","demand_SA_2040": "SA_data","demand_SI_2040": "SI_data","demand_TA_2040": "TA_data","demand_TU_2040": "TU_data","demand_VU_2040": "VU_data"}
)

# add columns and set them as index
demand_R4_R20_CH["years"] = "2040"
demand_R4_R20_CH["techs"] = "Demand"
demand_R4_R20_CH["commodity"] = "Elec"
demand_R4_R20_CH["type"] = "fixed"
demand_R4_R20_CH = demand_R4_R20_CH.set_index(
    ["years", "techs", "commodity", "type"], append=True
)

m.profile.add(demand_R4_R20_CH, "sourcesink_profile")
demand_R4_R20_CH.iloc[:, 0:8]

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
###################################################################################################
demand_R4_R3_CH = profiles[["MDV_el_CI_2040","MDV_el_FJ_2040","MDV_el_FSM_2040","MDV_el_KB_2040","MDV_el_MI_2040","MDV_el_NU_2040","MDV_el_NE_2040","MDV_el_PU_2040","MDV_el_PNG_2040","MDV_el_SA_2040","MDV_el_SI_2040","MDV_el_TA_2040","MDV_el_TU_2040","MDV_el_VU_2040"]]

demand_R4_R3_CH = demand_R4_R3_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R3_CH = demand_R4_R3_CH.T

demand_R4_R3_CH = demand_R4_R3_CH.rename(
    index={"MDV_el_CI_2040": "CI_data", "MDV_el_FJ_2040": "FJ_data", "MDV_el_FSM_2040": "FSM_data", "MDV_el_KB_2040": "KB_data", "MDV_el_MI_2040": "MI_data","MDV_el_NU_2040": "NU_data","MDV_el_NE_2040": "NE_data","MDV_el_PU_2040": "PU_data","MDV_el_PNG_2040": "PNG_data","MDV_el_SA_2040": "SA_data","MDV_el_SI_2040": "SI_data","MDV_el_TA_2040": "TA_data","MDV_el_TU_2040": "TU_data","MDV_el_VU_2040": "VU_data"}
)

# add columns and set them as index
demand_R4_R3_CH["years"] = "2040"
demand_R4_R3_CH["techs"] = "Demand"
demand_R4_R3_CH["commodity"] = "T_MDV_el"
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
        [m.set.nodesdata, m.set.yearssel, ["Demand"], ["T_MDV_el"]]
    )
)
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesFixedProfile"] = 1
sourcesink_config = sourcesink_config.dropna()

m.parameter.add(sourcesink_config, "sourcesink_config")
sourcesink_config
####################################################################################
demand_R4_R3_CH = profiles[["MDV_Th_CI_2040","MDV_Th_FJ_2040","MDV_Th_FSM_2040","MDV_Th_KB_2040","MDV_Th_MI_2040","MDV_Th_NU_2040","MDV_Th_NE_2040","MDV_Th_PU_2040","MDV_Th_PNG_2040","MDV_Th_SA_2040","MDV_Th_SI_2040","MDV_Th_TA_2040","MDV_Th_TU_2040","MDV_Th_VU_2040"]]

demand_R4_R3_CH = demand_R4_R3_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R3_CH = demand_R4_R3_CH.T

demand_R4_R3_CH = demand_R4_R3_CH.rename(
    index={"MDV_Th_CI_2040": "CI_data", "MDV_Th_FJ_2040": "FJ_data", "MDV_Th_FSM_2040": "FSM_data", "MDV_Th_KB_2040": "KB_data", "MDV_Th_MI_2040": "MI_data","MDV_Th_NU_2040": "NU_data","MDV_Th_NE_2040": "NE_data","MDV_Th_PU_2040": "PU_data","MDV_Th_PNG_2040": "PNG_data","MDV_Th_SA_2040": "SA_data","MDV_Th_SI_2040": "SI_data","MDV_Th_TA_2040": "TA_data","MDV_Th_TU_2040": "TU_data","MDV_Th_VU_2040": "VU_data"}
)

# add columns and set them as index
demand_R4_R3_CH["years"] = "2040"
demand_R4_R3_CH["techs"] = "Demand"
demand_R4_R3_CH["commodity"] = "T_MDV_th"
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
        [m.set.nodesdata, m.set.yearssel, ["Demand"], ["T_MDV_th"]]
    )
)
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesFixedProfile"] = 1
sourcesink_config = sourcesink_config.dropna()

m.parameter.add(sourcesink_config, "sourcesink_config")
sourcesink_config
########################################################
demand_R4_R3_CH = profiles[["MDV_BF_CI_2040","MDV_BF_FJ_2040","MDV_BF_FSM_2040","MDV_BF_KB_2040","MDV_BF_MI_2040","MDV_BF_NU_2040","MDV_BF_NE_2040","MDV_BF_PU_2040","MDV_BF_PNG_2040","MDV_BF_SA_2040","MDV_BF_SI_2040","MDV_BF_TA_2040","MDV_BF_TU_2040","MDV_BF_VU_2040"]]

demand_R4_R3_CH = demand_R4_R3_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R3_CH = demand_R4_R3_CH.T

demand_R4_R3_CH = demand_R4_R3_CH.rename(
    index={"MDV_BF_CI_2040": "CI_data", "MDV_BF_FJ_2040": "FJ_data", "MDV_BF_FSM_2040": "FSM_data", "MDV_BF_KB_2040": "KB_data", "MDV_BF_MI_2040": "MI_data","MDV_BF_NU_2040": "NU_data","MDV_BF_NE_2040": "NE_data","MDV_BF_PU_2040": "PU_data","MDV_BF_PNG_2040": "PNG_data","MDV_BF_SA_2040": "SA_data","MDV_BF_SI_2040": "SI_data","MDV_BF_TA_2040": "TA_data","MDV_BF_TU_2040": "TU_data","MDV_BF_VU_2040": "VU_data"}
)

# add columns and set them as index
demand_R4_R3_CH["years"] = "2040"
demand_R4_R3_CH["techs"] = "Demand"
demand_R4_R3_CH["commodity"] = "T_MDV_BF"
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
        [m.set.nodesdata, m.set.yearssel, ["Demand"], ["T_MDV_BF"]]
    )
)
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesFixedProfile"] = 1
sourcesink_config = sourcesink_config.dropna()

m.parameter.add(sourcesink_config, "sourcesink_config")
sourcesink_config
#####################################################################################HDV######################################################
demand_R4_R3_CH = profiles[["HDV_el_CI_2040","HDV_el_FJ_2040","HDV_el_FSM_2040","HDV_el_KB_2040","HDV_el_MI_2040","HDV_el_NU_2040","HDV_el_NE_2040","HDV_el_PU_2040","HDV_el_PNG_2040","HDV_el_SA_2040","HDV_el_SI_2040","HDV_el_TA_2040","HDV_el_TU_2040","HDV_el_VU_2040"]]

demand_R4_R3_CH = demand_R4_R3_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R3_CH = demand_R4_R3_CH.T

demand_R4_R3_CH = demand_R4_R3_CH.rename(
    index={"HDV_el_CI_2040": "CI_data", "HDV_el_FJ_2040": "FJ_data", "HDV_el_FSM_2040": "FSM_data", "HDV_el_KB_2040": "KB_data", "HDV_el_MI_2040": "MI_data","HDV_el_NU_2040": "NU_data","HDV_el_NE_2040": "NE_data","HDV_el_PU_2040": "PU_data","HDV_el_PNG_2040": "PNG_data","HDV_el_SA_2040": "SA_data","HDV_el_SI_2040": "SI_data","HDV_el_TA_2040": "TA_data","HDV_el_TU_2040": "TU_data","HDV_el_VU_2040": "VU_data"}
)

# add columns and set them as index
demand_R4_R3_CH["years"] = "2040"
demand_R4_R3_CH["techs"] = "Demand"
demand_R4_R3_CH["commodity"] = "T_HDV_el"
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
        [m.set.nodesdata, m.set.yearssel, ["Demand"], ["T_HDV_el"]]
    )
)
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesFixedProfile"] = 1
sourcesink_config = sourcesink_config.dropna()

m.parameter.add(sourcesink_config, "sourcesink_config")
sourcesink_config
####################################################################################
demand_R4_R3_CH = profiles[["HDV_Th_CI_2040","HDV_Th_FJ_2040","HDV_Th_FSM_2040","HDV_Th_KB_2040","HDV_Th_MI_2040","HDV_Th_NU_2040","HDV_Th_NE_2040","HDV_Th_PU_2040","HDV_Th_PNG_2040","HDV_Th_SA_2040","HDV_Th_SI_2040","HDV_Th_TA_2040","HDV_Th_TU_2040","HDV_Th_VU_2040"]]

demand_R4_R3_CH = demand_R4_R3_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R3_CH = demand_R4_R3_CH.T

demand_R4_R3_CH = demand_R4_R3_CH.rename(
    index={"HDV_Th_CI_2040": "CI_data", "HDV_Th_FJ_2040": "FJ_data", "HDV_Th_FSM_2040": "FSM_data", "HDV_Th_KB_2040": "KB_data", "HDV_Th_MI_2040": "MI_data","HDV_Th_NU_2040": "NU_data","HDV_Th_NE_2040": "NE_data","HDV_Th_PU_2040": "PU_data","HDV_Th_PNG_2040": "PNG_data","HDV_Th_SA_2040": "SA_data","HDV_Th_SI_2040": "SI_data","HDV_Th_TA_2040": "TA_data","HDV_Th_TU_2040": "TU_data","HDV_Th_VU_2040": "VU_data"}
)

# add columns and set them as index
demand_R4_R3_CH["years"] = "2040"
demand_R4_R3_CH["techs"] = "Demand"
demand_R4_R3_CH["commodity"] = "T_HDV_th"
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
        [m.set.nodesdata, m.set.yearssel, ["Demand"], ["T_HDV_th"]]
    )
)
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesFixedProfile"] = 1
sourcesink_config = sourcesink_config.dropna()

m.parameter.add(sourcesink_config, "sourcesink_config")
sourcesink_config
########################################################
demand_R4_R3_CH = profiles[["HDV_BF_CI_2040","HDV_BF_FJ_2040","HDV_BF_FSM_2040","HDV_BF_KB_2040","HDV_BF_MI_2040","HDV_BF_NU_2040","HDV_BF_NE_2040","HDV_BF_PU_2040","HDV_BF_PNG_2040","HDV_BF_SA_2040","HDV_BF_SI_2040","HDV_BF_TA_2040","HDV_BF_TU_2040","HDV_BF_VU_2040"]]

demand_R4_R3_CH = demand_R4_R3_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R3_CH = demand_R4_R3_CH.T

demand_R4_R3_CH = demand_R4_R3_CH.rename(
    index={"HDV_BF_CI_2040": "CI_data", "HDV_BF_FJ_2040": "FJ_data", "HDV_BF_FSM_2040": "FSM_data", "HDV_BF_KB_2040": "KB_data", "HDV_BF_MI_2040": "MI_data","HDV_BF_NU_2040": "NU_data","HDV_BF_NE_2040": "NE_data","HDV_BF_PU_2040": "PU_data","HDV_BF_PNG_2040": "PNG_data","HDV_BF_SA_2040": "SA_data","HDV_BF_SI_2040": "SI_data","HDV_BF_TA_2040": "TA_data","HDV_BF_TU_2040": "TU_data","HDV_BF_VU_2040": "VU_data"}
)

# add columns and set them as index
demand_R4_R3_CH["years"] = "2040"
demand_R4_R3_CH["techs"] = "Demand"
demand_R4_R3_CH["commodity"] = "T_HDV_BF"
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
        [m.set.nodesdata, m.set.yearssel, ["Demand"], ["T_HDV_BF"]]
    )
)
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesFixedProfile"] = 1
sourcesink_config = sourcesink_config.dropna()

m.parameter.add(sourcesink_config, "sourcesink_config")
sourcesink_config


#############################################################
demand_R4_R3_CH = profiles[["LDV_el_CI_2040","LDV_el_FJ_2040","LDV_el_FSM_2040","LDV_el_KB_2040","LDV_el_MI_2040","LDV_el_NU_2040","LDV_el_NE_2040","LDV_el_PU_2040","LDV_el_PNG_2040","LDV_el_SA_2040","LDV_el_SI_2040","LDV_el_TA_2040","LDV_el_TU_2040","LDV_el_VU_2040"]]

demand_R4_R3_CH = demand_R4_R3_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R3_CH = demand_R4_R3_CH.T

demand_R4_R3_CH = demand_R4_R3_CH.rename(
    index={"LDV_el_CI_2040": "CI_data", "LDV_el_FJ_2040": "FJ_data", "LDV_el_FSM_2040": "FSM_data", "LDV_el_KB_2040": "KB_data", "LDV_el_MI_2040": "MI_data","LDV_el_NU_2040": "NU_data","HDV_el_NE_2040": "NE_data","LDV_el_PU_2040": "PU_data","LDV_el_PNG_2040": "PNG_data","LDV_el_SA_2040": "SA_data","LDV_el_SI_2040": "SI_data","LDV_el_TA_2040": "TA_data","LDV_el_TU_2040": "TU_data","LDV_el_VU_2040": "VU_data"}
)

# add columns and set them as index
demand_R4_R3_CH["years"] = "2040"
demand_R4_R3_CH["techs"] = "Demand"
demand_R4_R3_CH["commodity"] = "T_LDV_el"
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
        [m.set.nodesdata, m.set.yearssel, ["Demand"], ["T_LDV_el"]]
    )
)
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesFixedProfile"] = 1
sourcesink_config = sourcesink_config.dropna()

m.parameter.add(sourcesink_config, "sourcesink_config")
sourcesink_config
####################################################################################
demand_R4_R3_CH = profiles[["LDV_Th_CI_2040","LDV_Th_FJ_2040","LDV_Th_FSM_2040","LDV_Th_KB_2040","LDV_Th_MI_2040","LDV_Th_NU_2040","LDV_Th_NE_2040","LDV_Th_PU_2040","LDV_Th_PNG_2040","LDV_Th_SA_2040","LDV_Th_SI_2040","LDV_Th_TA_2040","LDV_Th_TU_2040","LDV_Th_VU_2040"]]

demand_R4_R3_CH = demand_R4_R3_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R3_CH = demand_R4_R3_CH.T

demand_R4_R3_CH = demand_R4_R3_CH.rename(
    index={"LDV_Th_CI_2040": "CI_data", "LDV_Th_FJ_2040": "FJ_data", "LDV_Th_FSM_2040": "FSM_data", "LDV_Th_KB_2040": "KB_data", "LDV_Th_MI_2040": "MI_data","LDV_Th_NU_2040": "NU_data","LDV_Th_NE_2040": "NE_data","LDV_Th_PU_2040": "PU_data","LDV_Th_PNG_2040": "PNG_data","LDV_Th_SA_2040": "SA_data","LDV_Th_SI_2040": "SI_data","LDV_Th_TA_2040": "TA_data","LDV_Th_TU_2040": "TU_data","LDV_Th_VU_2040": "VU_data"}
)

# add columns and set them as index
demand_R4_R3_CH["years"] = "2040"
demand_R4_R3_CH["techs"] = "Demand"
demand_R4_R3_CH["commodity"] = "T_LDV_th"
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
        [m.set.nodesdata, m.set.yearssel, ["Demand"], ["T_LDV_th"]]
    )
)
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesFixedProfile"] = 1
sourcesink_config = sourcesink_config.dropna()

m.parameter.add(sourcesink_config, "sourcesink_config")
sourcesink_config
########################################################
demand_R4_R3_CH = profiles[["LDV_BF_CI_2040","LDV_BF_FJ_2040","LDV_BF_FSM_2040","LDV_BF_KB_2040","LDV_BF_MI_2040","LDV_BF_NU_2040","LDV_BF_NE_2040","LDV_BF_PU_2040","LDV_BF_PNG_2040","LDV_BF_SA_2040","LDV_BF_SI_2040","LDV_BF_TA_2040","LDV_BF_TU_2040","LDV_BF_VU_2040"]]

demand_R4_R3_CH = demand_R4_R3_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R3_CH = demand_R4_R3_CH.T

demand_R4_R3_CH = demand_R4_R3_CH.rename(
    index={"LDV_BF_CI_2040": "CI_data", "LDV_BF_FJ_2040": "FJ_data", "LDV_BF_FSM_2040": "FSM_data", "LDV_BF_KB_2040": "KB_data", "LDV_BF_MI_2040": "MI_data","LDV_BF_NU_2040": "NU_data","LDV_BF_NE_2040": "NE_data","LDV_BF_PU_2040": "PU_data","LDV_BF_PNG_2040": "PNG_data","LDV_BF_SA_2040": "SA_data","LDV_BF_SI_2040": "SI_data","LDV_BF_TA_2040": "TA_data","LDV_BF_TU_2040": "TU_data","LDV_BF_VU_2040": "VU_data"}
)

# add columns and set them as index
demand_R4_R3_CH["years"] = "2040"
demand_R4_R3_CH["techs"] = "Demand"
demand_R4_R3_CH["commodity"] = "T_LDV_BF"
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
        [m.set.nodesdata, m.set.yearssel, ["Demand"], ["T_LDV_BF"]]
    )
)
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesFixedProfile"] = 1
sourcesink_config = sourcesink_config.dropna()

m.parameter.add(sourcesink_config, "sourcesink_config")
sourcesink_config

############################################################
demand_R4_R3_CH = profiles[["BUS_el_CI_2040","BUS_el_FJ_2040","BUS_el_FSM_2040","BUS_el_KB_2040","BUS_el_MI_2040","BUS_el_NU_2040","BUS_el_NE_2040","BUS_el_PU_2040","BUS_el_PNG_2040","BUS_el_SA_2040","BUS_el_SI_2040","BUS_el_TA_2040","BUS_el_TU_2040","BUS_el_VU_2040"]]

demand_R4_R3_CH = demand_R4_R3_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R3_CH = demand_R4_R3_CH.T

demand_R4_R3_CH = demand_R4_R3_CH.rename(
    index={"BUS_el_CI_2040": "CI_data", "BUS_el_FJ_2040": "FJ_data", "BUS_el_FSM_2040": "FSM_data", "BUS_el_KB_2040": "KB_data", "BUS_el_MI_2040": "MI_data","BUS_el_NU_2040": "NU_data","BUS_el_NE_2040": "NE_data","BUS_el_PU_2040": "PU_data","BUS_el_PNG_2040": "PNG_data","BUS_el_SA_2040": "SA_data","BUS_el_SI_2040": "SI_data","BUS_el_TA_2040": "TA_data","BUS_el_TU_2040": "TU_data","BUS_el_VU_2040": "VU_data"}
)

# add columns and set them as index
demand_R4_R3_CH["years"] = "2040"
demand_R4_R3_CH["techs"] = "Demand"
demand_R4_R3_CH["commodity"] = "T_Bus_el"
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
        [m.set.nodesdata, m.set.yearssel, ["Demand"], ["T_Bus_el"]]
    )
)
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesFixedProfile"] = 1
sourcesink_config = sourcesink_config.dropna()

m.parameter.add(sourcesink_config, "sourcesink_config")
sourcesink_config
####################################################################################
demand_R4_R3_CH = profiles[["BUS_Th_CI_2040","BUS_Th_FJ_2040","BUS_Th_FSM_2040","BUS_Th_KB_2040","BUS_Th_MI_2040","BUS_Th_NU_2040","BUS_Th_NE_2040","BUS_Th_PU_2040","BUS_Th_PNG_2040","BUS_Th_SA_2040","BUS_Th_SI_2040","BUS_Th_TA_2040","BUS_Th_TU_2040","BUS_Th_VU_2040"]]

demand_R4_R3_CH = demand_R4_R3_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R3_CH = demand_R4_R3_CH.T

demand_R4_R3_CH = demand_R4_R3_CH.rename(
    index={"BUS_Th_CI_2040": "CI_data", "BUS_Th_FJ_2040": "FJ_data", "BUS_Th_FSM_2040": "FSM_data", "BUS_Th_KB_2040": "KB_data", "BUS_Th_MI_2040": "MI_data","BUS_Th_NU_2040": "NU_data","BUS_Th_NE_2040": "NE_data","BUS_Th_PU_2040": "PU_data","BUS_Th_PNG_2040": "PNG_data","BUS_Th_SA_2040": "SA_data","BUS_Th_SI_2040": "SI_data","BUS_Th_TA_2040": "TA_data","BUS_Th_TU_2040": "TU_data","BUS_Th_VU_2040": "VU_data"}
)

# add columns and set them as index
demand_R4_R3_CH["years"] = "2040"
demand_R4_R3_CH["techs"] = "Demand"
demand_R4_R3_CH["commodity"] = "T_Bus_th"
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
        [m.set.nodesdata, m.set.yearssel, ["Demand"], ["T_Bus_th"]]
    )
)
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesFixedProfile"] = 1
sourcesink_config = sourcesink_config.dropna()

m.parameter.add(sourcesink_config, "sourcesink_config")
sourcesink_config
############################################################
demand_R4_R7_CH = profiles[["2W_el_CI_2040", "2W_el_FJ_2040","2W_el_FSM_2040", "2W_el_KB_2040", "2W_el_MI_2040","2W_el_NU_2040","2W_el_NE_2040","2W_el_PU_2040","2W_el_PNG_2040","2W_el_SA_2040","2W_el_SI_2040","2W_el_TA_2040","2W_el_TU_2040","2W_el_VU_2040"]]

demand_R4_R7_CH = demand_R4_R7_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R7_CH = demand_R4_R7_CH.T

demand_R4_R7_CH = demand_R4_R7_CH.rename(
    index={"2W_el_CI_2040": "CI_data", "2W_el_FJ_2040": "FJ_data", "2W_el_FSM_2040": "FSM_data", "2W_el_KB_2040": "KB_data", "2W_el_MI_2040": "MI_data","2W_el_NU_2040": "NU_data","2W_el_NE_2040": "NE_data","2W_el_PU_2040": "PU_data","2W_el_PNG_2040": "PNG_data","2W_el_SA_2040": "SA_data","2W_el_SI_2040": "SI_data","2W_el_TA_2040": "TA_data","2W_el_TU_2040": "TU_data","2W_el_VU_2040": "VU_data"}
)

# add columns and set them as index
demand_R4_R7_CH["years"] = "2040"
demand_R4_R7_CH["techs"] = "Demand"
demand_R4_R7_CH["commodity"] = "T_Two_wheel_el"
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
        [m.set.nodesdata, m.set.yearssel, ["Demand"], ["T_Two_wheel_el"]]
    )
)
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesFixedProfile"] = 1
sourcesink_config = sourcesink_config.dropna()

m.parameter.add(sourcesink_config, "sourcesink_config")
sourcesink_config
############################################################
demand_R4_R7_CH = profiles[["2W_th_CI_2040", "2W_th_FJ_2040","2W_th_FSM_2040", "2W_th_KB_2040", "2W_th_MI_2040","2W_th_NU_2040","2W_th_NE_2040","2W_th_PU_2040","2W_th_PNG_2040","2W_th_SA_2040","2W_th_SI_2040","2W_th_TA_2040","2W_th_TU_2040","2W_th_VU_2040"]]

demand_R4_R7_CH = demand_R4_R7_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R7_CH = demand_R4_R7_CH.T

demand_R4_R7_CH = demand_R4_R7_CH.rename(
    index={"2W_th_CI_2040": "CI_data", "2W_th_FJ_2040": "FJ_data", "2W_th_FSM_2040": "FSM_data", "2W_th_KB_2040": "KB_data", "2W_th_MI_2040": "MI_data","2W_th_NU_2040": "NU_data","2W_th_NE_2040": "NE_data","2W_th_PU_2040": "PU_data","2W_th_PNG_2040": "PNG_data","2W_th_SA_2040": "SA_data","2W_th_SI_2040": "SI_data","2W_th_TA_2040": "TA_data","2W_th_TU_2040": "TU_data","2W_th_VU_2040": "VU_data"}
)

# add columns and set them as index
demand_R4_R7_CH["years"] = "2040"
demand_R4_R7_CH["techs"] = "Demand"
demand_R4_R7_CH["commodity"] = "T_Two_wheel_th"
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
        [m.set.nodesdata, m.set.yearssel, ["Demand"], ["T_Two_wheel_th"]]
    )
)
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesFixedProfile"] = 1
sourcesink_config = sourcesink_config.dropna()

m.parameter.add(sourcesink_config, "sourcesink_config")
sourcesink_config
#############################################################
demand_R4_R8_CH = profiles[["Marine_TH_CI_2040", "Marine_TH_FJ_2040","Marine_TH_FSM_2040", "Marine_TH_KB_2040", "Marine_TH_MI_2040","Marine_TH_NU_2040","Marine_TH_NE_2040","Marine_TH_PU_2040","Marine_TH_PNG_2040","Marine_TH_SA_2040","Marine_TH_SI_2040","Marine_TH_TA_2040","Marine_TH_TU_2040","Marine_TH_VU_2040"]]

demand_R4_R8_CH = demand_R4_R8_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R8_CH = demand_R4_R8_CH.T

demand_R4_R8_CH = demand_R4_R8_CH.rename(
    index={"Marine_TH_CI_2040": "CI_data", "Marine_TH_FJ_2040": "FJ_data", "Marine_TH_FSM_2040": "FSM_data", "Marine_TH_KB_2040": "KB_data", "Marine_TH_MI_2040": "MI_data","Marine_TH_NU_2040": "NU_data","Marine_TH_NE_2040": "NE_data","Marine_TH_PU_2040": "PU_data","Marine_TH_PNG_2040": "PNG_data","Marine_TH_SA_2040": "SA_data","Marine_TH_SI_2040": "SI_data","Marine_TH_TA_2040": "TA_data","Marine_TH_TU_2040": "TU_data","Marine_TH_VU_2040": "VU_data"}
)

# add columns and set them as index
demand_R4_R8_CH["years"] = "2040"
demand_R4_R8_CH["techs"] = "Demand"
demand_R4_R8_CH["commodity"] = "T_Marine_f_th"
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
        [m.set.nodesdata, m.set.yearssel, ["Demand"], ["T_Marine_f_th"]]
    )
)
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesFixedProfile"] = 1
sourcesink_config = sourcesink_config.dropna()

m.parameter.add(sourcesink_config, "sourcesink_config")
sourcesink_config
#############################################################
demand_R4_R8_CH = profiles[["Marine_E_CI_2040", "Marine_E_FJ_2040","Marine_E_FSM_2040", "Marine_E_KB_2040", "Marine_E_MI_2040","Marine_E_NU_2040","Marine_E_NE_2040","Marine_E_PU_2040","Marine_E_PNG_2040","Marine_E_SA_2040","Marine_E_SI_2040","Marine_E_TA_2040","Marine_E_TU_2040","Marine_E_VU_2040"]]

demand_R4_R8_CH = demand_R4_R8_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R8_CH = demand_R4_R8_CH.T

demand_R4_R8_CH = demand_R4_R8_CH.rename(
    index={"Marine_E_CI_2040": "CI_data", "Marine_E_FJ_2040": "FJ_data", "Marine_E_FSM_2040": "FSM_data", "Marine_E_KB_2040": "KB_data", "Marine_E_MI_2040": "MI_data","Marine_E_NU_2040": "NU_data","Marine_E_NE_2040": "NE_data","Marine_E_PU_2040": "PU_data","Marine_E_PNG_2040": "PNG_data","Marine_E_SA_2040": "SA_data","Marine_E_SI_2040": "SI_data","Marine_E_TA_2040": "TA_data","Marine_E_TU_2040": "TU_data","Marine_E_VU_2040": "VU_data"}
)

# add columns and set them as index
demand_R4_R8_CH["years"] = "2040"
demand_R4_R8_CH["techs"] = "Demand"
demand_R4_R8_CH["commodity"] = "Dummy_EL"
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
        [m.set.nodesdata, m.set.yearssel, ["Demand"], ["Dummy_EL"]]
    )
)
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesFixedProfile"] = 1
sourcesink_config = sourcesink_config.dropna()

m.parameter.add(sourcesink_config, "sourcesink_config")
sourcesink_config
###############################################################################
demand_R4_R9_CH = profiles[["AVIA_TH_CI_2040", "AVIA_TH_FJ_2040","AVIA_TH_FSM_2040", "AVIA_TH_KB_2040", "AVIA_TH_MI_2040","AVIA_TH_NU_2040","AVIA_TH_NE_2040","AVIA_TH_PU_2040","AVIA_TH_PNG_2040","AVIA_TH_SA_2040","AVIA_TH_SI_2040","AVIA_TH_TA_2040","AVIA_TH_TU_2040","AVIA_TH_VU_2040"]]

demand_R4_R9_CH = demand_R4_R9_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R9_CH = demand_R4_R9_CH.T

demand_R4_R9_CH = demand_R4_R9_CH.rename(
    index={"AVIA_TH_CI_2040": "CI_data", "AVIA_TH_FJ_2040": "FJ_data", "AVIA_TH_FSM_2040": "FSM_data", "AVIA_TH_KB_2040": "KB_data", "AVIA_TH_MI_2040": "MI_data","AVIA_TH_NU_2040": "NU_data","AVIA_TH_NE_2040": "NE_data","AVIA_TH_PU_2040": "PU_data","AVIA_TH_PNG_2040": "PNG_data","AVIA_TH_SA_2040": "SA_data","AVIA_TH_SI_2040": "SI_data","AVIA_TH_TA_2040": "TA_data","AVIA_TH_TU_2040": "TU_data","AVIA_TH_VU_2040": "VU_data"}
)

# add columns and set them as index
demand_R4_R9_CH["years"] = "2040"
demand_R4_R9_CH["techs"] = "Demand"
demand_R4_R9_CH["commodity"] = "T_Aviation_th"
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
        [m.set.nodesdata, m.set.yearssel, ["Demand"], ["T_Aviation_th"]]
    )
)
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesFixedProfile"] = 1
sourcesink_config = sourcesink_config.dropna()

m.parameter.add(sourcesink_config, "sourcesink_config")
sourcesink_config
###############################################################################
demand_R4_R9_CH = profiles[["AVIA_EL_CI_2040", "AVIA_EL_FJ_2040","AVIA_EL_FSM_2040", "AVIA_EL_KB_2040", "AVIA_EL_MI_2040","AVIA_EL_NU_2040","AVIA_EL_NE_2040","AVIA_EL_PU_2040","AVIA_EL_PNG_2040","AVIA_EL_SA_2040","AVIA_EL_SI_2040","AVIA_EL_TA_2040","AVIA_EL_TU_2040","AVIA_EL_VU_2040"]]

demand_R4_R9_CH = demand_R4_R9_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R9_CH = demand_R4_R9_CH.T

demand_R4_R9_CH = demand_R4_R9_CH.rename(
    index={"AVIA_EL_CI_2040": "CI_data", "AVIA_EL_FJ_2040": "FJ_data", "AVIA_EL_FSM_2040": "FSM_data", "AVIA_EL_KB_2040": "KB_data", "AVIA_EL_MI_2040": "MI_data","AVIA_EL_NU_2040": "NU_data","AVIA_EL_NE_2040": "NE_data","AVIA_EL_PU_2040": "PU_data","AVIA_EL_PNG_2040": "PNG_data","AVIA_EL_SA_2040": "SA_data","AVIA_EL_SI_2040": "SI_data","AVIA_EL_TA_2040": "TA_data","AVIA_EL_TU_2040": "TU_data","AVIA_EL_VU_2040": "VU_data"}
)

# add columns and set them as index
demand_R4_R9_CH["years"] = "2040"
demand_R4_R9_CH["techs"] = "Demand"
demand_R4_R9_CH["commodity"] = "eKerosene"
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
        [m.set.nodesdata, m.set.yearssel, ["Demand"], ["eKerosene"]]
    )
)
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesFixedProfile"] = 1
sourcesink_config = sourcesink_config.dropna()

m.parameter.add(sourcesink_config, "sourcesink_config")
sourcesink_config
###################################################################################
demand_R4_R9_CH = profiles[["AVIA_E_CI_2040", "AVIA_E_FJ_2040","AVIA_E_FSM_2040", "AVIA_E_KB_2040", "AVIA_E_MI_2040","AVIA_E_NU_2040","AVIA_E_NE_2040","AVIA_E_PU_2040","AVIA_E_PNG_2040","AVIA_E_SA_2040","AVIA_E_SI_2040","AVIA_E_TA_2040","AVIA_E_TU_2040","AVIA_E_VU_2040"]]

demand_R4_R9_CH = demand_R4_R9_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R9_CH = demand_R4_R9_CH.T

demand_R4_R9_CH = demand_R4_R9_CH.rename(
    index={"AVIA_E_CI_2040": "CI_data", "AVIA_E_FJ_2040": "FJ_data", "AVIA_E_FSM_2040": "FSM_data", "AVIA_E_KB_2040": "KB_data", "AVIA_E_MI_2040": "MI_data","AVIA_E_NU_2040": "NU_data","AVIA_E_NE_2040": "NE_data","AVIA_E_PU_2040": "PU_data","AVIA_E_PNG_2040": "PNG_data","AVIA_E_SA_2040": "SA_data","AVIA_E_SI_2040": "SI_data","AVIA_E_TA_2040": "TA_data","AVIA_E_TU_2040": "TU_data","AVIA_E_VU_2040": "VU_data"}
)

# add columns and set them as index
demand_R4_R9_CH["years"] = "2040"
demand_R4_R9_CH["techs"] = "Demand"
demand_R4_R9_CH["commodity"] = "T_Aviation_el"
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
        [m.set.nodesdata, m.set.yearssel, ["Demand"], ["T_Aviation_el"]]
    )
)
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesFixedProfile"] = 1
sourcesink_config = sourcesink_config.dropna()

m.parameter.add(sourcesink_config, "sourcesink_config")
sourcesink_config
###################################################################################

#################################################################################
demand_R4_R10_CH = profiles[["HC_B_CI_2040", "HC_B_FJ_2040","HC_B_FSM_2040", "HC_B_KB_2040", "HC_B_MI_2040","HC_B_NU_2040","HC_B_NE_2040","HC_B_PU_2040","HC_B_PNG_2040","HC_B_SA_2040","HC_B_SI_2040","HC_B_TA_2040","HC_B_TU_2040","HC_B_VU_2040"]]

demand_R4_R10_CH = demand_R4_R10_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R10_CH = demand_R4_R10_CH.T

demand_R4_R10_CH = demand_R4_R10_CH.rename(
    index={"HC_B_CI_2040": "CI_data", "HC_B_FJ_2040": "FJ_data", "HC_B_FSM_2040": "FSM_data", "HC_B_KB_2040": "KB_data", "HC_B_MI_2040": "MI_data","HC_B_NU_2040": "NU_data","HC_B_NE_2040": "NE_data","HC_B_PU_2040": "PU_data","HC_B_PNG_2040": "PNG_data","HC_B_SA_2040": "SA_data","HC_B_SI_2040": "SI_data","HC_B_TA_2040": "TA_data","HC_B_TU_2040": "TU_data","HC_B_VU_2040": "VU_data"}
)

# add columns and set them as index
demand_R4_R10_CH["years"] = "2040"
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
demand_R4_R10_CH = profiles[["HC_L_CI_2040", "HC_L_FJ_2040","HC_L_FSM_2040", "HC_L_KB_2040", "HC_L_MI_2040","HC_L_NU_2040","HC_L_NE_2040","HC_L_PU_2040","HC_L_PNG_2040","HC_L_SA_2040","HC_L_SI_2040","HC_L_TA_2040","HC_L_TU_2040","HC_L_VU_2040"]]

demand_R4_R10_CH = demand_R4_R10_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R10_CH = demand_R4_R10_CH.T

demand_R4_R10_CH = demand_R4_R10_CH.rename(
    index={"HC_L_CI_2040": "CI_data", "HC_L_FJ_2040": "FJ_data", "HC_L_FSM_2040": "FSM_data", "HC_L_KB_2040": "KB_data", "HC_L_MI_2040": "MI_data","HC_L_NU_2040": "NU_data","HC_L_NE_2040": "NE_data","HC_L_PU_2040": "PU_data","HC_L_PNG_2040": "PNG_data","HC_L_SA_2040": "SA_data","HC_L_SI_2040": "SI_data","HC_L_TA_2040": "TA_data","HC_L_TU_2040": "TU_data","HC_L_VU_2040": "VU_data"}
)

# add columns and set them as index
demand_R4_R10_CH["years"] = "2040"
demand_R4_R10_CH["techs"] = "Demand"
demand_R4_R10_CH["commodity"] = "T_cook_LPG"
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
        [m.set.nodesdata, m.set.yearssel, ["Demand"], ["T_cook_LPG"]]
    )
)
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesFixedProfile"] = 1
sourcesink_config = sourcesink_config.dropna()

m.parameter.add(sourcesink_config, "sourcesink_config")
sourcesink_config
############################################################
demand_R4_R10_CH = profiles[["HC_el_CI_2040", "HC_el_FJ_2040","HC_el_FSM_2040", "HC_el_KB_2040", "HC_el_MI_2040","HC_el_NU_2040","HC_el_NE_2040","HC_el_PU_2040","HC_el_PNG_2040","HC_el_SA_2040","HC_el_SI_2040","HC_el_TA_2040","HC_el_TU_2040","HC_el_VU_2040"]]

demand_R4_R10_CH = demand_R4_R10_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R10_CH = demand_R4_R10_CH.T

demand_R4_R10_CH = demand_R4_R10_CH.rename(
    index={"HC_el_CI_2040": "CI_data", "HC_el_FJ_2040": "FJ_data", "HC_el_FSM_2040": "FSM_data", "HC_el_KB_2040": "KB_data", "HC_el_MI_2040": "MI_data","HC_el_NU_2040": "NU_data","HC_el_NE_2040": "NE_data","HC_el_PU_2040": "PU_data","HC_el_PNG_2040": "PNG_data","HC_el_SA_2040": "SA_data","HC_el_SI_2040": "SI_data","HC_el_TA_2040": "TA_data","HC_el_TU_2040": "TU_data","HC_el_VU_2040": "VU_data"}
)

# add columns and set them as index
demand_R4_R10_CH["years"] = "2040"
demand_R4_R10_CH["techs"] = "Demand"
demand_R4_R10_CH["commodity"] = "T_cook_el"
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
        [m.set.nodesdata, m.set.yearssel, ["Demand"], ["T_cook_el"]]
    )
)
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesFixedProfile"] = 1
sourcesink_config = sourcesink_config.dropna()

m.parameter.add(sourcesink_config, "sourcesink_config")
sourcesink_config
###########################################################
demand_R4_R10_CH = profiles[["HI_D_CI_2040", "HI_D_FJ_2040","HI_D_FSM_2040", "HI_D_KB_2040", "HI_D_MI_2040","HI_D_NU_2040","HI_D_NE_2040","HI_D_PU_2040","HI_D_PNG_2040","HI_D_SA_2040","HI_D_SI_2040","HI_D_TA_2040","HI_D_TU_2040","HI_D_VU_2040"]]

demand_R4_R10_CH = demand_R4_R10_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R10_CH = demand_R4_R10_CH.T

demand_R4_R10_CH = demand_R4_R10_CH.rename(
    index={"HI_D_CI_2040": "CI_data", "HI_D_FJ_2040": "FJ_data", "HI_D_FSM_2040": "FSM_data", "HI_D_KB_2040": "KB_data", "HI_D_MI_2040": "MI_data","HI_D_NU_2040": "NU_data","HI_D_NE_2040": "NE_data","HI_D_PU_2040": "PU_data","HI_D_PNG_2040": "PNG_data","HI_D_SA_2040": "SA_data","HI_D_SI_2040": "SI_data","HI_D_TA_2040": "TA_data","HI_D_TU_2040": "TU_data","HI_D_VU_2040": "VU_data"}
)

# add columns and set them as index
demand_R4_R10_CH["years"] = "2040"
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
demand_R4_R10_CH = profiles[["HI_EH_CI_2040", "HI_EH_FJ_2040","HI_EH_FSM_2040", "HI_EH_KB_2040", "HI_EH_MI_2040","HI_EH_NU_2040","HI_EH_NE_2040","HI_EH_PU_2040","HI_EH_PNG_2040","HI_EH_SA_2040","HI_EH_SI_2040","HI_EH_TA_2040","HI_EH_TU_2040","HI_EH_VU_2040"]]

demand_R4_R10_CH = demand_R4_R10_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R10_CH = demand_R4_R10_CH.T

demand_R4_R10_CH = demand_R4_R10_CH.rename(
    index={"HI_EH_CI_2040": "CI_data", "HI_EH_FJ_2040": "FJ_data", "HI_EH_FSM_2040": "FSM_data", "HI_EH_KB_2040": "KB_data", "HI_EH_MI_2040": "MI_data","HI_EH_NU_2040": "NU_data","HI_EH_NE_2040": "NE_data","HI_EH_PU_2040": "PU_data","HI_EH_PNG_2040": "PNG_data","HI_EH_SA_2040": "SA_data","HI_EH_SI_2040": "SI_data","HI_EH_TA_2040": "TA_data","HI_EH_TU_2040": "TU_data","HI_EH_VU_2040": "VU_data"}
)

# add columns and set them as index
demand_R4_R10_CH["years"] = "2040"
demand_R4_R10_CH["techs"] = "Demand"
demand_R4_R10_CH["commodity"] = "T_Industry_EH"
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
        [m.set.nodesdata, m.set.yearssel, ["Demand"], ["T_Industry_EH"]]
    )
)
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesFixedProfile"] = 1
sourcesink_config = sourcesink_config.dropna()

m.parameter.add(sourcesink_config, "sourcesink_config")
sourcesink_config
##################################################################
demand_R4_R10_CH = profiles[["DHW_E_CI_2040", "DHW_E_FJ_2040","DHW_E_FSM_2040", "DHW_E_KB_2040", "DHW_E_MI_2040","DHW_E_NU_2040","DHW_E_NE_2040","DHW_E_PU_2040","DHW_E_PNG_2040","DHW_E_SA_2040","DHW_E_SI_2040","DHW_E_TA_2040","DHW_E_TU_2040","DHW_E_VU_2040"]]

demand_R4_R10_CH = demand_R4_R10_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R10_CH = demand_R4_R10_CH.T

demand_R4_R10_CH = demand_R4_R10_CH.rename(
    index={"DHW_E_CI_2040": "CI_data", "DHW_E_FJ_2040": "FJ_data", "DHW_E_FSM_2040": "FSM_data", "DHW_E_KB_2040": "KB_data", "DHW_E_MI_2040": "MI_data","DHW_E_NU_2040": "NU_data","DHW_E_NE_2040": "NE_data","DHW_E_PU_2040": "PU_data","DHW_E_PNG_2040": "PNG_data","DHW_E_SA_2040": "SA_data","DHW_E_SI_2040": "SI_data","DHW_E_TA_2040": "TA_data","DHW_E_TU_2040": "TU_data","DHW_E_VU_2040": "VU_data"}
)

# add columns and set them as index
demand_R4_R10_CH["years"] = "2040"
demand_R4_R10_CH["techs"] = "Demand"
demand_R4_R10_CH["commodity"] = "DHW_el"
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
        [m.set.nodesdata, m.set.yearssel, ["Demand"], ["DHW_el"]]
    )
)
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesFixedProfile"] = 1
sourcesink_config = sourcesink_config.dropna()

m.parameter.add(sourcesink_config, "sourcesink_config")
sourcesink_config
#############################################################
demand_R4_R10_CH = profiles[["DHW_L_CI_2040", "DHW_L_FJ_2040","DHW_L_FSM_2040", "DHW_L_KB_2040", "DHW_L_MI_2040","DHW_L_NU_2040","DHW_L_NE_2040","DHW_L_PU_2040","DHW_L_PNG_2040","DHW_L_SA_2040","DHW_L_SI_2040","DHW_L_TA_2040","DHW_L_TU_2040","DHW_L_VU_2040"]]

demand_R4_R10_CH = demand_R4_R10_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R10_CH = demand_R4_R10_CH.T

demand_R4_R10_CH = demand_R4_R10_CH.rename(
    index={"DHW_L_CI_2040": "CI_data", "DHW_L_FJ_2040": "FJ_data", "DHW_L_FSM_2040": "FSM_data", "DHW_L_KB_2040": "KB_data", "DHW_L_MI_2040": "MI_data","DHW_L_NU_2040": "NU_data","DHW_L_NE_2040": "NE_data","DHW_L_PU_2040": "PU_data","DHW_L_PNG_2040": "PNG_data","DHW_L_SA_2040": "SA_data","DHW_L_SI_2040": "SI_data","DHW_L_TA_2040": "TA_data","DHW_L_TU_2040": "TU_data","DHW_L_VU_2040": "VU_data"}
)

# add columns and set them as index
demand_R4_R10_CH["years"] = "2040"
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
####################################################################################
# demand_R4_R10_CH = profiles[["DHW_ST_CI_2040", "DHW_ST_FJ_2040","DHW_ST_FSM_2040", "DHW_ST_KB_2040", "DHW_ST_MI_2040","DHW_ST_NU_2040","DHW_ST_NE_2040","DHW_ST_PU_2040","DHW_ST_PNG_2040","DHW_ST_SA_2040","DHW_ST_SI_2040","DHW_ST_TA_2040","DHW_ST_TU_2040","DHW_ST_VU_2040"]]

# demand_R4_R10_CH = demand_R4_R10_CH.div(1e3).mul(-1)
# # transpose DataFrame for needed format
# demand_R4_R10_CH = demand_R4_R10_CH.T

# demand_R4_R10_CH = demand_R4_R10_CH.rename(
#     index={"DHW_ST_CI_2040": "CI_data", "DHW_ST_FJ_2040": "FJ_data", "DHW_ST_FSM_2040": "FSM_data", "DHW_ST_KB_2040": "KB_data", "DHW_ST_MI_2040": "MI_data","DHW_ST_NU_2040": "NU_data","DHW_ST_NE_2040": "NE_data","DHW_ST_PU_2040": "PU_data","DHW_ST_PNG_2040": "PNG_data","DHW_ST_SA_2040": "SA_data","DHW_ST_SI_2040": "SI_data","DHW_ST_TA_2040": "TA_data","DHW_ST_TU_2040": "TU_data","DHW_ST_VU_2040": "VU_data"}
# )

# # add columns and set them as index
# demand_R4_R10_CH["years"] = "2040"
# demand_R4_R10_CH["techs"] = "Demand"
# demand_R4_R10_CH["commodity"] = "DHW_el"
# demand_R4_R10_CH["type"] = "fixed"
# demand_R4_R10_CH = demand_R4_R10_CH.set_index(
#     ["years", "techs", "commodity", "type"], append=True
# )

# m.profile.add(demand_R4_R10_CH, "sourcesink_profile")
# demand_R4_R10_CH.iloc[:, 0:8]

# # load the profiles DataFrame, select the demand column
# # %% [markdown]
# # Now that we have created the profile, we need to create a config with the
# # information that the created profile is going to be integrated into the model
# # as fixed profile.

# # %%
# # "sourcesink_config" (demand configuration)
# sourcesink_config = pd.DataFrame(
#     index=pd.MultiIndex.from_product(
#         [m.set.nodesdata, m.set.yearssel, ["Demand"], ["DHW_el"]]
#     )
# )
# sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesFixedProfile"] = 1
# sourcesink_config = sourcesink_config.dropna()

# m.parameter.add(sourcesink_config, "sourcesink_config")
# sourcesink_config
#####################################################################  2050        ###################################################################################################################
demand_R4_R21_CH = profiles[["demand_CI_2050", "demand_FJ_2050","demand_FSM_2050", "demand_KB_2050", "demand_MI_2050","demand_NU_2050","demand_NE_2050","demand_PU_2050","demand_PNG_2050","demand_SA_2050","demand_SI_2050","demand_TA_2050","demand_TU_2050","demand_VU_2050"]]

demand_R4_R21_CH = demand_R4_R21_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R21_CH = demand_R4_R21_CH.T

demand_R4_R21_CH = demand_R4_R21_CH.rename(
    index={"demand_CI_2050": "CI_data", "demand_FJ_2050": "FJ_data", "demand_FSM_2050": "FSM_data", "demand_KB_2050": "KB_data", "demand_MI_2050": "MI_data","demand_NU_2050": "NU_data","demand_NE_2050": "NE_data","demand_PU_2050": "PU_data","demand_PNG_2050": "PNG_data","demand_SA_2050": "SA_data","demand_SI_2050": "SI_data","demand_TA_2050": "TA_data","demand_TU_2050": "TU_data","demand_VU_2050": "VU_data"}
)

# add columns and set them as index
demand_R4_R21_CH["years"] = "2050"
demand_R4_R21_CH["techs"] = "Demand"
demand_R4_R21_CH["commodity"] = "Elec"
demand_R4_R21_CH["type"] = "fixed"
demand_R4_R21_CH = demand_R4_R21_CH.set_index(
    ["years", "techs", "commodity", "type"], append=True
)

m.profile.add(demand_R4_R21_CH, "sourcesink_profile")
demand_R4_R21_CH.iloc[:, 0:8]

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
#################################################################################################
demand_R4_R3_CH = profiles[["MDV_el_CI_2050","MDV_el_FJ_2050","MDV_el_FSM_2050","MDV_el_KB_2050","MDV_el_MI_2050","MDV_el_NU_2050","MDV_el_NE_2050","MDV_el_PU_2050","MDV_el_PNG_2050","MDV_el_SA_2050","MDV_el_SI_2050","MDV_el_TA_2050","MDV_el_TU_2050","MDV_el_VU_2050"]]

demand_R4_R3_CH = demand_R4_R3_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R3_CH = demand_R4_R3_CH.T

demand_R4_R3_CH = demand_R4_R3_CH.rename(
    index={"MDV_el_CI_2050": "CI_data", "MDV_el_FJ_2050": "FJ_data", "MDV_el_FSM_2050": "FSM_data", "MDV_el_KB_2050": "KB_data", "MDV_el_MI_2050": "MI_data","MDV_el_NU_2050": "NU_data","MDV_el_NE_2050": "NE_data","MDV_el_PU_2050": "PU_data","MDV_el_PNG_2050": "PNG_data","MDV_el_SA_2050": "SA_data","MDV_el_SI_2050": "SI_data","MDV_el_TA_2050": "TA_data","MDV_el_TU_2050": "TU_data","MDV_el_VU_2050": "VU_data"}
)

# add columns and set them as index
demand_R4_R3_CH["years"] = "2050"
demand_R4_R3_CH["techs"] = "Demand"
demand_R4_R3_CH["commodity"] = "T_MDV_el"
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
        [m.set.nodesdata, m.set.yearssel, ["Demand"], ["T_MDV_el"]]
    )
)
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesFixedProfile"] = 1
sourcesink_config = sourcesink_config.dropna()

m.parameter.add(sourcesink_config, "sourcesink_config")
sourcesink_config
# ####################################################################################
demand_R4_R3_CH = profiles[["MDV_Th_CI_2050","MDV_Th_FJ_2050","MDV_Th_FSM_2050","MDV_Th_KB_2050","MDV_Th_MI_2050","MDV_Th_NU_2050","MDV_Th_NE_2050","MDV_Th_PU_2050","MDV_Th_PNG_2050","MDV_Th_SA_2050","MDV_Th_SI_2050","MDV_Th_TA_2050","MDV_Th_TU_2050","MDV_Th_VU_2050"]]

demand_R4_R3_CH = demand_R4_R3_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R3_CH = demand_R4_R3_CH.T

demand_R4_R3_CH = demand_R4_R3_CH.rename(
    index={"MDV_Th_CI_2050": "CI_data", "MDV_Th_FJ_2050": "FJ_data", "MDV_Th_FSM_2050": "FSM_data", "MDV_Th_KB_2050": "KB_data", "MDV_Th_MI_2050": "MI_data","MDV_Th_NU_2050": "NU_data","MDV_Th_NE_2050": "NE_data","MDV_Th_PU_2050": "PU_data","MDV_Th_PNG_2050": "PNG_data","MDV_Th_SA_2050": "SA_data","MDV_Th_SI_2050": "SI_data","MDV_Th_TA_2050": "TA_data","MDV_Th_TU_2050": "TU_data","MDV_Th_VU_2050": "VU_data"}
)

# add columns and set them as index
demand_R4_R3_CH["years"] = "2050"
demand_R4_R3_CH["techs"] = "Demand"
demand_R4_R3_CH["commodity"] = "T_MDV_th"
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

# # %%
# # "sourcesink_config" (demand configuration)
sourcesink_config = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [m.set.nodesdata, m.set.yearssel, ["Demand"], ["T_MDV_th"]]
    )
)
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesFixedProfile"] = 1
sourcesink_config = sourcesink_config.dropna()

m.parameter.add(sourcesink_config, "sourcesink_config")
sourcesink_config
# ########################################################
demand_R4_R3_CH = profiles[["MDV_BF_CI_2050","MDV_BF_FJ_2050","MDV_BF_FSM_2050","MDV_BF_KB_2050","MDV_BF_MI_2050","MDV_BF_NU_2050","MDV_BF_NE_2050","MDV_BF_PU_2050","MDV_BF_PNG_2050","MDV_BF_SA_2050","MDV_BF_SI_2050","MDV_BF_TA_2050","MDV_BF_TU_2050","MDV_BF_VU_2050"]]

demand_R4_R3_CH = demand_R4_R3_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R3_CH = demand_R4_R3_CH.T

demand_R4_R3_CH = demand_R4_R3_CH.rename(
    index={"MDV_BF_CI_2050": "CI_data", "MDV_BF_FJ_2050": "FJ_data", "MDV_BF_FSM_2050": "FSM_data", "MDV_BF_KB_2050": "KB_data", "MDV_BF_MI_2050": "MI_data","MDV_BF_NU_2050": "NU_data","MDV_BF_NE_2050": "NE_data","MDV_BF_PU_2050": "PU_data","MDV_BF_PNG_2050": "PNG_data","MDV_BF_SA_2050": "SA_data","MDV_BF_SI_2050": "SI_data","MDV_BF_TA_2050": "TA_data","MDV_BF_TU_2050": "TU_data","MDV_BF_VU_2050": "VU_data"}
)

# add columns and set them as index
demand_R4_R3_CH["years"] = "2050"
demand_R4_R3_CH["techs"] = "Demand"
demand_R4_R3_CH["commodity"] = "T_MDV_BF"
demand_R4_R3_CH["type"] = "fixed"
demand_R4_R3_CH = demand_R4_R3_CH.set_index(
    ["years", "techs", "commodity", "type"], append=True
)

m.profile.add(demand_R4_R3_CH, "sourcesink_profile")
demand_R4_R3_CH.iloc[:, 0:8]

# # load the profiles DataFrame, select the demand column
# # %% [markdown]
# # Now that we have created the profile, we need to create a config with the
# # information that the created profile is going to be integrated into the model
# # as fixed profile.

# # %%
# # "sourcesink_config" (demand configuration)
sourcesink_config = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [m.set.nodesdata, m.set.yearssel, ["Demand"], ["T_MDV_BF"]]
    )
)
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesFixedProfile"] = 1
sourcesink_config = sourcesink_config.dropna()

m.parameter.add(sourcesink_config, "sourcesink_config")
sourcesink_config
# #####################################################################################HDV######################################################
demand_R4_R3_CH = profiles[["HDV_el_CI_2050","HDV_el_FJ_2050","HDV_el_FSM_2050","HDV_el_KB_2050","HDV_el_MI_2050","HDV_el_NU_2050","HDV_el_NE_2050","HDV_el_PU_2050","HDV_el_PNG_2050","HDV_el_SA_2050","HDV_el_SI_2050","HDV_el_TA_2050","HDV_el_TU_2050","HDV_el_VU_2050"]]

demand_R4_R3_CH = demand_R4_R3_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R3_CH = demand_R4_R3_CH.T

demand_R4_R3_CH = demand_R4_R3_CH.rename(
    index={"HDV_el_CI_2050": "CI_data", "HDV_el_FJ_2050": "FJ_data", "HDV_el_FSM_2050": "FSM_data", "HDV_el_KB_2050": "KB_data", "HDV_el_MI_2050": "MI_data","HDV_el_NU_2050": "NU_data","HDV_el_NE_2050": "NE_data","HDV_el_PU_2050": "PU_data","HDV_el_PNG_2050": "PNG_data","HDV_el_SA_2050": "SA_data","HDV_el_SI_2050": "SI_data","HDV_el_TA_2050": "TA_data","HDV_el_TU_2050": "TU_data","HDV_el_VU_2050": "VU_data"}
)

# add columns and set them as index
demand_R4_R3_CH["years"] = "2050"
demand_R4_R3_CH["techs"] = "Demand"
demand_R4_R3_CH["commodity"] = "T_HDV_el"
demand_R4_R3_CH["type"] = "fixed"
demand_R4_R3_CH = demand_R4_R3_CH.set_index(
    ["years", "techs", "commodity", "type"], append=True
)

m.profile.add(demand_R4_R3_CH, "sourcesink_profile")
demand_R4_R3_CH.iloc[:, 0:8]

# # load the profiles DataFrame, select the demand column
# # %% [markdown]
# # Now that we have created the profile, we need to create a config with the
# # information that the created profile is going to be integrated into the model
# # as fixed profile.

# # %%
# # "sourcesink_config" (demand configuration)
sourcesink_config = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [m.set.nodesdata, m.set.yearssel, ["Demand"], ["T_HDV_el"]]
    )
)
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesFixedProfile"] = 1
sourcesink_config = sourcesink_config.dropna()

m.parameter.add(sourcesink_config, "sourcesink_config")
sourcesink_config
# ####################################################################################
demand_R4_R3_CH = profiles[["HDV_Th_CI_2050","HDV_Th_FJ_2050","HDV_Th_FSM_2050","HDV_Th_KB_2050","HDV_Th_MI_2050","HDV_Th_NU_2050","HDV_Th_NE_2050","HDV_Th_PU_2050","HDV_Th_PNG_2050","HDV_Th_SA_2050","HDV_Th_SI_2050","HDV_Th_TA_2050","HDV_Th_TU_2050","HDV_Th_VU_2050"]]

demand_R4_R3_CH = demand_R4_R3_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R3_CH = demand_R4_R3_CH.T

demand_R4_R3_CH = demand_R4_R3_CH.rename(
    index={"HDV_Th_CI_2050": "CI_data", "HDV_Th_FJ_2050": "FJ_data", "HDV_Th_FSM_2050": "FSM_data", "HDV_Th_KB_2050": "KB_data", "HDV_Th_MI_2050": "MI_data","HDV_Th_NU_2050": "NU_data","HDV_Th_NE_2050": "NE_data","HDV_Th_PU_2050": "PU_data","HDV_Th_PNG_2050": "PNG_data","HDV_Th_SA_2050": "SA_data","HDV_Th_SI_2050": "SI_data","HDV_Th_TA_2050": "TA_data","HDV_Th_TU_2050": "TU_data","HDV_Th_VU_2050": "VU_data"}
)

# add columns and set them as index
demand_R4_R3_CH["years"] = "2050"
demand_R4_R3_CH["techs"] = "Demand"
demand_R4_R3_CH["commodity"] = "T_HDV_th"
demand_R4_R3_CH["type"] = "fixed"
demand_R4_R3_CH = demand_R4_R3_CH.set_index(
    ["years", "techs", "commodity", "type"], append=True
)

m.profile.add(demand_R4_R3_CH, "sourcesink_profile")
demand_R4_R3_CH.iloc[:, 0:8]

# # load the profiles DataFrame, select the demand column
# # %% [markdown]
# # Now that we have created the profile, we need to create a config with the
# # information that the created profile is going to be integrated into the model
# # as fixed profile.

# # %%
# # "sourcesink_config" (demand configuration)
sourcesink_config = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [m.set.nodesdata, m.set.yearssel, ["Demand"], ["T_HDV_th"]]
    )
)
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesFixedProfile"] = 1
sourcesink_config = sourcesink_config.dropna()

m.parameter.add(sourcesink_config, "sourcesink_config")
sourcesink_config
# ########################################################
demand_R4_R3_CH = profiles[["HDV_BF_CI_2050","HDV_BF_FJ_2050","HDV_BF_FSM_2050","HDV_BF_KB_2050","HDV_BF_MI_2050","HDV_BF_NU_2050","HDV_BF_NE_2050","HDV_BF_PU_2050","HDV_BF_PNG_2050","HDV_BF_SA_2050","HDV_BF_SI_2050","HDV_BF_TA_2050","HDV_BF_TU_2050","HDV_BF_VU_2050"]]

demand_R4_R3_CH = demand_R4_R3_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R3_CH = demand_R4_R3_CH.T

demand_R4_R3_CH = demand_R4_R3_CH.rename(
    index={"HDV_BF_CI_2050": "CI_data", "HDV_BF_FJ_2050": "FJ_data", "HDV_BF_FSM_2050": "FSM_data", "HDV_BF_KB_2050": "KB_data", "HDV_BF_MI_2050": "MI_data","HDV_BF_NU_2050": "NU_data","HDV_BF_NE_2050": "NE_data","HDV_BF_PU_2050": "PU_data","HDV_BF_PNG_2050": "PNG_data","HDV_BF_SA_2050": "SA_data","HDV_BF_SI_2050": "SI_data","HDV_BF_TA_2050": "TA_data","HDV_BF_TU_2050": "TU_data","HDV_BF_VU_2050": "VU_data"}
)

# add columns and set them as index
demand_R4_R3_CH["years"] = "2050"
demand_R4_R3_CH["techs"] = "Demand"
demand_R4_R3_CH["commodity"] = "T_HDV_BF"
demand_R4_R3_CH["type"] = "fixed"
demand_R4_R3_CH = demand_R4_R3_CH.set_index(
    ["years", "techs", "commodity", "type"], append=True
)

m.profile.add(demand_R4_R3_CH, "sourcesink_profile")
demand_R4_R3_CH.iloc[:, 0:8]

# # load the profiles DataFrame, select the demand column
# # %% [markdown]
# # Now that we have created the profile, we need to create a config with the
# # information that the created profile is going to be integrated into the model
# # as fixed profile.

# # %%
# # "sourcesink_config" (demand configuration)
sourcesink_config = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [m.set.nodesdata, m.set.yearssel, ["Demand"], ["T_HDV_BF"]]
    )
)
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesFixedProfile"] = 1
sourcesink_config = sourcesink_config.dropna()

m.parameter.add(sourcesink_config, "sourcesink_config")
sourcesink_config


# #############################################################
demand_R4_R3_CH = profiles[["LDV_el_CI_2050","LDV_el_FJ_2050","LDV_el_FSM_2050","LDV_el_KB_2050","LDV_el_MI_2050","LDV_el_NU_2050","LDV_el_NE_2050","LDV_el_PU_2050","LDV_el_PNG_2050","LDV_el_SA_2050","LDV_el_SI_2050","LDV_el_TA_2050","LDV_el_TU_2050","LDV_el_VU_2050"]]

demand_R4_R3_CH = demand_R4_R3_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R3_CH = demand_R4_R3_CH.T

demand_R4_R3_CH = demand_R4_R3_CH.rename(
    index={"LDV_el_CI_2050": "CI_data", "LDV_el_FJ_2050": "FJ_data", "LDV_el_FSM_2050": "FSM_data", "LDV_el_KB_2050": "KB_data", "LDV_el_MI_2050": "MI_data","LDV_el_NU_2050": "NU_data","HDV_el_NE_2050": "NE_data","LDV_el_PU_2050": "PU_data","LDV_el_PNG_2050": "PNG_data","LDV_el_SA_2050": "SA_data","LDV_el_SI_2050": "SI_data","LDV_el_TA_2050": "TA_data","LDV_el_TU_2050": "TU_data","LDV_el_VU_2050": "VU_data"}
)

# add columns and set them as index
demand_R4_R3_CH["years"] = "2050"
demand_R4_R3_CH["techs"] = "Demand"
demand_R4_R3_CH["commodity"] = "T_LDV_el"
demand_R4_R3_CH["type"] = "fixed"
demand_R4_R3_CH = demand_R4_R3_CH.set_index(
    ["years", "techs", "commodity", "type"], append=True
)

m.profile.add(demand_R4_R3_CH, "sourcesink_profile")
demand_R4_R3_CH.iloc[:, 0:8]

# # load the profiles DataFrame, select the demand column
# # %% [markdown]
# # Now that we have created the profile, we need to create a config with the
# # information that the created profile is going to be integrated into the model
# # as fixed profile.

# # %%
# # "sourcesink_config" (demand configuration)
sourcesink_config = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [m.set.nodesdata, m.set.yearssel, ["Demand"], ["T_LDV_el"]]
    )
)
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesFixedProfile"] = 1
sourcesink_config = sourcesink_config.dropna()

m.parameter.add(sourcesink_config, "sourcesink_config")
sourcesink_config
# ####################################################################################
demand_R4_R3_CH = profiles[["LDV_Th_CI_2050","LDV_Th_FJ_2050","LDV_Th_FSM_2050","LDV_Th_KB_2050","LDV_Th_MI_2050","LDV_Th_NU_2050","LDV_Th_NE_2050","LDV_Th_PU_2050","LDV_Th_PNG_2050","LDV_Th_SA_2050","LDV_Th_SI_2050","LDV_Th_TA_2050","LDV_Th_TU_2050","LDV_Th_VU_2050"]]

demand_R4_R3_CH = demand_R4_R3_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R3_CH = demand_R4_R3_CH.T

demand_R4_R3_CH = demand_R4_R3_CH.rename(
    index={"LDV_Th_CI_2050": "CI_data", "LDV_Th_FJ_2050": "FJ_data", "LDV_Th_FSM_2050": "FSM_data", "LDV_Th_KB_2050": "KB_data", "LDV_Th_MI_2050": "MI_data","LDV_Th_NU_2050": "NU_data","LDV_Th_NE_2050": "NE_data","LDV_Th_PU_2050": "PU_data","LDV_Th_PNG_2050": "PNG_data","LDV_Th_SA_2050": "SA_data","LDV_Th_SI_2050": "SI_data","LDV_Th_TA_2050": "TA_data","LDV_Th_TU_2050": "TU_data","LDV_Th_VU_2050": "VU_data"}
)

# add columns and set them as index
demand_R4_R3_CH["years"] = "2050"
demand_R4_R3_CH["techs"] = "Demand"
demand_R4_R3_CH["commodity"] = "T_LDV_th"
demand_R4_R3_CH["type"] = "fixed"
demand_R4_R3_CH = demand_R4_R3_CH.set_index(
    ["years", "techs", "commodity", "type"], append=True
)

m.profile.add(demand_R4_R3_CH, "sourcesink_profile")
demand_R4_R3_CH.iloc[:, 0:8]

# # load the profiles DataFrame, select the demand column
# # %% [markdown]
# # Now that we have created the profile, we need to create a config with the
# # information that the created profile is going to be integrated into the model
# # as fixed profile.

# # %%
# # "sourcesink_config" (demand configuration)
sourcesink_config = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [m.set.nodesdata, m.set.yearssel, ["Demand"], ["T_LDV_th"]]
    )
)
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesFixedProfile"] = 1
sourcesink_config = sourcesink_config.dropna()

m.parameter.add(sourcesink_config, "sourcesink_config")
sourcesink_config
# ########################################################
demand_R4_R3_CH = profiles[["LDV_BF_CI_2050","LDV_BF_FJ_2050","LDV_BF_FSM_2050","LDV_BF_KB_2050","LDV_BF_MI_2050","LDV_BF_NU_2050","LDV_BF_NE_2050","LDV_BF_PU_2050","LDV_BF_PNG_2050","LDV_BF_SA_2050","LDV_BF_SI_2050","LDV_BF_TA_2050","LDV_BF_TU_2050","LDV_BF_VU_2050"]]

demand_R4_R3_CH = demand_R4_R3_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R3_CH = demand_R4_R3_CH.T

demand_R4_R3_CH = demand_R4_R3_CH.rename(
    index={"LDV_BF_CI_2050": "CI_data", "LDV_BF_FJ_2050": "FJ_data", "LDV_BF_FSM_2050": "FSM_data", "LDV_BF_KB_2050": "KB_data", "LDV_BF_MI_2050": "MI_data","LDV_BF_NU_2050": "NU_data","LDV_BF_NE_2050": "NE_data","LDV_BF_PU_2050": "PU_data","LDV_BF_PNG_2050": "PNG_data","LDV_BF_SA_2050": "SA_data","LDV_BF_SI_2050": "SI_data","LDV_BF_TA_2050": "TA_data","LDV_BF_TU_2050": "TU_data","LDV_BF_VU_2050": "VU_data"}
)

# add columns and set them as index
demand_R4_R3_CH["years"] = "2050"
demand_R4_R3_CH["techs"] = "Demand"
demand_R4_R3_CH["commodity"] = "T_LDV_BF"
demand_R4_R3_CH["type"] = "fixed"
demand_R4_R3_CH = demand_R4_R3_CH.set_index(
    ["years", "techs", "commodity", "type"], append=True
)

m.profile.add(demand_R4_R3_CH, "sourcesink_profile")
demand_R4_R3_CH.iloc[:, 0:8]

# # load the profiles DataFrame, select the demand column
# # %% [markdown]
# # Now that we have created the profile, we need to create a config with the
# # information that the created profile is going to be integrated into the model
# # as fixed profile.

# # %%
# # "sourcesink_config" (demand configuration)
sourcesink_config = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [m.set.nodesdata, m.set.yearssel, ["Demand"], ["T_LDV_BF"]]
    )
)
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesFixedProfile"] = 1
sourcesink_config = sourcesink_config.dropna()

m.parameter.add(sourcesink_config, "sourcesink_config")
sourcesink_config

# ############################################################
demand_R4_R3_CH = profiles[["BUS_el_CI_2050","BUS_el_FJ_2050","BUS_el_FSM_2050","BUS_el_KB_2050","BUS_el_MI_2050","BUS_el_NU_2050","BUS_el_NE_2050","BUS_el_PU_2050","BUS_el_PNG_2050","BUS_el_SA_2050","BUS_el_SI_2050","BUS_el_TA_2050","BUS_el_TU_2050","BUS_el_VU_2050"]]

demand_R4_R3_CH = demand_R4_R3_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R3_CH = demand_R4_R3_CH.T

demand_R4_R3_CH = demand_R4_R3_CH.rename(
    index={"BUS_el_CI_2050": "CI_data", "BUS_el_FJ_2050": "FJ_data", "BUS_el_FSM_2050": "FSM_data", "BUS_el_KB_2050": "KB_data", "BUS_el_MI_2050": "MI_data","BUS_el_NU_2050": "NU_data","BUS_el_NE_2050": "NE_data","BUS_el_PU_2050": "PU_data","BUS_el_PNG_2050": "PNG_data","BUS_el_SA_2050": "SA_data","BUS_el_SI_2050": "SI_data","BUS_el_TA_2050": "TA_data","BUS_el_TU_2050": "TU_data","BUS_el_VU_2050": "VU_data"}
)

# add columns and set them as index
demand_R4_R3_CH["years"] = "2050"
demand_R4_R3_CH["techs"] = "Demand"
demand_R4_R3_CH["commodity"] = "T_Bus_el"
demand_R4_R3_CH["type"] = "fixed"
demand_R4_R3_CH = demand_R4_R3_CH.set_index(
    ["years", "techs", "commodity", "type"], append=True
)

m.profile.add(demand_R4_R3_CH, "sourcesink_profile")
demand_R4_R3_CH.iloc[:, 0:8]

# # load the profiles DataFrame, select the demand column
# # %% [markdown]
# # Now that we have created the profile, we need to create a config with the
# # information that the created profile is going to be integrated into the model
# # as fixed profile.

# # %%
# # "sourcesink_config" (demand configuration)
sourcesink_config = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [m.set.nodesdata, m.set.yearssel, ["Demand"], ["T_Bus_el"]]
    )
)
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesFixedProfile"] = 1
sourcesink_config = sourcesink_config.dropna()

m.parameter.add(sourcesink_config, "sourcesink_config")
sourcesink_config
# ####################################################################################
demand_R4_R3_CH = profiles[["BUS_Th_CI_2050","BUS_Th_FJ_2050","BUS_Th_FSM_2050","BUS_Th_KB_2050","BUS_Th_MI_2050","BUS_Th_NU_2050","BUS_Th_NE_2050","BUS_Th_PU_2050","BUS_Th_PNG_2050","BUS_Th_SA_2050","BUS_Th_SI_2050","BUS_Th_TA_2050","BUS_Th_TU_2050","BUS_Th_VU_2050"]]

demand_R4_R3_CH = demand_R4_R3_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R3_CH = demand_R4_R3_CH.T

demand_R4_R3_CH = demand_R4_R3_CH.rename(
    index={"BUS_Th_CI_2050": "CI_data", "BUS_Th_FJ_2050": "FJ_data", "BUS_Th_FSM_2050": "FSM_data", "BUS_Th_KB_2050": "KB_data", "BUS_Th_MI_2050": "MI_data","BUS_Th_NU_2050": "NU_data","BUS_Th_NE_2050": "NE_data","BUS_Th_PU_2050": "PU_data","BUS_Th_PNG_2050": "PNG_data","BUS_Th_SA_2050": "SA_data","BUS_Th_SI_2050": "SI_data","BUS_Th_TA_2050": "TA_data","BUS_Th_TU_2050": "TU_data","BUS_Th_VU_2050": "VU_data"}
)

# add columns and set them as index
demand_R4_R3_CH["years"] = "2050"
demand_R4_R3_CH["techs"] = "Demand"
demand_R4_R3_CH["commodity"] = "T_Bus_th"
demand_R4_R3_CH["type"] = "fixed"
demand_R4_R3_CH = demand_R4_R3_CH.set_index(
    ["years", "techs", "commodity", "type"], append=True
)

m.profile.add(demand_R4_R3_CH, "sourcesink_profile")
demand_R4_R3_CH.iloc[:, 0:8]

# # load the profiles DataFrame, select the demand column
# # %% [markdown]
# # Now that we have created the profile, we need to create a config with the
# # information that the created profile is going to be integrated into the model
# # as fixed profile.

# # %%
# # "sourcesink_config" (demand configuration)
sourcesink_config = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [m.set.nodesdata, m.set.yearssel, ["Demand"], ["T_Bus_th"]]
    )
)
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesFixedProfile"] = 1
sourcesink_config = sourcesink_config.dropna()

m.parameter.add(sourcesink_config, "sourcesink_config")
sourcesink_config
# # ############################################################
demand_R4_R7_CH = profiles[["2W_el_CI_2050", "2W_el_FJ_2050","2W_el_FSM_2050", "2W_el_KB_2050", "2W_el_MI_2050","2W_el_NU_2050","2W_el_NE_2050","2W_el_PU_2050","2W_el_PNG_2050","2W_el_SA_2050","2W_el_SI_2050","2W_el_TA_2050","2W_el_TU_2050","2W_el_VU_2050"]]

demand_R4_R7_CH = demand_R4_R7_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R7_CH = demand_R4_R7_CH.T

demand_R4_R7_CH = demand_R4_R7_CH.rename(
    index={"2W_el_CI_2050": "CI_data", "2W_el_FJ_2050": "FJ_data", "2W_el_FSM_2050": "FSM_data", "2W_el_KB_2050": "KB_data", "2W_el_MI_2050": "MI_data","2W_el_NU_2050": "NU_data","2W_el_NE_2050": "NE_data","2W_el_PU_2050": "PU_data","2W_el_PNG_2050": "PNG_data","2W_el_SA_2050": "SA_data","2W_el_SI_2050": "SI_data","2W_el_TA_2050": "TA_data","2W_el_TU_2050": "TU_data","2W_el_VU_2050": "VU_data"}
)

# add columns and set them as index
demand_R4_R7_CH["years"] = "2050"
demand_R4_R7_CH["techs"] = "Demand"
demand_R4_R7_CH["commodity"] = "T_Two_wheel_el"
demand_R4_R7_CH["type"] = "fixed"
demand_R4_R7_CH = demand_R4_R7_CH.set_index(
    ["years", "techs", "commodity", "type"], append=True
)

m.profile.add(demand_R4_R7_CH, "sourcesink_profile")
demand_R4_R7_CH.iloc[:, 0:8]

# # load the profiles DataFrame, select the demand column
# # %% [markdown]
# # Now that we have created the profile, we need to create a config with the
# # information that the created profile is going to be integrated into the model
# # as fixed profile.

# # %%
# # "sourcesink_config" (demand configuration)
sourcesink_config = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [m.set.nodesdata, m.set.yearssel, ["Demand"], ["T_Two_wheel_el"]]
    )
)
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesFixedProfile"] = 1
sourcesink_config = sourcesink_config.dropna()

m.parameter.add(sourcesink_config, "sourcesink_config")
sourcesink_config
# # ############################################################
demand_R4_R7_CH = profiles[["2W_th_CI_2050", "2W_th_FJ_2050","2W_th_FSM_2050", "2W_th_KB_2050", "2W_th_MI_2050","2W_th_NU_2050","2W_th_NE_2050","2W_th_PU_2050","2W_th_PNG_2050","2W_th_SA_2050","2W_th_SI_2050","2W_th_TA_2050","2W_th_TU_2050","2W_th_VU_2050"]]

demand_R4_R7_CH = demand_R4_R7_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R7_CH = demand_R4_R7_CH.T

demand_R4_R7_CH = demand_R4_R7_CH.rename(
    index={"2W_th_CI_2050": "CI_data", "2W_th_FJ_2050": "FJ_data", "2W_th_FSM_2050": "FSM_data", "2W_th_KB_2050": "KB_data", "2W_th_MI_2050": "MI_data","2W_th_NU_2050": "NU_data","2W_th_NE_2050": "NE_data","2W_th_PU_2050": "PU_data","2W_th_PNG_2050": "PNG_data","2W_th_SA_2050": "SA_data","2W_th_SI_2050": "SI_data","2W_th_TA_2050": "TA_data","2W_th_TU_2050": "TU_data","2W_th_VU_2050": "VU_data"}
)

# add columns and set them as index
demand_R4_R7_CH["years"] = "2050"
demand_R4_R7_CH["techs"] = "Demand"
demand_R4_R7_CH["commodity"] = "T_Two_wheel_th"
demand_R4_R7_CH["type"] = "fixed"
demand_R4_R7_CH = demand_R4_R7_CH.set_index(
    ["years", "techs", "commodity", "type"], append=True
)

m.profile.add(demand_R4_R7_CH, "sourcesink_profile")
demand_R4_R7_CH.iloc[:, 0:8]

# # load the profiles DataFrame, select the demand column
# # %% [markdown]
# # Now that we have created the profile, we need to create a config with the
# # information that the created profile is going to be integrated into the model
# # as fixed profile.

# # %%
# # "sourcesink_config" (demand configuration)
sourcesink_config = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [m.set.nodesdata, m.set.yearssel, ["Demand"], ["T_Two_wheel_th"]]
    )
)
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesFixedProfile"] = 1
sourcesink_config = sourcesink_config.dropna()

m.parameter.add(sourcesink_config, "sourcesink_config")
sourcesink_config
# #############################################################
demand_R4_R8_CH = profiles[["Marine_TH_CI_2050", "Marine_TH_FJ_2050","Marine_TH_FSM_2050", "Marine_TH_KB_2050", "Marine_TH_MI_2050","Marine_TH_NU_2050","Marine_TH_NE_2050","Marine_TH_PU_2050","Marine_TH_PNG_2050","Marine_TH_SA_2050","Marine_TH_SI_2050","Marine_TH_TA_2050","Marine_TH_TU_2050","Marine_TH_VU_2050"]]

demand_R4_R8_CH = demand_R4_R8_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R8_CH = demand_R4_R8_CH.T

demand_R4_R8_CH = demand_R4_R8_CH.rename(
    index={"Marine_TH_CI_2050": "CI_data", "Marine_TH_FJ_2050": "FJ_data", "Marine_TH_FSM_2050": "FSM_data", "Marine_TH_KB_2050": "KB_data", "Marine_TH_MI_2050": "MI_data","Marine_TH_NU_2050": "NU_data","Marine_TH_NE_2050": "NE_data","Marine_TH_PU_2050": "PU_data","Marine_TH_PNG_2050": "PNG_data","Marine_TH_SA_2050": "SA_data","Marine_TH_SI_2050": "SI_data","Marine_TH_TA_2050": "TA_data","Marine_TH_TU_2050": "TU_data","Marine_TH_VU_2050": "VU_data"}
)

# add columns and set them as index
demand_R4_R8_CH["years"] = "2050"
demand_R4_R8_CH["techs"] = "Demand"
demand_R4_R8_CH["commodity"] = "T_Marine_f_th"
demand_R4_R8_CH["type"] = "fixed"
demand_R4_R8_CH = demand_R4_R8_CH.set_index(
    ["years", "techs", "commodity", "type"], append=True
)

m.profile.add(demand_R4_R8_CH, "sourcesink_profile")
demand_R4_R8_CH.iloc[:, 0:8]

# # load the profiles DataFrame, select the demand column
# # %% [markdown]
# # Now that we have created the profile, we need to create a config with the
# # information that the created profile is going to be integrated into the model
# # as fixed profile.

# # %%
# # "sourcesink_config" (demand configuration)
sourcesink_config = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [m.set.nodesdata, m.set.yearssel, ["Demand"], ["T_Marine_f_th"]]
    )
)
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesFixedProfile"] = 1
sourcesink_config = sourcesink_config.dropna()

m.parameter.add(sourcesink_config, "sourcesink_config")
sourcesink_config
# #############################################################
demand_R4_R8_CH = profiles[["Marine_E_CI_2050", "Marine_E_FJ_2050","Marine_E_FSM_2050", "Marine_E_KB_2050", "Marine_E_MI_2050","Marine_E_NU_2050","Marine_E_NE_2050","Marine_E_PU_2050","Marine_E_PNG_2050","Marine_E_SA_2050","Marine_E_SI_2050","Marine_E_TA_2050","Marine_E_TU_2050","Marine_E_VU_2050"]]

demand_R4_R8_CH = demand_R4_R8_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R8_CH = demand_R4_R8_CH.T

demand_R4_R8_CH = demand_R4_R8_CH.rename(
    index={"Marine_E_CI_2050": "CI_data", "Marine_E_FJ_2050": "FJ_data", "Marine_E_FSM_2050": "FSM_data", "Marine_E_KB_2050": "KB_data", "Marine_E_MI_2050": "MI_data","Marine_E_NU_2050": "NU_data","Marine_E_NE_2050": "NE_data","Marine_E_PU_2050": "PU_data","Marine_E_PNG_2050": "PNG_data","Marine_E_SA_2050": "SA_data","Marine_E_SI_2050": "SI_data","Marine_E_TA_2050": "TA_data","Marine_E_TU_2050": "TU_data","Marine_E_VU_2050": "VU_data"}
)

# add columns and set them as index
demand_R4_R8_CH["years"] = "2050"
demand_R4_R8_CH["techs"] = "Demand"
demand_R4_R8_CH["commodity"] = "Dummy_EL"
demand_R4_R8_CH["type"] = "fixed"
demand_R4_R8_CH = demand_R4_R8_CH.set_index(
    ["years", "techs", "commodity", "type"], append=True
)

m.profile.add(demand_R4_R8_CH, "sourcesink_profile")
demand_R4_R8_CH.iloc[:, 0:8]

# # # load the profiles DataFrame, select the demand column
# # # %% [markdown]
# # # Now that we have created the profile, we need to create a config with the
# # # information that the created profile is going to be integrated into the model
# # # as fixed profile.

# # # %%
# # # "sourcesink_config" (demand configuration)
sourcesink_config = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [m.set.nodesdata, m.set.yearssel, ["Demand"], ["Dummy_EL"]]
    )
)
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesFixedProfile"] = 1
sourcesink_config = sourcesink_config.dropna()

m.parameter.add(sourcesink_config, "sourcesink_config")
sourcesink_config
# ############################################################
demand_R4_R9_CH = profiles[["AVIA_TH_CI_2050", "AVIA_TH_FJ_2050","AVIA_TH_FSM_2050", "AVIA_TH_KB_2050", "AVIA_TH_MI_2050","AVIA_TH_NU_2050","AVIA_TH_NE_2050","AVIA_TH_PU_2050","AVIA_TH_PNG_2050","AVIA_TH_SA_2050","AVIA_TH_SI_2050","AVIA_TH_TA_2050","AVIA_TH_TU_2050","AVIA_TH_VU_2050"]]

demand_R4_R9_CH = demand_R4_R9_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R9_CH = demand_R4_R9_CH.T

demand_R4_R9_CH = demand_R4_R9_CH.rename(
    index={"AVIA_TH_CI_2050": "CI_data", "AVIA_TH_FJ_2050": "FJ_data", "AVIA_TH_FSM_2050": "FSM_data", "AVIA_TH_KB_2050": "KB_data", "AVIA_TH_MI_2050": "MI_data","AVIA_TH_NU_2050": "NU_data","AVIA_TH_NE_2050": "NE_data","AVIA_TH_PU_2050": "PU_data","AVIA_TH_PNG_2050": "PNG_data","AVIA_TH_SA_2050": "SA_data","AVIA_TH_SI_2050": "SI_data","AVIA_TH_TA_2050": "TA_data","AVIA_TH_TU_2050": "TU_data","AVIA_TH_VU_2050": "VU_data"}
)

# add columns and set them as index
demand_R4_R9_CH["years"] = "2050"
demand_R4_R9_CH["techs"] = "Demand"
demand_R4_R9_CH["commodity"] = "T_Aviation_th"
demand_R4_R9_CH["type"] = "fixed"
demand_R4_R9_CH = demand_R4_R9_CH.set_index(
    ["years", "techs", "commodity", "type"], append=True
)

m.profile.add(demand_R4_R9_CH, "sourcesink_profile")
demand_R4_R9_CH.iloc[:, 0:8]

# # load the profiles DataFrame, select the demand column
# # %% [markdown]
# # Now that we have created the profile, we need to create a config with the
# # information that the created profile is going to be integrated into the model
# # as fixed profile.

# # %%
# # "sourcesink_config" (demand configuration)
sourcesink_config = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [m.set.nodesdata, m.set.yearssel, ["Demand"], ["T_Aviation_th"]]
    )
)
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesFixedProfile"] = 1
sourcesink_config = sourcesink_config.dropna()

m.parameter.add(sourcesink_config, "sourcesink_config")
sourcesink_config
# #################################################################################
demand_R4_R9_CH = profiles[["AVIA_EL_CI_2050", "AVIA_EL_FJ_2050","AVIA_EL_FSM_2050", "AVIA_EL_KB_2050", "AVIA_EL_MI_2050","AVIA_EL_NU_2050","AVIA_EL_NE_2050","AVIA_EL_PU_2050","AVIA_EL_PNG_2050","AVIA_EL_SA_2050","AVIA_EL_SI_2050","AVIA_EL_TA_2050","AVIA_EL_TU_2050","AVIA_EL_VU_2050"]]

demand_R4_R9_CH = demand_R4_R9_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R9_CH = demand_R4_R9_CH.T

demand_R4_R9_CH = demand_R4_R9_CH.rename(
    index={"AVIA_EL_CI_2050": "CI_data", "AVIA_EL_FJ_2050": "FJ_data", "AVIA_EL_FSM_2050": "FSM_data", "AVIA_EL_KB_2050": "KB_data", "AVIA_EL_MI_2050": "MI_data","AVIA_EL_NU_2050": "NU_data","AVIA_EL_NE_2050": "NE_data","AVIA_EL_PU_2050": "PU_data","AVIA_EL_PNG_2050": "PNG_data","AVIA_EL_SA_2050": "SA_data","AVIA_EL_SI_2050": "SI_data","AVIA_EL_TA_2050": "TA_data","AVIA_EL_TU_2050": "TU_data","AVIA_EL_VU_2050": "VU_data"}
)

# add columns and set them as index
demand_R4_R9_CH["years"] = "2050"
demand_R4_R9_CH["techs"] = "Demand"
demand_R4_R9_CH["commodity"] = "T_Aviation_el"
demand_R4_R9_CH["type"] = "fixed"
demand_R4_R9_CH = demand_R4_R9_CH.set_index(
    ["years", "techs", "commodity", "type"], append=True
)

m.profile.add(demand_R4_R9_CH, "sourcesink_profile")
demand_R4_R9_CH.iloc[:, 0:8]

# # load the profiles DataFrame, select the demand column
# # %% [markdown]
# # Now that we have created the profile, we need to create a config with the
# # information that the created profile is going to be integrated into the model
# # as fixed profile.

# # %%
# # "sourcesink_config" (demand configuration)
sourcesink_config = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [m.set.nodesdata, m.set.yearssel, ["Demand"], ["T_Aviation_el"]]
    )
)
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesFixedProfile"] = 1
sourcesink_config = sourcesink_config.dropna()

m.parameter.add(sourcesink_config, "sourcesink_config")
sourcesink_config
###################################################################################
demand_R4_R9_CH = profiles[["AVIA_E_CI_2050", "AVIA_E_FJ_2050","AVIA_E_FSM_2050", "AVIA_E_KB_2050", "AVIA_E_MI_2050","AVIA_E_NU_2050","AVIA_E_NE_2050","AVIA_E_PU_2050","AVIA_E_PNG_2050","AVIA_E_SA_2050","AVIA_E_SI_2050","AVIA_E_TA_2050","AVIA_E_TU_2050","AVIA_E_VU_2050"]]

demand_R4_R9_CH = demand_R4_R9_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R9_CH = demand_R4_R9_CH.T

demand_R4_R9_CH = demand_R4_R9_CH.rename(
    index={"AVIA_E_CI_2050": "CI_data", "AVIA_E_FJ_2050": "FJ_data", "AVIA_E_FSM_2050": "FSM_data", "AVIA_E_KB_2050": "KB_data", "AVIA_E_MI_2050": "MI_data","AVIA_E_NU_2050": "NU_data","AVIA_E_NE_2050": "NE_data","AVIA_E_PU_2050": "PU_data","AVIA_E_PNG_2050": "PNG_data","AVIA_E_SA_2050": "SA_data","AVIA_E_SI_2050": "SI_data","AVIA_E_TA_2050": "TA_data","AVIA_E_TU_2050": "TU_data","AVIA_E_VU_2050": "VU_data"}
)

# add columns and set them as index
demand_R4_R9_CH["years"] = "2050"
demand_R4_R9_CH["techs"] = "Demand"
demand_R4_R9_CH["commodity"] = "eKerosene"
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
        [m.set.nodesdata, m.set.yearssel, ["Demand"], ["eKerosene"]]
    )
)
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesFixedProfile"] = 1
sourcesink_config = sourcesink_config.dropna()

m.parameter.add(sourcesink_config, "sourcesink_config")
sourcesink_config
###################################################################################

#################################################################################
demand_R4_R10_CH = profiles[["HC_B_CI_2050", "HC_B_FJ_2050","HC_B_FSM_2050", "HC_B_KB_2050", "HC_B_MI_2050","HC_B_NU_2050","HC_B_NE_2050","HC_B_PU_2050","HC_B_PNG_2050","HC_B_SA_2050","HC_B_SI_2050","HC_B_TA_2050","HC_B_TU_2050","HC_B_VU_2050"]]

demand_R4_R10_CH = demand_R4_R10_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R10_CH = demand_R4_R10_CH.T

demand_R4_R10_CH = demand_R4_R10_CH.rename(
    index={"HC_B_CI_2050": "CI_data", "HC_B_FJ_2050": "FJ_data", "HC_B_FSM_2050": "FSM_data", "HC_B_KB_2050": "KB_data", "HC_B_MI_2050": "MI_data","HC_B_NU_2050": "NU_data","HC_B_NE_2050": "NE_data","HC_B_PU_2050": "PU_data","HC_B_PNG_2050": "PNG_data","HC_B_SA_2050": "SA_data","HC_B_SI_2050": "SI_data","HC_B_TA_2050": "TA_data","HC_B_TU_2050": "TU_data","HC_B_VU_2050": "VU_data"}
)

# add columns and set them as index
demand_R4_R10_CH["years"] = "2050"
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
demand_R4_R10_CH = profiles[["HC_L_CI_2050", "HC_L_FJ_2050","HC_L_FSM_2050", "HC_L_KB_2050", "HC_L_MI_2050","HC_L_NU_2050","HC_L_NE_2050","HC_L_PU_2050","HC_L_PNG_2050","HC_L_SA_2050","HC_L_SI_2050","HC_L_TA_2050","HC_L_TU_2050","HC_L_VU_2050"]]

demand_R4_R10_CH = demand_R4_R10_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R10_CH = demand_R4_R10_CH.T

demand_R4_R10_CH = demand_R4_R10_CH.rename(
    index={"HC_L_CI_2050": "CI_data", "HC_L_FJ_2050": "FJ_data", "HC_L_FSM_2050": "FSM_data", "HC_L_KB_2050": "KB_data", "HC_L_MI_2050": "MI_data","HC_L_NU_2050": "NU_data","HC_L_NE_2050": "NE_data","HC_L_PU_2050": "PU_data","HC_L_PNG_2050": "PNG_data","HC_L_SA_2050": "SA_data","HC_L_SI_2050": "SI_data","HC_L_TA_2050": "TA_data","HC_L_TU_2050": "TU_data","HC_L_VU_2050": "VU_data"}
)

# add columns and set them as index
demand_R4_R10_CH["years"] = "2050"
demand_R4_R10_CH["techs"] = "Demand"
demand_R4_R10_CH["commodity"] = "T_cook_LPG"
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
        [m.set.nodesdata, m.set.yearssel, ["Demand"], ["T_cook_LPG"]]
    )
)
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesFixedProfile"] = 1
sourcesink_config = sourcesink_config.dropna()

m.parameter.add(sourcesink_config, "sourcesink_config")
sourcesink_config
############################################################
demand_R4_R10_CH = profiles[["HC_el_CI_2050", "HC_el_FJ_2050","HC_el_FSM_2050", "HC_el_KB_2050", "HC_el_MI_2050","HC_el_NU_2050","HC_el_NE_2050","HC_el_PU_2050","HC_el_PNG_2050","HC_el_SA_2050","HC_el_SI_2050","HC_el_TA_2050","HC_el_TU_2050","HC_el_VU_2050"]]

demand_R4_R10_CH = demand_R4_R10_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R10_CH = demand_R4_R10_CH.T

demand_R4_R10_CH = demand_R4_R10_CH.rename(
    index={"HC_el_CI_2050": "CI_data", "HC_el_FJ_2050": "FJ_data", "HC_el_FSM_2050": "FSM_data", "HC_el_KB_2050": "KB_data", "HC_el_MI_2050": "MI_data","HC_el_NU_2050": "NU_data","HC_el_NE_2050": "NE_data","HC_el_PU_2050": "PU_data","HC_el_PNG_2050": "PNG_data","HC_el_SA_2050": "SA_data","HC_el_SI_2050": "SI_data","HC_el_TA_2050": "TA_data","HC_el_TU_2050": "TU_data","HC_el_VU_2050": "VU_data"}
)

# add columns and set them as index
demand_R4_R10_CH["years"] = "2050"
demand_R4_R10_CH["techs"] = "Demand"
demand_R4_R10_CH["commodity"] = "T_cook_el"
demand_R4_R10_CH["type"] = "fixed"
demand_R4_R10_CH = demand_R4_R10_CH.set_index(
    ["years", "techs", "commodity", "type"], append=True
)

m.profile.add(demand_R4_R10_CH, "sourcesink_profile")
demand_R4_R10_CH.iloc[:, 0:8]

# # load the profiles DataFrame, select the demand column
# # %% [markdown]
# # Now that we have created the profile, we need to create a config with the
# # information that the created profile is going to be integrated into the model
# # as fixed profile.

# # %%
# # "sourcesink_config" (demand configuration)
sourcesink_config = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [m.set.nodesdata, m.set.yearssel, ["Demand"], ["T_cook_el"]]
    )
)
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesFixedProfile"] = 1
sourcesink_config = sourcesink_config.dropna()

m.parameter.add(sourcesink_config, "sourcesink_config")
sourcesink_config
###########################################################
demand_R4_R10_CH = profiles[["HI_D_CI_2050", "HI_D_FJ_2050","HI_D_FSM_2050", "HI_D_KB_2050", "HI_D_MI_2050","HI_D_NU_2050","HI_D_NE_2050","HI_D_PU_2050","HI_D_PNG_2050","HI_D_SA_2050","HI_D_SI_2050","HI_D_TA_2050","HI_D_TU_2050","HI_D_VU_2050"]]

demand_R4_R10_CH = demand_R4_R10_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R10_CH = demand_R4_R10_CH.T

demand_R4_R10_CH = demand_R4_R10_CH.rename(
    index={"HI_D_CI_2050": "CI_data", "HI_D_FJ_2050": "FJ_data", "HI_D_FSM_2050": "FSM_data", "HI_D_KB_2050": "KB_data", "HI_D_MI_2050": "MI_data","HI_D_NU_2050": "NU_data","HI_D_NE_2050": "NE_data","HI_D_PU_2050": "PU_data","HI_D_PNG_2050": "PNG_data","HI_D_SA_2050": "SA_data","HI_D_SI_2050": "SI_data","HI_D_TA_2050": "TA_data","HI_D_TU_2050": "TU_data","HI_D_VU_2050": "VU_data"}
)

# add columns and set them as index
demand_R4_R10_CH["years"] = "2050"
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
demand_R4_R10_CH = profiles[["HI_EH_CI_2050", "HI_EH_FJ_2050","HI_EH_FSM_2050", "HI_EH_KB_2050", "HI_EH_MI_2050","HI_EH_NU_2050","HI_EH_NE_2050","HI_EH_PU_2050","HI_EH_PNG_2050","HI_EH_SA_2050","HI_EH_SI_2050","HI_EH_TA_2050","HI_EH_TU_2050","HI_EH_VU_2050"]]

demand_R4_R10_CH = demand_R4_R10_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R10_CH = demand_R4_R10_CH.T

demand_R4_R10_CH = demand_R4_R10_CH.rename(
    index={"HI_EH_CI_2050": "CI_data", "HI_EH_FJ_2050": "FJ_data", "HI_EH_FSM_2050": "FSM_data", "HI_EH_KB_2050": "KB_data", "HI_EH_MI_2050": "MI_data","HI_EH_NU_2050": "NU_data","HI_EH_NE_2050": "NE_data","HI_EH_PU_2050": "PU_data","HI_EH_PNG_2050": "PNG_data","HI_EH_SA_2050": "SA_data","HI_EH_SI_2050": "SI_data","HI_EH_TA_2050": "TA_data","HI_EH_TU_2050": "TU_data","HI_EH_VU_2050": "VU_data"}
)

# add columns and set them as index
demand_R4_R10_CH["years"] = "2050"
demand_R4_R10_CH["techs"] = "Demand"
demand_R4_R10_CH["commodity"] = "T_Industry_EH"
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
        [m.set.nodesdata, m.set.yearssel, ["Demand"], ["T_Industry_EH"]]
    )
)
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesFixedProfile"] = 1
sourcesink_config = sourcesink_config.dropna()

m.parameter.add(sourcesink_config, "sourcesink_config")
sourcesink_config
##################################################################
demand_R4_R10_CH = profiles[["DHW_E_CI_2050", "DHW_E_FJ_2050","DHW_E_FSM_2050", "DHW_E_KB_2050", "DHW_E_MI_2050","DHW_E_NU_2050","DHW_E_NE_2050","DHW_E_PU_2050","DHW_E_PNG_2050","DHW_E_SA_2050","DHW_E_SI_2050","DHW_E_TA_2050","DHW_E_TU_2050","DHW_E_VU_2050"]]

demand_R4_R10_CH = demand_R4_R10_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R10_CH = demand_R4_R10_CH.T

demand_R4_R10_CH = demand_R4_R10_CH.rename(
    index={"DHW_E_CI_2050": "CI_data", "DHW_E_FJ_2050": "FJ_data", "DHW_E_FSM_2050": "FSM_data", "DHW_E_KB_2050": "KB_data", "DHW_E_MI_2050": "MI_data","DHW_E_NU_2050": "NU_data","DHW_E_NE_2050": "NE_data","DHW_E_PU_2050": "PU_data","DHW_E_PNG_2050": "PNG_data","DHW_E_SA_2050": "SA_data","DHW_E_SI_2050": "SI_data","DHW_E_TA_2050": "TA_data","DHW_E_TU_2050": "TU_data","DHW_E_VU_2050": "VU_data"}
)

# add columns and set them as index
demand_R4_R10_CH["years"] = "2050"
demand_R4_R10_CH["techs"] = "Demand"
demand_R4_R10_CH["commodity"] = "DHW_el"
demand_R4_R10_CH["type"] = "fixed"
demand_R4_R10_CH = demand_R4_R10_CH.set_index(
    ["years", "techs", "commodity", "type"], append=True
)

m.profile.add(demand_R4_R10_CH, "sourcesink_profile")
demand_R4_R10_CH.iloc[:, 0:8]

# # load the profiles DataFrame, select the demand column
# # %% [markdown]
# # Now that we have created the profile, we need to create a config with the
# # information that the created profile is going to be integrated into the model
# # as fixed profile.

# # %%
# # "sourcesink_config" (demand configuration)
sourcesink_config = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [m.set.nodesdata, m.set.yearssel, ["Demand"], ["DHW_el"]]
    )
)
sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesFixedProfile"] = 1
sourcesink_config = sourcesink_config.dropna()

m.parameter.add(sourcesink_config, "sourcesink_config")
sourcesink_config
#############################################################
demand_R4_R10_CH = profiles[["DHW_L_CI_2050", "DHW_L_FJ_2050","DHW_L_FSM_2050", "DHW_L_KB_2050", "DHW_L_MI_2050","DHW_L_NU_2050","DHW_L_NE_2050","DHW_L_PU_2050","DHW_L_PNG_2050","DHW_L_SA_2050","DHW_L_SI_2050","DHW_L_TA_2050","DHW_L_TU_2050","DHW_L_VU_2050"]]

demand_R4_R10_CH = demand_R4_R10_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R10_CH = demand_R4_R10_CH.T

demand_R4_R10_CH = demand_R4_R10_CH.rename(
    index={"DHW_L_CI_2050": "CI_data", "DHW_L_FJ_2050": "FJ_data", "DHW_L_FSM_2050": "FSM_data", "DHW_L_KB_2050": "KB_data", "DHW_L_MI_2050": "MI_data","DHW_L_NU_2050": "NU_data","DHW_L_NE_2050": "NE_data","DHW_L_PU_2050": "PU_data","DHW_L_PNG_2050": "PNG_data","DHW_L_SA_2050": "SA_data","DHW_L_SI_2050": "SI_data","DHW_L_TA_2050": "TA_data","DHW_L_TU_2050": "TU_data","DHW_L_VU_2050": "VU_data"}
)

# add columns and set them as index
demand_R4_R10_CH["years"] = "2050"
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
# ####################################################################################
# demand_R4_R10_CH = profiles[["DHW_ST_CI_2050", "DHW_ST_FJ_2050","DHW_ST_FSM_2050", "DHW_ST_KB_2050", "DHW_ST_MI_2050","DHW_ST_NU_2050","DHW_ST_NE_2050","DHW_ST_PU_2050","DHW_ST_PNG_2050","DHW_ST_SA_2050","DHW_ST_SI_2050","DHW_ST_TA_2050","DHW_ST_TU_2050","DHW_ST_VU_2050"]]

# demand_R4_R10_CH = demand_R4_R10_CH.div(1e3).mul(-1)
# # transpose DataFrame for needed format
# demand_R4_R10_CH = demand_R4_R10_CH.T

# demand_R4_R10_CH = demand_R4_R10_CH.rename(
#     index={"DHW_ST_CI_2050": "CI_data", "DHW_ST_FJ_2050": "FJ_data", "DHW_ST_FSM_2050": "FSM_data", "DHW_ST_KB_2050": "KB_data", "DHW_ST_MI_2050": "MI_data","DHW_ST_NU_2050": "NU_data","DHW_ST_NE_2050": "NE_data","DHW_ST_PU_2050": "PU_data","DHW_ST_PNG_2050": "PNG_data","DHW_ST_SA_2050": "SA_data","DHW_ST_SI_2050": "SI_data","DHW_ST_TA_2050": "TA_data","DHW_ST_TU_2050": "TU_data","DHW_ST_VU_2050": "VU_data"}
# )

# # add columns and set them as index
# demand_R4_R10_CH["years"] = "2050"
# demand_R4_R10_CH["techs"] = "Demand"
# demand_R4_R10_CH["commodity"] = "DHW_el"
# demand_R4_R10_CH["type"] = "fixed"
# demand_R4_R10_CH = demand_R4_R10_CH.set_index(
#     ["years", "techs", "commodity", "type"], append=True
# )

# m.profile.add(demand_R4_R10_CH, "sourcesink_profile")
# demand_R4_R10_CH.iloc[:, 0:8]

# # load the profiles DataFrame, select the demand column
# # %% [markdown]
# # Now that we have created the profile, we need to create a config with the
# # information that the created profile is going to be integrated into the model
# # as fixed profile.

# # %%
# # "sourcesink_config" (demand configuration)
# sourcesink_config = pd.DataFrame(
#     index=pd.MultiIndex.from_product(
#         [m.set.nodesdata, m.set.yearssel, ["Demand"], ["DHW_el"]]
#     )
# )
# sourcesink_config.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, :, :], "usesFixedProfile"] = 1
# sourcesink_config = sourcesink_config.dropna()

# m.parameter.add(sourcesink_config, "sourcesink_config")
# sourcesink_config
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
biomass_limits = [1000000, 1000000, 1000000, 1000000, 1000000,1000000,100000,1000000, 1000000,1000000, 1000000, 1000000,1000000, 1000000] 
#biomass_limits = [12, 2380, 168, 221, 22,5,4,12100, 1, 295, 1507, 211, 9, 671] 

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
HFOO_limits = [1000000, 1000000, 1000000, 1000000, 1000000,1000000,100000,1000000, 1000000,1000000, 1000000, 1000000,1000000, 1000000] 
#Dictionary####dic - pd df 
# GW or other units for R1_data, R2_data
lower_limit = 0  # same for all in this example

sourcesink_annualSum = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [m.set.nodesdata, m.set.yearssel, ["FuelImport"], ["HFOO"]]
    )
)

for node, limit in zip(m.set.nodesdata, HFOO_limits):
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
#################################################################################

##################################################################################
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
################################################################################
sourcesink_config = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [m.set.nodesdata, m.set.yearssel, ["FuelImport"], ["HFOO"]]
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

####################################################################################
# %%
# "accounting_sourcesinkFlow"
# setting a cost for methane imports
# User inputs perFlow prices for Biomass for each node
biomass_prices = [0.032, 0.032, 0.032, 0.032,0.032, 0.032, 0.032, 0.032, 0.032, 0.032, 0.032, 0.032,0.032, 0.032]  # Mio EUR per GWh_ch CH4 for R1_data, R2_data

accounting_sourcesinkFlow = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [["FuelCost"], m.set.nodesdata, ["2020"], ["FuelImport"], ["Biomass"]]
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
        [["FuelCost"], m.set.nodesdata, ["2020"], ["FuelImport"], ["NG"]]
    )
)

for node, price in zip(m.set.nodesdata, NG_prices):
    accounting_sourcesinkFlow.loc[idx["FuelCost", node, :, :, :], "perFlow"] = price

accounting_sourcesinkFlow = accounting_sourcesinkFlow.dropna()

m.parameter.add(accounting_sourcesinkFlow, "accounting_sourcesinkflow")
######################################################################
HFOO_prices = [0.031, 0.031, 0.031, 0.031,0.031, 0.031, 0.031, 0.031, 0.031, 0.031, 0.031, 0.031,0.031, 0.031]  # Mio EUR per GWh_ch CH4 for R1_data, R2_data

accounting_sourcesinkFlow = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [["FuelCost"], m.set.nodesdata, ["2020"], ["FuelImport"], ["HFOO"]]
    )
)

for node, price in zip(m.set.nodesdata, HFOO_prices):
    accounting_sourcesinkFlow.loc[idx["FuelCost", node, :, :, :], "perFlow"] = price

accounting_sourcesinkFlow = accounting_sourcesinkFlow.dropna()

m.parameter.add(accounting_sourcesinkFlow, "accounting_sourcesinkflow")
########################################################################
Diesel_prices = [0.095, 0.095, 0.095, 0.095,0.095, 0.095, 0.095, 0.095, 0.095, 0.095, 0.095, 0.095,0.095, 0.095]  # Mio EUR per GWh_ch CH4 for R1_data, R2_data

accounting_sourcesinkFlow = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [["FuelCost"], m.set.nodesdata,["2020"], ["FuelImport"], ["Diesel"]]
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
        [["FuelCost"], m.set.nodesdata, ["2020"], ["FuelImport"], ["LPG"]]
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
        [["FuelCost"], m.set.nodesdata, ["2020"], ["FuelImport"], ["Gasoline"]]
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
        [["FuelCost"], m.set.nodesdata,  ["2020"], ["FuelImport"], ["JetA1"]]
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
        [["FuelCost"], m.set.nodesdata,  ["2020"], ["FuelImport"], ["MDO"]]
    )
)

for node, price in zip(m.set.nodesdata, MDO_prices):
    accounting_sourcesinkFlow.loc[idx["FuelCost", node, :, :, :], "perFlow"] = price

accounting_sourcesinkFlow = accounting_sourcesinkFlow.dropna()

m.parameter.add(accounting_sourcesinkFlow, "accounting_sourcesinkflow")
###########################################################################2030
biomass_prices = [0.032, 0.032, 0.032, 0.032,0.032, 0.032, 0.032, 0.032, 0.032, 0.032, 0.032, 0.032,0.032, 0.032]  # Mio EUR per GWh_ch CH4 for R1_data, R2_data

accounting_sourcesinkFlow = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [["FuelCost"], m.set.nodesdata, ["2030", "2040", "2050"], ["FuelImport"], ["Biomass"]]
    )
)

for node, price in zip(m.set.nodesdata, biomass_prices):
    accounting_sourcesinkFlow.loc[idx["FuelCost", node, :, :, :], "perFlow"] = price

accounting_sourcesinkFlow = accounting_sourcesinkFlow.dropna()

m.parameter.add(accounting_sourcesinkFlow, "accounting_sourcesinkflow")
########################################################################
NG_prices = [0.025, 0.025, 0.025, 0.025,0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025,0.025, 0.025]  # Mio EUR per GWh_ch CH4 for R1_data, R2_data

accounting_sourcesinkFlow = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [["FuelCost"], m.set.nodesdata, ["2030", "2040", "2050"], ["FuelImport"], ["NG"]]
    )
)

for node, price in zip(m.set.nodesdata, NG_prices):
    accounting_sourcesinkFlow.loc[idx["FuelCost", node, :, :, :], "perFlow"] = price

accounting_sourcesinkFlow = accounting_sourcesinkFlow.dropna()

m.parameter.add(accounting_sourcesinkFlow, "accounting_sourcesinkflow")
######################################################################
HFOO_prices = [0.029, 0.029, 0.029, 0.029,0.029, 0.029, 0.029, 0.029, 0.029, 0.029, 0.029, 0.029,0.029, 0.029]  # Mio EUR per GWh_ch CH4 for R1_data, R2_data

accounting_sourcesinkFlow = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [["FuelCost"], m.set.nodesdata, ["2030", "2040", "2050"], ["FuelImport"], ["HFOO"]]
    )
)

for node, price in zip(m.set.nodesdata, HFOO_prices):
    accounting_sourcesinkFlow.loc[idx["FuelCost", node, :, :, :], "perFlow"] = price

accounting_sourcesinkFlow = accounting_sourcesinkFlow.dropna()

m.parameter.add(accounting_sourcesinkFlow, "accounting_sourcesinkflow")
########################################################################
Diesel_prices = [0.090, 0.090, 0.090, 0.090,0.090, 0.090, 0.090, 0.090, 0.090, 0.090, 0.090, 0.090,0.090, 0.090]  # Mio EUR per GWh_ch CH4 for R1_data, R2_data

accounting_sourcesinkFlow = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [["FuelCost"], m.set.nodesdata,["2030", "2040", "2050"], ["FuelImport"], ["Diesel"]]
    )
)

for node, price in zip(m.set.nodesdata, Diesel_prices):
    accounting_sourcesinkFlow.loc[idx["FuelCost", node, :, :, :], "perFlow"] = price

accounting_sourcesinkFlow = accounting_sourcesinkFlow.dropna()

m.parameter.add(accounting_sourcesinkFlow, "accounting_sourcesinkflow")
########################################################################
LPG_prices = [0.245, 0.245, 0.245, 0.245,0.245, 0.245, 0.245, 0.245, 0.245, 0.245, 0.245, 0.245,0.245, 0.245] # Mio EUR per GWh_ch CH4 for R1_data, R2_data

accounting_sourcesinkFlow = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [["FuelCost"], m.set.nodesdata, ["2030", "2040", "2050"], ["FuelImport"], ["LPG"]]
    )
)

for node, price in zip(m.set.nodesdata, LPG_prices):
    accounting_sourcesinkFlow.loc[idx["FuelCost", node, :, :, :], "perFlow"] = price

accounting_sourcesinkFlow = accounting_sourcesinkFlow.dropna()

m.parameter.add(accounting_sourcesinkFlow, "accounting_sourcesinkflow")
#########################################################################
Gasoline_prices = [0.099, 0.099, 0.099, 0.099,0.099, 0.099, 0.099, 0.099, 0.099, 0.099, 0.099, 0.099,0.099, 0.099]  # Mio EUR per GWh_ch CH4 for R1_data, R2_data

accounting_sourcesinkFlow = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [["FuelCost"], m.set.nodesdata, ["2030", "2040", "2050"], ["FuelImport"], ["Gasoline"]]
    )
)

for node, price in zip(m.set.nodesdata, Gasoline_prices):
    accounting_sourcesinkFlow.loc[idx["FuelCost", node, :, :, :], "perFlow"] = price

accounting_sourcesinkFlow = accounting_sourcesinkFlow.dropna()

m.parameter.add(accounting_sourcesinkFlow, "accounting_sourcesinkflow")
#########################################################################
JetA1_prices = [0.047, 0.047, 0.047, 0.047,0.047, 0.047, 0.047, 0.047, 0.047, 0.047, 0.047, 0.047,0.047, 0.047]  # Mio EUR per GWh_ch CH4 for R1_data, R2_data

accounting_sourcesinkFlow = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [["FuelCost"], m.set.nodesdata,  ["2030", "2040", "2050"], ["FuelImport"], ["JetA1"]]
    )
)

for node, price in zip(m.set.nodesdata, JetA1_prices):
    accounting_sourcesinkFlow.loc[idx["FuelCost", node, :, :, :], "perFlow"] = price

accounting_sourcesinkFlow = accounting_sourcesinkFlow.dropna()

m.parameter.add(accounting_sourcesinkFlow, "accounting_sourcesinkflow")
#########################################################################
MDO_prices = [0.042, 0.042, 0.042, 0.042,0.042, 0.042, 0.042, 0.042, 0.042, 0.042, 0.042, 0.042,0.042, 0.042]  # Mio EUR per GWh_ch CH4 for R1_data, R2_data

accounting_sourcesinkFlow = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [["FuelCost"], m.set.nodesdata,  ["2030", "2040", "2050"], ["FuelImport"], ["MDO"]]
    )
)

for node, price in zip(m.set.nodesdata, MDO_prices):
    accounting_sourcesinkFlow.loc[idx["FuelCost", node, :, :, :], "perFlow"] = price

accounting_sourcesinkFlow = accounting_sourcesinkFlow.dropna()

m.parameter.add(accounting_sourcesinkFlow, "accounting_sourcesinkflow")
#################################################################################
#################################################################################
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
converter_techParam = pd.DataFrame(
    index=pd.MultiIndex.from_product([["Battery"], ['2030', "2040", "2050"]])
)
converter_techParam.loc[idx["Battery", :], "lifeTime"] = 20
converter_techParam.loc[idx["Battery", :], "activityUpperLimit"] = 1

m.parameter.add(converter_techParam, "converter_techparam")
converter_techParam
# %%
# "converter_capacityParam"
converter_capacityParam = pd.DataFrame(
    index=pd.MultiIndex.from_product([m.set.nodesdata, ['2030', "2040", "2050"], ["Battery"]])
)
converter_capacityParam.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, "Battery"], "unitsUpperLimit"] = (
  300  # GW_el Converter upper limit
)
  # GW_el Converter upper limit

converter_capacityParam = converter_capacityParam.dropna()

m.parameter.add(converter_capacityParam, "converter_capacityparam")
converter_capacityParam
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
converter_coefficient = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [["Battery"], ['2030', "2040", "2050"], ["Charge", "Discharge"], ["Elec", "Elec_LiIon"]]
    )
)

converter_coefficient.loc[
    idx["Battery", :, "Charge", "Elec"], "coefficient"
] = -1  # GW_el
converter_coefficient.loc[idx["Battery", :, "Charge", "Elec_LiIon"], "coefficient"] = (
    0.95  # GW_el in LiIon
)
converter_coefficient.loc[idx["Battery", :, "Discharge", "Elec"], "coefficient"] = (
    1  # GW_el
)
converter_coefficient.loc[
    idx["Battery", :, "Discharge", "Elec_LiIon"], "coefficient"
] = -1.05  # GW_el in LiIon


# converter_coefficient.loc[
#     idx["Battery", :, "build","Elec_LiIon"], "coefficient"
# ] = 1  # GW_el in LiIon
# converter_coefficient.loc[
#     idx["Battery", :,"build", "Elec"], "coefficient"
# ] = 1
converter_coefficient = converter_coefficient.dropna(how="all")

m.parameter.add(converter_coefficient, "converter_coefficient")
converter_coefficient
# %%
# "accounting_converterUnits"
accounting_converterUnits = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [["Invest", "OMFix"], ["global"], ["horizon"], ["Battery"], ['2030', "2040", "2050"]]
    )
)

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Battery", "2030"], "perUnitBuild"
] = 150  # million EUR / unit
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Battery", "2030"], "useAnnuity"
] = 1  # binary yes/no
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Battery", "2030"], "amorTime"
] = 25  # years
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Battery", "2030"], "interest"
] = 0.06  # percent/100
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "Battery", "2030"], "perUnitTotal"
] = 4.5 

accounting_converterUnits = accounting_converterUnits.fillna(0)

m.parameter.add(accounting_converterUnits, "accounting_converterunits")
accounting_converterUnits



accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Battery", "2040"], "perUnitBuild"
] = 150  # million EUR / unit
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Battery", "2040"], "useAnnuity"
] = 1  # binary yes/no
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Battery", "2040"], "amorTime"
] = 25  # years
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Battery", "2040"], "interest"
] = 0.06  # percent/100
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "Battery", "2040"], "perUnitTotal"
] = 4.5 

accounting_converterUnits = accounting_converterUnits.fillna(0)

m.parameter.add(accounting_converterUnits, "accounting_converterunits")
accounting_converterUnits

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Battery", "2050"], "perUnitBuild"
] = 150  # million EUR / unit
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Battery", "2050"], "useAnnuity"
] = 1  # binary yes/no
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Battery", "2050"], "amorTime"
] = 25  # years
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Battery", "2050"], "interest"
] = 0.06  # percent/100
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "Battery", "2050"], "perUnitTotal"
] = 4.5 

accounting_converterUnits = accounting_converterUnits.fillna(0)

m.parameter.add(accounting_converterUnits, "accounting_converterunits")
accounting_converterUnits

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
# "storage_techParam"
storage_techParam = pd.DataFrame(
    index=pd.MultiIndex.from_product([["Battery"], ['2030', "2040", "2050"]])
)
storage_techParam.loc[idx["Battery", :], "lifeTime"] = 20
storage_techParam.loc[idx["Battery", :], "levelUpperLimit"] = 1

m.parameter.add(storage_techParam, "storage_techparam")
storage_techParam
# %% [markdown]
# For the storage size, we need to associate a commodity (here "Elec_LiIon") and
# a rated capacity for every storage reservoir unit.

# %%
# "storage_sizeParam"
# size of each storage unit
storage_sizeParam = pd.DataFrame(
    index=pd.MultiIndex.from_product([["Battery"], ['2030', "2040", "2050"], ["Elec_LiIon"]])
)
storage_sizeParam.loc[idx["Battery", :, "Elec_LiIon"], "size"] = 1 # GWh_ch/unit
storage_sizeParam = storage_sizeParam.dropna()

m.parameter.add(storage_sizeParam, "storage_sizeparam")
storage_sizeParam
# %% [markdown]
# Now we can set the storage reservoir upper limit to 30 units for a specific
# model region, therefore the model can build up to 240 GWh_ch of storage
# reservoir (8 GWh_ch / unit * 30 units = 240 GWh_ch).

# %%
# "storage_reservoirParam"
# installed storage reservoir units
storage_reservoirParam = pd.DataFrame(
    index=pd.MultiIndex.from_product([m.set.nodesdata, ['2030', "2040", "2050"], ["Battery"]])
)
storage_reservoirParam.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, "Battery"], "unitsUpperLimit"] = (
    1000 
)# units

storage_reservoirParam = storage_reservoirParam.dropna()

m.parameter.add(storage_reservoirParam, "storage_reservoirparam")
storage_reservoirParam
# %%
# "accounting_storageUnits"
# accounting for costs of storage
accounting_storageUnits = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [["Invest", "OMFix"], ["global"], ["horizon"], ["Battery"], ['2030', "2040", "2050"]]
    )
)

accounting_storageUnits.loc[idx["Invest", :, :, :, :], "perUnitBuild"] = (
    105 
) 

# Since our storage unit can store 8 GWh we need to scale the million EUR/GWh value with 8
accounting_storageUnits.loc[idx["Invest", :, :, :, "2030"], "useAnnuity"] = 1
accounting_storageUnits.loc[idx["Invest", :, :, :, "2030"], "amorTime"] = 20
accounting_storageUnits.loc[idx["Invest", :, :, :, "2030"], "interest"] = 0.06
accounting_storageUnits.loc[idx["OMFix", :, :, :, "2030"], "perUnitTotal"] = (
  5
)
accounting_storageUnits = accounting_storageUnits.fillna(0)

m.parameter.add(accounting_storageUnits, "accounting_storageunits")
accounting_storageUnits

accounting_storageUnits.loc[idx["Invest", :, :, :, :], "perUnitBuild"] = (
   90)

# Since our storage unit can store 8 GWh we need to scale the million EUR/GWh value with 8
accounting_storageUnits.loc[idx["Invest", :, :, :, "2040"], "useAnnuity"] = 1
accounting_storageUnits.loc[idx["Invest", :, :, :, "2040"], "amorTime"] = 20
accounting_storageUnits.loc[idx["Invest", :, :, :, "2040"], "interest"] = 0.06
accounting_storageUnits.loc[idx["OMFix", :, :, :, "2040"], "perUnitTotal"] = (
  5
)
accounting_storageUnits = accounting_storageUnits.fillna(0)

m.parameter.add(accounting_storageUnits, "accounting_storageunits")
accounting_storageUnits

accounting_storageUnits.loc[idx["Invest", :, :, :, :], "perUnitBuild"] = (
    72
) 

# Since our storage unit can store 8 GWh we need to scale the million EUR/GWh value with 8
accounting_storageUnits.loc[idx["Invest", :, :, :, "2050"], "useAnnuity"] = 1
accounting_storageUnits.loc[idx["Invest", :, :, :, "2050"], "amorTime"] = 20
accounting_storageUnits.loc[idx["Invest", :, :, :, "2050"], "interest"] = 0.06
accounting_storageUnits.loc[idx["OMFix", :, :, :, "2050"], "perUnitTotal"] = (
  5
)
accounting_storageUnits = accounting_storageUnits.fillna(0)

m.parameter.add(accounting_storageUnits, "accounting_storageunits")
accounting_storageUnits

##########################################thermal storage####################
converter_techParam = pd.DataFrame(
    index=pd.MultiIndex.from_product([["THSS"], ['2030', "2040", "2050"]])
)
converter_techParam.loc[idx["THSS", :], "lifeTime"] = 30
converter_techParam.loc[idx["THSS", :], "activityUpperLimit"] = 1

m.parameter.add(converter_techParam, "converter_techparam")
converter_techParam
# %%
# "converter_capacityParam"
converter_capacityParam = pd.DataFrame(
    index=pd.MultiIndex.from_product([m.set.nodesdata, ['2030', "2040", "2050"], ["THSS"]])
)
converter_capacityParam.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, "THSS"], "unitsUpperLimit"] = (
  300  # GW_el Converter upper limit
)
  # GW_el Converter upper limit

converter_capacityParam = converter_capacityParam.dropna()

m.parameter.add(converter_capacityParam, "converter_capacityparam")
converter_capacityParam


converter_coefficient = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [["THSS"], ['2030', "2040", "2050"], ["Charge", "Discharge"], ["Heat", "Heat_T"]]
    )
)

converter_coefficient.loc[
    idx["THSS", :, "Charge", "Heat"], "coefficient"
] = -1  # GW_el
converter_coefficient.loc[idx["THSS", :, "Charge", "Heat_T"], "coefficient"] = (
    0.99  # GW_el in LiIon
)
converter_coefficient.loc[idx["THSS", :, "Discharge", "Heat"], "coefficient"] = (
    1  # GW_el
)
converter_coefficient.loc[
    idx["THSS", :, "Discharge", "Heat_T"], "coefficient"
] = -1.01  # GW_el in LiIon


# converter_coefficient.loc[
#     idx["Battery", :, "build","Elec_LiIon"], "coefficient"
# ] = 1  # GW_el in LiIon
# converter_coefficient.loc[
#     idx["Battery", :,"build", "Elec"], "coefficient"
# ] = 1
converter_coefficient = converter_coefficient.dropna(how="all")

m.parameter.add(converter_coefficient, "converter_coefficient")
converter_coefficient
# %%
# "accounting_converterUnits"
accounting_converterUnits = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [["Invest", "OMFix"], ["global"], ["horizon"], ["THSS"], ['2030', "2040", "2050"]]
    )
)

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "THSS", "2030"], "perUnitBuild"
] = 0  # million EUR / unit
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "THSS", "2030"], "useAnnuity"
] = 1  # binary yes/no
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "THSS", "2030"], "amorTime"
] = 25  # years
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "THSS", "2030"], "interest"
] = 0.06  # percent/100
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "THSS", "2030"], "perUnitTotal"
] = 1

accounting_converterUnits = accounting_converterUnits.fillna(0)

m.parameter.add(accounting_converterUnits, "accounting_converterunits")
accounting_converterUnits


accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "THSS", "2040"], "perUnitBuild"
] = 0  # million EUR / unit
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "THSS", "2040"], "useAnnuity"
] = 1  # binary yes/no
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "THSS", "2040"], "amorTime"
] = 25  # years
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "THSS", "2040"], "interest"
] = 0.06  # percent/100
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "THSS", "2040"], "perUnitTotal"
] = 1

accounting_converterUnits = accounting_converterUnits.fillna(0)

m.parameter.add(accounting_converterUnits, "accounting_converterunits")
accounting_converterUnits

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "THSS", "2050"], "perUnitBuild"
] = 0  # million EUR / unit
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "THSS", "2050"], "useAnnuity"
] = 1  # binary yes/no
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "THSS", "2050"], "amorTime"
] = 25  # years
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "THSS", "2050"], "interest"
] = 0.06  # percent/100
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "THSS", "2050"], "perUnitTotal"
] = 1
accounting_converterUnits = accounting_converterUnits.fillna(0)

m.parameter.add(accounting_converterUnits, "accounting_converterunits")
accounting_converterUnits

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
# "storage_techParam"
storage_techParam = pd.DataFrame(
    index=pd.MultiIndex.from_product([["THSS"], ['2030', "2040", "2050"]])
)
storage_techParam.loc[idx["THSS", :], "lifeTime"] = 30
storage_techParam.loc[idx["THSS", :], "levelUpperLimit"] = 1

m.parameter.add(storage_techParam, "storage_techparam")
storage_techParam
# %% [markdown]
# For the storage size, we need to associate a commodity (here "Elec_LiIon") and
# a rated capacity for every storage reservoir unit.

# %%
# "storage_sizeParam"
# size of each storage unit
storage_sizeParam = pd.DataFrame(
    index=pd.MultiIndex.from_product([["THSS"], ['2030', "2040", "2050"], ["Heat_T"]])
)
storage_sizeParam.loc[idx["THSS", :, "Heat_T"], "size"] = 1 # GWh_ch/unit
storage_sizeParam = storage_sizeParam.dropna()

m.parameter.add(storage_sizeParam, "storage_sizeparam")
storage_sizeParam
# %% [markdown]
# Now we can set the storage reservoir uppermit to 30 units for a specific
# model region, therefore the model can build up to 240 GWh_ch of storage
# reservoir (8 GWh_ch / unit * 30 units = 240 GWh_ch).

# %%
# "storage_reservoirParam"
# installed storage reservoir units
storage_reservoirParam = pd.DataFrame(
    index=pd.MultiIndex.from_product([m.set.nodesdata, ['2030', "2040", "2050"], ["THSS"]])
)
storage_reservoirParam.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, "THSS"], "unitsUpperLimit"] = (
    1000 
)# units

storage_reservoirParam = storage_reservoirParam.dropna()

m.parameter.add(storage_reservoirParam, "storage_reservoirparam")
storage_reservoirParam
# %%
# "accounting_storageUnits"
# accounting for costs of storage
accounting_storageUnits = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [["Invest", "OMFix"], ["global"], ["horizon"], ["THSS"], ['2030', "2040", "2050"]]
    )
)

accounting_storageUnits.loc[idx["Invest", :, :, :, :], "perUnitBuild"] = (
30
) 

# Since our storage unit can store 8 GWh we need to scale the million EUR/GWh value with 8
accounting_storageUnits.loc[idx["Invest", :, :, :, "2030"], "useAnnuity"] = 1
accounting_storageUnits.loc[idx["Invest", :, :, :, "2030"], "amorTime"] = 30
accounting_storageUnits.loc[idx["Invest", :, :, :, "2030"], "interest"] = 0.06
accounting_storageUnits.loc[idx["OMFix", :, :, :, "2030"], "perUnitTotal"] = (
0
)
accounting_storageUnits = accounting_storageUnits.fillna(0)

m.parameter.add(accounting_storageUnits, "accounting_storageunits")
accounting_storageUnits

accounting_storageUnits.loc[idx["Invest", :, :, :, :], "perUnitBuild"] = (
23
) 

# Since our storage unit can store 8 GWh we need to scale the million EUR/GWh value with 8
accounting_storageUnits.loc[idx["Invest", :, :, :, "2040"], "useAnnuity"] = 1
accounting_storageUnits.loc[idx["Invest", :, :, :, "2040"], "amorTime"] = 30
accounting_storageUnits.loc[idx["Invest", :, :, :, "2040"], "interest"] = 0.06
accounting_storageUnits.loc[idx["OMFix", :, :, :, "2040"], "perUnitTotal"] = (
0
)
accounting_storageUnits = accounting_storageUnits.fillna(0)

m.parameter.add(accounting_storageUnits, "accounting_storageunits")
accounting_storageUnits

accounting_storageUnits.loc[idx["Invest", :, :, :, :], "perUnitBuild"] = (
17.6
) 

# Since our storage unit can store 8 GWh we need to scale the million EUR/GWh value with 8
accounting_storageUnits.loc[idx["Invest", :, :, :, "2050"], "useAnnuity"] = 1
accounting_storageUnits.loc[idx["Invest", :, :, :, "2050"], "amorTime"] = 30
accounting_storageUnits.loc[idx["Invest", :, :, :, "2050"], "interest"] = 0.06
accounting_storageUnits.loc[idx["OMFix", :, :, :, "2050"], "perUnitTotal"] = (
0
)
accounting_storageUnits = accounting_storageUnits.fillna(0)

m.parameter.add(accounting_storageUnits, "accounting_storageunits")
accounting_storageUnits
##################################water storage################################
converter_techParam = pd.DataFrame(
    index=pd.MultiIndex.from_product([["H20_storage"], ["2040", "2050"]])
)
converter_techParam.loc[idx["H20_storage", :], "lifeTime"] = 30
converter_techParam.loc[idx["H20_storage", :], "activityUpperLimit"] = 1

m.parameter.add(converter_techParam, "converter_techparam")
converter_techParam
# %%
# "converter_capacityParam"
converter_capacityParam = pd.DataFrame(
    index=pd.MultiIndex.from_product([m.set.nodesdata, ["2040", "2050"], ["H20_storage"]])
)
converter_capacityParam.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, "H20_storage"], "unitsUpperLimit"] = (
  300  # GW_el Converter upper limit
)
  # GW_el Converter upper limit

converter_capacityParam = converter_capacityParam.dropna()

m.parameter.add(converter_capacityParam, "converter_capacityparam")
converter_capacityParam


converter_coefficient = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [["H20_storage"], ["2040", "2050"], ["Charge", "Discharge"], ["Pure_water", "Pure_water_T"]]
    )
)

converter_coefficient.loc[
    idx["H20_storage", :, "Charge", "Pure_water"], "coefficient"
] = -1  # GW_el
converter_coefficient.loc[idx["H20_storage", :, "Charge", "Pure_water_T"], "coefficient"] = (
   1  # GW_el in LiIon
)
converter_coefficient.loc[idx["H20_storage", :, "Discharge", "Pure_water"], "coefficient"] = (
    1  # GW_el
)
converter_coefficient.loc[
    idx["H20_storage", :, "Discharge", "Pure_water_T"], "coefficient"
] = -1  # GW_el in LiIon


# converter_coefficient.loc[
#     idx["Battery", :, "build","Elec_LiIon"], "coefficient"
# ] = 1  # GW_el in LiIon
# converter_coefficient.loc[
#     idx["Battery", :,"build", "Elec"], "coefficient"
# ] = 1
converter_coefficient = converter_coefficient.dropna(how="all")

m.parameter.add(converter_coefficient, "converter_coefficient")
converter_coefficient
# %%
# "accounting_converterUnits"
accounting_converterUnits = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [["Invest", "OMFix"], ["global"], ["horizon"], ["H20_storage"], ["2040", "2050"]]
    )
)


accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "H20_storage", "2040"], "perUnitBuild"
] = 0  # million EUR / unit
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "H20_storage", "2040"], "useAnnuity"
] = 0  # binary yes/no
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "H20_storage", "2040"], "amorTime"
] = 0  # years
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "H20_storage", "2040"], "interest"
] = 0  # percent/100
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "H20_storage", "2040"], "perUnitTotal"
] = 0

accounting_converterUnits = accounting_converterUnits.fillna(0)

m.parameter.add(accounting_converterUnits, "accounting_converterunits")
accounting_converterUnits

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "H20_storage", "2050"], "perUnitBuild"
] = 0  # million EUR / unit
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "H20_storage", "2050"], "useAnnuity"
] = 0  # binary yes/no
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "H20_storage", "2050"], "amorTime"
] = 0  # years
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "H20_storage", "2050"], "interest"
] = 0  # percent/100
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "H20_storage", "2050"], "perUnitTotal"
] = 0
accounting_converterUnits = accounting_converterUnits.fillna(0)

m.parameter.add(accounting_converterUnits, "accounting_converterunits")
accounting_converterUnits

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
# "storage_techParam"
storage_techParam = pd.DataFrame(
    index=pd.MultiIndex.from_product([["H20_storage"], ["2040", "2050"]])
)
storage_techParam.loc[idx["H20_storage", :], "lifeTime"] = 30
storage_techParam.loc[idx["H20_storage", :], "levelUpperLimit"] = 1

m.parameter.add(storage_techParam, "storage_techparam")
storage_techParam
# %% [markdown]
# For the storage size, we need to associate a commodity (here "Elec_LiIon") and
# a rated capacity for every storage reservoir unit.

# %%
# "storage_sizeParam"
# size of each storage unit
storage_sizeParam = pd.DataFrame(
    index=pd.MultiIndex.from_product([["H20_storage"], ["2040", "2050"], ["Pure_water_T"]])
)
storage_sizeParam.loc[idx["H20_storage", :, "Pure_water_T"], "size"] = 1 # ((1000* m3))/unit
storage_sizeParam = storage_sizeParam.dropna()

m.parameter.add(storage_sizeParam, "storage_sizeparam")
storage_sizeParam
# %% [markdown]
# Now we can set the storage reservoir uppermit to 30 units for a specific
# model region, therefore the model can build up to 240 GWh_ch of storage
# reservoir (8 GWh_ch / unit * 30 units = 240 GWh_ch).

# %%
# "storage_reservoirParam"
# installed storage reservoir units
storage_reservoirParam = pd.DataFrame(
    index=pd.MultiIndex.from_product([m.set.nodesdata, ["2040", "2050"], ["H20_storage"]])
)
storage_reservoirParam.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, "H20_storage"], "unitsUpperLimit"] = (
    1000 
)# units

storage_reservoirParam = storage_reservoirParam.dropna()

m.parameter.add(storage_reservoirParam, "storage_reservoirparam")
storage_reservoirParam
# %%
# "accounting_storageUnits"
# accounting for costs of storage
accounting_storageUnits = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [["Invest", "OMFix"], ["global"], ["horizon"], ["H20_storage"], ["2040", "2050"]]
    )
)

accounting_storageUnits.loc[idx["Invest", :, :, :, :], "perUnitBuild"] = (
0.065
) 

# Since our storage unit can store 8 GWh we need to scale the million EUR/GWh value with 8
accounting_storageUnits.loc[idx["Invest", :, :, :, :], "useAnnuity"] = 1
accounting_storageUnits.loc[idx["Invest", :, :, :, :], "amorTime"] = 30
accounting_storageUnits.loc[idx["Invest", :, :, :, :], "interest"] = 0.06
accounting_storageUnits.loc[idx["OMFix", :, :, :, :], "perUnitTotal"] = (
.002
)
accounting_storageUnits = accounting_storageUnits.fillna(0)

m.parameter.add(accounting_storageUnits, "accounting_storageunits")
accounting_storageUnits

#############################H2 storage ###############################

converter_techParam = pd.DataFrame(
    index=pd.MultiIndex.from_product([["H2_storage"], ['2030', "2040", "2050"]])
)
converter_techParam.loc[idx["H2_storage", :], "lifeTime"] = 30
converter_techParam.loc[idx["H2_storage", :], "activityUpperLimit"] = 1

m.parameter.add(converter_techParam, "converter_techparam")
converter_techParam
# %%
# "converter_capacityParam"
converter_capacityParam = pd.DataFrame(
    index=pd.MultiIndex.from_product([m.set.nodesdata, ['2030', "2040", "2050"], ["H2_storage"]])
)
converter_capacityParam.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, "H2_storage"], "unitsUpperLimit"] = (
  300  # GW_el Converter upper limit
)
  # GW_el Converter upper limit

converter_capacityParam = converter_capacityParam.dropna()

m.parameter.add(converter_capacityParam, "converter_capacityparam")
converter_capacityParam


converter_coefficient = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [["H2_storage"], ['2030', "2040", "2050"], ["Charge", "Discharge"], ["Hydrogen", "Hydrogen_T"]]
    )
)

converter_coefficient.loc[
    idx["H2_storage", :, "Charge", "Hydrogen"], "coefficient"
] = -1  # GW_el
converter_coefficient.loc[idx["H2_storage", :, "Charge", "Hydrogen_T"], "coefficient"] = (
    1  # GW_el in LiIon
)
converter_coefficient.loc[idx["H2_storage", :, "Discharge", "Hydrogen"], "coefficient"] = (
    1  # GW_el
)
converter_coefficient.loc[
    idx["H2_storage", :, "Discharge", "Hydrogen_T"], "coefficient"
] = -1  # GW_el in LiIon


# converter_coefficient.loc[
#     idx["Battery", :, "build","Elec_LiIon"], "coefficient"
# ] = 1  # GW_el in LiIon
# converter_coefficient.loc[
#     idx["Battery", :,"build", "Elec"], "coefficient"
# ] = 1
converter_coefficient = converter_coefficient.dropna(how="all")

m.parameter.add(converter_coefficient, "converter_coefficient")
converter_coefficient
# %%
# "accounting_converterUnits"
accounting_converterUnits = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [["Invest", "OMFix"], ["global"], ["horizon"], ["H2_storage"], ['2030', "2040", "2050"]]
    )
)

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "H2_storage", "2030"], "perUnitBuild"
] = 0  # million EUR / unit
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "H2_storage", "2030"], "useAnnuity"
] = 1  # binary yes/no
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "H2_storage", "2030"], "amorTime"
] = 25  # years
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "H2_storage", "2030"], "interest"
] = 0.06  # percent/100
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "H2_storage", "2030"], "perUnitTotal"
] = 1

accounting_converterUnits = accounting_converterUnits.fillna(0)

m.parameter.add(accounting_converterUnits, "accounting_converterunits")
accounting_converterUnits

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "H2_storage", "2040"], "perUnitBuild"
] = 0  # million EUR / unit
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "H2_storage", "2040"], "useAnnuity"
] = 1  # binary yes/no
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "H2_storage", "2040"], "amorTime"
] = 25  # years
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "H2_storage", "2040"], "interest"
] = 0.06  # percent/100
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "H2_storage", "2040"], "perUnitTotal"
] = 1

accounting_converterUnits = accounting_converterUnits.fillna(0)

m.parameter.add(accounting_converterUnits, "accounting_converterunits")
accounting_converterUnits

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "H2_storage", "2050"], "perUnitBuild"
] = 0  # million EUR / unit
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "H2_storage", "2050"], "useAnnuity"
] = 1  # binary yes/no
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "H2_storage", "2050"], "amorTime"
] = 25  # years
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "H2_storage", "2050"], "interest"
] = 0.06  # percent/100
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "H2_storage", "2050"], "perUnitTotal"
] = 1

accounting_converterUnits = accounting_converterUnits.fillna(0)

m.parameter.add(accounting_converterUnits, "accounting_converterunits")
accounting_converterUnits

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
# "storage_techParam"
storage_techParam = pd.DataFrame(
    index=pd.MultiIndex.from_product([["H2_storage"], ['2030', "2040", "2050"]])
)
storage_techParam.loc[idx["H2_storage", :], "lifeTime"] = 30
storage_techParam.loc[idx["H2_storage", :], "levelUpperLimit"] = 1

m.parameter.add(storage_techParam, "storage_techparam")
storage_techParam
# %% [markdown]
# For the storage size, we need to associate a commodity (here "Elec_LiIon") and
# a rated capacity for every storage reservoir unit.

# %%
# "storage_sizeParam"
# size of each storage unit
storage_sizeParam = pd.DataFrame(
    index=pd.MultiIndex.from_product([["H2_storage"], ['2030', "2040", "2050"], ["Hydrogen_T"]])
)
storage_sizeParam.loc[idx["H2_storage", :, "Hydrogen_T"], "size"] = 1 # GWh_ch/unit
storage_sizeParam = storage_sizeParam.dropna()

m.parameter.add(storage_sizeParam, "storage_sizeparam")
storage_sizeParam
# %% [markdown]
# Now we can set the storage reservoir uppermit to 30 units for a specific
# model region, therefore the model can build up to 240 GWh_ch of storage
# reservoir (8 GWh_ch / unit * 30 units = 240 GWh_ch).

# %%
# "storage_reservoirParam"
# installed storage reservoir units
storage_reservoirParam = pd.DataFrame(
    index=pd.MultiIndex.from_product([m.set.nodesdata, ['2030', "2040", "2050"], ["H2_storage"]])
)
storage_reservoirParam.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, "H2_storage"], "unitsUpperLimit"] = (
    1000 
)# units

storage_reservoirParam = storage_reservoirParam.dropna()

m.parameter.add(storage_reservoirParam, "storage_reservoirparam")
storage_reservoirParam
# %%
# "accounting_storageUnits"
# accounting for costs of storage
accounting_storageUnits = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [["Invest", "OMFix"], ["global"], ["horizon"], ["H2_storage"], ['2030', "2040", "2050"]]
    )
)

accounting_storageUnits.loc[idx["Invest", :, :, :, :], "perUnitBuild"] = (
0.30
) 

# Since our storage unit can store 8 GWh we need to scale the million EUR/GWh value with 8
accounting_storageUnits.loc[idx["Invest", :, :, :, :], "useAnnuity"] = 1
accounting_storageUnits.loc[idx["Invest", :, :, :, :], "amorTime"] = 30
accounting_storageUnits.loc[idx["Invest", :, :, :, :], "interest"] = 0.06
accounting_storageUnits.loc[idx["OMFix", :, :, :, :], "perUnitTotal"] = (
0
)
accounting_storageUnits = accounting_storageUnits.fillna(0)

m.parameter.add(accounting_storageUnits, "accounting_storageunits")
accounting_storageUnits
#######################ammonia storage########################################

converter_techParam = pd.DataFrame(
    index=pd.MultiIndex.from_product([["Ammonia_storage"], ['2030', "2040", "2050"]])
)
converter_techParam.loc[idx["Ammonia_storage", :], "lifeTime"] = 30
converter_techParam.loc[idx["Ammonia_storage", :], "activityUpperLimit"] = 1

m.parameter.add(converter_techParam, "converter_techparam")
converter_techParam
# %%
# "converter_capacityParam"
converter_capacityParam = pd.DataFrame(
    index=pd.MultiIndex.from_product([m.set.nodesdata, ['2030', "2040", "2050"], ["Ammonia_storage"]])
)
converter_capacityParam.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, "Ammonia_storage"], "unitsUpperLimit"] = (
  300  # GW_el Converter upper limit
)
  # GW_el Converter upper limit

converter_capacityParam = converter_capacityParam.dropna()

m.parameter.add(converter_capacityParam, "converter_capacityparam")
converter_capacityParam


converter_coefficient = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [["Ammonia_storage"], ['2030', "2040", "2050"], ["Charge", "Discharge"], ["Ammonia", "Ammonia_T"]]
    )
)

converter_coefficient.loc[
    idx["Ammonia_storage", :, "Charge", "Ammonia"], "coefficient"
] = -1  # GW_el
converter_coefficient.loc[idx["Ammonia_storage", :, "Charge", "Ammonia_T"], "coefficient"] = (
    1  # GW_el in LiIon
)
converter_coefficient.loc[idx["Ammonia_storage", :, "Discharge", "Ammonia"], "coefficient"] = (
    1  # GW_el
)
converter_coefficient.loc[
    idx["Ammonia_storage", :, "Discharge", "Ammonia_T"], "coefficient"
] = -1  # GW_el in LiIon


# converter_coefficient.loc[
#     idx["Battery", :, "build","Elec_LiIon"], "coefficient"
# ] = 1  # GW_el in LiIon
# converter_coefficient.loc[
#     idx["Battery", :,"build", "Elec"], "coefficient"
# ] = 1
converter_coefficient = converter_coefficient.dropna(how="all")

m.parameter.add(converter_coefficient, "converter_coefficient")
converter_coefficient
# %%
# "accounting_converterUnits"
accounting_converterUnits = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [["Invest", "OMFix"], ["global"], ["horizon"], ["Ammonia_storage"], ['2030', "2040", "2050"]]
    )
)

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Ammonia_storage", "2030"], "perUnitBuild"
] = 0  # million EUR / unit
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Ammonia_storage", "2030"], "useAnnuity"
] = 1  # binary yes/no
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Ammonia_storage", "2030"], "amorTime"
] = 25  # years
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Ammonia_storage", "2030"], "interest"
] = 0.06  # percent/100
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "Ammonia_storage", "2030"], "perUnitTotal"
] = 1

accounting_converterUnits = accounting_converterUnits.fillna(0)

m.parameter.add(accounting_converterUnits, "accounting_converterunits")
accounting_converterUnits

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Ammonia_storage", "2040"], "perUnitBuild"
] = 0  # million EUR / unit
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Ammonia_storage", "2040"], "useAnnuity"
] = 1  # binary yes/no
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Ammonia_storage", "2040"], "amorTime"
] = 25  # years
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Ammonia_storage", "2040"], "interest"
] = 0.06  # percent/100
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "Ammonia_storage", "2040"], "perUnitTotal"
] = 1

accounting_converterUnits = accounting_converterUnits.fillna(0)

m.parameter.add(accounting_converterUnits, "accounting_converterunits")
accounting_converterUnits

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Ammonia_storage", "2050"], "perUnitBuild"
] = 0  # million EUR / unit
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Ammonia_storage", "2050"], "useAnnuity"
] = 1  # binary yes/no
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Ammonia_storage", "2050"], "amorTime"
] = 25  # years
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Ammonia_storage", "2050"], "interest"
] = 0.06  # percent/100
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "Ammonia_storage", "2050"], "perUnitTotal"
] = 1
accounting_converterUnits = accounting_converterUnits.fillna(0)

m.parameter.add(accounting_converterUnits, "accounting_converterunits")
accounting_converterUnits

accounting_converterUnits = accounting_converterUnits.fillna(0)

m.parameter.add(accounting_converterUnits, "accounting_converterunits")
accounting_converterUnits

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
# "storage_techParam"
storage_techParam = pd.DataFrame(
    index=pd.MultiIndex.from_product([["Ammonia_storage"], ['2030', "2040", "2050"]])
)
storage_techParam.loc[idx["Ammonia_storage", :], "lifeTime"] = 30
storage_techParam.loc[idx["Ammonia_storage", :], "levelUpperLimit"] = 1

m.parameter.add(storage_techParam, "storage_techparam")
storage_techParam
# %% [markdown]
# For the storage size, we need to associate a commodity (here "Elec_LiIon") and
# a rated capacity for every storage reservoir unit.

# %%
# "storage_sizeParam"
# size of each storage unit
storage_sizeParam = pd.DataFrame(
    index=pd.MultiIndex.from_product([["Ammonia_storage"], ['2030', "2040", "2050"], ["Ammonia_T"]])
)
storage_sizeParam.loc[idx["Ammonia_storage", :, "Ammonia_T"], "size"] = 1 # GWh_ch/unit
storage_sizeParam = storage_sizeParam.dropna()

m.parameter.add(storage_sizeParam, "storage_sizeparam")
storage_sizeParam
# %% [markdown]
# Now we can set the storage reservoir uppermit to 30 units for a specific
# model region, therefore the model can build up to 240 GWh_ch of storage
# reservoir (8 GWh_ch / unit * 30 units = 240 GWh_ch).

# %%
# "storage_reservoirParam"
# installed storage reservoir units
storage_reservoirParam = pd.DataFrame(  
    index=pd.MultiIndex.from_product([m.set.nodesdata, ['2030', "2040", "2050"], ["Ammonia_storage"]])
)
storage_reservoirParam.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, "Ammonia_storage"], "unitsUpperLimit"] = (
    1000 
)# units

storage_reservoirParam = storage_reservoirParam.dropna()

m.parameter.add(storage_reservoirParam, "storage_reservoirparam")
storage_reservoirParam
# %%
# "accounting_storageUnits"
# accounting for costs of storage
accounting_storageUnits = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [["Invest", "OMFix"], ["global"], ["horizon"], ["Ammonia_storage"], ['2030', "2040", "2050"]]
    )
)

accounting_storageUnits.loc[idx["Invest", :, :, :, :], "perUnitBuild"] = (
0.20
) 

# Since our storage unit can store 8 GWh we need to scale the million EUR/GWh value with 8
accounting_storageUnits.loc[idx["Invest", :, :, :, :], "useAnnuity"] = 1
accounting_storageUnits.loc[idx["Invest", :, :, :, :], "amorTime"] = 30
accounting_storageUnits.loc[idx["Invest", :, :, :, :], "interest"] = 0.06
accounting_storageUnits.loc[idx["OMFix", :, :, :, :], "perUnitTotal"] = (
0
)
accounting_storageUnits = accounting_storageUnits.fillna(0)

m.parameter.add(accounting_storageUnits, "accounting_storageunits")
accounting_storageUnits

###############################################Methanol#################################
converter_techParam = pd.DataFrame(
    index=pd.MultiIndex.from_product([["Methanol_storage"], ["2040", "2050"]])
)
converter_techParam.loc[idx["Methanol_storage", :], "lifeTime"] = 30
converter_techParam.loc[idx["Methanol_storage", :], "activityUpperLimit"] = 1

m.parameter.add(converter_techParam, "converter_techparam")
converter_techParam
# %%
# "converter_capacityParam"
converter_capacityParam = pd.DataFrame(
    index=pd.MultiIndex.from_product([m.set.nodesdata, ["2040", "2050"], ["Methanol_storage"]])
)
converter_capacityParam.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, "Methanol_storage"], "unitsUpperLimit"] = (
  300  # GW_el Converter upper limit
)
  # GW_el Converter upper limit

converter_capacityParam = converter_capacityParam.dropna()

m.parameter.add(converter_capacityParam, "converter_capacityparam")
converter_capacityParam


converter_coefficient = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [["Methanol_storage"], ["2040", "2050"], ["Charge", "Discharge"], ["Methanol", "Methanol_T"]]
    )
)

converter_coefficient.loc[
    idx["Methanol_storage", :, "Charge", "Methanol"], "coefficient"
] = -1  # GW_el
converter_coefficient.loc[idx["Methanol_storage", :, "Charge", "Methanol_T"], "coefficient"] = (
    1  # GW_el in LiIon
)
converter_coefficient.loc[idx["Methanol_storage", :, "Discharge", "Methanol"], "coefficient"] = (
    1  # GW_el
)
converter_coefficient.loc[
    idx["Methanol_storage", :, "Discharge", "Methanol_T"], "coefficient"
] = -1  # GW_el in LiIon


# converter_coefficient.loc[
#     idx["Battery", :, "build","Elec_LiIon"], "coefficient"
# ] = 1  # GW_el in LiIon
# converter_coefficient.loc[
#     idx["Battery", :,"build", "Elec"], "coefficient"
# ] = 1
converter_coefficient = converter_coefficient.dropna(how="all")

m.parameter.add(converter_coefficient, "converter_coefficient")
converter_coefficient
# %%
# "accounting_converterUnits"
accounting_converterUnits = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [["Invest", "OMFix"], ["global"], ["horizon"], ["Methanol_storage"], ["2040", "2050"]]
    )
)


accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Methanol_storage", "2040"], "perUnitBuild"
] = 0  # million EUR / unit
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Methanol_storage", "2040"], "useAnnuity"
] = 1  # binary yes/no
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Methanol_storage", "2040"], "amorTime"
] = 25  # years
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Methanol_storage", "2040"], "interest"
] = 0.06  # percent/100
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "Methanol_storage", "2040"], "perUnitTotal"
] = 0

accounting_converterUnits = accounting_converterUnits.fillna(0)

m.parameter.add(accounting_converterUnits, "accounting_converterunits")
accounting_converterUnits

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Methanol_storage", "2050"], "perUnitBuild"
] = 0  # million EUR / unit
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Methanol_storage", "2050"], "useAnnuity"
] = 1  # binary yes/no
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Methanol_storage", "2050"], "amorTime"
] = 25  # years
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Methanol_storage", "2050"], "interest"
] = 0.06  # percent/100
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "Methanol_storage", "2050"], "perUnitTotal"
] = 0
accounting_converterUnits = accounting_converterUnits.fillna(0)

m.parameter.add(accounting_converterUnits, "accounting_converterunits")
accounting_converterUnits

accounting_converterUnits = accounting_converterUnits.fillna(0)

m.parameter.add(accounting_converterUnits, "accounting_converterunits")
accounting_converterUnits

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
# "storage_techParam"
storage_techParam = pd.DataFrame(
    index=pd.MultiIndex.from_product([["Methanol_storage"], ["2040", "2050"]])
)
storage_techParam.loc[idx["Methanol_storage", :], "lifeTime"] = 30
storage_techParam.loc[idx["Methanol_storage", :], "levelUpperLimit"] = 1

m.parameter.add(storage_techParam, "storage_techparam")
storage_techParam
# %% [markdown]
# For the storage size, we need to associate a commodity (here "Elec_LiIon") and
# a rated capacity for every storage reservoir unit.

# %%
# "storage_sizeParam"
# size of each storage unit
storage_sizeParam = pd.DataFrame(
    index=pd.MultiIndex.from_product([["Methanol_storage"], ["2040", "2050"], ["Methanol_T"]])
)
storage_sizeParam.loc[idx["Methanol_storage", :, "Methanol_T"], "size"] = 1 # GWh_ch/unit
storage_sizeParam = storage_sizeParam.dropna()

m.parameter.add(storage_sizeParam, "storage_sizeparam")
storage_sizeParam
# %% [markdown]
# Now we can set the storage reservoir uppermit to 30 units for a specific
# model region, therefore the model can build up to 240 GWh_ch of storage
# reservoir (8 GWh_ch / unit * 30 units = 240 GWh_ch).

# %%
# "storage_reservoirParam"
# installed storage reservoir units
storage_reservoirParam = pd.DataFrame(  
    index=pd.MultiIndex.from_product([m.set.nodesdata, ["2040", "2050"], ["Methanol_storage"]])
)
storage_reservoirParam.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, "Methanol_storage"], "unitsUpperLimit"] = (
    1000 
)# units

storage_reservoirParam = storage_reservoirParam.dropna()

m.parameter.add(storage_reservoirParam, "storage_reservoirparam")
storage_reservoirParam
# %%
# "accounting_storageUnits"
# accounting for costs of storage
accounting_storageUnits = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [["Invest", "OMFix"], ["global"], ["horizon"], ["Methanol_storage"], ["2040", "2050"]]
    )
)

accounting_storageUnits.loc[idx["Invest", :, :, :, :], "perUnitBuild"] = (
0.058
) 

# Since our storage unit can store 8 GWh we need to scale the million EUR/GWh value with 8
accounting_storageUnits.loc[idx["Invest", :, :, :, :], "useAnnuity"] = 1
accounting_storageUnits.loc[idx["Invest", :, :, :, :], "amorTime"] = 30
accounting_storageUnits.loc[idx["Invest", :, :, :, :], "interest"] = 0.06
accounting_storageUnits.loc[idx["OMFix", :, :, :, :], "perUnitTotal"] = (
0
)
accounting_storageUnits = accounting_storageUnits.fillna(0)

m.parameter.add(accounting_storageUnits, "accounting_storageunits")
accounting_storageUnits
#####################eKerosene####################################################
converter_techParam = pd.DataFrame(
    index=pd.MultiIndex.from_product([["eKerosene_storage"], ["2040", "2050"]])
)
converter_techParam.loc[idx["eKerosene_storage", :], "lifeTime"] = 30
converter_techParam.loc[idx["eKerosene_storage", :], "activityUpperLimit"] = 1

m.parameter.add(converter_techParam, "converter_techparam")
converter_techParam
# %%
# "converter_capacityParam"
converter_capacityParam = pd.DataFrame(
    index=pd.MultiIndex.from_product([m.set.nodesdata, ["2040", "2050"], ["eKerosene_storage"]])
)
converter_capacityParam.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, "eKerosene_storage"], "unitsUpperLimit"] = (
  300  # GW_el Converter upper limit
)
  # GW_el Converter upper limit

converter_capacityParam = converter_capacityParam.dropna()

m.parameter.add(converter_capacityParam, "converter_capacityparam")
converter_capacityParam


converter_coefficient = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [["eKerosene_storage"], ['2040', "2050"], ["Charge", "Discharge"], ["eKerosene", "eKerosene_T"]]
    )
)

converter_coefficient.loc[
    idx["eKerosene_storage", :, "Charge", "eKerosene"], "coefficient"
] = -1  # GW_el
converter_coefficient.loc[idx["eKerosene_storage", :, "Charge", "eKerosene_T"], "coefficient"] = (
    1  # GW_el in LiIon
)
converter_coefficient.loc[idx["eKerosene_storage", :, "Discharge", "eKerosene"], "coefficient"] = (
    1  # GW_el
)
converter_coefficient.loc[
    idx["eKerosene_storage", :, "Discharge", "eKerosene_T"], "coefficient"
] = -1  # GW_el in LiIon


# converter_coefficient.loc[
#     idx["Battery", :, "build","Elec_LiIon"], "coefficient"
# ] = 1  # GW_el in LiIon
# converter_coefficient.loc[
#     idx["Battery", :,"build", "Elec"], "coefficient"
# ] = 1
converter_coefficient = converter_coefficient.dropna(how="all")

m.parameter.add(converter_coefficient, "converter_coefficient")
converter_coefficient
# %%
# "accounting_converterUnits"
accounting_converterUnits = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [["Invest", "OMFix"], ["global"], ["horizon"], ["eKerosene_storage"], ["2040", "2050"]]
 )
)

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "eKerosene_storage", "2040"], "perUnitBuild"
] = 0  # million EUR / unit
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "eKerosene_storage", "2040"], "useAnnuity"
] = 1  # binary yes/no
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "eKerosene_storage", "2040"], "amorTime"
] = 25  # years
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "eKerosene_storage", "2040"], "interest"
] = 0.06  # percent/100
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "eKerosene_storage", "2040"], "perUnitTotal"
] = 0

accounting_converterUnits = accounting_converterUnits.fillna(0)

m.parameter.add(accounting_converterUnits, "accounting_converterunits")
accounting_converterUnits

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "eKerosene_storage", "2050"], "perUnitBuild"
] = 0  # million EUR / unit
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "eKerosene_storage", "2050"], "useAnnuity"
] = 1  # binary yes/no
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "eKerosene_storage", "2050"], "amorTime"
] = 25  # years
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "eKerosene_storage", "2050"], "interest"
] = 0.06  # percent/100
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "eKerosene_storage", "2050"], "perUnitTotal"
] = 0
accounting_converterUnits = accounting_converterUnits.fillna(0)

m.parameter.add(accounting_converterUnits, "accounting_converterunits")
accounting_converterUnits

accounting_converterUnits = accounting_converterUnits.fillna(0)

m.parameter.add(accounting_converterUnits, "accounting_converterunits")
accounting_converterUnits

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
# "storage_techParam"
storage_techParam = pd.DataFrame(
    index=pd.MultiIndex.from_product([["eKerosene_storage"], ["2040", "2050"]])
)
storage_techParam.loc[idx["eKerosene_storage", :], "lifeTime"] = 30
storage_techParam.loc[idx["eKerosene_storage", :], "levelUpperLimit"] = 1

m.parameter.add(storage_techParam, "storage_techparam")
storage_techParam
# %% [markdown]
# For the storage size, we need to associate a commodity (here "Elec_LiIon") and
# a rated capacity for every storage reservoir unit.

# %%
# "storage_sizeParam"
# size of each storage unit
storage_sizeParam = pd.DataFrame(
    index=pd.MultiIndex.from_product([["eKerosene_storage"], ["2040", "2050"], ["Ammonia_T"]])
)
storage_sizeParam.loc[idx["eKerosene_storage", :, "Ammonia_T"], "size"] = 1 # GWh_ch/unit
storage_sizeParam = storage_sizeParam.dropna()

m.parameter.add(storage_sizeParam, "storage_sizeparam")
storage_sizeParam
# %% [markdown]
# Now we can set the storage reservoir uppermit to 30 units for a specific
# model region, therefore the model can build up to 240 GWh_ch of storage
# reservoir (8 GWh_ch / unit * 30 units = 240 GWh_ch).

# %%
# "storage_reservoirParam"
# installed storage reservoir units
storage_reservoirParam = pd.DataFrame(  
    index=pd.MultiIndex.from_product([m.set.nodesdata, ["2040", "2050"], ["eKerosene_storage"]])
)
storage_reservoirParam.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, "eKerosene_storage"], "unitsUpperLimit"] = (
    1000 
)# units

storage_reservoirParam = storage_reservoirParam.dropna()

m.parameter.add(storage_reservoirParam, "storage_reservoirparam")
storage_reservoirParam
# %%
# "accounting_storageUnits"
# accounting for costs of storage
accounting_storageUnits = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [["Invest", "OMFix"], ["global"], ["horizon"], ["eKerosene_storage"], ["2040", "2050"]]
    )
)

accounting_storageUnits.loc[idx["Invest", :, :, :, :], "perUnitBuild"] = (
0.058
) 

# Since our storage unit can store 8 GWh we need to scale the million EUR/GWh value with 8
accounting_storageUnits.loc[idx["Invest", :, :, :, :], "useAnnuity"] = 1
accounting_storageUnits.loc[idx["Invest", :, :, :, :], "amorTime"] = 30
accounting_storageUnits.loc[idx["Invest", :, :, :, :], "interest"] = 0.06
accounting_storageUnits.loc[idx["OMFix", :, :, :, :], "perUnitTotal"] = (
0
)
accounting_storageUnits = accounting_storageUnits.fillna(0)

m.parameter.add(accounting_storageUnits, "accounting_storageunits")
accounting_storageUnits

#####################co2 storage##################################################
converter_techParam = pd.DataFrame(
    index=pd.MultiIndex.from_product([["co2_storage"], ["2040", "2050"]])
)
converter_techParam.loc[idx["co2_storage", :], "lifeTime"] = 30
converter_techParam.loc[idx["co2_storage", :], "activityUpperLimit"] = 1

m.parameter.add(converter_techParam, "converter_techparam")
converter_techParam
# %%
# "converter_capacityParam"
converter_capacityParam = pd.DataFrame(
    index=pd.MultiIndex.from_product([m.set.nodesdata, ["2040", "2050"], ["co2_storage"]])
)
converter_capacityParam.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, "co2_storage"], "unitsUpperLimit"] = (
  300  # GW_el Converter upper limit
)
  # GW_el Converter upper limit

converter_capacityParam = converter_capacityParam.dropna()

m.parameter.add(converter_capacityParam, "converter_capacityparam")
converter_capacityParam


converter_coefficient = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [["co2_storage"], ['2040', "2050"], ["Charge", "Discharge"], ["co", "co_T"]]
    )
)

converter_coefficient.loc[
    idx["co2_storage", :, "Charge", "co"], "coefficient"
] = -1  # GW_el
converter_coefficient.loc[idx["co2_storage", :, "Charge", "co_T"], "coefficient"] = (
    1  # GW_el in LiIon
)
converter_coefficient.loc[idx["co2_storage", :, "Discharge", "co"], "coefficient"] = (
    1  # GW_el
)
converter_coefficient.loc[
    idx["co2_storage", :, "Discharge", "co_T"], "coefficient"
] = -1  # GW_el in LiIon


# converter_coefficient.loc[
#     idx["Battery", :, "build","Elec_LiIon"], "coefficient"
# ] = 1  # GW_el in LiIon
# converter_coefficient.loc[
#     idx["Battery", :,"build", "Elec"], "coefficient"
# ] = 1
converter_coefficient = converter_coefficient.dropna(how="all")

m.parameter.add(converter_coefficient, "converter_coefficient")
converter_coefficient
# %%
# "accounting_converterUnits"
accounting_converterUnits = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [["Invest", "OMFix"], ["global"], ["horizon"], ["co2_storage"], ["2040", "2050"]]
 )
)

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "co2_storage", "2040"], "perUnitBuild"
] = 0  # million EUR / unit
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "co2_storage", "2040"], "useAnnuity"
] = 1  # binary yes/no
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "co2_storage", "2040"], "amorTime"
] = 25  # years
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "co2_storage", "2040"], "interest"
] = 0.06  # percent/100
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "co2_storage", "2040"], "perUnitTotal"
] = 0

accounting_converterUnits = accounting_converterUnits.fillna(0)

m.parameter.add(accounting_converterUnits, "accounting_converterunits")
accounting_converterUnits

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "co2_storage", "2050"], "perUnitBuild"
] = 0  # million EUR / unit
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "co2_storage", "2050"], "useAnnuity"
] = 1  # binary yes/no
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "co2_storage", "2050"], "amorTime"
] = 25  # years
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "co2_storage", "2050"], "interest"
] = 0.06  # percent/100
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "co2_storage", "2050"], "perUnitTotal"
] = 0
accounting_converterUnits = accounting_converterUnits.fillna(0)

m.parameter.add(accounting_converterUnits, "accounting_converterunits")
accounting_converterUnits

accounting_converterUnits = accounting_converterUnits.fillna(0)

m.parameter.add(accounting_converterUnits, "accounting_converterunits")
accounting_converterUnits
############################################################################
accounting_converteractivity = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [["OMVar"], ["global"], ["horizon"], ["co2_storage"], ["2040"], ['Charge']]
 )
).sort_index()

accounting_converteractivity.loc[
    idx["OMVar", "global", "horizon", "co2_storage", "2040", "Charge"], "perActivity"
] = 0.035

accounting_converteractivity = accounting_converteractivity.fillna(0)

m.parameter.add(accounting_converteractivity, "accounting_converteractivity")
accounting_converteractivity
############################################################################################
accounting_converteractivity = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [["OMVar"], ["global"], ["horizon"], ["co2_storage"], ["2050"], ['Charge']]
 )
).sort_index()

accounting_converteractivity.loc[
    idx["OMVar", "global", "horizon", "co2_storage", "2050", "Charge"], "perActivity"
] = 0.035

accounting_converteractivity = accounting_converteractivity.fillna(0)

m.parameter.add(accounting_converteractivity, "accounting_converteractivity")
accounting_converteractivity
#############################################################################################
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
# "storage_techParam"
storage_techParam = pd.DataFrame(
    index=pd.MultiIndex.from_product([["co2_storage"], ["2040", "2050"]])
)
storage_techParam.loc[idx["co2_storage", :], "lifeTime"] = 30
storage_techParam.loc[idx["co2_storage", :], "levelUpperLimit"] = 1

m.parameter.add(storage_techParam, "storage_techparam")
storage_techParam
# %% [markdown]
# For the storage size, we need to associate a commodity (here "Elec_LiIon") and
# a rated capacity for every storage reservoir unit.

# %%
# "storage_sizeParam"
# size of each storage unit
storage_sizeParam = pd.DataFrame(
    index=pd.MultiIndex.from_product([["co2_storage"], ["2040", "2050"], ["co_T"]])
)
storage_sizeParam.loc[idx["co2_storage", :, "co_T"], "size"] = 1 # GWh_ch/unit
storage_sizeParam = storage_sizeParam.dropna()

m.parameter.add(storage_sizeParam, "storage_sizeparam")
storage_sizeParam
# %% [markdown]
# Now we can set the storage reservoir uppermit to 30 units for a specific
# model region, therefore the model can build up to 240 GWh_ch of storage
# reservoir (8 GWh_ch / unit * 30 units = 240 GWh_ch).

# %%
# "storage_reservoirParam"
# installed storage reservoir units
storage_reservoirParam = pd.DataFrame(  
    index=pd.MultiIndex.from_product([m.set.nodesdata, ["2040", "2050"], ["co2_storage"]])
)
storage_reservoirParam.loc[idx[["CI_data","FJ_data","FSM_data","KB_data","MI_data","NU_data","NE_data","PU_data","PNG_data","SA_data","SI_data","TA_data","TU_data","VU_data"], :, "co2_storage"], "unitsUpperLimit"] = (
    1000 
)# units

storage_reservoirParam = storage_reservoirParam.dropna()

m.parameter.add(storage_reservoirParam, "storage_reservoirparam")
storage_reservoirParam
# %%
# "accounting_storageUnits"
# accounting for costs of storage
accounting_storageUnits = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [["Invest", "OMFix"], ["global"], ["horizon"], ["co2_storage"], ["2040", "2050"]]
    )
)

accounting_storageUnits.loc[idx["Invest", :, :, :, :], "perUnitBuild"] = (0

) 

# Since our storage unit can store 8 GWh we need to scale the million EUR/GWh value with 8
accounting_storageUnits.loc[idx["Invest", :, :, :, :], "useAnnuity"] = 1
accounting_storageUnits.loc[idx["Invest", :, :, :, :], "amorTime"] = 30
accounting_storageUnits.loc[idx["Invest", :, :, :, :], "interest"] = 0.06
accounting_storageUnits.loc[idx["OMFix", :, :, :, :], "perUnitTotal"] = (
0
)

accounting_storageUnits = accounting_storageUnits.fillna(0)

m.parameter.add(accounting_storageUnits, "accounting_storageunits")
accounting_storageUnits
######################################################################################

############################################################################################
# %%
# write all files to `data/` directory
m.write(fileformat="dat")
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
    resultfile="IP_2050_3",
    lo=3,
    postcalc=1,
    roundts=1,
    pathopt="myopic") 
#######evaluation#################
