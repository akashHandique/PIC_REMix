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
profiles = pd.read_csv("../_input/profiles.csv", index_col=0)


# %%
# "map_aggregateNodesModel"
# DataFrame for aggregation from data to model regions
df = pd.DataFrame(
    [
        ["R1_data", "R1_model", 1],
        ["R2_data", "R2_model", 1],  # not strictly necessary for tutorial 1 and 2
        ["R3_data", "R3_model", 1],  # not strictly necessary for tutorial 1 and 2
        ["R4_data", "R4_model", 1],
        ["R5_data", "R5_model", 1],# not strictly necessary for tutorial 1 and 2
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
accounting_indicatorBounds["discount"] = 0.07  # social discount rate for the indicators

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
converter_techParam = pd.DataFrame(
    index=pd.MultiIndex.from_product([["CCGT", "PV", "WindOnshore"], m.set.yearssel])
)
converter_techParam.loc[idx["CCGT"], "lifeTime"] = 30  # years
converter_techParam.loc[idx["CCGT"], "activityUpperLimit"] = (
    1  # availability of technology
)

converter_techParam.loc[idx["PV"], "lifeTime"] = 20  # years
converter_techParam.loc[idx["PV"], "activityUpperLimit"] = (
    0  # this value will be replaced later on with the normalized feed-in profile
)

converter_techParam.loc[idx["WindOnshore"], "lifeTime"] = 25
converter_techParam.loc[idx["WindOnshore"], "activityUpperLimit"] = (
    0  # this value will be replaced later on with the normalized feed-in profile
)

m.parameter.add(converter_techParam, "converter_techparam")
converter_techParam
# %%
# "converter_capacityParam"
# defining upper and/or lower limits for converter technologies
converter_capacityParam = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [m.set.nodesdata, m.set.yearssel, ["CCGT", "PV", "WindOnshore"]]
    )
)
converter_capacityParam.loc[idx["R1_data", :, "CCGT"], "unitsUpperLimit"] = 100  # GW_el
converter_capacityParam.loc[idx["R1_data", :, "CCGT"], "unitsLowerLimit"] = 0  # GW_el
converter_capacityParam.loc[idx["R1_data", :, "PV"], "unitsUpperLimit"] = 80  # GW_el
converter_capacityParam.loc[idx["R1_data", :, "PV"], "unitsLowerLimit"] = 0  # GW_el
converter_capacityParam.loc[idx["R1_data", :, "WindOnshore"], "unitsUpperLimit"] = (
    120  # GW_el
)
converter_capacityParam.loc[idx["R1_data", :, "WindOnshore"], "unitsLowerLimit"] = (
    0  # GW_el
)
converter_capacityParam = converter_capacityParam.dropna(how="all")

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
            ["CCGT", "PV", "WindOnshore"],
            m.set.yearssel,
            ["Powergen"],
            ["CH4", "Elec", "CO2"],
        ]
    )
)
converter_coefficient.loc[idx["CCGT", :, :, "Elec"], "coefficient"] = 1  # GWh_el
converter_coefficient.loc[idx["CCGT", :, :, "CH4"], "coefficient"] = -1.587  # GWh_ch
converter_coefficient.loc[idx["CCGT", :, :, "CO2"], "coefficient"] = 0.320  # kt CO2

converter_coefficient.loc[idx["PV", :, :, "Elec"], "coefficient"] = 1  # GWh_el

converter_coefficient.loc[idx["WindOnshore", :, :, "Elec"], "coefficient"] = 1  # GWh_el
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
converter_activityProfile = profiles[["PV", "WindOnshore"]]

# convert from MW to GW
converter_activityProfile = converter_activityProfile.div(1e3).T

converter_activityProfile = converter_activityProfile.div(
    converter_activityProfile.max(axis=1), axis=0
)
converter_activityProfile.index.names = ["techs"]

# add columns and set them as index
converter_activityProfile["region"] = "R1_data"
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
            ["CCGT", "PV", "WindOnshore"],
            m.set.yearssel,
        ]
    )
).sort_index()

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "CCGT", "2030"], "perUnitBuild"
] = 700.0  # Mio EUR per unit
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "CCGT", "2030"], "useAnnuity"
] = 1  # binary yes/no
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "CCGT", "2030"], "amorTime"
] = 30  # years
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "CCGT", "2030"], "interest"
] = 0.06  # percent/100
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "CCGT", "2030"], "perUnitTotal"
] = 28.0  # Mio EUR per unit

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "PV", "2030"], "perUnitBuild"
] = 518.0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "PV", "2030"], "useAnnuity"
] = 1
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "PV", "2030"], "amorTime"
] = 20
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "PV", "2030"], "interest"
] = 0.06
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "PV", "2030"], "perUnitTotal"
] = 7.7

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "WindOnshore", "2030"], "perUnitBuild"
] = 1368.0
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "WindOnshore", "2030"], "useAnnuity"
] = 1
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "WindOnshore", "2030"], "amorTime"
] = 25
accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "WindOnshore", "2030"], "interest"
] = 0.025
accounting_converterUnits.loc[
    idx["OMFix", "global", "horizon", "WindOnshore", "2030"], "perUnitTotal"
] = 25.8
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
# load the profiles DataFrame, select the demand column
sourcesink_profile = profiles[["demand_R1"]]

# divide by 1000 to convert to GW, multiply with -1 because this is the
# REMix convention for accounting for sinks/demand
sourcesink_profile = sourcesink_profile.div(1e3).mul(-1)
# transpose DataFrame for needed format
sourcesink_profile = sourcesink_profile.T

# add columns and set them as index
sourcesink_profile["nodesData"] = "R1_data"
sourcesink_profile["years"] = "2030"
sourcesink_profile["techs"] = "Demand"
sourcesink_profile["commodity"] = "Elec"
sourcesink_profile["type"] = "fixed"
sourcesink_profile = sourcesink_profile.set_index(
    ["nodesData", "years", "techs", "commodity", "type"]
)

m.profile.add(sourcesink_profile, "sourcesink_profile")
sourcesink_profile.iloc[:, 0:8]
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
sourcesink_config.loc[idx["R1_data", :, :, :], "usesFixedProfile"] = 1
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
sourcesink_annualSum = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [m.set.nodesdata, m.set.yearssel, ["FuelImport"], ["CH4"]]
    )
)
sourcesink_annualSum.loc[idx["R1_data", :, :, :], "upper"] = np.inf
sourcesink_annualSum.loc[idx["R1_data", :, :, :], "lower"] = 0
sourcesink_annualSum = sourcesink_annualSum.dropna()

m.parameter.add(sourcesink_annualSum, "sourcesink_annualsum")
sourcesink_annualSum
# %%
# "sourcesink_config" (import configuration)
sourcesink_config = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [m.set.nodesdata, m.set.yearssel, ["FuelImport"], ["CH4"]]
    )
)
sourcesink_config.loc[idx["R1_data", :, :, :], "usesUpperSum"] = 1
sourcesink_config.loc[idx["R1_data", :, :, :], "usesLowerProfile"] = 1
sourcesink_config = sourcesink_config.dropna()

m.parameter.add(sourcesink_config, "sourcesink_config")
sourcesink_config
# %%
# "accounting_sourcesinkFlow"
# setting a cost for methane imports
accounting_sourcesinkFlow = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [["FuelCost"], ["global"], m.set.yearssel, ["FuelImport"], ["CH4"]]
    )
)
accounting_sourcesinkFlow["perFlow"] = 0.03060  # Mio EUR per GWh_ch CH4

m.parameter.add(accounting_sourcesinkFlow, "accounting_sourcesinkflow")
accounting_sourcesinkFlow
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
sourcesink_annualSum.loc[idx["R1_data", :, :, :], "lower"] = -np.inf
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
sourcesink_config.loc[idx["R1_data", :, :, :], "usesLowerSum"] = 1
sourcesink_config.loc[idx["R1_data", :, :, :], "usesUpperProfile"] = 1
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

# %%
# write all files to the datadir
m.write(fileformat="dat")
# %% [markdown]
# We have finished building our REMix data model now. In part b of this
# tutorial, we are looking into how we can execute it.
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
# importing dependencies
from remix.framework import Instance
import pandas as pd
import pathlib as pt

# reading in model built in `tutorial_101a_build`
_path_tut1_data = pt.Path("../tutorial_101/data")

if not _path_tut1_data.exists():
    raise IOError("You need to run tutorial 1a first!")

m = Instance.from_path(_path_tut1_data)

m.datadir = "./data"

# define often-used shortcut
idx = pd.IndexSlice
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
    index=pd.MultiIndex.from_product([["Battery"], m.set.yearssel])
)
converter_techParam.loc[idx["Battery", :], "lifeTime"] = 25
converter_techParam.loc[idx["Battery", :], "activityUpperLimit"] = 1

m.parameter.add(converter_techParam, "converter_techparam")
converter_techParam
# %%
# "converter_capacityParam"
converter_capacityParam = pd.DataFrame(
    index=pd.MultiIndex.from_product([m.set.nodesdata, m.set.yearssel, ["Battery"]])
)
converter_capacityParam.loc[idx["R1_data", :, "Battery"], "unitsUpperLimit"] = (
    30  # GW_el
)
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
        [["Battery"], m.set.yearssel, ["Charge", "Discharge"], ["Elec", "Elec_LiIon"]]
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
converter_coefficient = converter_coefficient.dropna(how="all")

m.parameter.add(converter_coefficient, "converter_coefficient")
converter_coefficient
# %%
# "accounting_converterUnits"
accounting_converterUnits = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [["Invest", "OMFix"], ["global"], ["horizon"], ["Battery"], m.set.yearssel]
    )
)

accounting_converterUnits.loc[
    idx["Invest", "global", "horizon", "Battery", "2030"], "perUnitBuild"
] = 50  # million EUR / unit
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
] = 0.75  # million EUR per unit and year
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
    index=pd.MultiIndex.from_product([["Battery"], m.set.yearssel])
)
storage_techParam.loc[idx["Battery", :], "lifeTime"] = 25
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
    index=pd.MultiIndex.from_product([["Battery"], m.set.yearssel, ["Elec_LiIon"]])
)
storage_sizeParam.loc[idx["Battery", :, "Elec_LiIon"], "size"] = 8  # GWh_ch/unit
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
    index=pd.MultiIndex.from_product([m.set.nodesdata, m.set.yearssel, ["Battery"]])
)
storage_reservoirParam.loc[idx["R1_data", :, "Battery"], "unitsUpperLimit"] = (
    30  # units
)
storage_reservoirParam = storage_reservoirParam.dropna()

m.parameter.add(storage_reservoirParam, "storage_reservoirparam")
storage_reservoirParam
# %%
# "accounting_storageUnits"
# accounting for costs of storage
accounting_storageUnits = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [["Invest", "OMFix"], ["global"], ["horizon"], ["Battery"], m.set.yearssel]
    )
)

accounting_storageUnits.loc[idx["Invest", :, :, :, :], "perUnitBuild"] = (
    105.5 * 8
)  # Since our storage unit can store 8 GWh we need to scale the million EUR/GWh value with 8
accounting_storageUnits.loc[idx["Invest", :, :, :, :], "useAnnuity"] = 1
accounting_storageUnits.loc[idx["Invest", :, :, :, :], "amorTime"] = 25
accounting_storageUnits.loc[idx["Invest", :, :, :, :], "interest"] = 0.06
accounting_storageUnits.loc[idx["OMFix", :, :, :, :], "perUnitTotal"] = (
    105.5 * 8 * 0.015
)
accounting_storageUnits = accounting_storageUnits.fillna(0)

m.parameter.add(accounting_storageUnits, "accounting_storageunits")
accounting_storageUnits
# %%
# write all files to `data/` directory
m.write(fileformat="dat")
# %% [markdown]
# That's it. We have successfully added a lithium-ion battery as storage
# technology to our model. We can now start a GAMS optimization run (part b).
# %% [markdown]
# (tutorial_103_label)=
#
# # Tutorial 103 - Inter-regional energy transfer
#
# The image below shows the overview of different regions, that will be modeled
# in this tutorial.
# The individual regions are modeled identically to the previous tutrial, but
# they are connected with links.
#
# <div style="text-align: center;">
#
# ![Model regions overview for tutorial 103](../../img/REMix_tutorial103.svg "Model regions overview for tutorial 103")
#
# Model regions overview for tutorial 103
#
# </div>
#
# <div style="text-align: center;">
#
# ![Per region model for tutorial 103](../../img/REMix_tutorial102.svg "Per region model for tutorial 103")
#
# Per region model for tutorial 103
#
# </div>
#
# ## Part a: setting up the model
#
# In this tutorial we have a closer look at **transfer technologies**.
# In the second tutorial we had added technologies to store the electrical
# energy from the volatile renewable sources.
# As a next step, in this tutorial we include the possibility to transfer
# energy between the model nodes.
#
# As done before, we will use the previous tutorial_102 as a base model by
# reading its files into an Instance object `m` and adding a transfer
# technology to it.

# %%
# import dependencies
from remix.framework import Instance
import pandas as pd
import numpy as np
import pathlib as pt

# reading previous tutorial into Instance object `m`
_path_tut102_data = pt.Path("../tutorial_101/data")

if not _path_tut102_data.exists():
    raise IOError("You need to run tutorial 102a first!")

m = Instance.from_path(_path_tut102_data)

m.datadir = "./data"

# define often-used shortcut
idx = pd.IndexSlice
# %% [markdown]
# ### Adding a transfer technology
#
# After loading the model and dependencies from our base model, we can now
# simply add the components of the transfer technology.
# In this tutorial, this will be a high-voltage direct current ("HVDC") grid.

# %% [markdown]
# First we need to set up the transfer connections in the data by defining the
# starting and ending node of each link
# %%
# "transfer_linkStartEnd"
link_names = ["R1__R2", "R2__R3", "R1__R3", "R3__R4"]
data_nodes = m.set.nodesdata

transfer_connections = pd.DataFrame(
    index=pd.MultiIndex.from_product([link_names, data_nodes])
)
transfer_connections.loc[idx["R1__R2", "R1_data"], ["start"]] = 1
transfer_connections.loc[idx["R1__R2", "R2_data"], ["end"]] = 1
transfer_connections.loc[idx["R2__R3", "R2_data"], ["start"]] = 1
transfer_connections.loc[idx["R2__R3", "R3_data"], ["end"]] = 1
transfer_connections.loc[idx["R1__R3", "R1_data"], ["start"]] = 1
transfer_connections.loc[idx["R1__R3", "R3_data"], ["end"]] = 1
transfer_connections.loc[idx["R3__R4", "R3_data"], ["start"]] = 1
transfer_connections.loc[idx["R3__R4", "R4_data"], ["end"]] = 1
transfer_connections = transfer_connections.dropna(how="all").fillna(0)

m.parameter.add(transfer_connections, "transfer_linkstartend")
transfer_connections
# %% [markdown]
# Next we define the lengths for each corridor.
# We can use different link types.
# %%
# "transfer_lengthParam"
link_types = ["land", "sea"]

transfer_lengths = pd.DataFrame(
    index=pd.MultiIndex.from_product([link_names, link_types])
)
transfer_lengths.loc[idx["R1__R2", "land"], ["length"]] = 1006.3
transfer_lengths.loc[idx["R2__R3", "land"], ["length"]] = 660.1
transfer_lengths.loc[idx["R1__R3", "land"], ["length"]] = 528.8
transfer_lengths.loc[idx["R3__R4", "land"], ["length"]] = 630.0
transfer_lengths.loc[idx["R1__R2", "sea"], ["length"]] = 0.0
transfer_lengths.loc[idx["R2__R3", "sea"], ["length"]] = 0.0
transfer_lengths.loc[idx["R1__R3", "sea"], ["length"]] = 0.0
transfer_lengths.loc[idx["R3__R4", "sea"], ["length"]] = 0.0
# transfer_lengths = transfer_lengths.dropna()

m.parameter.add(transfer_lengths, "transfer_lengthparam")
# %% [markdown]
# With the  corridors now defined, we can start adding links to be optimized
# to the model.
# %%
# "transfer_linksParam"
transfer_techs = ["HVDC"]

transfer_caps = pd.DataFrame(
    index=pd.MultiIndex.from_product([link_names, m.set.yearssel, transfer_techs])
)
transfer_caps.loc[:, ["linksUpperLimit"]] = (
    100  # Allow to build 100 GW for all links as the upper limit
)

m.parameter.add(transfer_caps, "transfer_linksparam")
transfer_caps
# %% [markdown]
# Define the technology information of the network
# %%
# "transfer_techParam"
tech_params = pd.DataFrame(
    index=pd.MultiIndex.from_product([transfer_techs, m.set.yearssel])
)
tech_params.loc[:, "lifeTime"] = 40
tech_params.loc[:, "flowUpperLimit"] = 1

m.parameter.add(tech_params, "transfer_techparam")
tech_params
# %% [markdown]
# Define the commodity and rated capacity of the network technology
# %%
# "transfer_coefficient"
commodity = ["Elec"]

transfer_coefficient = pd.DataFrame(
    index=pd.MultiIndex.from_product([transfer_techs, m.set.yearssel, commodity])
)
transfer_coefficient["coefficient"] = 1  # GWh / h

m.parameter.add(transfer_coefficient, "transfer_coefficient")
transfer_coefficient
# %% [markdown]
# Define the losses for the converter stations
# %%
# "transfer_coefPerFlow"
coef_per_flow = pd.DataFrame(
    index=pd.MultiIndex.from_product([transfer_techs, m.set.yearssel, commodity])
)
coef_per_flow[
    "coefPerFlow"
] = -0.014  # electrical losses of 14 MWh/h for each flow of 1 GWh/h

m.parameter.add(coef_per_flow, "transfer_coefperflow")
coef_per_flow

# %% [markdown]
# Define the losses for the links per km
# %%
# "transfer_coefPerLength"
coef_per_len = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [transfer_techs, m.set.yearssel, commodity, link_types]
    )
)
coef_per_len.loc[
    idx[:, :, :, "land"], idx["coefPerLength"]
] = -0.00004  # electrical losses of 40 kWh / h for each flow of 1 GWh / h and 1 km  length ~ 24 MWh / h for 600 km length
coef_per_len.loc[idx[:, :, :, "sea"], idx["coefPerLength"]] = -0.00003

m.parameter.add(coef_per_len, "transfer_coefperlength")
coef_per_len
# %% [markdown]
# Define indicators for each  built
# (for HVDC this is an AC/DC converter station at the beginning and end of the
# )
# %%
# "accounting_transferLinks"
cost_indicators = ["Invest", "OMFix"]
area = ["global"]
horizon = ["horizon"]

transfer_indicators = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [cost_indicators, area, horizon, transfer_techs, m.set.yearssel]
    )
)
transfer_indicators.loc[idx["Invest", "global", "horizon"], "perLinkBuild"] = 180
transfer_indicators.loc[idx["Invest", "global", "horizon"], "interest"] = 0.06
transfer_indicators.loc[idx["Invest", "global", "horizon"], "amorTime"] = 40
transfer_indicators.loc[idx["Invest", "global", "horizon"], "useAnnuity"] = 1
transfer_indicators.loc[idx["OMFix", "global", "horizon"], "perLinkTotal"] = 1.8
transfer_indicators = transfer_indicators.fillna(0)

m.parameter.add(transfer_indicators, "accounting_transferlinks")
transfer_indicators
# %% [markdown]
# Define indicators for each -km built (this needs the additional set for
# length-type modifiers, such as land and sea)
# %%
# "accounting_transferPerLength"
indicators_length = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [cost_indicators, area, horizon, transfer_techs, m.set.yearssel, link_types]
    )
)
indicators_length.loc[
    idx["Invest", "global", "horizon", :, :, "land"], "perLengthBuild"
] = 0.544
indicators_length.loc[idx["Invest", "global", "horizon", :, :, "land"], "interest"] = (
    0.06
)
indicators_length.loc[idx["Invest", "global", "horizon", :, :, "land"], "amorTime"] = 40
indicators_length.loc[
    idx["Invest", "global", "horizon", :, :, "land"], "useAnnuity"
] = 1
indicators_length.loc[
    idx["OMFix", "global", "horizon", :, :, "land"], "perLengthTotal"
] = 0.00544

indicators_length.loc[
    idx["Invest", "global", "horizon", :, :, "sea"], "perLengthBuild"
] = 0.975
indicators_length.loc[idx["Invest", "global", "horizon", :, :, "sea"], "interest"] = (
    0.06
)
indicators_length.loc[idx["Invest", "global", "horizon", :, :, "sea"], "amorTime"] = 40
indicators_length.loc[idx["Invest", "global", "horizon", :, :, "sea"], "useAnnuity"] = 1
indicators_length.loc[
    idx["OMFix", "global", "horizon", :, :, "sea"], "perLengthTotal"
] = 0.00975
indicators_length = indicators_length.fillna(0)

m.parameter.add(indicators_length, "accounting_transferperlength")
indicators_length
# %% [markdown]
# ### Adding additional demand and converters to the model
#
# Add profiles for PV and Wind onshore in for data nodes of R2, R3 and R4.
# %%
# "converter_activityProfile"
profiles = pd.read_csv("../_input/profiles.csv", index_col=0)

for data_node in ["R3_data", "R2_data", "R4_data"]:
    converter_activityProfile = profiles[["PV", "WindOnshore"]]

    # convert from MW to GW
    converter_activityProfile = converter_activityProfile.div(1e3).T

    converter_activityProfile = converter_activityProfile.div(
        converter_activityProfile.max(axis=1), axis=0
    )
    converter_activityProfile.index.names = ["techs"]

    converter_activityProfile["region"] = data_node
    converter_activityProfile["years"] = "2030"
    converter_activityProfile["type"] = "upper"
    converter_activityProfile = converter_activityProfile.reset_index().set_index(
        ["region", "years", "techs", "type"]
    )

    m.profile.add(converter_activityProfile, "converter_activityprofile")

m.profile.converter_activityprofile
converter_activityProfile.iloc[:, 0:8]
# %% [markdown]
# Add demand data nodes of R2, R3 and R4.
# %%
# "sourcesink_profile"
demand_R4_R2_CH = profiles[["demand_R4", "demand_R2", "demand_R3"]]

demand_R4_R2_CH = demand_R4_R2_CH.div(1e3).mul(-1)
# transpose DataFrame for needed format
demand_R4_R2_CH = demand_R4_R2_CH.T

demand_R4_R2_CH = demand_R4_R2_CH.rename(
    index={"demand_R4": "R4_data", "demand_R2": "R2_data", "demand_R3": "R3_data"}
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
# %%
# "sourcesink_config" (demand configuration)
demand_cfg = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [["R4_data", "R2_data", "R3_data"], m.set.yearssel, ["Demand"], ["Elec"]]
    )
)
demand_cfg["usesFixedProfile"] = 1

m.parameter.add(demand_cfg, "sourcesink_config")
demand_cfg
# %% [markdown]
# Add converter capacities to the data nodes of R2, R3 and R4.
# %%
# "converter_capacityParam"
converter_capacityParam = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [m.set.nodesdata, m.set.yearssel, ["CCGT", "PV", "WindOnshore"]]
    )
)
converter_capacityParam.loc[
    idx[["R2_data", "R3_data", "R4_data"], :, "CCGT"], "unitsUpperLimit"
] = 50  # GW_el
converter_capacityParam.loc[
    idx[["R2_data", "R3_data", "R4_data"], :, "CCGT"], "unitsLowerLimit"
] = 0  # GW_el
converter_capacityParam.loc[
    idx[["R2_data", "R3_data", "R4_data"], :, "WindOnshore"], "unitsUpperLimit"
] = 100  # GW_el
converter_capacityParam.loc[
    idx[["R2_data", "R3_data", "R4_data"], :, "WindOnshore"], "unitsLowerLimit"
] = 0  # GW_el
converter_capacityParam.loc[
    idx[["R2_data", "R3_data", "R4_data"], :, "PV"], "unitsUpperLimit"
] = 100  # GW_el
converter_capacityParam.loc[
    idx[["R2_data", "R3_data", "R4_data"], :, "PV"], "unitsLowerLimit"
] = 0  # GW_el

converter_capacityParam = converter_capacityParam.dropna(how="all")

m.parameter.add(converter_capacityParam, "converter_capacityparam")
converter_capacityParam
# %%
# limiting the annual sum of fuel imports into a model region
# "sourcesink_annualSum"
sourcesink_annualSum = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [m.set.nodesdata, m.set.yearssel, ["FuelImport"], ["CH4"]]
    )
)
sourcesink_annualSum.loc[idx[["R2_data", "R3_data", "R4_data"], :, :, :], "upper"] = (
    np.inf
)
sourcesink_annualSum = sourcesink_annualSum.dropna()

m.parameter.add(sourcesink_annualSum, "sourcesink_annualsum")
sourcesink_annualSum

# %%
# limiting annual sum of carbon emissions
sourcesink_annualSum = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [m.set.nodesdata, m.set.yearssel, ["Emission"], ["CO2"]]
    )
)
sourcesink_annualSum.loc[
    idx[["R2_data", "R3_data", "R4_data"], :, :, :], "lower"
] = -np.inf
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
sourcesink_config.loc[
    idx[["R2_data", "R3_data", "R4_data"], :, :, :], "usesLowerSum"
] = 1
sourcesink_config.loc[
    idx[["R2_data", "R3_data", "R4_data"], :, :, :], "usesUpperProfile"
] = 1
sourcesink_config = sourcesink_config.dropna()

m.parameter.add(sourcesink_config, "sourcesink_config")
sourcesink_config

# %%
sourcesink_config = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [m.set.nodesdata, m.set.yearssel, ["FuelImport"], ["CH4"]]
    )
)
sourcesink_config.loc[
    idx[["R2_data", "R3_data", "R4_data"], :, :, :], "usesUpperSum"
] = 1
sourcesink_config.loc[
    idx[["R2_data", "R3_data", "R4_data"], :, :, :], "usesLowerProfile"
] = 1
sourcesink_config = sourcesink_config.dropna()

m.parameter.add(sourcesink_config, "sourcesink_config")
sourcesink_config
# %%
# writing files to `data/` directory
m.write(fileformat="dat")
# %% [markdown]
# That's it. We have successfully added tlinks to our model.
# We can now start a GAMS optimization run (part b).
from remix.framework import Instance
import pathlib as pt

_path_tut103_data = pt.Path("../tutorial_101/data")

if not _path_tut103_data.exists():
    raise IOError("You need to run tutorial 103a first!")

m = Instance.from_path(_path_tut103_data)

# %%
# running GAMS from Python script
m.run(
    resultfile="tutorial_103",
    lo=3,
    names=1,
    postcalc=1,
    roundts=1,
)