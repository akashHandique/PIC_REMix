"""
REMix Pacific Islands sector-coupled energy system model
========================================================

A multi-node, multi-year (2020 / 2030 / 2040 / 2050) myopic optimisation of the
Pacific Island energy system, built on the DLR REMix framework. The model covers
14 Pacific data nodes and couples power, transport (land / marine / aviation),
cooking, industry, domestic hot water, water desalination and synthetic-fuel
production (hydrogen, ammonia, methanol, e-kerosene), together with battery,
thermal, water, hydrogen, ammonia, methanol, e-kerosene and CO2 storage and
inter-island ammonia / methanol / e-kerosene shipping links.

The objective is minimisation of total discounted system cost.

------------------------------------------------------------------------------
IMPORTANT
------------------------------------------------------------------------------
This file is a *structural* refactor of the original notebook-style script.
- All numeric parameters and all REMix keywords are preserved unchanged.
- The only behavioural cleanup is that node labels in the demand profiles are now
  derived systematically from the region code in each column name (see
  `_region_to_node`). This corrects a handful of copy/paste typos in the original
  rename maps (e.g. an `HDV_el_NE_*` key inside the `LDV_el` blocks) that silently
  left some node rows unlabelled. See README.md ("Notes on the refactor").
- Run a diff of the written `data/` folder against your original output if you
  want byte-for-byte confirmation before publishing.

Usage
-----
    python src/remix_pacific_model.py

Expects the input profile CSV at:
    _input/Copy of IP_2040_2050_14_PIC - Copy.csv
"""

import numpy as np
import pandas as pd

from remix.framework import Instance

# Frequently used pandas IndexSlice shortcut
idx = pd.IndexSlice


# ============================================================================
# Region / node definitions
# ============================================================================
# Two-letter (or short) region codes, in the canonical column order used by the
# input CSV. Every "<prefix>_<REGION>[_<year>]" column follows this ordering.
NODE_ORDER = [
    "CI", "FJ", "FSM", "KB", "MI", "NU",
    "NE", "PU", "PNG", "SA", "SI", "TA", "TU", "VU",
]

# Set of valid region codes, used to locate the region token inside a column name
REGION_CODES = set(NODE_ORDER)

# Full list of data-node names ("<REGION>_data") in canonical order
DATA_NODES = [f"{r}_data" for r in NODE_ORDER]

# Horizon years shared across storage technologies
STORAGE_YEARS = ["2030", "2040", "2050"]


def _region_to_node(column_name):
    """Map a profile column name (e.g. 'MDV_el_CI_2040') to its node ('CI_data').

    The region code is the first token in the underscore-split name that is a
    known region code. This is robust to the various prefixes used across years
    (demand_, MDV_, MDV_el_, 2W_el_, Marine_TH_, HC_B_, DHWE_, ...).
    """
    for token in column_name.split("_"):
        if token in REGION_CODES:
            return f"{token}_data"
    raise ValueError(f"No region code found in column name: {column_name!r}")


# ============================================================================
# Reusable helper functions
# ============================================================================
def add_demand_profile(m, profiles, prefix, suffix, year, commodity):
    """Add a fixed demand profile for one commodity / year across all nodes.

    Builds the column list as ``f"{prefix}_{region}{suffix}"`` for every region
    in NODE_ORDER, converts MWh -> GWh (``/1e3``), flips sign to a sink
    (``* -1``), transposes, labels and registers it as a `sourcesink_profile`.
    """
    columns = [f"{prefix}_{region}{suffix}" for region in NODE_ORDER]

    df = profiles[columns].div(1e3).mul(-1).T
    df = df.rename(index={c: _region_to_node(c) for c in columns})

    df["years"] = year
    df["techs"] = "Demand"
    df["commodity"] = commodity
    df["type"] = "fixed"
    df = df.set_index(["years", "techs", "commodity", "type"], append=True)

    m.profile.add(df, "sourcesink_profile")
    return df


def add_demand_config(m, commodity):
    """Register the `sourcesink_config` entry that consumes a fixed demand profile."""
    cfg = pd.DataFrame(
        index=pd.MultiIndex.from_product(
            [m.set.nodesdata, m.set.yearssel, ["Demand"], [commodity]]
        )
    )
    cfg.loc[idx[DATA_NODES, :, :, :], "usesFixedProfile"] = 1
    cfg = cfg.dropna()
    m.parameter.add(cfg, "sourcesink_config")
    return cfg


def add_fuel_import_limits(m, nodes, commodity, limits, lower_limit=0):
    """Register annual-sum upper/lower bounds for a fuel-import commodity."""
    ss = pd.DataFrame(
        index=pd.MultiIndex.from_product(
            [nodes, m.set.yearssel, ["FuelImport"], [commodity]]
        )
    )
    for node, limit in zip(nodes, limits):
        ss.loc[idx[node, :, :, :], "upper"] = limit
        ss.loc[idx[node, :, :, :], "lower"] = lower_limit
    ss = ss.dropna()
    m.parameter.add(ss, "sourcesink_annualsum")
    return ss


def add_fuel_import_config(m, nodes, commodity):
    """Register the `sourcesink_config` entry for a bounded fuel-import commodity."""
    cfg = pd.DataFrame(
        index=pd.MultiIndex.from_product(
            [m.set.nodesdata, m.set.yearssel, ["FuelImport"], [commodity]]
        )
    )
    cfg.loc[idx[nodes, :, :, :], "usesUpperSum"] = 1
    cfg.loc[idx[nodes, :, :, :], "usesLowerProfile"] = 1
    cfg = cfg.dropna()
    m.parameter.add(cfg, "sourcesink_config")
    return cfg


def add_fuel_prices(m, nodes, years, commodity, prices):
    """Register `accounting_sourcesinkflow` FuelCost per-flow prices."""
    flow = pd.DataFrame(
        index=pd.MultiIndex.from_product(
            [["FuelCost"], nodes, years, ["FuelImport"], [commodity]]
        )
    )
    for node, price in zip(nodes, prices):
        flow.loc[idx["FuelCost", node, :, :, :], "perFlow"] = price
    flow = flow.dropna()
    m.parameter.add(flow, "accounting_sourcesinkflow")
    return flow


def add_converter_capacity(m, year, capacity_limits):
    """Register `converter_capacityparam` for one build year.

    `capacity_limits` maps node -> {tech: (lower_limit, upper_limit)} in GW.
    """
    all_techs = list({tech for node in capacity_limits for tech in capacity_limits[node]})
    cap = pd.DataFrame(
        index=pd.MultiIndex.from_product([m.set.nodesdata, [year], all_techs])
    )
    for node, techs in capacity_limits.items():
        for tech, (lower, upper) in techs.items():
            cap.loc[idx[node, :, tech], "unitsLowerLimit"] = lower
            cap.loc[idx[node, :, tech], "unitsUpperLimit"] = upper
    cap = cap.dropna(how="all")
    m.parameter.add(cap, "converter_capacityparam")
    return cap


def add_converter_unit_costs(m, year, global_specs, node_specs=None):
    """Register accounting_converterunits for one year.

    global_specs : dict[tech] -> cost dict, written against the "global" label.
    node_specs   : optional dict[node] -> dict[tech] -> cost dict, written
                   against specific node labels (used for per-country AEL).
    """
    techs = list(global_specs.keys())
    if node_specs:
        extra = {t for specs in node_specs.values() for t in specs}
        techs = list(dict.fromkeys(techs + list(extra)))

    frame = _new_converter_units_frame(techs, year)

    for tech, spec in global_specs.items():
        _set_converter_unit_cost(frame, "global", tech, year, spec)

    if node_specs:
        for node, specs in node_specs.items():
            for tech, spec in specs.items():
                _set_converter_unit_cost(frame, node, tech, year, spec)

    frame = frame.fillna(0)
    m.parameter.add(frame, "accounting_converterunits")
    return frame


def add_transfer_link_costs(m, techs, per_link_build, per_flow_along):
    """Register accounting_transferlinks for one carrier (port_A/M/F)."""
    cost_indicators = ["Invest", "OMFix", "OMVar"]
    frame = pd.DataFrame(
        index=pd.MultiIndex.from_product(
            [cost_indicators, ["global"], ["horizon"], techs, Transfer_year]
        )
    ).sort_index()
    frame.loc[idx["Invest", "global", "horizon"], "perLinkBuild"] = per_link_build
    frame.loc[idx["Invest", "global", "horizon"], "interest"] = 0.06
    frame.loc[idx["Invest", "global", "horizon"], "amorTime"] = 40
    frame.loc[idx["Invest", "global", "horizon"], "useAnnuity"] = 1
    frame.loc[idx["OMFix", "global", "horizon"], "perLinkTotal"] = 0
    frame.loc[idx["OMVar", "global", "horizon"], "perFlowAlong"] = per_flow_along
    frame = frame.fillna(0)
    m.parameter.add(frame, "accounting_transferlinks")
    return frame


def _new_converter_units_frame(techs, year):
    """Empty accounting_converterunits frame for the given techs and year."""
    return pd.DataFrame(
        index=pd.MultiIndex.from_product(
            [["Invest", "OMFix"], DATA_NODES, ["horizon"], techs, [year]]
        )
    ).sort_index()


def _set_converter_unit_cost(frame, node, tech, year, spec):
    """Write one technology's Invest + OMFix rows into the frame."""
    frame.loc[idx["Invest", node, "horizon", tech, year], "perUnitBuild"] = spec["perUnitBuild"]
    frame.loc[idx["Invest", node, "horizon", tech, year], "useAnnuity"] = spec["useAnnuity"]
    frame.loc[idx["Invest", node, "horizon", tech, year], "amorTime"] = spec["amorTime"]
    frame.loc[idx["Invest", node, "horizon", tech, year], "interest"] = spec["interest"]
    frame.loc[idx["OMFix", node, "horizon", tech, year], "perUnitTotal"] = spec["perUnitTotal"]


def _annuity(per_unit_build, om_fix, amor_time, interest=0.06, use_annuity=1):
    """Convenience constructor for a converter cost spec dict."""
    return {
        "perUnitBuild": per_unit_build,
        "useAnnuity": use_annuity,
        "amorTime": amor_time,
        "interest": interest,
        "perUnitTotal": om_fix,
    }


def _zero_cost():
    """Cost spec for free / non-invested technologies (transport end-uses)."""
    return {
        "perUnitBuild": 0,
        "useAnnuity": 0,
        "amorTime": 0,
        "interest": 0,
        "perUnitTotal": 0,
    }


def add_activity_profiles(m, profiles, year, techs):
    """Build and register normalised renewable activity profiles for one year.

    For each data node, selects the per-tech CSV columns ("<tech>_<region>"),
    converts MW->GW, transposes, normalises each row by its maximum, and adds
    the result under the (region, year, tech, "upper") index.
    """
    for data_node in DATA_NODES:
        region_code = data_node.split("_")[0]
        techs_region = [f"{t}_{region_code}" for t in techs]

        activity_profile = profiles[techs_region].div(1e3).T
        activity_profile.index = techs
        activity_profile = activity_profile.div(
            activity_profile.max(axis=1), axis=0
        )
        activity_profile.index.names = ["techs"]

        activity_profile["region"] = data_node
        activity_profile["years"] = year
        activity_profile["type"] = "upper"
        activity_profile = activity_profile.reset_index().set_index(
            ["region", "years", "techs", "type"]
        )

        m.profile.add(activity_profile, "converter_activityprofile")


def add_storage_tech(
    m,
    tech,
    years,
    commodity_stored,
    converter_coeff,
    converter_lifetime,
    converter_activity_upper,
    converter_unit_upper,
    storage_lifetime,
    storage_level_upper,
    storage_size,
    storage_reservoir_upper,
    converter_costs_by_year,
    storage_costs,
    converter_var_costs=None,
    converter_var_costs_activity="Powergen",
):
    """Register all REMix parameters for one storage technology.

    Parameters
    ----------
    tech : str
        Technology name (e.g. "Battery", "THSS", "H2_storage").
    years : list[str]
        Build years this technology is available.
    commodity_stored : str
        The internal storage commodity (e.g. "Elec_LiIon", "Heat_T").
    converter_coeff : dict
        Mapping of (activity, commodity) -> coefficient for the converter.
        Example: {("Charge", "Elec"): -1, ("Charge", "Elec_LiIon"): 0.975, ...}
    converter_lifetime : int
        Converter lifetime in years.
    converter_activity_upper : float
        Converter activityUpperLimit (0 or 1).
    converter_unit_upper : float
        unitsUpperLimit for converter capacity.
    storage_lifetime : int
        Storage reservoir lifetime in years.
    storage_level_upper : float
        Storage levelUpperLimit.
    storage_size : float
        Size of each storage unit (GWh/unit).
    storage_reservoir_upper : float
        unitsUpperLimit for storage reservoir.
    converter_costs_by_year : dict[str, dict]
        Mapping of year -> cost spec dict (from _annuity / _zero_cost).
    storage_costs : dict
        Single cost spec dict applied uniformly across all years (Invest + OMFix).
        Keys: perUnitBuild, useAnnuity, amorTime, interest, perUnitTotal.
    converter_var_costs : dict[str, float], optional
        Mapping of year -> perActivity OMVar cost for the converter.
    converter_var_costs_activity : str, optional
        The converter activity name used for variable O&M (default: \"Powergen\").
        Set to \"Charge\" for storage technologies like co2_storage.
    """
    # --- converter_techparam ---
    tech_param = pd.DataFrame(
        index=pd.MultiIndex.from_product([[tech], years])
    )
    tech_param.loc[idx[tech, :], "lifeTime"] = converter_lifetime
    tech_param.loc[idx[tech, :], "activityUpperLimit"] = converter_activity_upper
    m.parameter.add(tech_param, "converter_techparam")

    # --- converter_capacityparam ---
    cap_param = pd.DataFrame(
        index=pd.MultiIndex.from_product([m.set.nodesdata, years, [tech]])
    )
    cap_param.loc[idx[DATA_NODES, :, tech], "unitsUpperLimit"] = converter_unit_upper
    cap_param = cap_param.dropna()
    m.parameter.add(cap_param, "converter_capacityparam")

    # --- converter_coefficient ---
    activities = list({a for a, _ in converter_coeff})
    commodities = list({c for _, c in converter_coeff})
    coeff_frame = pd.DataFrame(
        index=pd.MultiIndex.from_product([[tech], years, activities, commodities])
    )
    for (activity, commodity), value in converter_coeff.items():
        coeff_frame.loc[idx[tech, :, activity, commodity], "coefficient"] = value
    coeff_frame = coeff_frame.dropna(how="all")
    m.parameter.add(coeff_frame, "converter_coefficient")

    # --- accounting_converterunits (per year) ---
    for year, spec in converter_costs_by_year.items():
        frame = pd.DataFrame(
            index=pd.MultiIndex.from_product(
                [["Invest", "OMFix"], DATA_NODES, ["horizon"], [tech], [year]]
            )
        ).sort_index()
        _set_converter_unit_cost(frame, "global", tech, year, spec)
        frame = frame.fillna(0)
        m.parameter.add(frame, "accounting_converterunits")

    # --- accounting_converteractivity (optional variable O&M) ---
    if converter_var_costs:
        for year, per_activity in converter_var_costs.items():
            var_frame = pd.DataFrame(
                index=pd.MultiIndex.from_product(
                    [["OMVar"], ["global"], ["horizon"], [tech], [year], [converter_var_costs_activity]]
                )
            ).sort_index()
            var_frame.loc[
                idx["OMVar", "global", "horizon", tech, year, converter_var_costs_activity], "perActivity"
            ] = per_activity
            var_frame = var_frame.fillna(0)
            m.parameter.add(var_frame, "accounting_converteractivity")

    # --- storage_techparam ---
    storage_tech_param = pd.DataFrame(
        index=pd.MultiIndex.from_product([[tech], years])
    )
    storage_tech_param.loc[idx[tech, :], "lifeTime"] = storage_lifetime
    storage_tech_param.loc[idx[tech, :], "levelUpperLimit"] = storage_level_upper
    m.parameter.add(storage_tech_param, "storage_techparam")

    # --- storage_sizeparam ---
    size_param = pd.DataFrame(
        index=pd.MultiIndex.from_product([[tech], years, [commodity_stored]])
    )
    size_param.loc[idx[tech, :, commodity_stored], "size"] = storage_size
    size_param = size_param.dropna()
    m.parameter.add(size_param, "storage_sizeparam")

    # --- storage_reservoirparam ---
    reservoir_param = pd.DataFrame(
        index=pd.MultiIndex.from_product([m.set.nodesdata, years, [tech]])
    )
    reservoir_param.loc[idx[DATA_NODES, :, tech], "unitsUpperLimit"] = storage_reservoir_upper
    reservoir_param = reservoir_param.dropna()
    m.parameter.add(reservoir_param, "storage_reservoirparam")

    # --- accounting_storageunits ---
    if storage_costs is not None:
        storage_frame = pd.DataFrame(
            index=pd.MultiIndex.from_product(
                [["Invest", "OMFix"], DATA_NODES, ["horizon"], [tech], years]
            )
        )
        storage_frame.loc[idx["Invest", :, :, :, :], "perUnitBuild"] = storage_costs["perUnitBuild"]
        storage_frame.loc[idx["Invest", :, :, :, :], "useAnnuity"] = storage_costs["useAnnuity"]
        storage_frame.loc[idx["Invest", :, :, :, :], "amorTime"] = storage_costs["amorTime"]
        storage_frame.loc[idx["Invest", :, :, :, :], "interest"] = storage_costs["interest"]
        storage_frame.loc[idx["OMFix", :, :, :, :], "perUnitTotal"] = storage_costs["perUnitTotal"]
        storage_frame = storage_frame.fillna(0)
        m.parameter.add(storage_frame, "accounting_storageunits")


def add_storage_accounting_by_year(m, tech, years, cost_by_year, index_nodes=DATA_NODES):
    """Register accounting_storageunits with per-year cost variation.

    Used when perUnitBuild differs between build years (e.g. Battery).
    `cost_by_year` maps year -> {perUnitBuild, useAnnuity, amorTime, interest, perUnitTotal}.
    """
    for year, spec in cost_by_year.items():
        frame = pd.DataFrame(
            index=pd.MultiIndex.from_product(
                [["Invest", "OMFix"], index_nodes, ["horizon"], [tech], [year]]
            )
        )
        frame.loc[idx["Invest", :, :, :, year], "perUnitBuild"] = spec["perUnitBuild"]
        frame.loc[idx["Invest", :, :, :, year], "useAnnuity"] = spec["useAnnuity"]
        frame.loc[idx["Invest", :, :, :, year], "amorTime"] = spec["amorTime"]
        frame.loc[idx["Invest", :, :, :, year], "interest"] = spec["interest"]
        frame.loc[idx["OMFix", :, :, :, year], "perUnitTotal"] = spec["perUnitTotal"]
        frame = frame.fillna(0)
        m.parameter.add(frame, "accounting_storageunits")


# ============================================================================
# Model initialisation
# ============================================================================
m = Instance()

# Directory the model data is written to ("./data" is the REMix default)
m.datadir = "./data"

# Load input profile data (hourly resource / demand profiles)
profiles = pd.read_csv("../_input/Copy of IP_2040_2050_14_PIC - Copy.csv", index_col=0)


# ============================================================================
# Node mapping (data nodes -> model nodes)
# ============================================================================
df = pd.DataFrame(
    [
        ["CI_data", "CI_model", 1],
        ["FJ_data", "FJ_model", 1],
        ["FSM_data", "FSM_model", 1],
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
        ["VU_data", "VU_model", 1],
    ]
)
df.columns = ["nodesData", "nodesModel", "aggregate"]
df = df.set_index(["nodesData", "nodesModel"])
df["aggregate"] = ""
df.columns = [""]

m.map.add(df, "aggregatenodesmodel")

# Derive the data / model region sets from the mapping
m.set.add(list(set(m.map.aggregatenodesmodel.index.get_level_values(0))), "nodesdata")
m.set.add(list(set(m.map.aggregatenodesmodel.index.get_level_values(1))), "nodesmodel")

# Years considered and years optimised
m.set.add(["2020", "2030", "2040", "2050"], "years")
m.set.add(["2020", "2030", "2040", "2050"], "yearssel")


# ============================================================================
# Objective function and indicator accounting
# ============================================================================
accounting_indicatorBounds = pd.DataFrame(
    index=pd.MultiIndex.from_product([["global"], ["horizon"], ["SystemCost"]])
)
accounting_indicatorBounds["obj"] = -1         # minimisation of system costs
accounting_indicatorBounds["discount"] = 0.08  # social discount rate
m.parameter.add(accounting_indicatorBounds, "accounting_indicatorbounds")

accounting_perIndicator = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [
            ["SystemCost"],
            ["Invest", "OMFix", "FuelCost", "OMVar"],
            ["global"],
            m.set.yearssel,
        ]
    )
)
accounting_perIndicator["perIndicator"] = 1
m.parameter.add(accounting_perIndicator, "accounting_perindicator")


# ============================================================================
# Converter technology parameters (lifetime / activity limits) per build year
# ============================================================================
# --- 2020 stock technologies ---
tech_specs = {
    "DG": {"lifeTime": 25, "activityUpperLimit": 1},
    "NG_plant": {"lifeTime": 25, "activityUpperLimit": 1},
    "BG_B": {"lifeTime": 25, "activityUpperLimit": 0},
    "PV_B": {"lifeTime": 25, "activityUpperLimit": 0},
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

converter_techParam = pd.DataFrame(
    index=pd.MultiIndex.from_product([list(tech_specs.keys()), ["2020"]])
)
for tech, specs in tech_specs.items():
    converter_techParam.loc[idx[tech], "lifeTime"] = specs["lifeTime"]
    converter_techParam.loc[idx[tech], "activityUpperLimit"] = specs["activityUpperLimit"]
m.parameter.add(converter_techParam, "converter_techparam")

# --- 2030 new technologies ---
tech_specss = {
    "BG_N": {"lifeTime": 25, "activityUpperLimit": 0},
    "PV_N": {"lifeTime": 25, "activityUpperLimit": 0},
    "WindOnshore_N": {"lifeTime": 25, "activityUpperLimit": 0},
    "Hydro_N": {"lifeTime": 50, "activityUpperLimit": 0},
    "Wave_N": {"lifeTime": 25, "activityUpperLimit": 0},
    "DW_Electric_converter_D": {"lifeTime": 25, "activityUpperLimit": 1},
    "WindOffshore_N": {"lifeTime": 25, "activityUpperLimit": 0},
}
converter_techParam = pd.DataFrame(
    index=pd.MultiIndex.from_product([list(tech_specss.keys()), ["2030"]])
)
for tech, specs in tech_specss.items():
    converter_techParam.loc[idx[tech], "lifeTime"] = specs["lifeTime"]
    converter_techParam.loc[idx[tech], "activityUpperLimit"] = specs["activityUpperLimit"]
m.parameter.add(converter_techParam, "converter_techparam")

# --- 2040 new technologies (incl. activityLowerLimit) ---
tech_specss = {
    "BG_N": {"lifeTime": 25, "activityUpperLimit": 0, "activityLowerLimit": 0},
    "PV_N": {"lifeTime": 25, "activityUpperLimit": 0, "activityLowerLimit": 0},
    "WindOnshore_N": {"lifeTime": 25, "activityUpperLimit": 0, "activityLowerLimit": 0},
    "Hydro_N": {"lifeTime": 50, "activityUpperLimit": 0, "activityLowerLimit": 0},
    "Wave_N": {"lifeTime": 25, "activityUpperLimit": 0, "activityLowerLimit": 0},
    "ST_N": {"lifeTime": 25, "activityUpperLimit": 0, "activityLowerLimit": 0},
    "DW_Electric_converter_2": {"lifeTime": 25, "activityUpperLimit": 1, "activityLowerLimit": 0},
    "LDV_BF": {"lifeTime": 25, "activityUpperLimit": 1, "activityLowerLimit": 0},
    "RO": {"lifeTime": 25, "activityUpperLimit": 1, "activityLowerLimit": 0},
    "AEL": {"lifeTime": 25, "activityUpperLimit": 1, "activityLowerLimit": 0.20},
    "Ammonia_synthesis": {"lifeTime": 25, "activityUpperLimit": 1, "activityLowerLimit": 0},
    "DAC": {"lifeTime": 25, "activityUpperLimit": 1, "activityLowerLimit": 0},
    "Methanol_synthesis": {"lifeTime": 25, "activityUpperLimit": 1, "activityLowerLimit": 0},
    "HP": {"lifeTime": 25, "activityUpperLimit": 1, "activityLowerLimit": 0},
    "FTL": {"lifeTime": 25, "activityUpperLimit": 1, "activityLowerLimit": 0},
    "LDV_el": {"lifeTime": 25, "activityUpperLimit": 1, "activityLowerLimit": 0},
    "HDV_el": {"lifeTime": 25, "activityUpperLimit": 1, "activityLowerLimit": 0},
    "HDV_BF": {"lifeTime": 25, "activityUpperLimit": 1, "activityLowerLimit": 0},
    "MDV_el": {"lifeTime": 25, "activityUpperLimit": 1, "activityLowerLimit": 0},
    "MDV_BF": {"lifeTime": 25, "activityUpperLimit": 1, "activityLowerLimit": 0},
    "Two_wheel_el": {"lifeTime": 25, "activityUpperLimit": 1, "activityLowerLimit": 0},
    "Bus_el": {"lifeTime": 25, "activityUpperLimit": 1, "activityLowerLimit": 0},
    "Marine_e": {"lifeTime": 25, "activityUpperLimit": 1, "activityLowerLimit": 0},
    "Aviation_el": {"lifeTime": 25, "activityUpperLimit": 1, "activityLowerLimit": 0},
    "Aviation_e": {"lifeTime": 25, "activityUpperLimit": 1, "activityLowerLimit": 0},
    "cook_el": {"lifeTime": 25, "activityUpperLimit": 1, "activityLowerLimit": 0},
    "cook_LPG": {"lifeTime": 25, "activityUpperLimit": 1, "activityLowerLimit": 0},
    "Industry_EH": {"lifeTime": 25, "activityUpperLimit": 1, "activityLowerLimit": 0},
    "DW_heat": {"lifeTime": 25, "activityUpperLimit": 1, "activityLowerLimit": 0},
    "Dummy_Ammonia": {"lifeTime": 25, "activityUpperLimit": 1, "activityLowerLimit": 0},
    "Dummy_Methanol": {"lifeTime": 25, "activityUpperLimit": 1, "activityLowerLimit": 0},
    "Industry_EL": {"lifeTime": 25, "activityUpperLimit": 1, "activityLowerLimit": 0},
    "WindOffshore_N": {"lifeTime": 25, "activityUpperLimit": 0, "activityLowerLimit": 0},
    "Ship_BEV": {"lifeTime": 25, "activityUpperLimit": 1, "activityLowerLimit": 0},
}
converter_techParam = pd.DataFrame(
    index=pd.MultiIndex.from_product([list(tech_specss.keys()), ["2040"]])
)
for tech, specs in tech_specss.items():
    converter_techParam.loc[idx[tech], "lifeTime"] = specs["lifeTime"]
    converter_techParam.loc[idx[tech], "activityUpperLimit"] = specs["activityUpperLimit"]
    converter_techParam.loc[idx[tech], "activityLowerLimit"] = specs["activityLowerLimit"]
m.parameter.add(converter_techParam, "converter_techparam")

# --- 2050 new technologies (identical to 2040 except Dummy_* lifeTime = 50) ---
tech_specss = {
    "BG_N": {"lifeTime": 25, "activityUpperLimit": 0, "activityLowerLimit": 0},
    "PV_N": {"lifeTime": 25, "activityUpperLimit": 0, "activityLowerLimit": 0},
    "WindOnshore_N": {"lifeTime": 25, "activityUpperLimit": 0, "activityLowerLimit": 0},
    "Hydro_N": {"lifeTime": 50, "activityUpperLimit": 0, "activityLowerLimit": 0},
    "Wave_N": {"lifeTime": 25, "activityUpperLimit": 0, "activityLowerLimit": 0},
    "ST_N": {"lifeTime": 25, "activityUpperLimit": 0, "activityLowerLimit": 0},
    "DW_Electric_converter_2": {"lifeTime": 25, "activityUpperLimit": 1, "activityLowerLimit": 0},
    "LDV_BF": {"lifeTime": 25, "activityUpperLimit": 1, "activityLowerLimit": 0},
    "RO": {"lifeTime": 25, "activityUpperLimit": 1, "activityLowerLimit": 0},
    "AEL": {"lifeTime": 25, "activityUpperLimit": 1, "activityLowerLimit": 0.20},
    "Ammonia_synthesis": {"lifeTime": 25, "activityUpperLimit": 1, "activityLowerLimit": 0},
    "DAC": {"lifeTime": 25, "activityUpperLimit": 1, "activityLowerLimit": 0},
    "Methanol_synthesis": {"lifeTime": 25, "activityUpperLimit": 1, "activityLowerLimit": 0},
    "HP": {"lifeTime": 25, "activityUpperLimit": 1, "activityLowerLimit": 0},
    "FTL": {"lifeTime": 25, "activityUpperLimit": 1, "activityLowerLimit": 0},
    "LDV_el": {"lifeTime": 25, "activityUpperLimit": 1, "activityLowerLimit": 0},
    "HDV_el": {"lifeTime": 25, "activityUpperLimit": 1, "activityLowerLimit": 0},
    "HDV_BF": {"lifeTime": 25, "activityUpperLimit": 1, "activityLowerLimit": 0},
    "MDV_el": {"lifeTime": 25, "activityUpperLimit": 1, "activityLowerLimit": 0},
    "MDV_BF": {"lifeTime": 25, "activityUpperLimit": 1, "activityLowerLimit": 0},
    "Two_wheel_el": {"lifeTime": 25, "activityUpperLimit": 1, "activityLowerLimit": 0},
    "Bus_el": {"lifeTime": 25, "activityUpperLimit": 1, "activityLowerLimit": 0},
    "Marine_e": {"lifeTime": 25, "activityUpperLimit": 1, "activityLowerLimit": 0},
    "Aviation_el": {"lifeTime": 25, "activityUpperLimit": 1, "activityLowerLimit": 0},
    "Aviation_e": {"lifeTime": 25, "activityUpperLimit": 1, "activityLowerLimit": 0},
    "cook_el": {"lifeTime": 25, "activityUpperLimit": 1, "activityLowerLimit": 0},
    "cook_LPG": {"lifeTime": 25, "activityUpperLimit": 1, "activityLowerLimit": 0},
    "Industry_EH": {"lifeTime": 25, "activityUpperLimit": 1, "activityLowerLimit": 0},
    "DW_heat": {"lifeTime": 25, "activityUpperLimit": 1, "activityLowerLimit": 0},
    "Dummy_Ammonia": {"lifeTime": 50, "activityUpperLimit": 1, "activityLowerLimit": 0},
    "Dummy_Methanol": {"lifeTime": 50, "activityUpperLimit": 1, "activityLowerLimit": 0},
    "Industry_EL": {"lifeTime": 25, "activityUpperLimit": 1, "activityLowerLimit": 0},
    "WindOffshore_N": {"lifeTime": 25, "activityUpperLimit": 0, "activityLowerLimit": 0},
    "Ship_BEV": {"lifeTime": 25, "activityUpperLimit": 1, "activityLowerLimit": 0},
}
converter_techParam = pd.DataFrame(
    index=pd.MultiIndex.from_product([list(tech_specss.keys()), ["2050"]])
)
for tech, specs in tech_specss.items():
    converter_techParam.loc[idx[tech], "lifeTime"] = specs["lifeTime"]
    converter_techParam.loc[idx[tech], "activityUpperLimit"] = specs["activityUpperLimit"]
    converter_techParam.loc[idx[tech], "activityLowerLimit"] = specs["activityLowerLimit"]
m.parameter.add(converter_techParam, "converter_techparam")


# ============================================================================
# Converter capacity parameters (per-node lower/upper unit limits, in GW)
# ============================================================================
# ---- 2020 installed / existing capacities ------------------------------------
CAPACITY_LIMITS_2020 = {
    "CI_data": {
        "DG": (0.018, 0.018),
        "PV_B": (0.0052, 0.0052),
        "MDV": (0, 1000), "HDV": (0, 1000), "LDV": (0, 1000), "Bus": (0, 1000),
        "Two_wheel": (0, 1000), "Aviation": (0, 1000), "Marine": (0, 1000),
        "cook_b": (0, 1000), "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000), "DW_Electric_converter": (0, 1000),
        "HFO": (0, 1000),
    },
    "FJ_data": {
        "DG": (0.172, 0.172),
        "BG_B": (0.0580, 0.0580),
        "PV_B": (0.0090, 0.0090),
        "Hydro_B": (0.0625, 0.0625),
        "MDV": (0, 1000), "HDV": (0, 1000), "LDV": (0, 1000), "Bus": (0, 1000),
        "Two_wheel": (0, 1000), "Aviation": (0, 1000), "Marine": (0, 1000),
        "cook_b": (0, 1000), "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000), "DW_Electric_converter": (0, 1000),
        "HFO": (0, 1000),
    },
    "FSM_data": {
        "DG": (0.0388, 0.0388),
        "PV_B": (0.0028, 0.0028),
        "WindOnshore_B": (0.0009, 0.0009),
        "Hydro_B": (0.000225, 0.000225),
        "MDV": (0, 1000), "HDV": (0, 1000), "LDV": (0, 1000), "Bus": (0, 1000),
        "Two_wheel": (0, 1000), "Aviation": (0, 1000), "Marine": (0, 1000),
        "cook_b": (0, 1000), "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000), "DW_Electric_converter": (0, 1000),
        "HFO": (0, 1000),
    },
    "KB_data": {
        "DG": (0.0066, 0.0067),
        "PV_B": (0.0030, 0.0030),
        "MDV": (0, 1000), "HDV": (0, 1000), "LDV": (0, 1000), "Bus": (0, 1000),
        "Two_wheel": (0, 1000), "Aviation": (0, 1000), "Marine": (0, 1000),
        "cook_b": (0, 1000), "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000), "DW_Electric_converter": (0, 1000),
        "HFO": (0, 1000),
    },
    "MI_data": {
        "DG": (0.0287, 0.0287),
        "PV_B": (0.0017, 0.0017),
        "MDV": (0, 1000), "HDV": (0, 1000), "LDV": (0, 1000), "Bus": (0, 1000),
        "Two_wheel": (0, 1000), "Aviation": (0, 1000), "Marine": (0, 1000),
        "cook_b": (0, 1000), "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000), "DW_Electric_converter": (0, 1000),
        "HFO": (0, 1000),
    },
    "NU_data": {
        "DG": (0.0245, 0.0245),
        "PV_B": (0.0028, 0.0028),
        "MDV": (0, 1000), "HDV": (0, 1000), "LDV": (0, 1000), "Bus": (0, 1000),
        "Two_wheel": (0, 1000), "Aviation": (0, 1000), "Marine": (0, 1000),
        "cook_b": (0, 1000), "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000), "DW_Electric_converter": (0, 1000),
        "HFO": (0, 1000),
    },
    "NE_data": {
        "DG": (0.0021, 0.0021),
        "PV_B": (0.0010, 0.0011),
        "MDV": (0, 1000), "HDV": (0, 1000), "LDV": (0, 1000), "Bus": (0, 1000),
        "Two_wheel": (0, 1000), "Aviation": (0, 1000), "Marine": (0, 1000),
        "cook_b": (0, 1000), "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000), "DW_Electric_converter": (0, 1000),
        "HFO": (0, 1000),
    },
    "PU_data": {
        "DG": (0.0303, 0.0303),
        "PV_B": (0.0030, 0.0032),
        "MDV": (0, 1000), "HDV": (0, 1000), "LDV": (0, 1000), "Bus": (0, 1000),
        "Two_wheel": (0, 1000), "Aviation": (0, 1000), "Marine": (0, 1000),
        "cook_b": (0, 1000), "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000), "DW_Electric_converter": (0, 1000),
        "HFO": (0, 1000),
    },
    "PNG_data": {
        "DG": (0.280, 0.350),
        "NG_plant": (0.082, 0.082),
        "BG_B": (0.0182, 0.0182),
        "PV_B": (0.0031, 0.0031),
        "Hydro_B": (0.115, 0.115),
        "Geothermal_B": (0.011, 0.011),
        "MDV": (0, 1000), "HDV": (0, 1000), "LDV": (0, 1000), "Bus": (0, 1000),
        "Two_wheel": (0, 1000), "Aviation": (0, 1000), "Marine": (0, 1000),
        "cook_b": (0, 1000), "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000), "DW_Electric_converter": (0, 1000),
        "HFO": (0, 1000),
    },
    "SA_data": {
        "DG": (0.0315, 0.0315),
        "BG_B": (0.0011, 0.0011),
        "PV_B": (0.0138, 0.0138),
        "WindOnshore_B": (0.0005, 0.0005),
        "Hydro_B": (0.0063, 0.0063),
        "MDV": (0, 1000), "HDV": (0, 1000), "LDV": (0, 1000), "Bus": (0, 1000),
        "Two_wheel": (0, 1000), "Aviation": (0, 1000), "Marine": (0, 1000),
        "cook_b": (0, 1000), "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000), "DW_Electric_converter": (0, 1000),
        "HFO": (0, 1000),
    },
    "SI_data": {
        "DG": (0.0527, 0.0527),
        "BG_B": (0.0008, 0.0008),
        "PV_B": (0.0023, 0.0023),
        "Hydro_B": (0.00018, 0.00018),
        "MDV": (0, 1000), "HDV": (0, 1000), "LDV": (0, 1000), "Bus": (0, 1000),
        "Two_wheel": (0, 1000), "Aviation": (0, 1000), "Marine": (0, 1000),
        "cook_b": (0, 1000), "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000), "DW_Electric_converter": (0, 1000),
        "HFO": (0, 1000),
    },
    "TA_data": {
        "DG": (0.0167, 0.0167),
        "PV_B": (0.0071, 0.0071),
        "WindOnshore_B": (0.00151, 0.00151),
        "MDV": (0, 1000), "HDV": (0, 1000), "LDV": (0, 1000), "Bus": (0, 1000),
        "Two_wheel": (0, 1000), "Aviation": (0, 1000), "Marine": (0, 1000),
        "cook_b": (0, 1), "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000), "DW_Electric_converter": (0, 1000),
        "HFO": (0, 1000),
    },
    "TU_data": {
        "DG": (0.003, 0.003),
        "PV_B": (0.0029, 0.0029),
        "MDV": (0, 1000), "HDV": (0, 1000), "LDV": (0, 1000), "Bus": (0, 1000),
        "Two_wheel": (0, 1000), "Aviation": (0, 1000), "Marine": (0, 1000),
        "cook_b": (0, 1), "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000), "DW_Electric_converter": (0, 1000),
        "HFO": (0, 1000),
    },
    "VU_data": {
        "DG": (0.0232, 0.0232),
        "PV_B": (0.0044, 0.0044),
        "WindOnshore_B": (0.0032, 0.0032),
        "Hydro_B": (0.00054, 0.00054),
        "MDV": (0, 1000), "HDV": (0, 1000), "LDV": (0, 1000), "Bus": (0, 1000),
        "Two_wheel": (0, 1000), "Aviation": (0, 1000), "Marine": (0, 1000),
        "cook_b": (0, 1), "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000), "DW_Electric_converter": (0, 1000),
        "HFO": (0, 1000),
    },
}
add_converter_capacity(m, "2020", CAPACITY_LIMITS_2020)

# ---- 2030 capacity options ---------------------------------------------------
CAPACITY_LIMITS_2030 = {
    "CI_data": {
        "PV_B": (0.0052, 0.0052),
        "MDV": (0, 1000), "HDV": (0, 1000), "LDV": (0, 1000), "LDV_BF": (0, 1000),
        "Bus": (0, 1000), "Two_wheel": (0, 1000), "Aviation": (0, 1000),
        "Marine": (0, 1000), "cook_b": (0, 1000), "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000), "DW_Electric_converter": (0, 1000),
        "DW_Electric_converter_D": (0, 1000), "HFO": (0, 1000),
        "BG_N": (0, 0.011), "PV_N": (0, 1), "WindOnshore_N": (0, 0.08),
    },
    "FJ_data": {
        "BG_B": (0.0580, 0.0580), "PV_B": (0.0090, 0.0090), "Hydro_B": (0.0625, 0.0625),
        "MDV": (0, 1000), "HDV": (0, 1000), "LDV": (0, 1000), "LDV_BF": (0, 1000),
        "Bus": (0, 1000), "Two_wheel": (0, 1000), "Aviation": (0, 1000),
        "Marine": (0, 1000), "cook_b": (0, 1000), "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000), "DW_Electric_converter": (0, 1000),
        "DW_Electric_converter_D": (0, 1000), "HFO": (0, 1000),
        "BG_N": (0, 1), "PV_N": (0, 82), "WindOnshore_N": (0, 6),
        "Wave_N": (0, 1), "WindOffshore_N": (0, 1),
    },
    "FSM_data": {
        "PV_B": (0.0028, 0.0028), "WindOnshore_B": (0.0009, 0.0009),
        "Hydro_B": (0.000225, 0.000225),
        "MDV": (0, 1000), "HDV": (0, 1000), "LDV": (0, 1000), "LDV_BF": (0, 1000),
        "Bus": (0, 1000), "Two_wheel": (0, 1000), "Aviation": (0, 1000),
        "Marine": (0, 1000), "cook_b": (0, 1000), "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000), "DW_Electric_converter": (0, 1000),
        "DW_Electric_converter_D": (0, 1000), "HFO": (0, 1000),
        "BG_N": (0, 0.160), "PV_N": (0, 3.1), "WindOnshore_N": (0, 0.23),
        "Wave_N": (0, 20), "WindOffshore_N": (0, 20),
    },
    "KB_data": {
        "PV_B": (0.0030, 0.0030),
        "MDV": (0, 1000), "HDV": (0, 1000), "LDV": (0, 1000), "LDV_BF": (0, 1000),
        "Bus": (0, 1000), "Two_wheel": (0, 1000), "Aviation": (0, 1000),
        "Marine": (0, 1000), "cook_b": (0, 1000), "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000), "DW_Electric_converter": (0, 1000),
        "DW_Electric_converter_D": (0, 1000), "HFO": (0, 1000),
        "BG_N": (0, 0.211), "PV_N": (0, 3.2), "WindOnshore_N": (0, 0.24),
    },
    "MI_data": {
        "PV_B": (0.0017, 0.0017),
        "MDV": (0, 1000), "HDV": (0, 1000), "LDV": (0, 1000), "LDV_BF": (0, 1000),
        "Bus": (0, 1000), "Two_wheel": (0, 1000), "Aviation": (0, 1000),
        "Marine": (0, 1000), "cook_b": (0, 1000), "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000), "DW_Electric_converter": (0, 1000),
        "DW_Electric_converter_D": (0, 1000), "HFO": (0, 1000),
        "BG_N": (0, 0.02), "PV_N": (0, 0.82), "WindOnshore_N": (0, 0.06),
        "Wave_N": (0, 20), "WindOffshore_N": (0, 20),
    },
    "NU_data": {
        "PV_B": (0.0028, 0.0028),
        "MDV": (0, 1000), "HDV": (0, 1000), "LDV": (0, 1000), "LDV_BF": (0, 1000),
        "Bus": (0, 1000), "Two_wheel": (0, 1000), "Aviation": (0, 1000),
        "Marine": (0, 1000), "cook_b": (0, 1000), "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000), "DW_Electric_converter": (0, 1000),
        "DW_Electric_converter_D": (0, 1000), "HFO": (0, 1000),
        "BG_N": (0, 0.0038), "PV_N": (0, 0.095), "WindOnshore_N": (0, 0.007),
    },
    "NE_data": {
        "PV_B": (0.0010, 0.0011),
        "MDV": (0, 1000), "HDV": (0, 1000), "LDV": (0, 1000), "LDV_BF": (0, 1000),
        "Bus": (0, 1000), "Two_wheel": (0, 1000), "Aviation": (0, 1000),
        "Marine": (0, 1000), "cook_b": (0, 1000), "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000), "DW_Electric_converter": (0, 1000),
        "DW_Electric_converter_D": (0, 1000), "HFO": (0, 1000),
        "BG_N": (0, 0.0047), "PV_N": (0, 1.1), "WindOnshore_N": (0, 0.087),
    },
    "PU_data": {
        "PV_B": (0.0030, 0.0032),
        "MDV": (0, 1000), "HDV": (0, 1000), "LDV": (0, 1000), "LDV_BF": (0, 1000),
        "Bus": (0, 1000), "Two_wheel": (0, 1000), "Aviation": (0, 1000),
        "Marine": (0, 1000), "cook_b": (0, 1000), "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000), "DW_Electric_converter": (0, 1000),
        "DW_Electric_converter_D": (0, 1000), "HFO": (0, 1000),
        "BG_N": (0, 0.0009), "PV_N": (0, 2), "WindOnshore_N": (0, 0.15),
        "WindOffshore_N": (0, 20),
    },
    "PNG_data": {
        "BG_B": (0.0182, 0.0182), "PV_B": (0.0031, 0.0031), "Hydro_B": (0.115, 0.115),
        "Geothermal_B": (0.011, 0.011),
        "MDV": (0, 1000), "HDV": (0, 1000), "LDV": (0, 1000), "LDV_BF": (0, 1000),
        "Bus": (0, 1000), "Two_wheel": (0, 1000), "Aviation": (0, 1000),
        "Marine": (0, 1000), "cook_b": (0, 1000), "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000), "DW_Electric_converter": (0, 1000),
        "DW_Electric_converter_D": (0, 1000), "HFO": (0, 1000),
        "BG_N": (0, 0.452), "PV_N": (0, 20), "WindOnshore_N": (0, 20),
        "Wave_N": (0, 20), "WindOffshore_N": (0, 20),
    },
    "SA_data": {
        "BG_B": (0.0011, 0.0011), "PV_B": (0.0138, 0.0138),
        "WindOnshore_B": (0.0005, 0.0005), "Hydro_B": (0.0063, 0.0063),
        "MDV": (0, 1000), "HDV": (0, 1000), "LDV": (0, 1000), "LDV_BF": (0, 1000),
        "Bus": (0, 1000), "Two_wheel": (0, 1000), "Aviation": (0, 1000),
        "Marine": (0, 1000), "cook_b": (0, 1000), "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000), "DW_Electric_converter": (0, 1000),
        "DW_Electric_converter_D": (0, 1000), "HFO": (0, 1000),
        "BG_N": (0, 0.28), "PV_N": (0, 12), "WindOnshore_N": (0, 0.95),
        "Wave_N": (0, 20), "WindOffshore_N": (0, 20),
    },
    "SI_data": {
        "BG_B": (0.0008, 0.0008), "PV_B": (0.0023, 0.0023),
        "Hydro_B": (0.00018, 0.00018),
        "MDV": (0, 1000), "HDV": (0, 1000), "LDV": (0, 1000), "LDV_BF": (0, 1000),
        "Bus": (0, 1000), "Two_wheel": (0, 1000), "Aviation": (0, 1000),
        "Marine": (0, 1000), "cook_b": (0, 1000), "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000), "DW_Electric_converter": (0, 1000),
        "DW_Electric_converter_D": (0, 1000), "HFO": (0, 1000),
        "BG_N": (0, 1.44), "PV_N": (0, 20), "WindOnshore_N": (0, 9),
        "Wave_N": (0, 20), "WindOffshore_N": (0, 20),
        "Hydro_N": (0.00675, 0.00675),
    },
    "TA_data": {
        "PV_B": (0.0071, 0.0071), "WindOnshore_B": (0.00151, 0.00151),
        "MDV": (0, 1000), "HDV": (0, 1000), "LDV": (0, 1000), "LDV_BF": (0, 1000),
        "Bus": (0, 1000), "Two_wheel": (0, 1000), "Aviation": (0, 1000),
        "Marine": (0, 1000), "cook_b": (0, 1), "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000), "DW_Electric_converter": (0, 1000),
        "DW_Electric_converter_D": (0, 1000), "HFO": (0, 1000),
        "BG_N": (0, 0.20), "PV_N": (0, 3), "WindOnshore_N": (0, 0.25),
        "Wave_N": (0, 20), "WindOffshore_N": (0, 20),
    },
    "TU_data": {
        "PV_B": (0.0029, 0.0029),
        "MDV": (0, 1000), "HDV": (0, 1000), "LDV": (0, 1000), "LDV_BF": (0, 1000),
        "Bus": (0, 1000), "Two_wheel": (0, 1000), "Aviation": (0, 1000),
        "Marine": (0, 1000), "cook_b": (0, 1), "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000), "DW_Electric_converter": (0, 1000),
        "DW_Electric_converter_D": (0, 1000), "HFO": (0, 1000),
        "BG_N": (0, 0.0084), "PV_N": (0, 0.11), "WindOnshore_N": (0, 0.009),
        "Wave_N": (0, 20),
    },
    "VU_data": {
        "PV_B": (0.0044, 0.0044), "WindOnshore_B": (0.0032, 0.0032),
        "Hydro_B": (0.00054, 0.00054),
        "MDV": (0, 1000), "HDV": (0, 1000), "LDV": (0, 1000), "LDV_BF": (0, 1000),
        "Bus": (0, 1000), "Two_wheel": (0, 1000), "Aviation": (0, 1000),
        "Marine": (0, 1000), "cook_b": (0, 1), "Industry": (0, 1000),
        "DW_LPG_converter": (0, 1000), "DW_Electric_converter": (0, 1000),
        "DW_Electric_converter_D": (0, 1000), "HFO": (0, 1000),
        "BG_N": (0, 0.062), "PV_N": (0, 20), "WindOnshore_N": (0, 4),
        "Wave_N": (0, 10), "WindOffshore_N": (0, 10),
    },
}
add_converter_capacity(m, "2030", CAPACITY_LIMITS_2030)

# ---- 2040 / 2050 capacity options --------------------------------------------
_SECTORS_PTX = {
    "MDV": (0, 1000), "HDV": (0, 1000), "LDV": (0, 1000), "LDV_BF": (0, 1000),
    "Bus": (0, 1000), "Two_wheel": (0, 1000), "Aviation": (0, 1000), "Marine": (0, 1000),
    "Industry": (0, 1000),
    "DW_LPG_converter": (0, 1000), "DW_Electric_converter": (0, 1000),
    "DW_Electric_converter_2": (0, 1000), "ST_N": (0, 1000000), "HFO": (0, 1000),
    "RO": (0, 1000), "AEL": (0, 10),
    "Ammonia_synthesis": (0, 1000), "Methanol_synthesis": (0, 1000), "HP": (0, 1000),
    "DAC": (0, 1000), "FTL": (0, 1000),
    "LDV_el": (0, 1000), "HDV_el": (0, 1000), "HDV_BF": (0, 1000),
    "MDV_el": (0, 1000), "MDV_BF": (0, 1000), "Two_wheel_el": (0, 1000),
    "Bus_el": (0, 1000), "Marine_e": (0, 1000), "Aviation_el": (0, 1000),
    "Aviation_e": (0, 1000), "cook_el": (0, 1000), "cook_LPG": (0, 1000),
    "Industry_EH": (0, 1000), "DW_heat": (0, 1000),
    "Dummy_Ammonia": (0, 1000), "Dummy_Methanol": (0, 1000),
    "Ship_BEV": (0, 1000), "Industry_EL": (0, 1000),
}

_RES_2040 = {
    "CI_data":  (1000, {}, {"BG_N": (0, 0.011), "PV_N": (0, 1), "WindOnshore_N": (0, 0.08)}),
    "FJ_data":  (1000, {"Hydro_B": (0.0625, 0.0625)},
                 {"BG_N": (0, 20), "PV_N": (0, 82), "WindOnshore_N": (0, 6),
                  "Wave_N": (0, 20), "WindOffshore_N": (0, 20)}),
    "FSM_data": (1000, {"Hydro_B": (0.000225, 0.000225)},
                 {"BG_N": (0, 0.160), "PV_N": (0, 3.1), "WindOnshore_N": (0, 0.23),
                  "Wave_N": (0, 20), "WindOffshore_N": (0, 20)}),
    "KB_data":  (1000, {}, {"BG_N": (0, 0.211), "PV_N": (0, 3.2), "WindOnshore_N": (0, 0.24)}),
    "MI_data":  (1000, {}, {"BG_N": (0, 0.02), "PV_N": (0, 0.82), "WindOnshore_N": (0, 0.06),
                            "Wave_N": (0, 20), "WindOffshore_N": (0, 20)}),
    "NU_data":  (1000, {}, {"BG_N": (0, 0.0038), "PV_N": (0, 0.095), "WindOnshore_N": (0, 0.007)}),
    "NE_data":  (1000, {}, {"BG_N": (0, 0.0047), "PV_N": (0, 1.1), "WindOnshore_N": (0, 0.087)}),
    "PU_data":  (1000, {}, {"BG_N": (0, 0.0009), "PV_N": (0, 2), "WindOnshore_N": (0, 0.15),
                            "WindOffshore_N": (0, 20)}),
    "PNG_data": (1000, {"Hydro_B": (0.115, 0.115), "Geothermal_B": (0.011, 0.011)},
                 {"BG_N": (0, 0.452), "PV_N": (0, 20), "WindOnshore_N": (0, 20),
                  "Wave_N": (0, 20), "WindOffshore_N": (0, 20)}),
    "SA_data":  (1000, {"Hydro_B": (0.0063, 0.0063)},
                 {"BG_N": (0, 0.28), "PV_N": (0, 12), "WindOnshore_N": (0, 0.95),
                  "Wave_N": (0, 20), "WindOffshore_N": (0, 20)}),
    "SI_data":  (1000, {"Hydro_B": (0.00018, 0.00018)},
                 {"BG_N": (0, 1.44), "PV_N": (0, 20), "WindOnshore_N": (0, 9),
                  "Wave_N": (0, 20), "WindOffshore_N": (0, 20), "Hydro_N": (0.00675, 0.00675)}),
    "TA_data":  (1, {}, {"BG_N": (0, 0.20), "PV_N": (0, 3), "WindOnshore_N": (0, 0.25),
                         "Wave_N": (0, 20), "WindOffshore_N": (0, 20)}),
    "TU_data":  (1, {}, {"BG_N": (0, 0.0084), "PV_N": (0, 0.11), "WindOnshore_N": (0, 0.009),
                         "Wave_N": (0, 20)}),
    "VU_data":  (1, {"Hydro_B": (0.00054, 0.00054)},
                 {"BG_N": (0, 0.062), "PV_N": (0, 20), "WindOnshore_N": (0, 4),
                  "Wave_N": (0, 10), "WindOffshore_N": (0, 10)}),
}


# Per-node techs to drop from _SECTORS_PTX (none in the current model;
# define as empty dict so _build_capacity_dict works without change).
_DROP_PTX = {}


def _build_capacity_dict(res_table):
    caps = {}
    for node, (cook_b, existing, res) in res_table.items():
        node_caps = {**_SECTORS_PTX, "cook_b": (0, cook_b), **existing, **res}
        for tech in _DROP_PTX.get(node, ()):
            node_caps.pop(tech, None)
        caps[node] = node_caps
    return caps


CAPACITY_LIMITS_2040 = _build_capacity_dict(_RES_2040)
add_converter_capacity(m, "2040", CAPACITY_LIMITS_2040)

# 2050 renewables are identical to 2040 except for these overrides
_RES_2050 = {node: (cook_b, dict(existing), dict(res))
             for node, (cook_b, existing, res) in _RES_2040.items()}
_RES_2050["NU_data"][2]["PV_N"] = (0, 0.150)
_RES_2050["PNG_data"][2]["PV_N"] = (0, 500)
_RES_2050["PNG_data"][2]["WindOnshore_N"] = (0, 100)

CAPACITY_LIMITS_2050 = _build_capacity_dict(_RES_2050)
add_converter_capacity(m, "2050", CAPACITY_LIMITS_2050)


# ============================================================================
# Converter coefficients (input/output ratios per technology)
# ============================================================================
# ---- 2020 existing technologies ----------------------------------------------
converter_coefficient = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [
            ["DG", "BG_B", "PV_B", "WindOnshore_B", "Hydro_B", "Geothermal_B", "MDV",
             "HDV", "LDV", "Bus", "Two_wheel", "Aviation", "Marine", "cook_b",
             "Industry", "DW_LPG_converter", "DW_Electric_converter", "NG_plant", "HFO"],
            ["2020"],
            ["Powergen"],
            ["Biomass", "Elec", "CO2", "Diesel", "Gasoline", "JetA1", "MDO",
             "T_MDV_th", "T_HDV_th", "T_LDV_th", "T_Bus_th", "T_Two_wheel_th",
             "T_Aviation_th", "T_Marine_th", "T_Marine_f_th", "Heat_cooking",
             "Heat_industry", "LPG", "DHW_LPG", "DHW_el", "NG", "HFOO"],
        ]
    )
)
converter_coefficient.loc[idx["DG", :, :, "Elec"], "coefficient"] = 1
converter_coefficient.loc[idx["DG", :, :, "Diesel"], "coefficient"] = -2.85
converter_coefficient.loc[idx["DG", :, :, "CO2"], "coefficient"] = 0.76

converter_coefficient.loc[idx["NG_plant", :, :, "Elec"], "coefficient"] = 1
converter_coefficient.loc[idx["NG_plant", :, :, "NG"], "coefficient"] = -2
converter_coefficient.loc[idx["NG_plant", :, :, "CO2"], "coefficient"] = 0.40

converter_coefficient.loc[idx["BG_B", :, :, "Elec"], "coefficient"] = 1
converter_coefficient.loc[idx["BG_B", :, :, "Biomass"], "coefficient"] = -2.85
converter_coefficient.loc[idx["BG_B", :, :, "CO2"], "coefficient"] = 0

converter_coefficient.loc[idx["PV_B", :, :, "Elec"], "coefficient"] = 1
converter_coefficient.loc[idx["WindOnshore_B", :, :, "Elec"], "coefficient"] = 1
converter_coefficient.loc[idx["Hydro_B", :, :, "Elec"], "coefficient"] = 1
converter_coefficient.loc[idx["Geothermal_B", :, :, "Elec"], "coefficient"] = 1

converter_coefficient.loc[idx["cook_b", :, :, "Heat_cooking"], "coefficient"] = 1
converter_coefficient.loc[idx["cook_b", :, :, "Biomass"], "coefficient"] = -1
converter_coefficient.loc[idx["cook_b", :, :, "CO2"], "coefficient"] = 0

converter_coefficient.loc[idx["Industry", :, :, "Heat_industry"], "coefficient"] = 1
converter_coefficient.loc[idx["Industry", :, :, "Diesel"], "coefficient"] = -1.17
converter_coefficient.loc[idx["Industry", :, :, "CO2"], "coefficient"] = 0.31

converter_coefficient.loc[idx["DW_LPG_converter", :, :, "DHW_LPG"], "coefficient"] = 1
converter_coefficient.loc[idx["DW_LPG_converter", :, :, "LPG"], "coefficient"] = -1.17
converter_coefficient.loc[idx["DW_LPG_converter", :, :, "CO2"], "coefficient"] = 0.22

converter_coefficient.loc[idx["DW_Electric_converter", :, :, "DHW_el"], "coefficient"] = 1
converter_coefficient.loc[idx["DW_Electric_converter", :, :, "Elec"], "coefficient"] = -1.05
converter_coefficient.loc[idx["DW_Electric_converter", :, :, "CO2"], "coefficient"] = 0

converter_coefficient.loc[idx["MDV", :, :, "T_MDV_th"], "coefficient"] = 1
converter_coefficient.loc[idx["MDV", :, :, "Diesel"], "coefficient"] = -1
converter_coefficient.loc[idx["MDV", :, :, "CO2"], "coefficient"] = 0.26

converter_coefficient.loc[idx["HDV", :, :, "T_HDV_th"], "coefficient"] = 1
converter_coefficient.loc[idx["HDV", :, :, "Diesel"], "coefficient"] = -1
converter_coefficient.loc[idx["HDV", :, :, "CO2"], "coefficient"] = 0.26

converter_coefficient.loc[idx["LDV", :, :, "T_LDV_th"], "coefficient"] = 1
converter_coefficient.loc[idx["LDV", :, :, "Gasoline"], "coefficient"] = -1
converter_coefficient.loc[idx["LDV", :, :, "CO2"], "coefficient"] = 0.25

converter_coefficient.loc[idx["Bus", :, :, "T_Bus_th"], "coefficient"] = 1
converter_coefficient.loc[idx["Bus", :, :, "Diesel"], "coefficient"] = -1
converter_coefficient.loc[idx["Bus", :, :, "CO2"], "coefficient"] = 0.26

converter_coefficient.loc[idx["Two_wheel", :, :, "T_Two_wheel_th"], "coefficient"] = 1
converter_coefficient.loc[idx["Two_wheel", :, :, "Gasoline"], "coefficient"] = -1
converter_coefficient.loc[idx["Two_wheel", :, :, "CO2"], "coefficient"] = 0.25

converter_coefficient.loc[idx["Aviation", :, :, "T_Aviation_th"], "coefficient"] = 1
converter_coefficient.loc[idx["Aviation", :, :, "JetA1"], "coefficient"] = -1
converter_coefficient.loc[idx["Aviation", :, :, "CO2"], "coefficient"] = 0.26

converter_coefficient.loc[idx["Marine", :, :, "T_Marine_th"], "coefficient"] = 1
converter_coefficient.loc[idx["Marine", :, :, "MDO"], "coefficient"] = -1
converter_coefficient.loc[idx["Marine", :, :, "CO2"], "coefficient"] = 0.27

converter_coefficient.loc[idx["HFO", :, :, "T_Marine_f_th"], "coefficient"] = 1
converter_coefficient.loc[idx["HFO", :, :, "HFOO"], "coefficient"] = -1
converter_coefficient.loc[idx["HFO", :, :, "CO2"], "coefficient"] = 0.29

converter_coefficient = converter_coefficient.dropna(how="all")
m.parameter.add(converter_coefficient, "converter_coefficient")

# ---- 2030 new renewable generation -------------------------------------------
converter_coefficient = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [
            ["BG_N", "PV_N", "WindOnshore_N", "Wave_N", "WindOffshore_N", "Hydro_N"],
            ["2030"],
            ["Powergen"],
            ["Biomass", "Elec", "CO2"],
        ]
    )
)
converter_coefficient.loc[idx["BG_N", :, :, "Elec"], "coefficient"] = 1
converter_coefficient.loc[idx["BG_N", :, :, "Biomass"], "coefficient"] = -2.85
converter_coefficient.loc[idx["BG_N", :, :, "CO2"], "coefficient"] = 0
converter_coefficient.loc[idx["PV_N", :, :, "Elec"], "coefficient"] = 1
converter_coefficient.loc[idx["WindOnshore_N", :, :, "Elec"], "coefficient"] = 1
converter_coefficient.loc[idx["Wave_N", :, :, "Elec"], "coefficient"] = 1
converter_coefficient.loc[idx["WindOffshore_N", :, :, "Elec"], "coefficient"] = 1
converter_coefficient.loc[idx["Hydro_N", :, :, "Elec"], "coefficient"] = 1

converter_coefficient = converter_coefficient.dropna(how="all")
m.parameter.add(converter_coefficient, "converter_coefficient")

# ---- 2040 / 2050 full power-to-X and sector coupling -------------------------
converter_coefficient = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [
            ["BG_N", "PV_N", "WindOnshore_N", "Wave_N", "WindOffshore_N", "ST_N",
             "Industry_EL", "LDV_BF", "RO", "Ammonia_synthesis", "DAC",
             "Methanol_synthesis", "HP", "FTL", "AEL", "LDV_el", "HDV_el", "HDV_BF",
             "MDV_el", "MDV_BF", "Two_wheel_el", "Bus_el", "Marine_e", "Aviation_el",
             "Aviation_e", "cook_el", "cook_LPG", "Industry_EH", "DW_heat",
             "Dummy_Ammonia", "Dummy_Methanol", "DW_Electric_converter_2", "Ship_BEV"],
            ["2040", "2050"],
            ["Powergen"],
            ["Biomass", "Elec", "LPG", "CO2", "T_LDV_BF", "Pure_water", "Hydrogen",
             "Heat", "ST_Heat", "Ammonia", "co", "Methanol", "eKerosene", "T_LDV_el",
             "T_HDV_el", "T_HDV_BF", "T_MDV_el", "T_MDV_BF", "T_Two_wheel_el",
             "T_Bus_el", "T_Marine_e", "T_Aviation_el", "T_Aviation_e", "DHW_ST",
             "T_cook_el", "T_cook_LPG", "T_Industry_EH", "T_DHW_heat", "Dummy_EL",
             "DHW_el", "T_ship_el"],
        ]
    )
)
converter_coefficient.loc[idx["BG_N", :, :, "Elec"], "coefficient"] = 1
converter_coefficient.loc[idx["BG_N", :, :, "Biomass"], "coefficient"] = -2.85
converter_coefficient.loc[idx["BG_N", :, :, "CO2"], "coefficient"] = 0
converter_coefficient.loc[idx["PV_N", :, :, "Elec"], "coefficient"] = 1
converter_coefficient.loc[idx["WindOnshore_N", :, :, "Elec"], "coefficient"] = 1
converter_coefficient.loc[idx["Wave_N", :, :, "Elec"], "coefficient"] = 1
converter_coefficient.loc[idx["WindOffshore_N", :, :, "Elec"], "coefficient"] = 1
converter_coefficient.loc[idx["ST_N", :, :, "Heat"], "coefficient"] = 1
converter_coefficient.loc[idx["LDV_BF", :, :, "T_LDV_BF"], "coefficient"] = 1
converter_coefficient.loc[idx["LDV_BF", :, :, "Biomass"], "coefficient"] = -2.85
converter_coefficient.loc[idx["LDV_BF", :, :, "CO2"], "coefficient"] = 0.25
converter_coefficient.loc[idx["LDV_el", :, :, "T_LDV_el"], "coefficient"] = 1
converter_coefficient.loc[idx["LDV_el", :, :, "Elec"], "coefficient"] = -1
converter_coefficient.loc[idx["HDV_el", :, :, "T_HDV_el"], "coefficient"] = 1
converter_coefficient.loc[idx["HDV_el", :, :, "Elec"], "coefficient"] = -1
converter_coefficient.loc[idx["HDV_BF", :, :, "T_HDV_BF"], "coefficient"] = 1
converter_coefficient.loc[idx["HDV_BF", :, :, "Biomass"], "coefficient"] = -2.85
converter_coefficient.loc[idx["HDV_BF", :, :, "CO2"], "coefficient"] = 0.25
converter_coefficient.loc[idx["MDV_BF", :, :, "T_MDV_BF"], "coefficient"] = 1
converter_coefficient.loc[idx["MDV_BF", :, :, "Biomass"], "coefficient"] = -2.85
converter_coefficient.loc[idx["MDV_BF", :, :, "CO2"], "coefficient"] = 0.25
converter_coefficient.loc[idx["MDV_el", :, :, "T_MDV_el"], "coefficient"] = 1
converter_coefficient.loc[idx["MDV_el", :, :, "Elec"], "coefficient"] = -1
converter_coefficient.loc[idx["Bus_el", :, :, "T_Bus_el"], "coefficient"] = 1
converter_coefficient.loc[idx["Bus_el", :, :, "Elec"], "coefficient"] = -1
converter_coefficient.loc[idx["Two_wheel_el", :, :, "T_Two_wheel_el"], "coefficient"] = 1
converter_coefficient.loc[idx["Two_wheel_el", :, :, "Elec"], "coefficient"] = -1
converter_coefficient.loc[idx["Aviation_el", :, :, "T_Aviation_el"], "coefficient"] = 1
converter_coefficient.loc[idx["Aviation_el", :, :, "Elec"], "coefficient"] = -1
converter_coefficient.loc[idx["RO", :, :, "Pure_water"], "coefficient"] = 1
converter_coefficient.loc[idx["RO", :, :, "Elec"], "coefficient"] = -0.00315
converter_coefficient.loc[idx["Ammonia_synthesis", :, :, "Ammonia"], "coefficient"] = 1
converter_coefficient.loc[idx["Ammonia_synthesis", :, :, "Elec"], "coefficient"] = -0.02
converter_coefficient.loc[idx["Ammonia_synthesis", :, :, "Hydrogen"], "coefficient"] = -1.14
converter_coefficient.loc[idx["DAC", :, :, "co"], "coefficient"] = 1
converter_coefficient.loc[idx["DAC", :, :, "Elec"], "coefficient"] = -0.25
converter_coefficient.loc[idx["DAC", :, :, "Heat"], "coefficient"] = -1.7
converter_coefficient.loc[idx["Methanol_synthesis", :, :, "Methanol"], "coefficient"] = 1
converter_coefficient.loc[idx["Methanol_synthesis", :, :, "Hydrogen"], "coefficient"] = -1.127
converter_coefficient.loc[idx["Methanol_synthesis", :, :, "co"], "coefficient"] = -0.2485
converter_coefficient.loc[idx["HP", :, :, "Heat"], "coefficient"] = 1
converter_coefficient.loc[idx["HP", :, :, "Elec"], "coefficient"] = -0.285
converter_coefficient.loc[idx["FTL", :, :, "eKerosene"], "coefficient"] = 1
converter_coefficient.loc[idx["FTL", :, :, "Hydrogen"], "coefficient"] = -1.2
converter_coefficient.loc[idx["FTL", :, :, "co"], "coefficient"] = -0.305
converter_coefficient.loc[idx["AEL", :, :, "Hydrogen"], "coefficient"] = 1
converter_coefficient.loc[idx["AEL", :, :, "Pure_water"], "coefficient"] = -0.450
converter_coefficient.loc[idx["AEL", :, :, "Elec"], "coefficient"] = -1.42
converter_coefficient.loc[idx["cook_el", :, :, "T_cook_el"], "coefficient"] = 1
converter_coefficient.loc[idx["cook_el", :, :, "Elec"], "coefficient"] = -1
converter_coefficient.loc[idx["cook_LPG", :, :, "T_cook_LPG"], "coefficient"] = 1
converter_coefficient.loc[idx["cook_LPG", :, :, "LPG"], "coefficient"] = -1.17
converter_coefficient.loc[idx["cook_LPG", :, :, "CO2"], "coefficient"] = 0.27
converter_coefficient.loc[idx["Industry_EH", :, :, "T_Industry_EH"], "coefficient"] = 1
converter_coefficient.loc[idx["Industry_EH", :, :, "Heat"], "coefficient"] = -1
converter_coefficient.loc[idx["Industry_EL", :, :, "T_Industry_EH"], "coefficient"] = 1
converter_coefficient.loc[idx["Industry_EL", :, :, "Elec"], "coefficient"] = -1.05
converter_coefficient.loc[idx["Dummy_Ammonia", :, :, "Dummy_EL"], "coefficient"] = 1
converter_coefficient.loc[idx["Dummy_Ammonia", :, :, "Ammonia"], "coefficient"] = -1
converter_coefficient.loc[idx["Dummy_Methanol", :, :, "Dummy_EL"], "coefficient"] = 1
converter_coefficient.loc[idx["Dummy_Methanol", :, :, "Methanol"], "coefficient"] = -1
converter_coefficient.loc[idx["Ship_BEV", :, :, "T_ship_el"], "coefficient"] = 1
converter_coefficient.loc[idx["Ship_BEV", :, :, "Elec"], "coefficient"] = -1

converter_coefficient = converter_coefficient.dropna(how="all")
m.parameter.add(converter_coefficient, "converter_coefficient")


# ============================================================================
# Renewable activity profiles ("converter_activityprofile")
# ============================================================================
ACTIVITY_TECHS_2020 = ["PV_B", "WindOnshore_B", "Hydro_B", "Geothermal_B", "BG_B"]
ACTIVITY_TECHS_2030 = [
    "PV_B", "WindOnshore_B", "Hydro_B", "Geothermal_B", "BG_B",
    "PV_N", "WindOnshore_N", "Wave_N", "WindOffshore_N", "BG_N", "Hydro_N",
]
ACTIVITY_TECHS_2040_2050 = ACTIVITY_TECHS_2030 + ["ST_N"]

add_activity_profiles(m, profiles, "2030", ACTIVITY_TECHS_2030)
add_activity_profiles(m, profiles, "2020", ACTIVITY_TECHS_2020)
add_activity_profiles(m, profiles, "2040", ACTIVITY_TECHS_2040_2050)
add_activity_profiles(m, profiles, "2050", ACTIVITY_TECHS_2040_2050)


# ============================================================================
# Converter investment & fixed O&M costs ("accounting_converterunits")
# ============================================================================
# --- 2020 existing-capacity technologies ----------------------------------
COSTS_2020 = {
    "DG": {"perUnitBuild": 0, "useAnnuity": 1, "amorTime": 2, "interest": 0.06, "perUnitTotal": 160},
    "NG_plant": _annuity(0, 87.6, 25),
    "BG_B": _annuity(0, 78, 25),
    "PV_B": _annuity(0, 14, 25),
    "WindOnshore_B": _annuity(0, 22, 25),
    "Hydro_B": _annuity(0, 168 * 2.22, 25),
    "Geothermal_B": _annuity(0, 118 * 4.54, 25),
    "MDV": _zero_cost(), "HDV": _zero_cost(), "LDV": _zero_cost(), "Bus": _zero_cost(),
    "Two_wheel": _zero_cost(), "Aviation": _zero_cost(), "Marine": _zero_cost(),
    "cook_b": _zero_cost(), "Industry": _zero_cost(),
    "DW_LPG_converter": _zero_cost(), "DW_Electric_converter": _zero_cost(),
    "HFO": _zero_cost(),
}
add_converter_unit_costs(m, "2020", COSTS_2020)

# --- 2030 new renewables --------------------------------------------------
COSTS_2030 = {
    "BG_N": _annuity(2700.0 * 1.2, 78 * 1.2, 25),
    "PV_N": _annuity(331 * 1.2, 7 * 1.2, 25),
    "WindOnshore_N": _annuity(1081 * 1.2, 22 * 1.2, 25),
    "Wave_N": _annuity(3030 * 1.2, 83 * 1.2, 25),
    "WindOffshore_N": _annuity(2670 * 1.2, 80 * 1.2, 25),
    "Hydro_N": _annuity(12400 * 2.22, 2.22 * 490, 50),
}
add_converter_unit_costs(m, "2030", COSTS_2030)

# --- 2040 / 2050 per-country electrolyser (AEL) costs ---------------------
AEL_BUILD_2040 = {
    "CI_data": 946, "FJ_data": 637, "FSM_data": 856, "KB_data": 1000,
    "MI_data": 1513, "NU_data": 1187, "NE_data": 2101, "PU_data": 843,
    "PNG_data": 471, "SA_data": 780, "SI_data": 741, "TA_data": 803,
    "TU_data": 2092, "VU_data": 787,
}
AEL_BUILD_2050 = {
    "CI_data": 622, "FJ_data": 416, "FSM_data": 505, "KB_data": 542,
    "MI_data": 780, "NU_data": 712, "NE_data": 1193, "PU_data": 537,
    "PNG_data": 344, "SA_data": 498, "SI_data": 444, "TA_data": 498,
    "TU_data": 978, "VU_data": 456,
}


def _ael_node_specs(build_by_node, om_fix):
    """Build a node_specs dict for the per-country AEL electrolyser."""
    return {
        node: {"AEL": _annuity(build * 1.2, om_fix * 1.2, 25)}
        for node, build in build_by_node.items()
    }


# --- 2040 global technologies ---------------------------------------------
COSTS_2040 = {
    "DW_Electric_converter_2": _zero_cost(),
    "Ship_BEV": _zero_cost(),
    "Dummy_Ammonia": _zero_cost(),
    "Dummy_Methanol": _zero_cost(),
    "Industry_EL": _annuity(140 * 1.2, 5 * 1.2, 20),
    "BG_N": _annuity(2600.0 * 1.2, 70 * 1.2, 30),
    "PV_N": _annuity(240 * 1.2, 4.8 * 1.2, 35),
    "WindOnshore_N": _annuity(1080 * 1.2, 19 * 1.2, 25),
    "Wave_N": _annuity(2300 * 1.2, 58 * 1.2, 30),
    "WindOffshore_N": _annuity(2520 * 1.2, 71 * 1.2, 25),
    "Hydro_N": _annuity(12400 * 2.22, 2.22 * 490, 50),
    "ST_N": _annuity(530 * 1.2, 1.8 * 1.2, 30),
    "LDV_BF": _annuity(1200 * 1.2, 48 * 1.2, 20),
    "RO": _annuity(0.003200 * 1.2, 0.000128 * 1.2, 30),
    "Ammonia_synthesis": _annuity(1348 * 1.2, 75 * 1.2, 30),
    "DAC": _annuity(0.277000 * 1.2, 0.011000 * 1.2, 30),
    "Methanol_synthesis": _annuity(976 * 1.2, 39 * 1.2, 30),
    "HP": _annuity(650 * 1.2, 2.8 * 1.2, 30),
    "FTL": _annuity(1070 * 1.2, 32 * 1.2, 30),
    "LDV_el": _zero_cost(), "HDV_el": _zero_cost(), "HDV_BF": _annuity(1200 * 1.2, 48 * 1.2, 20),
    "MDV_el": _zero_cost(), "MDV_BF": _annuity(1200 * 1.2, 48 * 1.2, 20),
    "Two_wheel_el": _zero_cost(), "Bus_el": _zero_cost(), "Marine_e": _zero_cost(),
    "Aviation_el": _zero_cost(), "Aviation_e": _zero_cost(),
    "cook_el": _zero_cost(), "cook_LPG": _zero_cost(),
    "Industry_EH": _zero_cost(), "DW_heat": _zero_cost(),
}
add_converter_unit_costs(m, "2040", COSTS_2040, _ael_node_specs(AEL_BUILD_2040, 14))

# Variable O&M for 2040 AEL and Industry_EL
for tech_var, per_activity in [("AEL", 0.0014), ("Industry_EL", 0.002)]:
    accounting_converteractivity = pd.DataFrame(
        index=pd.MultiIndex.from_product(
            [["OMVar"], ["global"], ["horizon"], [tech_var], ["2040"], ["Powergen"]]
        )
    ).sort_index()
    accounting_converteractivity.loc[
        idx["OMVar", "global", "horizon", tech_var, "2040", "Powergen"], "perActivity"
    ] = per_activity
    accounting_converteractivity = accounting_converteractivity.fillna(0)
    m.parameter.add(accounting_converteractivity, "accounting_converteractivity")

# Shipping-link costs (registered after 2040 converter costs)

# --- 2050 global technologies ---------------------------------------------
COSTS_2050 = {
    "Ship_BEV": _zero_cost(),
    "BG_N": _annuity(2600.0 * 1.2, 70 * 1.2, 30),
    "Industry_EL": _annuity(140 * 1.2, 5 * 1.2, 20),
    "Dummy_Ammonia": _zero_cost(),
    "Dummy_Methanol": _zero_cost(),
    "PV_N": _annuity(213 * 1.2, 4.2 * 1.2, 35),
    "WindOnshore_N": _annuity(1040 * 1.2, 19 * 1.2, 25),
    "Wave_N": _annuity(2015 * 1.2, 48 * 1.2, 30),
    "WindOffshore_N": _annuity(2480 * 1.2, 70 * 1.2, 25),
    "Hydro_N": _annuity(12400 * 2.22, 2.22 * 490, 50),
    "ST_N": _annuity(506 * 1.2, 1.8 * 1.2, 30),
    "LDV_BF": _annuity(1200 * 1.2, 70 * 1.2, 30),
    "RO": _annuity(0.001300 * 1.2, 0.000052 * 1.2, 25),
    "Ammonia_synthesis": _annuity(1348 * 1.2, 75 * 1.2, 30),
    "DAC": _annuity(0.232000 * 1.2, 0.009300 * 1.2, 30),
    "Methanol_synthesis": _annuity(976 * 1.2, 39 * 1.2, 30),
    "HP": _annuity(630 * 1.2, 2.6 * 1.2, 30),
    "FTL": _annuity(1070 * 1.2, 32 * 1.2, 30),
    "LDV_el": _zero_cost(), "HDV_el": _zero_cost(), "HDV_BF": _annuity(1200 * 1.2, 70 * 1.2, 30),
    "MDV_el": _zero_cost(), "MDV_BF": _annuity(1200 * 1.2, 70 * 1.2, 30),
    "Two_wheel_el": _zero_cost(), "Bus_el": _zero_cost(), "Marine_e": _zero_cost(),
    "Aviation_el": _zero_cost(), "Aviation_e": _zero_cost(),
    "cook_el": _zero_cost(), "cook_LPG": _zero_cost(),
    "Industry_EH": _zero_cost(), "DW_heat": _zero_cost(),
}
add_converter_unit_costs(m, "2050", COSTS_2050, _ael_node_specs(AEL_BUILD_2050, 12))

# Variable O&M for 2050 AEL and Industry_EL
for tech_var, per_activity in [("AEL", 0.0014), ("Industry_EL", 0.002)]:
    accounting_converteractivity = pd.DataFrame(
        index=pd.MultiIndex.from_product(
            [["OMVar"], ["global"], ["horizon"], [tech_var], ["2050"], ["Powergen"]]
        )
    ).sort_index()
    accounting_converteractivity.loc[
        idx["OMVar", "global", "horizon", tech_var, "2050", "Powergen"], "perActivity"
    ] = per_activity
    accounting_converteractivity = accounting_converteractivity.fillna(0)
    m.parameter.add(accounting_converteractivity, "accounting_converteractivity")

# Shipping-link costs (registered again for 2050 horizon)


# ============================================================================
# Demand profiles ("sourcesink_profile" + "sourcesink_config")
# ============================================================================
DEMAND_SPEC_2020 = [
    ("Elec", "demand", "_2020"),
    ("T_MDV_th", "MDV", ""),
    ("T_HDV_th", "HDV", ""),
    ("T_LDV_th", "LDV", ""),
    ("T_Bus_th", "Bus", ""),
    ("T_Two_wheel_th", "Two_wheel", ""),
    ("T_Marine_th", "Marine", ""),
    ("T_Aviation_th", "Aviation", ""),
    ("T_Marine_f_th", "Marinef", ""),
    ("Heat_cooking", "HC", ""),
    ("Heat_industry", "HI", ""),
    ("DHW_el", "DHWE", ""),
    ("DHW_LPG", "DHWL", ""),
]

DEMAND_SPEC_2030 = [
    ("Elec", "demand", "_2030"),
    ("T_MDV_th", "MDV", "_2030"),
    ("T_HDV_th", "HDV", "_2030"),
    ("T_LDV_th", "LDV", "_2030"),
    ("T_Bus_th", "Bus", "_2030"),
    ("T_Two_wheel_th", "Two_wheel", "_2030"),
    ("T_Marine_th", "Marine", "_2030"),
    ("T_Aviation_th", "Aviation", "_2030"),
    ("T_Marine_f_th", "Marinef", "_2030"),
    ("Heat_cooking", "HC", "_2030"),
    ("Heat_industry", "HI", "_2030"),
    ("DHW_el", "DHWE", "_2030"),
    ("DHW_LPG", "DHWL", "_2030"),
]


def _demand_spec_transport(suffix):
    return [
        ("Elec", "demand", suffix),
        ("T_MDV_el", "MDV_el", suffix),
        ("T_MDV_th", "MDV_Th", suffix),
        ("T_MDV_BF", "MDV_BF", suffix),
        ("T_HDV_el", "HDV_el", suffix),
        ("T_HDV_th", "HDV_Th", suffix),
        ("T_HDV_BF", "HDV_BF", suffix),
        ("T_LDV_el", "LDV_el", suffix),
        ("T_LDV_th", "LDV_Th", suffix),
        ("T_LDV_BF", "LDV_BF", suffix),
        ("T_Bus_el", "BUS_el", suffix),
        ("T_Bus_th", "BUS_Th", suffix),
        ("T_Two_wheel_el", "2W_el", suffix),
        ("T_Two_wheel_th", "2W_th", suffix),
        ("T_Marine_f_th", "Marine_TH", suffix),
        ("Dummy_EL", "Marine_E", suffix),
        ("Methanol", "Marine_M", suffix),
        ("T_ship_el", "Marine_BEV", suffix),
        ("T_Aviation_th", "AVIA_TH", suffix),
        ("T_Aviation_el", "AVIA_EL", suffix),
        ("eKerosene", "AVIA_E", suffix),
        ("Heat_cooking", "HC_B", suffix),
        ("T_cook_LPG", "HC_L", suffix),
        ("T_cook_el", "HC_el", suffix),
        ("Heat_industry", "HI_D", suffix),
        ("T_Industry_EH", "HI_EH", suffix),
        ("DHW_el", "DHW_E", suffix),
        ("DHW_LPG", "DHW_L", suffix),
    ]


DEMAND_SPEC_2040 = _demand_spec_transport("_2040")
DEMAND_SPEC_2050 = _demand_spec_transport("_2050")


def register_demand(m, profiles, year, spec):
    """Register every (commodity, prefix, suffix) demand row for one year."""
    for commodity, prefix, suffix in spec:
        add_demand_profile(m, profiles, prefix, suffix, year, commodity)
        add_demand_config(m, commodity)


register_demand(m, profiles, "2020", DEMAND_SPEC_2020)
register_demand(m, profiles, "2030", DEMAND_SPEC_2030)
register_demand(m, profiles, "2040", DEMAND_SPEC_2040)
register_demand(m, profiles, "2050", DEMAND_SPEC_2050)


# ============================================================================
# Fuel imports ("sourcesink_annualsum" + "sourcesink_config")
# ============================================================================
EFUEL_NODES = [
    "FJ_data", "PNG_data", "VU_data", "TA_data",
    "SA_data", "SI_data", "CI_data", "NE_data",
]

BIOMASS_LIMITS = [12, 2380, 168, 221, 22, 5, 4, 12100, 1, 295, 1507, 211, 9, 671]
add_fuel_import_limits(m, m.set.nodesdata, "Biomass", BIOMASS_LIMITS)


CONV_FUEL_LIMITS = [1000000] * 14
CONV_FUEL_LIMITS[6] = 100000  # NE_data
for commodity in ["NG", "HFOO", "Diesel", "LPG", "Gasoline", "JetA1", "MDO"]:
    add_fuel_import_limits(m, m.set.nodesdata, commodity, CONV_FUEL_LIMITS)

add_fuel_import_config(m, m.set.nodesdata, "Biomass")
for commodity in ["NG", "HFOO", "Diesel", "LPG", "Gasoline", "JetA1", "MDO"]:
    add_fuel_import_config(m, m.set.nodesdata, commodity)


# ============================================================================
# Fuel prices ("accounting_sourcesinkflow")
# ============================================================================
ALL_YEARS = ["2030", "2040", "2050"]


def _flat(value, nodes):
    """A flat per-node price list."""
    return [value] * len(nodes)


CONV_FUEL_PRICES = {
    "Biomass": 0.025, "NG": 0.05, "HFOO": 0.08,
    "Diesel": 0.090, "LPG": 0.13, "Gasoline": 0.10, "JetA1": 0.06, "MDO": 0.08,
}
for commodity, price in CONV_FUEL_PRICES.items():
    add_fuel_prices(m, m.set.nodesdata, ["2020"], commodity, _flat(price, m.set.nodesdata))
    add_fuel_prices(m, m.set.nodesdata, ALL_YEARS, commodity, _flat(price, m.set.nodesdata))



# ============================================================================
# CO2 emission accounting ("sourcesink_annualsum" + "sourcesink_config")
# ============================================================================
sourcesink_annualSum = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [m.set.nodesdata, m.set.yearssel, ["Emission"], ["CO2"]]
    )
)
sourcesink_annualSum.loc[idx[DATA_NODES, :, :, :], "lower"] = -np.inf
sourcesink_annualSum = sourcesink_annualSum.dropna()
m.parameter.add(sourcesink_annualSum, "sourcesink_annualsum")

sourcesink_config = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [m.set.nodesdata, m.set.yearssel, ["Emission"], ["CO2"]]
    )
)
sourcesink_config.loc[idx[DATA_NODES, :, :, :], "usesLowerSum"] = 1
sourcesink_config.loc[idx[DATA_NODES, :, :, :], "usesUpperProfile"] = 1
sourcesink_config = sourcesink_config.dropna()
m.parameter.add(sourcesink_config, "sourcesink_config")


# ============================================================================
# Storage technologies
# ============================================================================

# ---------------------------------------------------------------------------
# Battery (Li-Ion) — 2030 / 2040 / 2050
# ---------------------------------------------------------------------------
add_storage_tech(
    m,
    tech="Battery",
    years=STORAGE_YEARS,
    commodity_stored="Elec_LiIon",
    converter_coeff={
        ("Charge", "Elec"): -1,
        ("Charge", "Elec_LiIon"): 0.975,
        ("Discharge", "Elec"): 1,
        ("Discharge", "Elec_LiIon"): -1.025,
    },
    converter_lifetime=20,
    converter_activity_upper=1,
    converter_unit_upper=300,
    storage_lifetime=20,
    storage_level_upper=1,
    storage_size=1,            # GWh/unit
    storage_reservoir_upper=1000,
    converter_costs_by_year={
        "2030": _annuity(110 * 1.0, 2.7, 25),
        "2040": _annuity(85 * 1.0, 2.1, 25),
        "2050": _annuity(65 * 1.0, 1.7, 25),
    },
    storage_costs=None,        # handled separately (per-year variation)
)

add_storage_accounting_by_year(
    m,
    tech="Battery",
    years=STORAGE_YEARS,
    cost_by_year={
        "2030": {"perUnitBuild": 105, "useAnnuity": 1, "amorTime": 20, "interest": 0.06, "perUnitTotal": 2.6},
        "2040": {"perUnitBuild": 62,  "useAnnuity": 1, "amorTime": 20, "interest": 0.06, "perUnitTotal": 1.5},
        "2050": {"perUnitBuild": 52,  "useAnnuity": 1, "amorTime": 20, "interest": 0.06, "perUnitTotal": 1.3},
    },
)

# ---------------------------------------------------------------------------
# Thermal Short-term Storage (THSS) — 2030 / 2040 / 2050
# ---------------------------------------------------------------------------
add_storage_tech(
    m,
    tech="THSS",
    years=STORAGE_YEARS,
    commodity_stored="Heat_T",
    converter_coeff={
        ("Charge", "Heat"): -1,
        ("Charge", "Heat_T"): 0.90,
        ("Discharge", "Heat"): 1,
        ("Discharge", "Heat_T"): -1.11,
    },
    converter_lifetime=30,
    converter_activity_upper=1,
    converter_unit_upper=300,
    storage_lifetime=30,
    storage_level_upper=1,
    storage_size=1,
    storage_reservoir_upper=1000,
    converter_costs_by_year={
        "2030": _annuity(0, 1, 25),
        "2040": _annuity(0, 1, 25),
        "2050": _annuity(0, 1, 25),
    },
    storage_costs={"perUnitBuild": 12, "useAnnuity": 1, "amorTime": 30, "interest": 0.06, "perUnitTotal": 0.2},
)

# ---------------------------------------------------------------------------
# Water storage (H2O) — 2040 / 2050
# ---------------------------------------------------------------------------
add_storage_tech(
    m,
    tech="H20_storage",
    years=["2040", "2050"],
    commodity_stored="Pure_water_T",
    converter_coeff={
        ("Charge", "Pure_water"): -1,
        ("Charge", "Pure_water_T"): 1,
        ("Discharge", "Pure_water"): 1,
        ("Discharge", "Pure_water_T"): -1,
    },
    converter_lifetime=30,
    converter_activity_upper=1,
    converter_unit_upper=300,
    storage_lifetime=30,
    storage_level_upper=1,
    storage_size=1,            # (1000 m3)/unit
    storage_reservoir_upper=1000,
    converter_costs_by_year={
        "2040": _annuity(0, 0, 0, interest=0, use_annuity=0),
        "2050": _annuity(0, 0, 0, interest=0, use_annuity=0),
    },
    storage_costs={"perUnitBuild": 0.076, "useAnnuity": 1, "amorTime": 30, "interest": 0.06, "perUnitTotal": 0.002},
)

# ---------------------------------------------------------------------------
# Hydrogen storage (H2) — 2030 / 2040 / 2050
# ---------------------------------------------------------------------------
add_storage_tech(
    m,
    tech="H2_storage",
    years=STORAGE_YEARS,
    commodity_stored="Hydrogen_T",
    converter_coeff={
        ("Charge", "Hydrogen"): -1,
        ("Charge", "Hydrogen_T"): 1,
        ("Discharge", "Hydrogen"): 1,
        ("Discharge", "Hydrogen_T"): -1,
    },
    converter_lifetime=30,
    converter_activity_upper=1,
    converter_unit_upper=300,
    storage_lifetime=30,
    storage_level_upper=1,
    storage_size=1,
    storage_reservoir_upper=1000,
    converter_costs_by_year={
        "2030": _annuity(0, 1, 25),
        "2040": _annuity(0, 1, 25),
        "2050": _annuity(0, 1, 25),
    },
    storage_costs={"perUnitBuild": 0.33, "useAnnuity": 1, "amorTime": 30, "interest": 0.06, "perUnitTotal": 0.01},
)

# ---------------------------------------------------------------------------
# Ammonia storage — 2030 / 2040 / 2050
# ---------------------------------------------------------------------------
add_storage_tech(
    m,
    tech="Ammonia_storage",
    years=STORAGE_YEARS,
    commodity_stored="Ammonia_T",
    converter_coeff={
        ("Charge", "Ammonia"): -1,
        ("Charge", "Ammonia_T"): 1,
        ("Discharge", "Ammonia"): 1,
        ("Discharge", "Ammonia_T"): -1,
    },
    converter_lifetime=30,
    converter_activity_upper=1,
    converter_unit_upper=300,
    storage_lifetime=30,
    storage_level_upper=1,
    storage_size=1,            # GWh/unit
    storage_reservoir_upper=1000,
    converter_costs_by_year={
        "2030": _annuity(0, 1, 25),
        "2040": _annuity(0, 1, 25),
        "2050": _annuity(0, 1, 25),
    },
    storage_costs={"perUnitBuild": 0.20, "useAnnuity": 1, "amorTime": 30, "interest": 0.06, "perUnitTotal": 0},
)

# ---------------------------------------------------------------------------
# Methanol storage — 2040 / 2050
# ---------------------------------------------------------------------------
add_storage_tech(
    m,
    tech="Methanol_storage",
    years=["2040", "2050"],
    commodity_stored="Methanol_T",
    converter_coeff={
        ("Charge", "Methanol"): -1,
        ("Charge", "Methanol_T"): 1,
        ("Discharge", "Methanol"): 1,
        ("Discharge", "Methanol_T"): -1,
    },
    converter_lifetime=30,
    converter_activity_upper=1,
    converter_unit_upper=300,
    storage_lifetime=30,
    storage_level_upper=1,
    storage_size=1,
    storage_reservoir_upper=1000,
    converter_costs_by_year={
        "2040": _annuity(0, 0, 25),
        "2050": _annuity(0, 0, 25),
    },
    storage_costs={"perUnitBuild": 0.06, "useAnnuity": 1, "amorTime": 30, "interest": 0.06, "perUnitTotal": 0},
)

# ---------------------------------------------------------------------------
# E-kerosene storage — 2040 / 2050
# ---------------------------------------------------------------------------
add_storage_tech(
    m,
    tech="eKerosene_storage",
    years=["2040", "2050"],
    commodity_stored="eKerosene_T",
    converter_coeff={
        ("Charge", "eKerosene"): -1,
        ("Charge", "eKerosene_T"): 1,
        ("Discharge", "eKerosene"): 1,
        ("Discharge", "eKerosene_T"): -1,
    },
    converter_lifetime=30,
    converter_activity_upper=1,
    converter_unit_upper=300,
    storage_lifetime=30,
    storage_level_upper=1,
    storage_size=1,
    storage_reservoir_upper=1000,
    converter_costs_by_year={
        "2040": _annuity(0, 0, 25),
        "2050": _annuity(0, 0, 25),
    },
    storage_costs={"perUnitBuild": 0.05, "useAnnuity": 1, "amorTime": 30, "interest": 0.06, "perUnitTotal": 0},
)

# ---------------------------------------------------------------------------
# CO2 storage — 2040 / 2050
# ---------------------------------------------------------------------------
add_storage_tech(
    m,
    tech="co2_storage",
    years=["2040", "2050"],
    commodity_stored="co_T",
    converter_coeff={
        ("Charge", "co"): -1,
        ("Charge", "co_T"): 1,
        ("Discharge", "co"): 1,
        ("Discharge", "co_T"): -1,
    },
    converter_lifetime=30,
    converter_activity_upper=1,
    converter_unit_upper=300,
    storage_lifetime=30,
    storage_level_upper=1,
    storage_size=1,
    storage_reservoir_upper=1000,
    converter_costs_by_year={
        "2040": _annuity(0, 0, 25, interest=0.06, use_annuity=1),
        "2050": _annuity(0, 0, 25, interest=0.06, use_annuity=1),
    },
    converter_var_costs={"2040": 0.042, "2050": 0.035},
    converter_var_costs_activity="Charge",
    storage_costs={"perUnitBuild": 0, "useAnnuity": 1, "amorTime": 30, "interest": 0.06, "perUnitTotal": 0},
)


# ============================================================================
# Write data files and run optimisation
# ============================================================================
m.write(fileformat="dat")

m.run(
    resultfile="IP_2050_Final_SS1_minload",
    lo=3,
    postcalc=1,
    roundts=1,
    pathopt="myopic",
)
