# =============================================================================
#  PIC GDX Visualisation Toolkit
#  Plotting & post-processing tools for REMix / HOMER Pacific-Island-Countries
#  GDX results (LCOE, capacities, generation profiles, storage SOC, ...).
#
#  This file is a TOOLBOX: each section below is an independent tool with its
#  own `main()` and command-line interface. Run one at a time (see README.md).
#
#  LAYOUT: this file lives in  <repo>/visualization/  and is run from there.
#  It reads GAMS results from the repo-root  GDX_results/  folder, i.e. one
#  level up:  ../GDX_results/  . Override on the command line, e.g.
#      ...  --gdx ../GDX_results/IP_2050_Final_S23_minload.gdx  --out figures/  --dpi 600
#
#  The LCO results .xlsx (produced by pic_lco_assessment, read by pic_lco_plots)
#  is written here inside visualization/.
#  Needs: gdxpds (requires a GAMS installation), pandas, numpy, matplotlib, openpyxl.
# =============================================================================

"""
Pacific Island Countries (PICs) — Levelized Cost Assessment
============================================================
Extracts commodity balance and cost indicators from a GAMS GDX file
and computes levelized costs (LCO) for:

    Electricity · Heat · Water · Hydrogen · CO₂ ·
    Ammonia · Methanol · e-Kerosene · System Energy

Each LCO is computed as:
    LCO = (CAPEX + OPEX + Input energy costs) / Total output demand

Results are written to a single Excel file with one sheet per carrier.

Usage
-----
    python pic_lco_assessment.py
    python pic_lco_assessment.py --gdx path/to/results.gdx --out results.xlsx
"""

import argparse
import logging
from pathlib import Path

import gdxpds
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s %(message)s"
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

GDX_PATH = "../GDX_results/IP_2050_Final_S1_minload.gdx"

data = gdxpds.to_dataframes(GDX_PATH)
YEARS = ["2020", "2030", "2040", "2050"]
OUTPUT_PATH = "LCO_results_IP_2050_Final_S23.xlsx"
ISLANDS = [
    "CI_model", "FJ_model", "FSM_model", "KB_model", "MI_model",
    "NU_model", "NE_model", "PU_model", "PNG_model", "SA_model",
    "SI_model", "TA_model", "TU_model", "VU_model",
]

# ── Population projections (*1000 persons) ────────────────────────────────────
# Source: user-provided projections; years 2020/2030/2040/2050 used directly.
# Keys are (island_model_code, year_int) -> population in thousands.
POPULATION = {
    # Cook Islands
    ("CI_model",  2020): 14,   ("CI_model",  2030): 14,
    ("CI_model",  2040): 18,   ("CI_model",  2050): 26,
    # FSM
    ("FSM_model", 2020): 113,  ("FSM_model", 2030): 126,
    ("FSM_model", 2040): 156,  ("FSM_model", 2050): 195,
    # Fiji
    ("FJ_model",  2020): 924,  ("FJ_model",  2030): 998,
    ("FJ_model",  2040): 1155, ("FJ_model",  2050): 1321,
    # Kiribati
    ("KB_model",  2020): 128,  ("KB_model",  2030): 159,
    ("KB_model",  2040): 211,  ("KB_model",  2050): 275,
    # Marshall Islands
    ("MI_model",  2020): 41,   ("MI_model",  2030): 36,
    ("MI_model",  2040): 41,   ("MI_model",  2050): 52,
    # Nauru
    ("NU_model",  2020): 12,   ("NU_model",  2030): 14,
    ("NU_model",  2040): 20,   ("NU_model",  2050): 28,
    # Niue
    ("NE_model",  2020): 1.7,  ("NE_model",  2030): 2,
    ("NE_model",  2040): 3,    ("NE_model",  2050): 5,
    # Palau
    ("PU_model",  2020): 18,   ("PU_model",  2030): 19,
    ("PU_model",  2040): 23,   ("PU_model",  2050): 27,
    # Papua New Guinea
    ("PNG_model", 2020): 9950, ("PNG_model", 2030): 12366,
    ("PNG_model", 2040): 15626,("PNG_model", 2050): 19450,
    # Samoa
    ("SA_model",  2020): 218,  ("SA_model",  2030): 242,
    ("SA_model",  2040): 307,  ("SA_model",  2050): 404,
    # Solomon Islands
    ("SI_model",  2020): 708,  ("SI_model",  2030): 998,
    ("SI_model",  2040): 1354, ("SI_model",  2050): 1773,
    # Tonga
    ("TA_model",  2020): 105,  ("TA_model",  2030): 111,
    ("TA_model",  2040): 133,  ("TA_model",  2050): 168,
    # Tuvalu
    ("TU_model",  2020): 10,   ("TU_model",  2030): 10,
    ("TU_model",  2040): 13,   ("TU_model",  2050): 18,
    # Vanuatu
    ("VU_model",  2020): 319,  ("VU_model",  2030): 391,
    ("VU_model",  2040): 513,  ("VU_model",  2050): 666,
}

# Human-readable labels for the chart legend
ISLAND_LABELS = {
    "CI_model":  "Cook Islands",
    "FJ_model":  "Fiji",
    "FSM_model": "FSM",
    "KB_model":  "Kiribati",
    "MI_model":  "Marshall Islands",
    "NU_model":  "Nauru",
    "NE_model":  "Niue",
    "PU_model":  "Palau",
    "PNG_model": "Papua New Guinea",
    "SA_model":  "Samoa",
    "SI_model":  "Solomon Islands",
    "TA_model":  "Tonga",
    "TU_model":  "Tuvalu",
    "VU_model":  "Vanuatu",
}

# ── Technology groups ─────────────────────────────────────────────────────────
TECHS = {
    "el_production":   ["DG", "PV_B", "NG_plant", "BG_B", "WindOnshore_B",
                        "Hydro_B", "BG_N", "PV_N", "WindOnshore_N",
                        "Wave_N", "WindOffshore_N", "Hydro_N", "Geothermal_B"],
    "el_storage":      ["Battery"],
    "heat_production": ["cook_b", "Industry", "DW_LPG_converter",
                        "DW_Electric_converter", "ST_N", "HP",
                        "cook_el", "cook_LPG", "Industry_EL"],
    "heat_storage":    ["THSS"],
    "water_production":["RO"],
    "water_storage":   ["H20_storage"],
    "h2_production":   ["AEL"],
    "h2_storage":      ["H2_storage"],
    "co2_production":  ["DAC"],
    "co2_storage":     ["co2_storage"],
    "ammonia_production": ["Ammonia_synthesis"],
    "ammonia_storage":    ["Ammonia_storage"],
    "methanol_production": ["Methanol_synthesis"],
    "methanol_storage":    ["Methanol_storage"],
    "ekerosene_production": ["FTL"],
    "ekerosene_storage":    ["eKerosene_storage"],
}

# Converter techs for LCOE electricity demand (unchanged from original)
ALL_CONVERTER_TECHS = [
    "BG_N", "PV_N", "WindOnshore_N", "Wave_N", "WindOffshore_N", "ST_N",
    "Industry_EL", "LDV_BF", "RO", "Ammonia_synthesis", "DAC",
    "Methanol_synthesis", "HP", "FTL", "AEL", "LDV_el", "HDV_el",
    "HDV_BF", "MDV_el", "MDV_BF", "Two_wheel_el", "Bus_el", "Marine_e",
    "Aviation_el", "Aviation_e", "cook_el", "cook_LPG", "Industry_EH",
    "DW_heat", "Dummy_Ammonia", "Dummy_Methanol",
    "DW_Electric_converter", "Ship_BEV", "Battery"
]

# Converter techs for LCOEnergy system demand
All_CONVERTERS = [
    "DW_Electric_converter",
    "DW_LPG_converter",
    "FTL",
    "Marine",
    "cook_b",
    "Industry",
    "Methanol_synthesis",
    "Ammonia_synthesis",
    "Aviation_el",
    "Aviation",
    "Bus_el",
    "Bus",
    "cook_el",
    "cook_LPG",
    "HDV_BF",
    "HDV_el",
    "HDV",
    "Industry_EL",
    "Industry_EH",
    "LDV_BF",
    "LDV_el",
    "LDV",
    "HFO",
    "MDV_BF",
    "MDV_el",
    "MDV",
    "Ship_BEV",
    "Two_wheel_el",
    "Two_wheel", "FuelImport"
]

# Final commodities for LCOEnergy — Elec excluded here; handled separately
ALL_FINAL_COMMODITIES = [
    "T_Two_wheel_th", "T_Two_wheel_el", "T_MDV_th", "T_MDV_el", "T_MDV_BF",
    "T_Marine_f_th", "T_LDV_th", "T_LDV_el", "T_LDV_BF", "T_Industry_EH",
    "T_HDV_th", "T_HDV_el", "T_HDV_BF", "T_Bus_th", "T_Bus_el",
    "T_Aviation_th", "T_Aviation_el", "Heat_industry",
    "eKerosene", "DHW_LPG", "DHW_el", "T_Marine_th",
    "Heat_cooking", "T_cook_LPG", "T_cook_el", "Methanol", "Ammonia", "T_ship_el"
]

HEAT_DEMAND_COMMODITIES = [
    "Heat_cooking", "T_cook_LPG", "T_cook_el",
    "Heat_industry", "T_Industry_EH", "DHW_el", "DHW_LPG", "Heat",
]

FUEL_COSTS = {
    "Diesel":   0.090,
    "LPG":      0.065,
    "Biomass":  0.032,
    "NG":       0.025,
}

EL_FUEL_MAP = {
    "DG":       "Diesel",
    "NG_plant": "NG",
    "BG_B":     "Biomass",
    "BG_N":     "Biomass",
}

HEAT_FUEL_MAP = {
    "cook_b":                "Biomass",
    "Industry":              "Diesel",
    "cook_LPG":              "LPG",
    "DW_LPG_converter":      "LPG",
    "DW_Electric_converter": None,
    "HP":                    None,
    "cook_el":               None,
    "Industry_EL":           None,
}

CAPEX_OPEX_INDICATORS = ["Invest", "OMFix", "OMVar"]


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_gdx(path: str) -> dict:
    log.info("Loading GDX: %s", path)
    data = gdxpds.to_dataframes(path)
    log.info("Symbols loaded: %s", list(data.keys()))
    return data


# ─────────────────────────────────────────────────────────────────────────────
# QUERY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def capex_opex(ind: pd.DataFrame, island: str, year: str,
               techs: list[str]) -> float:
    mask = (
        (ind["nodesModel"] == island) &
        (ind["years"] == year) &
        (ind["techs"].isin(techs)) &
        (ind["indicator"].isin(CAPEX_OPEX_INDICATORS))
    )
    return float(ind.loc[mask, "Value"].sum())


def commodity_flow(cb: pd.DataFrame, island: str, year: str,
                   techs: list[str], commodity: str,
                   direction: str = "input") -> float:
    mask = (
        (cb["accNodesModel"] == island) &
        (cb["accYears"] == year) &
        (cb["balanceType"] == "net") &
        (cb["techs"].isin(techs)) &
        (cb["commodity"] == commodity)
    )
    subset = cb.loc[mask, "Value"]
    if direction == "input":
        return float(abs(subset[subset < 0].sum()))
    else:
        return float(subset[subset > 0].sum())


def demand_flow(cb: pd.DataFrame, island: str, year: str,
                commodities: list[str],
                demand_tech: str = "Demand") -> float:
    mask = (
        (cb["accNodesModel"] == island) &
        (cb["accYears"] == year) &
        (cb["balanceType"] == "net") &
        (cb["techs"] == demand_tech) &
        (cb["commodity"].isin(commodities)) &
        (cb["Value"] < 0)
    )
    return float(abs(cb.loc[mask, "Value"].sum()))


# ─────────────────────────────────────────────────────────────────────────────
# LCO COMPUTATION FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def lco_electricity(data: dict, island: str, year: str) -> dict:
    ind = data["indicator_accounting_detailed"]
    cb  = data["commodity_balance_annual"]

    el_techs = TECHS["el_production"] + TECHS["el_storage"]
    cost = capex_opex(ind, island, year, el_techs)

    for tech, fuel in EL_FUEL_MAP.items():
        use = commodity_flow(cb, island, year, [tech], fuel, "input")
        cost += use * FUEL_COSTS.get(fuel, 0.0)

    direct   = demand_flow(cb, island, year, ["Elec"])
    via_conv = commodity_flow(cb, island, year, ALL_CONVERTER_TECHS, "Elec", "input")
    demand   = direct + via_conv

    return _lco_result("LCOE_Electricity", cost, demand)


def lco_heat(data: dict, island: str, year: str,
             lcoe_lookup: dict) -> dict:
    ind = data["indicator_accounting_detailed"]
    cb  = data["commodity_balance_annual"]

    heat_techs = TECHS["heat_production"] + TECHS["heat_storage"]
    cost = capex_opex(ind, island, year, heat_techs)

    for tech, fuel in HEAT_FUEL_MAP.items():
        if fuel is None:
            elec_use = commodity_flow(cb, island, year, [tech], "Elec", "input")
            cost += elec_use * _safe_lookup(lcoe_lookup, island, year, "LCOE")
        else:
            fuel_use = commodity_flow(cb, island, year, [tech], fuel, "input")
            cost += fuel_use * FUEL_COSTS.get(fuel, 0.0)

    demand = (
        demand_flow(cb, island, year, HEAT_DEMAND_COMMODITIES)
        + commodity_flow(cb, island, year, TECHS["co2_production"], "Heat", "input")
    )

    return _lco_result("LCOHeat", cost, demand)


def lco_water(data: dict, island: str, year: str,
              lcoe_lookup: dict) -> dict:
    ind = data["indicator_accounting_detailed"]
    cb  = data["commodity_balance_annual"]

    water_techs = TECHS["water_production"] + TECHS["water_storage"]
    cost = capex_opex(ind, island, year, water_techs)

    elec_use = commodity_flow(cb, island, year, TECHS["water_production"], "Elec", "input")
    cost += elec_use * _safe_lookup(lcoe_lookup, island, year, "LCOE")

    demand = commodity_flow(cb, island, year, TECHS["water_production"],
                            "Pure_water", "output")

    return _lco_result("LCOWater", cost, demand)


def lco_hydrogen(data: dict, island: str, year: str,
                 lcoe_lookup: dict, lco_water_lookup: dict) -> dict:
    ind = data["indicator_accounting_detailed"]
    cb  = data["commodity_balance_annual"]

    h2_techs = TECHS["h2_production"] + TECHS["h2_storage"]
    cost = capex_opex(ind, island, year, h2_techs)

    elec_use  = commodity_flow(cb, island, year, TECHS["h2_production"], "Elec", "input")
    water_use = commodity_flow(cb, island, year, TECHS["h2_production"], "Pure_water", "input")

    cost += elec_use  * _safe_lookup(lcoe_lookup,      island, year, "LCOE")
    cost += water_use * _safe_lookup(lco_water_lookup, island, year, "LCOWater")

    demand = commodity_flow(cb, island, year, TECHS["h2_production"], "Hydrogen", "output")

    return _lco_result("LCOH2", cost, demand)


def lco_co2(data: dict, island: str, year: str,
            lcoe_lookup: dict, lco_heat_lookup: dict) -> dict:
    ind = data["indicator_accounting_detailed"]
    cb  = data["commodity_balance_annual"]

    co2_techs = TECHS["co2_production"] + TECHS["co2_storage"]
    cost = capex_opex(ind, island, year, co2_techs)

    elec_use = commodity_flow(cb, island, year, TECHS["co2_production"], "Elec", "input")
    heat_use = commodity_flow(cb, island, year, TECHS["co2_production"], "Heat", "input")

    cost += elec_use * _safe_lookup(lcoe_lookup,     island, year, "LCOE")
    cost += heat_use * _safe_lookup(lco_heat_lookup, island, year, "LCOHeat")

    demand = commodity_flow(cb, island, year, TECHS["co2_production"], "co", "output")

    return _lco_result("LCOCO2", cost, demand)


def lco_ammonia(data: dict, island: str, year: str,
                lcoe_lookup: dict, lco_h2_lookup: dict) -> dict:
    ind = data["indicator_accounting_detailed"]
    cb  = data["commodity_balance_annual"]

    ammo_techs = TECHS["ammonia_production"] + TECHS["ammonia_storage"]
    cost = capex_opex(ind, island, year, ammo_techs)

    elec_use = commodity_flow(cb, island, year, TECHS["ammonia_production"], "Elec", "input")
    h2_use   = commodity_flow(cb, island, year, TECHS["ammonia_production"], "Hydrogen", "input")

    cost += elec_use * _safe_lookup(lcoe_lookup,   island, year, "LCOE")
    cost += h2_use   * _safe_lookup(lco_h2_lookup, island, year, "LCOH2")

    demand = commodity_flow(cb, island, year,
                            TECHS["ammonia_production"], "Ammonia", "output")

    return _lco_result("LCOAmmonia", cost, demand)


def lco_methanol(data: dict, island: str, year: str,
                 lco_h2_lookup: dict, lco_co2_lookup: dict) -> dict:
    ind = data["indicator_accounting_detailed"]
    cb  = data["commodity_balance_annual"]

    meoh_techs = TECHS["methanol_production"] + TECHS["methanol_storage"]
    cost = capex_opex(ind, island, year, meoh_techs)

    h2_use = commodity_flow(cb, island, year, TECHS["methanol_production"], "Hydrogen", "input")
    co_use = commodity_flow(cb, island, year, TECHS["methanol_production"], "co", "input")

    cost += h2_use * _safe_lookup(lco_h2_lookup,  island, year, "LCOH2")
    cost += co_use * _safe_lookup(lco_co2_lookup, island, year, "LCOCO2")

    demand = commodity_flow(cb, island, year,
                            TECHS["methanol_production"], "Methanol", "output")

    return _lco_result("LCOMethanol", cost, demand)


def lco_ekerosene(data: dict, island: str, year: str,
                  lco_h2_lookup: dict, lco_co2_lookup: dict) -> dict:
    ind = data["indicator_accounting_detailed"]
    cb  = data["commodity_balance_annual"]

    ekero_techs = TECHS["ekerosene_production"] + TECHS["ekerosene_storage"]
    cost = capex_opex(ind, island, year, ekero_techs)

    h2_use = commodity_flow(cb, island, year, TECHS["ekerosene_production"], "Hydrogen", "input")
    co_use = commodity_flow(cb, island, year, TECHS["ekerosene_production"], "co", "input")

    cost += h2_use * _safe_lookup(lco_h2_lookup,  island, year, "LCOH2")
    cost += co_use * _safe_lookup(lco_co2_lookup, island, year, "LCOCO2")

    demand = commodity_flow(cb, island, year, TECHS["ekerosene_production"],
                            "eKerosene", "output")

    return _lco_result("LCOeKerosene", cost, demand)


def lco_system_energy(data: dict, island: str, year: str) -> dict:
    """LCOEnergy — total system cost / total final energy demand.

    Demand has two components added together:

    1. Electricity demand — read via the Demand pseudo-tech for commodity "Elec".
       Negative values (consumption) are taken as absolute.
       This correctly captures only local electricity consumption and avoids
       counting inter-island exports that would appear in the generation-side
       balance.

    2. All other final energy — read directly from commodity_balance_annual
       using All_CONVERTERS (excludes electricity generation techs) and
       ALL_FINAL_COMMODITIES (excludes "Elec").
       Negative values (consumption by converter techs) are taken as absolute.

    The two components are summed to give total final energy demand.
    System cost is unchanged (total SystemCost from indicator_accounting).
    """
    ind = data["indicator_accounting"]
    cb  = data["commodity_balance_annual"]

    # ── System cost (unchanged) ───────────────────────────────────────────────
    mask_cost = (
        (ind["accNodesModel"] == island) &
        (ind["accYears"]      == year)   &
        (ind["indicator"]     == "SystemCost")
    )
    cost = float(ind.loc[mask_cost, "Value"].sum())

    # ── Component 1: electricity demand via Demand pseudo-tech ───────────────
    mask_elec = (
        (cb["accNodesModel"] == island) &
        (cb["accYears"]      == year)   &
        (cb["balanceType"]   == "net")  &
        (cb["techs"]         == "Demand") &
        (cb["commodity"]     == "Elec") &
        (cb["Value"] < 0)
    )
    elec_demand = float(abs(cb.loc[mask_elec, "Value"].sum()))

    # ── Component 2: all other final energy via converter techs ──────────────
    mask_other = (
        (cb["accNodesModel"] == island)                  &
        (cb["accYears"]      == year)                    &
        (cb["balanceType"]   == "net")                   &
        (cb["techs"].isin(All_CONVERTERS))               &
        (cb["commodity"].isin(ALL_FINAL_COMMODITIES))    &
        (cb["Value"] > 0)
    )
    other_demand = float(abs(cb.loc[mask_other, "Value"].sum()))

    demand = elec_demand + other_demand

    log.debug(
        "LCOEnergy [%s, %s]: elec_demand=%.2f  other_demand=%.2f  total=%.2f",
        island, year, elec_demand, other_demand, demand
    )

    return _lco_result("LCOEnergy", cost, demand)


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _lco_result(label: str, cost: float, demand: float) -> dict:
    lco = (cost / demand) if demand > 0 else None
    if demand == 0:
        log.debug("Zero demand for %s — LCO set to None.", label)
    return {label: lco, "Total_Cost": cost, "Total_Demand": demand}


def _safe_lookup(lookup: dict, island: str, year: str, label: str) -> float:
    val = lookup.get((island, int(year)))
    if val is None or (isinstance(val, float) and pd.isna(val)):
        log.warning("Missing %s for (%s, %s) — using 0.", label, island, year)
        return 0.0
    return float(val)


def build_lookup(df: pd.DataFrame, value_col: str) -> dict:
    return {(row["Island"], row["Year"]): row[value_col]
            for _, row in df.iterrows()}


def run_all(data: dict, lco_fn, *lookup_args, label: str,
            extra_cols: dict | None = None) -> pd.DataFrame:
    rows = []
    for island in ISLANDS:
        for year in YEARS:
            out = lco_fn(data, island, year, *lookup_args)
            row = {"Island": island, "Year": int(year)}
            row.update(out)
            if extra_cols:
                row.update(extra_cols)
            rows.append(row)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# PER-CAPITA ELECTRICITY DEMAND CHART  (NEW)
# ─────────────────────────────────────────────────────────────────────────────

def plot_electricity_demand_per_capita(lcoe_df: pd.DataFrame,
                                       output_png: str = "electricity_demand_per_capita.png") -> None:
    """
    Plot per-capita electricity demand (MWh/person/year) for all 14 islands
    across 2020, 2030, 2040, 2050 using dotted lines with markers.

    The electricity demand column used is 'Total_Electricity_Demand' from the
    LCOE sheet — it already combines direct demand (via Demand pseudo-tech)
    and via-converter demand, exactly mirroring the lco_electricity() logic.

    Population is sourced from the POPULATION dict (in *1000 persons), so the
    demand (assumed to be in GWh or the model's native energy unit — here kept
    as-is and labelled accordingly) is divided by population * 1000 to get the
    per-person value.  Units on the y-axis label should be adjusted if the
    model energy unit differs from MWh.
    """
    years_int = [int(y) for y in YEARS]

    # Use a qualitatively distinct colour palette for 14 lines
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i) for i in range(len(ISLANDS))]

    fig, ax = plt.subplots(figsize=(12, 7))

    for idx, island in enumerate(ISLANDS):
        per_capita_values = []
        valid = True

        for year in YEARS:
            row = lcoe_df[
                (lcoe_df["Island"] == island) & (lcoe_df["Year"] == int(year))
            ]
            if row.empty:
                log.warning("No LCOE row for (%s, %s) — skipping island in chart.", island, year)
                valid = False
                break

            total_demand = float(row["Total_Electricity_Demand"].iloc[0])
            pop_thousands = POPULATION.get((island, int(year)))

            if pop_thousands is None or pop_thousands == 0:
                log.warning("No population for (%s, %s) — skipping island in chart.", island, year)
                valid = False
                break

            # population in thousands → multiply by 1000 for actual headcount
            pop = pop_thousands 
            per_capita_values.append(total_demand / pop)

        if not valid:
            continue

        label = ISLAND_LABELS.get(island, island)
        ax.plot(
            years_int,
            per_capita_values,
            linestyle=":",          # dotted line
            linewidth=2.0,
            marker="o",
            markersize=6,
            color=colors[idx],
            label=label,
        )

        # Annotate the 2050 end-point with the island name for readability
        ax.annotate(
            label,
            xy=(years_int[-1], per_capita_values[-1]),
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=7,
            color=colors[idx],
            va="center",
        )

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Electricity Demand per Capita\n(MWh / person)", fontsize=11)
    ax.set_title("",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(years_int)
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    ax.legend(
        loc="upper left",
        fontsize=8,
        framealpha=0.7,
        ncol=2,
        title="Island / Country",
        title_fontsize=9,
    )
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_png, dpi=150, bbox_inches="tight")
    log.info("Per-capita electricity demand chart saved to: %s", output_png)
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main(gdx_path: str, output_path: str,preloaded_data=None) -> None:


    # if preloaded_data is not None:
    #     data = preloaded_data
    # else:
    #     data = load_gdx(gdx_path)

    log.info("Computing LCOE Electricity …")
    lcoe_df = run_all(data, lco_electricity, label="LCOE_Electricity")
    lcoe_df.rename(columns={"Total_Cost": "Total_Electricity_Cost",
                             "Total_Demand": "Total_Electricity_Demand"}, inplace=True)
    lcoe_lookup = build_lookup(lcoe_df, "LCOE_Electricity")

    log.info("Computing LCOHeat …")
    heat_df = run_all(data, lco_heat, lcoe_lookup, label="LCOHeat")
    heat_df.rename(columns={"Total_Cost": "Total_Heat_Cost",
                             "Total_Demand": "Total_Heat_Demand"}, inplace=True)
    lco_heat_lookup = build_lookup(heat_df, "LCOHeat")

    log.info("Computing LCOWater …")
    water_df = run_all(data, lco_water, lcoe_lookup, label="LCOWater")
    water_df.rename(columns={"Total_Cost": "Total_Water_Cost",
                              "Total_Demand": "Total_Water_Demand"}, inplace=True)
    lco_water_lookup = build_lookup(water_df, "LCOWater")

    log.info("Computing LCOH2 …")
    h2_df = run_all(data, lco_hydrogen, lcoe_lookup, lco_water_lookup, label="LCOH2")
    h2_df.rename(columns={"Total_Cost": "Total_H2_Cost",
                           "Total_Demand": "Total_H2_Demand"}, inplace=True)
    lco_h2_lookup = build_lookup(h2_df, "LCOH2")

    log.info("Computing LCOCO2 …")
    co_df = run_all(data, lco_co2, lcoe_lookup, lco_heat_lookup, label="LCOCO2")
    co_df.rename(columns={"Total_Cost": "Total_CO2_Cost",
                           "Total_Demand": "Total_CO2_Demand"}, inplace=True)
    lco_co2_lookup = build_lookup(co_df, "LCOCO2")

    log.info("Computing LCOAmmonia …")
    ammonia_df = run_all(data, lco_ammonia, lcoe_lookup, lco_h2_lookup,
                         label="LCOAmmonia")
    ammonia_df.rename(columns={"Total_Cost": "Total_Ammonia_Cost",
                                "Total_Demand": "Total_Ammonia_Demand"}, inplace=True)

    log.info("Computing LCOMethanol …")
    methanol_df = run_all(data, lco_methanol, lco_h2_lookup, lco_co2_lookup,
                          label="LCOMethanol")
    methanol_df.rename(columns={"Total_Cost": "Total_Methanol_Cost",
                                 "Total_Demand": "Total_Methanol_Demand"}, inplace=True)

    log.info("Computing LCOeKerosene …")
    ekerosene_df = run_all(data, lco_ekerosene, lco_h2_lookup, lco_co2_lookup,
                           label="LCOeKerosene")
    ekerosene_df.rename(columns={"Total_Cost": "Total_eKerosene_Cost",
                                  "Total_Demand": "Total_eKerosene_Demand"}, inplace=True)

    log.info("Computing LCOEnergy (system) …")
    system_df = run_all(data, lco_system_energy, label="LCOEnergy")
    system_df.rename(columns={"Total_Cost": "Total_Energy_Cost",
                               "Total_Demand": "Total_Energy_Demand"}, inplace=True)

    sheets = {
        "LCOE_Electricity": lcoe_df,
        "LCOHeat":          heat_df,
        "LCOWater":         water_df,
        "LCOH2":            h2_df,
        "LCOCO2":           co_df,
        "LCOAmmonia":       ammonia_df,
        "LCOMethanol":      methanol_df,
        "LCOeKerosene":     ekerosene_df,
        "LCOEnergy":        system_df,
    }

    log.info("Writing results to: %s", output_path)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            ws = writer.sheets[sheet_name]
            for col in ws.columns:
                max_len = max(len(str(cell.value or "")) for cell in col) + 2
                ws.column_dimensions[col[0].column_letter].width = min(max_len, 30)

    log.info("Done. %d sheets written.", len(sheets))

    print("\n── Summary (non-null LCO counts per carrier) ──────────────────")
    for sheet, df in sheets.items():
        col = sheet
        if col in df.columns:
            n_valid = df[col].notna().sum()
            n_total = len(df)
            print(f"  {sheet:<22} {n_valid}/{n_total} island-year pairs have results")
    print()

    # ── NEW: per-capita electricity demand chart ──────────────────────────────
    log.info("Generating per-capita electricity demand chart …")
    chart_path = Path(output_path).stem + "_electricity_per_capita.png"
    plot_electricity_demand_per_capita(lcoe_df, output_png=chart_path)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gdx", default=GDX_PATH,
                        help="Path to GAMS GDX results file")
    parser.add_argument("--out", default=OUTPUT_PATH,
                        help="Output Excel file path")
    args = parser.parse_args()

    main(args.gdx, args.out)
##########################all commodity values are written to an excel###############################
######now for plotiing these results###############################################################
# Default — PNG at 300 dpi, saved to figures/
"""
Pacific Island Countries (PICs) — LCO Results Visualisation
============================================================
Reads the LCO Excel output and produces publication-quality bar charts
for each energy carrier. All plots share a consistent style suitable
for scientific journals (Nature Energy, Applied Energy, etc.).

Usage
-----
    python pic_lco_plots.py
    python pic_lco_plots.py --file path/to/results.xlsx --dpi 600 --fmt pdf
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib import rcParams

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL PLOT STYLE  (journal-ready)
# ─────────────────────────────────────────────────────────────────────────────

rcParams.update({
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":          10,
    "axes.titlesize":     11,
    "axes.labelsize":     10,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "legend.title_fontsize": 9,
    "axes.linewidth":     0.8,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    "xtick.major.size":   3,
    "ytick.major.size":   3,
    "xtick.major.width":  0.8,
    "ytick.major.width":  0.8,
    "grid.linewidth":     0.5,
    "grid.alpha":         0.4,
    "grid.color":         "#888888",
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.facecolor":  "white",
})

# ── Colour palette — colorblind-safe (Wong 2011, extended) ───────────────────
YEAR_COLORS = {
    2020: "#0072B2",   # blue
    2030: "#E69F00",   # amber
    2040: "#009E73",   # teal-green
    2050: "#D55E00",   # vermillion
}

YEARS      = [2020, 2030, 2040, 2050]
BAR_WIDTH  = 0.18
FIG_SIZE   = (14, 5)

# ── Clean island labels (strip "_model" suffix) ──────────────────────────────
def clean_labels(islands: np.ndarray) -> list[str]:
    return [s.replace("_model", "") for s in islands]


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION TABLE
# Each entry defines one chart completely.
# ─────────────────────────────────────────────────────────────────────────────

# reference_line: (y_value_in_MWh, label_text) or None
CHART_CONFIG = [
    {
        "sheet":      "LCOE_Electricity",
        "value_col":  "LCOE_Electricity",
        "ylabel":     "Levelized cost of electricity ($/MWh)",
        "title":      "Levelized Cost of Electricity by Island and Year",
        "filename":   "LCOE_Electricity",
        "reference":  None,
        "show_legend": True,
    },
    {
        "sheet":      "LCOHeat",
        "value_col":  "LCOHeat",
        "ylabel":     "Levelized cost of heat ($/MWh)",
        "title":      "Levelized Cost of Heat by Island and Year",
        "filename":   "LCOHeat",
        "reference":  None,
        "show_legend": False,
    },
    {
        "sheet":      "LCOEnergy",
        "value_col":  "LCOEnergy",
        "ylabel":     "Levelized cost of energy ($/MWh)",
        "title":      "System-Level Levelized Cost of Energy by Island and Year",
        "filename":   "LCOEnergy",
        "reference":  None,
        "show_legend": False,
    },
    {
        "sheet":      "LCOH2",
        "value_col":  "LCOH2",
        "ylabel":     "Levelized cost of hydrogen ($/MWh)",
        "title":      "Levelized Cost of Hydrogen by Island and Year",
        "filename":   "LCOH2",
        "reference":  (0,  ""),
        "show_legend": False,
    },
    {
        "sheet":      "LCOAmmonia",
        "value_col":  "LCOAmmonia",
        "ylabel":     "Levelized cost of ammonia ($/MWh)",
        "title":      "Levelized Cost of Ammonia  by Island and Year",
        "filename":   "LCOAmmonia",
        "reference":  (0, ""),
        "show_legend": False,
    },
    {
        "sheet":      "LCOMethanol",
        "value_col":  "LCOMethanol",
        "ylabel":     "Levelized cost of methanol ($/MWh)",
        "title":      "Levelized Cost of Methanol by Island and Year",
        "filename":   "LCOMethanol",
        "reference":  (0,  ""),
        "show_legend": False,
    },
    {
        "sheet":      "LCOeKerosene",
        "value_col":  "LCOeKerosene",
        "ylabel":     "Levelized cost of e-kerosene ($/MWh)",
        "title":      "Levelized Cost of e-Kerosene  by Island and Year",
        "filename":   "LCOeKerosene",
        "reference":  (0, ""),
        "show_legend": False,
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# CORE PLOT FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def plot_lco(
    df: pd.DataFrame,
    value_col: str,
    ylabel: str,
    title: str,
    reference: tuple | None,
    show_legend: bool,
    output_path: Path,
    fmt: str,
    dpi: int,
) -> None:
    """
    Render a grouped bar chart for one LCO carrier.

    Parameters
    ----------
    df          : DataFrame with columns [Island, Year, value_col]
    value_col   : name of the LCO column
    ylabel      : y-axis label
    title       : chart title
    reference   : (y_value_MWh, label_str) for an optional reference line, or None
    show_legend : whether to draw the year legend
    output_path : full save path (stem + suffix applied automatically)
    fmt         : file format ("png", "pdf", "svg", "tiff")
    dpi         : save resolution
    """
    df = df[["Island", "Year", value_col]].sort_values(["Island", "Year"])
    islands = df["Island"].unique()
    labels  = clean_labels(islands)
    x       = np.arange(len(islands))

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # ── Bars ─────────────────────────────────────────────────────────────────
    for i, year in enumerate(YEARS):
        values = (
            df[df["Year"] == year]
            .set_index("Island")
            .reindex(islands)[value_col]
            .values
        ) * 1000   # convert model units → USD/MWh

        bars = ax.bar(
            x + i * BAR_WIDTH,
            values,
            width=BAR_WIDTH,
            label=str(year),
            color=YEAR_COLORS[year],
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )

    # ── Reference line (drawn once, outside the year loop) ───────────────────
    if reference is not None:
        ref_val, ref_label = reference
        ax.axhline(
            y=ref_val,
            color="#CC0000",
            linestyle="--",
            linewidth=1.2,
            zorder=4,
            label=ref_label,
        )
        # Annotate on the right margin
        ax.annotate(
            ref_label,
            xy=(1, ref_val),
            xycoords=("axes fraction", "data"),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color="#CC0000",
        )

    # ── Axes formatting ───────────────────────────────────────────────────────
    ax.set_xlabel("Island", labelpad=6)
    ax.set_ylabel(ylabel, labelpad=6)
    ax.set_title(title, pad=10, fontweight="bold")

    ax.set_xticks(x + BAR_WIDTH * (len(YEARS) - 1) / 2)
    ax.set_xticklabels(labels, rotation=40, ha="right")

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda val, _: f"{val:,.0f}"
    ))
    ax.set_ylim(bottom=0)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)

    # ── Legend ────────────────────────────────────────────────────────────────
    if show_legend:
        handles = [
            plt.Rectangle((0, 0), 1, 1, fc=YEAR_COLORS[y], ec="white", lw=0.4)
            for y in YEARS
        ]
        ax.legend(
            handles,
            [str(y) for y in YEARS],
            title="Year",
            ncol=4,
            frameon=False,
            loc="upper right",
        )

    fig.tight_layout()

    save_path = output_path.with_suffix(f".{fmt}")
    fig.savefig(save_path, dpi=dpi)
    # NOTE: fig is NOT closed here so plt.show() displays all figures together
    print(f"  Saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(file_path: str, out_dir: str, fmt: str, dpi: int) -> None:
    xlsx    = Path(file_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading: {xlsx}")
    print(f"Saving to: {out_dir}/  [{fmt.upper()}, {dpi} dpi]\n")

    for cfg in CHART_CONFIG:
        sheet = cfg["sheet"]
        print(f"Plotting {sheet} …")

        try:
            df = pd.read_excel(xlsx, sheet_name=sheet)
        except Exception as e:
            print(f"  [SKIP] Could not read sheet '{sheet}': {e}")
            continue

        # Validate column exists
        if cfg["value_col"] not in df.columns:
            print(f"  [SKIP] Column '{cfg['value_col']}' not found in sheet '{sheet}'.")
            continue

        plot_lco(
            df          = df,
            value_col   = cfg["value_col"],
            ylabel      = cfg["ylabel"],
            title       = cfg["title"],
            reference   = cfg["reference"],
            show_legend = cfg["show_legend"],
            output_path = out_dir / cfg["filename"],
            fmt         = fmt,
            dpi         = dpi,
        )

    print("\nAll plots complete.")
    plt.show()   # display all figures; blocks until windows are closed


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--file", default="LCO_results_IP_2050_Final_S1.xlsx",
        help="Path to the LCO Excel results file (default: LCO_results_IP_2050_Final_S1.xlsx)"
    )
    parser.add_argument(
        "--out", default="figures/S_1/",
        help="Output directory for figures (default: figures/S_1/)"
    )
    parser.add_argument(
        "--fmt", default="png", choices=["png", "pdf", "svg", "tiff"],
        help="Output file format (default: png)"
    )
    parser.add_argument(
        "--dpi", default=600, type=int,
        help="Save resolution in DPI (default: 600)"
    )
    args = parser.parse_args()

    main(args.file, args.out, args.fmt, args.dpi)
##########################hourly elec generation profile#############################################################################################################################################################
"""
Pacific Island Countries (PICs) — Hourly Electricity Generation Profiles
=========================================================================
Opens the GAMS GDX file and reads the hourly commodity_balance table
(8760 timesteps × 14 islands × 4 years) to produce stacked-area generation
profiles for each island.

Layout
------
One figure per island  →  2 × 2 grid of subplots (one per target year).
Each subplot shows:
  • Stacked area chart  — hourly generation by technology (GWh)
  • Demand line         — total electricity demand (GWh, black)
  • Curtailment shade   — generation above demand, hatched on top (grey)
  • Curtailment label   — annual curtailment (GWh) and % of demand

Output
------
  figures/generation_profiles/
      CI_model.png
      FJ_model.png
      ...  (one file per island)

Usage
-----
    python pic_generation_profiles.py
    python pic_generation_profiles.py --gdx path/to/results.gdx
                                      --out figures/generation_profiles
                                      --fmt png --dpi 200
                                      --islands CI_model FJ_model
                                      --years 2020 2050
"""

"""
Pacific Island Countries (PICs) — Hourly Electricity Generation Profiles
=========================================================================
Opens the GAMS GDX file and reads the hourly commodity_balance table
(8760 timesteps × 14 islands × 4 years) to produce stacked-area generation
profiles for each island.

Layout
------
One figure per island  →  2 × 2 grid of subplots (one per target year).
Each subplot shows:
  • Stacked area chart  — 2-week centred rolling-average generation by
                          technology (MWh)
  • Demand line         — 2-week rolling-average total electricity demand
                          (MWh, black)
  • Curtailment shade   — rolling-smoothed excess generation above demand,
                          hatched grey on top
  • Top annotation      — annual totals: Gen / Demand / Curtailment (MWh)
                          and curtailment as % of demand

Rolling average
---------------
  window = 336 h (14 days), centred, min_periods=1.
  Annual totals in the annotation are computed from the raw hourly data
  before smoothing, so they are exact.

Output
------
  figures/generation_profiles/
      CI_model.png
      FJ_model.png
      ...  (one PNG per island)

Usage
-----
    python pic_generation_profiles.py
    python pic_generation_profiles.py --gdx path/to/results.gdx
                                      --out figures/generation_profiles
                                      --fmt png --dpi 200
                                      --islands CI_model FJ_model
                                      --years 2020 2050
"""

import argparse
import logging
from pathlib import Path

import gdxpds
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib.patches import Patch

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────
GDX_PATH   = "../GDX_results/IP_2050_Final_S23_minload.gdx"
OUTPUT_DIR = "figures/S_23/generation_profiles_minload_S23"
YEARS      = ["2020", "2030", "2040", "2050"]
ISLANDS    = [
    "CI_model", "FJ_model", "FSM_model", "KB_model",  "MI_model",
    "NU_model", "NE_model", "PU_model",  "PNG_model", "SA_model",
    "SI_model", "TA_model", "TU_model",  "VU_model",
]

# ── Electricity generation technologies (supply side, positive flows) ─────────
EL_PRODUCTION_TECHS = [
    "DG", "PV_B", "NG_plant", "BG_B", "WindOnshore_B", "Hydro_B",
    "BG_N", "PV_N", "WindOnshore_N", "Wave_N", "WindOffshore_N", "Hydro_N", "Geothermal_B", "Battery"
]

# ── Electricity demand techs (consumption side, negative flows in cb) ─────────
All_Elec_demand_tech = ['DW_Electric_converter',
                        'Ammonia_synthesis', 'Aviation_el',  'Bus_el', 'DAC',
                        'HDV_el', 'HP', 'Industry_EL', 'LDV_el', 'MDV_el', 'RO', 'Ship_BEV', 'Two_wheel_el',
                        'cook_el', 'AEL', 'Battery']

# ── Technology display names & colours ────────────────────────────────────────
# Colorblind-safe palette (Wong 2011 + extensions)
TECH_STYLE = {
    "PV_B":           ("Solar PV (behind-meter)", "#F0C234"),
    "PV_N":           ("Solar PV (new)",           "#E6A817"),
    "WindOnshore_B":  ("Wind onshore (existing)",  "#56B4E9"),
    "WindOnshore_N":  ("Wind onshore (new)",       "#0072B2"),
    "WindOffshore_N": ("Wind offshore",            "#004D80"),
    "Wave_N":         ("Wave",                     "#009E73"),
    "Hydro_B":        ("Hydro (existing)",         "#44AA99"),
    "Hydro_N":        ("Hydro (new)",              "#117733"),
    "BG_B":           ("Biogas (existing)",        "#CC79A7"),
    "BG_N":           ("Biogas (new)",             "#882255"),
    "NG_plant":       ("Natural gas",              "#E69F00"),
    "DG":             ("Diesel generator",         "#D55E00"),
    "Geothermal_B":   ("Geothermal",               "#8C564B"),
    "Battery":        ("Battery_dis",              "#B87333"),
}

FALLBACK_COLORS = ["#999999", "#BBBBBB", "#666666", "#444444", "#CCCCCC"]

# ── Rolling-average window ────────────────────────────────────────────────────
ROLLING_WINDOW = 14 * 24   # 336 hours = 2 weeks, centred

# ── Month layout helpers ──────────────────────────────────────────────────────
MONTH_BOUNDARIES = [0, 744, 1416, 2160, 2880, 3624, 4344,
                    5088, 5832, 6552, 7296, 8016, 8760]
MONTH_LABELS     = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTH_MIDPOINTS  = [(MONTH_BOUNDARIES[i] + MONTH_BOUNDARIES[i + 1]) // 2
                    for i in range(12)]

# ─────────────────────────────────────────────────────────────────────────────
# PLOT STYLE  (journal-ready)
# ─────────────────────────────────────────────────────────────────────────────
rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":         9,
    "axes.titlesize":    10,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "axes.linewidth":    0.7,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "xtick.direction":   "out",
    "ytick.direction":   "out",
    "xtick.major.size":  3,
    "ytick.major.size":  3,
    "grid.linewidth":    0.4,
    "grid.alpha":        0.35,
    "grid.color":        "#888888",
    "figure.dpi":        120,
    "savefig.dpi":       200,
    "savefig.bbox":      "tight",
    "savefig.facecolor": "white",
})


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_gdx(path: str) -> dict:
    log.info("Loading GDX: %s", path)
    data = gdxpds.to_dataframes(path)
    log.info("Available symbols: %s", sorted(data.keys()))
    return data


def extract_hourly_cb(data: dict) -> pd.DataFrame:
    """
    Pull the hourly commodity_balance table and pre-filter to Elec only.
    Converts timeModel strings ('tm1'...'tm8760') to 0-based integer hour index.

    Returns DataFrame with columns:
        hour (int 0-8759), island, year, tech, value_gwh
    Positive values = production; negative = consumption.
    """
    cb = data["commodity_balance"]
    log.info("commodity_balance rows: %d", len(cb))

    cb_elec = cb[cb["commodity"] == "Elec"].copy()
    cb_elec["hour"] = (
        cb_elec["timeModel"]
        .str.replace("tm", "", regex=False)
        .astype(int) - 1
    )
    cb_elec.rename(columns={
        "accNodesModel": "island",
        "accYears":      "year",
        "techs":         "tech",
        "Value":         "value_gwh",
    }, inplace=True)

    return cb_elec[["hour", "island", "year", "tech", "value_gwh"]]


# ─────────────────────────────────────────────────────────────────────────────
# PER-ISLAND / PER-YEAR DATA PREP
# ─────────────────────────────────────────────────────────────────────────────

def get_generation(cb_elec: pd.DataFrame, island: str, year: str) -> pd.DataFrame:
    """
    Returns an (8760 x n_techs) DataFrame of raw hourly generation (GWh >= 0)
    for electricity production technologies.
    """
    mask = (
        (cb_elec["island"] == island) &
        (cb_elec["year"]   == year) &
        (cb_elec["tech"].isin(EL_PRODUCTION_TECHS)) &
        (cb_elec["value_gwh"] > 0)
    )
    sub = cb_elec.loc[mask, ["hour", "tech", "value_gwh"]]

    pivot = (
        sub.pivot_table(index="hour", columns="tech",
                        values="value_gwh", aggfunc="sum")
           .reindex(range(8760), fill_value=0.0)
           .fillna(0.0)
    )
    # Drop techs with zero generation in this island-year
    pivot = pivot.loc[:, (pivot > 0).any(axis=0)]
    return pivot


def get_demand(cb_elec: pd.DataFrame, island: str, year: str) -> pd.Series:
    """
    Hourly electricity demand (GWh, positive), indexed 0-8759.
    = |Demand pseudo-tech| + |all converter techs consuming Elec|
    Mirrors the denominator logic from lco_electricity().
    """
    base_mask = (
        (cb_elec["island"] == island) &
        (cb_elec["year"]   == year) &
        (cb_elec["value_gwh"] < 0)
    )
    direct = (
        cb_elec.loc[base_mask & (cb_elec["tech"] == "Demand")]
        .groupby("hour")["value_gwh"].sum()
        .abs()
    )
    conv = (
        cb_elec.loc[base_mask & cb_elec["tech"].isin(All_Elec_demand_tech)]
        .groupby("hour")["value_gwh"].sum()
        .abs()
    )
    return direct.add(conv, fill_value=0.0).reindex(range(8760), fill_value=0.0)


def compute_curtailment(gen: pd.DataFrame, demand: pd.Series):
    """
    Compute curtailment as the hourly surplus of generation over demand.

    curtailment[h] = max(0,  total_gen[h] - demand[h])   [GWh]

    Annual curtailment (GWh) is the sum of these hourly values.
    Curtailment % is expressed relative to annual demand.

    Returns
    -------
    total_gen     : pd.Series  hourly total generation (GWh)
    curtailment   : pd.Series  hourly curtailed energy  (GWh, >= 0)
    curt_annual   : float      annual curtailment sum    (GWh)
    curt_pct      : float      curtailment as % of annual demand
    """
    total_gen     = gen.sum(axis=1)
    curtailment   = (total_gen - demand).clip(lower=0.0)   # surplus each hour
    curt_annual   = float(curtailment.sum())               # GWh
    demand_annual = float(demand.sum())
    curt_pct      = (curt_annual / demand_annual * 100) if demand_annual > 0 else 0.0
    return total_gen, curtailment, curt_annual, curt_pct


# ─────────────────────────────────────────────────────────────────────────────
# ROLLING AVERAGE
# ─────────────────────────────────────────────────────────────────────────────

def rolling_avg(arr: np.ndarray) -> np.ndarray:
    """
    Centred rolling average with window = ROLLING_WINDOW (336 h = 2 weeks).
    Returns a full-length 8760-point smoothed array.
    min_periods=1 prevents NaN at the first/last 168 hours where the full
    window is not available.
    """
    return (
        pd.Series(arr)
        .rolling(window=ROLLING_WINDOW, center=True, min_periods=1)
        .mean()
        .values
    )


# ─────────────────────────────────────────────────────────────────────────────
# COLOUR HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def assign_colors(techs: list) -> dict:
    """Return a {tech: hex_colour} map for every tech in the list."""
    color_map    = {}
    fallback_idx = 0
    for tech in techs:
        if tech in TECH_STYLE:
            color_map[tech] = TECH_STYLE[tech][1]
        else:
            color_map[tech] = FALLBACK_COLORS[fallback_idx % len(FALLBACK_COLORS)]
            fallback_idx   += 1
    return color_map


def tech_label(tech: str) -> str:
    return TECH_STYLE.get(tech, (tech, ""))[0]


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE SUBPLOT RENDERER
# ─────────────────────────────────────────────────────────────────────────────

def draw_subplot(ax, gen: pd.DataFrame, demand: pd.Series,
                 year: str, color_map: dict, show_ylab: bool):
    """
    Draw one year panel into ax.

    Steps
    -----
    1. Compute exact annual totals from raw hourly GWh data.
    2. Apply 2-week centred rolling average to all series.
    3. Convert smoothed values GWh -> MWh for plotting.
    4. Stacked-area chart (techs) + demand line + curtailment hatch.
    5. Print annual summary above the subplot title.
       Curtailment is reported in GWh (summed from raw hourly values).
    """

    # ── Step 1: exact annual totals (raw hourly GWh) ──────────────────────────
    total_gen, curtailment, curt_annual_gwh, curt_pct = compute_curtailment(gen, demand)

    gen_annual_mwh    = float(total_gen.sum()) * 1e3
    demand_annual_mwh = float(demand.sum())    * 1e3

    if gen.empty:
        ax.text(0.5, 0.5, "No generation data",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=9, color="#888888")
        ax.set_title(year, fontweight="bold")
        return

    # ── Steps 2 & 3: rolling average + GWh -> MWh ────────────────────────────
    hours     = np.arange(8760)
    tech_cols = list(gen.columns)

    gen_smooth    = {t: rolling_avg(gen[t].values) * 1e3 for t in tech_cols}
    demand_smooth = rolling_avg(demand.values) * 1e3
    curt_smooth   = rolling_avg(curtailment.values) * 1e3

    # ── Step 4a: stacked area ─────────────────────────────────────────────────
    stack_data = [gen_smooth[t] for t in tech_cols]
    colors     = [color_map.get(t, "#999999") for t in tech_cols]

    ax.stackplot(hours, stack_data,
                 labels=[tech_label(t) for t in tech_cols],
                 colors=colors, linewidth=0, zorder=2)

    # ── Step 4b: curtailment hatch on top of demand ───────────────────────────
    curt_bottom = np.clip(demand_smooth, a_min=0, a_max=None)
    ax.fill_between(hours,
                    curt_bottom,
                    curt_bottom + curt_smooth,
                    color="#AAAAAA", alpha=0.55, hatch="////",
                    linewidth=0, label="Curtailment", zorder=3)

    # ── Step 4c: demand line ──────────────────────────────────────────────────
    ax.plot(hours, demand_smooth,
            color="black", linewidth=1.0, label="Demand", zorder=5)

    # ── Step 5: annual summary above subplot title ────────────────────────────
    # Curtailment is the raw hourly sum (GWh), not converted to MWh, per request
    ax.set_title(year, fontweight="bold", pad=16)
    ax.text(
        0.5, 1.02,
        (f"Gen: {gen_annual_mwh:,.0f} MWh  |  "
         f"Demand: {demand_annual_mwh:,.0f} MWh  |  "
         f"Curtailment: {curt_annual_gwh * 1e3:,.0f} MWh"),
        transform=ax.transAxes,
        ha="center", va="bottom",
        fontsize=7.2, color="#333333", fontstyle="italic",
    )

    # ── X-axis: month labels ──────────────────────────────────────────────────
    ax.set_xlim(0, 8759)
    ax.set_xticks(MONTH_MIDPOINTS)
    ax.set_xticklabels(MONTH_LABELS, fontsize=7.5)

    for mb in MONTH_BOUNDARIES[1:-1]:
        ax.axvline(mb, color="#CCCCCC", linewidth=0.4, zorder=1)

    # ── Y-axis: MWh, comma-formatted ─────────────────────────────────────────
    ax.set_ylim(bottom=0)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)

    if show_ylab:
        ax.set_ylabel("Generation - 2-week rolling avg (MWh)", labelpad=4)

    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v:,.0f}")
    )


# ─────────────────────────────────────────────────────────────────────────────
# PER-ISLAND FIGURE  (2 x 2 subplots)
# ─────────────────────────────────────────────────────────────────────────────

def plot_island(cb_elec: pd.DataFrame, island: str,
                years: list, color_map: dict,
                out_dir: Path, fmt: str, dpi: int):
    """Create and save the 2x2 generation profile figure for one island."""

    island_label = island.replace("_model", "")
    fig, axes    = plt.subplots(2, 2, figsize=(18, 9), sharey=False)
    axes_flat    = axes.flatten()

    fig.suptitle(
        f"{island_label} - Electricity Generation Profile "
        f"(2-week centred rolling average)",
        fontsize=12, fontweight="bold", y=1.02,
    )

    # Collect legend handles across all subplots (deduplicated)
    legend_handles = {}

    for idx, year in enumerate(years):
        ax        = axes_flat[idx]
        show_ylab = (idx % 2 == 0)    # left column only

        gen    = get_generation(cb_elec, island, year)
        demand = get_demand(cb_elec, island, year)

        draw_subplot(ax, gen, demand, year, color_map, show_ylab)

        for handle, lbl in zip(*ax.get_legend_handles_labels()):
            if lbl not in legend_handles:
                legend_handles[lbl] = handle

    # ── Shared legend: generation techs first, then Demand & Curtailment ──────
    special     = {"Demand", "Curtailment"}
    gen_handles = {k: v for k, v in legend_handles.items() if k not in special}
    spc_handles = {k: v for k, v in legend_handles.items() if k in special}

    all_handles = list(gen_handles.values()) + list(spc_handles.values())
    all_labels  = list(gen_handles.keys())   + list(spc_handles.keys())

    fig.legend(
        all_handles, all_labels,
        loc="lower center",
        ncol=min(len(all_labels), 7),
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.04),
    )

    plt.tight_layout(rect=[0, 0.0, 1, 1])

    save_path = out_dir / f"{island}.{fmt}"
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    log.info("  Saved -> %s", save_path)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(gdx_path: str, out_dir: str, fmt: str, dpi: int,
         islands: list, years: list, preloaded_data=None):

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # # Load GDX and extract hourly Elec balance
    # if preloaded_data is not None:
    #     data = preloaded_data
    # else:
    #     data = load_gdx(gdx_path)
    cb_elec = extract_hourly_cb(data)

    log.info("Unique techs in hourly Elec balance: %s",
             sorted(cb_elec["tech"].unique()))

    # Build global colour map (RE first, thermal last)
    present_techs = (
        cb_elec.loc[
            cb_elec["tech"].isin(EL_PRODUCTION_TECHS) & (cb_elec["value_gwh"] > 0),
            "tech",
        ].unique().tolist()
    )
    ordered   = [t for t in EL_PRODUCTION_TECHS if t in present_techs]
    color_map = assign_colors(ordered)

    # Plot each island
    for island in islands:
        log.info("Plotting %s ...", island)
        plot_island(cb_elec, island, years, color_map, out_path, fmt, dpi)

    log.info("Done. Figures saved to: %s", out_path)
    print(f"\nAll {len(islands)} island figures written to: {out_path}/")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--gdx",     default=GDX_PATH,
                        help="Path to GAMS GDX file")
    parser.add_argument("--out",     default=OUTPUT_DIR,
                        help="Output directory for figures")
    parser.add_argument("--fmt",     default="png",
                        choices=["png", "pdf", "svg", "tiff"],
                        help="Output file format (default: png)")
    parser.add_argument("--dpi",     default=200, type=int,
                        help="Save resolution in DPI (default: 200)")
    parser.add_argument("--islands", nargs="+", default=ISLANDS,
                        help="Subset of islands to plot (default: all 14)")
    parser.add_argument("--years",   nargs="+", default=YEARS,
                        help="Target years (default: 2020 2030 2040 2050)")
    args = parser.parse_args()

    main(args.gdx, args.out, args.fmt, args.dpi, args.islands, args.years)
    ################################################################################################################battery heatmap##############################
"""
Pacific Island Countries (PICs) — Battery State of Charge Heatmaps
====================================================================
Reads 'Storage_level_out' from the GAMS GDX file and produces
24 × 365 heatmaps of hourly battery State of Charge (SOC) expressed
as a percentage of the annual peak storage level.

Layout
------
One figure per island  →  1 × 3 row of subplots (one per target year).
Each subplot is a heatmap where:
    x-axis  : month labels (Jan–Dec)
    y-axis  : hour of day (0–23)
    colour  : SOC (%) = value / max(value for that island-year) × 100

SOC = 0 % → battery empty   |   SOC = 100 % → battery at peak capacity

Output
------
  figures/battery_soc/
      CI_model.png
      FJ_model.png
      ...  (one PNG per island)

Usage
-----
    python pic_battery_soc.py
    python pic_battery_soc.py --gdx path/to/results.gdx
                              --out figures/battery_soc
                              --fmt png --dpi 200
                              --islands CI_model FJ_model
                              --years 2030 2040 2050
"""

import argparse
import logging
from pathlib import Path

import gdxpds
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────
GDX_PATH   = "../GDX_results/IP_2050_Final_S23_minload.gdx"
OUTPUT_DIR = "figures/S_23/battery_soc_minload"
YEARS      = ["2030", "2040", "2050"]
ISLANDS    = [
    "CI_model", "FJ_model", "FSM_model", "KB_model",  "MI_model",
    "NU_model", "NE_model", "PU_model",  "PNG_model", "SA_model",
    "SI_model", "TA_model", "TU_model",  "VU_model",
]

BATTERY_TECH = "Battery"

# ── Month axis ────────────────────────────────────────────────────────────────
MONTH_BOUNDARIES = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]
MONTH_LABELS     = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]
MONTH_MIDPOINTS  = [(MONTH_BOUNDARIES[i] + MONTH_BOUNDARIES[i + 1]) / 2
                    for i in range(12)]

# ─────────────────────────────────────────────────────────────────────────────
# COLOUR MAP
# YlGnBu-inspired: pale yellow (empty) → mint → teal → deep ocean blue (full)
# Perceptually uniform, colorblind-safe, prints well in greyscale too.
# ─────────────────────────────────────────────────────────────────────────────
SOC_CMAP = LinearSegmentedColormap.from_list(
    "soc",
    ["#FFFFCC", "#C7E9B4", "#41B6C4", "#1D91C0", "#0C2C84"],
    N=256,
)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT STYLE
# IMPORTANT: savefig.bbox is intentionally NOT "tight".
# "tight" re-expands the canvas after layout and drags the colorbar axis
# back over the rightmost heatmap panel.  All margins are controlled via
# gridspec left/right/top/bottom fractions instead.
# ─────────────────────────────────────────────────────────────────────────────
rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":         9,
    "axes.titlesize":    10,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "axes.linewidth":    0.7,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        120,
    "savefig.dpi":       200,
    "savefig.facecolor": "white",
    # savefig.bbox deliberately omitted (defaults to "standard")
})


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_gdx(path: str) -> dict:
    log.info("Loading GDX: %s", path)
    data = gdxpds.to_dataframes(path)
    log.info("Available symbols: %s", sorted(data.keys()))
    return data


def extract_battery_soc(data: dict) -> pd.DataFrame:
    sl  = data["storage_level_out"]
    bat = sl[sl["techs"] == BATTERY_TECH].copy()

    if bat.empty:
        log.warning("No rows for tech='%s'.", BATTERY_TECH)
        return pd.DataFrame(columns=["hour", "island", "year", "soc_gwh"])

    bat["hour"] = (
        bat["timeModel"].str.replace("tm", "", regex=False).astype(int) - 1
    )
    bat.rename(columns={
        "accNodesModel": "island",
        "accYears":      "year",
        "Value":         "soc_gwh",
    }, inplace=True)

    bat = bat.groupby(["hour", "island", "year"], as_index=False)["soc_gwh"].sum()
    log.info("Battery rows: %d", len(bat))
    return bat[["hour", "island", "year", "soc_gwh"]]


# ─────────────────────────────────────────────────────────────────────────────
# SOC MATRIX BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_soc_matrix(bat, island, year):
    mask = (bat["island"] == island) & (bat["year"] == year)
    sub  = bat.loc[mask, ["hour", "soc_gwh"]].copy()

    if sub.empty:
        return np.full((24, 365), np.nan), 0.0, 0.0, 0.0

    full = pd.Series(0.0, index=range(8760))
    full.update(sub.set_index("hour")["soc_gwh"])

    peak_gwh = float(full.max())
    if peak_gwh == 0:
        return np.full((24, 365), np.nan), 0.0, 0.0, 0.0

    soc_pct        = (full / peak_gwh * 100).values
    matrix         = soc_pct.reshape(365, 24).T
    avg_soc        = float(soc_pct.mean())
    hours_above_80 = float((soc_pct > 80).mean() * 100)

    return matrix, peak_gwh, avg_soc, hours_above_80


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE SUBPLOT RENDERER
# ─────────────────────────────────────────────────────────────────────────────

def draw_soc_heatmap(ax, matrix, year, peak_gwh, avg_soc,
                     hours_above_80, show_ylab):

    if np.all(np.isnan(matrix)):
        ax.text(0.5, 0.5, "No battery data", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="#888888")
        ax.set_title(year, fontweight="bold")
        return None

    im = ax.imshow(
        matrix,
        aspect="auto",
        origin="upper",
        cmap=SOC_CMAP,
        vmin=0, vmax=100,
        interpolation="nearest",
        extent=[1, 365, 23.5, -0.5],
    )

    # ── X-axis: month names, no overlapping day numbers ───────────────────────
    ax.set_xticks(MONTH_MIDPOINTS)
    ax.set_xticklabels(MONTH_LABELS, fontsize=8)
    ax.set_xticks(MONTH_BOUNDARIES, minor=True)
    ax.tick_params(axis="x", which="major", length=0)          # labels only
    ax.tick_params(axis="x", which="minor", length=3,
                   color="#aaaaaa", width=0.5)
    ax.set_xlim(1, 365)
    ax.set_xlabel("Month", labelpad=4)

    for mb in MONTH_BOUNDARIES[1:-1]:
        ax.axvline(mb, color="white", linewidth=0.4, alpha=0.55)

    # ── Y-axis ────────────────────────────────────────────────────────────────
    ax.set_yticks([0, 6, 12, 18, 23])
    ax.set_yticklabels(["00:00", "06:00", "12:00", "18:00", "23:00"],
                       fontsize=7.5)
    ax.set_ylim(23.5, -0.5)
    if show_ylab:
        ax.set_ylabel("Hour of day", labelpad=4)

    # ── Title + stats ─────────────────────────────────────────────────────────
    ax.set_title(year, fontweight="bold", pad=14)
    ax.text(
        0.5, 1.01,
        (f"Peak: {peak_gwh:.3f} GWh  |  "
         f"Avg SOC: {avg_soc:.1f}%  |  "
         f"Hours >80%: {hours_above_80:.1f}%"),
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=7.2, color="#444444", fontstyle="italic",
    )

    return im


# ─────────────────────────────────────────────────────────────────────────────
# PER-ISLAND FIGURE
# ─────────────────────────────────────────────────────────────────────────────

def plot_island_soc(bat, island, years, out_dir, fmt, dpi):

    island_label = island.replace("_model", "")
    n_years      = len(years)

    panel_w = 5.5                              # inches per heatmap panel
    cbar_w  = 0.3                              # inches for colorbar column
    gap     = 0.55                             # right-side whitespace (inches)
    fig_w   = panel_w * n_years + cbar_w + gap + 0.7   # 0.7 for left margin
    fig_h   = 5.2

    fig = plt.figure(figsize=(fig_w, fig_h))

    # right edge fraction: leaves exactly (cbar_w + gap) inches for cbar+space
    right_frac = 1.0 - (cbar_w + gap) / fig_w

    gs = fig.add_gridspec(
        1, n_years + 1,
        width_ratios=[panel_w] * n_years + [cbar_w],
        wspace=0.06,
        left=0.07,
        right=right_frac,
        top=0.80,
        bottom=0.14,
    )

    axes    = [fig.add_subplot(gs[0, i]) for i in range(n_years)]
    cbar_ax = fig.add_subplot(gs[0, n_years])

    for ax in axes[1:]:
        ax.sharey(axes[0])

    fig.suptitle(
        f"{island_label} — Battery State of Charge (% of annual peak)",
        fontsize=11, fontweight="bold", y=0.95,
    )

    last_im = None
    for idx, year in enumerate(years):
        matrix, peak_gwh, avg_soc, h80 = build_soc_matrix(bat, island, year)
        im = draw_soc_heatmap(
            axes[idx], matrix, year, peak_gwh, avg_soc, h80,
            show_ylab=(idx == 0),
        )
        if im is not None:
            last_im = im
        if idx > 0:
            plt.setp(axes[idx].get_yticklabels(), visible=False)

    # ── Colorbar anchored to its own dedicated axis ───────────────────────────
    if last_im is not None:
        cbar = fig.colorbar(last_im, cax=cbar_ax)
        cbar.set_label("State of Charge (%)", fontsize=9, labelpad=8)
        cbar.set_ticks([0, 20, 40, 60, 80, 100])
        cbar.ax.tick_params(labelsize=8)
        for t in [20, 40, 60, 80]:
            cbar.ax.axhline(t, color="white", linewidth=0.5, alpha=0.6)
    else:
        cbar_ax.set_visible(False)

    save_path = out_dir / f"{island}.{fmt}"
    fig.savefig(save_path, dpi=dpi)      # no bbox_inches="tight"
    plt.close(fig)
    log.info("  Saved -> %s", save_path)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(gdx_path, out_dir, fmt, dpi, islands, years, preloaded_data=None):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    # if preloaded_data is not None:
    #     data = preloaded_data
    # else:
    #     data = load_gdx(gdx_path)

    bat  = extract_battery_soc(data)

    if bat.empty:
        log.error("No battery data extracted.")
        return

    log.info("Islands: %s", sorted(bat["island"].unique()))
    log.info("Years  : %s", sorted(bat["year"].unique()))

    for island in islands:
        log.info("Plotting %s ...", island)
        plot_island_soc(bat, island, years, out_path, fmt, dpi)

    log.info("Done. Figures saved to: %s", out_path)
    print(f"\nAll {len(islands)} island SOC figures written to: {out_path}/")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--gdx",     default=GDX_PATH)
    parser.add_argument("--out",     default=OUTPUT_DIR)
    parser.add_argument("--fmt",     default="png",
                        choices=["png", "pdf", "svg", "tiff"])
    parser.add_argument("--dpi",     default=200, type=int)
    parser.add_argument("--islands", nargs="+", default=ISLANDS)
    parser.add_argument("--years",   nargs="+", default=YEARS)
    parser.add_argument("--tech",    default=BATTERY_TECH)
    args = parser.parse_args()

    BATTERY_TECH = args.tech
    main(args.gdx, args.out, args.fmt, args.dpi, args.islands, args.years)
####################thermal storage SOC##################################################################
import argparse
import logging
from pathlib import Path

import gdxpds
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
log = logging.getLogger(__name__)

GDX_PATH   = "../GDX_results/IP_2050_Final_S23_minload.gdx"
OUTPUT_DIR = "figures/S_23/thermal_soc_minload"
YEARS      = ["2030", "2040", "2050"]
ISLANDS    = [
    "CI_model", "FJ_model", "FSM_model", "KB_model",  "MI_model",
    "NU_model", "NE_model", "PU_model",  "PNG_model", "SA_model",
    "SI_model", "TA_model", "TU_model",  "VU_model",
]

STORAGE_TECH = "THSS"   # Thermal Heat Storage System

MONTH_BOUNDARIES = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]
MONTH_LABELS     = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]
MONTH_MIDPOINTS  = [(MONTH_BOUNDARIES[i] + MONTH_BOUNDARIES[i + 1]) / 2
                    for i in range(12)]

# Warm orange-red palette — visually distinct from the blue battery SOC map
SOC_CMAP = LinearSegmentedColormap.from_list(
    "thermal_soc",
    ["#FFFFB2", "#FECC5C", "#FD8D3C", "#E31A1C", "#800026"],
    N=256,
)

rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":         9,
    "axes.titlesize":    10,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "axes.linewidth":    0.7,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        120,
    "savefig.dpi":       200,
    "savefig.facecolor": "white",
})


def load_gdx(path: str) -> dict:
    log.info("Loading GDX: %s", path)
    data = gdxpds.to_dataframes(path)
    log.info("Available symbols: %s", sorted(data.keys()))
    return data


def extract_storage_soc(data: dict) -> pd.DataFrame:
    sl   = data["storage_level_out"]
    thss = sl[sl["techs"] == STORAGE_TECH].copy()   # commodity filter not needed

    if thss.empty:
        log.warning("No rows for tech='%s'.", STORAGE_TECH)
        return pd.DataFrame(columns=["hour", "island", "year", "soc_gwh"])

    thss["hour"] = (
        thss["timeModel"].str.replace("tm", "", regex=False).astype(int) - 1
    )
    thss.rename(columns={
        "accNodesModel": "island",
        "accYears":      "year",
        "Value":         "soc_gwh",
    }, inplace=True)

    thss = thss.groupby(["hour", "island", "year"], as_index=False)["soc_gwh"].sum()
    log.info("Thermal storage rows: %d", len(thss))
    return thss[["hour", "island", "year", "soc_gwh"]]


def build_soc_matrix(soc_df, island, year):
    mask = (soc_df["island"] == island) & (soc_df["year"] == year)
    sub  = soc_df.loc[mask, ["hour", "soc_gwh"]].copy()

    if sub.empty:
        return np.full((24, 365), np.nan), 0.0, 0.0, 0.0

    full = pd.Series(0.0, index=range(8760))
    full.update(sub.set_index("hour")["soc_gwh"])

    peak_gwh = float(full.max())
    if peak_gwh == 0:
        return np.full((24, 365), np.nan), 0.0, 0.0, 0.0

    soc_pct        = (full / peak_gwh * 100).values
    matrix         = soc_pct.reshape(365, 24).T
    avg_soc        = float(soc_pct.mean())
    hours_above_80 = float((soc_pct > 80).mean() * 100)

    return matrix, peak_gwh, avg_soc, hours_above_80


def draw_soc_heatmap(ax, matrix, year, peak_gwh, avg_soc,
                     hours_above_80, show_ylab):
    if np.all(np.isnan(matrix)):
        ax.text(0.5, 0.5, "No thermal storage data",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=9, color="#888888")
        ax.set_title(year, fontweight="bold")
        return None

    im = ax.imshow(
        matrix, aspect="auto", origin="upper",
        cmap=SOC_CMAP, vmin=0, vmax=100,
        interpolation="nearest",
        extent=[1, 365, 23.5, -0.5],
    )

    ax.set_xticks(MONTH_MIDPOINTS)
    ax.set_xticklabels(MONTH_LABELS, fontsize=8)
    ax.set_xticks(MONTH_BOUNDARIES, minor=True)
    ax.tick_params(axis="x", which="major", length=0)
    ax.tick_params(axis="x", which="minor", length=3, color="#aaaaaa", width=0.5)
    ax.set_xlim(1, 365)
    ax.set_xlabel("Month", labelpad=4)

    for mb in MONTH_BOUNDARIES[1:-1]:
        ax.axvline(mb, color="white", linewidth=0.4, alpha=0.55)

    ax.set_yticks([0, 6, 12, 18, 23])
    ax.set_yticklabels(["00:00","06:00","12:00","18:00","23:00"], fontsize=7.5)
    ax.set_ylim(23.5, -0.5)
    if show_ylab:
        ax.set_ylabel("Hour of day", labelpad=4)

    ax.set_title(year, fontweight="bold", pad=14)
    ax.text(
        0.5, 1.01,
        (f"Peak: {peak_gwh:.3f} GWh  |  "
         f"Avg SOC: {avg_soc:.1f}%  |  "
         f"Hours >80%: {hours_above_80:.1f}%"),
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=7.2, color="#444444", fontstyle="italic",
    )
    return im


def plot_island_soc(soc_df, island, years, out_dir, fmt, dpi):
    island_label = island.replace("_model", "")
    n_years      = len(years)

    panel_w = 5.5
    cbar_w  = 0.3
    gap     = 0.55
    fig_w   = panel_w * n_years + cbar_w + gap + 0.7
    fig_h   = 5.2

    fig        = plt.figure(figsize=(fig_w, fig_h))
    right_frac = 1.0 - (cbar_w + gap) / fig_w

    gs = fig.add_gridspec(
        1, n_years + 1,
        width_ratios=[panel_w] * n_years + [cbar_w],
        wspace=0.06, left=0.07, right=right_frac, top=0.80, bottom=0.14,
    )

    axes    = [fig.add_subplot(gs[0, i]) for i in range(n_years)]
    cbar_ax = fig.add_subplot(gs[0, n_years])

    for ax in axes[1:]:
        ax.sharey(axes[0])

    fig.suptitle(
        f"{island_label} — Thermal Storage State of Charge (% of annual peak)",
        fontsize=11, fontweight="bold", y=0.95,
    )

    last_im = None
    for idx, year in enumerate(years):
        matrix, peak_gwh, avg_soc, h80 = build_soc_matrix(soc_df, island, year)
        im = draw_soc_heatmap(
            axes[idx], matrix, year, peak_gwh, avg_soc, h80,
            show_ylab=(idx == 0),
        )
        if im is not None:
            last_im = im
        if idx > 0:
            plt.setp(axes[idx].get_yticklabels(), visible=False)

    if last_im is not None:
        cbar = fig.colorbar(last_im, cax=cbar_ax)
        cbar.set_label("State of Charge (%)", fontsize=9, labelpad=8)
        cbar.set_ticks([0, 20, 40, 60, 80, 100])
        cbar.ax.tick_params(labelsize=8)
        for t in [20, 40, 60, 80]:
            cbar.ax.axhline(t, color="white", linewidth=0.5, alpha=0.6)
    else:
        cbar_ax.set_visible(False)

    save_path = out_dir / f"{island}.{fmt}"
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    log.info("  Saved -> %s", save_path)


def main(gdx_path, out_dir, fmt, dpi, islands, years, preloaded_data=None):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # if preloaded_data is not None:
    #     log.info("Using pre-loaded GDX data — skipping file read.")
    #     data = preloaded_data
    # else:
    #     data = load_gdx(gdx_path)

    soc_df = extract_storage_soc(data)

    if soc_df.empty:
        log.error("No thermal storage data extracted.")
        return

    log.info("Islands: %s", sorted(soc_df["island"].unique()))
    log.info("Years  : %s", sorted(soc_df["year"].unique()))

    for island in islands:
        log.info("Plotting %s ...", island)
        plot_island_soc(soc_df, island, years, out_path, fmt, dpi)

    log.info("Done. Figures saved to: %s", out_path)
    print(f"\nAll {len(islands)} island thermal SOC figures written to: {out_path}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gdx",     default=GDX_PATH)
    parser.add_argument("--out",     default=OUTPUT_DIR)
    parser.add_argument("--fmt",     default="png", choices=["png","pdf","svg","tiff"])
    parser.add_argument("--dpi",     default=200, type=int)
    parser.add_argument("--islands", nargs="+", default=ISLANDS)
    parser.add_argument("--years",   nargs="+", default=YEARS)
    args = parser.parse_args()

    main(args.gdx, args.out, args.fmt, args.dpi, args.islands, args.years)
##########################################battery capacity and their generation######################


"""
battery_viz.py
--------------
For each of the 14 Pacific island models, produces a dual-axis 6-bar chart:
  LEFT  3 bars  — Battery storage capacity (GWh)   [2030, 2040, 2050]
  RIGHT 3 bars  — Power generation from battery (GWh) [2030, 2040, 2050]
  A dotted vertical line separates the two groups.

  Left  y-axis → capacity scale
  Right y-axis → generation scale

Source tables:
  storage_caps         → techs=Battery, capType=total, commodity=Elec_LiIon, +ve
  storage_flows_annual → techs=Battery, commodity=Elec_LiIon, balanceType=positive, +ve

Usage:
    python battery_viz.py --gdx path/to/results.gdx --out battery_charts/
"""

import argparse
import logging
from pathlib import Path

import gdxpds
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
log = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
GDX_PATH   = "../GDX_results/IP_2050_Final_S23_minload.gdx"
OUTPUT_DIR = "figures/S_23/battery_charts"

YEARS = ["2030", "2040", "2050"]

ISLANDS = [
    "CI_model", "FJ_model", "FSM_model", "KB_model", "MI_model",
    "NU_model", "NE_model", "PU_model", "PNG_model", "SA_model",
    "SI_model", "TA_model", "TU_model", "VU_model",
]

ISLAND_LABELS = {
    "CI_model":  "Cook Islands",
    "FJ_model":  "Fiji",
    "FSM_model": "Federated States of Micronesia",
    "KB_model":  "Kiribati",
    "MI_model":  "Marshall Islands",
    "NU_model":  "Nauru",
    "NE_model":  "Niue",
    "PU_model":  "Palau",
    "PNG_model": "Papua New Guinea",
    "SA_model":  "Samoa",
    "SI_model":  "Solomon Islands",
    "TA_model":  "Tonga",
    "TU_model":  "Tuvalu",
    "VU_model":  "Vanuatu",
}

# Capacity bars (left axis) — solid colours
CAP_COLORS  = {"2030": "#4472C4", "2040": "#ED7D31", "2050": "#70AD47"}
# Generation bars (right axis) — hatched, slightly lighter tones
GEN_COLORS  = {"2030": "#A9C0E8", "2040": "#F7C09A", "2050": "#B8DDA0"}

BAR_W = 0.28   # width of every bar

# X-positions: capacity group centred at 0, generation group centred at 1
# Each group has 3 bars offset by BAR_W around the group centre
GROUP_GAP = 1.0          # distance between the two group centres
CAP_CTR   = 0.0
GEN_CTR   = CAP_CTR + GROUP_GAP

OFFSETS = {              # offset within a group for each year
    "2030": -BAR_W,
    "2040":  0.0,
    "2050":  BAR_W,
}

CAP_POSITIONS  = {y: CAP_CTR + OFFSETS[y] for y in YEARS}
GEN_POSITIONS  = {y: GEN_CTR + OFFSETS[y] for y in YEARS}

# Dotted separator x position — midway between the two groups
SEP_X = (CAP_CTR + BAR_W/2 + GEN_CTR - BAR_W/2) / 2   # ≈ 0.5

# Unit conversion: storage_caps values come out of the GDX in GWh.
# Capacities are displayed in MWh, generation stays in GWh.
CAP_UNIT_SCALE = 1000.0   # GWh -> MWh


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(gdx_path: str):
    log.info("Loading GDX: %s", gdx_path)
    # if preloaded_data is not None:
    #     data = preloaded_data
    # else:
    #     data = load_data(gdx_path)

    caps = data["storage_caps"].copy()
    log.info("storage_caps columns: %s", caps.columns.tolist())
    caps = caps[
        (caps["techs"]        == "Battery") &
        (caps["capType"]      == "total") &
        (caps["commodity"]    == "Elec_LiIon") &
        (caps["accYears"].isin(YEARS)) &
        (caps["accNodesModel"].isin(ISLANDS)) &
        (caps["Value"]        >  0)
    ].copy()
    log.info("storage_caps after filter: %d rows", len(caps))

    flows = data["storage_flows_annual"].copy()
    log.info("storage_flows_annual columns: %s", flows.columns.tolist())
    flows = flows[
        (flows["techs"]        == "Battery") &
        (flows["commodity"]    == "Elec_LiIon") &
        (flows["balanceType"]  == "positive") &
        (flows["accYears"].isin(YEARS)) &
        (flows["accNodesModel"].isin(ISLANDS)) &
        (flows["Value"]        >  0)
    ].copy()
    log.info("storage_flows_annual after filter: %d rows", len(flows))

    return caps, flows


def get_value(df, island, year):
    mask = (df["accNodesModel"] == island) & (df["accYears"] == year)
    return float(df.loc[mask, "Value"].sum())


# ── Single-island plot ────────────────────────────────────────────────────────

def plot_island(island, caps, flows, output_dir):
    label    = ISLAND_LABELS.get(island, island)
    cap_vals = {y: get_value(caps,  island, y) * CAP_UNIT_SCALE for y in YEARS}
    gen_vals = {y: get_value(flows, island, y) for y in YEARS}

    if all(v == 0 for v in cap_vals.values()) and all(v == 0 for v in gen_vals.values()):
        log.info("No battery data for %s — skipping.", label)
        return

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    cap_max  = max(cap_vals.values())  or 1
    gen_max  = max(gen_vals.values())  or 1

    # ── 3 capacity bars (left axis) ───────────────────────────────────────────
    for year in YEARS:
        x = CAP_POSITIONS[year]
        v = cap_vals[year]
        ax1.bar(x, v, width=BAR_W * 0.90,
                color=CAP_COLORS[year], alpha=0.88, zorder=3,
                label=f"Capacity {year}")
        if v > 0:
            ax1.text(x, v + cap_max * 0.025, f"{v:,.1f}",
                     ha="center", va="bottom", fontsize=7.5,
                     color=CAP_COLORS[year], fontweight="bold")

    # ── 3 generation bars (right axis) ────────────────────────────────────────
    for year in YEARS:
        x = GEN_POSITIONS[year]
        v = gen_vals[year]
        ax2.bar(x, v, width=BAR_W * 0.90,
                color=GEN_COLORS[year], alpha=0.88, zorder=3,
                hatch="//", edgecolor="grey", linewidth=0.5,
                label=f"Generation {year}")
        if v > 0:
            ax2.text(x, v + gen_max * 0.025, f"{v:.1f}",
                     ha="center", va="bottom", fontsize=7.5,
                     color="dimgrey", fontweight="bold")

    # ── Dotted separator ──────────────────────────────────────────────────────
    ax1.axvline(x=SEP_X, color="grey", linestyle=":", linewidth=1.5, alpha=0.8, zorder=5)

    # ── Group header labels ───────────────────────────────────────────────────
    y_header = cap_max * 1.18
    ax1.text(CAP_CTR, y_header, "Battery Capacity",
             ha="center", va="bottom", fontsize=9, color="#2F4F8F",
             fontweight="bold", transform=ax1.transData)
    # For generation label use ax1 transform but position on gen side
    ax1.text(GEN_CTR, y_header, "Power Generation",
             ha="center", va="bottom", fontsize=9, color="#555555",
             fontweight="bold", transform=ax1.transData)

    # ── Year tick labels (shared x-axis) ─────────────────────────────────────
    tick_positions = (
        [CAP_POSITIONS[y] for y in YEARS] +
        [GEN_POSITIONS[y] for y in YEARS]
    )
    tick_labels = YEARS + YEARS
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels, fontsize=8, color="dimgrey")

    # ── Axis limits & labels ──────────────────────────────────────────────────
    ax1.set_xlim(CAP_CTR - BAR_W * 2, GEN_CTR + BAR_W * 2)
    ax1.set_ylim(0, cap_max * 1.30)
    ax2.set_ylim(0, gen_max * 1.30)

    ax1.set_ylabel("Battery Storage Capacity (MWh)", fontsize=10, color="#2F4F8F")
    ax2.set_ylabel("Power Output from Battery (GWh)", fontsize=10, color="#555555")
    ax1.tick_params(axis="y", labelcolor="#2F4F8F")
    ax2.tick_params(axis="y", labelcolor="#555555")

    ax1.set_title(f"{label}",
                  fontsize=12, fontweight="bold", pad=12)

    # ── Legend ────────────────────────────────────────────────────────────────
    cap_patches = [mpatches.Patch(facecolor=CAP_COLORS[y], label=f"Capacity {y}",
                                  alpha=0.88) for y in YEARS]
    gen_patches = [mpatches.Patch(facecolor=GEN_COLORS[y], label=f"Generation {y}",
                                  alpha=0.88, hatch="//", edgecolor="grey") for y in YEARS]
    ax1.legend(handles=cap_patches + gen_patches,
               loc="upper left", fontsize=7.5, framealpha=0.8, ncol=2)

    ax1.yaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
    ax1.set_axisbelow(True)

    fig.tight_layout()
    out_file = output_dir / f"battery_{island}.png"
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", out_file)


# ── Summary: all 14 islands, 4×4 subplots ────────────────────────────────────

def plot_all_islands_summary(caps, flows, output_dir):
    fig, axes = plt.subplots(4, 4, figsize=(22, 18))
    axes_flat = axes.flatten()

    for idx, island in enumerate(ISLANDS):
        ax1 = axes_flat[idx]
        ax2 = ax1.twinx()
        label    = ISLAND_LABELS.get(island, island)
        cap_vals = {y: get_value(caps,  island, y) * CAP_UNIT_SCALE for y in YEARS}
        gen_vals = {y: get_value(flows, island, y) for y in YEARS}

        cap_max = max(cap_vals.values()) or 1
        gen_max = max(gen_vals.values()) or 1

        for year in YEARS:
            ax1.bar(CAP_POSITIONS[year], cap_vals[year], width=BAR_W * 0.88,
                    color=CAP_COLORS[year], alpha=0.88, zorder=3)
            ax2.bar(GEN_POSITIONS[year], gen_vals[year], width=BAR_W * 0.88,
                    color=GEN_COLORS[year], alpha=0.88, zorder=3,
                    hatch="//", edgecolor="grey", linewidth=0.4)

        ax1.axvline(x=SEP_X, color="grey", linestyle=":", linewidth=1.2, alpha=0.7)

        tick_positions = [CAP_POSITIONS[y] for y in YEARS] + [GEN_POSITIONS[y] for y in YEARS]
        tick_labels    = [y[-2:] for y in YEARS] + [y[-2:] for y in YEARS]
        ax1.set_xticks(tick_positions)
        ax1.set_xticklabels(tick_labels, fontsize=5.5, color="dimgrey")

        ax1.set_xlim(CAP_CTR - BAR_W * 2, GEN_CTR + BAR_W * 2)
        ax1.set_ylim(0, cap_max * 1.30)
        ax2.set_ylim(0, gen_max * 1.30)

        ax1.set_title(label, fontsize=7.5, fontweight="bold", pad=3)
        ax1.tick_params(axis="y", labelsize=5.5, labelcolor="#2F4F8F")
        ax2.tick_params(axis="y", labelsize=5.5, labelcolor="#555555")
        ax1.yaxis.grid(True, linestyle="--", alpha=0.3, zorder=0)
        ax1.set_axisbelow(True)

    for idx in range(len(ISLANDS), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Shared legend
    cap_patches = [mpatches.Patch(facecolor=CAP_COLORS[y], label=f"Capacity {y}",
                                  alpha=0.88) for y in YEARS]
    gen_patches = [mpatches.Patch(facecolor=GEN_COLORS[y], label=f"Generation {y}",
                                  alpha=0.88, hatch="//", edgecolor="grey") for y in YEARS]
    fig.legend(handles=cap_patches + gen_patches,
               loc="lower center", ncol=6, fontsize=9,
               framealpha=0.85, bbox_to_anchor=(0.5, 0.005))

    fig.suptitle(
        "Battery Storage Capacity (solid bars, left axis, MWh)  |  "
        "Power Generation from Battery (hatched bars, right axis, GWh)\n"
        "Pacific Island Countries — 2030 / 2040 / 2050",
        fontsize=12, fontweight="bold", y=0.998)

    fig.tight_layout(rect=[0, 0.055, 1, 0.998])
    out_file = output_dir / "battery_ALL_ISLANDS_summary.png"
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Summary chart saved: %s", out_file)


# ── Entry point ───────────────────────────────────────────────────────────────

def main(gdx_path, output_dir_str):
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)
    caps, flows = load_data(gdx_path)
    for island in ISLANDS:
        plot_island(island, caps, flows, output_dir)
    plot_all_islands_summary(caps, flows, output_dir)
    log.info("All charts written to: %s", output_dir.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gdx", default=GDX_PATH)
    parser.add_argument("--out", default=OUTPUT_DIR)
    args = parser.parse_args()
    main(args.gdx, args.out)
    ##################################H2_storage##########################################
"""
Pacific Island Countries (PICs) — Hydrogen Storage Level Heatmaps
==================================================================
Reads 'storage_level_out' from the GAMS GDX file and produces
24 x 365 heatmaps of hydrogen storage level expressed as a percentage
of the annual peak storage level.

    Storage level (%) = value / max(value for that island-year) x 100

    0 %   -> H2 storage empty
    100 % -> H2 storage at annual peak capacity

Layout
------
One figure per island  ->  1 x 3 row of subplots (2030 | 2040 | 2050).
Each subplot is a heatmap where:
    x-axis  : month labels (Jan-Dec)
    y-axis  : hour of day (0-23, midnight at top)
    colour  : storage level (%)

Source
------
    Symbol    : storage_level_out
    Tech      : H2_storage
    Commodity : Hydrogen_T

Output
------
  figures/hydrogen_storage/
      CI_model.png
      FJ_model.png
      ...  (one PNG per island)

Usage
-----
    python pic_h2_storage.py
    python pic_h2_storage.py --gdx path/to/results.gdx
                              --out figures/hydrogen_storage
                              --fmt png --dpi 200
                              --islands CI_model FJ_model
                              --years 2030 2040 2050
"""

import argparse
import logging
from pathlib import Path

import gdxpds
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────
GDX_PATH    = "../GDX_results/IP_2050_Final_S23_minload.gdx"
OUTPUT_DIR  = "figures/S_23/hydrogen_storage_minload"
YEARS       = ["2030", "2040", "2050"]
ISLANDS     = [
    "CI_model", "FJ_model", "FSM_model", "KB_model",  "MI_model",
    "NU_model", "NE_model", "PU_model",  "PNG_model", "SA_model",
    "SI_model", "TA_model", "TU_model",  "VU_model",
]

H2_TECH      = "H2_storage"
H2_COMMODITY = "Hydrogen_T"

# ── Month axis ────────────────────────────────────────────────────────────────
MONTH_BOUNDARIES = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]
MONTH_LABELS     = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]
MONTH_MIDPOINTS  = [(MONTH_BOUNDARIES[i] + MONTH_BOUNDARIES[i + 1]) / 2
                    for i in range(12)]

# ─────────────────────────────────────────────────────────────────────────────
# COLOUR MAP
# Warm yellow-green -> teal -> deep blue-green (same family as battery SOC
# but shifted toward green to visually distinguish H2 from battery plots)
# ─────────────────────────────────────────────────────────────────────────────
H2_CMAP = LinearSegmentedColormap.from_list(
    "h2_storage",
    ["#FFFFCC", "#C7E9B4", "#41B6C4", "#1D91C0", "#0C2C84"],
    N=256,
)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT STYLE
# NOTE: savefig.bbox is intentionally NOT "tight" — margins are controlled
# explicitly via gridspec to prevent the colorbar from overlapping panels.
# ─────────────────────────────────────────────────────────────────────────────
rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":         9,
    "axes.titlesize":    10,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "axes.linewidth":    0.7,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        120,
    "savefig.dpi":       200,
    "savefig.facecolor": "white",
    # savefig.bbox deliberately omitted
})


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_gdx(path: str) -> dict:
    log.info("Loading GDX: %s", path)
    data = gdxpds.to_dataframes(path)
    log.info("Available symbols: %s", sorted(data.keys()))
    return data


def extract_h2_storage(data: dict) -> pd.DataFrame:
    """
    Pull storage_level_out, filter to H2_storage tech + Hydrogen_T commodity.

    Columns returned:
        hour   (int 0-8759)
        island (str)
        year   (str)
        h2_gwh (float, GWh)
    """
    sl   = data["storage_level_out"]
    mask = (sl["techs"] == H2_TECH) & (sl["commodity"] == H2_COMMODITY)
    h2   = sl.loc[mask].copy()

    if h2.empty:
        log.warning("No rows for tech='%s', commodity='%s'.", H2_TECH, H2_COMMODITY)
        return pd.DataFrame(columns=["hour", "island", "year", "h2_gwh"])

    h2["hour"] = (
        h2["timeModel"].str.replace("tm", "", regex=False).astype(int) - 1
    )
    h2.rename(columns={
        "accNodesModel": "island",
        "accYears":      "year",
        "Value":         "h2_gwh",
    }, inplace=True)

    h2 = h2.groupby(["hour", "island", "year"], as_index=False)["h2_gwh"].sum()

    log.info("H2 storage rows: %d", len(h2))
    log.info("Islands: %s", sorted(h2["island"].unique()))
    log.info("Years  : %s", sorted(h2["year"].unique()))

    return h2[["hour", "island", "year", "h2_gwh"]]


# ─────────────────────────────────────────────────────────────────────────────
# STORAGE LEVEL MATRIX BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_h2_matrix(h2: pd.DataFrame, island: str, year: str):
    mask = (h2["island"] == island) & (h2["year"] == year)
    sub  = h2.loc[mask, ["hour", "h2_gwh"]].copy()

    if sub.empty:
        return np.full((24, 365), np.nan), 0.0, 0.0, 0.0

    full = pd.Series(0.0, index=range(8760))
    full.update(sub.set_index("hour")["h2_gwh"])

    peak_gwh = float(full.max())
    if peak_gwh == 0:
        return np.full((24, 365), np.nan), 0.0, 0.0, 0.0

    level_pct      = (full / peak_gwh * 100).values
    matrix         = level_pct.reshape(365, 24).T
    avg_level      = float(level_pct.mean())
    hours_above_80 = float((level_pct > 80).mean() * 100)

    return matrix, peak_gwh, avg_level, hours_above_80


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE SUBPLOT RENDERER
# ─────────────────────────────────────────────────────────────────────────────

def draw_h2_heatmap(ax, matrix: np.ndarray, year: str,
                    peak_gwh: float, avg_level: float,
                    hours_above_80: float, show_ylab: bool):

    if np.all(np.isnan(matrix)):
        ax.text(0.5, 0.5, "No H2 storage data\n(not deployed)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=9, color="#888888")
        ax.set_title(year, fontweight="bold")
        return None

    im = ax.imshow(
        matrix,
        aspect="auto",
        origin="upper",
        cmap=H2_CMAP,
        vmin=0, vmax=100,
        interpolation="nearest",
        extent=[1, 365, 23.5, -0.5],
    )

    # ── X-axis: month names, no overlapping day numbers ───────────────────────
    ax.set_xticks(MONTH_MIDPOINTS)
    ax.set_xticklabels(MONTH_LABELS, fontsize=8)
    ax.set_xticks(MONTH_BOUNDARIES, minor=True)
    ax.tick_params(axis="x", which="major", length=0)
    ax.tick_params(axis="x", which="minor", length=3, color="#aaaaaa", width=0.5)
    ax.set_xlim(1, 365)
    ax.set_xlabel("Month", labelpad=4)

    for mb in MONTH_BOUNDARIES[1:-1]:
        ax.axvline(mb, color="white", linewidth=0.4, alpha=0.55)

    # ── Y-axis ────────────────────────────────────────────────────────────────
    ax.set_yticks([0, 6, 12, 18, 23])
    ax.set_yticklabels(["00:00", "06:00", "12:00", "18:00", "23:00"], fontsize=7.5)
    ax.set_ylim(23.5, -0.5)
    if show_ylab:
        ax.set_ylabel("Hour of day", labelpad=4)

    # ── Title + stats ─────────────────────────────────────────────────────────
    ax.set_title(year, fontweight="bold", pad=14)
    ax.text(
        0.5, 1.01,
        (f"Peak: {peak_gwh:.3f} GWh  |  "
         f"Avg level: {avg_level:.1f}%  |  "
         f"Hours >80%: {hours_above_80:.1f}%"),
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=7.2, color="#444444", fontstyle="italic",
    )

    return im


# ─────────────────────────────────────────────────────────────────────────────
# PER-ISLAND FIGURE
# ─────────────────────────────────────────────────────────────────────────────

def plot_island_h2(h2: pd.DataFrame, island: str,
                   years: list, out_dir: Path, fmt: str, dpi: int):

    island_label = island.replace("_model", "")
    n_years      = len(years)

    panel_w = 5.5
    cbar_w  = 0.3
    gap     = 0.55
    fig_w   = panel_w * n_years + cbar_w + gap + 0.7
    fig_h   = 5.2

    fig = plt.figure(figsize=(fig_w, fig_h))

    right_frac = 1.0 - (cbar_w + gap) / fig_w

    gs = fig.add_gridspec(
        1, n_years + 1,
        width_ratios=[panel_w] * n_years + [cbar_w],
        wspace=0.06,
        left=0.07,
        right=right_frac,
        top=0.80,
        bottom=0.14,
    )

    axes    = [fig.add_subplot(gs[0, i]) for i in range(n_years)]
    cbar_ax = fig.add_subplot(gs[0, n_years])

    for ax in axes[1:]:
        ax.sharey(axes[0])

    fig.suptitle(
        f"{island_label} — Hydrogen Storage Level (% of annual peak)",
        fontsize=11, fontweight="bold", y=0.95,
    )

    last_im = None
    for idx, year in enumerate(years):
        matrix, peak_gwh, avg_level, h80 = build_h2_matrix(h2, island, year)
        im = draw_h2_heatmap(
            axes[idx], matrix, year, peak_gwh, avg_level, h80,
            show_ylab=(idx == 0),
        )
        if im is not None:
            last_im = im
        if idx > 0:
            plt.setp(axes[idx].get_yticklabels(), visible=False)

    # ── Colorbar anchored to its own dedicated axis ───────────────────────────
    if last_im is not None:
        cbar = fig.colorbar(last_im, cax=cbar_ax)
        cbar.set_label("Storage level (%)", fontsize=9, labelpad=8)
        cbar.set_ticks([0, 20, 40, 60, 80, 100])
        cbar.ax.tick_params(labelsize=8)
        for t in [20, 40, 60, 80]:
            cbar.ax.axhline(t, color="white", linewidth=0.5, alpha=0.6)
    else:
        cbar_ax.set_visible(False)

    save_path = out_dir / f"{island}.{fmt}"
    fig.savefig(save_path, dpi=dpi)      # no bbox_inches="tight"
    plt.close(fig)
    log.info("  Saved -> %s", save_path)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(gdx_path: str, out_dir: str, fmt: str, dpi: int,
         islands: list, years: list, preloaded_data=None):

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # if preloaded_data is not None:
    #     data = preloaded_data
    # else:
    #     data = load_data(gdx_path)
    h2   = extract_h2_storage(data)

    if h2.empty:
        log.error("No H2 storage data. Check tech='%s', commodity='%s'.",
                  H2_TECH, H2_COMMODITY)
        return

    for island in islands:
        log.info("Plotting %s ...", island)
        plot_island_h2(h2, island, years, out_path, fmt, dpi)

    log.info("Done. Figures saved to: %s", out_path)
    print(f"\nAll {len(islands)} island H2 storage figures written to: {out_path}/")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--gdx",       default=GDX_PATH)
    parser.add_argument("--out",       default=OUTPUT_DIR)
    parser.add_argument("--fmt",       default="png",
                        choices=["png", "pdf", "svg", "tiff"])
    parser.add_argument("--dpi",       default=200, type=int)
    parser.add_argument("--islands",   nargs="+", default=ISLANDS)
    parser.add_argument("--years",     nargs="+", default=YEARS)
    parser.add_argument("--tech",      default=H2_TECH)
    parser.add_argument("--commodity", default=H2_COMMODITY)
    args = parser.parse_args()

    H2_TECH      = args.tech
    H2_COMMODITY = args.commodity

    main(args.gdx, args.out, args.fmt, args.dpi, args.islands, args.years)

##################################Sustainable fuel storage SOC##########################################
##################################Sustainable fuel storage SOC##########################################
"""
Pacific Island Countries (PICs) — Sustainable-Fuel Storage Level Heatmaps
=========================================================================
Mirrors the Hydrogen storage heatmaps, applied to the three synthetic-fuel
storages defined in TECHS:

    Ammonia    (tech: Ammonia_storage,   commodity: Ammonia_T)
    Methanol   (tech: Methanol_storage,  commodity: Methanol_T)
    eKerosene  (tech: eKerosene_storage, commodity: eKerosene_T)

    Storage level (%) = value / max(value for that island-year) x 100
        0 %   -> store empty
        100 % -> store at annual peak

Layout (identical to the H2 figures)
------------------------------------
One figure per island per fuel -> 1 x 3 row of subplots (2030 | 2040 | 2050).
    x-axis : month labels (Jan-Dec)
    y-axis : hour of day (0-23, midnight at top)
    colour : storage level (%)

Source
------
    Symbol : storage_level_out
    Tech / commodity per fuel (see FUEL_STORAGES below)

Output
------
  figures/S_23/ammonia_storage_minload/   CI_model.png ...
  figures/S_23/methanol_storage_minload/  ...
  figures/S_23/ekerosene_storage_minload/ ...
"""

import argparse
import logging
from pathlib import Path

import gdxpds
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────
GDX_PATH = "../GDX_results/IP_2050_Final_S1_minload.gdx"
YEARS    = ["2030", "2040", "2050"]
ISLANDS  = [
    "CI_model", "FJ_model", "FSM_model", "KB_model",  "MI_model",
    "NU_model", "NE_model", "PU_model",  "PNG_model", "SA_model",
    "SI_model", "TA_model", "TU_model",  "VU_model",
]

# ── Per-fuel configuration ────────────────────────────────────────────────────
# Each fuel gets its own tech, commodity, output folder and (distinct) colormap.
# `unit` is only used for the "Peak: … " annotation; storage_level_out is assumed
# to be in GWh (energy), matching the H2 figures — change if your model differs.
FUEL_STORAGES = [
    {
        "name":      "Ammonia",
        "tech":      "Ammonia_storage",
        "commodity": "Ammonia_T",
        "out_dir":   "figures/S_1/ammonia_storage_minload",
        "unit":      "GWh",
        "cmap":      LinearSegmentedColormap.from_list(
                         "ammonia_storage",
                         ["#FFFFE5", "#D9F0A3", "#78C679", "#238443", "#004529"],
                         N=256),
    },
    {
        "name":      "Methanol",
        "tech":      "Methanol_storage",
        "commodity": "Methanol_T",
        "out_dir":   "figures/S_1/methanol_storage_minload",
        "unit":      "GWh",
        "cmap":      LinearSegmentedColormap.from_list(
                         "methanol_storage",
                         ["#FFF7EC", "#FDD49E", "#FC8D59", "#D7301F", "#7F0000"],
                         N=256),
    },
    {
        "name":      "eKerosene",
        "tech":      "eKerosene_storage",
        "commodity": "eKerosene_T",
        "out_dir":   "figures/S_1/ekerosene_storage_minload",
        "unit":      "GWh",
        "cmap":      LinearSegmentedColormap.from_list(
                         "ekerosene_storage",
                         ["#F7FCFD", "#BFD3E6", "#8C96C6", "#88419D", "#4D004B"],
                         N=256),
    },
]

# ── Month axis ────────────────────────────────────────────────────────────────
MONTH_BOUNDARIES = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]
MONTH_LABELS     = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTH_MIDPOINTS  = [(MONTH_BOUNDARIES[i] + MONTH_BOUNDARIES[i + 1]) / 2
                    for i in range(12)]

# ─────────────────────────────────────────────────────────────────────────────
# PLOT STYLE  (savefig.bbox intentionally NOT "tight" — margins via gridspec)
# ─────────────────────────────────────────────────────────────────────────────
rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":         9,
    "axes.titlesize":    10,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "axes.linewidth":    0.7,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        120,
    "savefig.dpi":       200,
    "savefig.facecolor": "white",
})


# ─────────────────────────────────────────────────────────────────────────────
# DATA EXTRACTION  (one fuel at a time)
# ─────────────────────────────────────────────────────────────────────────────
def extract_fuel_storage(data: dict, tech: str, commodity: str) -> pd.DataFrame:
    """storage_level_out filtered to one storage tech + commodity.

    Returns columns: hour (0-8759), island, year, level_gwh.
    """
    sl   = data["storage_level_out"]
    mask = (sl["techs"] == tech) & (sl["commodity"] == commodity)
    df   = sl.loc[mask].copy()

    if df.empty:
        log.warning("No rows for tech='%s', commodity='%s'.", tech, commodity)
        return pd.DataFrame(columns=["hour", "island", "year", "level_gwh"])

    df["hour"] = (
        df["timeModel"].str.replace("tm", "", regex=False).astype(int) - 1
    )
    df.rename(columns={
        "accNodesModel": "island",
        "accYears":      "year",
        "Value":         "level_gwh",
    }, inplace=True)

    df = df.groupby(["hour", "island", "year"], as_index=False)["level_gwh"].sum()

    log.info("[%s] rows: %d | islands: %s | years: %s",
             tech, len(df), sorted(df["island"].unique()),
             sorted(df["year"].unique()))
    return df[["hour", "island", "year", "level_gwh"]]


# ─────────────────────────────────────────────────────────────────────────────
# STORAGE-LEVEL MATRIX BUILDER
# ─────────────────────────────────────────────────────────────────────────────
def build_fuel_matrix(df: pd.DataFrame, island: str, year: str):
    mask = (df["island"] == island) & (df["year"] == year)
    sub  = df.loc[mask, ["hour", "level_gwh"]].copy()

    if sub.empty:
        return np.full((24, 365), np.nan), 0.0, 0.0, 0.0

    full = pd.Series(0.0, index=range(8760))
    full.update(sub.set_index("hour")["level_gwh"])

    peak_gwh = float(full.max())
    if peak_gwh == 0:
        return np.full((24, 365), np.nan), 0.0, 0.0, 0.0

    level_pct      = (full / peak_gwh * 100).values
    matrix         = level_pct.reshape(365, 24).T
    avg_level      = float(level_pct.mean())
    hours_above_80 = float((level_pct > 80).mean() * 100)

    return matrix, peak_gwh, avg_level, hours_above_80


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE SUBPLOT RENDERER
# ─────────────────────────────────────────────────────────────────────────────
def draw_fuel_heatmap(ax, matrix, year, peak_gwh, avg_level, hours_above_80,
                      show_ylab, cmap, fuel_name, unit):

    if np.all(np.isnan(matrix)):
        ax.text(0.5, 0.5, f"No {fuel_name} storage data\n(not deployed)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=9, color="#888888")
        ax.set_title(year, fontweight="bold")
        return None

    im = ax.imshow(
        matrix, aspect="auto", origin="upper",
        cmap=cmap, vmin=0, vmax=100,
        interpolation="nearest", extent=[1, 365, 23.5, -0.5],
    )

    # ── X-axis: month names ───────────────────────────────────────────────────
    ax.set_xticks(MONTH_MIDPOINTS)
    ax.set_xticklabels(MONTH_LABELS, fontsize=8)
    ax.set_xticks(MONTH_BOUNDARIES, minor=True)
    ax.tick_params(axis="x", which="major", length=0)
    ax.tick_params(axis="x", which="minor", length=3, color="#aaaaaa", width=0.5)
    ax.set_xlim(1, 365)
    ax.set_xlabel("Month", labelpad=4)
    for mb in MONTH_BOUNDARIES[1:-1]:
        ax.axvline(mb, color="white", linewidth=0.4, alpha=0.55)

    # ── Y-axis ────────────────────────────────────────────────────────────────
    ax.set_yticks([0, 6, 12, 18, 23])
    ax.set_yticklabels(["00:00", "06:00", "12:00", "18:00", "23:00"], fontsize=7.5)
    ax.set_ylim(23.5, -0.5)
    if show_ylab:
        ax.set_ylabel("Hour of day", labelpad=4)

    # ── Title + stats ─────────────────────────────────────────────────────────
    ax.set_title(year, fontweight="bold", pad=14)
    ax.text(
        0.5, 1.01,
        (f"Peak: {peak_gwh:.3f} {unit}  |  "
         f"Avg level: {avg_level:.1f}%  |  "
         f"Hours >80%: {hours_above_80:.1f}%"),
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=7.2, color="#444444", fontstyle="italic",
    )
    return im


# ─────────────────────────────────────────────────────────────────────────────
# PER-ISLAND FIGURE
# ─────────────────────────────────────────────────────────────────────────────
def plot_island_fuel(df, island, years, out_dir, fmt, dpi, cmap, fuel_name, unit):
    island_label = island.replace("_model", "")
    n_years      = len(years)

    panel_w = 5.5
    cbar_w  = 0.3
    gap     = 0.55
    fig_w   = panel_w * n_years + cbar_w + gap + 0.7
    fig_h   = 5.2

    fig = plt.figure(figsize=(fig_w, fig_h))
    right_frac = 1.0 - (cbar_w + gap) / fig_w

    gs = fig.add_gridspec(
        1, n_years + 1,
        width_ratios=[panel_w] * n_years + [cbar_w],
        wspace=0.06, left=0.07, right=right_frac, top=0.80, bottom=0.14,
    )

    axes    = [fig.add_subplot(gs[0, i]) for i in range(n_years)]
    cbar_ax = fig.add_subplot(gs[0, n_years])
    for ax in axes[1:]:
        ax.sharey(axes[0])

    fig.suptitle(
        f"{island_label} — {fuel_name} Storage Level (% of annual peak)",
        fontsize=11, fontweight="bold", y=0.95,
    )

    last_im = None
    for idx, year in enumerate(years):
        matrix, peak_gwh, avg_level, h80 = build_fuel_matrix(df, island, year)
        im = draw_fuel_heatmap(
            axes[idx], matrix, year, peak_gwh, avg_level, h80,
            show_ylab=(idx == 0), cmap=cmap, fuel_name=fuel_name, unit=unit,
        )
        if im is not None:
            last_im = im
        if idx > 0:
            plt.setp(axes[idx].get_yticklabels(), visible=False)

    if last_im is not None:
        cbar = fig.colorbar(last_im, cax=cbar_ax)
        cbar.set_label("Storage level (%)", fontsize=9, labelpad=8)
        cbar.set_ticks([0, 20, 40, 60, 80, 100])
        cbar.ax.tick_params(labelsize=8)
        for t in [20, 40, 60, 80]:
            cbar.ax.axhline(t, color="white", linewidth=0.5, alpha=0.6)
    else:
        cbar_ax.set_visible(False)

    save_path = out_dir / f"{island}.{fmt}"
    fig.savefig(save_path, dpi=dpi)      # no bbox_inches="tight"
    plt.close(fig)
    log.info("  Saved -> %s", save_path)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — loops over the three fuels
# ─────────────────────────────────────────────────────────────────────────────
def main(gdx_path: str, fmt: str, dpi: int,
         islands: list, years: list, preloaded_data=None):

    # Uses the shared global `data` loaded once at the top of Gdx_plot.py
    # (same pattern as the H2 storage block).
    # if preloaded_data is not None:
    #     data = preloaded_data
    # else:
    #     data = load_data(gdx_path)

    for fuel in FUEL_STORAGES:
        log.info("=== %s storage (tech=%s, commodity=%s) ===",
                 fuel["name"], fuel["tech"], fuel["commodity"])

        df = extract_fuel_storage(data, fuel["tech"], fuel["commodity"])
        if df.empty:
            log.error("No %s storage data — skipping. "
                      "Check tech='%s', commodity='%s'.",
                      fuel["name"], fuel["tech"], fuel["commodity"])
            continue

        out_path = Path(fuel["out_dir"])
        out_path.mkdir(parents=True, exist_ok=True)

        for island in islands:
            log.info("Plotting %s %s ...", island, fuel["name"])
            plot_island_fuel(df, island, years, out_path, fmt, dpi,
                             fuel["cmap"], fuel["name"], fuel["unit"])

        print(f"All {len(islands)} island {fuel['name']} storage figures "
              f"written to: {out_path}/")

    log.info("Done — all sustainable-fuel storage figures generated.")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--gdx",     default=GDX_PATH)
    parser.add_argument("--fmt",     default="png",
                        choices=["png", "pdf", "svg", "tiff"])
    parser.add_argument("--dpi",     default=200, type=int)
    parser.add_argument("--islands", nargs="+", default=ISLANDS)
    parser.add_argument("--years",   nargs="+", default=YEARS)
    args = parser.parse_args()

    main(args.gdx, args.fmt, args.dpi, args.islands, args.years)
    ######################################Efuel synthesis activity##########################
######################e-Fuel synthesis activity####################################
"""
Pacific Island Countries (PICs) — e-Fuel Synthesis Activity Heatmaps
====================================================================
24 x 365 heatmaps of synthetic-fuel synthesis activity, expressed as a
percentage of the annual peak hourly production.  Flows are read from
'commodity_balance'.

    Activity (%) = flow(hour) / max(flow over the year) x 100
        0 %   -> converter off
        100 % -> converter at peak throughput

Converters (see CONVERTERS below) — output flow, Value > 0
----------------------------------------------------------
    Ammonia_synthesis  -> Ammonia produced
    Methanol_synthesis -> Methanol produced
    FTL                -> eKerosene produced

NOTE: the synthesis commodities use the bare name (Ammonia / Methanol /
eKerosene) — no "_T" suffix.

Layout
------
One figure per island per converter -> 1 x 2 row (2040 | 2050).
    x-axis : month labels (Jan-Dec)
    y-axis : hour of day (0-23, 00:00 at top)
    colour : activity (%)

Output
------
  figures/S_1/ammonia_synthesis_activity_minload/   CI_model.png ...
  figures/S_1/methanol_synthesis_activity_minload/  ...
  figures/S_1/ekerosene_synthesis_activity_minload/ ...
"""

import argparse
import logging
from pathlib import Path

import gdxpds
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────
GDX_PATH = "../GDX_results/IP_2050_Final_S1_minload.gdx"
YEARS    = ["2040", "2050"]
ISLANDS  = [
    "CI_model", "FJ_model", "FSM_model", "KB_model",  "MI_model",
    "NU_model", "NE_model", "PU_model",  "PNG_model", "SA_model",
    "SI_model", "TA_model", "TU_model",  "VU_model",
]

# ── Colour maps ───────────────────────────────────────────────────────────────
# Distinct hues per fuel (matching the fuel-storage figures) so they are easy
# to tell apart.
CMAP_AMMONIA = LinearSegmentedColormap.from_list(
    "ammonia_act", ["#FFFFE5", "#D9F0A3", "#78C679", "#238443", "#004529"], N=256)
CMAP_METHANOL = LinearSegmentedColormap.from_list(
    "methanol_act", ["#FFF7EC", "#FDD49E", "#FC8D59", "#D7301F", "#7F0000"], N=256)
CMAP_EKEROSENE = LinearSegmentedColormap.from_list(
    "ekerosene_act", ["#F7FCFD", "#BFD3E6", "#8C96C6", "#88419D", "#4D004B"], N=256)

# ── Per-converter configuration ───────────────────────────────────────────────
# flow: "in"  -> electricity/feedstock consumed (filter Value < 0, take abs)
#       "out" -> product generated              (filter Value > 0)
CONVERTERS = [
    {
        "name":       "Ammonia synthesis",
        "tech":       "Ammonia_synthesis",
        "commodity":  "Ammonia",
        "flow":       "out",
        "out_dir":    "figures/S_1/ammonia_synthesis_activity_minload",
        "cmap":       CMAP_AMMONIA,
        "cbar_label": "Ammonia synthesis activity (%)",
    },
    {
        "name":       "Methanol synthesis",
        "tech":       "Methanol_synthesis",
        "commodity":  "Methanol",
        "flow":       "out",
        "out_dir":    "figures/S_1/methanol_synthesis_activity_minload",
        "cmap":       CMAP_METHANOL,
        "cbar_label": "Methanol synthesis activity (%)",
    },
    {
        "name":       "eKerosene synthesis (FTL)",
        "tech":       "FTL",
        "commodity":  "eKerosene",
        "flow":       "out",
        "out_dir":    "figures/S_1/ekerosene_synthesis_activity_minload",
        "cmap":       CMAP_EKEROSENE,
        "cbar_label": "eKerosene synthesis activity (%)",
    },
]

# ── Month axis ────────────────────────────────────────────────────────────────
MONTH_BOUNDARIES = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]
MONTH_LABELS     = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]
MONTH_MIDPOINTS  = [(MONTH_BOUNDARIES[i] + MONTH_BOUNDARIES[i + 1]) / 2
                    for i in range(12)]

# ─────────────────────────────────────────────────────────────────────────────
# PLOT STYLE  (savefig.bbox intentionally NOT "tight")
# ─────────────────────────────────────────────────────────────────────────────
rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":         9,
    "axes.titlesize":    10,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "axes.linewidth":    0.7,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        120,
    "savefig.dpi":       200,
    "savefig.facecolor": "white",
})


# ─────────────────────────────────────────────────────────────────────────────
# DATA EXTRACTION  (one converter at a time)
# ─────────────────────────────────────────────────────────────────────────────
def extract_converter_flow(data: dict, tech: str, commodity: str,
                           flow: str) -> pd.DataFrame:
    """commodity_balance filtered to one tech + commodity + flow direction.

    flow="in"  -> consumption (Value < 0), returned as +ve magnitude
    flow="out" -> production  (Value > 0)

    Returns columns: hour (0-8759), island, year, value_gwh (>= 0).
    """
    cb = data["commodity_balance"]

    if flow == "in":
        mask = ((cb["techs"] == tech) & (cb["commodity"] == commodity) &
                (cb["Value"] < 0))
    else:  # "out"
        mask = ((cb["techs"] == tech) & (cb["commodity"] == commodity) &
                (cb["Value"] > 0))

    df = cb.loc[mask].copy()

    if df.empty:
        log.warning("No rows for tech='%s', commodity='%s', flow='%s'.",
                    tech, commodity, flow)
        return pd.DataFrame(columns=["hour", "island", "year", "value_gwh"])

    df["hour"] = (
        df["timeModel"].str.replace("tm", "", regex=False).astype(int) - 1
    )
    df.rename(columns={
        "accNodesModel": "island",
        "accYears":      "year",
        "Value":         "value_gwh",
    }, inplace=True)

    df["value_gwh"] = df["value_gwh"].abs()
    df = df.groupby(["hour", "island", "year"], as_index=False)["value_gwh"].sum()

    log.info("[%s] rows: %d | islands: %s | years: %s",
             tech, len(df), sorted(df["island"].unique()),
             sorted(df["year"].unique()))
    return df[["hour", "island", "year", "value_gwh"]]


# ─────────────────────────────────────────────────────────────────────────────
# ACTIVITY MATRIX BUILDER
# ─────────────────────────────────────────────────────────────────────────────
def build_activity_matrix(df: pd.DataFrame, island: str, year: str):
    mask = (df["island"] == island) & (df["year"] == year)
    sub  = df.loc[mask, ["hour", "value_gwh"]].copy()

    if sub.empty:
        return np.full((24, 365), np.nan), 0.0, 0.0, 0.0, 0.0

    full = pd.Series(0.0, index=range(8760))
    full.update(sub.set_index("hour")["value_gwh"])

    peak_gwh  = float(full.max())
    total_gwh = float(full.sum())
    if peak_gwh == 0:
        return np.full((24, 365), np.nan), 0.0, 0.0, 0.0, 0.0

    activity_pct = (full / peak_gwh * 100).values
    matrix       = activity_pct.reshape(365, 24).T
    avg_activity = float(activity_pct.mean())
    hours_active = float((activity_pct > 0).mean() * 100)

    return matrix, peak_gwh, avg_activity, hours_active, total_gwh


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE SUBPLOT RENDERER
# ─────────────────────────────────────────────────────────────────────────────
def draw_activity_heatmap(ax, matrix, year, peak_gwh, avg_activity,
                          hours_active, total_gwh, show_ylab, cmap, name):

    if np.all(np.isnan(matrix)):
        ax.text(0.5, 0.5, f"No {name} data\n(not deployed)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=9, color="#888888")
        ax.set_title(year, fontweight="bold")
        return None

    im = ax.imshow(
        matrix, aspect="auto", origin="upper",
        cmap=cmap, vmin=0, vmax=100,
        interpolation="nearest", extent=[1, 365, 23.5, -0.5],
    )

    # ── X-axis: month names ───────────────────────────────────────────────────
    ax.set_xticks(MONTH_MIDPOINTS)
    ax.set_xticklabels(MONTH_LABELS, fontsize=8)
    ax.set_xticks(MONTH_BOUNDARIES, minor=True)
    ax.tick_params(axis="x", which="major", length=0)
    ax.tick_params(axis="x", which="minor", length=3, color="#aaaaaa", width=0.5)
    ax.set_xlim(1, 365)
    ax.set_xlabel("Month", labelpad=4)
    for mb in MONTH_BOUNDARIES[1:-1]:
        ax.axvline(mb, color="white", linewidth=0.4, alpha=0.55)

    # ── Y-axis ────────────────────────────────────────────────────────────────
    ax.set_yticks([0, 6, 12, 18, 23])
    ax.set_yticklabels(["00:00", "06:00", "12:00", "18:00", "23:00"], fontsize=7.5)
    ax.set_ylim(23.5, -0.5)
    if show_ylab:
        ax.set_ylabel("Hour of day", labelpad=4)

    # ── Title + stats ─────────────────────────────────────────────────────────
    ax.set_title(year, fontweight="bold", pad=14)
    ax.text(
        0.5, 1.01,
        (f"Peak: {peak_gwh:.4f} GWh/h  |  "
         f"Total: {total_gwh:.2f} GWh/yr  |  "
         f"Avg activity: {avg_activity:.1f}%  |  "
         f"Hours active: {hours_active:.1f}%"),
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=7.2, color="#444444", fontstyle="italic",
    )
    return im


# ─────────────────────────────────────────────────────────────────────────────
# PER-ISLAND FIGURE
# ─────────────────────────────────────────────────────────────────────────────
def plot_island_converter(df, island, years, out_dir, fmt, dpi,
                          cmap, name, cbar_label, flow):
    island_label = island.replace("_model", "")
    n_years      = len(years)

    panel_w = 5.5
    cbar_w  = 0.3
    gap     = 0.55
    fig_w   = panel_w * n_years + cbar_w + gap + 0.7
    fig_h   = 5.2

    fig = plt.figure(figsize=(fig_w, fig_h))
    right_frac = 1.0 - (cbar_w + gap) / fig_w

    gs = fig.add_gridspec(
        1, n_years + 1,
        width_ratios=[panel_w] * n_years + [cbar_w],
        wspace=0.06, left=0.07, right=right_frac, top=0.80, bottom=0.14,
    )

    axes    = [fig.add_subplot(gs[0, i]) for i in range(n_years)]
    cbar_ax = fig.add_subplot(gs[0, n_years])
    for ax in axes[1:]:
        ax.sharey(axes[0])

    flow_word = "consumption" if flow == "in" else "production"
    fig.suptitle(
        f"{island_label} — {name} Activity  "
        f"(% of annual peak hourly {flow_word})",
        fontsize=11, fontweight="bold", y=0.95,
    )

    last_im = None
    for idx, year in enumerate(years):
        matrix, peak_gwh, avg_act, h_active, total_gwh = \
            build_activity_matrix(df, island, year)
        im = draw_activity_heatmap(
            axes[idx], matrix, year, peak_gwh, avg_act, h_active, total_gwh,
            show_ylab=(idx == 0), cmap=cmap, name=name,
        )
        if im is not None:
            last_im = im
        if idx > 0:
            plt.setp(axes[idx].get_yticklabels(), visible=False)

    if last_im is not None:
        cbar = fig.colorbar(last_im, cax=cbar_ax)
        cbar.set_label(cbar_label, fontsize=9, labelpad=8)
        cbar.set_ticks([0, 20, 40, 60, 80, 100])
        cbar.ax.tick_params(labelsize=8)
        for t in [20, 40, 60, 80]:
            cbar.ax.axhline(t, color="white", linewidth=0.5, alpha=0.6)
    else:
        cbar_ax.set_visible(False)

    save_path = out_dir / f"{island}.{fmt}"
    fig.savefig(save_path, dpi=dpi)      # no bbox_inches="tight"
    plt.close(fig)
    log.info("  Saved -> %s", save_path)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — loops over the four converters
# ─────────────────────────────────────────────────────────────────────────────
def main(gdx_path: str, fmt: str, dpi: int,
         islands: list, years: list, preloaded_data=None):

    # Uses the shared global `data` loaded once at the top of Gdx_plot.py
    # (same pattern as the original activity block).
    # if preloaded_data is not None:
    #     data = preloaded_data
    # else:
    #     data = load_data(gdx_path)

    for conv in CONVERTERS:
        log.info("=== %s (tech=%s, commodity=%s, flow=%s) ===",
                 conv["name"], conv["tech"], conv["commodity"], conv["flow"])

        df = extract_converter_flow(
            data, conv["tech"], conv["commodity"], conv["flow"])
        if df.empty:
            log.error("No %s data — skipping. Check tech='%s', commodity='%s'.",
                      conv["name"], conv["tech"], conv["commodity"])
            continue

        out_path = Path(conv["out_dir"])
        out_path.mkdir(parents=True, exist_ok=True)

        for island in islands:
            log.info("Plotting %s %s ...", island, conv["name"])
            plot_island_converter(df, island, years, out_path, fmt, dpi,
                                   conv["cmap"], conv["name"],
                                   conv["cbar_label"], conv["flow"])

        print(f"All {len(islands)} island {conv['name']} figures "
              f"written to: {out_path}/")

    log.info("Done — all converter activity figures generated.")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--gdx",     default=GDX_PATH)
    parser.add_argument("--fmt",     default="png",
                        choices=["png", "pdf", "svg", "tiff"])
    parser.add_argument("--dpi",     default=200, type=int)
    parser.add_argument("--islands", nargs="+", default=ISLANDS)
    parser.add_argument("--years",   nargs="+", default=YEARS)
    args = parser.parse_args()

    main(args.gdx, args.fmt, args.dpi, args.islands, args.years)
    
    ######################AEL activity####################################################
"""
Pacific Island Countries (PICs) — Electrolyzer (AEL) Activity Heatmaps
=======================================================================
Reads hourly electricity consumption of the AEL (alkaline electrolyser)
from 'commodity_balance' in the GAMS GDX file and produces 24 × 365
heatmaps of electrolyser activity expressed as a percentage of the
annual peak consumption.

    Activity (%) = |AEL Elec consumption (hour)| /
                   max(|AEL Elec consumption|) over the year  × 100

    0 %   → electrolyser off
    100 % → electrolyser running at peak rated load

Layout
------
One figure per island  →  1 × 2 row of subplots (2040 | 2050).
Each subplot is a heatmap where:
    x-axis  : month labels (Jan-Dec)
    y-axis  : hour of day (0-23,  00:00 at top)
    colour  : activity (%)

Output
------
  figures/electrolyzer_activity/
      CI_model.png
      FJ_model.png
      ...  (one PNG per island)

Usage
-----
    python pic_electrolyzer_activity.py
    python pic_electrolyzer_activity.py --gdx path/to/results.gdx
                                        --out figures/electrolyzer_activity
                                        --fmt png --dpi 200
                                        --islands CI_model FJ_model
                                        --years 2040 2050
"""

import argparse
import logging
from pathlib import Path

import gdxpds
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────
GDX_PATH   = "../GDX_results/IP_2050_Final_S23_minload.gdx"
OUTPUT_DIR = "figures/S_23/electrolyzer_activity_minload"
YEARS      = ["2040", "2050"]
ISLANDS    = [
    "CI_model", "FJ_model", "FSM_model", "KB_model",  "MI_model",
    "NU_model", "NE_model", "PU_model",  "PNG_model", "SA_model",
    "SI_model", "TA_model", "TU_model",  "VU_model",
]

AEL_TECH      = "AEL"
AEL_COMMODITY = "Elec"

# ── Month axis ────────────────────────────────────────────────────────────────
MONTH_BOUNDARIES = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]
MONTH_LABELS     = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]
MONTH_MIDPOINTS  = [(MONTH_BOUNDARIES[i] + MONTH_BOUNDARIES[i + 1]) / 2
                    for i in range(12)]

# ─────────────────────────────────────────────────────────────────────────────
# COLOUR MAP
# YlGnBu-inspired (matching battery SOC design):
# pale yellow (idle) → mint → teal → deep ocean blue (full load)
# Perceptually uniform, colorblind-safe, prints well in greyscale too.
# ─────────────────────────────────────────────────────────────────────────────
AEL_CMAP = LinearSegmentedColormap.from_list(
    "soc",
    ["#FFFFCC", "#C7E9B4", "#41B6C4", "#1D91C0", "#0C2C84"],
    N=256,
)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT STYLE
# IMPORTANT: savefig.bbox is intentionally NOT "tight".
# "tight" re-expands the canvas after layout and drags the colorbar axis
# back over the rightmost heatmap panel.  All margins are controlled via
# gridspec left/right/top/bottom fractions instead.
# ─────────────────────────────────────────────────────────────────────────────
rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":         9,
    "axes.titlesize":    10,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "axes.linewidth":    0.7,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        120,
    "savefig.dpi":       200,
    "savefig.facecolor": "white",
    # savefig.bbox deliberately omitted (defaults to "standard")
})


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_gdx(path: str) -> dict:
    log.info("Loading GDX: %s", path)
    data = gdxpds.to_dataframes(path)
    log.info("Available symbols: %s", sorted(data.keys()))
    return data


def extract_ael_consumption(data: dict) -> pd.DataFrame:
    """
    Pull commodity_balance, filter to AEL tech + Elec commodity.
    Take absolute value of the negative consumption rows.

    Returns DataFrame with columns:
        hour      (int 0-8759)
        island    (str)
        year      (str)
        elec_gwh  (float, GWh, always >= 0)
    """
    cb   = data["commodity_balance"]
    mask = (
        (cb["techs"]     == AEL_TECH) &
        (cb["commodity"] == AEL_COMMODITY) &
        (cb["Value"]     <  0)
    )
    ael = cb.loc[mask].copy()

    if ael.empty:
        log.warning("No rows for tech='%s', commodity='%s', Value<0.",
                    AEL_TECH, AEL_COMMODITY)
        return pd.DataFrame(columns=["hour", "island", "year", "elec_gwh"])

    ael["hour"] = (
        ael["timeModel"].str.replace("tm", "", regex=False).astype(int) - 1
    )
    ael.rename(columns={
        "accNodesModel": "island",
        "accYears":      "year",
        "Value":         "elec_gwh",
    }, inplace=True)

    ael["elec_gwh"] = ael["elec_gwh"].abs()
    ael = ael.groupby(["hour", "island", "year"], as_index=False)["elec_gwh"].sum()

    log.info("AEL rows: %d", len(ael))
    log.info("Islands: %s", sorted(ael["island"].unique()))
    log.info("Years  : %s", sorted(ael["year"].unique()))

    return ael[["hour", "island", "year", "elec_gwh"]]


# ─────────────────────────────────────────────────────────────────────────────
# ACTIVITY MATRIX BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_activity_matrix(ael: pd.DataFrame, island: str, year: str):
    mask = (ael["island"] == island) & (ael["year"] == year)
    sub  = ael.loc[mask, ["hour", "elec_gwh"]].copy()

    if sub.empty:
        return np.full((24, 365), np.nan), 0.0, 0.0, 0.0, 0.0

    full = pd.Series(0.0, index=range(8760))
    full.update(sub.set_index("hour")["elec_gwh"])

    peak_gwh  = float(full.max())
    total_gwh = float(full.sum())

    if peak_gwh == 0:
        return np.full((24, 365), np.nan), 0.0, 0.0, 0.0, 0.0

    activity_pct = (full / peak_gwh * 100).values
    matrix       = activity_pct.reshape(365, 24).T

    avg_activity = float(activity_pct.mean())
    hours_active = float((activity_pct > 0).mean() * 100)

    return matrix, peak_gwh, avg_activity, hours_active, total_gwh


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE SUBPLOT RENDERER
# ─────────────────────────────────────────────────────────────────────────────

def draw_activity_heatmap(ax, matrix: np.ndarray, year: str,
                          peak_gwh: float, avg_activity: float,
                          hours_active: float, total_gwh: float,
                          show_ylab: bool):

    if np.all(np.isnan(matrix)):
        ax.text(0.5, 0.5, "No AEL data\n(electrolyser not deployed)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=9, color="#888888")
        ax.set_title(year, fontweight="bold")
        return None

    im = ax.imshow(
        matrix,
        aspect="auto",
        origin="upper",
        cmap=AEL_CMAP,
        vmin=0, vmax=100,
        interpolation="nearest",
        extent=[1, 365, 23.5, -0.5],
    )

    # ── X-axis: month names, no overlapping day numbers ───────────────────────
    ax.set_xticks(MONTH_MIDPOINTS)
    ax.set_xticklabels(MONTH_LABELS, fontsize=8)
    ax.set_xticks(MONTH_BOUNDARIES, minor=True)
    ax.tick_params(axis="x", which="major", length=0)          # labels only
    ax.tick_params(axis="x", which="minor", length=3,
                   color="#aaaaaa", width=0.5)
    ax.set_xlim(1, 365)
    ax.set_xlabel("Month", labelpad=4)

    for mb in MONTH_BOUNDARIES[1:-1]:
        ax.axvline(mb, color="white", linewidth=0.4, alpha=0.55)

    # ── Y-axis ────────────────────────────────────────────────────────────────
    ax.set_yticks([0, 6, 12, 18, 23])
    ax.set_yticklabels(["00:00", "06:00", "12:00", "18:00", "23:00"],
                       fontsize=7.5)
    ax.set_ylim(23.5, -0.5)
    if show_ylab:
        ax.set_ylabel("Hour of day", labelpad=4)

    # ── Title + stats ─────────────────────────────────────────────────────────
    ax.set_title(year, fontweight="bold", pad=14)
    ax.text(
        0.5, 1.01,
        (f"Peak: {peak_gwh:.4f} GWh/h  |  "
         f"Total: {total_gwh:.2f} GWh/yr  |  "
         f"Avg activity: {avg_activity:.1f}%  |  "
         f"Hours active: {hours_active:.1f}%"),
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=7.2, color="#444444", fontstyle="italic",
    )

    return im


# ─────────────────────────────────────────────────────────────────────────────
# PER-ISLAND FIGURE
# ─────────────────────────────────────────────────────────────────────────────

def plot_island_ael(ael: pd.DataFrame, island: str,
                    years: list, out_dir: Path, fmt: str, dpi: int):

    island_label = island.replace("_model", "")
    n_years      = len(years)

    panel_w = 5.5                              # inches per heatmap panel
    cbar_w  = 0.3                              # inches for colorbar column
    gap     = 0.55                             # right-side whitespace (inches)
    fig_w   = panel_w * n_years + cbar_w + gap + 0.7   # 0.7 for left margin
    fig_h   = 5.2

    fig = plt.figure(figsize=(fig_w, fig_h))

    # right edge fraction: leaves exactly (cbar_w + gap) inches for cbar+space
    right_frac = 1.0 - (cbar_w + gap) / fig_w

    gs = fig.add_gridspec(
        1, n_years + 1,
        width_ratios=[panel_w] * n_years + [cbar_w],
        wspace=0.06,
        left=0.07,
        right=right_frac,
        top=0.80,
        bottom=0.14,
    )

    axes    = [fig.add_subplot(gs[0, i]) for i in range(n_years)]
    cbar_ax = fig.add_subplot(gs[0, n_years])

    for ax in axes[1:]:
        ax.sharey(axes[0])

    fig.suptitle(
        f"{island_label} — Electrolyser (AEL) Activity  "
        f"(% of annual peak hourly consumption)",
        fontsize=11, fontweight="bold", y=0.95,
    )

    last_im = None
    for idx, year in enumerate(years):
        matrix, peak_gwh, avg_act, h_active, total_gwh = \
            build_activity_matrix(ael, island, year)
        im = draw_activity_heatmap(
            axes[idx], matrix, year, peak_gwh, avg_act, h_active, total_gwh,
            show_ylab=(idx == 0),
        )
        if im is not None:
            last_im = im
        if idx > 0:
            plt.setp(axes[idx].get_yticklabels(), visible=False)

    # ── Colorbar anchored to its own dedicated axis ───────────────────────────
    if last_im is not None:
        cbar = fig.colorbar(last_im, cax=cbar_ax)
        cbar.set_label("Electrolyser activity (%)", fontsize=9, labelpad=8)
        cbar.set_ticks([0, 20, 40, 60, 80, 100])
        cbar.ax.tick_params(labelsize=8)
        for t in [20, 40, 60, 80]:
            cbar.ax.axhline(t, color="white", linewidth=0.5, alpha=0.6)
    else:
        cbar_ax.set_visible(False)

    save_path = out_dir / f"{island}.{fmt}"
    fig.savefig(save_path, dpi=dpi)      # no bbox_inches="tight"
    plt.close(fig)
    log.info("  Saved -> %s", save_path)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(gdx_path: str, out_dir: str, fmt: str, dpi: int,
         islands: list, years: list, preloaded_data=None):

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # if preloaded_data is not None:
    #     data = preloaded_data
    # else:
    #     data = load_data(gdx_path)
    ael  = extract_ael_consumption(data)

    if ael.empty:
        log.error("No AEL data. Verify tech='%s', commodity='%s' in commodity_balance.",
                  AEL_TECH, AEL_COMMODITY)
        return

    for island in islands:
        log.info("Plotting %s ...", island)
        plot_island_ael(ael, island, years, out_path, fmt, dpi)

    log.info("Done. Figures saved to: %s", out_path)
    print(f"\nAll {len(islands)} island figures written to: {out_path}/")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--gdx",     default=GDX_PATH)
    parser.add_argument("--out",     default=OUTPUT_DIR)
    parser.add_argument("--fmt",     default="png",
                        choices=["png", "pdf", "svg", "tiff"])
    parser.add_argument("--dpi",     default=200, type=int)
    parser.add_argument("--islands", nargs="+", default=ISLANDS)
    parser.add_argument("--years",   nargs="+", default=YEARS)
    parser.add_argument("--tech",    default=AEL_TECH)
    args = parser.parse_args()

    AEL_TECH = args.tech
    main(args.gdx, args.out, args.fmt, args.dpi, args.islands, args.years)
##############Heat pump activity ###############################################
import argparse
import logging
from pathlib import Path

import gdxpds
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────
GDX_PATH   = "../GDX_results/IP_2050_Final_S23_minload.gdx"
OUTPUT_DIR = "figures/S_23/heat_pump_activity_minload"
YEARS      = ["2040", "2050"]
ISLANDS    = [
    "CI_model", "FJ_model", "FSM_model", "KB_model",  "MI_model",
    "NU_model", "NE_model", "PU_model",  "PNG_model", "SA_model",
    "SI_model", "TA_model", "TU_model",  "VU_model",
]

HP_TECH      = "HP"
HP_COMMODITY = "Elec"

# ── Month axis ────────────────────────────────────────────────────────────────
MONTH_BOUNDARIES = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]
MONTH_LABELS     = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]
MONTH_MIDPOINTS  = [(MONTH_BOUNDARIES[i] + MONTH_BOUNDARIES[i + 1]) / 2
                    for i in range(12)]

# ─────────────────────────────────────────────────────────────────────────────
# COLOUR MAP
# YlGnBu-inspired (matching battery SOC design):
# pale yellow (idle) → mint → teal → deep ocean blue (full load)
# Perceptually uniform, colorblind-safe, prints well in greyscale too.
# ─────────────────────────────────────────────────────────────────────────────
HP_CMAP = LinearSegmentedColormap.from_list(
    "soc",
    ["#FFFFCC", "#C7E9B4", "#41B6C4", "#1D91C0", "#0C2C84"],
    N=256,
)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT STYLE
# IMPORTANT: savefig.bbox is intentionally NOT "tight".
# "tight" re-expands the canvas after layout and drags the colorbar axis
# back over the rightmost heatmap panel.  All margins are controlled via
# gridspec left/right/top/bottom fractions instead.
# ─────────────────────────────────────────────────────────────────────────────
rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":         9,
    "axes.titlesize":    10,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "axes.linewidth":    0.7,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        120,
    "savefig.dpi":       200,
    "savefig.facecolor": "white",
    # savefig.bbox deliberately omitted (defaults to "standard")
})


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_gdx(path: str) -> dict:
    log.info("Loading GDX: %s", path)
    data = gdxpds.to_dataframes(path)
    log.info("Available symbols: %s", sorted(data.keys()))
    return data


def extract_hp_consumption(data: dict) -> pd.DataFrame:
    """
    Pull commodity_balance, filter to HP tech + Elec commodity.
    Take absolute value of the negative consumption rows.

    Returns DataFrame with columns:
        hour      (int 0-8759)
        island    (str)
        year      (str)
        elec_gwh  (float, GWh, always >= 0)
    """
    cb   = data["commodity_balance"]
    mask = (
        (cb["techs"]     == HP_TECH) &
        (cb["commodity"] == HP_COMMODITY) &
        (cb["Value"]     <  0)
    )
    hp = cb.loc[mask].copy()

    if hp.empty:
        log.warning("No rows for tech='%s', commodity='%s', Value<0.",
                    HP_TECH, HP_COMMODITY)
        return pd.DataFrame(columns=["hour", "island", "year", "elec_gwh"])

    hp["hour"] = (
        hp["timeModel"].str.replace("tm", "", regex=False).astype(int) - 1
    )
    hp.rename(columns={
        "accNodesModel": "island",
        "accYears":      "year",
        "Value":         "elec_gwh",
    }, inplace=True)

    hp["elec_gwh"] = hp["elec_gwh"].abs()
    hp = hp.groupby(["hour", "island", "year"], as_index=False)["elec_gwh"].sum()

    log.info("HP rows: %d", len(hp))
    log.info("Islands: %s", sorted(hp["island"].unique()))
    log.info("Years  : %s", sorted(hp["year"].unique()))

    return hp[["hour", "island", "year", "elec_gwh"]]


# ─────────────────────────────────────────────────────────────────────────────
# ACTIVITY MATRIX BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_activity_matrix(hp: pd.DataFrame, island: str, year: str):
    mask = (hp["island"] == island) & (hp["year"] == year)
    sub  = hp.loc[mask, ["hour", "elec_gwh"]].copy()

    if sub.empty:
        return np.full((24, 365), np.nan), 0.0, 0.0, 0.0, 0.0

    full = pd.Series(0.0, index=range(8760))
    full.update(sub.set_index("hour")["elec_gwh"])

    peak_gwh  = float(full.max())
    total_gwh = float(full.sum())

    if peak_gwh == 0:
        return np.full((24, 365), np.nan), 0.0, 0.0, 0.0, 0.0

    activity_pct = (full / peak_gwh * 100).values
    matrix       = activity_pct.reshape(365, 24).T

    avg_activity = float(activity_pct.mean())
    hours_active = float((activity_pct > 0).mean() * 100)

    return matrix, peak_gwh, avg_activity, hours_active, total_gwh


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE SUBPLOT RENDERER
# ─────────────────────────────────────────────────────────────────────────────

def draw_activity_heatmap(ax, matrix: np.ndarray, year: str,
                          peak_gwh: float, avg_activity: float,
                          hours_active: float, total_gwh: float,
                          show_ylab: bool):

    if np.all(np.isnan(matrix)):
        ax.text(0.5, 0.5, "No HP data\n(heat pump not deployed)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=9, color="#888888")
        ax.set_title(year, fontweight="bold")
        return None

    im = ax.imshow(
        matrix,
        aspect="auto",
        origin="upper",
        cmap=HP_CMAP,
        vmin=0, vmax=100,
        interpolation="nearest",
        extent=[1, 365, 23.5, -0.5],
    )

    # ── X-axis: month names, no overlapping day numbers ───────────────────────
    ax.set_xticks(MONTH_MIDPOINTS)
    ax.set_xticklabels(MONTH_LABELS, fontsize=8)
    ax.set_xticks(MONTH_BOUNDARIES, minor=True)
    ax.tick_params(axis="x", which="major", length=0)          # labels only
    ax.tick_params(axis="x", which="minor", length=3,
                   color="#aaaaaa", width=0.5)
    ax.set_xlim(1, 365)
    ax.set_xlabel("Month", labelpad=4)

    for mb in MONTH_BOUNDARIES[1:-1]:
        ax.axvline(mb, color="white", linewidth=0.4, alpha=0.55)

    # ── Y-axis ────────────────────────────────────────────────────────────────
    ax.set_yticks([0, 6, 12, 18, 23])
    ax.set_yticklabels(["00:00", "06:00", "12:00", "18:00", "23:00"],
                       fontsize=7.5)
    ax.set_ylim(23.5, -0.5)
    if show_ylab:
        ax.set_ylabel("Hour of day", labelpad=4)

    # ── Title + stats ─────────────────────────────────────────────────────────
    ax.set_title(year, fontweight="bold", pad=14)
    ax.text(
        0.5, 1.01,
        (f"Peak: {peak_gwh:.4f} GWh/h  |  "
         f"Total: {total_gwh:.2f} GWh/yr  |  "
         f"Avg activity: {avg_activity:.1f}%  |  "
         f"Hours active: {hours_active:.1f}%"),
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=7.2, color="#444444", fontstyle="italic",
    )

    return im


# ─────────────────────────────────────────────────────────────────────────────
# PER-ISLAND FIGURE
# ─────────────────────────────────────────────────────────────────────────────

def plot_island_hp(hp: pd.DataFrame, island: str,
                   years: list, out_dir: Path, fmt: str, dpi: int):

    island_label = island.replace("_model", "")
    n_years      = len(years)

    panel_w = 5.5                              # inches per heatmap panel
    cbar_w  = 0.3                              # inches for colorbar column
    gap     = 0.55                             # right-side whitespace (inches)
    fig_w   = panel_w * n_years + cbar_w + gap + 0.7   # 0.7 for left margin
    fig_h   = 5.2

    fig = plt.figure(figsize=(fig_w, fig_h))

    # right edge fraction: leaves exactly (cbar_w + gap) inches for cbar+space
    right_frac = 1.0 - (cbar_w + gap) / fig_w

    gs = fig.add_gridspec(
        1, n_years + 1,
        width_ratios=[panel_w] * n_years + [cbar_w],
        wspace=0.06,
        left=0.07,
        right=right_frac,
        top=0.80,
        bottom=0.14,
    )

    axes    = [fig.add_subplot(gs[0, i]) for i in range(n_years)]
    cbar_ax = fig.add_subplot(gs[0, n_years])

    for ax in axes[1:]:
        ax.sharey(axes[0])

    fig.suptitle(
        f"{island_label} — Heat Pump (HP) Activity  "
        f"(% of annual peak hourly consumption)",
        fontsize=11, fontweight="bold", y=0.95,
    )

    last_im = None
    for idx, year in enumerate(years):
        matrix, peak_gwh, avg_act, h_active, total_gwh = \
            build_activity_matrix(hp, island, year)
        im = draw_activity_heatmap(
            axes[idx], matrix, year, peak_gwh, avg_act, h_active, total_gwh,
            show_ylab=(idx == 0),
        )
        if im is not None:
            last_im = im
        if idx > 0:
            plt.setp(axes[idx].get_yticklabels(), visible=False)

    # ── Colorbar anchored to its own dedicated axis ───────────────────────────
    if last_im is not None:
        cbar = fig.colorbar(last_im, cax=cbar_ax)
        cbar.set_label("Heat pump activity (%)", fontsize=9, labelpad=8)
        cbar.set_ticks([0, 20, 40, 60, 80, 100])
        cbar.ax.tick_params(labelsize=8)
        for t in [20, 40, 60, 80]:
            cbar.ax.axhline(t, color="white", linewidth=0.5, alpha=0.6)
    else:
        cbar_ax.set_visible(False)

    save_path = out_dir / f"{island}.{fmt}"
    fig.savefig(save_path, dpi=dpi)      # no bbox_inches="tight"
    plt.close(fig)
    log.info("  Saved -> %s", save_path)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(gdx_path: str, out_dir: str, fmt: str, dpi: int,
         islands: list, years: list, preloaded_data=None):

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # if preloaded_data is not None:
    #     data = preloaded_data
    # else:
    #     data = load_data(gdx_path)
    hp  = extract_hp_consumption(data)

    if hp.empty:
        log.error("No HP data. Verify tech='%s', commodity='%s' in commodity_balance.",
                  HP_TECH, HP_COMMODITY)
        return

    for island in islands:
        log.info("Plotting %s ...", island)
        plot_island_hp(hp, island, years, out_path, fmt, dpi)

    log.info("Done. Figures saved to: %s", out_path)
    print(f"\nAll {len(islands)} island figures written to: {out_path}/")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--gdx",     default=GDX_PATH)
    parser.add_argument("--out",     default=OUTPUT_DIR)
    parser.add_argument("--fmt",     default="png",
                        choices=["png", "pdf", "svg", "tiff"])
    parser.add_argument("--dpi",     default=200, type=int)
    parser.add_argument("--islands", nargs="+", default=ISLANDS)
    parser.add_argument("--years",   nargs="+", default=YEARS)
    parser.add_argument("--tech",    default=HP_TECH)
    args = parser.parse_args()

    HP_TECH = args.tech
    main(args.gdx, args.out, args.fmt, args.dpi, args.islands, args.years)
 
##############################original for capacities, generation and end use cases####################
import argparse
import logging
from pathlib import Path

import gdxpds
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib.patches import Patch

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
GDX_PATH   = "../GDX_results/IP_2050_Final_S23_minload.gdx"
OUTPUT_DIR = "figures/S_23/capacity_overview_minload"
YEARS      = ["2020", "2030", "2040", "2050"]
ISLANDS    = [
    "CI_model", "FJ_model", "FSM_model", "KB_model",  "MI_model",
    "NU_model", "NE_model", "PU_model",  "PNG_model", "SA_model",
    "SI_model", "TA_model", "TU_model",  "VU_model",
]

# ── Generation technologies (panels 1 & 2) ───────────────────────────────────
EL_PRODUCTION_TECHS = [
    "PV_B", "PV_N",
    "WindOnshore_B", "WindOnshore_N", "WindOffshore_N",
    "Wave_N",
    "Hydro_B", "Hydro_N",
    "BG_B", "BG_N",
    "NG_plant",
    "DG",
    "Geothermal_B",
]

# ── Capacity correction factors (model scaling artefacts) ────────────────────
# Capacities are UNDERREPORTED in the GDX by these percentages.
# True value = reported / (1 - underreport_fraction)
#   Hydro_B, Hydro_N  : underreported by 55 % → scale = 1 / 0.45 ≈ 2.222
#   Geothermal_B      : underreported by 78 % → scale = 1 / 0.22 ≈ 4.545
CAPACITY_SCALE = {
    "Hydro_B":      1 / (1 - 0.55),   # ÷ 0.45
    "Hydro_N":      1 / (1 - 0.55),   # ÷ 0.45
    "Geothermal_B": 1 / (1 - 0.78),   # ÷ 0.22
}

# ── End-use / converter technologies (panel 3) ───────────────────────────────
END_USE_TECHS = [
    "Battery",
    "DW_Electric_converter",
    "HP",
    "cook_el",
    "Industry_EL",
    "RO",
    "AEL",
    "Ammonia_synthesis",
    "DAC",
    "LDV_el",
    "HDV_el",
    "MDV_el",
    "Bus_el",
    "Two_wheel_el",
    "Aviation_el",
    "Ship_BEV",
    "Demand",
]

# ── Technology display labels ─────────────────────────────────────────────────
TECH_LABELS = {
    # Generation
    "PV_B":              "Solar PV (existing)",
    "PV_N":              "Solar PV (new)",
    "WindOnshore_B":     "Wind onshore (existing)",
    "WindOnshore_N":     "Wind onshore (new)",
    "WindOffshore_N":    "Wind offshore",
    "Wave_N":            "Wave",
    "Hydro_B":           "Hydro (existing)",
    "Hydro_N":           "Hydro (new)",
    "BG_B":              "Biogas (existing)",
    "BG_N":              "Biogas (new)",
    "NG_plant":          "Natural gas",
    "DG":                "Diesel generator",
    "Geothermal_B":      "Geothermal",
    # End-use
    "Battery":               "Battery storage",
    "DW_Electric_converter": "Electric hot water",
    "HP":                    "Heat pump",
    "cook_el":               "Electric cooking",
    "Industry_EL":           "Industrial electr.",
    "RO":                    "Desalination (RO)",
    "AEL":                   "Electrolyser (AEL)",
    "Ammonia_synthesis":     "Ammonia synthesis",
    "DAC":                   "Direct air capture",
    "LDV_el":                "Light-duty EV",
    "HDV_el":                "Heavy-duty EV",
    "MDV_el":                "Medium-duty EV",
    "Bus_el":                "Electric bus",
    "Two_wheel_el":          "E-motorbike/scooter",
    "Aviation_el":           "Electric aviation",
    "Ship_BEV":              "Electric shipping",
    "Demand":                "Residential demand",
}

# ── Colour palettes ───────────────────────────────────────────────────────────
GEN_COLORS = {
    "PV_B":           "#F0C234",
    "PV_N":           "#E6A817",
    "WindOnshore_B":  "#56B4E9",
    "WindOnshore_N":  "#0072B2",
    "WindOffshore_N": "#004D80",
    "Wave_N":         "#009E73",
    "Hydro_B":        "#44AA99",
    "Hydro_N":        "#117733",
    "BG_B":           "#CC79A7",
    "BG_N":           "#882255",
    "NG_plant":       "#E69F00",
    "DG":             "#D55E00",
    "Geothermal_B":   "#8B4513",
}

ENDUSE_COLORS = {
    "Battery":               "#1A85FF",
    "DW_Electric_converter": "#D41159",
    "HP":                    "#FFC20A",
    "cook_el":               "#994F00",
    "Industry_EL":           "#006CD1",
    "RO":                    "#26C6DA",
    "AEL":                   "#40B0A6",
    "Ammonia_synthesis":     "#E1BE6A",
    "DAC":                   "#A0522D",
    "LDV_el":                "#5D3A9B",
    "HDV_el":                "#E66100",
    "MDV_el":                "#009E73",
    "Bus_el":                "#0072B2",
    "Two_wheel_el":          "#CC79A7",
    "Aviation_el":           "#882255",
    "Ship_BEV":              "#D55E00",
    "Demand":                "#888888",
}

FALLBACK_COLOR = "#AAAAAA"

# ─────────────────────────────────────────────────────────────────────────────
# PLOT STYLE
# ─────────────────────────────────────────────────────────────────────────────
rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":         9,
    "axes.titlesize":    10,
    "axes.labelsize":    9,
    "xtick.labelsize":   9,
    "ytick.labelsize":   8,
    "legend.fontsize":   7.5,
    "axes.linewidth":    0.7,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "xtick.direction":   "out",
    "ytick.direction":   "out",
    "xtick.major.size":  3,
    "ytick.major.size":  3,
    "grid.linewidth":    0.4,
    "grid.alpha":        0.35,
    "grid.color":        "#888888",
    "figure.dpi":        120,
    "savefig.dpi":       200,
    "savefig.bbox":      "tight",
    "savefig.facecolor": "white",
})

BAR_WIDTH = 0.6


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_gdx(path: str) -> dict:
    log.info("Loading GDX: %s", path)
    data = gdxpds.to_dataframes(path)
    log.info("Available symbols: %s", sorted(data.keys()))
    return data


# ─────────────────────────────────────────────────────────────────────────────
# PANEL 1 — INSTALLED CAPACITY
# ─────────────────────────────────────────────────────────────────────────────

def extract_capacity(data: dict) -> pd.DataFrame:
    """
    converter_caps | capType == 'total' | commodity == 'Elec'
    Returns DataFrame: island, year (str), tech, capacity_mw (float)

    Unit conversion : GW → MW  (multiply by 1000)
    Capacity corrections (model scaling artefacts):
        Hydro_B, Hydro_N  : multiply by (1 - 0.55) = 0.45
        Geothermal_B      : multiply by (1 - 0.78) = 0.22
    """
    cc = data["converter_caps"]
    mask = (
        (cc["capType"]   == "total") &
        (cc["commodity"] == "Elec") &
        (cc["techs"].isin(EL_PRODUCTION_TECHS))
    )
    df = cc.loc[mask].copy()
    df.rename(columns={
        "accNodesModel": "island",
        "accYears":      "year",
        "techs":         "tech",
        "Value":         "capacity_mw",
    }, inplace=True)
    df["year"] = df["year"].astype(str)

    # Step 1: GW → MW
    df["capacity_mw"] = df["capacity_mw"] * 1000.0

    # Step 2: apply per-tech correction factors
    for tech, scale in CAPACITY_SCALE.items():
        mask_tech = df["tech"] == tech
        df.loc[mask_tech, "capacity_mw"] *= scale
        n = mask_tech.sum()
        if n > 0:
            log.info(
                "Capacity correction applied: %s × %.2f  (%d rows)",
                tech, scale, n,
            )

    return df[["island", "year", "tech", "capacity_mw"]]


def capacity_pivot(cap: pd.DataFrame, island: str) -> pd.DataFrame:
    sub = cap[cap["island"] == island]
    if sub.empty:
        return pd.DataFrame(index=YEARS)
    pivot = (
        sub.pivot_table(index="year", columns="tech",
                        values="capacity_mw", aggfunc="sum")
           .reindex(YEARS, fill_value=0.0)
           .fillna(0.0)
    )
    ordered = [t for t in EL_PRODUCTION_TECHS if t in pivot.columns
               and (pivot[t] > 0).any()]
    return pivot[ordered] if ordered else pd.DataFrame(index=YEARS)


# ─────────────────────────────────────────────────────────────────────────────
# PANEL 2 — ELECTRICITY GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def extract_generation(data: dict) -> pd.DataFrame:
    """
    commodity_balance_annual | commodity == 'Elec' | balanceType == 'net' | Value > 0
    Returns DataFrame: island, year (str), tech, generation_gwh (float)
    """
    cb = data["commodity_balance_annual"]
    mask = (
        (cb["commodity"]   == "Elec") &
        (cb["balanceType"] == "net") &
        (cb["techs"].isin(EL_PRODUCTION_TECHS)) &
        (cb["Value"]       >  0)
    )
    df = cb.loc[mask].copy()
    df.rename(columns={
        "accNodesModel": "island",
        "accYears":      "year",
        "techs":         "tech",
        "Value":         "generation_gwh",
    }, inplace=True)
    df["year"] = df["year"].astype(str)
    return df[["island", "year", "tech", "generation_gwh"]]


def generation_pivot(gen: pd.DataFrame, island: str) -> pd.DataFrame:
    sub = gen[gen["island"] == island]
    if sub.empty:
        return pd.DataFrame(index=YEARS)
    pivot = (
        sub.pivot_table(index="year", columns="tech",
                        values="generation_gwh", aggfunc="sum")
           .reindex(YEARS, fill_value=0.0)
           .fillna(0.0)
    )
    ordered = [t for t in EL_PRODUCTION_TECHS if t in pivot.columns
               and (pivot[t] > 0).any()]
    return pivot[ordered] if ordered else pd.DataFrame(index=YEARS)


# ─────────────────────────────────────────────────────────────────────────────
# PANEL 3 — ELECTRICITY END-USE
# ─────────────────────────────────────────────────────────────────────────────

def extract_enduse(data: dict) -> pd.DataFrame:
    """
    commodity_balance_annual | commodity == 'Elec' | balanceType == 'net' | Value < 0
    Filters to END_USE_TECHS.
    Returns DataFrame: island, year (str), tech, enduse_gwh (float, positive)
    """
    cb = data["commodity_balance_annual"]
    mask = (
        (cb["commodity"]   == "Elec") &
        (cb["balanceType"] == "net") &
        (cb["techs"].isin(END_USE_TECHS)) &
        (cb["Value"]       <  0)
    )
    df = cb.loc[mask].copy()
    df.rename(columns={
        "accNodesModel": "island",
        "accYears":      "year",
        "techs":         "tech",
        "Value":         "enduse_gwh",
    }, inplace=True)
    df["enduse_gwh"] = df["enduse_gwh"].abs()
    df["year"] = df["year"].astype(str)
    return df[["island", "year", "tech", "enduse_gwh"]]


def enduse_pivot(eu: pd.DataFrame, island: str) -> pd.DataFrame:
    sub = eu[eu["island"] == island]
    if sub.empty:
        return pd.DataFrame(index=YEARS)
    pivot = (
        sub.pivot_table(index="year", columns="tech",
                        values="enduse_gwh", aggfunc="sum")
           .reindex(YEARS, fill_value=0.0)
           .fillna(0.0)
    )
    ordered = [t for t in END_USE_TECHS if t in pivot.columns
               and (pivot[t] > 0).any()]
    return pivot[ordered] if ordered else pd.DataFrame(index=YEARS)


# ─────────────────────────────────────────────────────────────────────────────
# STACKED BAR HELPER
# ─────────────────────────────────────────────────────────────────────────────

def draw_stacked_bar(ax, pivot: pd.DataFrame, color_map: dict,
                     ylabel: str, title: str, unit_scale: float = 1.0):
    if pivot.empty or pivot.shape[1] == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="#888888")
        ax.set_title(title, fontweight="bold", pad=6)
        return

    x      = np.arange(len(YEARS))
    bottom = np.zeros(len(YEARS))

    for tech in pivot.columns:
        values = pivot[tech].values * unit_scale
        color  = color_map.get(tech, FALLBACK_COLOR)
        label  = TECH_LABELS.get(tech, tech)
        ax.bar(x, values, BAR_WIDTH,
               bottom=bottom, label=label, color=color,
               edgecolor="white", linewidth=0.3, zorder=3)
        bottom += values

    for i, total in enumerate(bottom):
        if total > 0:
            ax.text(x[i], total * 1.01, f"{total:,.0f}",
                    ha="center", va="bottom", fontsize=7, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels(YEARS, fontsize=9)
    ax.set_xlabel("Year", labelpad=4)
    ax.set_ylabel(ylabel, labelpad=4)
    ax.set_title(title, fontweight="bold", pad=6)
    ax.set_xlim(-0.5, len(YEARS) - 0.5)
    ax.set_ylim(bottom=0)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v:,.0f}")
    )


# ─────────────────────────────────────────────────────────────────────────────
# PER-ISLAND FIGURE
# ─────────────────────────────────────────────────────────────────────────────

def plot_island(cap_df: pd.DataFrame, gen_df: pd.DataFrame,
                eu_df: pd.DataFrame, island: str,
                out_dir: Path, fmt: str, dpi: int):
    island_label = island.replace("_model", "")

    cap_piv = capacity_pivot(cap_df, island)
    gen_piv = generation_pivot(gen_df, island)
    eu_piv  = enduse_pivot(eu_df, island)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"{island_label} — Installed Capacity, Electricity Generation "
        f"& End-Use  (2020 – 2050)",
        fontsize=12, fontweight="bold", y=1.02,
    )

    draw_stacked_bar(axes[0], cap_piv, GEN_COLORS,
                     "Installed capacity (MW)", "Installed Capacity")
    draw_stacked_bar(axes[1], gen_piv, GEN_COLORS,
                     "Generation (GWh)", "Electricity Generation")
    draw_stacked_bar(axes[2], eu_piv,  ENDUSE_COLORS,
                     "Electricity consumed (GWh)", "Electricity End-Use")

    # ── Generation legend (panels 1 & 2) ─────────────────────────────────────
    gen_techs_present = list(dict.fromkeys(
        list(cap_piv.columns) + list(gen_piv.columns)
    ))
    gen_handles = [
        Patch(facecolor=GEN_COLORS.get(t, FALLBACK_COLOR),
              edgecolor="white", linewidth=0.3,
              label=TECH_LABELS.get(t, t))
        for t in EL_PRODUCTION_TECHS if t in gen_techs_present
    ]
    if gen_handles:
        leg1 = axes[0].legend(
            handles=gen_handles,
            loc="upper center", bbox_to_anchor=(1.0, -0.18),
            ncol=min(len(gen_handles), 4), frameon=False,
            fontsize=7.5, title="Generation technologies", title_fontsize=8,
        )
        axes[0].add_artist(leg1)

    # ── End-use legend (panel 3) ──────────────────────────────────────────────
    eu_handles = [
        Patch(facecolor=ENDUSE_COLORS.get(t, FALLBACK_COLOR),
              edgecolor="white", linewidth=0.3,
              label=TECH_LABELS.get(t, t))
        for t in END_USE_TECHS if t in list(eu_piv.columns)
    ]
    if eu_handles:
        axes[2].legend(
            handles=eu_handles,
            loc="upper center", bbox_to_anchor=(0.5, -0.18),
            ncol=min(len(eu_handles), 4), frameon=False,
            fontsize=7.5, title="End-use sectors", title_fontsize=8,
        )

    plt.tight_layout(rect=[0, 0.12, 1, 1])
    save_path = out_dir / f"{island}.{fmt}"
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    log.info("  Saved -> %s", save_path)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(gdx_path: str, out_dir: str, fmt: str, dpi: int,
         islands: list, years: list, preloaded_data=None):

    global YEARS
    YEARS = years

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # if preloaded_data is not None:
    #     data = preloaded_data
    # else:
    #     data = load_data(gdx_path)

    log.info("Extracting installed capacities ...")
    cap_df = extract_capacity(data)

    log.info("Extracting electricity generation ...")
    gen_df = extract_generation(data)

    log.info("Extracting electricity end-use ...")
    eu_df  = extract_enduse(data)

    for island in islands:
        log.info("Plotting %s ...", island)
        plot_island(cap_df, gen_df, eu_df, island, out_path, fmt, dpi)

    log.info("Done. All figures saved to: %s", out_path)
    print(f"\nAll {len(islands)} island figures written to: {out_path}/")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--gdx",     default=GDX_PATH)
    parser.add_argument("--out",     default=OUTPUT_DIR)
    parser.add_argument("--fmt",     default="png",
                        choices=["png", "pdf", "svg", "tiff"])
    parser.add_argument("--dpi",     default=200, type=int)
    parser.add_argument("--islands", nargs="+", default=ISLANDS)
    parser.add_argument("--years",   nargs="+",
                        default=["2020", "2030", "2040", "2050"])
    args = parser.parse_args()

    main(args.gdx, args.out, args.fmt, args.dpi, args.islands, args.years)
    ######################################################all commodity demand #############################

    ####################heat generation #####################################################
    """
Pacific Island Countries (PICs) — Heat Generation by Technology
================================================================
Reads 'commodity_balance_annual' from the GAMS GDX file and extracts
positive (production) flows for specific tech–commodity pairs representing
all forms of heat supply across 14 islands and 4 target years.

Filter applied
--------------
    balanceType == "net"
    Value       >  0          (positive = output / production)
    (tech, commodity) in the table below:

    Tech                  Commodity        Heat type
    ────────────────────  ───────────────  ──────────────────────────
    DW_Electric_converter DHW_el           Electric domestic hot water
    DW_LPG_converter      DHW_LPG          LPG domestic hot water
    Industry              Heat_industry    Industrial heat (diesel)
    cook_b                Heat_cooking     Biomass cooking
    cook_el               T_cook_el        Electric cooking
    cook_LPG              T_cook_LPG       LPG cooking
    HP                    Heat             Heat-pump space/water heat
    Industry_EL           T_Industry_EH    Industrial heat (elec/HP/ST)
    ST_N                  Heat             Solar thermal heat

All commodity types are treated as "heat" in the plot; the legend
uses technology names to distinguish the source.

Output
------
  figures/heat_generation/
      CI_model.png  ...  VU_model.png   ← per-island figures
      all_islands_overview.png          ← combined overview

Usage
-----
    python pic_heat_generation.py
    python pic_heat_generation.py --gdx path/to/results.gdx
                                  --out figures/heat_generation
                                  --fmt png --dpi 200
"""

import argparse
import logging
from pathlib import Path

import gdxpds
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib.patches import Patch

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
GDX_PATH   = "../GDX_results/IP_2050_Final_S23_minload.gdx"
OUTPUT_DIR = "figures/S_23/heat_generation"
YEARS      = ["2020","2030", "2040", "2050"]
ISLANDS    = [
    "CI_model", "FJ_model", "FSM_model", "KB_model",  "MI_model",
    "NU_model", "NE_model", "PU_model",  "PNG_model", "SA_model",
    "SI_model", "TA_model", "TU_model",  "VU_model",
]

# ── Tech–commodity pairs to extract (exact matching, no cross-contamination) ──
# Each tuple is (tech, commodity_produced).
HEAT_PAIRS = [
    ("DW_Electric_converter", "DHW_el"),
    ("DW_LPG_converter",      "DHW_LPG"),
    ("Industry",              "Heat_industry"),
    ("cook_b",                "Heat_cooking"),
    ("cook_el",               "T_cook_el"),
    ("cook_LPG",              "T_cook_LPG"),
    ("HP",                    "Heat"),
    ("Industry_EL",           "T_Industry_EH"),
    ("ST_N",                  "Heat"),
]

HEAT_PAIRS_SET = set(HEAT_PAIRS)

# Ordered list of unique techs (defines stack order bottom → top)
HEAT_TECHS_ORDER = list(dict.fromkeys(t for t, _ in HEAT_PAIRS))

# ── Technology display labels ─────────────────────────────────────────────────
TECH_LABELS = {
    "DW_Electric_converter": "Electric domestic hot water",
    "DW_LPG_converter":      "LPG domestic hot water",
    "Industry":              "Industry heat (diesel)",
    "cook_b":                "Biomass cooking",
    "cook_el":               "Electric cooking",
    "cook_LPG":              "LPG cooking",
    "HP":                    "Heat pump",
    "Industry_EL":           "Industry direct electric heat",
    "ST_N":                  "Solar thermal",
}

# ── Colour palette — colorblind-safe (Wong 2011 + extensions) ─────────────────
TECH_COLORS = {
    "DW_Electric_converter": "#0072B2",   # blue
    "DW_LPG_converter":      "#E69F00",   # amber
    "Industry":              "#D55E00",   # vermillion
    "cook_b":                "#117733",   # dark green
    "cook_el":               "#56B4E9",   # sky blue
    "cook_LPG":              "#F0E442",   # yellow
    "HP":                    "#009E73",   # teal
    "Industry_EL":           "#CC79A7",   # pink
    "ST_N":                  "#E6A817",   # dark amber / solar
}

BAR_WIDTH = 0.6

# ─────────────────────────────────────────────────────────────────────────────
# PLOT STYLE
# ─────────────────────────────────────────────────────────────────────────────
rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":         9,
    "axes.titlesize":    10,
    "axes.labelsize":    9,
    "xtick.labelsize":   9,
    "ytick.labelsize":   8,
    "legend.fontsize":   7.5,
    "axes.linewidth":    0.7,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "xtick.direction":   "out",
    "ytick.direction":   "out",
    "xtick.major.size":  3,
    "ytick.major.size":  3,
    "grid.linewidth":    0.4,
    "grid.alpha":        0.35,
    "grid.color":        "#888888",
    "figure.dpi":        120,
    "savefig.dpi":       200,
    "savefig.bbox":      "tight",
    "savefig.facecolor": "white",
})


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_gdx(path: str) -> dict:
    log.info("Loading GDX: %s", path)
    data = gdxpds.to_dataframes(path)
    log.info("Available symbols: %s", sorted(data.keys()))
    return data


def extract_heat(data: dict) -> pd.DataFrame:
    """
    commodity_balance_annual
        | balanceType == "net"
        | Value       >  0              (positive = production output)
        | (tech, commodity) in HEAT_PAIRS_SET

    Returns DataFrame: island, year (str), tech, heat_gwh (float, positive)
    All commodity types are aggregated to a single "heat" column per tech.
    """
    cb = data["commodity_balance_annual"]

    # Pre-filter to candidate techs and commodities (fast)
    all_techs = [t for t, _ in HEAT_PAIRS]
    all_comms = [c for _, c in HEAT_PAIRS]

    pre = cb.loc[
        (cb["balanceType"] == "positive") &
        (cb["Value"]       >  0) &
        (cb["techs"].isin(all_techs)) &
        (cb["commodity"].isin(all_comms))
    ].copy()

    if pre.empty:
        log.warning("No heat production rows found — check tech/commodity names.")
        return pd.DataFrame(columns=["island", "year", "tech", "heat_gwh"])

    # Exact pair filtering (removes e.g. HP producing DHW_el, which is not in the table)
    pre["_pair"] = list(zip(pre["techs"], pre["commodity"]))
    df = pre[pre["_pair"].isin(HEAT_PAIRS_SET)].copy()
    df.drop(columns=["_pair"], inplace=True)

    df.rename(columns={
        "accNodesModel": "island",
        "accYears":      "year",
        "techs":         "tech",
        "Value":         "heat_gwh",
    }, inplace=True)
    df["year"] = df["year"].astype(str)

    # Sum over commodities — all output from a tech counts as "heat"
    df = (
        df.groupby(["island", "year", "tech"], as_index=False)["heat_gwh"]
        .sum()
    )

    log.info("Heat rows extracted: %d", len(df))
    log.info("Techs found        : %s", sorted(df["tech"].unique()))
    return df[["island", "year", "tech", "heat_gwh"]]


def heat_pivot(df: pd.DataFrame, island: str) -> pd.DataFrame:
    """
    Returns (years × techs) pivot of heat production in GWh for one island.
    Rows = YEARS, columns = techs active for this island (non-zero only).
    """
    sub = df[df["island"] == island]
    if sub.empty:
        return pd.DataFrame(index=YEARS)

    pivot = (
        sub.pivot_table(index="year", columns="tech",
                        values="heat_gwh", aggfunc="sum")
           .reindex(YEARS, fill_value=0.0)
           .fillna(0.0)
    )
    ordered = [t for t in HEAT_TECHS_ORDER
               if t in pivot.columns and (pivot[t] > 0).any()]
    return pivot[ordered] if ordered else pd.DataFrame(index=YEARS)


# ─────────────────────────────────────────────────────────────────────────────
# LEGEND BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_legend_handles(techs_present: list) -> list:
    return [
        Patch(facecolor=TECH_COLORS.get(t, "#AAAAAA"),
              edgecolor="white", linewidth=0.3,
              label=TECH_LABELS.get(t, t))
        for t in HEAT_TECHS_ORDER
        if t in techs_present
    ]


# ─────────────────────────────────────────────────────────────────────────────
# STACKED BAR HELPER
# ─────────────────────────────────────────────────────────────────────────────

def draw_stacked_bar(ax, pivot: pd.DataFrame, ylabel: str, title: str):
    """Draw a stacked heat bar chart for one island into ax."""
    if pivot.empty or pivot.shape[1] == 0:
        ax.text(0.5, 0.5, "No heat data", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="#888888")
        ax.set_title(title, fontweight="bold", pad=6)
        return

    x      = np.arange(len(YEARS))
    bottom = np.zeros(len(YEARS))

    for tech in pivot.columns:
        values = pivot[tech].values
        color  = TECH_COLORS.get(tech, "#AAAAAA")
        label  = TECH_LABELS.get(tech, tech)
        ax.bar(x, values, BAR_WIDTH,
               bottom=bottom, label=label, color=color,
               edgecolor="white", linewidth=0.3, zorder=3)
        bottom += values

    # Total label on top of each bar
    for i, total in enumerate(bottom):
        if total > 0:
            ax.text(x[i], total * 1.01, f"{total:,.1f}",
                    ha="center", va="bottom", fontsize=7.5, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels(YEARS, fontsize=9)
    ax.set_xlabel("Year", labelpad=4)
    ax.set_ylabel(ylabel, labelpad=4)
    ax.set_title(title, fontweight="bold", pad=6)
    ax.set_xlim(-0.5, len(YEARS) - 0.5)
    ax.set_ylim(bottom=0)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v:,.1f}")
    )


# ─────────────────────────────────────────────────────────────────────────────
# PER-ISLAND FIGURE
# ─────────────────────────────────────────────────────────────────────────────

def plot_island_heat(pivot: pd.DataFrame, island: str,
                     out_dir: Path, fmt: str, dpi: int):
    """One figure per island — 4 year bars stacked by heat technology."""
    island_label = island.replace("_model", "")

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle(
        f"{island_label} — Heat Generation by Technology (2020–2050)",
        fontsize=11, fontweight="bold", y=1.02,
    )

    draw_stacked_bar(ax, pivot,
                     ylabel="Heat generation (GWh)",
                     title="")

    handles = build_legend_handles(list(pivot.columns) if not pivot.empty else [])
    if handles:
        ax.legend(
            handles=handles,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            fontsize=7.2,
            title="Heat technology",
            title_fontsize=7.5,
        )

    plt.tight_layout()
    save_path = out_dir / f"{island}.{fmt}"
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved -> %s", save_path)


# ─────────────────────────────────────────────────────────────────────────────
# COMBINED OVERVIEW FIGURE
# ─────────────────────────────────────────────────────────────────────────────

def plot_overview(df: pd.DataFrame, islands: list,
                  out_dir: Path, fmt: str, dpi: int):
    """All 14 islands on one chart — 4 year-bars each, stacked by tech."""
    n_islands   = len(islands)
    n_years     = len(YEARS)
    group_width = n_years * BAR_WIDTH + 0.4
    x_groups    = np.arange(n_islands) * group_width
    year_offsets = (np.arange(n_years) * BAR_WIDTH
                    - (n_years - 1) * BAR_WIDTH / 2)

    all_techs_present = [t for t in HEAT_TECHS_ORDER
                         if t in df["tech"].values]

    fig, ax = plt.subplots(figsize=(max(18, n_islands * 2.0), 6))
    fig.suptitle(
        "Heat Generation by Technology — All Pacific Islands (2020–2050)",
        fontsize=12, fontweight="bold", y=1.02,
    )

    for g_idx, island in enumerate(islands):
        pivot = heat_pivot(df, island)
        if pivot.empty or pivot.shape[1] == 0:
            continue

        for y_idx, year in enumerate(YEARS):
            bar_x  = x_groups[g_idx] + year_offsets[y_idx]
            bottom = 0.0
            for tech in all_techs_present:
                if tech not in pivot.columns:
                    continue
                val   = float(pivot.loc[year, tech]) if year in pivot.index else 0.0
                color = TECH_COLORS.get(tech, "#AAAAAA")
                ax.bar(bar_x, val, BAR_WIDTH * 0.92,
                       bottom=bottom, color=color,
                       edgecolor="white", linewidth=0.2, zorder=3)
                bottom += val

    # X-axis island labels
    island_labels = [i.replace("_model", "") for i in islands]
    ax.set_xticks(x_groups)
    ax.set_xticklabels(island_labels, rotation=35, ha="right", fontsize=8)
    ax.set_xlabel("Island", labelpad=6)
    ax.set_ylabel("Heat generation (GWh)", labelpad=4)
    ax.set_ylim(bottom=0)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v:,.1f}")
    )

    # Year sub-labels below each group
    for g_idx in range(n_islands):
        for y_idx, year in enumerate(YEARS):
            ax.text(
                x_groups[g_idx] + year_offsets[y_idx],
                -ax.get_ylim()[1] * 0.04,
                year[-2:],
                ha="center", va="top", fontsize=6.5, color="#555555",
            )

    # Legend on the right
    handles = build_legend_handles(all_techs_present)
    ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        fontsize=7.2,
        title="Heat technology",
        title_fontsize=7.5,
    )

    plt.tight_layout()
    save_path = out_dir / f"all_islands_overview.{fmt}"
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("Overview saved -> %s", save_path)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(gdx_path: str, out_dir: str, fmt: str, dpi: int,
         islands: list, years: list, preloaded_data=None):

    global YEARS
    YEARS = years

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # if preloaded_data is not None:
    #     data = preloaded_data
    # else:
    #     data = load_data(gdx_path)

    log.info("Extracting heat generation ...")
    df = extract_heat(data)

    for island in islands:
        log.info("Plotting %s ...", island)
        pivot = heat_pivot(df, island)
        plot_island_heat(pivot, island, out_path, fmt, dpi)

    log.info("Plotting combined overview ...")
    plot_overview(df, islands, out_path, fmt, dpi)

    log.info("Done. Figures saved to: %s", out_path)
    print(f"\n{len(islands)} island figures + overview written to: {out_path}/")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--gdx",     default=GDX_PATH,
                        help="Path to GAMS GDX file")
    parser.add_argument("--out",     default=OUTPUT_DIR,
                        help="Output directory for figures")
    parser.add_argument("--fmt",     default="png",
                        choices=["png", "pdf", "svg", "tiff"],
                        help="Output file format (default: png)")
    parser.add_argument("--dpi",     default=200, type=int,
                        help="Save resolution in DPI (default: 200)")
    parser.add_argument("--islands", nargs="+", default=ISLANDS,
                        help="Subset of islands to plot (default: all 14)")
    parser.add_argument("--years",   nargs="+", default=YEARS,
                        help="Target years (default: 2020 2030 2040 2050)")
    args = parser.parse_args()

    main(args.gdx, args.out, args.fmt, args.dpi, args.islands, args.years)

############################################################################


####################AEL capacities##################################################
"""
Pacific Island Countries (PICs) — AEL Installed Capacity (2040 & 2050)
=======================================================================
Reads 'converter_caps' from the GAMS GDX file and plots the total
installed capacity of the AEL electrolyser (Hydrogen commodity) for
all 14 islands across 2040 and 2050.

    Source : converter_caps | techs == 'AEL' | commodity == 'Hydrogen' | capType == 'total'
    Unit   : GW → MW (× 1000)

Output
------
  figures/S_23/ael_capacity_minload/
      ael_capacity_overview.png
"""

"""
Pacific Island Countries (PICs) — AEL Installed Capacity (2040 & 2050)
=======================================================================
One figure per island — simple grouped bar chart showing AEL total
installed capacity for 2040 and 2050.

    Source : converter_caps | techs == 'AEL' | commodity == 'Hydrogen' | capType == 'total'
    Unit   : GW → MW (× 1000)
"""

import argparse
import logging
from pathlib import Path

import gdxpds
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib.patches import Patch

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
GDX_PATH   = "../GDX_results/IP_2050_Final_S23_minload.gdx"
OUTPUT_DIR = "figures/S_23/ael_capacity_minload"
YEARS      = ["2040", "2050"]
ISLANDS    = [
    "CI_model", "FJ_model", "FSM_model", "KB_model",  "MI_model",
    "NU_model", "NE_model", "PU_model",  "PNG_model", "SA_model",
    "SI_model", "TA_model", "TU_model",  "VU_model",
]

AEL_TECH      = "AEL"
AEL_COMMODITY = "Hydrogen"

# Colorbrewer-safe, print-friendly
YEAR_COLORS = {
    "2040": "#2196F3",   # medium blue
    "2050": "#FF5722",   # deep orange
}

BAR_WIDTH = 0.45

rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "xtick.labelsize":   10,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "axes.linewidth":    0.7,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "xtick.direction":   "out",
    "ytick.direction":   "out",
    "xtick.major.size":  3,
    "ytick.major.size":  3,
    "grid.linewidth":    0.4,
    "grid.alpha":        0.35,
    "grid.color":        "#888888",
    "figure.dpi":        120,
    "savefig.dpi":       200,
    "savefig.bbox":      "tight",
    "savefig.facecolor": "white",
})


# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_gdx(path: str) -> dict:
    log.info("Loading GDX: %s", path)
    return gdxpds.to_dataframes(path)


def extract_ael_capacity(data: dict) -> pd.DataFrame:
    cc   = data["converter_caps"]
    mask = (
        (cc["techs"]     == AEL_TECH) &
        (cc["commodity"] == AEL_COMMODITY) &
        (cc["capType"]   == "total")
    )
    df = cc.loc[mask].copy()

    if df.empty:
        log.warning("No rows found — check tech/commodity/capType names.")
        return pd.DataFrame(columns=["island", "year", "capacity_mw"])

    df.rename(columns={"accNodesModel": "island",
                        "accYears":      "year",
                        "Value":         "capacity_mw"}, inplace=True)
    df["year"]        = df["year"].astype(str)
    df["capacity_mw"] = df["capacity_mw"] * 1000.0   # GW → MW
    df = df[df["year"].isin(YEARS)]
    log.info("Rows extracted: %d", len(df))
    return df[["island", "year", "capacity_mw"]]


# ─────────────────────────────────────────────────────────────────────────────
# PER-ISLAND FIGURE
# ─────────────────────────────────────────────────────────────────────────────

def plot_island(df: pd.DataFrame, island: str, out_dir: Path, fmt: str, dpi: int):
    island_label = island.replace("_model", "")

    sub = df[df["island"] == island]
    # Build value dict {year: MW}, default 0 if not deployed
    values = {y: 0.0 for y in YEARS}
    for _, row in sub.iterrows():
        values[row["year"]] = row["capacity_mw"]

    x       = np.arange(len(YEARS))
    heights = [values[y] for y in YEARS]
    colors  = [YEAR_COLORS[y] for y in YEARS]
    y_max   = max(heights) if max(heights) > 0 else 1.0

    fig, ax = plt.subplots(figsize=(5, 4.5))

    bars = ax.bar(x, heights, BAR_WIDTH, color=colors,
                  edgecolor="white", linewidth=0.4, zorder=3)

    # Value labels above bars
    for bar, val in zip(bars, heights):
        if val > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + y_max * 0.02,
                f"{val:,.0f}",
                ha="center", va="bottom", fontsize=9, color="#333333",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(YEARS, fontsize=10)
    ax.set_xlabel("Year", labelpad=4)
    ax.set_ylabel("Installed capacity (MW)", labelpad=4)
    ax.set_title(f"{island_label} — AEL Total Installed Capacity",
                 fontweight="bold", pad=8)
    ax.set_xlim(-0.6, len(YEARS) - 0.4)
    ax.set_ylim(0, y_max * 1.18)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    handles = [Patch(facecolor=YEAR_COLORS[y], edgecolor="white",
                     linewidth=0.3, label=y) for y in YEARS]
    ax.legend(handles=handles, frameon=False, fontsize=9,
              title="Year", title_fontsize=9)

    plt.tight_layout()
    save_path = out_dir / f"{island}.{fmt}"
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    log.info("  Saved -> %s", save_path)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(gdx_path, out_dir, fmt, dpi, preloaded_data=None):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # if preloaded_data is not None:
    #     data = preloaded_data
    # else:
    #     data = load_data(gdx_path)
    df   = extract_ael_capacity(data)

    if df.empty:
        log.error("No data extracted.")
        return

    for island in ISLANDS:
        log.info("Plotting %s ...", island)
        plot_island(df, island, out_path, fmt, dpi)

    log.info("Done. Figures saved to: %s", out_path)
    print(f"\n14 island figures written to: {out_path}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gdx", default=GDX_PATH)
    parser.add_argument("--out", default=OUTPUT_DIR)
    parser.add_argument("--fmt", default="png",
                        choices=["png", "pdf", "svg", "tiff"])
    parser.add_argument("--dpi", default=200, type=int)
    args = parser.parse_args()
    main(args.gdx, args.out, args.fmt, args.dpi)
    
    #############################Trade flow practice figures###############################

# #     ############################system costs deduction####################################
import argparse
import logging
from pathlib import Path

import gdxpds
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib import rcParams

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
log = logging.getLogger(__name__)

GDX_PATH   = "../GDX_results/IP_2050_Final_S23_minload.gdx"
OUTPUT_DIR = "figures/S_23/system_cost"
YEARS      = ["2020", "2030", "2040", "2050"]

COST_COMPONENTS = ["Invest", "OMFix", "OMVar", "FuelCost"]

COMPONENT_COLORS = {
    "Invest":   "#1B3A5C",
    "OMFix":    "#2E6F95",
    "OMVar":    "#3D9DAA",
    "FuelCost": "#C06B3A",
}

COMPONENT_LABELS = {
    "Invest":   "Investment (CAPEX)",
    "OMFix":    "Fixed O&M",
    "OMVar":    "Variable O&M",
    "FuelCost": "Fuel Cost",
}

rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   9,
    "axes.linewidth":    0.8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "xtick.direction":   "out",
    "ytick.direction":   "out",
    "xtick.major.size":  4,
    "ytick.major.size":  4,
    "grid.linewidth":    0.4,
    "grid.alpha":        0.4,
    "grid.color":        "#AAAAAA",
    "figure.dpi":        120,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.facecolor": "white",
})


def load_gdx(path: str) -> dict:
    log.info("Loading GDX: %s", path)
    data = gdxpds.to_dataframes(path)
    log.info("Available symbols: %s", sorted(data.keys()))
    return data


def extract_cost_components(data: dict, years: list) -> pd.DataFrame:
    ia = data["indicator_accounting"]
    log.info("indicator_accounting columns: %s", list(ia.columns))

    all_indicators = COST_COMPONENTS + ["SystemCost"]

    mask = (
        (ia["accNodesModel"].str.lower() == "global") &
        (ia["accYears"].astype(str).isin(years)) &
        (ia["indicator"].isin(all_indicators))
    )
    sub = ia.loc[mask, ["accYears", "indicator", "Value"]].copy()
    sub["accYears"] = sub["accYears"].astype(str)

    log.info("Rows extracted:\n%s", sub.to_string(index=False))

    wide = sub.pivot_table(
        index="accYears", columns="indicator", values="Value", aggfunc="sum"
    ).reset_index()
    wide.rename(columns={"accYears": "year"}, inplace=True)
    wide = wide.sort_values("year").reset_index(drop=True)

    for comp in COST_COMPONENTS:
        if comp not in wide.columns:
            log.warning("Component '%s' not found — filling with 0.", comp)
            wide[comp] = 0.0

    wide["components_sum"] = wide[COST_COMPONENTS].sum(axis=1)
    if "SystemCost" in wide.columns:
        wide["diff_pct"] = (
            (wide["components_sum"] - wide["SystemCost"]) / wide["SystemCost"] * 100
        ).round(2)
        log.info("Component sum vs SystemCost:\n%s",
                  wide[["year", "components_sum", "SystemCost", "diff_pct"]].to_string(index=False))

    return wide


def plot_stacked_cost(df: pd.DataFrame, out_dir: Path, fmt: str, dpi: int):
    if df.empty:
        log.error("No cost data to plot.")
        return

    years  = df["year"].tolist()
    x      = np.arange(len(years))
    bar_w  = 0.52

    # Wider figure to accommodate the external legend on the right
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    bottoms = np.zeros(len(years))

    for comp in COST_COMPONENTS:
        values = df[comp].fillna(0).values
        bars = ax.bar(
            x, values,
            bottom=bottoms,
            width=bar_w,
            color=COMPONENT_COLORS[comp],
            edgecolor="white",
            linewidth=0.5,
            label=COMPONENT_LABELS[comp],
            zorder=3,
        )
        for bar, val, bot in zip(bars, values, bottoms):
            if val > df["components_sum"].max() * 0.04:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bot + val / 2,
                    f"{val:,.1f}",
                    ha="center", va="center",
                    fontsize=7.5, color="white", fontweight="600",
                )
        bottoms = bottoms + values

    # Total label on top of each bar
    for i, total in enumerate(df["components_sum"].values):
        ax.text(
            x[i],
            total + df["components_sum"].max() * 0.012,
            f"{total:,.1f}",
            ha="center", va="bottom",
            fontsize=9, color="#222222", fontweight="600",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(years, fontsize=10)
    ax.set_xlabel("Year", labelpad=6)
    ax.set_ylabel("Total System Cost (M$)", labelpad=6)
    ax.set_title(
        "Total System Cost by Component — Pacific Island Countries\n"
        "(All Islands, Cumulative)",
        fontsize=11, fontweight="bold", pad=12,
    )

    ax.set_xlim(-0.5, len(years) - 0.5)
    ax.set_ylim(0, df["components_sum"].max() * 1.16)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    # ── Legend placed OUTSIDE the axes to the right — never covers bars ───────
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
        frameon=True,
        framealpha=0.95,
        edgecolor="#cccccc",
        fancybox=False,
    )

    # Shrink axes to leave room for legend; bbox_inches="tight" captures it all
    fig.subplots_adjust(right=0.78)

    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / f"system_cost_stacked.{fmt}"
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved -> %s", save_path)
    print(f"\nFigure saved to: {save_path}")


def main(gdx_path: str, out_dir: str, fmt: str, dpi: int, years: list, preloaded_data=None):
    # if preloaded_data is not None:
    #     data = preloaded_data
    # else:
    #     data = load_data(gdx_path)
    df   = extract_cost_components(data, years)

    if df.empty:
        log.error(
            "No rows matched. Check accNodesModel='global', "
            "indicators %s, years %s exist in indicator_accounting.",
            COST_COMPONENTS, years,
        )
        return

    plot_stacked_cost(df, Path(out_dir), fmt, dpi)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stacked bar chart of system cost components for Pacific islands.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--gdx",   default=GDX_PATH)
    parser.add_argument("--out",   default=OUTPUT_DIR)
    parser.add_argument("--fmt",   default="png", choices=["png", "pdf", "svg", "tiff"])
    parser.add_argument("--dpi",   default=300, type=int)
    parser.add_argument("--years", nargs="+", default=YEARS)
    args = parser.parse_args()

    main(args.gdx, args.out, args.fmt, args.dpi, args.years)
