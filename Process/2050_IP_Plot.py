# -*- coding: utf-8 -*-
"""
Created on Mon Dec 22 12:32:01 2025

@author: ajh287
"""

import gdxpds


#Load all symbols
data = gdxpds.to_dataframes(
r"C:\Local\remix-pic\REMix-Pacific_Island_Countries\Process\results\IP_2050_3.gdx"
)
print(data.keys())
el_Production_tech = ["DG", "PV_B", "NG_plant", "BG_B", "WindOnshore_B", "Hydro_B", "BG_N", "PV_N", "WindOnshore_N", "Wave_N", "WindOffshore_N", "Hydro_N"]
el_Storage_tech = ['Battery']
Water_production_tech= ['RO']
Water_storage_tech= ['H20_storage']
co_production_costs = ['DAC']
co_storage_costs =['co2_storage']
Heat_production_tech= ['cook_b', "Industry", "DW_LPG_converter", "DW_Electric_converter", "ST_N", "HP", "cook_el", "cook_LPG", "Industry_EL"]
Heat_storage_tech= ['THSS']
H2_production_tech= ["AEL_100"]
H2_production_tech= ['H2_storage']
Methanol_production_tech= ['Methanol_synthesis']
Methanol_production_tech= ["Methanol_storage"]
Ammonia_production_tech= ['Ammonia_synthesis']
Ammonia_storage_tech= ["Ammonia_storage"]
Ekerosene_tech= ["FTL"]
Ekerosene_tech= ["eKerosene_storage"]
Technologies_dep = ["DG","NG_plant", "BG_B", "BG_N", 'RO','DAC', 'cook_b' ,"Industry", "DW_LPG_converter", "DW_Electric_converter","HP", "cook_el", "cook_LPG","Industry_EL"]
All_techs = ["DG", "PV_B", "NG_plant", "BG_B", "WindOnshore_B", "Hydro_B", "BG_N", "PV_N", "WindOnshore_N", "Wave_N", "WindOffshore_N", "Hydro_N", 'RO', 'DAC', 'cook_b', "Industry", "DW_LPG_converter", "DW_Electric_converter", "ST_N", "HP", "cook_el", "cook_LPG", "Industry_EL","AEL_100", 'Methanol_synthesis','Ammonia_synthesis', "FTL"  ]
All_final_tech = ["BG_N","PV_N","WindOnshore_N", "Wave_N","WindOffshore_N", "ST_N","Industry_EL", "LDV_BF", "RO","Ammonia_synthesis", "DAC", "Methanol_synthesis", "HP", "FTL","AEL_100", "LDV_el", "HDV_el", "HDV_BF", "MDV_el", "MDV_BF", "Two_wheel_el", "Bus_el", "Marine_e", "Aviation_el", "Aviation_e", "cook_el", "cook_LPG", "Industry_EH", "DW_heat", "Dummy_Ammonia", "Dummy_Methanol", "ST_N_DW", "DW_Electric_converter"]
# # ##################################################################################################
el_Production_tech = [
    "DG", "PV_B", "NG_plant", "BG_B", "WindOnshore_B", "Hydro_B",
    "BG_N", "PV_N", "WindOnshore_N", "Wave_N", "WindOffshore_N", "Hydro_N"
]

el_Storage_tech = ["Battery"]

el_tech_all = el_Production_tech + el_Storage_tech
import pandas as pd

# Load table
ind = data["indicator_accounting_detailed"]
print(ind)

# Filter for island + year
ind_ci_2020 = ind[
    (ind["nodesModel"] == "CI_model") &
    (ind["years"] == "2020") &
    (ind["techs"].isin(el_tech_all)) &
    (ind["indicator"].isin(["Invest", "OMFix", "OMVar"]))
]

# Sum all available costs (missing indicators are naturally ignored)
part1_direct_costs = ind_ci_2020["Value"].sum()

print("PART 1 – Direct system costs (Invest + OMFix + OMVar):")
print(part1_direct_costs)
fuel_costs = {
    "Biomass": .032,   # USD/MWh_th or USD/unit (consistent with REMix)
    "NG": .025,
    "Diesel": .090
}
fuel_map = {
    "BG_B": "Biomass",
    "BG_N": "Biomass",
    "NG_plant": "NG",
    "DG": "Diesel"
}
# Load annual commodity balance
cb = data["commodity_balance_annual"]

# Filter for island + year + electricity technologies
cb_ci_2020 = cb[
    (cb["accNodesModel"] == "CI_model") &
    (cb["accYears"] == "2020") &
    (cb["balanceType"] == "net") &
    (cb["techs"].isin(fuel_map.keys())) &
    (cb["Value"] < 0)   # inputs only
]

# Compute fuel costs
fuel_cost_total = 0.0

for tech, fuel in fuel_map.items():
    if fuel not in fuel_costs:
        continue

    fuel_use = cb_ci_2020[
        (cb_ci_2020["techs"] == tech) &
        (cb_ci_2020["commodity"] == fuel) 
    ]["Value"].sum()

    # fuel_use is negative → take absolute
    fuel_cost = abs(fuel_use) * fuel_costs[fuel]
    fuel_cost_total += fuel_cost

    print(f"{tech}: {abs(fuel_use)} units × {fuel_costs[fuel]} = {fuel_cost}")

print("\nPART 2 – Indirect fuel/input costs:")
print(fuel_cost_total)
# # ################################################
elec_consuming_techs = (
    Water_production_tech
    + co_production_costs
    + Heat_production_tech
    + H2_production_tech
    + Methanol_production_tech
    + Ammonia_production_tech
)

# remove duplicates
elec_consuming_techs = list(set(elec_consuming_techs))
cb = data["commodity_balance_annual"]

final_demand = cb[
    (cb["accNodesModel"] == "CI_model") &
    (cb["accYears"] == "2020") &
    (cb["balanceType"] == "net") &
    (cb["techs"] == "Demand") &
    (cb["commodity"] == "Elec")
]["Value"].sum()

final_demand = abs(final_demand)

print("Final electricity demand:")
print(final_demand)
converter_elec_demand = cb[
    (cb["accNodesModel"] == "CI_model") &
    (cb["accYears"] == "2020") &
    (cb["balanceType"] == "net") &
    (cb["commodity"] == "Elec") &
    (cb["techs"].isin(All_final_tech)) &
    (cb["Value"] < 0)   # consumption only
]["Value"].sum()

converter_elec_demand = abs(converter_elec_demand)

print("Electricity consumed by converters (no storage):")
print(converter_elec_demand)
total_electricity_demand = final_demand + converter_elec_demand

print("\nTOTAL ELECTRICITY DEMAND (CI_model, 2020):")
print(total_electricity_demand)
# TOTAL electricity system cost
total_electricity_cost = part1_direct_costs + fuel_cost_total

print("TOTAL ELECTRICITY COST (CI_model, 2020):")
print(total_electricity_cost)

LCOE_elec = total_electricity_cost / total_electricity_demand

print("\nLCOE Electricity (USD per unit Elec): LCOE_elec")
###########run from here#########################################################
el_Production_tech = ["DG", "PV_B", "NG_plant", "BG_B", "WindOnshore_B", "Hydro_B", "BG_N", "PV_N", "WindOnshore_N", "Wave_N", "WindOffshore_N", "Hydro_N"]
el_Storage_tech = ['Battery']
Water_production_tech= ['RO']
Water_storage_tech= ['H20_storage']
co_production_costs = ['DAC']
co_storage_costs =['co2_storage']
Heat_production_tech= ['cook_b', "Industry", "DW_LPG_converter", "DW_Electric_converter", "ST_N", "HP", "cook_el", "cook_LPG", "Industry_EL"]
Heat_storage_tech= ['THSS']
H2_production_tech= ["AEL_100"]
H2_production_tech= ['H2_storage']
Methanol_production_tech= ['Methanol_synthesis']
Methanol_production_tech= ["Methanol_storage"]
Ammonia_production_tech= ['Ammonia_synthesis']
Ammonia_storage_tech= ["Ammonia_storage"]
Ekerosene_tech= ["FTL"]
Ekerosene_tech= ["eKerosene_storage"]
Technologies_dep = ["DG","NG_plant", "BG_B", "BG_N", 'RO','DAC', 'cook_b' ,"Industry", "DW_LPG_converter", "DW_Electric_converter","HP", "cook_el", "cook_LPG","Industry_EL"]
All_techs = ["DG", "PV_B", "NG_plant", "BG_B", "WindOnshore_B", "Hydro_B", "BG_N", "PV_N", "WindOnshore_N", "Wave_N", "WindOffshore_N", "Hydro_N", 'RO', 'DAC', 'cook_b', "Industry", "DW_LPG_converter", "DW_Electric_converter", "ST_N", "HP", "cook_el", "cook_LPG", "Industry_EL","AEL_100", 'Methanol_synthesis','Ammonia_synthesis', "FTL"]
All_final_commodity = [
    "T_Two_wheel_th",
    "T_Two_wheel_el",
    "T_MDV_th",
    "T_MDV_el",
    "T_MDV_BF",
    "T_Marine_f_th",
    "T_LDV_th",
    "T_LDV_el",
    "T_LDV_BF",
    "T_Industry_EH",
    "T_HDV_th",
    "T_HDV_el",
    "T_HDV_BF",
    "T_Bus_th",
    "T_Bus_el",
    "T_Aviation_th",
    "T_Aviation_el",
    "Heat_industry",
    "Elec",
    "eKerosene",
    "Dummy_EL",
    "DHW_LPG",
    "DHW_el", "T_Marine_th"
]

# # ##################################################################################################
el_Production_tech = [
    "DG", "PV_B", "NG_plant", "BG_B", "WindOnshore_B", "Hydro_B",
    "BG_N", "PV_N", "WindOnshore_N", "Wave_N", "WindOffshore_N", "Hydro_N"
]

el_Storage_tech = ["Battery"]

el_tech_all = el_Production_tech + el_Storage_tech
import pandas as pd
pic_models = [
    "CI_model",
    "FJ_model",
    "FSM_model",
    "KB_model",
    "MI_model",
    "NU_model",
    "NE_model",
    "PU_model",
    "PNG_model",
    "SA_model",
    "SI_model",
    "TA_model",
    "TU_model",
    "VU_model"
]
def compute_LCOE_elec(island, year):
    # ---------- PART 1: Direct costs ----------
    ind = data["indicator_accounting_detailed"]

    ind_filt = ind[
        (ind["nodesModel"] == island) &
        (ind["years"] == year) &
        (ind["techs"].isin(el_tech_all)) &
        (ind["indicator"].isin(["Invest", "OMFix", "OMVar"]))
    ]

    part1_direct_costs = ind_filt["Value"].sum()

    # ---------- PART 2: Fuel / input costs ----------
    cb = data["commodity_balance_annual"]

    cb_fuel = cb[
        (cb["accNodesModel"] == island) &
        (cb["accYears"] == year) &
        (cb["balanceType"] == "net") &
        (cb["techs"].isin(fuel_map.keys())) &
        (cb["Value"] < 0)
    ]

    fuel_cost_total = 0.0
    for tech, fuel in fuel_map.items():
        fuel_use = cb_fuel[
            (cb_fuel["techs"] == tech) &
            (cb_fuel["commodity"] == fuel)
        ]["Value"].sum()

        fuel_cost_total += abs(fuel_use) * fuel_costs.get(fuel, 0.0)

    total_electricity_cost = part1_direct_costs + fuel_cost_total

    # ---------- DEMAND PART ----------
    final_demand = cb[
        (cb["accNodesModel"] == island) &
        (cb["accYears"] == year) &
        (cb["balanceType"] == "net") &
        (cb["techs"] == "Demand") &
        (cb["commodity"] == "Elec")
    ]["Value"].sum()

    final_demand = abs(final_demand)

    converter_demand = cb[
        (cb["accNodesModel"] == island) &
        (cb["accYears"] == year) &
        (cb["balanceType"] == "net") &
        (cb["commodity"] == "Elec") &
        (cb["techs"].isin(All_final_tech)) &
        (cb["Value"] < 0)
    ]["Value"].sum()

    converter_demand = abs(converter_demand)

    total_electricity_demand = final_demand + converter_demand

    if total_electricity_demand == 0:
        return {
            "LCOE_Electricity": None,
            "Total_Elec_Demand": 0.0,
            "Total_Elec_Cost": total_electricity_cost
        }

    return {
        "LCOE_Electricity": total_electricity_cost / total_electricity_demand,
        "Total_Elec_Demand": total_electricity_demand,
        "Total_Elec_Cost": total_electricity_cost
    }

# # ##########################
years = ["2020", "2030", "2040", "2050"]
results = []

for island in pic_models:
    for year in years:
        out = compute_LCOE_elec(island, year)

        results.append({
            "Island": island,
            "Year": int(year),
            "LCOE_Electricity": out["LCOE_Electricity"],
            "Total_Electricity_Demand": out["Total_Elec_Demand"],
            "Total_Electricity_Cost": out["Total_Elec_Cost"]
        })

lcoe_df = pd.DataFrame(results)

print(lcoe_df)
import pandas as pd

# Assuming your DataFrame is lcoe_df
lcoe_df.to_excel("LCOE_results_2020.xlsx", index=False)
#################################################################################
import pandas as pd

# Heat demand commodities
Heat_demand_co = [
    "Heat_cooking", "T_cook_LPG", "T_cook_el",
    "Heat_industry", "T_Industry_EH", "DHW_el", "DHW_LPG", "Heat"
]

# Heat production techs
Heat_production_tech = [
    'cook_b', "Industry", "DW_LPG_converter", "DW_Electric_converter",
    "ST_N", "HP", "cook_el", "cook_LPG", "Industry_EL"
]

# Heat storage techs
Heat_storage_tech = ['THSS']

# Fuel costs
heat_fuel_costs = {
    "Biomass": 0.032,
    "Diesel": 0.090,
    "LPG": 0.065,
    "Elec": 0  # use electricity LCOE
}

# Map techs to fuels
heat_fuel_map = {
    "cook_b": "Biomass",
    "Industry": "Diesel",
    "cook_LPG": "LPG",
    "DW_LPG_converter": "LPG",
    "DW_Electric_converter": "Elec",
    "HP": "Elec",
    "cook_el": "Elec",
    "Industry_EL": "Elec"
}

import pandas as pd

# -------------------------
# Prepare electricity LCOE lookup
# -------------------------
LCOE_elec_lookup = {(row['Island'], row['Year']): row['LCOE_Electricity']
                    for _, row in lcoe_df.iterrows()}

# -------------------------
# Compute LCOHeat dynamically using LCOE
# -------------------------
def compute_LCOHeat(island, year, LCOE_elec_lookup):
    cb = data["commodity_balance_annual"]
    ind = data["indicator_accounting_detailed"]

    # ---------- PART 1: Direct costs ----------
    heat_techs = Heat_production_tech + Heat_storage_tech
    ind_filt = ind[
        (ind["nodesModel"] == island) &
        (ind["years"] == year) &
        (ind["techs"].isin(heat_techs)) &
        (ind["indicator"].isin(["Invest", "OMFix", "OMVar"]))
    ]
    part1_direct_costs = ind_filt["Value"].sum()

    # ---------- PART 2: Fuel / electricity costs ----------
    cb_heat = cb[
        (cb["accNodesModel"] == island) &
        (cb["accYears"] == year) &
        (cb["balanceType"] == "net") &
        (cb["techs"].isin(heat_fuel_map.keys())) &
        (cb["Value"] < 0)
    ]
    fuel_cost_total = 0.0
    for tech, fuel in heat_fuel_map.items():
        fuel_use = cb_heat[
            (cb_heat["techs"] == tech) &
            (cb_heat["commodity"] == fuel)
        ]["Value"].sum()

        if fuel_use == 0:
            continue

        if fuel == "Elec":
            # Get LCOE dynamically; fallback to 0 if not available
            elec_cost = LCOE_elec_lookup.get((island, year))
            if elec_cost is None:
                # Try computing it dynamically if missing
                elec_data = compute_LCOE_elec(island, year)
                elec_cost = elec_data["LCOE_Electricity"] or 0.0
                # Update the lookup for future use
                LCOE_elec_lookup[(island, year)] = elec_cost
            fuel_cost_total += abs(fuel_use) * elec_cost
        else:
            fuel_cost_total += abs(fuel_use) * heat_fuel_costs.get(fuel, 0.0)

    total_heat_cost = part1_direct_costs + fuel_cost_total

    # ---------- PART 3: Heat demand ----------
    cb_heat_demand = cb[
        (cb["accNodesModel"] == island) &
        (cb["accYears"] == year) &
        (cb["balanceType"] == "net") &
        (cb["commodity"].isin(Heat_demand_co)) &
        (cb["techs"] == "Demand") &
        (cb["Value"] < 0)
    ]["Value"].sum()
    cb_heat_demand = abs(cb_heat_demand)

    converter_demand = cb[
        (cb["accNodesModel"] == island) &
        (cb["accYears"] == year) &
        (cb["balanceType"] == "net") &
        (cb["commodity"] == "Heat") &
        (cb["techs"] == "DAC") &
        (cb["Value"] < 0)
    ]["Value"].sum()
    converter_demand = abs(converter_demand)

    total_heat_demand = cb_heat_demand + converter_demand

    if total_heat_demand == 0:
        return {
            "LCOHeat": None,
            "Total_Heat_Demand": 0.0,
            "Total_Heat_Cost": total_heat_cost
        }

    return {
        "LCOHeat": total_heat_cost / total_heat_demand,
        "Total_Heat_Demand": total_heat_demand,
        "Total_Heat_Cost": total_heat_cost
    }

# -------------------------
# Compute heat results for all islands and years
# -------------------------
heat_results = []

for island in pic_models:
    for year in years:
        out = compute_LCOHeat(island, year, LCOE_elec_lookup)
        heat_results.append({
            "Island": island,
            "Year": int(year),
            "LCOHeat": out["LCOHeat"],
            "Total_Heat_Demand": out["Total_Heat_Demand"],
            "Total_Heat_Cost": out["Total_Heat_Cost"]
        })

heat_df = pd.DataFrame(heat_results)

# -------------------------
# Export to Excel
# -------------------------
output_path = "LCOE_and_LCOHeat_PICs.xlsx"
with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    lcoe_df.to_excel(writer, sheet_name="LCOE_Electricity", index=False)
    heat_df.to_excel(writer, sheet_name="LCOHeat", index=False)

print(f"Excel file successfully written to: {output_path}")
#---------------------LCOEnergy------------------------------------
#######################################################################
def compute_LCOE_elec(island, year):
    # ---------- PART 1: Direct costs ----------
    ind = data["indicator_accounting"]

    ind_filt = ind[
        (ind["accNodesModel"] == island) &
        (ind["accYears"] == year) &
        (ind["indicator"] == "SystemCost")
    ]

    part1_direct_costs = ind_filt["Value"].sum()


    total_electricity_cost = part1_direct_costs 

    # ---------- DEMAND PART ----------

    converter_demand = cb[
        (cb["accNodesModel"] == island) &
        (cb["accYears"] == year) &
        (cb["balanceType"] == "net") &
        (cb["commodity"].isin(All_final_commodity)) &
        (cb["techs"] == "Demand") &
        (cb["Value"] < 0)
    ]["Value"].sum()

    converter_demand = abs(converter_demand)

    total_electricity_demand = converter_demand

    if total_electricity_demand == 0:
        return {
            "LCOE_Electricity": None,
            "Total_Elec_Demand": 0.0,
            "Total_Elec_Cost": total_electricity_cost
        }

    return {
        "LCOE_Electricity": total_electricity_cost / total_electricity_demand,
        "Total_Elec_Demand": total_electricity_demand,
        "Total_Elec_Cost": total_electricity_cost
    }

# # ##########################
years = ["2020", "2030", "2040", "2050"]
results = []

for island in pic_models:
    for year in years:
        out = compute_LCOE_elec(island, year)

        results.append({
            "Island": island,
            "Year": int(year),
            "LCOE_Electricity": out["LCOE_Electricity"],
            "Total_Electricity_Demand": out["Total_Elec_Demand"],
            "Total_Electricity_Cost": out["Total_Elec_Cost"]
        })

lcoE_df = pd.DataFrame(results)

print(lcoE_df)



##############################################################################pure_water#############
import pandas as pd

# Water technologies
Water_production_tech = ['RO']
Water_storage_tech = ['H20_storage']

# Water technologies
Water_production_tech = ["RO"]
Water_storage_tech = ["H20_storage"]

def compute_LCOWater(island, year, LCOE_elec_lookup):
    cb = data["commodity_balance_annual"]
    ind = data["indicator_accounting_detailed"]

    # ---------- PART 1: Direct costs ----------
    water_techs = Water_production_tech + Water_storage_tech
    ind_filt = ind[
        (ind["nodesModel"] == island) &
        (ind["years"] == year) &
        (ind["techs"].isin(water_techs)) &
        (ind["indicator"].isin(["Invest", "OMFix", "OMVar"]))
    ]

    part1_direct_costs = ind_filt["Value"].sum()

    # ---------- PART 2: Electricity cost (ROBUST) ----------
    cb_water = cb[
        (cb["accNodesModel"] == island) &
        (cb["accYears"] == year) &
        (cb["balanceType"] == "net") &
        (cb["techs"].isin(Water_production_tech)) &
        (cb["commodity"] == "Elec") &
        (cb["Value"] < 0)
    ]

    elec_use = abs(cb_water["Value"].sum())

    elec_cost = LCOE_elec_lookup.get((island, year))
    if elec_cost is None or pd.isna(elec_cost):
        elec_data = compute_LCOE_elec(island, year)
        elec_cost = elec_data["LCOE_Electricity"] or 0.0
        LCOE_elec_lookup[(island, year)] = elec_cost

    elec_cost_total = elec_use * elec_cost

    total_water_cost = part1_direct_costs + elec_cost_total

    # ---------- PART 3: Water demand ----------
    cb_water_demand = cb[
        (cb["accNodesModel"] == island) &
        (cb["accYears"] == year) &
        (cb["balanceType"] == "net") &
        (cb["commodity"] == "Pure_water") &
        (cb["techs"].isin(Water_production_tech)) &
        (cb["Value"] > 0)
    ]

    water_demand = cb_water_demand["Value"].sum()

    if water_demand == 0:
        return {
            "LCOWater": None,
            "Total_Water_Demand": 0.0,
            "Total_Water_Cost": total_water_cost
        }

    return {
        "LCOWater": total_water_cost / water_demand,
        "Total_Water_Demand": water_demand,
        "Total_Water_Cost": total_water_cost
    }


# Make a dictionary for LCOE electricity
LCOE_elec_lookup = {(row['Island'], row['Year']): row['LCOE_Electricity']
                    for _, row in lcoe_df.iterrows()}

# Compute for all islands and years
years = ["2020", "2030", "2040", "2050"]
water_results = []

for island in pic_models:
    for year in years:
        out = compute_LCOWater(island, year, LCOE_elec_lookup)
        water_results.append({
            "Island": island,
            "Year": int(year),
            "LCOWater": out["LCOWater"],
            "Total_Water_Demand": out["Total_Water_Demand"],
            "Total_Water_Cost": out["Total_Water_Cost"]
        })

water_df = pd.DataFrame(water_results)
print(water_df)
#######################################H2########################################
# H2 technologies
H2_production_tech = ["AEL_100"]
H2_storage_tech = ["H2_storage"]

def compute_LCOH2(island, year, LCOE_elec_lookup, LCOWater_lookup):
    cb = data["commodity_balance_annual"]
    ind = data["indicator_accounting_detailed"]

    # ---------- PART 1: Direct costs ----------
    h2_techs = H2_production_tech + H2_storage_tech
    ind_filt = ind[
        (ind["nodesModel"] == island) &
        (ind["years"] == year) &
        (ind["techs"].isin(h2_techs)) &
        (ind["indicator"].isin(["Invest", "OMFix", "OMVar"]))
    ]
    part1_direct_costs = ind_filt["Value"].sum()

    # ---------- PART 2a: Electricity cost ----------
    cb_elec = cb[
        (cb["accNodesModel"] == island) &
        (cb["accYears"] == year) &
        (cb["techs"].isin(H2_production_tech)) &
        (cb["balanceType"] == "net") &
        (cb["commodity"] == "Elec") &
        (cb["Value"] < 0)
    ]
    elec_use = abs(cb_elec["Value"].sum())
    elec_cost = LCOE_elec_lookup.get((island, year))
    if elec_cost is None or pd.isna(elec_cost):
        elec_data = compute_LCOE_elec(island, year)
        elec_cost = elec_data["LCOE_Electricity"] or 0.0
        LCOE_elec_lookup[(island, year)] = elec_cost
    elec_total_cost = elec_use * elec_cost

    # ---------- PART 2b: Water cost ----------
    cb_water = cb[
        (cb["accNodesModel"] == island) &
        (cb["accYears"] == year) &
        (cb["techs"].isin(H2_production_tech)) &
        (cb["balanceType"] == "net") &
        (cb["commodity"] == "Pure_water") &
        (cb["Value"] < 0)
    ]
    water_use = abs(cb_water["Value"].sum())
    water_cost = LCOWater_lookup.get((island, year), 0.0)
    water_total_cost = water_use * water_cost

    total_h2_cost = part1_direct_costs + elec_total_cost + water_total_cost

    # ---------- PART 3: Hydrogen demand ----------
    cb_h2_demand = cb[
        (cb["accNodesModel"] == island) &
        (cb["accYears"] == year) &
        (cb["techs"].isin(All_techs)) &
        (cb["balanceType"] == "net") &
        (cb["commodity"] == "Hydrogen") &
        (cb["Value"] < 0)
    ]
    h2_demand = abs(cb_h2_demand["Value"].sum())

    if h2_demand == 0:
        return {
            "LCOH2": None,
            "Total_H2_Demand": 0.0,
            "Total_H2_Cost": total_h2_cost
        }

    return {
        "LCOH2": total_h2_cost / h2_demand,
        "Total_H2_Demand": h2_demand,
        "Total_H2_Cost": total_h2_cost
    }

# Make dictionaries for input costs
LCOE_elec_lookup = {(row['Island'], row['Year']): row['LCOE_Electricity'] for _, row in lcoe_df.iterrows()}
LCOWater_lookup = {(row['Island'], row['Year']): row['LCOWater'] for _, row in water_df.iterrows()}

# Compute LCOH2 for all islands and years
years = ["2020", "2030", "2040", "2050"]
h2_results = []

for island in pic_models:
    for year in years:
        out = compute_LCOH2(island, year, LCOE_elec_lookup, LCOWater_lookup)
        h2_results.append({
            "Island": island,
            "Year": int(year),
            "LCOH2": out["LCOH2"],
            "Total_H2_Demand": out["Total_H2_Demand"],
            "Total_H2_Cost": out["Total_H2_Cost"]
        })

h2_df = pd.DataFrame(h2_results)
print(h2_df)
#############################################################################
Ammonia_production_tech = ["Ammonia_synthesis"]
Ammonia_storage_tech = ["Ammonia_storage"]

def compute_LCOAmmonia(island, year, LCOE_elec_lookup, LCOH2_lookup):
    """
    Compute Levelized Cost of Ammonia (LCOAmmonia)
    Uses electricity (LCOE) and hydrogen (LCOH2) as input costs.
    """

    cb = data["commodity_balance_annual"]
    ind = data["indicator_accounting_detailed"]

    # ---------- PART 1: Direct costs ----------
    ammo_techs = Ammonia_production_tech + Ammonia_storage_tech

    ind_filt = ind[
        (ind["nodesModel"] == island) &
        (ind["years"] == year) &
        (ind["techs"].isin(ammo_techs)) &
        (ind["indicator"].isin(["Invest", "OMFix", "OMVar"]))
    ]

    part1_direct_costs = ind_filt["Value"].sum()

    # ---------- PART 2: Input costs ----------

    # --- Electricity input ---
    cb_elec = cb[
        (cb["accNodesModel"] == island) &
        (cb["accYears"] == year) &
        (cb["balanceType"] == "net") &
        (cb["techs"].isin(Ammonia_production_tech)) &
        (cb["commodity"] == "Elec") &
        (cb["Value"] < 0)
    ]

    elec_use = abs(cb_elec["Value"].sum())

    elec_cost = LCOE_elec_lookup.get((island, year))
    if elec_cost is None or pd.isna(elec_cost):
        elec_data = compute_LCOE_elec(island, year)
        elec_cost = elec_data["LCOE_Electricity"] or 0.0
        LCOE_elec_lookup[(island, year)] = elec_cost

    elec_total_cost = elec_use * elec_cost

    # --- Hydrogen input (ROBUST, same as water logic) ---
    cb_h2 = cb[
        (cb["accNodesModel"] == island) &
        (cb["accYears"] == year) &
        (cb["balanceType"] == "net") &
        (cb["techs"].isin(Ammonia_production_tech)) &
        (cb["commodity"] == "Hydrogen") &
        (cb["Value"] < 0)
    ]

    h2_use = abs(cb_h2["Value"].sum())

    h2_cost = LCOH2_lookup.get((island, year))
    if h2_cost is None or pd.isna(h2_cost):
        h2_data = compute_LCOH2(island, year, LCOE_elec_lookup, LCOWater_lookup)
        h2_cost = h2_data["LCOH2"] or 0.0
        LCOH2_lookup[(island, year)] = h2_cost

    h2_total_cost = h2_use * h2_cost

    # ---------- TOTAL COST ----------
    total_ammonia_cost = (
        part1_direct_costs +
        elec_total_cost +
        h2_total_cost
    )

    # ---------- PART 3: Ammonia demand ----------
    cb_ammonia_demand = cb[
        (cb["accNodesModel"] == island) &
        (cb["accYears"] == year) &
        (cb["balanceType"] == "net") &
        (cb["techs"] == "Dummy_Ammonia") &
        (cb["commodity"] == "Ammonia") &
        (cb["Value"] < 0)
    ]

    ammonia_demand = abs(cb_ammonia_demand["Value"].sum())

    if ammonia_demand == 0:
        return {
            "LCOAmmonia": None,
            "Total_Ammonia_Demand": 0.0,
            "Total_Ammonia_Cost": total_ammonia_cost
        }

    return {
        "LCOAmmonia": total_ammonia_cost / ammonia_demand,
        "Total_Ammonia_Demand": ammonia_demand,
        "Total_Ammonia_Cost": total_ammonia_cost
    }


LCOE_elec_lookup = {
    (row["Island"], row["Year"]): row["LCOE_Electricity"]
    for _, row in lcoe_df.iterrows()}

LCOH2_lookup = {
    (row["Island"], row["Year"]): row["LCOH2"]
    for _, row in h2_df.iterrows()
  }
years = ["2020", "2030", "2040", "2050"]
ammonia_results = []

for island in pic_models:
    for year in years:
        out = compute_LCOAmmonia(island, year, LCOE_elec_lookup, LCOH2_lookup)
        ammonia_results.append({
            "Island": island,
            "Year": int(year),
            "LCOAmmonia": out["LCOAmmonia"],
            "Total_Ammonia_Demand": out["Total_Ammonia_Demand"],
            "Total_Ammonia_Cost": out["Total_Ammonia_Cost"]
        })

ammonia_df = pd.DataFrame(ammonia_results)
print(ammonia_df)
################co2 costs##########################################################
co_production_costs = ["DAC"]
co_storage_costs = ["co2_storage"]

def compute_LCOCO2(island, year, LCOE_elec_lookup, LCOHeat_lookup):
    """
    Compute Levelized Cost of CO2 (LCOCO2) for a given island and year.
    Uses electricity LCOE and LCOHeat for input energy costs (ROBUST).
    """

    cb = data["commodity_balance_annual"]
    ind = data["indicator_accounting_detailed"]

    # ---------- PART 1: Direct costs ----------
    co_techs = co_production_costs + co_storage_costs

    ind_filt = ind[
        (ind["nodesModel"] == island) &
        (ind["years"] == year) &
        (ind["techs"].isin(co_techs)) &
        (ind["indicator"].isin(["Invest", "OMFix", "OMVar"]))
    ]

    part1_direct_costs = ind_filt["Value"].sum()

    # ---------- PART 2: Input energy costs (ROBUST) ----------

    # Electricity consumption
    cb_elec = cb[
        (cb["accNodesModel"] == island) &
        (cb["accYears"] == year) &
        (cb["techs"].isin(co_production_costs)) &
        (cb["balanceType"] == "net") &
        (cb["commodity"] == "Elec") &
        (cb["Value"] < 0)
    ]

    elec_use = abs(cb_elec["Value"].sum())

    elec_cost = LCOE_elec_lookup.get((island, year))
    if elec_cost is None or pd.isna(elec_cost):
        elec_data = compute_LCOE_elec(island, year)
        elec_cost = elec_data["LCOE_Electricity"] or 0.0
        LCOE_elec_lookup[(island, year)] = elec_cost

    elec_cost_total = elec_use * elec_cost

    # Heat consumption
    cb_heat = cb[
        (cb["accNodesModel"] == island) &
        (cb["accYears"] == year) &
        (cb["techs"].isin(co_production_costs)) &
        (cb["balanceType"] == "net") &
        (cb["commodity"] == "Heat") &
        (cb["Value"] < 0)
    ]

    heat_use = abs(cb_heat["Value"].sum())

    heat_cost = LCOHeat_lookup.get((island, year))
    if heat_cost is None or pd.isna(heat_cost):
        heat_data = compute_LCOHeat(island, year, LCOE_elec_lookup)
        heat_cost = heat_data["LCOHeat"] or 0.0
        LCOHeat_lookup[(island, year)] = heat_cost

    heat_cost_total = heat_use * heat_cost

    total_co_cost = (
        part1_direct_costs
        + elec_cost_total
        + heat_cost_total
    )

    # ---------- PART 3: CO2 production (functional demand) ----------
    cb_co_demand = cb[
        (cb["accNodesModel"] == island) &
        (cb["accYears"] == year) &
        (cb["techs"] == "DAC") &
        (cb["balanceType"] == "net") &
        (cb["commodity"] == "co") &
        (cb["Value"] > 0)
    ]

    co_demand = cb_co_demand["Value"].sum()

    if co_demand == 0:
        return {
            "LCOCO2": None,
            "Total_CO_Demand": 0.0,
            "Total_CO_Cost": total_co_cost
        }

    return {
        "LCOCO2": total_co_cost / co_demand,
        "Total_CO_Demand": co_demand,
        "Total_CO_Cost": total_co_cost
    }

LCOHeat_lookup = {
    (row["Island"], row["Year"]): row["LCOHeat"]
    for _, row in heat_df.iterrows()
}
years = ["2020", "2030", "2040", "2050"]
co_results = []

for island in pic_models:
    for year in years:
        out = compute_LCOCO2(
            island,
            year,
            LCOE_elec_lookup,
            LCOHeat_lookup
        )
        co_results.append({
            "Island": island,
            "Year": int(year),
            "LCOCO2": out["LCOCO2"],
            "Total_CO_Demand": out["Total_CO_Demand"],
            "Total_CO_Cost": out["Total_CO_Cost"]
        })

co_df = pd.DataFrame(co_results)
print(co_df)
################Methanol####################################
# Methanol technologies
Methanol_production_tech = ["Methanol_synthesis"]
Methanol_storage_tech = ["Methanol_storage"]

def compute_LCOMethanol(island, year, LCOH2_lookup, LCOCO2_lookup):
    """
    Compute Levelized Cost of Methanol (LCOMethanol) for a given island and year.
    Uses LCOH2 and LCOCO2 for input feedstock costs (ROBUST).
    """

    cb = data["commodity_balance_annual"]
    ind = data["indicator_accounting_detailed"]

    # ---------- PART 1: Direct costs ----------
    methanol_techs = Methanol_production_tech + Methanol_storage_tech

    ind_filt = ind[
        (ind["nodesModel"] == island) &
        (ind["years"] == year) &
        (ind["techs"].isin(methanol_techs)) &
        (ind["indicator"].isin(["Invest", "OMFix", "OMVar"]))
    ]

    part1_direct_costs = ind_filt["Value"].sum()

    # ---------- PART 2: Input feedstock costs (ROBUST) ----------

    # ----- Hydrogen consumption -----
    cb_h2 = cb[
        (cb["accNodesModel"] == island) &
        (cb["accYears"] == year) &
        (cb["techs"].isin(Methanol_production_tech)) &
        (cb["balanceType"] == "net") &
        (cb["commodity"] == "Hydrogen") &
        (cb["Value"] < 0)
    ]

    h2_use = abs(cb_h2["Value"].sum())

    h2_cost = LCOH2_lookup.get((island, year))
    if h2_cost is None or pd.isna(h2_cost):
        h2_data = compute_LCOH2(island, year, LCOE_elec_lookup, LCOWater_lookup)
        h2_cost = h2_data["LCOH2"] or 0.0
        LCOH2_lookup[(island, year)] = h2_cost

    h2_cost_total = h2_use * h2_cost

    # ----- CO / CO2 consumption -----
    cb_co = cb[
        (cb["accNodesModel"] == island) &
        (cb["accYears"] == year) &
        (cb["techs"].isin(Methanol_production_tech)) &
        (cb["balanceType"] == "net") &
        (cb["commodity"] == "co") &
        (cb["Value"] < 0)
    ]

    co_use = abs(cb_co["Value"].sum())

    co_cost = LCOCO2_lookup.get((island, year))
    if co_cost is None or pd.isna(co_cost):
        co_data = compute_LCOCO2(island, year, LCOE_elec_lookup, LCOHeat_lookup)
        co_cost = co_data["LCOCO2"] or 0.0
        LCOCO2_lookup[(island, year)] = co_cost

    co_cost_total = co_use * co_cost

    total_methanol_cost = (
        part1_direct_costs +
        h2_cost_total +
        co_cost_total
    )

    # ---------- PART 3: Methanol demand ----------
    cb_meoh_demand = cb[
        (cb["accNodesModel"] == island) &
        (cb["accYears"] == year) &
        (cb["techs"] == "Dummy_Methanol") &
        (cb["balanceType"] == "net") &
        (cb["commodity"] == "Methanol") &
        (cb["Value"] < 0)
    ]

    methanol_demand = abs(cb_meoh_demand["Value"].sum())

    if methanol_demand == 0:
        return {
            "LCOMethanol": None,
            "Total_Methanol_Demand": 0.0,
            "Total_Methanol_Cost": total_methanol_cost
        }

    return {
        "LCOMethanol": total_methanol_cost / methanol_demand,
        "Total_Methanol_Demand": methanol_demand,
        "Total_Methanol_Cost": total_methanol_cost
    }

LCOH2_lookup = {
    (row["Island"], row["Year"]): row["LCOH2"]
    for _, row in h2_df.iterrows()
}

LCOCO2_lookup = {
    (row["Island"], row["Year"]): row["LCOCO2"]
    for _, row in co_df.iterrows()
}
years = ["2020", "2030", "2040", "2050"]
methanol_results = []

for island in pic_models:
    for year in years:
        out = compute_LCOMethanol(
            island,
            year,
            LCOH2_lookup,
            LCOCO2_lookup
        )

        methanol_results.append({
            "Island": island,
            "Year": int(year),
            "LCOMethanol": out["LCOMethanol"],
            "Total_Methanol_Demand": out["Total_Methanol_Demand"],
            "Total_Methanol_Cost": out["Total_Methanol_Cost"]
        })

methanol_df = pd.DataFrame(methanol_results)
print(methanol_df)
##########################################EKerosene############################
Ekerosene_production_tech = ["FTL"]
Ekerosene_storage_tech = ["eKerosene_storage"]
def compute_LCOeKerosene(
    island,
    year,
    LCOH2_lookup,
    LCOCO2_lookup
):
    """
    Compute Levelized Cost of e-Kerosene (LCOeKerosene) for a given island and year.
    Uses LCOH2 and LCOCO2 as feedstock input costs.
    """

    cb = data["commodity_balance_annual"]
    ind = data["indicator_accounting_detailed"]

    # ---------- PART 1: Direct costs ----------
    ekero_techs = Ekerosene_production_tech + Ekerosene_storage_tech

    ind_filt = ind[
        (ind["nodesModel"] == island) &
        (ind["years"] == year) &
        (ind["techs"].isin(ekero_techs)) &
        (ind["indicator"].isin(["Invest", "OMFix", "OMVar"]))
    ]

    part1_direct_costs = ind_filt["Value"].sum()

    # ---------- PART 2: Input feedstock costs ----------

    # --- Hydrogen consumption ---
    cb_h2 = cb[
        (cb["accNodesModel"] == island) &
        (cb["accYears"] == year) &
        (cb["balanceType"] == "net") &
        (cb["techs"].isin(Ekerosene_production_tech)) &
        (cb["commodity"] == "Hydrogen") &
        (cb["Value"] < 0)
    ]

    h2_use = abs(cb_h2["Value"].sum())

    h2_lco = LCOH2_lookup.get((island, year))
    if h2_lco is None or pd.isna(h2_lco):
        h2_data = compute_LCOH2(island, year, LCOE_elec_lookup, LCOWater_lookup)
        h2_lco = h2_data["LCOH2"] or 0.0
        LCOH2_lookup[(island, year)] = h2_lco

    h2_cost_total = h2_use * h2_lco

    # --- CO / CO₂ consumption ---
    cb_co = cb[
        (cb["accNodesModel"] == island) &
        (cb["accYears"] == year) &
        (cb["balanceType"] == "net") &
        (cb["techs"].isin(Ekerosene_production_tech)) &
        (cb["commodity"] == "co") &
        (cb["Value"] < 0)
    ]

    co_use = abs(cb_co["Value"].sum())

    co_lco = LCOCO2_lookup.get((island, year))
    if co_lco is None or pd.isna(co_lco):
        co_data = compute_LCOCO2(island, year, LCOE_elec_lookup, LCOHeat_lookup)
        co_lco = co_data["LCOCO2"] or 0.0
        LCOCO2_lookup[(island, year)] = co_lco

    co_cost_total = co_use * co_lco

    total_ekerosene_cost = (
        part1_direct_costs +
        h2_cost_total +
        co_cost_total
    )

    # ---------- PART 3: e-Kerosene production (demand proxy) ----------
    cb_ekerosene_prod = cb[
        (cb["accNodesModel"] == island) &
        (cb["accYears"] == year) &
        (cb["balanceType"] == "net") &
        (cb["techs"].isin(Ekerosene_production_tech)) &
        (cb["commodity"] == "eKerosene") &
        (cb["Value"] > 0)
    ]

    ekerosene_demand = cb_ekerosene_prod["Value"].sum()

    if ekerosene_demand == 0:
        return {
            "LCOeKerosene": None,
            "Total_eKerosene_Demand": 0.0,
            "Total_eKerosene_Cost": total_ekerosene_cost
        }

    return {
        "LCOeKerosene": total_ekerosene_cost / ekerosene_demand,
        "Total_eKerosene_Demand": ekerosene_demand,
        "Total_eKerosene_Cost": total_ekerosene_cost
    }
ekerosene_results = []

for island in pic_models:
    for year in years:
        out = compute_LCOeKerosene(
            island,
            year,
            LCOH2_lookup,
            LCOCO2_lookup,
        )

        ekerosene_results.append({
            "Island": island,
            "Year": int(year),
            "LCOeKerosene": out["LCOeKerosene"],
            "Total_eKerosene_Demand": out["Total_eKerosene_Demand"],
            "Total_eKerosene_Cost": out["Total_eKerosene_Cost"]
        })

ekerosene_df = pd.DataFrame(ekerosene_results)
print(ekerosene_df)
output_file = "LCO_results_IP_2050.xlsx"

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    lcoe_df.to_excel(writer, sheet_name="LCOE_Electricity", index=False)
    heat_df.to_excel(writer, sheet_name="LCOHeat", index=False)
    water_df.to_excel(writer, sheet_name="LCOWater", index=False)
    h2_df.to_excel(writer, sheet_name="LCOH2", index=False)
    co_df.to_excel(writer, sheet_name="LCOCO2", index=False)
    ammonia_df.to_excel(writer, sheet_name="LCOAmmonia", index=False)
    methanol_df.to_excel(writer, sheet_name="LCOMethanol", index=False)
    ekerosene_df.to_excel(writer, sheet_name="LCOeKerosene", index=False)
    lcoE_df.to_excel(writer, sheet_name="LCOenergy", index=False)

print(f"Results written to {output_file}")
#################################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Load data
# -----------------------------
file_path = "LCO_results_IP_2050.xlsx"
df = pd.read_excel(file_path, sheet_name="LCOE_Electricity")

# Keep only needed columns
df = df[["Island", "Year", "LCOE_Electricity"]]

# Sort islands alphabetically (optional but clean)
df = df.sort_values(["Island", "Year"])

# -----------------------------
# Prepare data for plotting
# -----------------------------
islands = df["Island"].unique()
years = [2020, 2030, 2040, 2050]

x = np.arange(len(islands))          # island positions
bar_width = 0.18

# Color map for decades
year_colors = {
    2020: "#1f77b4",  # blue
    2030: "#ff7f0e",  # orange
    2040: "#2ca02c",  # green
    2050: "#d62728"   # red
}

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(16, 6))

for i, year in enumerate(years):
    year_values = (
        df[df["Year"] == year]
        .set_index("Island")
        .reindex(islands)["LCOE_Electricity"]
        .values
    ) * 1000


    plt.bar(
        x + i * bar_width,
        year_values,
        width=bar_width,
        label=str(year),
        color=year_colors[year]
    )

# -----------------------------
# Formatting
# -----------------------------
plt.xlabel("Island", fontsize=12)
plt.ylabel("Electricity cost (USD/MWh)", fontsize=12)
plt.title("LCOE Electricity by Island and Year", fontsize=14)

plt.xticks(
    x + bar_width * 1.5,
    islands,
    rotation=45,
    ha="right"
)

plt.legend(title="Year", ncol=4)
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()
##################################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Load data
# -----------------------------
file_path = "LCO_results_IP_2050.xlsx"
df = pd.read_excel(file_path, sheet_name="LCOHeat")

# Keep only needed columns
df = df[["Island", "Year", "LCOHeat"]]

# Sort islands alphabetically (optional but clean)
df = df.sort_values(["Island", "Year"])

# -----------------------------
# Prepare data for plotting
# -----------------------------
islands = df["Island"].unique()
years = [2020, 2030, 2040, 2050]

x = np.arange(len(islands))          # island positions
bar_width = 0.18

# Color map for decades
year_colors = {
    2020: "#1f77b4",  # blue
    2030: "#ff7f0e",  # orange
    2040: "#2ca02c",  # green
    2050: "#d62728"   # red
}

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(16, 6))

for i, year in enumerate(years):
    year_values = (
        df[df["Year"] == year]
        .set_index("Island")
        .reindex(islands)["LCOHeat"]
        .values
    ) * 1000


    plt.bar(
        x + i * bar_width,
        year_values,
        width=bar_width,
        label=str(year),
        color=year_colors[year]
    )

# -----------------------------
# Formatting
# -----------------------------
plt.xlabel("Island", fontsize=12)
plt.ylabel("Levelized cost of heat (USD/MWh)", fontsize=12)
plt.title("LCOH by Island and Year", fontsize=14)

plt.xticks(
    x + bar_width * 1.5,
    islands,
    rotation=45,
    ha="right"
)

#plt.legend(title="Year", ncol=4)
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()
#################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Load data
# -----------------------------
file_path = "LCO_results_IP_2050.xlsx"
df = pd.read_excel(file_path, sheet_name="LCOenergy")

# Keep only needed columns
df = df[["Island", "Year", "LCOenergy"]]

# Sort islands alphabetically (optional but clean)
df = df.sort_values(["Island", "Year"])

# -----------------------------
# Prepare data for plotting
# -----------------------------
islands = df["Island"].unique()
years = [2020, 2030, 2040, 2050]

x = np.arange(len(islands))          # island positions
bar_width = 0.18

# Color map for decades
year_colors = {
    2020: "#1f77b4",  # blue
    2030: "#ff7f0e",  # orange
    2040: "#2ca02c",  # green
    2050: "#d62728"   # red
}

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(16, 6))

for i, year in enumerate(years):
    year_values = (
        df[df["Year"] == year]
        .set_index("Island")
        .reindex(islands)["LCOenergy"]
        .values
    ) * 1000


    plt.bar(
        x + i * bar_width,
        year_values,
        width=bar_width,
        label=str(year),
        color=year_colors[year]
    )

# -----------------------------
# Formatting
# -----------------------------
plt.xlabel("Island", fontsize=12)
plt.ylabel("Levelized cost of energy (USD/MWh)", fontsize=12)
plt.title("LCOenergy by Island and Year", fontsize=14)

plt.xticks(
    x + bar_width * 1.5,
    islands,
    rotation=45,
    ha="right"
)

#plt.legend(title="Year", ncol=4)
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()
#############################################################

############################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Load data
# -----------------------------
file_path = "LCO_results_IP_2050.xlsx"
df = pd.read_excel(file_path, sheet_name="LCOH2")

# Keep only needed columns
df = df[["Island", "Year", "LCOH2"]]

# Sort islands alphabetically (optional but clean)
df = df.sort_values(["Island", "Year"])

# -----------------------------
# Prepare data for plotting
# -----------------------------
islands = df["Island"].unique()
years = [2020, 2030, 2040, 2050]

x = np.arange(len(islands))          # island positions
bar_width = 0.18

# Color map for decades
year_colors = {
    2020: "#1f77b4",  # blue
    2030: "#ff7f0e",  # orange
    2040: "#2ca02c",  # green
    2050: "#d62728"   # red
}

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(16, 6))

for i, year in enumerate(years):
    year_values = (
        df[df["Year"] == year]
        .set_index("Island")
        .reindex(islands)["LCOH2"]
        .values
    ) * 1000


    plt.bar(
        x + i * bar_width,
        year_values,
        width=bar_width,
        label=str(year),
        color=year_colors[year]
    )
    plt.axhline(
        y=81,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Reference: 100 USD/MWh"
    )

# -----------------------------
# Formatting
# -----------------------------
plt.xlabel("Island", fontsize=12)
plt.ylabel("Levelized cost of Hydrogen (USD/MWh)", fontsize=12)
plt.title("LCOH2 by Island and Year", fontsize=14)

plt.xticks(
    x + bar_width * 1.5,
    islands,
    rotation=45,
    ha="right"
)

#plt.legend(title="Year", ncol=4)
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()
###########################################################33
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Load data
# -----------------------------
file_path = "LCO_results_IP_2050.xlsx"
df = pd.read_excel(file_path, sheet_name="LCOAmmonia")

# Keep only needed columns
df = df[["Island", "Year", "LCOAmmonia"]]

# Sort islands alphabetically (optional but clean)
df = df.sort_values(["Island", "Year"])

# -----------------------------
# Prepare data for plotting
# -----------------------------
islands = df["Island"].unique()
years = [2020, 2030, 2040, 2050]

x = np.arange(len(islands))          # island positions
bar_width = 0.18

# Color map for decades
year_colors = {
    2020: "#1f77b4",  # blue
    2030: "#ff7f0e",  # orange
    2040: "#2ca02c",  # green
    2050: "#d62728"   # red
}

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(16, 6))

for i, year in enumerate(years):
    year_values = (
        df[df["Year"] == year]
        .set_index("Island")
        .reindex(islands)["LCOAmmonia"]
        .values
    ) * 1000


    plt.bar(
        x + i * bar_width,
        year_values,
        width=bar_width,
        label=str(year),
        color=year_colors[year]
    )
    plt.axhline(
        y=114,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Reference: 100 USD/MWh"
    )

# -----------------------------
# Formatting
# -----------------------------
plt.xlabel("Island", fontsize=12)
plt.ylabel("Levelized cost of Ammonia (USD/MWh)", fontsize=12)
plt.title("LCOAmmonia by Island and Year", fontsize=14)

plt.xticks(
    x + bar_width * 1.5,
    islands,
    rotation=45,
    ha="right"
)

#plt.legend(title="Year", ncol=4)
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()
################################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Load data
# -----------------------------
file_path = "LCO_results_IP_2050.xlsx"
df = pd.read_excel(file_path, sheet_name="LCOMethanol")

# Keep only needed columns
df = df[["Island", "Year", "LCOMethanol"]]

# Sort islands alphabetically (optional but clean)
df = df.sort_values(["Island", "Year"])

# -----------------------------
# Prepare data for plotting
# -----------------------------
islands = df["Island"].unique()
years = [2020, 2030, 2040, 2050]

x = np.arange(len(islands))          # island positions
bar_width = 0.18

# Color map for decades
year_colors = {
    2020: "#1f77b4",  # blue
    2030: "#ff7f0e",  # orange
    2040: "#2ca02c",  # green
    2050: "#d62728"   # red
}

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(16, 6))

for i, year in enumerate(years):
    year_values = (
        df[df["Year"] == year]
        .set_index("Island")
        .reindex(islands)["LCOMethanol"]
        .values
    ) * 1000


    plt.bar(
        x + i * bar_width,
        year_values,
        width=bar_width,
        label=str(year),
        color=year_colors[year]
    )
    plt.axhline(
        y=98,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Reference: 100 USD/MWh"
    )

# -----------------------------
# Formatting
# -----------------------------
plt.xlabel("Island", fontsize=12)
plt.ylabel("Levelized cost of Methanol (USD/MWh)", fontsize=12)
plt.title("LCOMethanol by Island and Year", fontsize=14)

plt.xticks(
    x + bar_width * 1.5,
    islands,
    rotation=45,
    ha="right"
)

#plt.legend(title="Year", ncol=4)
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()
##############################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Load data
# -----------------------------
file_path = "LCO_results_IP_2050.xlsx"
df = pd.read_excel(file_path, sheet_name="LCOeKerosene")

# Keep only needed columns
df = df[["Island", "Year", "LCOeKerosene"]]

# Sort islands alphabetically (optional but clean)
df = df.sort_values(["Island", "Year"])

# -----------------------------
# Prepare data for plotting
# -----------------------------
islands = df["Island"].unique()
years = [2020, 2030, 2040, 2050]

x = np.arange(len(islands))          # island positions
bar_width = 0.18

# Color map for decades
year_colors = {
    2020: "#1f77b4",  # blue
    2030: "#ff7f0e",  # orange
    2040: "#2ca02c",  # green
    2050: "#d62728"   # red
}

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(16, 6))

for i, year in enumerate(years):
    year_values = (
        df[df["Year"] == year]
        .set_index("Island")
        .reindex(islands)["LCOeKerosene"]
        .values
    ) * 1000


    plt.bar(
        x + i * bar_width,
        year_values,
        width=bar_width,
        label=str(year),
        color=year_colors[year]
    )
    plt.axhline(
        y=130,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Reference: 100 USD/MWh"
    )

# -----------------------------
# Formatting
# -----------------------------
plt.xlabel("Island", fontsize=12)
plt.ylabel("Levelized cost of ekerosene (USD/MWh)", fontsize=12)
plt.title("LCOKerosene by Island and Year", fontsize=14)

plt.xticks(
    x + bar_width * 1.5,
    islands,
    rotation=45,
    ha="right"
)

#plt.legend(title="Year", ncol=4)
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()
########################################
import pandas as pd
import matplotlib.pyplot as plt
tech_colors = {
    "DG": "gray",

    "PV_B": "#FFD700",   # yellow
    "PV_N": "#FFD700",

    "BG_B": "#2ca02c",   # green
    "BG_N": "#2ca02c",

    "WindOnshore_B": "#ff7f0e",  # orange
    "WindOnshore_N": "#ff7f0e",

    "Hydro_B": "#1f77b4",  # blue
    "Hydro_N": "#1f77b4",

    "Wave_N": "#87CEFA",  # light blue

    "WindOffshore_N": "#ff9999"  # light red
}
# -----------------------------
# Load data
# -----------------------------
file_path = "LCO_results_IP_2050.xlsx"
df_raw = pd.read_excel(file_path, sheet_name="Elec_generation")

# ⚠️ Change this if your column name is different
string_col = df_raw.columns[0]

# -----------------------------
# Split the string column
# -----------------------------
df = df_raw[string_col].str.split(",", expand=True)
df.columns = ["Island", "Year", "Tech", "Commodity", "BalanceType", "Value"]

# Convert types
df["Year"] = df["Year"].astype(int)
df["Value"] = df["Value"].astype(float)

# -----------------------------
# Filter: electricity generation only
# -----------------------------
df = df[
    (df["Commodity"] == "Elec") &
    (df["BalanceType"] == "net") &
    (df["Value"] > 0)
]

# -----------------------------
# Aggregate generation
# -----------------------------
gen_agg = (
    df.groupby(["Island", "Year", "Tech"], as_index=False)["Value"]
    .sum()
)

# Total generation per island-year
total_gen = (
    gen_agg.groupby(["Island", "Year"], as_index=False)["Value"]
    .sum()
    .rename(columns={"Value": "TotalGen"})
)

# Merge totals
gen_agg = gen_agg.merge(total_gen, on=["Island", "Year"])

# -----------------------------
# Compute percentage share
# -----------------------------
gen_agg["Share_pct"] = 100 * gen_agg["Value"] / gen_agg["TotalGen"]
years = sorted(gen_agg["Year"].unique())
techs = gen_agg["Tech"].unique()

for year in years:
    df_year = gen_agg[gen_agg["Year"] == year]

    pivot = df_year.pivot(
        index="Island",
        columns="Tech",
        values="Share_pct"
    ).fillna(0)

    pivot = pivot.sort_index()

    plt.figure(figsize=(16, 6))
    bottom = None

    for tech in pivot.columns:
        plt.bar(
            pivot.index,
            pivot[tech],
            bottom=bottom,
            label=tech,
            color=tech_colors.get(tech, "lightgray")
        )

        bottom = pivot[tech] if bottom is None else bottom + pivot[tech]

    plt.ylabel("Electricity generation share (%)")
    plt.title(f"Electricity generation mix by technology – {year}")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 100)
    plt.legend(ncol=4, fontsize=9)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()
##############################################################################
import pandas as pd
import matplotlib.pyplot as plt
tech_colors = {
    # Industry
    "Industry": "gray",
    "Industry_EL": "#1f77b4",   # blue

    # Cooking – electric
    "cook_el": "#1f77b4",       # blue
    "DW_Electric_converter": "#1f77b4",  # blue

    # Biomass cooking
    "cook_b": "#2ca02c",        # green

    # Solar thermal
    "ST_N": "#FFD700",          # yellow

    # Heat pumps
    "HP": "#ff7f0e",            # orange

    # LPG technologies
    "cook_LPG": "#4d4d4d",      # dark gray
    "DW_LPG_converter": "#4d4d4d"  # dark gray
}

# -----------------------------
# Load data
# -----------------------------
file_path = "LCO_results_IP_2050.xlsx"
df_raw = pd.read_excel(file_path, sheet_name="Heat_share")

# ⚠️ Change this if your column name is different
string_col = df_raw.columns[0]

# -----------------------------
# Split the string column
# -----------------------------
df = df_raw[string_col].str.split(",", expand=True)
df.columns = ["Island", "Year", "Tech", "Commodity", "BalanceType", "Value"]

# Convert types
df["Year"] = df["Year"].astype(int)
df["Value"] = df["Value"].astype(float)

# -----------------------------
# Filter: electricity generation only
# -----------------------------
# df = df[
#     (df["Commodity"] == "Elec") &
#     (df["BalanceType"] == "net") &
#     (df["Value"] > 0)
# ]

# -----------------------------
# Aggregate generation
# -----------------------------
gen_agg = (
    df.groupby(["Island", "Year", "Tech"], as_index=False)["Value"]
    .sum()
)

# Total generation per island-year
total_gen = (
    gen_agg.groupby(["Island", "Year"], as_index=False)["Value"]
    .sum()
    .rename(columns={"Value": "TotalGen"})
)

# Merge totals
gen_agg = gen_agg.merge(total_gen, on=["Island", "Year"])

# -----------------------------
# Compute percentage share
# -----------------------------
gen_agg["Share_pct"] = 100 * gen_agg["Value"] / gen_agg["TotalGen"]
years = sorted(gen_agg["Year"].unique())
techs = gen_agg["Tech"].unique()

for year in years:
    df_year = gen_agg[gen_agg["Year"] == year]

    pivot = df_year.pivot(
        index="Island",
        columns="Tech",
        values="Share_pct"
    ).fillna(0)

    pivot = pivot.sort_index()

    plt.figure(figsize=(16, 6))
    bottom = None

    for tech in pivot.columns:
        plt.bar(
            pivot.index,
            pivot[tech],
            bottom=bottom,
            label=tech,
            color=tech_colors.get(tech, "lightgray")
        )

        bottom = pivot[tech] if bottom is None else bottom + pivot[tech]

    plt.ylabel("Heat generation share (%)")
    plt.title(f"Heat generation mix by technology – {year}")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 100)
    plt.legend(ncol=4, fontsize=9)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()
##########################################################
    import pandas as pd
    import matplotlib.pyplot as plt
    tech_colors = {
        "Aviation": "gray",

        "Aviation_el": "#FFD700",   # yellow


        "Bus": "gray",   # green


        "Bus_el": "#FFD700",  # orange


        "Dummy_Ammonia": "#1f77b4",
        "Dummy_Methanol": "#ff7f0e",

        "HDV": "gray",   # yellow


        "HDV_el": "#FFD700",   # green


        "HDV_BF": "#2ca02c",  # orange


        "LDV": "gray",
        "LDV_el": "#FFD700",

        "LDV_BF": "#2ca02c",   # yellow


        "MDV": "gray",   # green


        "MDV_el": "#FFD700",  # orange


        "MDV_BF": "#2ca02c",
        "Two_Wheel": "gray",
        "Marine": "gray", # orange


        "Two_Wheel_el": "#FFD700",
        "FTL": "#d62728"   ,
        # blue
     # light red
    }
    # -----------------------------
    # Load data
    # -----------------------------
    file_path = "LCO_results_IP_2050.xlsx"
    df_raw = pd.read_excel(file_path, sheet_name="Transportation")

    # ⚠️ Change this if your column name is different
    string_col = df_raw.columns[0]

    # -----------------------------
    # Split the string column
    # -----------------------------
    df = df_raw[string_col].str.split(",", expand=True)
    df.columns = ["Island", "Year", "Tech", "Commodity", "BalanceType", "Value"]

    # Convert types
    df["Year"] = df["Year"].astype(int)
    df["Value"] = df["Value"].astype(float)

    # -----------------------------
    # Filter: electricity generation only
    # -----------------------------
    # df = df[
    #     (df["Commodity"] == "Elec") &
    #     (df["BalanceType"] == "net") &
    #     (df["Value"] > 0)
    # ]

    # -----------------------------
    # Aggregate generation
    # -----------------------------
    gen_agg = (
        df.groupby(["Island", "Year", "Tech"], as_index=False)["Value"]
        .sum()
    )

    # Total generation per island-year
    total_gen = (
        gen_agg.groupby(["Island", "Year"], as_index=False)["Value"]
        .sum()
        .rename(columns={"Value": "TotalGen"})
    )

    # Merge totals
    gen_agg = gen_agg.merge(total_gen, on=["Island", "Year"])

    # -----------------------------
    # Compute percentage share
    # -----------------------------
    gen_agg["Share_pct"] = 100 * gen_agg["Value"] / gen_agg["TotalGen"]
    years = sorted(gen_agg["Year"].unique())
    techs = gen_agg["Tech"].unique()

    for year in years:
        df_year = gen_agg[gen_agg["Year"] == year]

        pivot = df_year.pivot(
            index="Island",
            columns="Tech",
            values="Share_pct"
        ).fillna(0)

        pivot = pivot.sort_index()

        plt.figure(figsize=(16, 6))
        bottom = None

        for tech in pivot.columns:
            plt.bar(
                pivot.index,
                pivot[tech],
                bottom=bottom,
                label=tech,
                color=tech_colors.get(tech, "lightgray")
            )

            bottom = pivot[tech] if bottom is None else bottom + pivot[tech]

        plt.ylabel("Commodity share share (%)")
        plt.title(f"Transportation commodity share – {year}")
        plt.xticks(rotation=45, ha="right")
        plt.ylim(0, 100)
        plt.legend(ncol=4, fontsize=9)
        plt.grid(axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.show()
        
##############################
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

# -----------------------------
# Raw data (as provided)
# -----------------------------
raw_data = """global,2030,Battery,Elec_LiIon,total,11.3871
global,2030,THSS,Heat_T,total,2.35343e-05
global,2040,Ammonia_storage,Ammonia_T,total,0.008269
global,2040,Battery,Elec_LiIon,total,30.1439
global,2040,co2_storage,co_T,total,14000
global,2040,eKerosene_storage,Ammonia_T,total,1.24454
global,2040,H20_storage,Pure_water_T,total,17.1959
global,2040,H2_storage,Hydrogen_T,total,250.281
global,2040,Methanol_storage,Methanol_T,total,26.0593
global,2040,THSS,Heat_T,total,61.8665
global,2050,Ammonia_storage,Ammonia_T,total,121.968
global,2050,Battery,Elec_LiIon,total,41.2547
global,2050,co2_storage,co_T,total,14000
global,2050,eKerosene_storage,Ammonia_T,total,1001.35
global,2050,H20_storage,Pure_water_T,total,118.744
global,2050,H2_storage,Hydrogen_T,total,1367.81
global,2050,Methanol_storage,Methanol_T,total,819.351
global,2050,THSS,Heat_T,total,187.048"""

# -----------------------------
# Load into DataFrame
# -----------------------------
df = pd.read_csv(
    StringIO(raw_data),
    header=None,
    names=["Scope", "Year", "Storage", "Commodity", "Type", "Value"]
)

df["Year"] = df["Year"].astype(int)
df["Value"] = df["Value"].astype(float)

# -----------------------------
# Plot: one figure per storage
# -----------------------------
storages = df["Storage"].unique()

for storage in storages:
    df_s = df[df["Storage"] == storage].sort_values("Year")

    plt.figure(figsize=(6, 4))
    plt.bar(
        df_s["Year"].astype(str),
        df_s["Value"]
    )

    plt.xlabel("Year")
    plt.ylabel("Installed storage capacity (GWh)")
    plt.title(f"{storage} – Global")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()
######################################
import pandas as pd
import matplotlib.pyplot as plt

file_path = "LCO_results_IP_2050.xlsx"   # change if needed
sheet_name = "Methanol_storage"               # change if needed

df_raw = pd.read_excel(file_path, sheet_name=sheet_name)

# first column contains the full string
string_col = df_raw.columns[0]
df = df_raw[string_col].astype(str).str.split(",", expand=True)
df.columns = [
    "timeModel",
    "accNodesModel",
    "accYears",
    "techs",
    "commodity",
    "Value"
]

df["Value"] = df["Value"].astype(float)
df["accYears"] = df["accYears"].astype(int)
meth_sl = df[
    (df["accNodesModel"] == "global") &
    (df["accYears"] == 2050) &
    (df["techs"] == "Methanol_storage") &
    (df["commodity"] == "Methanol_T")
].copy()
meth_sl["Hour"] = (
    meth_sl["timeModel"]
    .str.replace("tm", "", regex=False)
    .astype(int)
)

meth_sl = meth_sl.sort_values("Hour")
plt.figure(figsize=(15, 5))
plt.plot(meth_sl["Hour"], meth_sl["Value"])
plt.xlabel("Hour of year")
plt.ylabel("Methanol storage level")
plt.title("Hourly Methanol Storage Level – Global (2050)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
##########PNG_generation_testing####

#############################
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Load Excel
# -----------------------------
file_path = "IP_2050_2_biomasscap_export.xlsx"   # <-- your file
df = pd.read_excel(file_path, header=None)

df.columns = ["timeModel", "Island", "Year", "Tech", "Commodity", "Value"]

# -----------------------------
# FIX merged cells
# -----------------------------
df[["timeModel", "Island", "Year"]] = df[["timeModel", "Island", "Year"]].ffill()

# -----------------------------
# Filter PNG electricity (2050)
# -----------------------------
df = df[
    (df["Island"] == "PNG_model") &
    (df["Year"] == 2050) &
    (df["Commodity"] == "Elec")
].copy()

# -----------------------------
# Convert tm → hour
# -----------------------------
df["Hour"] = (
    df["timeModel"]
    .astype(str)
    .str.replace("tm", "", regex=False)
    .astype(int)
)

# -----------------------------
# Pivot: Hour × Tech
# -----------------------------
pivot = (
    df.pivot_table(
        index="Hour",
        columns="Tech",
        values="Value",
        aggfunc="sum"
    )
    .sort_index()
)

# -----------------------------
# 2-week rolling average (336 hours)
# -----------------------------
rolling_hours = 24 * 14
pivot_smooth = pivot.rolling(
    window=rolling_hours,
    center=True,
    min_periods=1
).mean()

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(16, 6))

for tech in pivot_smooth.columns:
    plt.plot(
        pivot_smooth.index,
        pivot_smooth[tech],
        label=tech,
        linewidth=1.5
    )

plt.xlabel("Hour of year")
plt.ylabel("Electricity generation")
plt.title("PNG electricity generation by technology (2-week average, 2050)")
plt.legend(ncol=3, fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
###############methanol production timings################
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Load Excel (single-column CSV-style)
# -----------------------------
file_path = "LCO_results_IP_2050.xlsx"
df_raw = pd.read_excel(file_path, sheet_name="Sheet1")

# Identify the only column
string_col = df_raw.columns[0]

# -----------------------------
# Split into proper columns
# -----------------------------
df = df_raw[string_col].astype(str).str.split(",", expand=True)

df.columns = [
    "timeModel",
    "accNodesModel",
    "accYears",
    "techs",
    "commodity",
    "Value"
]

# -----------------------------
# Convert types
# -----------------------------
df["accYears"] = df["accYears"].astype(int)
df["Value"] = df["Value"].astype(float)

# -----------------------------
# Filter Methanol production
# -----------------------------
df_methanol = df[
    (df["accNodesModel"] == "PNG_model") &
    (df["accYears"] == 2050) &
    (df["techs"] == "Dummy_Methanol") &
    (df["commodity"] == "Dummy_EL")
].copy()

# -----------------------------
# Convert tmX → hour index
# -----------------------------
df_methanol["Hour"] = (
    df_methanol["timeModel"]
    .str.replace("tm", "", regex=False)
    .astype(int)
)

df_methanol = df_methanol.sort_values("Hour")

# -----------------------------
# Show extracted data
# -----------------------------
print("Number of rows:", len(df_methanol))
print("\nFirst 10 rows:")
print(df_methanol.head(10))
print("\nLast 10 rows:")
print(df_methanol.tail(10))
print("\nStatistics:")
print(df_methanol["Value"].describe())

# -----------------------------
# Plot hourly production
# -----------------------------
plt.figure(figsize=(14, 5))
plt.plot(df_methanol["Hour"], df_methanol["Value"])
plt.xlabel("Hour of year")
plt.ylabel("Methanol production")
plt.title("Hourly Methanol Production – PNG (2050)")
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# 2-week rolling average (336 hours)
# -----------------------------
df_methanol["Value_2week_avg"] = (
    df_methanol["Value"]
    .rolling(window=336, center=True)
    .mean()
)

plt.figure(figsize=(14, 5))
plt.plot(df_methanol["Hour"], df_methanol["Value_2week_avg"])
plt.xlabel("Hour of year")
plt.ylabel("Methanol production (2-week avg)")
plt.title("Methanol Production – 2-Week Average (PNG, 2050)")
plt.grid(True)
plt.tight_layout()
plt.show()
