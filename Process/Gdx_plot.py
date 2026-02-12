# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 11:20:11 2025

@author: ajh287
"""

import gdxpds


#Load all symbols
data = gdxpds.to_dataframes(
r"C:\Local\remix-pic\REMix-Pacific_Island_Countries\Process\results\IP_2050_2.gdx"
)
print(data.keys())
# # # #######################################Final computations##################################

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
All_final_tech = ["BG_N","PV_N","WindOnshore_N", "Wave_N","WindOffshore_N", "ST_N","Industry_EL", "LDV_BF", "RO","Ammonia_synthesis", "DAC", "Methanol_synthesis", "HP", "FTL","AEL_100", "LDV_el", "HDV_el", "HDV_BF", "MDV_el", "MDV_BF", "Two_wheel_el", "Bus_el", "Marine_e", "Aviation_el", "Aviation_e", "cook_el", "cook_LPG", "Industry_EH", "DW_heat", "Dummy_Ammonia", "Dummy_Methanol", "ST_N_DW", "DW_Electric_converter_2"]
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

print("\nLCOE Electricity (USD per unit Elec):")
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
years = ["2020"]
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
lcoe_df.to_excel("LCOE_results_2050.xlsx", index=False)


# # #################################################################################heat
# # heat_fuel_costs = {
# #     "Biomass": 0.032,  # USD/unit
# #     "Diesel": 0.090,
# #     "LPG": 0.065,
# #     "Elec": None  # electricity cost will use previously computed LCOE_elec
# # }

# # # Map technologies to fuel
# # heat_fuel_map = {
# #     "cook_b": "Biomass",
# #     "Industry": "Diesel",
# #     "cook_LPG": "LPG",
# #     "DW_LPG_converter": "LPG",
# #     "DW_Electric_converter": "Elec",
# #     "HP": "Elec",
# #     "cook_el": "Elec",
# #     "Industry_EL": "Elec",
# #   # if applicable
# # }
# # def compute_LCOHeat(island, year, LCOE_elec_lookup):
# #     """
# #     LCO of heat for a given island and year.
# #     LCOE_elec_lookup: dict {(island, year): LCOE_electricity}
# #     """
# #     ind = data["indicator_accounting_detailed"]

# #     # ----- PART 1: Direct costs (Invest + O&M) -----
# #     heat_techs = Heat_production_tech + Heat_storage_tech
# #     ind_filt = ind[
# #         (ind["nodesModel"] == island) &
# #         (ind["years"] == year) &
# #         (ind["techs"].isin(heat_techs)) &
# #         (ind["indicator"].isin(["Invest", "OMFix", "OMVar"]))
# #     ]
# #     part1_direct_costs = ind_filt["Value"].sum()

# #     # ----- PART 2: Fuel / electricity costs -----
# #     cb = data["commodity_balance_annual"]

# #     cb_heat = cb[
# #         (cb["accNodesModel"] == island) &
# #         (cb["accYears"] == year) &
# #         (cb["balanceType"] == "net") &
# #         (cb["techs"].isin(heat_fuel_map.keys())) &
# #         (cb["Value"] < 0)  # inputs only
# #     ]

# #     fuel_cost_total = 0.0
# #     for tech, fuel in heat_fuel_map.items():
# #         fuel_use = cb_heat[
# #             (cb_heat["techs"] == tech) &
# #             (cb_heat["commodity"] == fuel)
# #         ]["Value"].sum()

# #         if fuel_use == 0:
# #             continue

# #         # Electricity cost handled separately
# #         if fuel == "Elec":
# #             elec_cost = LCOE_elec_lookup.get((island, year), 0.0)
# #             fuel_cost_total += abs(fuel_use) * elec_cost
# #         else:
# #             fuel_cost_total += abs(fuel_use) * heat_fuel_costs.get(fuel, 0.0)

# #     total_heat_cost = part1_direct_costs + fuel_cost_total

# #     # ----- DEMAND -----
# #     heat_demand = cb[
# #         (cb["accNodesModel"] == island) &
# #         (cb["accYears"] == year) &
# #         (cb["balanceType"] == "net") &
# #         (cb["commodity"] == "Heat") &
# #         (cb["techs"].isin(heat_techs)) &
# #         (cb["Value"] < 0)  # heat consumption only
# #     ]["Value"].sum()

# #     heat_demand = abs(heat_demand)

# #     if heat_demand == 0:
# #         return {
# #             "LCOHeat": None,
# #             "Total_Heat_Demand": 0.0,
# #             "Total_Heat_Cost": total_heat_cost
# #         }

# #     return {
# #         "LCOHeat": total_heat_cost / heat_demand,
# #         "Total_Heat_Demand": heat_demand,
# #         "Total_Heat_Cost": total_heat_cost
# #     }
# # # Make a dictionary for LCOE electricity
# # LCOE_elec_lookup = {(row['Island'], row['Year']): row['LCOE_Electricity']
# #                     for _, row in lcoe_df.iterrows()}
# # years = ["2020", "2030"]
# # heat_results = []

# # for island in pic_models:
# #     for year in years:
# #         out = compute_LCOHeat(island, year, LCOE_elec_lookup)
# #         heat_results.append({
# #             "Island": island,
# #             "Year": int(year),
# #             "LCOHeat": out["LCOHeat"],
# #             "Total_Heat_Demand": out["Total_Heat_Demand"],
# #             "Total_Heat_Cost": out["Total_Heat_Cost"]
# #         })

# # heat_df = pd.DataFrame(heat_results)
# # print(heat_df)
# # output_path_heat = "LCOHeat_PICs_2020_2050.xlsx"

# # with pd.ExcelWriter(output_path_heat, engine="openpyxl") as writer:
# #     heat_df.to_excel(writer, sheet_name="LCOHeat_full", index=False)
# #     heat_df.pivot(index="Island", columns="Year", values="LCOHeat").to_excel(writer, sheet_name="LCOHeat")
# #     heat_df.pivot(index="Island", columns="Year", values="Total_Heat_Demand").to_excel(writer, sheet_name="Demand")
# #     heat_df.pivot(index="Island", columns="Year", values="Total_Heat_Cost").to_excel(writer, sheet_name="Cost")

# # print(f"Excel file written to: {output_path_heat}")
# # #########Heating demand#####################
# # Heat_demand_co = ["Heat_cooking", "T_cook_LPG", "T_cook_el", "Heat_industry", "T_Industry_EH", "DHW_el", "DHW_LPG"]
# # ##################
# # # Heat demand commodities
# # Heat_demand_co = [
# #     "Heat_cooking", "T_cook_LPG", "T_cook_el",
# #     "Heat_industry", "T_Industry_EH", "DHW_el", "DHW_LPG"
# # ]

# # # All heat producing techs
# # Heat_production_tech = [
# #     'cook_b', "Industry", "DW_LPG_converter", "DW_Electric_converter",
# #     "ST_N", "HP", "cook_el", "cook_LPG", "Industry_EL"
# # ]

# # def compute_heat_demand(island, year):
# #     cb = data["commodity_balance_annual"]

# #     # Filter for island, year, balanceType, commodity in list, techs in heat techs
# #     cb_heat = cb[
# #         (cb["accNodesModel"] == island) &
# #         (cb["accYears"] == year) &
# #         (cb["balanceType"] == "net") &
# #         (cb["commodity"].isin(Heat_demand_co)) &
# #         (cb["techs"].isin(Heat_production_tech))
# #     ]

# #     # Sum the values directly
# #     heat_demand = cb_heat["Value"].sum()

# #     return heat_demand

# # # Example usage for CI_model in 2020
# # ci_2020_heat_demand = compute_heat_demand("CI_model", "2020")
# # print(f"CI_model 2020 Heat Demand (MWh): {ci_2020_heat_demand}")
# # ####################################
# # import pandas as pd

# # # Heat demand commodities
# # Heat_demand_co = [
# #     "Heat_cooking", "T_cook_LPG", "T_cook_el",
# #     "Heat_industry", "T_Industry_EH", "DHW_el", "DHW_LPG"
# # ]

# # # Heat production techs
# # Heat_production_tech = [
# #     'cook_b', "Industry", "DW_LPG_converter", "DW_Electric_converter",
# #     "ST_N", "HP", "cook_el", "cook_LPG", "Industry_EL"
# # ]

# # # Heat storage techs
# # Heat_storage_tech = ['THSS']

# # # Fuel costs
# # heat_fuel_costs = {
# #     "Biomass": 0.032,
# #     "Diesel": 0.090,
# #     "LPG": 0.065,
# #     "Elec": None  # use electricity LCOE
# # }

# # # Map techs to fuels
# # heat_fuel_map = {
# #     "cook_b": "Biomass",
# #     "Industry": "Diesel",
# #     "cook_LPG": "LPG",
# #     "DW_LPG_converter": "LPG",
# #     "DW_Electric_converter": "Elec",
# #     "HP": "Elec",
# #     "cook_el": "Elec",
# #     "Industry_EL": "Elec"
# # }

# # def compute_LCOHeat(island, year, LCOE_elec_lookup):
# #     cb = data["commodity_balance_annual"]
# #     ind = data["indicator_accounting_detailed"]

# #     # ---------- PART 1: Direct costs ----------
# #     heat_techs = Heat_production_tech + Heat_storage_tech
# #     ind_filt = ind[
# #         (ind["nodesModel"] == island) &
# #         (ind["years"] == year) &
# #         (ind["techs"].isin(heat_techs)) &
# #         (ind["indicator"].isin(["Invest", "OMFix", "OMVar"]))
# #     ]
# #     part1_direct_costs = ind_filt["Value"].sum()

# #     # ---------- PART 2: Fuel / electricity costs ----------
# #     cb_heat = cb[
# #         (cb["accNodesModel"] == island) &
# #         (cb["accYears"] == year) &
# #         (cb["balanceType"] == "net") &
# #         (cb["techs"].isin(heat_fuel_map.keys())) &
# #         (cb["Value"] < 0)  # only inputs
# #     ]
# #     fuel_cost_total = 0.0
# #     for tech, fuel in heat_fuel_map.items():
# #         fuel_use = cb_heat[
# #             (cb_heat["techs"] == tech) &
# #             (cb_heat["commodity"] == fuel)
# #         ]["Value"].sum()

# #         if fuel_use == 0:
# #             continue

# #         if fuel == "Elec":
# #             elec_cost = LCOE_elec_lookup.get((island, year), 0.0)
# #             fuel_cost_total += abs(fuel_use) * elec_cost
# #         else:
# #             fuel_cost_total += abs(fuel_use) * heat_fuel_costs.get(fuel, 0.0)

# #     total_heat_cost = part1_direct_costs + fuel_cost_total

# #     # ---------- PART 3: Heat demand ----------
# #     cb_heat_demand = cb[
# #         (cb["accNodesModel"] == island) &
# #         (cb["accYears"] == year) &
# #         (cb["balanceType"] == "net") &
# #         (cb["commodity"].isin(Heat_demand_co)) &
# #         (cb["techs"].isin(Heat_production_tech))
# #     ]
# #     total_heat_demand = cb_heat_demand["Value"].sum()

# #     if total_heat_demand == 0:
# #         return {
# #             "LCOHeat": None,
# #             "Total_Heat_Demand": 0.0,
# #             "Total_Heat_Cost": total_heat_cost
# #         }

# #     return {
# #         "LCOHeat": total_heat_cost / total_heat_demand,
# #         "Total_Heat_Demand": total_heat_demand,
# #         "Total_Heat_Cost": total_heat_cost
# #     }

# # # Prepare electricity LCOE lookup
# # LCOE_elec_lookup = {(row['Island'], row['Year']): row['LCOE_Electricity']
# #                     for _, row in lcoe_df.iterrows()}

# # # Compute for all islands and years
# # years = ["2020", "2030"]
# # heat_results = []

# # for island in pic_models:
# #     for year in years:
# #         out = compute_LCOHeat(island, year, LCOE_elec_lookup)
# #         heat_results.append({
# #             "Island": island,
# #             "Year": int(year),
# #             "LCOHeat": out["LCOHeat"],
# #             "Total_Heat_Demand": out["Total_Heat_Demand"],
# #             "Total_Heat_Cost": out["Total_Heat_Cost"]
# #         })

# # heat_df = pd.DataFrame(heat_results)
# # print(heat_df)

# # # # Optional: export to Excel
# # # output_path_heat = "LCOHeat_PICs_2020_2050.xlsx"
# # # with pd.ExcelWriter(output_path_heat, engine="openpyxl") as writer:
# # #     heat_df.to_excel(writer, sheet_name="LCOHeat_full", index=False)
# # #     heat_df.pivot(index="Island", columns="Year", values="LCOHeat").to_excel(writer, sheet_name="LCOHeat")
# # #     heat_df.pivot(index="Island", columns="Year", values="Total_Heat_Demand").to_excel(writer, sheet_name="Demand")
# # #     heat_df.pivot(index="Island", columns="Year", values="Total_Heat_Cost").to_excel(writer, sheet_name="Cost")

# # # print(f"Excel file written to: {output_path_heat}")
# # ############################################################LCOEnergy##########
# # import pandas as pd

# # # List of 14 PIC models
# pic_models = [
#     "CI_model", "FJ_model", "FSM_model", "KB_model", "MI_model",
#     "NU_model", "NE_model", "PU_model", "PNG_model", "SA_model",
#     "SI_model", "TA_model", "TU_model", "VU_model"
# ]

# # # Commodities to include in total energy demand
# # total_energy_commodities = [
# #     "DHW_el", "DHW_LPG", "Elec", "Heat_cooking", "Heat_industry",
# #     "T_Aviation_th", "T_Bus_th", "T_HDV_th", "T_LDV_th",
# #     "T_Marine_f_th", "T_Marine_th", "T_MDV_th", "T_Two_wheel_th"
# # ]

# # def compute_LCOEnergy(island, year):
# #     # --- PART 1: Total system costs ---
# #     ind = data["indicator_accounting"]
# #     cost_total = ind[
# #         (ind["accNodesModel"] == island) &
# #         (ind["accYears"] == year) &
# #         (ind["indicator"] == "SystemCost")
# #     ]["Value"].sum()
    
# #     # --- PART 2: Total energy demand ---
# #     cb = data["commodity_balance_annual"]
# #     cb_demand = cb[
# #         (cb["accNodesModel"] == island) &
# #         (cb["accYears"] == year) &
# #         (cb["techs"] == "Demand") &
# #         (cb["balanceType"] == "net") &
# #         (cb["commodity"].isin(total_energy_commodities)) &
# #         (cb["Value"] < 0)
# #     ]["Value"].sum()
    
# #     total_demand = abs(cb_demand)
    
# #     # Avoid division by zero
# #     if total_demand == 0:
# #         return {
# #             "LCOEnergy": None,
# #             "Total_Energy_Demand": 0.0,
# #             "Total_System_Cost": cost_total
# #         }
    
# #     return {
# #         "LCOEnergy": cost_total / total_demand,
# #         "Total_Energy_Demand": total_demand,
# #         "Total_System_Cost": cost_total
# #     }

# # # Compute for all islands and years
# # years = ["2020", "2030"]
# # results = []

# # for island in pic_models:
# #     for year in years:
# #         out = compute_LCOEnergy(island, year)
# #         results.append({
# #             "Island": island,
# #             "Year": int(year),
# #             "LCOEnergy": out["LCOEnergy"],
# #             "Total_Energy_Demand": out["Total_Energy_Demand"],
# #             "Total_System_Cost": out["Total_System_Cost"]
# #         })

# # # Convert to DataFrame
# # lco_energy_df = pd.DataFrame(results)
# # print(lco_energy_df)
# # ########################################Pure water##################################
# # import pandas as pd

# # # Water technologies
# # Water_production_tech = ['RO']
# # Water_storage_tech = ['H20_storage']

# # def compute_LCOWater(island, year, LCOE_elec_lookup):
# #     """
# #     Compute Levelized Cost of Water (LCOH) for a given island and year.
# #     Uses LCOE electricity for cost of RO electricity input.
# #     """
# #     # ---------- PART 1: Direct costs ----------
# #     ind = data["indicator_accounting_detailed"]
# #     water_techs = Water_production_tech + Water_storage_tech

# #     ind_filt = ind[
# #         (ind["nodesModel"] == island) &
# #         (ind["years"] == year) &
# #         (ind["techs"].isin(water_techs)) &
# #         (ind["indicator"].isin(["Invest", "OMFix", "OMVar"]))
# #     ]

# #     part1_direct_costs = ind_filt["Value"].sum()

# #     # ---------- PART 2: Electricity cost ----------
# #     cb = data["commodity_balance_annual"]
# #     cb_ro_elec = cb[
# #         (cb["accNodesModel"] == island) &
# #         (cb["accYears"] == year) &
# #         (cb["techs"].isin(Water_production_tech)) &
# #         (cb["balanceType"] == "net") &
# #         (cb["commodity"] == "Elec") &
# #         (cb["Value"] < 0)  # consumption only
# #     ]

# #     elec_use = abs(cb_ro_elec["Value"].sum())
# #     elec_cost = LCOE_elec_lookup.get((island, year), 0.0)

# #     total_elec_cost = elec_use * elec_cost

# #     total_water_cost = part1_direct_costs + total_elec_cost

# #     # ---------- PART 3: Water demand ----------
# #     cb_ro_demand = cb[
# #         (cb["accNodesModel"] == island) &
# #         (cb["accYears"] == year) &
# #         (cb["techs"].isin(Water_production_tech)) &
# #         (cb["balanceType"] == "net") &
# #         (cb["commodity"] == "Pure_water")
# #     ]

# #     water_demand = cb_ro_demand["Value"].sum()  # Already positive

# #     if water_demand == 0:
# #         return {
# #             "LCOWater": None,
# #             "Total_Water_Demand": 0.0,
# #             "Total_Water_Cost": total_water_cost
# #         }

# #     return {
# #         "LCOWater": total_water_cost / water_demand,
# #         "Total_Water_Demand": water_demand,
# #         "Total_Water_Cost": total_water_cost
# #     }

# # # Make a dictionary for LCOE electricity
# # LCOE_elec_lookup = {(row['Island'], row['Year']): row['LCOE_Electricity']
# #                     for _, row in lcoe_df.iterrows()}

# # # Compute for all islands and years
# # years = ["2020", "2030"]
# # water_results = []

# # for island in pic_models:
# #     for year in years:
# #         out = compute_LCOWater(island, year, LCOE_elec_lookup)
# #         water_results.append({
# #             "Island": island,
# #             "Year": int(year),
# #             "LCOWater": out["LCOWater"],
# #             "Total_Water_Demand": out["Total_Water_Demand"],
# #             "Total_Water_Cost": out["Total_Water_Cost"]
# #         })

# # water_df = pd.DataFrame(water_results)
# # print(water_df)

# # # Export to Excel
# # ####################H2########################
# # # H2 technologies
# # H2_production_tech = ["AEL_100"]
# # H2_storage_tech = ["H2_storage"]

# # def compute_LCOH2(island, year, LCOE_elec_lookup, LCOWater_lookup):
# #     """
# #     Compute Levelized Cost of Hydrogen (LCOH2) for a given island and year.
# #     Uses LCOE electricity and LCOWater for cost of inputs.
# #     """
# #     # ---------- PART 1: Direct costs ----------
# #     ind = data["indicator_accounting_detailed"]
# #     h2_techs = H2_production_tech + H2_storage_tech

# #     ind_filt = ind[
# #         (ind["nodesModel"] == island) &
# #         (ind["years"] == year) &
# #         (ind["techs"].isin(h2_techs)) &
# #         (ind["indicator"].isin(["Invest", "OMFix", "OMVar"]))
# #     ]

# #     part1_direct_costs = ind_filt["Value"].sum()

# #     # ---------- PART 2: Input costs ----------
# #     cb = data["commodity_balance_annual"]

# #     # Electricity consumption for H2 production
# #     cb_elec = cb[
# #         (cb["accNodesModel"] == island) &
# #         (cb["accYears"] == year) &
# #         (cb["techs"].isin(H2_production_tech)) &
# #         (cb["balanceType"] == "net") &
# #         (cb["commodity"] == "Elec") &
# #         (cb["Value"] < 0)
# #     ]
# #     elec_use = abs(cb_elec["Value"].sum())
# #     elec_cost = LCOE_elec_lookup.get((island, year), 0.0)
# #     elec_total_cost = elec_use * elec_cost

# #     # Water consumption for H2 production
# #     cb_water = cb[
# #         (cb["accNodesModel"] == island) &
# #         (cb["accYears"] == year) &
# #         (cb["techs"].isin(H2_production_tech)) &
# #         (cb["balanceType"] == "net") &
# #         (cb["commodity"] == "Pure_water") &
# #         (cb["Value"] < 0)
# #     ]
# #     water_use = abs(cb_water["Value"].sum())
# #     water_cost = LCOWater_lookup.get((island, year), 0.0)
# #     water_total_cost = water_use * water_cost

# #     total_h2_cost = part1_direct_costs + elec_total_cost + water_total_cost

# #     # ---------- PART 3: Hydrogen demand ----------
# #     cb_h2_demand = cb[
# #         (cb["accNodesModel"] == island) &
# #         (cb["accYears"] == year) &
# #         (cb["techs"].isin(H2_production_tech)) &
# #         (cb["balanceType"] == "net") &
# #         (cb["commodity"] == "Hydrogen")
# #     ]
# #     h2_demand = cb_h2_demand["Value"].sum()  # already positive

# #     if h2_demand == 0:
# #         return {
# #             "LCOH2": None,
# #             "Total_H2_Demand": 0.0,
# #             "Total_H2_Cost": total_h2_cost
# #         }

# #     return {
# #         "LCOH2": total_h2_cost / h2_demand,
# #         "Total_H2_Demand": h2_demand,
# #         "Total_H2_Cost": total_h2_cost
# #     }


# # # Make dictionaries for input costs
# # LCOE_elec_lookup = {(row['Island'], row['Year']): row['LCOE_Electricity'] for _, row in lcoe_df.iterrows()}
# # LCOWater_lookup = {(row['Island'], row['Year']): row['LCOWater'] for _, row in water_df.iterrows()}

# # # Compute LCOH2 for all islands and years
# # years = ["2020", "2030"]
# # h2_results = []

# # for island in pic_models:
# #     for year in years:
# #         out = compute_LCOH2(island, year, LCOE_elec_lookup, LCOWater_lookup)
# #         h2_results.append({
# #             "Island": island,
# #             "Year": int(year),
# #             "LCOH2": out["LCOH2"],
# #             "Total_H2_Demand": out["Total_H2_Demand"],
# #             "Total_H2_Cost": out["Total_H2_Cost"]
# #         })

# # h2_df = pd.DataFrame(h2_results)
# # print(h2_df)
# # ###########################################################################
# # Ammonia_production_tech = ["Ammonia_synthesis"]
# # Ammonia_storage_tech = ["Ammonia_storage"]
# # def compute_LCOAmmonia(island, year, LCOE_elec_lookup, LCOH2_lookup):
# #     """
# #     Compute Levelized Cost of Ammonia (LCOA) for a given island and year.
# #     Uses electricity LCOE and hydrogen LCOH2 as input costs.
# #     """

# #     # ---------- PART 1: Direct costs ----------
# #     ind = data["indicator_accounting_detailed"]
# #     ammo_techs = Ammonia_production_tech + Ammonia_storage_tech

# #     ind_filt = ind[
# #         (ind["nodesModel"] == island) &
# #         (ind["years"] == year) &
# #         (ind["techs"].isin(ammo_techs)) &
# #         (ind["indicator"].isin(["Invest", "OMFix", "OMVar"]))
# #     ]

# #     part1_direct_costs = ind_filt["Value"].sum()

# #     # ---------- PART 2: Input costs ----------
# #     cb = data["commodity_balance_annual"]

# #     # Electricity input
# #     cb_elec = cb[
# #         (cb["accNodesModel"] == island) &
# #         (cb["accYears"] == year) &
# #         (cb["techs"].isin(Ammonia_production_tech)) &
# #         (cb["balanceType"] == "net") &
# #         (cb["commodity"] == "Elec") &
# #         (cb["Value"] < 0)
# #     ]
# #     elec_use = abs(cb_elec["Value"].sum())
# #     elec_cost = LCOE_elec_lookup.get((island, year), 0.0)
# #     elec_total_cost = elec_use * elec_cost

# #     # Hydrogen input
# #     cb_h2 = cb[
# #         (cb["accNodesModel"] == island) &
# #         (cb["accYears"] == year) &
# #         (cb["techs"].isin(Ammonia_production_tech)) &
# #         (cb["balanceType"] == "net") &
# #         (cb["commodity"] == "Hydrogen") &
# #         (cb["Value"] < 0)
# #     ]
# #     h2_use = abs(cb_h2["Value"].sum())
# #     h2_cost = LCOH2_lookup.get((island, year), 0.0)
# #     h2_total_cost = h2_use * h2_cost

# #     total_ammonia_cost = (
# #         part1_direct_costs +
# #         elec_total_cost +
# #         h2_total_cost
# #     )

# #     # ---------- PART 3: Ammonia demand ----------
# #     cb_ammonia_demand = cb[
# #         (cb["accNodesModel"] == island) &
# #         (cb["accYears"] == year) &
# #         (cb["techs"] == "Dummy_Ammonia") &
# #         (cb["balanceType"] == "net") &
# #         (cb["commodity"] == "Ammonia") &
# #         (cb["Value"] < 0)
# #     ]

# #     ammonia_demand = abs(cb_ammonia_demand["Value"].sum())

# #     if ammonia_demand == 0:
# #         return {
# #             "LCOAmmonia": None,
# #             "Total_Ammonia_Demand": 0.0,
# #             "Total_Ammonia_Cost": total_ammonia_cost
# #         }

# #     return {
# #         "LCOAmmonia": total_ammonia_cost / ammonia_demand,
# #         "Total_Ammonia_Demand": ammonia_demand,
# #         "Total_Ammonia_Cost": total_ammonia_cost
# #     }
# # LCOE_elec_lookup = {
# #     (row["Island"], row["Year"]): row["LCOE_Electricity"]
# #     for _, row in lcoe_df.iterrows()
# # }

#   LCOH2_lookup = {
# #     (row["Island"], row["Year"]): row["LCOH2"]
# #     for _, row in h2_df.iterrows()
 # }
# # years = ["2020", "2030"]
# # ammonia_results = []

# # for island in pic_models:
# #     for year in years:
# #         out = compute_LCOAmmonia(island, year, LCOE_elec_lookup, LCOH2_lookup)
# #         ammonia_results.append({
# #             "Island": island,
# #             "Year": int(year),
# #             "LCOAmmonia": out["LCOAmmonia"],
# #             "Total_Ammonia_Demand": out["Total_Ammonia_Demand"],
# #             "Total_Ammonia_Cost": out["Total_Ammonia_Cost"]
# #         })

# # ammonia_df = pd.DataFrame(ammonia_results)
# # print(ammonia_df)
-------------------------------------------------------------------------------------------------------
