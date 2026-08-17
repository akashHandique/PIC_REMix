# REMix Pacific Islands Energy System Model — Scenario S1

`remix_pacific_model_S1_std.py` builds and runs a multi-node, multi-year
(2020 / 2030 / 2040 / 2050) myopic capacity-expansion optimisation of the
Pacific Island energy system on the DLR REMix framework. It covers power,
land / marine / aviation transport, cooking, industry, domestic hot water,
water desalination and synthetic-fuel production, together with the storage
and fuel-supply infrastructure needed to balance them, across 14 Pacific
data nodes. The objective is minimisation of total discounted system cost.

This README explains what the script builds, node by node and sector by
sector, so a new reader can find their way around the code without having
to trace every `m.parameter.add(...)` call by hand.

---

## 1. Nodes and horizon

- **Nodes** (`NODE_ORDER`): `CI, FJ, FSM, KB, MI, NU, NE, PU, PNG, SA, SI, TA, TU, VU` —
  14 Pacific Island data nodes, each represented internally as `<CODE>_data`.
- **Years**: `2020, 2030, 2040, 2050`, solved myopically (`pathopt="myopic"`),
  i.e. each horizon is optimised in sequence, building on the installed
  capacity carried over from the previous one.
- Demand and renewable-resource profiles are read from a single input CSV
  (`_input/Copy of IP_2040_2050_14_PIC - Copy.csv`); node labels are
  recovered from each column name via `_region_to_node`.

## 2. How a converter works in this model

Every technology in REMix is a **converter**: it takes one or more input
commodities and turns them into one or more output commodities, at fixed
ratios (`converter_coefficient`), subject to a build-out capacity
(`converter_capacityparam`) and investment/O&M cost
(`accounting_converterunits`). A converter with no coefficients defined has
no commodity flow at all — it is a "dead" object that carries cost/capacity
bookkeeping but can never be dispatched. (See §6.)

Two naming conventions recur throughout:
- **`_B`** = existing / brownfield converter (installed 2020 base capacity).
- **`_N`** = new-build converter (available for investment from 2030 on).

## 3. Converter glossary, by sector

### 3.1 Power generation
| Tech | Meaning |
|---|---|
| `DG` | Diesel generator |
| `NG_plant` | Natural-gas power plant |
| `BG_B` / `BG_N` | Biomass gasification, existing / new |
| `PV_B` / `PV_N` | Solar PV, existing / new |
| `WindOnshore_B` / `WindOnshore_N` | Onshore wind, existing / new |
| `WindOffshore_N` | Offshore wind (new-build only) |
| `Hydro_B` / `Hydro_N` | Hydropower, existing / new |
| `Geothermal_B` | Geothermal power (existing only) |
| `Wave_N` | Wave power (new-build only) |
| `ST_N` | Solar thermal (new-build only) |

### 3.2 Road transport (fossil baseline, `MDV`/`HDV`/`LDV`/`Bus`/`Two_wheel`)
`MDV`, `HDV`, `LDV` = medium / heavy / light-duty vehicles. Fossil versions
of these plus `Bus` and `Two_wheel` are fuel-based road transport (diesel or
gasoline, per the input data) and exist from 2020.

From 2040, each of these gets two low-carbon variants:
| Suffix | Meaning |
|---|---|
| `_el` (e.g. `LDV_el`, `HDV_el`, `MDV_el`, `Two_wheel_el`, `Bus_el`) | Battery-electric version |
| `_BF` (e.g. `LDV_BF`, `HDV_BF`, `MDV_BF`) | Biofuel-powered version |

### 3.3 Marine and aviation
| Tech | Meaning |
|---|---|
| `Marine` | Fossil (diesel/MDO) marine transport |
| `HFO` | Heavy-fuel-oil-based marine transport |
| `Aviation` | Fossil (jet fuel) aviation |
| `Aviation_el` | Electric aviation (short-range) |
| `Ship_BEV` | Electric (battery) ships |

### 3.4 Cooking
| Tech | Meaning |
|---|---|
| `cook_b` | Biomass cooking |
| `cook_LPG` | LPG-based cooking |
| `cook_el` | Electric cooking |

### 3.5 Industry
| Tech | Meaning |
|---|---|
| `Industry` | Diesel-fired industrial boilers |
| `Industry_EL` | Electric industrial boilers |
| `Industry_EH` | Placeholder / dummy converter linking industrial electric-heat demand into the model — not a distinct physical technology |

### 3.6 Domestic hot water (DHW)
| Tech | Meaning |
|---|---|
| `DW_LPG_converter` | LPG-based water heating |
| `DW_Electric_converter` | Electric water heating |

### 3.7 Water, hydrogen and power-to-X (all new-build, from 2040)
| Tech | Meaning |
|---|---|
| `RO` | Reverse osmosis (seawater desalination) |
| `AEL` | Alkaline electrolyser (hydrogen production) |
| `DAC` | Direct air capture (of CO2, for e-fuel synthesis) |
| `HP` | Heat pump |
| `Ammonia_synthesis` | Hydrogen → ammonia synthesis |
| `Methanol_synthesis` | Hydrogen + CO2 → methanol synthesis |
| `FTL` | Fischer–Tropsch liquids (e-kerosene synthesis) |
| `Dummy_Ammonia` / `Dummy_Methanol` | Placeholder converters linking ammonia/methanol into a generic downstream demand commodity (`Dummy_EL`) |

### 3.8 Storage technologies
Registered via `add_storage_tech` (each with a `Charge`/`Discharge` pair):
`Battery`, `THSS` (thermal short-term storage), `H20_storage`, `H2_storage`,
`Ammonia_storage`, `Methanol_storage`, `eKerosene_storage`, `co2_storage`.

## 4. Fuel imports and prices

Bounded, priced fuel-import sinks/sources are registered for:
`Biomass, NG, HFOO, Diesel, LPG, Gasoline, JetA1, MDO` (plus a per-node
biomass availability limit, `BIOMASS_LIMITS`). Prices are flat per
commodity across all nodes (`CONV_FUEL_PRICES`), fixed for 2020 and again
across 2030/2040/2050.

## 5. Demand, emissions accounting

Demand is registered per commodity/year via `register_demand`, which reads
fixed profiles from the input CSV and wires them to a `Demand`
sourcesink entry per node (`add_demand_profile` + `add_demand_config`).
A CO2 sourcesink annual-sum tracks total system emissions
(`sourcesink_annualsum` / `sourcesink_config` on `Emission`/`CO2`).

## 6. Known structural gaps (as of this refactor)

Two things are worth flagging to anyone extending this script, since they
affect what the model can actually do versus what the code/docstring
implies:

- **`add_transfer_link_costs` is defined but never called.** The module
  docstring advertises "inter-island ammonia / methanol / e-kerosene
  shipping links," and two comments (around the 2040 and 2050 converter-cost
  blocks) say shipping-link costs are "registered" there — but no
  `accounting_transferlinks` parameters are actually written anywhere in
  the script. Inter-island shipping is not currently active in this model
  version.
- **Four converters are instantiated (tech params, capacity limits, and
  zero investment cost) but never given a `converter_coefficient`, so they
  have no commodity input/output and cannot be dispatched.** These are
  effectively dead in the current script and were excluded from the
  glossary above. See the prior conversation for the full trace.

## 7. Output

`m.write(fileformat="dat")` writes the REMix `.dat` input files, and
`m.run(...)` solves the model (result file `IP_2050_Final_SS1_minload`,
myopic path optimisation, log level 3, post-calculation and time-series
rounding enabled).
