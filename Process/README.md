# REMix Pacific Islands Energy System Model — S1 / S2 / S13 / S23

This repository contains four scenario variants of the same underlying
model:

| Script | Result file | What it adds relative to the S1 baseline |
|---|---|---|
| `remix_pacific_model_S1_std.py` | `IP_2050_Final_SS1_minload` | Baseline — no inter-island shipping, no e-fuel imports |
| `remix_pacific_model_S2_std.py` | `IP_2050_Final_SS2_minload` | + inter-island **shipping links** (ammonia / methanol / e-kerosene) |
| `remix_pacific_model_S13_std.py` | `IP_2050_Final_SS13_minload` | + **e-fuel imports** (ammonia / methanol / e-kerosene bought in at select nodes) |
| `remix_pacific_model_S23_std.py` | `IP_2050_Final_SS23_minload` | + shipping links **and** e-fuel imports (S2 + S13 combined) |

All four are multi-node, multi-year (2020 / 2030 / 2040 / 2050) myopic
capacity-expansion optimisations of the Pacific Island energy system on the
DLR REMix framework, minimising total discounted system cost. They cover
power, land / marine / aviation transport, cooking, industry, domestic hot
water, water desalination and synthetic-fuel production, together with the
storage and fuel-supply infrastructure needed to balance them, across 14
Pacific data nodes. **Everything in §1–§5 below (nodes, converters, storage,
demand) is identical across all four scripts** — they share the same
model core and only differ in the inter-island trade options described in
§6. This README explains what the scripts build, sector by sector, so a
new reader can find their way around the code without tracing every
`m.parameter.add(...)` call by hand.

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
- A handful of per-node numeric parameters (2040/2050 electrolyser
  build-out and one 2050 PV resource cap) are re-calibrated slightly
  between scenarios — these are scenario-specific inputs, not structural
  differences, and aren't itemised here.

## 2. How a converter works in this model

Every technology in REMix is a **converter**: it takes one or more input
commodities and turns them into one or more output commodities, at fixed
ratios (`converter_coefficient`), subject to a build-out capacity
(`converter_capacityparam`) and investment/O&M cost
(`accounting_converterunits`). A converter with no coefficients defined has
no commodity flow at all — it is a "dead" object that carries cost/capacity
bookkeeping but can never be dispatched. (See §7.)

Two naming conventions recur throughout:
- **`_B`** = existing / brownfield converter (installed 2020 base capacity).
- **`_N`** = new-build converter (available for investment from 2030 on).

## 3. Converter glossary

Converters are grouped below by what they do: turn fuel/resource into
**power**, turn power/fuel into **heat**, move people/goods (**transport**),
or turn power into another energy carrier or product (**power-to-X**).
This grouping and every converter in it is common to all four scripts.

### 3.1 Power generation
Output commodity is electricity (`Elec`).

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

### 3.2 Heat generation
Output commodity is a heat carrier (space/water heating, cooking heat, or
industrial process heat).

| Tech | Meaning |
|---|---|
| `ST_N` | Solar thermal (new-build only) |
| `HP` | Heat pump (electricity → heat) |
| `cook_b` | Biomass cooking |
| `cook_LPG` | LPG-based cooking |
| `cook_el` | Electric cooking |
| `DW_LPG_converter` | LPG-based domestic water heating |
| `DW_Electric_converter` | Electric domestic water heating |
| `Industry` | Diesel-fired industrial boilers |
| `Industry_EL` | Electric industrial boilers |
| `Industry_EH` | Placeholder / dummy converter linking industrial heat demand into the model — not a distinct physical technology |

### 3.3 Transport
Fossil road-transport converters (`MDV`, `HDV`, `LDV` = medium / heavy /
light-duty vehicles, plus `Bus` and `Two_wheel`) exist from 2020. From
2040, each gets two low-carbon variants, and marine/aviation transport is
split into fossil and electric options too.

| Tech | Meaning |
|---|---|
| `MDV`, `HDV`, `LDV`, `Bus`, `Two_wheel` | Fossil (diesel/gasoline) road transport |
| `_el` variants: `LDV_el`, `HDV_el`, `MDV_el`, `Two_wheel_el`, `Bus_el` | Battery-electric road transport |
| `_BF` variants: `LDV_BF`, `HDV_BF`, `MDV_BF` | Biofuel-powered road transport |
| `Marine` | Fossil (diesel/MDO) marine transport |
| `HFO` | Heavy-fuel-oil-based marine transport |
| `Ship_BEV` | Electric (battery) ships |
| `Aviation` | Fossil (jet fuel) aviation |
| `Aviation_el` | Electric aviation (short-range) |

### 3.4 Power-to-X
Converters that use electricity (directly or via an intermediate energy
carrier) to produce water, hydrogen, or a synthetic fuel/feedstock. All
new-build, available from 2040.

| Tech | Meaning |
|---|---|
| `RO` | Reverse osmosis (seawater desalination) |
| `AEL` | Alkaline electrolyser (hydrogen production) |
| `DAC` | Direct air capture (of CO2, for e-fuel synthesis) |
| `Ammonia_synthesis` | Hydrogen → ammonia synthesis |
| `Methanol_synthesis` | Hydrogen + CO2 → methanol synthesis |
| `FTL` | Fischer–Tropsch liquids (e-kerosene synthesis) |
| `Dummy_Ammonia` / `Dummy_Methanol` | Placeholder converters linking ammonia/methanol into a generic downstream demand commodity (`Dummy_EL`) |

### 3.5 Storage technologies
Registered via `add_storage_tech` (each with a `Charge`/`Discharge` pair):
`Battery`, `THSS` (thermal short-term storage), `H20_storage`, `H2_storage`,
`Ammonia_storage`, `Methanol_storage`, `eKerosene_storage`, `co2_storage`.

## 4. Fuel imports and prices (all scenarios)

Bounded, priced fuel-import sinks/sources are registered for:
`Biomass, NG, HFOO, Diesel, LPG, Gasoline, JetA1, MDO` (plus a per-node
biomass availability limit, `BIOMASS_LIMITS`). Prices are flat per
commodity across all nodes (`CONV_FUEL_PRICES`), fixed for 2020 and again
across 2030/2040/2050. This block is identical across all four scripts;
see §6.2 for the additional e-fuel imports in S13/S23.

## 5. Demand, emissions accounting

Demand is registered per commodity/year via `register_demand`, which reads
fixed profiles from the input CSV and wires them to a `Demand`
sourcesink entry per node (`add_demand_profile` + `add_demand_config`).
A CO2 sourcesink annual-sum tracks total system emissions
(`sourcesink_annualsum` / `sourcesink_config` on `Emission`/`CO2`).

## 6. What differs between scenarios

### 6.1 Inter-island shipping links (S2, S23 only)
S2 and S23 add 22 fixed inter-island transfer links (`Ship__Z_1` …
`Ship__Z_22`, e.g. FJ↔VU, FJ↔TA, KB↔MI, MI↔FSM, PU↔PNG, …), each carrying
up to 100 units/year of one of three energy carriers via a dedicated
"port" converter:

| Carrier | Transfer tech |
|---|---|
| Ammonia | `port_A` |
| Methanol | `port_M` |
| e-Kerosene | `port_F` |

These are available from 2040 and registered via `transfer_linkstartend`,
`transfer_linksparam`, `transfer_techparam`, `transfer_coefficient`, plus
build/flow costs from `add_transfer_link_costs`. S1 and S13 do **not**
build this block at all — inter-island shipping is structurally absent in
those two scripts, even though the module docstring in every script
advertises it (see §7).

### 6.2 E-fuel imports (S13, S23 only)
S13 and S23 additionally allow direct import of `Ammonia`, `Methanol` and
`eKerosene` (on top of the S1 baseline's fossil-fuel imports in §4) at a
fixed subset of nodes:

`EFUEL_NODES = [FJ, PNG, VU, TA, SA, SI, CI, NE]`

with an effectively unconstrained annual limit (`1,000,000` units/node) and
flat per-node prices for 2040/2050:

| Commodity | 2040 price | 2050 price |
|---|---|---|
| Ammonia | 0.080 | 0.066 |
| Methanol | 0.0884 | 0.0735 |
| e-Kerosene | 0.114 | 0.0971 |

S1 and S2 do **not** register these imports — in those two scripts, the
only way to meet ammonia/methanol/e-kerosene demand is domestic synthesis
(`Ammonia_synthesis`, `Methanol_synthesis`, `FTL`) or, for S2, shipping
from another island.

### 6.3 Summary matrix

| | Shipping links | E-fuel imports |
|---|---|---|
| **S1** | — | — |
| **S2** | ✅ | — |
| **S13** | — | ✅ |
| **S23** | ✅ | ✅ |

## 7. Known structural gap common to all four scripts

Four converters are instantiated in every script (tech params, capacity
limits, and zero investment cost) but never given a
`converter_coefficient`, so they have no commodity input/output and cannot
be dispatched. They are effectively dead code and were excluded from the
glossary in §3. See the prior conversation for the full trace.

(Note: `add_transfer_link_costs` itself is *not* dead — it's defined once
and actually called, with real effect, in S2/S23 per §6.1. It's only
unused in S1/S13, where the module docstring's mention of shipping links
is aspirational rather than active.)

## 8. Output

`m.write(fileformat="dat")` writes the REMix `.dat` input files, and
`m.run(...)` solves the model (myopic path optimisation, log level 3,
post-calculation and time-series rounding enabled) to the scenario's
result file listed in the table at the top of this README.