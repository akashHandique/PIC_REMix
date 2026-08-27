# `_input/` — Hourly demand & resource profile data

This folder holds the single input file every `Process/remix_pacific_model_*`
script reads:

```python
profiles = pd.read_csv("../_input/Hourly_demand_and_resource_profiles.csv", index_col=0)
```

This is one hourly time-series table covering all 14 Pacific nodes, all
four converter-activity years, and all four demand years. If you rename it
or move it, update the `pd.read_csv(...)` path in each of the four scenario
scripts to match.

## Shape

- **8,760 rows** — one full year of hourly time steps. The row index
  (column `I`, values `t0001` … `t8760`) is read in as the DataFrame index
  (`index_col=0`) — it is **not** a data series the model uses, just the
  hour label.
- **1,317 data columns** — every other column, named
  `<prefix>_<REGION>[_<YEAR>]`, where `<REGION>` is one of the 14 node codes
  (`CI, FJ, FSM, KB, MI, NU, NE, PU, PNG, SA, SI, TA, TU, VU`) and the
  optional `<YEAR>` suffix (`_2030`, `_2040`, `_2050`, or none for 2020) is
  present only on the demand-side columns (see below). The scripts recover
  the node from each column name via `_region_to_node`, which scans every
  underscore-separated token for a known region code — so column order and
  exact prefix don't matter, only that a valid region code appears
  somewhere in the name.

There are two distinct families of columns, used by two different
functions in the scenario scripts.

## 1. Renewable/resource activity profiles → `add_activity_profiles`

168 columns (14 nodes × 12 technology prefixes). These are the raw,
un-normalised hourly output of each renewable/resource-driven converter —
read once and reused for every horizon year, since the same weather year is
assumed to repeat across 2020/2030/2040/2050.

| Column prefix | Converter (see `Process/README.md` §3) | First available (build) year |
|---|---|---|
| `PV_B` | `PV_B` — Solar PV, existing | 2020 |
| `WindOnshore_B` | `WindOnshore_B` — Onshore wind, existing | 2020 |
| `Hydro_B` | `Hydro_B` — Hydropower, existing | 2020 |
| `Geothermal_B` | `Geothermal_B` — Geothermal power, existing | 2020 |
| `BG_B` | `BG_B` — Biomass gasification, existing | 2020 |
| `PV_N` | `PV_N` — Solar PV, new-build | 2030 |
| `WindOnshore_N` | `WindOnshore_N` — Onshore wind, new-build | 2030 |
| `Wave_N` | `Wave_N` — Wave power, new-build | 2030 |
| `WindOffshore_N` | `WindOffshore_N` — Offshore wind, new-build | 2030 |
| `BG_N` | `BG_N` — Biomass gasification, new-build | 2030 |
| `Hydro_N` | `Hydro_N` — Hydropower, new-build | 2030 |
| `ST_N` | `ST_N` — Solar thermal, new-build (heat, not power) | 2040 |

**Transform applied by the script:** for each node, the 12 columns are
selected as `f"{tech}_{region_code}"`, divided by 1000 (MW → GW), then
**each row is normalised by that technology/node's own max value** to
produce a 0–1 capacity-factor profile. This is registered as
`converter_activityprofile` with `type="upper"` — it caps how much of a
converter's built capacity can run in each hour. Because of the
normalisation, the *absolute* units/scale of the raw column don't matter —
only its hour-to-hour shape does.

`ST_N`'s output commodity is `Heat`, not `Elec` — it's grouped under Heat
generation in `Process/README.md`, not Power generation, even though it's
read by the same function.

## 2. Demand profiles → `register_demand` / `add_demand_profile`

The remaining ~1,148 columns are fixed hourly demand series, one block per
node per horizon year, following the naming convention below. Each column
is one `(commodity, prefix)` pair from `DEMAND_SPEC_2020` /
`DEMAND_SPEC_2030` / `DEMAND_SPEC_2040` / `DEMAND_SPEC_2050` in the
scripts. Year suffix: 2020 = no suffix, then `_2030`, `_2040`, `_2050`.

### 2020 / 2030 naming (13 columns per node per year)

| Column prefix | Commodity | Feeds converter / sector |
|---|---|---|
| `demand` | `Elec` | Direct electricity demand |
| `MDV` | `T_MDV_th` | `MDV` — fossil medium-duty vehicles |
| `HDV` | `T_HDV_th` | `HDV` — fossil heavy-duty vehicles |
| `LDV` | `T_LDV_th` | `LDV` — fossil light-duty vehicles |
| `Bus` | `T_Bus_th` | `Bus` — fossil buses |
| `Two_wheel` | `T_Two_wheel_th` | `Two_wheel` — fossil two-wheelers |
| `Marine` | `T_Marine_th` | `Marine` — fossil marine transport |
| `Marinef` | `T_Marine_f_th` | `HFO` — heavy-fuel-oil marine transport |
| `Aviation` | `T_Aviation_th` | `Aviation` — fossil aviation |
| `HC` | `Heat_cooking` | `cook_b` — biomass cooking |
| `HI` | `Heat_industry` | `Industry` — diesel industrial boilers |
| `DHWE` | `DHW_el` | `DW_Electric_converter` — electric water heating |
| `DHWL` | `DHW_LPG` | `DW_LPG_converter` — LPG water heating |

### 2040 / 2050 naming (28 columns per node per year)

By 2040 each sector splits into its low-carbon variants (see
`Process/README.md` §3.3–3.4), so the demand columns split accordingly:

| Column prefix | Commodity | Feeds converter / sector |
|---|---|---|
| `demand` | `Elec` | Direct electricity demand |
| `MDV_el` | `T_MDV_el` | `MDV_el` — electric MDV |
| `MDV_Th` | `T_MDV_th` | `MDV` — fossil MDV (residual) |
| `MDV_BF` | `T_MDV_BF` | `MDV_BF` — biofuel MDV |
| `HDV_el` | `T_HDV_el` | `HDV_el` — electric HDV |
| `HDV_Th` | `T_HDV_th` | `HDV` — fossil HDV (residual) |
| `HDV_BF` | `T_HDV_BF` | `HDV_BF` — biofuel HDV |
| `LDV_el` | `T_LDV_el` | `LDV_el` — electric LDV |
| `LDV_Th` | `T_LDV_th` | `LDV` — fossil LDV (residual) |
| `LDV_BF` | `T_LDV_BF` | `LDV_BF` — biofuel LDV |
| `BUS_el` | `T_Bus_el` | `Bus_el` — electric bus |
| `BUS_Th` | `T_Bus_th` | `Bus` — fossil bus (residual) |
| `2W_el` | `T_Two_wheel_el` | `Two_wheel_el` — electric two-wheeler |
| `2W_th` | `T_Two_wheel_th` | `Two_wheel` — fossil two-wheeler (residual) |
| `Marine_TH` | `T_Marine_f_th` | `HFO` — heavy-fuel-oil marine transport |
| `Marine_E` | `Dummy_EL` | *(see note below)* |
| `Marine_M` | `Methanol` | Marine transport running directly on methanol |
| `Marine_BEV` | `T_ship_el` | `Ship_BEV` — electric ships |
| `AVIA_TH` | `T_Aviation_th` | `Aviation` — fossil aviation (residual) |
| `AVIA_EL` | `T_Aviation_el` | `Aviation_el` — electric aviation |
| `AVIA_E` | `eKerosene` | *(see note below)* |
| `HC_B` | `Heat_cooking` | `cook_b` — biomass cooking (residual) |
| `HC_L` | `T_cook_LPG` | `cook_LPG` — LPG cooking |
| `HC_el` | `T_cook_el` | `cook_el` — electric cooking |
| `HI_D` | `Heat_industry` | `Industry` — diesel industrial boilers (residual) |
| `HI_EH` | `T_Industry_EH` | `Industry_EH` — placeholder industrial-heat link |
| `DHW_E` | `DHW_el` | `DW_Electric_converter` — electric water heating |
| `DHW_L` | `DHW_LPG` | `DW_LPG_converter` — LPG water heating |

> **Note on `Marine_E` and `AVIA_E`:** these two columns feed the
> commodities `Dummy_EL` and `eKerosene` directly. Despite the naming,
> they do not route through any dedicated electric-marine or e-kerosene
> aviation converter — no such converter exists in the model.

**Transform applied by the script:** for each node/year, the relevant
columns are selected, divided by 1000 (MWh → GWh), sign-flipped (`* -1`,
since demand is a *sink*), then registered as a fixed `sourcesink_profile`
+ `sourcesink_config` for that commodity/node/year.

## Column-count check

- Activity profiles: 14 nodes × 12 techs = **168**
- Demand profiles: 14 nodes × (13 cols × 2 years [2020, 2030] + 28 cols × 2
  years [2040, 2050]) = 14 × (26 + 56) = **1,148**
- Plus the `I` index column = **1,317 total**, matching the file.

## Known quirk

One `demand_CI_2020` column appears out of its expected block position
(at the very end of the file rather than grouped with CI's other 2020
columns). This is exactly the kind of copy/paste ordering slip the model
scripts' `_region_to_node` helper is written to be robust to — it scans
every column name for a valid region code regardless of position, so this
doesn't cause a lookup failure, but it's worth knowing about if you're
diffing or re-deriving this file from a spreadsheet source.
