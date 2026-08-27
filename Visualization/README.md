# Visualization/

Plotting & post-processing tools for the REMix Pacific-Island-Countries
GDX results — levelized costs, installed capacities, hourly generation
profiles, and storage state-of-charge.

## Where it sits in the repo

```
<repo-root>/
├── _input/          # Model input data
├── Process/         # The four REMix scenario scripts
├── GDX_results/     # GAMS .gdx outputs        <-- read from here (untracked; create it)
└── Visualization/   # you are here
    └── Gdx_plot.py
```

**Run the tool from inside this `Visualization/` folder.** It reads the
`.gdx` from the repository-root `GDX_results/` folder one level up
(`../GDX_results/…`), so no paths need editing — just launch from here
(e.g. set `Visualization/` as the working directory in Spyder/VS Code).

Note the capital **G** in `Gdx_plot.py`: the filename is case-sensitive on
Linux and macOS.

## Choosing which scenario to plot

**Everything follows one line.** `Gdx_plot.py` loads exactly one `.gdx`,
once, at the top of the file, and derives the workbook name and the whole
figure tree from it:

```python
GDX_PATH    = "../GDX_results/IP_2050_Final_S1.gdx"          # line 63  <-- edit this
data        = gdxpds.to_dataframes(GDX_PATH)                 # line 65  the only load
OUTPUT_PATH = f"LCO_results_{Path(GDX_PATH).stem}.xlsx"      # line 67
FIG_ROOT    = f"figures/{SCENARIO}"                          # derived from GDX_PATH
```

**To plot a different scenario, change `GDX_PATH` and re-run.** Nothing
else needs editing — the workbook name and every figure directory move
with it, so results cannot end up labelled with the wrong scenario.

| `GDX_PATH` | LCO workbook (written *and* read) | Figure tree |
|---|---|---|
| `IP_2050_Final_S1.gdx` | `LCO_results_IP_2050_Final_S1.xlsx` | `figures/S_1/…` |
| `IP_2050_Final_S2.gdx` | `LCO_results_IP_2050_Final_S2.xlsx` | `figures/S_2/…` |
| `IP_2050_Final_S13.gdx` | `LCO_results_IP_2050_Final_S13.xlsx` | `figures/S_13/…` |
| `IP_2050_Final_S23.gdx` | `LCO_results_IP_2050_Final_S23.xlsx` | `figures/S_23/…` |

The scenario tag is taken from the last underscore-separated token of the
GDX filename (`…_S23` → `S_23`). If a filename doesn't end in `S<digits>`,
the last token is used verbatim rather than failing.

> **`--gdx` is not honoured.** Each section still accepts a `--gdx`
> argument for historical reasons, but the value is not used: the sections
> read the module-level `data` loaded at line 65. Change `GDX_PATH`
> instead. The `--out`, `--fmt`, `--dpi`, `--islands` and `--years`
> arguments do work.

## Sections

The file is a toolbox: 15 sections, each with its own `main()` and
`if __name__ == "__main__":` block. Running `python Gdx_plot.py` executes
**all of them in sequence**; to run one, execute its cell/selection from
an IDE.

Figure directories below are shown relative to `FIG_ROOT`
(= `figures/<SCENARIO>`).

| # | Section | Purpose | Figure output |
|---:|---|---|---|
| 1 | `pic_lco_assessment` | Levelized costs from the GDX → LCO results `.xlsx` | *(writes the workbook)* |
| 2 | `pic_lco_plots` | Plot LCO breakdowns from the `.xlsx` | `<FIG_ROOT>/` |
| 3 | `pic_generation_profiles` | Hourly electricity generation profiles per island | `generation_profiles` |
| 4 | `pic_battery_soc` | Battery state-of-charge heatmaps | `battery_soc` |
| 5 | thermal storage SOC | THSS state-of-charge heatmaps | `thermal_soc` |
| 6 | `battery_viz` | Battery capacity and generation charts | `battery_charts` |
| 7 | `pic_h2_storage` | Hydrogen storage state-of-charge heatmaps | `hydrogen_storage` |
| 8 | sustainable-fuel storage SOC | Ammonia / methanol / e-kerosene storage SOC | `ammonia_storage`, `methanol_storage`, `ekerosene_storage` |
| 9 | e-fuel synthesis activity | Ammonia / methanol / FTL converter activity | `ammonia_synthesis_activity`, `methanol_synthesis_activity`, `ekerosene_synthesis_activity` |
| 10 | `pic_electrolyzer_activity` | Electrolyser (AEL) activity heatmaps | `electrolyzer_activity` |
| 11 | heat pump activity | Heat-pump activity heatmaps | `heat_pump_activity` |
| 12 | capacity overview | Capacities, generation and end-use by island | `capacity_overview` |
| 13 | `pic_heat_generation` | Heat generation by technology | `heat_generation` |
| 14 | AEL capacity | Installed electrolyser capacity per island | `ael_capacity` |
| 15 | system cost | Stacked system-cost components by year | `system_cost` |

Sections 1 and 2 form a pipeline: section 1 computes the levelized costs
and writes the workbook; section 2 reads that workbook and plots it.
Section 2 is the only section that never touches the GDX, so it is the
only one that runs **without** a GAMS installation.

## Requirements

`gdxpds`, `pandas`, `numpy`, `matplotlib`, `openpyxl`. **`gdxpds` needs a
GAMS installation** to read `.gdx`. (Section 2 works from the generated
`.xlsx` without GAMS.)

## Generated files

- `LCO_results_*.xlsx` — written here by section 1.
- Figures — written under `figures/<SCENARIO>/`, created automatically.

Add these to the repository `.gitignore` if you don't want them committed:

```
Visualization/LCO_results_*.xlsx
Visualization/figures/
```
