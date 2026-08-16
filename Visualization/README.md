# visualization/

Plotting & post-processing tools for the REMix / HOMER Pacific-Island-Countries
GDX results — LCOE, installed capacities, hourly generation profiles, and
storage state-of-charge.

## Where it sits in the repo

```
<repo-root>/
├── input/            # Excel inputs for the REMix model
├── process/          # main REMix model script
├── GDX_results/      # GAMS .gdx outputs        <-- read from here
└── visualization/    # you are here
    └── gdx_plot.py
```

**Run the tools from inside this `visualization/` folder.** They read the `.gdx`
from the repo-root `GDX_results/` folder one level up (`../GDX_results/…`), so no
paths need editing — just launch from here (e.g. set `visualization/` as the
working directory in Spyder/VS Code). Override any input with `--gdx`:

```bash
python gdx_plot.py ... --gdx ../GDX_results/IP_2050_Final_S23_minload.gdx
```

## Tools (sections in `gdx_plot.py`)

| Tool | Purpose |
|---|---|
| `pic_lco_assessment` | Extract commodity values from a GDX → LCO results `.xlsx` |
| `pic_lco_plots` | Plot LCOE / cost breakdowns from the LCO `.xlsx` |
| `pic_generation_profiles` | Hourly electricity generation profiles per island |
| `pic_battery_soc` | Battery state-of-charge heatmaps |
| `battery_viz` | Battery capacity and generation charts |
| `pic_h2_storage` | Hydrogen storage state-of-charge heatmaps |
| `pic_electrolyzer_activity` | Electrolyser (AEL) activity heatmaps |
| `pic_heat_generation` | Heat generation by technology |

Each section is an independent tool with its own `main()` and CLI (see the
index in the header of `gdx_plot.py`). Run a section from an IDE, or copy it
into its own `.py` file.

## Requirements

`gdxpds`, `pandas`, `numpy`, `matplotlib`, `openpyxl` — add these to the main
repo's `requirements.txt`. **`gdxpds` needs a GAMS installation** to read `.gdx`.
(`pic_lco_plots` works from the generated `.xlsx` without GAMS.)

## Generated files

- `LCO_results_*.xlsx` — written here by `pic_lco_assessment`.
- Figures — written to `figures/` / `outputs/` (created automatically).

Add these to the main repo's `.gitignore` if you don't want them committed, e.g.:

```
visualization/LCO_results_*.xlsx
visualization/figures/
visualization/outputs/
```
