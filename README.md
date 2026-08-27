# REMix-PIC — Energy Systems Modelling for Pacific Island Countries

A multi-node, multi-year (2020 / 2030 / 2040 / 2050) capacity-expansion
energy system model for 14 Pacific Island Countries, built on the DLR
[REMix](https://dlr-ve.gitlab.io/esy/remix/framework/dev/index.html) framework and solved with GAMS.
The model co-optimises power, land / marine / aviation transport, cooking,
industry, domestic hot water, water desalination and synthetic-fuel
(hydrogen / ammonia / methanol / e-kerosene) production, across four
inter-island-trade scenarios, and ships with tooling to plot the results.

## Repository layout

```
REMix-PIC/
├── _input/          # Model input data (demand & renewable-resource profiles)
├── Process/         # The four REMix scenario scripts — start here to run the model
├── Visualization/   # Post-processing & plotting tools that read the .gdx files
├── LICENSE          # MIT
└── .gitignore
```

Each subfolder has its own README with the full detail; this page is the
map between them.

`GDX_results/` is **not** tracked in the repository (git does not store
empty directories). Create it at the repository root before your first
run — both the model output and the plotting tools expect it there:

```bash
mkdir GDX_results
```

## Workflow

```
_input/  →  Process/<scenario>.py  →  GDX_results/*.gdx  →  Visualization/Gdx_plot.py
 (data)        (build + solve)         (solver output)         (LCO, capacity,
                                                                dispatch & SOC plots)
```

1. **`_input/`** — holds the demand and renewable-resource profile CSV
   (`Hourly_demand_and_resource_profiles.csv`) that every scenario script
   reads. Node labels, converter activity profiles, and fixed demand
   series all come from this one file. See
   [`_input/README.md`](_input/README.md).
2. **`Process/`** — the model itself: four Python scripts, one per
   scenario, that build the REMix parameter set (nodes, converters,
   storage, fuel imports, demand, and optionally inter-island shipping)
   and hand it to GAMS to solve. **This is where you run the model.** See
   [`Process/README.md`](Process/README.md) for the full converter
   glossary and a breakdown of what differs between scenarios.
3. **`GDX_results/`** — where each scenario's solved `.gdx` output file
   lands (e.g. `IP_2050_Final_S23.gdx`). Untracked; create it yourself.
4. **`Visualization/`** — `Gdx_plot.py`, a toolbox of plotting sections
   that read one `.gdx` from `GDX_results/` and produce LCO breakdowns,
   capacity/generation charts, battery and hydrogen state-of-charge
   heatmaps, and electrolyser activity plots. See
   [`Visualization/README.md`](Visualization/README.md).

## The four scenarios

| Script | Result file | Inter-island shipping | E-fuel imports |
|---|---|:---:|:---:|
| `Process/remix_pacific_model_S1_std.py` | `IP_2050_Final_S1` | — | — |
| `Process/remix_pacific_model_S2_std.py` | `IP_2050_Final_S2` | ✅ | — |
| `Process/remix_pacific_model_S13_std.py` | `IP_2050_Final_S13` | — | ✅ |
| `Process/remix_pacific_model_S23_std.py` | `IP_2050_Final_S23` | ✅ | ✅ |

S1 is the baseline: all four scripts share the same 14-node model core
(power generation, heat generation, transport, power-to-X converters, and
storage) — they only differ in whether ammonia/methanol/e-kerosene can
move between islands by ship (S2/S23) and/or be imported directly (S13/S23).
Full detail in [`Process/README.md`](Process/README.md).

## Requirements

- Python 3 with `pandas`, `numpy`, and the `remix.framework` package
  (used by the `Process/` scripts).
- A licensed **GAMS** installation to solve the model and to read
  `.gdx` files back out (`gdxpds`, used by `Visualization/Gdx_plot.py`).
- `matplotlib`, `openpyxl` for plotting/export in `Visualization/`.

(No repo-level `requirements.txt` exists yet — see the subfolder READMEs
for the exact package list each part needs.)

## Quick start

```bash
mkdir -p GDX_results

cd Process
python remix_pacific_model_S1_std.py      # build + solve the baseline scenario
# → writes IP_2050_Final_S1.gdx

# move/copy the .gdx into GDX_results/ if your GAMS run does not write it there

cd ../Visualization
# set GDX_PATH at the top of Gdx_plot.py to the .gdx you want, then run one section
python Gdx_plot.py
```

Note the capital **G** in `Gdx_plot.py` — the filename is case-sensitive
on Linux and macOS.

The plotting toolbox reads **one** `.gdx`, chosen by the `GDX_PATH`
constant at the top of `Gdx_plot.py`. See
[`Visualization/README.md`](Visualization/README.md) before running it.

## License

MIT — see [`LICENSE`](LICENSE).
