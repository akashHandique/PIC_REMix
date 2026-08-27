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
├── _input/            # Model input data (demand & renewable-resource profiles)
├── Process/            # The four REMix scenario scripts — start here to run the model
├── GDX_results/        # GAMS .gdx solver outputs land here after a run
├── Visualization/       # Post-processing & plotting tools that read the .gdx files
├── LICENSE              # MIT
└── .gitignore
```

Each subfolder has its own README with the full detail; this page is the
map between them.

## Workflow

```
_input/  →  Process/<scenario>.py  →  GDX_results/*.gdx  →  Visualization/Gdx_plot.py
 (data)        (build + solve)         (solver output)         (LCOE, capacity,
                                                                  dispatch & SOC plots)
```

1. **`_input/`** — holds the demand and renewable-resource profile CSV
   (`Copy of IP_2040_2050_14_PIC - Copy.csv`) that every scenario script
   reads. Node labels, converter activity profiles, and fixed demand
   series all come from this one file.
2. **`Process/`** — the model itself: four Python scripts, one per
   scenario, that build the REMix parameter set (nodes, converters,
   storage, fuel imports, demand, and optionally inter-island shipping)
   and hand it to GAMS to solve. **This is where you run the model.** See
   [`Process/README.md`](Process/README.md) for the full converter
   glossary and a breakdown of what differs between scenarios.
3. **`GDX_results/`** — where each scenario's solved `.gdx` output file
   lands (e.g. `IP_2050_Final_S23_minload.gdx`). See
   [`GDX_results/README.md`](GDX_results/README.md).
4. **`Visualization/`** — `Gdx_plot.py`, a set of independent plotting
   tools that read a `.gdx` from `GDX_results/` and produce LCOE
   breakdowns, capacity/generation charts, battery and hydrogen
   state-of-charge heatmaps, and electrolyser activity plots. See
   [`Visualization/README.md`](Visualization/README.md).

## The four scenarios

| Script | Result file | Inter-island shipping | E-fuel imports |
|---|---|:---:|:---:|
| `Process/remix_pacific_model_S1_std.py` | `IP_2050_Final_SS1_minload` | — | — |
| `Process/remix_pacific_model_S2_std.py` | `IP_2050_Final_SS2_minload` | ✅ | — |
| `Process/remix_pacific_model_S13_std.py` | `IP_2050_Final_SS13_minload` | — | ✅ |
| `Process/remix_pacific_model_S23_std.py` | `IP_2050_Final_SS23_minload` | ✅ | ✅ |

S1 is the baseline: all four scripts share the same 14-node model core
(power generation, heat generation, transport, power-to-X converters, and
storage) — they only differ in whether ammonia/methanol/e-kerosene can
move between islands by ship (S2/S23) and/or be imported directly (S13/S23).
Full detail in [`Process/README.md`](Process/README.md).

## Requirements

- Python 3 with `pandas`, `numpy`, and the `remix.framework` package
  (used by the `Process/` scripts).
- A licensed **GAMS** installation to actually solve the model and to read
  `.gdx` files back out (`gdxpds`, used by `Visualization/Gdx_plot.py`).
- `matplotlib`, `openpyxl` for plotting/export in `Visualization/`.

(No repo-level `requirements.txt` exists yet — see the subfolder READMEs
for the exact package list each part needs.)

## Quick start

```bash
cd Process
python remix_pacific_model_S1_std.py      # build + solve the baseline scenario
# → writes IP_2050_Final_SS1_minload.gdx into ../GDX_results/

cd ../Visualization
python gdx_plot.py ... --gdx ../GDX_results/IP_2050_Final_SS1_minload.gdx
```

## License

MIT — see [`LICENSE`](LICENSE).
