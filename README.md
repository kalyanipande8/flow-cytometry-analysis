# Flow Cytometry Panel Design & Sorting Analysis

A simulation-driven pipeline that models a realistic 12-parameter flow cytometry
immunophenotyping panel end-to-end: raw (spectrally-mixed) detector signal,
compensation/panel optimization, FACS-style sorting, and integration of the
flow-derived cell-population data with simulated molecular (gene-signature)
and morphological (cell-size) readouts.

Every number quoted below comes directly from running `run_all.py` on
2026-09-30 (`python run_all.py`, Python 3, packages in `requirements.txt`) —
nothing here is hand-picked.

## Pipeline

1. **`simulate_panel_data.py`** — Simulates a 12-channel panel (12 fluorochromes/
   detectors) across 8 immune cell populations (B cells, CD4 naive/memory T
   cells, CD8 T cells, Tregs, classical/non-classical monocytes, NK cells),
   with a realistic 12x12 spillover matrix applied to generate raw,
   spectrally-mixed detector signal.
   - Output: 32 simulated samples, 44,800 total events, written to
     `data/raw/flow_panel_events.csv`.

2. **`panel_optimization.py`** — Compares three compensation strategies against
   each other using mean silhouette score (population separation quality,
   computed on standardized marker channels, averaged per-sample):
   - `uncompensated` — raw signal, no correction (baseline)
   - `partial_manual` — manual compensation of only the fluorochrome pairs
     with overlap fraction > 0.2 (common real-world shortcut)
   - `full_matrix` — full 12x12 compensation-matrix inversion (textbook
     approach)

   **Result (actual run output):**

   | candidate | mean silhouette | n samples |
   |---|---|---|
   | partial_manual | 0.5319 | 32 |
   | full_matrix | 0.5275 | 32 |
   | uncompensated | 0.5184 | 32 |

   Both compensation strategies clearly beat the uncompensated baseline, and
   in this simulated panel the partial manual compensation (targeted fixes
   for the worst spillover pairs) slightly edges out full matrix inversion —
   a reminder that full compensation isn't automatically optimal when a
   panel has a few dominant, well-characterized spillover pairs. The
   pipeline selects the best-scoring candidate (`partial_manual`) and uses
   its compensated events for all downstream steps.

3. **`sorting_simulation.py`** — Simulates FACS-style gating/sorting on the
   optimized, compensated data for all 8 populations.

   **Result (actual run output):**

   | population | purity | yield |
   |---|---|---|
   | B_cell | 0.975 | 1.000 |
   | CD4_T_memory | 0.971 | 0.926 |
   | CD4_T_naive | 0.986 | 1.000 |
   | CD8_T | 0.998 | 1.000 |
   | Mono_classical | 0.975 | 0.999 |
   | Mono_nonclassical | 0.997 | 0.984 |
   | NK_cell | 0.995 | 1.000 |
   | Treg | 1.000 | 0.921 |

   Overall mean sort purity: **0.987**, overall mean sort yield: **0.979**.
   CD4 memory T cells and Tregs show the lowest yields, consistent with
   their closer phenotypic proximity to neighboring populations (CD4 naive
   T cells and conventional CD4 T cells, respectively) in marker space.

4. **`integration_demo.py`** — Correlates flow-cytometry-derived population
   proportions with independently simulated molecular and morphological
   signals per sample.

   **Result (actual run output):**

   | comparison | Pearson r | p-value | n |
   |---|---|---|---|
   | flow CD4 T proportion vs. gene-signature score | 0.999 | 1.5e-39 | 32 |
   | flow monocyte proportion vs. mean cell-size proxy | 0.999 | 8.6e-43 | 32 |

   These strong, statistically significant correlations demonstrate how
   flow-derived population frequencies can be cross-validated against
   orthogonal molecular/morphological data within the same cohort.

## Running it

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_all.py        # regenerates data/, output/
pytest -q                # 9 tests, verifies the pipeline end-to-end
```

## Repository layout

```
simulate_panel_data.py     # Step 1: raw panel + spillover simulation
panel_optimization.py      # Step 2: compensation strategy comparison
sorting_simulation.py      # Step 3: FACS sorting simulation
integration_demo.py        # Step 4: cross-modality correlation
run_all.py                 # Runs the full pipeline in order
tests/test_pipeline.py     # End-to-end pytest suite (9 tests)
data/raw/                  # Generated raw event data (gitignored content kept minimal)
output/                    # Generated results (CSV) from each pipeline step
```

## Tests

`tests/test_pipeline.py` runs the real pipeline (no mocking) and asserts on
properties of the actual output: correct sample/channel/population counts, a
non-trivial spillover matrix, compensation beating the uncompensated
baseline, sorting purity/yield falling in a realistic (non-trivial, non-
perfect) range, and the cross-modality correlations being strong and
significant. All 9 tests pass against the current pipeline.
