"""
Cell-sorting yield/purity simulation.

Using the optimized (best-compensated) 12-parameter panel selected by
panel_optimization.py, this module simulates the sort-gate decision a real
FACS operator makes for each of the 8 target populations, for every one of
the 30+ primary samples, and computes real sort purity and yield:

    purity = (sorted events that are truly the target population)
             / (all sorted events)
    yield  = (sorted events that are truly the target population)
             / (all true target-population events in that sample)

Sort gates are simple boolean marker-threshold gates (the same kind of gate
an operator draws on a cytometer), defined once on the compensated marker
scale. Because the events were compensated with the *optimized* panel
config, these fixed-threshold gates should recover realistic-but-imperfect
purity/yield (spillover residual + biological overlap between adjacent
populations, e.g. CD4 memory/naive, classical/non-classical monocytes,
still causes some cross-contamination -- as in a real sort).
"""
import numpy as np
import pandas as pd

from simulate_panel_data import CHANNELS

POS_THRESH = 2.0
NEG_THRESH = 2.0

# marker channel names without the "comp_" prefix, matching CHANNELS entries
# CHANNELS: FITC_CD3, PE_CD25, PerCPCy55_CD127, PECy7_CD19, APC_CD56,
#           APCCy7_CD14, BV421_CD4, BV510_CD8, BV605_CD16, BV711_CD45RA,
#           BUV395_CCR7, BUV737_HLADR
GATES = {
    "CD4_T_naive":       [("FITC_CD3", ">"), ("BV421_CD4", ">"), ("BV711_CD45RA", ">"), ("BUV395_CCR7", ">")],
    "CD4_T_memory":      [("FITC_CD3", ">"), ("BV421_CD4", ">"), ("BV711_CD45RA", "<"), ("BUV395_CCR7", "<")],
    "CD8_T":             [("FITC_CD3", ">"), ("BV510_CD8", ">")],
    "Treg":              [("FITC_CD3", ">"), ("BV421_CD4", ">"), ("PE_CD25", ">"), ("PerCPCy55_CD127", "<")],
    "B_cell":            [("PECy7_CD19", ">")],
    "NK_cell":           [("APC_CD56", ">"), ("FITC_CD3", "<")],
    "Mono_classical":    [("APCCy7_CD14", ">"), ("BV605_CD16", "<")],
    "Mono_nonclassical": [("APCCy7_CD14", ">"), ("BV605_CD16", ">")],
}


def apply_gate(df, gate_rules):
    mask = np.ones(len(df), dtype=bool)
    for channel, sign in gate_rules:
        col = f"comp_{channel}"
        if sign == ">":
            mask &= df[col].to_numpy() > POS_THRESH
        else:
            mask &= df[col].to_numpy() < NEG_THRESH
    return mask


def simulate_sorting(comp_df):
    """Return a DataFrame with one row per (sample, target population):
    n_true, n_sorted, n_true_positive, purity, yield.
    """
    rows = []
    for sample_id, g in comp_df.groupby("sample_id"):
        g = g.reset_index(drop=True)
        for pop, rules in GATES.items():
            n_true = int((g["population"] == pop).sum())
            sorted_mask = apply_gate(g, rules)
            n_sorted = int(sorted_mask.sum())
            n_tp = int(((g["population"] == pop) & sorted_mask).sum())
            purity = n_tp / n_sorted if n_sorted > 0 else np.nan
            yield_ = n_tp / n_true if n_true > 0 else np.nan
            rows.append({
                "sample_id": sample_id,
                "population": pop,
                "n_true": n_true,
                "n_sorted": n_sorted,
                "n_true_positive": n_tp,
                "purity": purity,
                "yield": yield_,
            })
    return pd.DataFrame(rows)


def main():
    comp_df = pd.read_csv("output/compensated_events.csv")
    sort_df = simulate_sorting(comp_df)
    sort_df.to_csv("output/sorting_results.csv", index=False)

    summary = sort_df.groupby("population")[["purity", "yield"]].mean().reset_index()
    summary.to_csv("output/sorting_summary_by_population.csv", index=False)

    overall_purity = sort_df["purity"].mean()
    overall_yield = sort_df["yield"].mean()
    print(summary.to_string(index=False))
    print(f"\nOverall mean sort purity: {overall_purity:.3f}")
    print(f"Overall mean sort yield:  {overall_yield:.3f}")
    return sort_df, summary


if __name__ == "__main__":
    main()
