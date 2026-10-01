"""
Cross-modality integration demo.

For each of the 30+ primary samples, correlates the flow-cytometry-derived
immune population proportions (from the optimized panel's sort gates in
sorting_simulation.py) with simulated molecular (bulk gene-expression proxy)
and morphological (mean cell-size proxy) readouts on the *same* samples --
the kind of cross-modality check a real translational study performs to
confirm the flow phenotype is biologically consistent with orthogonal data.

Two biologically motivated linkages are simulated and then recovered purely
from the data (nothing is hardcoded into the correlation step):

1. CD4 T-cell proportion (flow) vs. a CD4/CD3 T-cell gene-signature score
   (molecular proxy) -- these should track together since both reflect
   T-helper cell abundance.
2. Monocyte proportion (flow) vs. mean forward-scatter-proxy cell size
   (morphological proxy) -- monocytes are the largest PBMC subset, so
   samples with more monocytes should show a larger mean cell size.
"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

MOL_NOISE_SD = 0.12
MORPH_NOISE_SD = 0.08
RNG_SEED = 7


def flow_proportions(sort_df):
    """Flow-derived population proportions per sample, from sort-gate calls
    (n_sorted / total cells in sample), not ground truth -- these are the
    numbers an actual flow analysis would report.
    """
    totals = sort_df.groupby("sample_id")["n_true"].sum() / 8  # n_true summed 8x (once per pop) -> /8 = total cells
    totals = totals.rename("total_cells")
    wide = sort_df.pivot(index="sample_id", columns="population", values="n_sorted")
    prop = wide.div(totals, axis=0)
    return prop, totals


def simulate_molecular_and_morphological(prop_df, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    cd4_prop = prop_df[["CD4_T_naive", "CD4_T_memory"]].sum(axis=1)
    mono_prop = prop_df[["Mono_classical", "Mono_nonclassical"]].sum(axis=1)

    # molecular proxy: CD4/CD3 T-helper gene signature score, linearly
    # driven by true CD4 T proportion plus independent biological/assay noise
    gene_signature = 2.0 + 6.0 * cd4_prop + rng.normal(0, MOL_NOISE_SD, size=len(prop_df))

    # morphological proxy: mean cell size (arbitrary units), driven by
    # monocyte proportion (monocytes are the largest PBMC subset)
    cell_size = 8.0 + 5.0 * mono_prop + rng.normal(0, MORPH_NOISE_SD, size=len(prop_df))

    out = pd.DataFrame({
        "sample_id": prop_df.index,
        "cd4_t_proportion_flow": cd4_prop.values,
        "monocyte_proportion_flow": mono_prop.values,
        "gene_signature_score": gene_signature,
        "mean_cell_size_um_proxy": cell_size,
    })
    return out


def main():
    sort_df = pd.read_csv("output/sorting_results.csv")
    prop_df, totals = flow_proportions(sort_df)
    integ_df = simulate_molecular_and_morphological(prop_df)
    integ_df.to_csv("output/integration_data.csv", index=False)

    r_gene, p_gene = pearsonr(integ_df["cd4_t_proportion_flow"], integ_df["gene_signature_score"])
    r_size, p_size = pearsonr(integ_df["monocyte_proportion_flow"], integ_df["mean_cell_size_um_proxy"])

    results = pd.DataFrame([
        {"comparison": "flow CD4 T proportion vs gene-signature score", "pearson_r": r_gene, "p_value": p_gene, "n": len(integ_df)},
        {"comparison": "flow monocyte proportion vs mean cell-size proxy", "pearson_r": r_size, "p_value": p_size, "n": len(integ_df)},
    ])
    results.to_csv("output/integration_correlations.csv", index=False)
    print(results.to_string(index=False))
    return integ_df, results


if __name__ == "__main__":
    main()
