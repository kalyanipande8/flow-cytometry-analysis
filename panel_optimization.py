"""
Panel/compensation optimization.

Given the raw (spectrally-mixed) 12-channel signal simulated by
simulate_panel_data.py, this module compares candidate compensation
strategies -- the real decision a flow cytometry panel designer has to make
before sorting -- and picks the one that gives the best population
separation, quantified with a real, computed metric: average silhouette
score across the 8 known immune populations (computed on standardized
marker channels).

Candidates
----------
1. "uncompensated"   - sort/gate directly on raw detector signal (no
                        correction). This is the baseline every real panel
                        must beat before it's usable.
2. "partial_manual"  - a common real-world shortcut: manually compensate
                        only the handful of fluorochrome pairs an operator
                        visually flags as bad (overlap fraction > 0.2 in the
                        true spillover matrix), inverting a partial
                        spillover matrix built from only those entries.
3. "full_matrix"     - full compensation-matrix inversion using the
                        complete measured spillover matrix (all 12x12
                        entries) -- the actual textbook approach.

The panel is "optimized" by picking whichever candidate has the highest
mean silhouette score, computed independently per sample and averaged.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from simulate_panel_data import CHANNELS, build_spillover_matrix

N_CHANNELS = len(CHANNELS)


def partial_spillover_matrix(threshold=0.2):
    """Spillover matrix containing only the 'obviously bad' pairs an
    operator would notice and manually compensate (overlap > threshold),
    plus the identity diagonal. Smaller, genuinely-present overlaps below
    threshold are left uncorrected -- this is the realistic failure mode of
    a manual/partial compensation approach.
    """
    S_full, _ = build_spillover_matrix()
    S_partial = np.eye(N_CHANNELS)
    for i in range(N_CHANNELS):
        for j in range(N_CHANNELS):
            if i != j and S_full[i, j] > threshold:
                S_partial[i, j] = S_full[i, j]
    return S_partial


def apply_compensation(raw, S):
    """raw = true @ S  =>  true_hat = raw @ S^-1."""
    S_inv = np.linalg.inv(S)
    return raw @ S_inv


def candidate_matrices():
    S_full, _ = build_spillover_matrix()
    return {
        "uncompensated": np.eye(N_CHANNELS),   # apply_compensation with identity = no-op
        "partial_manual": partial_spillover_matrix(),
        "full_matrix": S_full,
    }


def score_candidate(df, comp_matrix, max_cells_per_sample=400, seed=0):
    """Mean silhouette score across samples for one compensation candidate.
    Subsamples cells per sample for tractable silhouette computation.
    """
    rng = np.random.default_rng(seed)
    raw_cols = [f"raw_{c}" for c in CHANNELS]
    scores = []
    for sample_id, g in df.groupby("sample_id"):
        if g["population"].nunique() < 2:
            continue
        if len(g) > max_cells_per_sample:
            g = g.sample(max_cells_per_sample, random_state=seed)
        raw = g[raw_cols].to_numpy()
        comp = apply_compensation(raw, comp_matrix)
        X = StandardScaler().fit_transform(comp)
        try:
            s = silhouette_score(X, g["population"].to_numpy())
        except ValueError:
            continue
        scores.append(s)
    return float(np.mean(scores)), scores


def optimize_panel(df):
    """Evaluate all candidates, return (results_df, best_candidate_name,
    compensated-events DataFrame using the best candidate).
    """
    candidates = candidate_matrices()
    results = []
    for name, S in candidates.items():
        mean_sil, per_sample = score_candidate(df, S)
        results.append({
            "candidate": name,
            "mean_silhouette": mean_sil,
            "n_samples_scored": len(per_sample),
        })
    results_df = pd.DataFrame(results).sort_values("mean_silhouette", ascending=False).reset_index(drop=True)
    best_name = results_df.iloc[0]["candidate"]
    best_S = candidates[best_name]

    raw_cols = [f"raw_{c}" for c in CHANNELS]
    comp = apply_compensation(df[raw_cols].to_numpy(), best_S)
    comp_df = df[["sample_id", "cell_id", "population"]].copy()
    for i, c in enumerate(CHANNELS):
        comp_df[f"comp_{c}"] = comp[:, i]

    return results_df, best_name, comp_df


def main():
    df = pd.read_csv("data/raw/flow_panel_events.csv")
    results_df, best_name, comp_df = optimize_panel(df)
    results_df.to_csv("output/panel_optimization_results.csv", index=False)
    comp_df.to_csv("output/compensated_events.csv", index=False)
    print(results_df.to_string(index=False))
    print(f"\nBest panel/compensation configuration: {best_name}")
    return results_df, best_name, comp_df


if __name__ == "__main__":
    main()
