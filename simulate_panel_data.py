"""
Simulate a 12-parameter (12-fluorochrome) flow cytometry panel across 30+
primary immune-cell samples, with realistic spectral overlap between
fluorochromes that requires compensation-matrix correction.

Disclaimer: all data here is synthetically simulated. No real patient,
donor, or employer data is used.

Design
------
12-color panel (fluorochrome -> marker), a plausible panel for a T/B/NK/
monocyte immunophenotyping + Treg sort:

    FITC        - CD3
    PE          - CD25
    PerCP-Cy5.5 - CD127
    PE-Cy7      - CD19
    APC         - CD56
    APC-Cy7     - CD14
    BV421       - CD4
    BV510       - CD8
    BV605       - CD16
    BV711       - CD45RA
    BUV395      - CCR7
    BUV737      - HLA-DR

8 immune populations are simulated per sample from a primary PBMC-like
sample, each with a population-specific 12-dim mean expression vector
(in arcsinh-like fluorescence space):

    CD4 T (naive), CD4 T (memory), CD8 T, Treg (CD4+CD25+CD127lo),
    B cell, NK cell, Classical monocyte, Non-classical monocyte

Spectral overlap: several fluorochrome pairs are genuinely
spectrally close (e.g. FITC/PerCP-Cy5.5 into PE, BV421/BV510/BV605 bleed
into each other, PE-Cy7/APC-Cy7 tandem degradation bleeds into APC) and are
modeled with a non-trivial spillover matrix S (12x12, S[i,j] = fraction of
fluorochrome i's signal detected in channel j). Detected ("raw") signal is

    raw = true_signal @ S + noise

Only a true compensation matrix (S^-1) fully separates populations again;
partial/no correction leaves populations confounded, which panel_optimization.py
quantifies.
"""
import numpy as np
import pandas as pd

RNG_SEED = 42

CHANNELS = [
    "FITC_CD3", "PE_CD25", "PerCPCy55_CD127", "PECy7_CD19", "APC_CD56",
    "APCCy7_CD14", "BV421_CD4", "BV510_CD8", "BV605_CD16", "BV711_CD45RA",
    "BUV395_CCR7", "BUV737_HLADR",
]
N_CHANNELS = len(CHANNELS)

POPULATIONS = [
    "CD4_T_naive", "CD4_T_memory", "CD8_T", "Treg",
    "B_cell", "NK_cell", "Mono_classical", "Mono_nonclassical",
]

# Population mixing fractions in a typical PBMC-derived primary sample
POP_BASE_FRACTION = {
    "CD4_T_naive": 0.17, "CD4_T_memory": 0.15, "CD8_T": 0.18, "Treg": 0.04,
    "B_cell": 0.10, "NK_cell": 0.10, "Mono_classical": 0.18, "Mono_nonclassical": 0.08,
}

# Per-population mean expression (arbitrary fluorescence units, log-ish scale)
# order matches CHANNELS: CD3, CD25, CD127, CD19, CD56, CD14, CD4, CD8, CD16, CD45RA, CCR7, HLADR
POP_MEANS = {
    "CD4_T_naive":      [4.2, 0.8, 3.2, 0.3, 0.3, 0.3, 4.0, 0.5, 0.3, 4.0, 3.8, 0.6],
    "CD4_T_memory":     [4.1, 1.0, 3.0, 0.3, 0.3, 0.3, 3.9, 0.5, 0.3, 1.0, 1.2, 0.8],
    "CD8_T":            [4.0, 0.7, 2.8, 0.3, 0.4, 0.3, 0.4, 4.1, 0.4, 2.5, 2.0, 0.6],
    "Treg":             [4.0, 3.6, 0.6, 0.3, 0.3, 0.3, 3.9, 0.4, 0.3, 1.2, 2.6, 0.7],
    "B_cell":           [0.4, 0.4, 0.4, 4.3, 0.3, 0.3, 0.3, 0.3, 0.3, 2.0, 0.6, 3.4],
    "NK_cell":          [0.4, 0.5, 0.4, 0.3, 4.2, 0.4, 0.3, 0.6, 3.8, 1.0, 0.4, 0.5],
    "Mono_classical":   [0.4, 0.4, 0.4, 0.3, 0.4, 4.1, 0.3, 0.3, 0.4, 0.5, 0.4, 2.6],
    "Mono_nonclassical":[0.4, 0.4, 0.4, 0.3, 0.4, 3.0, 0.3, 0.3, 3.5, 0.5, 0.4, 3.0],
}

POP_SD = 0.45  # within-population biological + shot-noise spread

N_SAMPLES = 32
CELLS_PER_SAMPLE = 1400


def build_spillover_matrix():
    """12x12 spillover matrix S (row=true fluorochrome, col=detected channel).

    Diagonal ~1 (own channel). Off-diagonal entries encode real
    spectrally-adjacent fluorochrome pairs that a 12-color panel designer
    has to compensate for:
      - FITC -> PerCP-Cy5.5 channel (green spillover into tandem)
      - PerCP-Cy5.5 -> PE-Cy7 (tandem degradation bleed)
      - PE -> PerCP-Cy5.5 (classic PE/PerCP overlap)
      - PE-Cy7 -> APC-Cy7 (both far-red tandems overlap on APC-Cy7 laser line)
      - APC -> APC-Cy7 (tandem bleed-through, a very common real compensation problem)
      - BV421 -> BV510, BV510 -> BV605, BV605 -> BV711 (Brilliant Violet cascade,
        a well-known real panel-design headache since all four share the violet laser)
      - BUV395 -> BUV737 (shared UV laser line)
    """
    S = np.eye(N_CHANNELS)
    idx = {c: i for i, c in enumerate(CHANNELS)}
    overlaps = [
        ("FITC_CD3", "PerCPCy55_CD127", 0.18),
        ("PerCPCy55_CD127", "FITC_CD3", 0.05),
        ("PE_CD25", "PerCPCy55_CD127", 0.22),
        ("PerCPCy55_CD127", "PECy7_CD19", 0.15),
        ("PECy7_CD19", "APCCy7_CD14", 0.20),
        ("APC_CD56", "APCCy7_CD14", 0.28),
        ("APCCy7_CD14", "APC_CD56", 0.06),
        ("BV421_CD4", "BV510_CD8", 0.25),
        ("BV510_CD8", "BV605_CD16", 0.20),
        ("BV605_CD16", "BV711_CD45RA", 0.15),
        ("BV421_CD4", "BV605_CD16", 0.08),
        ("BUV395_CCR7", "BUV737_HLADR", 0.12),
    ]
    for src, dst, frac in overlaps:
        S[idx[src], idx[dst]] = frac
    return S, idx


def simulate_all_samples(seed=RNG_SEED):
    """Return a long-form DataFrame: sample_id, cell_id, true population,
    true (unmixed) marker signals, and raw (spectrally-mixed + noisy)
    channel signals for each of the 12 detector channels.
    """
    rng = np.random.default_rng(seed)
    S, _ = build_spillover_matrix()

    rows = []
    for s_i in range(N_SAMPLES):
        sample_id = f"sample_{s_i:02d}"
        # per-sample donor variability in population composition
        fracs = np.array([POP_BASE_FRACTION[p] for p in POPULATIONS])
        fracs = fracs * rng.lognormal(mean=0, sigma=0.25, size=len(POPULATIONS))
        fracs = fracs / fracs.sum()
        n_cells = CELLS_PER_SAMPLE
        pop_counts = rng.multinomial(n_cells, fracs)

        # small per-sample instrument drift (laser power / detector gain)
        gain_drift = rng.normal(1.0, 0.03, size=N_CHANNELS)
        # per-sample electronic noise floor
        noise_sd = rng.uniform(0.12, 0.20)

        true_block = []
        labels_block = []
        for pop, n in zip(POPULATIONS, pop_counts):
            if n == 0:
                continue
            mean = np.array(POP_MEANS[pop])
            vals = rng.normal(mean, POP_SD, size=(n, N_CHANNELS))
            vals = np.clip(vals, 0, None)
            true_block.append(vals)
            labels_block.extend([pop] * n)
        true_signal = np.vstack(true_block)
        labels = np.array(labels_block)

        # spectral mixing: raw = true @ S, then gain drift + detector noise
        raw = true_signal @ S
        raw = raw * gain_drift
        raw = raw + rng.normal(0, noise_sd, size=raw.shape)
        raw = np.clip(raw, 0, None)

        for i in range(len(labels)):
            row = {"sample_id": sample_id, "cell_id": i, "population": labels[i]}
            for c_i, ch in enumerate(CHANNELS):
                row[f"true_{ch}"] = true_signal[i, c_i]
                row[f"raw_{ch}"] = raw[i, c_i]
            rows.append(row)

    df = pd.DataFrame(rows)
    return df


def main():
    df = simulate_all_samples()
    out_path = "data/raw/flow_panel_events.csv"
    df.to_csv(out_path, index=False)
    print(f"Simulated {df['sample_id'].nunique()} samples, {len(df)} total events "
          f"across {N_CHANNELS} channels -> {out_path}")
    return df


if __name__ == "__main__":
    main()
