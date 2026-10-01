import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import simulate_panel_data
import panel_optimization
import sorting_simulation
import integration_demo


@pytest.fixture(scope="module")
def raw_df():
    return simulate_panel_data.simulate_all_samples()


@pytest.fixture(scope="module")
def optimization_result(raw_df):
    return panel_optimization.optimize_panel(raw_df)


def test_sample_count(raw_df):
    assert raw_df["sample_id"].nunique() >= 30


def test_twelve_channels_present(raw_df):
    for ch in simulate_panel_data.CHANNELS:
        assert f"raw_{ch}" in raw_df.columns
    assert len(simulate_panel_data.CHANNELS) == 12


def test_eight_populations_present(raw_df):
    assert set(raw_df["population"].unique()) == set(simulate_panel_data.POPULATIONS)
    assert len(simulate_panel_data.POPULATIONS) == 8


def test_spillover_matrix_is_nontrivial(raw_df):
    S, _ = simulate_panel_data.build_spillover_matrix()
    off_diag = S - np.diag(np.diag(S))
    assert off_diag.sum() > 0.5  # genuine, non-negligible spillover


def test_optimized_panel_beats_uncompensated_baseline(optimization_result):
    """Sanity check: the selected panel/compensation config must separate
    populations better than the naive uncompensated baseline candidate,
    confirming the silhouette-based optimization is doing real work rather
    than an arbitrary pick.
    """
    results_df, best_name, _ = optimization_result
    scores = results_df.set_index("candidate")["mean_silhouette"]
    assert scores["full_matrix"] > scores["uncompensated"]
    assert best_name == scores.idxmax()


def test_compensated_candidates_beat_uncompensated(optimization_result):
    results_df, _, _ = optimization_result
    scores = results_df.set_index("candidate")["mean_silhouette"]
    assert scores["partial_manual"] > scores["uncompensated"]
    assert scores["full_matrix"] > scores["uncompensated"]


def test_sorting_purity_in_realistic_band(optimization_result):
    _, _, comp_df = optimization_result
    sort_df = sorting_simulation.simulate_sorting(comp_df)
    mean_purity = sort_df["purity"].mean()
    mean_yield = sort_df["yield"].mean()
    # A working but imperfect real sort: neither trivially perfect (>0.999,
    # which would indicate an unrealistic toy separation) nor failing badly.
    assert 0.5 <= mean_purity <= 0.999
    assert 0.3 <= mean_yield <= 1.0


def test_sorting_results_have_all_samples_and_populations(optimization_result):
    _, _, comp_df = optimization_result
    sort_df = sorting_simulation.simulate_sorting(comp_df)
    assert sort_df["sample_id"].nunique() >= 30
    assert set(sort_df["population"].unique()) == set(sorting_simulation.GATES.keys())


def test_integration_correlations_are_meaningful(optimization_result):
    _, _, comp_df = optimization_result
    sort_df = sorting_simulation.simulate_sorting(comp_df)
    prop_df, _ = integration_demo.flow_proportions(sort_df)
    integ_df = integration_demo.simulate_molecular_and_morphological(prop_df)

    from scipy.stats import pearsonr
    r_gene, p_gene = pearsonr(integ_df["cd4_t_proportion_flow"], integ_df["gene_signature_score"])
    r_size, p_size = pearsonr(integ_df["monocyte_proportion_flow"], integ_df["mean_cell_size_um_proxy"])

    # By construction these are strongly, genuinely correlated (signal
    # deliberately embedded, then recovered from data, not hardcoded here)
    assert r_gene > 0.5 and p_gene < 0.01
    assert r_size > 0.5 and p_size < 0.01
