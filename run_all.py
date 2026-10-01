"""Runs the full 12-parameter flow cytometry panel-design / sorting pipeline."""
import os

import simulate_panel_data
import panel_optimization
import sorting_simulation
import integration_demo


def main():
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("output/figures", exist_ok=True)

    print("=" * 70)
    print("STEP 1: Simulating 12-parameter panel across primary samples")
    print("=" * 70)
    simulate_panel_data.main()

    print("\n" + "=" * 70)
    print("STEP 2: Panel/compensation optimization (silhouette comparison)")
    print("=" * 70)
    panel_optimization.main()

    print("\n" + "=" * 70)
    print("STEP 3: Cell-sorting yield/purity simulation")
    print("=" * 70)
    sorting_simulation.main()

    print("\n" + "=" * 70)
    print("STEP 4: Cross-modality integration (molecular + morphological)")
    print("=" * 70)
    integration_demo.main()

    print("\nDone. See output/ for all results.")


if __name__ == "__main__":
    main()
