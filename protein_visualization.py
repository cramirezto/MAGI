"""
protein_visualization.py
------------------------
Predict and visualize a protein's 3D structure from an amino acid sequence.

Usage
-----
    python protein_visualization.py                        # interactive prompt
    python protein_visualization.py -s ACDEFGHIKLMNPQRSTVWY
    python protein_visualization.py -s ACDE... -o my_protein.pdb
    python protein_visualization.py --pdb existing.pdb     # skip prediction

Dependencies: requests, py3Dmol, biopandas, biopython, matplotlib, numpy
"""

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import requests
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from biopandas.pdb import PandasPdb

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

# Kyte-Doolittle hydrophobicity scale
KD_SCALE = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

# ---------------------------------------------------------------------------
# Sequence validation
# ---------------------------------------------------------------------------

def validate_sequence(sequence: str) -> str:
    sequence = sequence.strip().upper().replace(" ", "").replace("\n", "")
    invalid = set(sequence) - VALID_AA
    if invalid:
        raise ValueError(f"Invalid amino acid characters: {invalid}")
    if len(sequence) < 6:
        raise ValueError("Sequence too short (minimum 6 residues).")
    if len(sequence) > 400:
        print(f"Warning: sequence is {len(sequence)} residues. "
              "ESMFold works best under 400 residues.")
    return sequence

# ---------------------------------------------------------------------------
# Structure prediction via ESMFold
# ---------------------------------------------------------------------------

def predict_structure(sequence: str, output_pdb: str = "predicted_structure.pdb") -> str:
    """Query the ESMFold API and save the predicted PDB."""
    print(f"Predicting structure for {len(sequence)}-residue sequence via ESMFold...")
    url = "https://api.esmatlas.com/foldSequence/v1/pdb/"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    try:
        response = requests.post(url, data=sequence, headers=headers, timeout=120)
        response.raise_for_status()
    except requests.exceptions.Timeout:
        sys.exit("Error: ESMFold API timed out. Try a shorter sequence or retry later.")
    except requests.exceptions.HTTPError as e:
        sys.exit(f"Error: ESMFold API returned {response.status_code}: {e}")

    pdb_string = response.text
    Path(output_pdb).write_text(pdb_string)
    print(f"Structure saved to '{output_pdb}'")
    return pdb_string

# ---------------------------------------------------------------------------
# 3D visualisation
# ---------------------------------------------------------------------------

def visualize_3d(pdb_string: str):
    """Open an interactive py3Dmol viewer in the browser."""
    try:
        import py3Dmol  # noqa: PLC0415
    except ImportError:
        print("py3Dmol not installed – skipping 3D viewer.")
        return

    view = py3Dmol.view(width=800, height=500)
    view.addModel(pdb_string, "pdb")

    # Cartoon coloured by spectrum (N-term blue → C-term red)
    view.setStyle({"cartoon": {"color": "spectrum"}})
    view.zoomTo()
    view.spin(True)

    # Save to HTML so it can be opened in any browser
    html_file = "protein_structure.html"
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Protein Structure Viewer</title>
    <script src="https://3dmol.org/build/3Dmol-min.js"></script>
</head>
<body>
<div id="viewer" style="width:800px;height:500px;position:relative;"></div>
<script>
  let viewer = $3Dmol.createViewer("viewer", {{backgroundColor:"white"}});
  viewer.addModel(`{pdb_string}`, "pdb");
  viewer.setStyle({{}}, {{cartoon: {{color: "spectrum"}}}});
  viewer.zoomTo();
  viewer.spin(true);
  viewer.render();
</script>
</body>
</html>"""
    Path(html_file).write_text(html_content)
    print(f"3D viewer saved to '{html_file}' – open it in your browser.")

# ---------------------------------------------------------------------------
# Sequence analysis plots
# ---------------------------------------------------------------------------

def plot_sequence_analysis(sequence: str, output_prefix: str = "analysis"):
    analysis = ProteinAnalysis(sequence)
    aa_percent  = analysis.get_amino_acids_percent()
    mw          = analysis.molecular_weight()
    ip          = analysis.isoelectric_point()
    gravy       = analysis.gravy()
    instability = analysis.instability_index()
    sec_struct  = analysis.secondary_structure_fraction()  # (helix, turn, sheet)

    print("\n" + "=" * 52)
    print("  PROTEIN PHYSICOCHEMICAL PROPERTIES")
    print("=" * 52)
    print(f"  Length            : {len(sequence)} residues")
    print(f"  Molecular weight  : {mw:,.1f} Da")
    print(f"  Isoelectric point : {ip:.2f}")
    print(f"  GRAVY index       : {gravy:.3f}  "
          f"({'hydrophobic' if gravy > 0 else 'hydrophilic'})")
    print(f"  Instability index : {instability:.1f}  "
          f"({'unstable' if instability > 40 else 'stable'})")
    print(f"  2° structure est. : {sec_struct[0]*100:.1f}% helix | "
          f"{sec_struct[2]*100:.1f}% sheet | {sec_struct[1]*100:.1f}% turn")
    print("=" * 52)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Sequence Analysis", fontsize=14, fontweight="bold")

    # Amino acid composition
    aas    = sorted(aa_percent, key=aa_percent.get, reverse=True)
    vals   = [aa_percent[a] * 100 for a in aas]
    colors = plt.cm.tab20(np.linspace(0, 1, len(aas)))
    axes[0].bar(aas, vals, color=colors)
    axes[0].set_title("Amino Acid Composition (%)")
    axes[0].set_xlabel("Amino Acid")
    axes[0].set_ylabel("Percentage")
    axes[0].tick_params(axis="x", rotation=45)

    # Secondary structure estimate
    labels     = ["α-Helix", "β-Sheet", "Turn"]
    sizes      = [sec_struct[0], sec_struct[2], sec_struct[1]]
    pie_colors = ["#4C72B0", "#DD8452", "#55A868"]
    axes[1].pie(sizes, labels=labels, colors=pie_colors, autopct="%1.1f%%", startangle=90)
    axes[1].set_title("Secondary Structure Estimate")

    # Kyte-Doolittle hydrophobicity sliding window
    window = min(9, len(sequence))
    scores = [KD_SCALE.get(aa, 0) for aa in sequence]
    hydro  = [np.mean(scores[i:i+window]) for i in range(len(scores) - window + 1)]
    x      = range(window // 2, len(hydro) + window // 2)
    axes[2].plot(x, hydro, color="steelblue", linewidth=1.5)
    axes[2].axhline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[2].fill_between(x, hydro, 0,
                         where=[h > 0 for h in hydro],
                         alpha=0.3, color="orange",    label="hydrophobic")
    axes[2].fill_between(x, hydro, 0,
                         where=[h < 0 for h in hydro],
                         alpha=0.3, color="steelblue", label="hydrophilic")
    axes[2].set_title(f"Kyte-Doolittle Hydrophobicity (window={window})")
    axes[2].set_xlabel("Residue position")
    axes[2].set_ylabel("Hydrophobicity score")
    axes[2].legend()

    plt.tight_layout()
    out = f"{output_prefix}_sequence.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Sequence analysis plot saved to '{out}'")
    plt.show()

# ---------------------------------------------------------------------------
# pLDDT confidence plot
# ---------------------------------------------------------------------------

def plot_plddt(pdb_file: str, output_prefix: str = "analysis"):
    ppdb = PandasPdb().read_pdb(pdb_file)
    ca   = (ppdb.df["ATOM"]
            .query("atom_name == 'CA'")[["residue_number", "b_factor"]]
            .reset_index(drop=True))

    colors = [
        "#0053D6" if v >= 90 else
        "#65CBF3" if v >= 70 else
        "#FFDB13" if v >= 50 else
        "#FF7D45"
        for v in ca["b_factor"]
    ]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(ca["residue_number"], ca["b_factor"], color=colors, width=1.0)
    for threshold, color in [(90, "#0053D6"), (70, "#65CBF3"), (50, "#FFDB13")]:
        ax.axhline(threshold, color=color, linestyle="--", linewidth=0.8)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Residue number")
    ax.set_ylabel("pLDDT score")
    ax.set_title("Per-residue pLDDT Confidence (ESMFold)")
    patches = [
        mpatches.Patch(color="#0053D6", label="Very high (≥90)"),
        mpatches.Patch(color="#65CBF3", label="Confident (70–90)"),
        mpatches.Patch(color="#FFDB13", label="Low (50–70)"),
        mpatches.Patch(color="#FF7D45", label="Very low (<50)"),
    ]
    ax.legend(handles=patches, loc="lower right")
    plt.tight_layout()
    out = f"{output_prefix}_plddt.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"pLDDT plot saved to '{out}'")
    plt.show()
    print(f"Mean pLDDT: {ca['b_factor'].mean():.1f}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict and visualize a protein structure from an amino acid sequence."
    )
    parser.add_argument("-s", "--sequence",
                        help="Amino acid sequence (single-letter codes)")
    parser.add_argument("-o", "--output", default="predicted_structure.pdb",
                        help="Output PDB filename (default: predicted_structure.pdb)")
    parser.add_argument("--pdb",
                        help="Skip prediction and load an existing PDB file")
    parser.add_argument("--prefix", default="analysis",
                        help="Prefix for output plot filenames (default: analysis)")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.pdb:
        pdb_file   = args.pdb
        pdb_string = Path(pdb_file).read_text()
        sequence   = None
        print(f"Loaded existing PDB: '{pdb_file}'")
    else:
        if args.sequence:
            sequence = args.sequence
        else:
            print("Enter amino acid sequence (single-letter codes, press Enter when done):")
            sequence = input("> ").strip()

        sequence   = validate_sequence(sequence)
        pdb_string = predict_structure(sequence, output_pdb=args.output)
        pdb_file   = args.output

    if sequence:
        plot_sequence_analysis(sequence, output_prefix=args.prefix)

    plot_plddt(pdb_file, output_prefix=args.prefix)
    visualize_3d(pdb_string)


if __name__ == "__main__":
    main()
