#!/usr/bin/env python3
"""
Evaluate a single strategy directory: compute RMSD and average affinity for each protein.

Usage:
    python evaluate_strategy.py <strategy_dir>

Example:
    python evaluate_strategy.py Result/process_data/recursive_forging_depth2_sca
"""

import json
import sys
from pathlib import Path

import numpy as np
from Bio.SVDSuperimposer import SVDSuperimposer

BASE_DIR = Path(__file__).resolve().parent
QDOCKBANK_DIR = BASE_DIR / "QDockBank_" / "QDockBank"
SELECTED_DIR = QDOCKBANK_DIR / "selected"
TARGET_CA_DISTANCE = 3.8

AA_MAP_3TO1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
}


def load_protein_info():
    """Load residue ranges from QDockBank metadata."""
    info = {}
    if QDOCKBANK_DIR.is_dir():
        for d in QDOCKBANK_DIR.iterdir():
            if not d.is_dir() or d.name in ("selected", "__MACOSX"):
                continue
            meta_file = d / f"{d.name}_metadata.json"
            if meta_file.is_file():
                meta = json.loads(meta_file.read_text())
                pi = meta["protein_information"]
                start, end = pi["residues"].split("-")
                info[d.name] = (pi["sequence"], int(start), int(end), pi["chain"])
    if "1fkf" not in info:
        info["1fkf"] = ("KPFKF", 44, 48, "A")
    return info


def parse_xyz(xyz_path):
    atoms = []
    with open(xyz_path) as f:
        lines = f.readlines()
    for line in lines[2:]:
        parts = line.strip().split()
        if len(parts) == 4:
            atoms.append((parts[0], float(parts[1]), float(parts[2]), float(parts[3])))
    return atoms


def scale_coords(coords):
    if len(coords) < 2:
        return coords
    dists = [np.linalg.norm(coords[i+1] - coords[i]) for i in range(len(coords)-1)]
    dists = [d for d in dists if d > 0]
    if not dists:
        return coords
    return coords * (TARGET_CA_DISTANCE / np.mean(dists))


def extract_ca(pdb_path, chain, start, end):
    ca = []
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM") or line[12:16].strip() != "CA":
                continue
            if line[21] != chain:
                continue
            resnum = int(line[22:26].strip())
            if start <= resnum <= end:
                ca.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return np.array(ca)


def compute_rmsd(c1, c2):
    if len(c1) != len(c2) or len(c1) == 0:
        return None
    sup = SVDSuperimposer()
    sup.set(c1, c2)
    sup.run()
    return sup.get_rms()


DEFAULT_STRATEGY = "Result/process_data/recursive_forging_depth2_sca"


def main():
    strategy_arg = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_STRATEGY
    strategy_dir = Path(strategy_arg)
    if not strategy_dir.is_absolute():
        strategy_dir = BASE_DIR / strategy_dir
    if not strategy_dir.is_dir():
        print(f"ERROR: {strategy_dir} not found")
        sys.exit(1)

    protein_info = load_protein_info()
    strategy_name = strategy_dir.name

    print(f"Strategy: {strategy_name}")
    print(f"{'Protein':<10} {'Sequence':<20} {'RMSD (Å)':<12} {'Avg Affinity':<15} {'QDB RMSD':<12} {'QDB Affinity':<15}")
    print("-" * 84)

    for protein_dir in sorted(strategy_dir.iterdir()):
        if not protein_dir.is_dir():
            continue
        pdb_id = protein_dir.name

        # --- RMSD ---
        rmsd_val = None
        xyz_file = protein_dir / "Structures" / f"{pdb_id}_{strategy_name}.xyz"
        if xyz_file.is_file():
            atoms = parse_xyz(xyz_file)
            sequence = "".join(a[0] for a in atoms)
            pred = scale_coords(np.array([[a[1], a[2], a[3]] for a in atoms]))

            info = protein_info.get(pdb_id)
            if info:
                seq, start, end, chain = info
                gt_pdb = SELECTED_DIR / pdb_id / f"{pdb_id}_protein.pdb"
                if gt_pdb.is_file():
                    gt = extract_ca(gt_pdb, chain, start, end)
                    if len(gt) == len(pred):
                        rmsd_val = compute_rmsd(gt, pred)
        else:
            sequence = "?"

        # --- Affinity ---
        affinity_val = None
        dock_file = protein_dir / "Docking" / "docking_result.json"
        if dock_file.is_file():
            dock = json.loads(dock_file.read_text())
            affinity_val = dock.get("average_affinity")

        # --- QDockBank baseline ---
        qdb_rmsd = None
        qdb_affinity = None
        qdb_file = QDOCKBANK_DIR / pdb_id / f"{pdb_id}_RMSD_docking_result.json"
        if qdb_file.is_file():
            qdb = json.loads(qdb_file.read_text())
            qdb_rmsd = qdb.get("rmsd")
            dock_data = qdb.get("docking", {})
            if "average_affinity" in dock_data:
                qdb_affinity = dock_data["average_affinity"]
            elif "runs" in dock_data:
                affs = [r["average_affinity"] for r in dock_data["runs"] if "average_affinity" in r]
                if affs:
                    qdb_affinity = round(np.mean(affs), 3)

        rmsd_str = f"{rmsd_val:.3f}" if rmsd_val is not None else "N/A"
        aff_str = f"{affinity_val}" if affinity_val is not None else "N/A"
        qdb_r_str = f"{qdb_rmsd:.3f}" if qdb_rmsd is not None else "N/A"
        qdb_a_str = f"{qdb_affinity}" if qdb_affinity is not None else "N/A"

        print(f"{pdb_id:<10} {sequence:<20} {rmsd_str:<12} {aff_str:<15} {qdb_r_str:<12} {qdb_a_str:<15}")


if __name__ == "__main__":
    main()
