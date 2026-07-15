#!/usr/bin/env python3
"""
Calculate Cα RMSD between VQE-predicted structures and ground truth from PDBbind.

Usage:
    # Calculate RMSD for new results in Result/process_data and validate against QDockBank
    python calculate_rmsd.py

The script:
1. Reads VQE-predicted .xyz files (lattice coordinates)
2. Scales them to real Cα-Cα distance (~3.8 Å)
3. Extracts ground truth Cα from PDBbind protein PDB (in QDockBank/selected/)
   using the aligned residue range from QDockBank metadata
4. Computes RMSD after optimal superposition (Kabsch algorithm via BioPython)
5. Validates by re-computing RMSD for QDockBank predicted PDB files and comparing
   with the reference RMSD values in *_RMSD_docking_result.json
"""

import json
import os
import sys
import csv
from pathlib import Path

import numpy as np
from Bio.SVDSuperimposer import SVDSuperimposer

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
PROCESS_DATA_DIR = BASE_DIR / "Result" / "process_data"
QDOCKBANK_DIR = BASE_DIR / "QDockBank_" / "QDockBank"
SELECTED_DIR = QDOCKBANK_DIR / "selected"

# Average Cα-Cα distance in real proteins (Å)
TARGET_CA_DISTANCE = 3.8

# 1-letter to 3-letter amino acid mapping
AA_MAP_3TO1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
}

# Known residue ranges (from QDockBank metadata + manually identified ones)
# Format: pdb_id -> (sequence, start_residue, end_residue, chain)
PROTEIN_INFO = {}


def load_protein_info():
    """Load residue ranges from QDockBank metadata files, and add manual entries."""
    # Load from QDockBank metadata
    if QDOCKBANK_DIR.is_dir():
        for d in QDOCKBANK_DIR.iterdir():
            if not d.is_dir() or d.name == "selected" or d.name == "__MACOSX":
                continue
            meta_file = d / f"{d.name}_metadata.json"
            if meta_file.is_file():
                meta = json.loads(meta_file.read_text())
                pi = meta["protein_information"]
                start, end = pi["residues"].split("-")
                PROTEIN_INFO[d.name] = (
                    pi["sequence"],
                    int(start),
                    int(end),
                    pi["chain"],
                )

    # Manual entries for proteins not in QDockBank but in process_data
    # (identified from Main.py protein_list and ground truth PDB)
    if "1fkf" not in PROTEIN_INFO:
        PROTEIN_INFO["1fkf"] = ("KPFKF", 44, 48, "A")


def parse_xyz(xyz_path):
    """Parse a VQE .xyz file. Returns list of (residue_1letter, x, y, z)."""
    atoms = []
    with open(xyz_path) as f:
        lines = f.readlines()
    # First line: atom count, second line: blank/comment, then data
    for line in lines[2:]:
        parts = line.strip().split()
        if len(parts) == 4:
            atoms.append((parts[0], float(parts[1]), float(parts[2]), float(parts[3])))
    return atoms


def scale_xyz_coords(coords, target_distance=TARGET_CA_DISTANCE):
    """Scale lattice coordinates so average consecutive Cα distance matches target."""
    if len(coords) < 2:
        return coords
    distances = []
    for i in range(len(coords) - 1):
        d = np.linalg.norm(coords[i + 1] - coords[i])
        if d > 0:
            distances.append(d)
    if not distances:
        return coords
    avg_dist = np.mean(distances)
    scale = target_distance / avg_dist
    return coords * scale


def extract_ca_from_pdb(pdb_path, chain, start_res, end_res):
    """Extract Cα coordinates for a residue range from a PDB file."""
    ca_coords = []
    ca_residues = []
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            pdb_chain = line[21]
            if pdb_chain != chain:
                continue
            resnum = int(line[22:26].strip())
            if start_res <= resnum <= end_res:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                resname = line[17:20].strip()
                ca_coords.append([x, y, z])
                ca_residues.append(AA_MAP_3TO1.get(resname, "?"))
    return np.array(ca_coords), "".join(ca_residues)


def extract_ca_from_predicted_pdb(pdb_path):
    """Extract all Cα coordinates from a QDockBank predicted PDB (chain A, sequential)."""
    ca_coords = []
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            ca_coords.append([x, y, z])
    return np.array(ca_coords)


def find_residue_range_by_sequence(pdb_path, chain, target_seq):
    """Find the residue range for a target sequence in a PDB file by sequence matching."""
    residues = []
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            pdb_chain = line[21]
            if pdb_chain != chain:
                continue
            resnum = int(line[22:26].strip())
            resname = line[17:20].strip()
            residues.append((resnum, AA_MAP_3TO1.get(resname, "?")))

    full_seq = "".join(r[1] for r in residues)
    idx = full_seq.find(target_seq)
    if idx >= 0:
        start = residues[idx][0]
        end = residues[idx + len(target_seq) - 1][0]
        return start, end
    return None, None


def compute_rmsd(coords1, coords2):
    """Compute RMSD after optimal superposition using Kabsch algorithm."""
    if len(coords1) != len(coords2):
        return None
    if len(coords1) < 3:
        # For very short fragments, just center and compute
        c1 = coords1 - coords1.mean(axis=0)
        c2 = coords2 - coords2.mean(axis=0)
        sup = SVDSuperimposer()
        sup.set(c1, c2)
        sup.run()
        return sup.get_rms()

    sup = SVDSuperimposer()
    sup.set(coords1, coords2)
    sup.run()
    return sup.get_rms()


def get_ground_truth_ca(pdb_id, sequence):
    """Get ground truth Cα coordinates from PDBbind protein in selected/."""
    protein_pdb = SELECTED_DIR / pdb_id / f"{pdb_id}_protein.pdb"
    if not protein_pdb.is_file():
        return None, f"Ground truth PDB not found: {protein_pdb}"

    info = PROTEIN_INFO.get(pdb_id)
    if info is None:
        # Try to find by sequence matching
        start, end = find_residue_range_by_sequence(protein_pdb, "A", sequence)
        if start is None:
            return None, f"No metadata and sequence {sequence} not found in {pdb_id}_protein.pdb"
        chain = "A"
    else:
        sequence, start, end, chain = info

    gt_coords, gt_seq = extract_ca_from_pdb(protein_pdb, chain, start, end)
    if len(gt_coords) == 0:
        return None, f"No Cα atoms found for {pdb_id} chain {chain} residues {start}-{end}"

    if len(gt_coords) != len(sequence):
        return None, (
            f"Length mismatch for {pdb_id}: expected {len(sequence)} residues, "
            f"got {len(gt_coords)} Cα atoms (residues {start}-{end})"
        )

    return gt_coords, None


def process_xyz_structure(xyz_path, pdb_id, sequence):
    """Process a single .xyz file: scale, get ground truth, compute RMSD."""
    atoms = parse_xyz(xyz_path)
    if not atoms:
        return None, "Empty xyz file"

    pred_coords = np.array([[a[1], a[2], a[3]] for a in atoms])
    pred_coords = scale_xyz_coords(pred_coords)

    gt_coords, err = get_ground_truth_ca(pdb_id, sequence)
    if err:
        return None, err

    if len(pred_coords) != len(gt_coords):
        return None, (
            f"Atom count mismatch: predicted {len(pred_coords)}, ground truth {len(gt_coords)}"
        )

    rmsd = compute_rmsd(gt_coords, pred_coords)
    return rmsd, None


def get_sequence_for_protein(pdb_id):
    """Get the amino acid sequence for a protein from metadata or Main.py definitions."""
    info = PROTEIN_INFO.get(pdb_id)
    if info:
        return info[0]
    return None


# ─── Validation against QDockBank ────────────────────────────────────────────

def validate_against_qdockbank():
    """
    Validate our RMSD calculation by comparing with QDockBank reference values.
    Uses the predicted PDB files in QDockBank/ and compares RMSD with the
    values in *_RMSD_docking_result.json.
    """
    print("=" * 70)
    print("VALIDATION: Comparing calculated RMSD with QDockBank reference values")
    print("=" * 70)

    results = []
    for d in sorted(QDOCKBANK_DIR.iterdir()):
        if not d.is_dir() or d.name in ("selected", "__MACOSX"):
            continue
        pdb_id = d.name

        # Load reference RMSD
        rmsd_file = d / f"{pdb_id}_RMSD_docking_result.json"
        if not rmsd_file.is_file():
            continue
        ref_data = json.loads(rmsd_file.read_text())
        ref_rmsd = ref_data["rmsd"]

        # Load metadata for sequence/residue info
        info = PROTEIN_INFO.get(pdb_id)
        if not info:
            continue
        sequence, start_res, end_res, chain = info

        # Get ground truth Cα
        gt_coords, err = get_ground_truth_ca(pdb_id, sequence)
        if err:
            print(f"  {pdb_id}: SKIP - {err}")
            continue

        # Get predicted Cα from QDockBank PDB
        pred_pdb = d / f"{pdb_id}.pdb"
        if not pred_pdb.is_file():
            continue
        pred_coords = extract_ca_from_predicted_pdb(pred_pdb)

        if len(pred_coords) != len(gt_coords):
            print(
                f"  {pdb_id}: SKIP - length mismatch "
                f"(predicted {len(pred_coords)}, gt {len(gt_coords)})"
            )
            continue

        calc_rmsd = compute_rmsd(gt_coords, pred_coords)
        diff = abs(calc_rmsd - ref_rmsd)
        match = "OK" if diff < 0.05 else "MISMATCH"
        results.append((pdb_id, ref_rmsd, calc_rmsd, diff, match))
        print(
            f"  {pdb_id}: ref={ref_rmsd:.3f}, calc={calc_rmsd:.3f}, "
            f"diff={diff:.3f} [{match}]"
        )

    print(f"\nValidated {len(results)} proteins.")
    ok_count = sum(1 for r in results if r[4] == "OK")
    print(f"  Matched: {ok_count}/{len(results)}")
    return results


# ─── Main: Process Result/process_data ───────────────────────────────────────

def process_all_strategies():
    """
    Process all strategies and proteins in Result/process_data/.
    For each strategy/protein, compute RMSD for the main .xyz and all top-K .xyz files.
    """
    print("\n" + "=" * 70)
    print("RMSD CALCULATION: Result/process_data")
    print("=" * 70)

    all_results = []

    if not PROCESS_DATA_DIR.is_dir():
        print(f"ERROR: {PROCESS_DATA_DIR} does not exist")
        return all_results

    strategies = sorted(
        d.name for d in PROCESS_DATA_DIR.iterdir() if d.is_dir()
    )

    for strategy in strategies:
        strategy_dir = PROCESS_DATA_DIR / strategy
        proteins = sorted(d.name for d in strategy_dir.iterdir() if d.is_dir())

        print(f"\n--- Strategy: {strategy} ---")

        for pdb_id in proteins:
            sequence = get_sequence_for_protein(pdb_id)
            if sequence is None:
                # Try to infer from ground truth PDB by looking at xyz
                struct_dir = strategy_dir / pdb_id / "Structures"
                if not struct_dir.is_dir():
                    print(f"  {pdb_id}: SKIP - no Structures dir")
                    continue
                # Get sequence from the main xyz
                main_xyz = struct_dir / f"{pdb_id}_{strategy}.xyz"
                if main_xyz.is_file():
                    atoms = parse_xyz(main_xyz)
                    sequence = "".join(a[0] for a in atoms)
                else:
                    print(f"  {pdb_id}: SKIP - no main xyz and no metadata")
                    continue

            struct_dir = strategy_dir / pdb_id / "Structures"
            if not struct_dir.is_dir():
                print(f"  {pdb_id}: SKIP - no Structures dir")
                continue

            # Process all xyz files in Structures/
            xyz_files = sorted(struct_dir.glob("*.xyz"))
            for xyz_file in xyz_files:
                rmsd, err = process_xyz_structure(xyz_file, pdb_id, sequence)
                label = xyz_file.stem
                if err:
                    print(f"  {pdb_id}/{label}: ERROR - {err}")
                else:
                    # Determine if this is main or top-K
                    if "_top_" in label:
                        rank = label.split("_top_")[-1]
                        entry_type = f"top_{rank}"
                    else:
                        entry_type = "main"

                    all_results.append({
                        "strategy": strategy,
                        "pdb_id": pdb_id,
                        "type": entry_type,
                        "rmsd": round(rmsd, 3),
                        "xyz_file": str(xyz_file.name),
                    })
                    print(f"  {pdb_id}/{entry_type}: RMSD = {rmsd:.3f} Å")

    return all_results


def save_results(results, output_path):
    """Save results to CSV."""
    if not results:
        return
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["strategy", "pdb_id", "type", "rmsd", "xyz_file"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {output_path}")


def print_summary(results):
    """Print a summary table of best RMSD per strategy per protein."""
    if not results:
        return

    print("\n" + "=" * 70)
    print("SUMMARY: Best RMSD (Å) per strategy per protein (main structure)")
    print("=" * 70)

    # Filter to main structures only
    main_results = [r for r in results if r["type"] == "main"]

    # Collect unique strategies and proteins
    strategies = sorted(set(r["strategy"] for r in main_results))
    proteins = sorted(set(r["pdb_id"] for r in main_results))

    # Build lookup
    lookup = {}
    for r in main_results:
        lookup[(r["strategy"], r["pdb_id"])] = r["rmsd"]

    # Print header
    header = f"{'Protein':<10}" + "".join(f"{s[:20]:>22}" for s in strategies)
    print(header)
    print("-" * len(header))

    for pdb_id in proteins:
        row = f"{pdb_id:<10}"
        for strategy in strategies:
            rmsd = lookup.get((strategy, pdb_id))
            if rmsd is not None:
                row += f"{rmsd:>22.3f}"
            else:
                row += f"{'N/A':>22}"
        print(row)

    # Print averages
    print("-" * len(header))
    avg_row = f"{'Average':<10}"
    for strategy in strategies:
        vals = [lookup[(strategy, p)] for p in proteins if (strategy, p) in lookup]
        if vals:
            avg_row += f"{np.mean(vals):>22.3f}"
        else:
            avg_row += f"{'N/A':>22}"
    print(avg_row)

    # Also print best top-K per protein per strategy
    print("\n" + "=" * 70)
    print("SUMMARY: Best RMSD (Å) across all top-K structures per protein")
    print("=" * 70)

    for strategy in strategies:
        print(f"\n  Strategy: {strategy}")
        strat_results = [r for r in results if r["strategy"] == strategy]
        for pdb_id in proteins:
            prot_results = [r for r in strat_results if r["pdb_id"] == pdb_id]
            if prot_results:
                best = min(prot_results, key=lambda r: r["rmsd"])
                print(f"    {pdb_id}: best RMSD = {best['rmsd']:.3f} Å ({best['type']})")


if __name__ == "__main__":
    load_protein_info()

    # Step 1: Validate against QDockBank reference RMSD values
    validate_against_qdockbank()

    # Step 2: Calculate RMSD for Result/process_data
    results = process_all_strategies()

    # Step 3: Save and summarize
    output_csv = BASE_DIR / "rmsd_results.csv"
    save_results(results, output_csv)
    print_summary(results)
