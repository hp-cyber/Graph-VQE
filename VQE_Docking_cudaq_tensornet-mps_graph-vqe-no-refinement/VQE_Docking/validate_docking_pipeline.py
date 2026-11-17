#!/usr/bin/env python3
"""
Validate our docking pipeline (obabel → Vina) against QDockBank reference values.

For each QDockBank protein:
  1. Convert the existing full-atom PDB → PDBQT via obabel
  2. Run 20 Vina docking trials (same seed count as QDockBank)
  3. Compare computed average affinity vs reference JSON value

Usage:
    python validate_docking_pipeline.py              # 20 runs per protein
    python validate_docking_pipeline.py --dock-runs 5  # fewer runs for quick test
"""

import json
import os
import shutil
import subprocess
import random
import sys
import tempfile
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
QDOCKBANK_DIR = BASE_DIR / "QDockBank_" / "QDockBank"
SELECTED_DIR = QDOCKBANK_DIR / "selected"
VALIDATION_DIR = BASE_DIR / "docking_validation"


def pdb_to_pdbqt(pdb_path, pdbqt_path):
    """Convert PDB to PDBQT via obabel, translating COM to origin."""
    from Bio.PDB import PDBParser, PDBIO

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("s", str(pdb_path))
    coords = np.array([a.get_coord() for a in structure.get_atoms()])
    com = coords.mean(axis=0)
    for atom in structure.get_atoms():
        atom.coord -= com

    trans_pdb = str(pdb_path).replace(".pdb", "_trans.pdb")
    io = PDBIO()
    io.set_structure(structure)
    io.save(trans_pdb)

    subprocess.run(
        ["obabel", trans_pdb, "-O", str(pdbqt_path),
         "--partialcharge", "gasteiger", "-h", "-xr"],
        check=True, capture_output=True,
    )
    os.remove(trans_pdb)


def translate_mol2(input_mol2, output_mol2):
    """Translate ligand MOL2 geometric center to origin."""
    with open(input_mol2) as f:
        lines = f.readlines()

    atom_section = False
    atom_lines = []
    atom_indices = []
    for i, line in enumerate(lines):
        if line.startswith("@<TRIPOS>ATOM"):
            atom_section = True
            continue
        elif line.startswith("@<TRIPOS>") and atom_section:
            atom_section = False
            continue
        if atom_section:
            parts = line.split()
            if len(parts) >= 5:
                atom_lines.append(parts)
                atom_indices.append(i)

    if not atom_lines:
        shutil.copy2(input_mol2, output_mol2)
        return

    coords = np.array([[float(p[2]), float(p[3]), float(p[4])] for p in atom_lines])
    center = coords.mean(axis=0)
    coords -= center

    out_lines = list(lines)
    for idx, al, c in zip(atom_indices, atom_lines, coords):
        al[2], al[3], al[4] = f"{c[0]:.4f}", f"{c[1]:.4f}", f"{c[2]:.4f}"
        out_lines[idx] = f"{al[0]:>6} {al[1]:<10} {al[2]:>8} {al[3]:>8} {al[4]:>8} {' '.join(al[5:])}\n"

    with open(output_mol2, "w") as f:
        f.writelines(out_lines)


def run_vina_single(receptor_pdbqt, ligand_mol2, output_dir, seed):
    """Run a single Vina docking trial via Python API. Returns average affinity or None."""
    os.makedirs(output_dir, exist_ok=True)
    ligand_pdbqt = os.path.join(output_dir, "ligand.pdbqt")

    subprocess.run(
        ["obabel", ligand_mol2, "-O", ligand_pdbqt,
         "--partialcharge", "gasteiger", "-h"],
        check=True, capture_output=True,
    )

    # Ligand center from MOL2
    atom_coords = []
    with open(ligand_mol2) as f:
        in_atoms = False
        for line in f:
            if line.startswith("@<TRIPOS>ATOM"):
                in_atoms = True
                continue
            elif line.startswith("@<TRIPOS>") and in_atoms:
                break
            if in_atoms:
                parts = line.split()
                if len(parts) >= 5:
                    atom_coords.append([float(parts[2]), float(parts[3]), float(parts[4])])
    if not atom_coords:
        return None
    center = np.mean(atom_coords, axis=0)

    output_file = os.path.join(output_dir, "docking_output.pdbqt")

    try:
        from vina import Vina
        v = Vina(sf_name='vina', seed=seed, verbosity=0)
        v.set_receptor(str(receptor_pdbqt))
        v.set_ligand_from_file(ligand_pdbqt)
        v.compute_vina_maps(center=[float(center[0]), float(center[1]), float(center[2])],
                            box_size=[18, 18, 18])
        v.dock(exhaustiveness=16, n_poses=9)
        v.write_poses(output_file, overwrite=True)

        energies = v.energies(n_poses=9)
        if energies is not None:
            return float(np.mean([e[0] for e in energies]))
        return None
    except Exception as e:
        print(f"      Vina error: {e}")
        return None


def main():
    num_runs = 20
    for i, arg in enumerate(sys.argv):
        if arg == "--dock-runs" and i + 1 < len(sys.argv):
            num_runs = int(sys.argv[i + 1])

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all QDockBank proteins with docking results
    proteins = []
    for d in sorted(QDOCKBANK_DIR.iterdir()):
        if not d.is_dir() or d.name in ("selected", "__MACOSX"):
            continue
        pdb_id = d.name
        result_json = d / f"{pdb_id}_RMSD_docking_result.json"
        pred_pdb = d / f"{pdb_id}.pdb"
        if not result_json.is_file() or not pred_pdb.is_file():
            continue

        data = json.loads(result_json.read_text())
        docking = data.get("docking", {})
        ref_aff = docking.get("average_affinity")
        if ref_aff is None:
            continue

        # Find ligand
        ligand_mol2 = SELECTED_DIR / pdb_id / f"{pdb_id}_ligand.mol2"
        if not ligand_mol2.is_file():
            ligand_mol2 = SELECTED_DIR / pdb_id / f"{pdb_id}_ligand.sdf"
        if not ligand_mol2.is_file():
            continue

        proteins.append({
            "pdb_id": pdb_id,
            "pdb_path": pred_pdb,
            "ligand_path": ligand_mol2,
            "ref_affinity": ref_aff,
            "ref_runs": len(docking.get("runs", [])),
        })

    print(f"Validating docking pipeline on {len(proteins)} QDockBank proteins ({num_runs} runs each)")
    print("=" * 90)
    print(f"{'PDB':<8} {'Ref Aff':>10} {'Calc Aff':>10} {'Diff':>10} {'% Diff':>10} {'Status':>10}")
    print("-" * 90)

    results = []
    for p in proteins:
        pdb_id = p["pdb_id"]
        work_dir = VALIDATION_DIR / pdb_id
        work_dir.mkdir(parents=True, exist_ok=True)

        # Check cache
        cache_file = work_dir / "validation_result.json"
        if cache_file.is_file():
            cached = json.loads(cache_file.read_text())
            calc_aff = cached["calc_affinity"]
        else:
            # Step 1: PDB → PDBQT
            receptor_pdbqt = work_dir / f"{pdb_id}.pdbqt"
            try:
                pdb_to_pdbqt(str(p["pdb_path"]), str(receptor_pdbqt))
            except Exception as e:
                print(f"{pdb_id:<8} {p['ref_affinity']:>10.3f} {'ERR':>10} {'':>10} {'':>10} {'SKIP':>10}  obabel: {e}")
                continue

            # Step 2: Translate ligand
            ligand_trans = work_dir / f"{pdb_id}_ligand_trans.mol2"
            if str(p["ligand_path"]).endswith(".mol2"):
                translate_mol2(str(p["ligand_path"]), str(ligand_trans))
            else:
                try:
                    subprocess.run(
                        ["obabel", str(p["ligand_path"]), "-O", str(ligand_trans)],
                        check=True, capture_output=True,
                    )
                    translate_mol2(str(ligand_trans), str(ligand_trans))
                except Exception as e:
                    print(f"{pdb_id:<8} {p['ref_affinity']:>10.3f} {'ERR':>10} {'':>10} {'':>10} {'SKIP':>10}  ligand: {e}")
                    continue

            # Step 3: Run Vina trials
            run_affinities = []
            for trial in range(1, num_runs + 1):
                seed = random.randint(1, 1000000)
                trial_dir = str(work_dir / f"trial_{trial}")
                aff = run_vina_single(str(receptor_pdbqt), str(ligand_trans), trial_dir, seed)
                if aff is not None:
                    run_affinities.append(aff)

            if not run_affinities:
                print(f"{pdb_id:<8} {p['ref_affinity']:>10.3f} {'FAIL':>10} {'':>10} {'':>10} {'FAIL':>10}")
                continue

            calc_aff = float(np.mean(run_affinities))

            # Cache
            cache_file.write_text(json.dumps({
                "calc_affinity": round(calc_aff, 3),
                "ref_affinity": p["ref_affinity"],
                "num_runs": len(run_affinities),
                "run_affinities": [round(a, 3) for a in run_affinities],
            }, indent=2))

        diff = calc_aff - p["ref_affinity"]
        pct_diff = (diff / abs(p["ref_affinity"])) * 100 if p["ref_affinity"] != 0 else 0
        status = "OK" if abs(pct_diff) < 20 else "WARN"
        results.append({
            "pdb_id": pdb_id,
            "ref": p["ref_affinity"],
            "calc": calc_aff,
            "diff": diff,
            "pct": pct_diff,
            "status": status,
        })
        print(f"{pdb_id:<8} {p['ref_affinity']:>10.3f} {calc_aff:>10.3f} {diff:>+10.3f} {pct_diff:>+9.1f}% {status:>10}")

    # Summary
    if results:
        print("-" * 90)
        diffs = [r["diff"] for r in results]
        pcts = [abs(r["pct"]) for r in results]
        ok_count = sum(1 for r in results if r["status"] == "OK")
        print(f"\nSummary: {ok_count}/{len(results)} within 20% tolerance")
        print(f"  Mean absolute diff:  {np.mean(np.abs(diffs)):.3f} kcal/mol")
        print(f"  Max absolute diff:   {np.max(np.abs(diffs)):.3f} kcal/mol")
        print(f"  Mean % difference:   {np.mean(pcts):.1f}%")
        print(f"  Correlation (ref vs calc): {np.corrcoef([r['ref'] for r in results], [r['calc'] for r in results])[0,1]:.4f}")


if __name__ == "__main__":
    main()
