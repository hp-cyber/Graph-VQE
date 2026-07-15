#!/usr/bin/env python3
"""
RMSD and average docking affinity comparison.

Ground truth: QDockBank_/QDockBank/selected/ (PDBbind pocket + ligand + protein PDB)
Baseline:     QDockBank_/QDockBank/<pdb_id>/ (VQE predicted PDB + RMSD/docking JSON)
New results:  Result/process_data/<strategy>/<pdb_id>/Structures/*.xyz

Workflow:
  1. Validate RMSD pipeline against QDockBank reference JSON values
  2. Load QDockBank average affinity as baseline
  3. Compute RMSD for Result/process_data strategies
  4. Run docking pipeline on process_data structures (Modeller + obabel + Vina)
  5. Print unified comparison tables + CSV

Usage:
    python compare_rmsd_affinity.py              # RMSD only (no docking)
    python compare_rmsd_affinity.py --dock        # RMSD + run docking for process_data
    python compare_rmsd_affinity.py --dock-runs 5 # RMSD + 5 docking runs (default 20)
"""

import json
import csv
import os
import sys
import random
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from Bio.SVDSuperimposer import SVDSuperimposer

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
QDOCKBANK_DIR = BASE_DIR / "QDockBank_" / "QDockBank"
SELECTED_DIR = QDOCKBANK_DIR / "selected"
PROCESS_DATA_DIR = BASE_DIR / "Result" / "process_data"

TARGET_CA_DISTANCE = 3.8

AA_MAP_3TO1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
}

PROTEIN_INFO = {}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_protein_info():
    """Load residue ranges from QDockBank metadata files."""
    if not QDOCKBANK_DIR.is_dir():
        return
    for d in QDOCKBANK_DIR.iterdir():
        if not d.is_dir() or d.name in ("selected", "__MACOSX"):
            continue
        meta_file = d / f"{d.name}_metadata.json"
        if meta_file.is_file():
            meta = json.loads(meta_file.read_text())
            pi = meta["protein_information"]
            start, end = pi["residues"].split("-")
            PROTEIN_INFO[d.name] = (pi["sequence"], int(start), int(end), pi["chain"])
    if "1fkf" not in PROTEIN_INFO:
        PROTEIN_INFO["1fkf"] = ("KPFKF", 44, 48, "A")


def parse_xyz(xyz_path):
    atoms = []
    with open(xyz_path) as f:
        lines = f.readlines()
    for line in lines[2:]:
        parts = line.strip().split()
        if len(parts) == 4:
            atoms.append((parts[0], float(parts[1]), float(parts[2]), float(parts[3])))
    return atoms


def scale_xyz_coords(coords, target_distance=TARGET_CA_DISTANCE):
    if len(coords) < 2:
        return coords
    distances = [np.linalg.norm(coords[i + 1] - coords[i])
                 for i in range(len(coords) - 1)
                 if np.linalg.norm(coords[i + 1] - coords[i]) > 0]
    if not distances:
        return coords
    return coords * (target_distance / np.mean(distances))


def extract_ca_from_pdb(pdb_path, chain, start_res, end_res):
    ca_coords = []
    ca_residues = []
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA":
                continue
            if line[21] != chain:
                continue
            resnum = int(line[22:26].strip())
            if start_res <= resnum <= end_res:
                ca_coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                ca_residues.append(AA_MAP_3TO1.get(line[17:20].strip(), "?"))
    return np.array(ca_coords), "".join(ca_residues)


def extract_ca_from_predicted_pdb(pdb_path):
    ca_coords = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                ca_coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return np.array(ca_coords)


def find_residue_range_by_sequence(pdb_path, chain, target_seq):
    residues = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA" and line[21] == chain:
                residues.append((int(line[22:26].strip()), AA_MAP_3TO1.get(line[17:20].strip(), "?")))
    full_seq = "".join(r[1] for r in residues)
    idx = full_seq.find(target_seq)
    if idx >= 0:
        return residues[idx][0], residues[idx + len(target_seq) - 1][0]
    return None, None


def compute_rmsd(coords1, coords2):
    if len(coords1) != len(coords2) or len(coords1) == 0:
        return None
    sup = SVDSuperimposer()
    if len(coords1) < 3:
        c1 = coords1 - coords1.mean(axis=0)
        c2 = coords2 - coords2.mean(axis=0)
        sup.set(c1, c2)
    else:
        sup.set(coords1, coords2)
    sup.run()
    return sup.get_rms()


def get_ground_truth_ca(pdb_id, sequence):
    protein_pdb = SELECTED_DIR / pdb_id / f"{pdb_id}_protein.pdb"
    if not protein_pdb.is_file():
        return None, f"Ground truth not found: {protein_pdb}"
    info = PROTEIN_INFO.get(pdb_id)
    if info is None:
        start, end = find_residue_range_by_sequence(protein_pdb, "A", sequence)
        if start is None:
            return None, f"Sequence {sequence} not found in {pdb_id}_protein.pdb"
        chain = "A"
    else:
        sequence, start, end, chain = info
    gt_coords, _ = extract_ca_from_pdb(protein_pdb, chain, start, end)
    if len(gt_coords) == 0:
        return None, f"No Cα for {pdb_id} chain {chain} res {start}-{end}"
    if len(gt_coords) != len(sequence):
        return None, f"Length mismatch: expected {len(sequence)}, got {len(gt_coords)} Cα"
    return gt_coords, None


def process_xyz_structure(xyz_path, pdb_id, sequence):
    atoms = parse_xyz(xyz_path)
    if not atoms:
        return None, "Empty xyz"
    pred_coords = scale_xyz_coords(np.array([[a[1], a[2], a[3]] for a in atoms]))
    gt_coords, err = get_ground_truth_ca(pdb_id, sequence)
    if err:
        return None, err
    if len(pred_coords) != len(gt_coords):
        return None, f"Atom count mismatch: pred {len(pred_coords)}, gt {len(gt_coords)}"
    return compute_rmsd(gt_coords, pred_coords), None


def get_sequence_for_protein(pdb_id):
    info = PROTEIN_INFO.get(pdb_id)
    return info[0] if info else None


# ─── QDockBank: RMSD + affinity baseline ─────────────────────────────────────

def load_qdockbank_baseline():
    """Load RMSD (computed + reference) and average affinity for every QDockBank protein.

    Returns:
        dict[pdb_id] -> {
            "ref_rmsd": float,
            "calc_rmsd": float|None,
            "avg_affinity": float|None,
            "num_runs": int,
            "error": str|None,
        }
    """
    results = {}
    for d in sorted(QDOCKBANK_DIR.iterdir()):
        if not d.is_dir() or d.name in ("selected", "__MACOSX"):
            continue
        pdb_id = d.name

        rmsd_file = d / f"{pdb_id}_RMSD_docking_result.json"
        pred_pdb = d / f"{pdb_id}.pdb"
        if not rmsd_file.is_file() or not pred_pdb.is_file():
            continue

        data = json.loads(rmsd_file.read_text())
        ref_rmsd = data["rmsd"]

        # Extract average affinity from docking results
        avg_affinity = None
        num_runs = 0
        docking = data.get("docking", {})
        if "average_affinity" in docking:
            avg_affinity = docking["average_affinity"]
            num_runs = len(docking.get("runs", []))

        info = PROTEIN_INFO.get(pdb_id)
        if not info:
            results[pdb_id] = {
                "ref_rmsd": ref_rmsd, "calc_rmsd": None,
                "avg_affinity": avg_affinity, "num_runs": num_runs,
                "error": "No metadata",
            }
            continue

        gt_coords, err = get_ground_truth_ca(pdb_id, info[0])
        if err:
            results[pdb_id] = {
                "ref_rmsd": ref_rmsd, "calc_rmsd": None,
                "avg_affinity": avg_affinity, "num_runs": num_runs,
                "error": err,
            }
            continue

        pred_coords = extract_ca_from_predicted_pdb(pred_pdb)
        if len(pred_coords) != len(gt_coords):
            results[pdb_id] = {
                "ref_rmsd": ref_rmsd, "calc_rmsd": None,
                "avg_affinity": avg_affinity, "num_runs": num_runs,
                "error": f"Length mismatch pred={len(pred_coords)} gt={len(gt_coords)}",
            }
            continue

        calc = compute_rmsd(gt_coords, pred_coords)
        results[pdb_id] = {
            "ref_rmsd": ref_rmsd,
            "calc_rmsd": round(calc, 3) if calc is not None else None,
            "avg_affinity": avg_affinity,
            "num_runs": num_runs,
            "error": None,
        }
    return results


# ─── process_data: RMSD for new strategies ────────────────────────────────────

def calc_strategy_rmsd():
    """Compute RMSD for every xyz in Result/process_data.

    Returns:
        list of {"strategy", "pdb_id", "type", "rmsd", "xyz_file"}
    """
    all_results = []
    if not PROCESS_DATA_DIR.is_dir():
        return all_results

    for strategy_dir in sorted(PROCESS_DATA_DIR.iterdir()):
        if not strategy_dir.is_dir():
            continue
        strategy = strategy_dir.name

        for prot_dir in sorted(strategy_dir.iterdir()):
            if not prot_dir.is_dir():
                continue
            pdb_id = prot_dir.name
            sequence = get_sequence_for_protein(pdb_id)
            if sequence is None:
                struct_dir = prot_dir / "Structures"
                main_xyz = struct_dir / f"{pdb_id}_{strategy}.xyz"
                if main_xyz.is_file():
                    atoms = parse_xyz(main_xyz)
                    sequence = "".join(a[0] for a in atoms)
                else:
                    continue

            struct_dir = prot_dir / "Structures"
            if not struct_dir.is_dir():
                continue

            for xyz_file in sorted(struct_dir.glob("*.xyz")):
                rmsd, err = process_xyz_structure(xyz_file, pdb_id, sequence)
                if err or rmsd is None:
                    continue
                label = xyz_file.stem
                if "_top_" in label:
                    entry_type = f"top_{label.split('_top_')[-1]}"
                else:
                    entry_type = "main"
                all_results.append({
                    "strategy": strategy,
                    "pdb_id": pdb_id,
                    "type": entry_type,
                    "rmsd": round(rmsd, 3),
                    "xyz_file": xyz_file.name,
                })
    return all_results


# ─── Docking pipeline ─────────────────────────────────────────────────────────

# Amino acid 1-letter → 3-letter
AA_1TO3 = {v: k for k, v in AA_MAP_3TO1.items()}


def _check_docking_tools():
    """Check if Modeller, obabel, and vina are available."""
    missing = []
    try:
        import modeller  # noqa: F401
    except ImportError:
        missing.append("modeller")
    if shutil.which("obabel") is None:
        missing.append("obabel")
    try:
        from vina import Vina  # noqa: F401
    except ImportError:
        if shutil.which("vina") is None:
            missing.append("vina (pip install vina)")
    return missing


def _build_full_atom_pdb(xyz_path, output_pdb, chain_id="A"):
    """Build full-atom PDB from XYZ via Modeller. Returns True on success."""
    from modeller import environ, log, alignment, model
    from modeller.automodel import automodel, assess

    atoms = parse_xyz(xyz_path)
    if not atoms:
        return False

    coords = np.array([[a[1], a[2], a[3]] for a in atoms])
    coords = scale_xyz_coords(coords)
    sequence = [a[0] for a in atoms]

    work_dir = tempfile.mkdtemp(prefix="dock_")
    ca_pdb = os.path.join(work_dir, "ca_model.pdb")
    ali_file = os.path.join(work_dir, "protein.ali")

    # Write Cα-only PDB
    with open(ca_pdb, "w") as f:
        for i, (res1, (x, y, z)) in enumerate(zip(sequence, coords), start=1):
            res3 = AA_1TO3.get(res1.upper(), "UNK")
            f.write(f"ATOM  {i:5d}  CA  {res3} {chain_id}{i:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n")
        f.write("END\n")

    # Write alignment file
    seq_str = "".join(sequence) + "*"
    n = len(sequence)
    with open(ali_file, "w") as f:
        f.write(f">P1;ca_model\n")
        f.write(f"structureX:ca_model:1:{chain_id}:{n}:{chain_id}::::\n")
        f.write(seq_str + "\n\n")
        f.write(f">P1;protein_full\n")
        f.write(f"sequence:protein_full:1:{chain_id}:{n}:{chain_id}::::\n")
        f.write(seq_str + "\n")

    # Run Modeller
    orig_dir = os.getcwd()
    try:
        os.chdir(work_dir)
        log.none()
        env = environ()
        env.io.atom_files_directory = [work_dir]

        aln = alignment(env)
        mdl = model(env, file=ca_pdb,
                     model_segment=(f"FIRST:{chain_id}", f"LAST:{chain_id}"))
        aln.append_model(mdl, align_codes="ca_model", atom_files=ca_pdb)
        aln.append(file=ali_file, align_codes="protein_full")
        aln.align2d()

        class _M(automodel):
            def special_patches(self, aln):
                self.rename_segments(segment_ids=[chain_id])

        a = _M(env, alnfile=ali_file, knowns="ca_model",
               sequence="protein_full",
               assess_methods=(assess.DOPE, assess.GA341))
        a.starting_model = 1
        a.ending_model = 1
        a.make()

        # Find the generated model file
        generated = [f for f in os.listdir(work_dir)
                     if f.startswith("protein_full") and f.endswith(".pdb")]
        if generated:
            shutil.copy2(os.path.join(work_dir, generated[0]), output_pdb)
            return True
        return False
    except Exception as e:
        print(f"    Modeller error: {e}")
        return False
    finally:
        os.chdir(orig_dir)
        shutil.rmtree(work_dir, ignore_errors=True)


def _pdb_to_pdbqt(pdb_path, pdbqt_path):
    """Convert PDB to PDBQT via obabel, translating COM to origin first."""
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


def _translate_mol2(input_mol2, output_mol2):
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
        # Reconstruct atom line preserving format
        al[2], al[3], al[4] = f"{c[0]:.4f}", f"{c[1]:.4f}", f"{c[2]:.4f}"
        out_lines[idx] = f"{al[0]:>6} {al[1]:<10} {al[2]:>8} {al[3]:>8} {al[4]:>8} {' '.join(al[5:])}\n"

    with open(output_mol2, "w") as f:
        f.writelines(out_lines)


def _run_vina_single(receptor_pdbqt, ligand_mol2, output_dir, seed):
    """Run a single Vina docking trial. Returns average affinity or None."""
    os.makedirs(output_dir, exist_ok=True)
    ligand_pdbqt = os.path.join(output_dir, "ligand.pdbqt")

    # Convert ligand MOL2 → PDBQT
    subprocess.run(
        ["obabel", ligand_mol2, "-O", ligand_pdbqt,
         "--partialcharge", "gasteiger", "-h"],
        check=True, capture_output=True,
    )

    # Calculate ligand center of mass from MOL2
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
    log_file = os.path.join(output_dir, "docking_log.txt")

    scores = []
    # Try Python API first, fall back to CLI
    try:
        from vina import Vina
        v = Vina(sf_name='vina', seed=seed, verbosity=0)
        v.set_receptor(str(receptor_pdbqt))
        v.set_ligand_from_file(ligand_pdbqt)
        v.compute_vina_maps(center=[float(center[0]), float(center[1]), float(center[2])],
                            box_size=[18, 18, 18])
        v.dock(exhaustiveness=16, n_poses=9)
        v.write_poses(output_file, overwrite=True)

        # Extract scores from poses
        energies = v.energies(n_poses=9)
        if energies is not None:
            scores = [float(e[0]) for e in energies]

        with open(log_file, "w") as f:
            f.write(f"Vina Python API, seed={seed}\n")
            for i, s in enumerate(scores, 1):
                f.write(f"  mode {i}: affinity = {s:.3f}\n")

    except Exception as e_api:
        # Fall back to CLI
        if shutil.which("vina"):
            result = subprocess.run(
                ["vina",
                 "--receptor", str(receptor_pdbqt),
                 "--ligand", ligand_pdbqt,
                 "--out", output_file,
                 "--center_x", str(center[0]),
                 "--center_y", str(center[1]),
                 "--center_z", str(center[2]),
                 "--size_x", "18", "--size_y", "18", "--size_z", "18",
                 "--exhaustiveness", "16",
                 "--seed", str(seed)],
                capture_output=True, text=True,
            )
            with open(log_file, "w") as f:
                f.write(result.stdout)
                if result.stderr:
                    f.write(result.stderr)

            if os.path.isfile(output_file):
                with open(output_file) as f:
                    for line in f:
                        if "REMARK VINA RESULT:" in line:
                            parts = line.strip().split()
                            scores.append(float(parts[3]))
        else:
            with open(log_file, "w") as f:
                f.write(f"Vina API error: {e_api}\n")
    return np.mean(scores) if scores else None


def run_docking_for_strategy(strat_results, num_runs=20):
    """Run the full docking pipeline for each structure in strat_results.

    Returns:
        dict[(strategy, pdb_id, type)] -> avg_affinity (float)
    """
    missing = _check_docking_tools()
    if missing:
        print(f"\nDocking SKIPPED — missing tools: {', '.join(missing)}")
        print("Install with: conda install -c conda-forge modeller openbabel autodock-vina")
        return {}

    if not strat_results:
        return {}

    print(f"\n{'=' * 110}")
    print(f"DOCKING: Running {num_runs}-run docking for {len(strat_results)} structures")
    print(f"{'=' * 110}")

    affinity_results = {}

    for entry in strat_results:
        strategy = entry["strategy"]
        pdb_id = entry["pdb_id"]
        entry_type = entry["type"]
        struct_dir = PROCESS_DATA_DIR / strategy / pdb_id / "Structures"
        xyz_file = struct_dir / entry["xyz_file"]

        if not xyz_file.is_file():
            print(f"  {pdb_id}/{strategy}/{entry_type}: SKIP — xyz not found")
            continue

        # Check for ligand in selected/
        ligand_mol2 = SELECTED_DIR / pdb_id / f"{pdb_id}_ligand.mol2"
        if not ligand_mol2.is_file():
            ligand_mol2 = SELECTED_DIR / pdb_id / f"{pdb_id}_ligand.sdf"
        if not ligand_mol2.is_file():
            print(f"  {pdb_id}/{strategy}/{entry_type}: SKIP — no ligand in selected/")
            continue

        # Set up per-type docking directory
        dock_dir = PROCESS_DATA_DIR / strategy / pdb_id / "Docking" / entry_type
        dock_dir.mkdir(parents=True, exist_ok=True)

        # Check if docking already done
        result_json = dock_dir / "docking_result.json"
        if result_json.is_file():
            data = json.loads(result_json.read_text())
            avg_aff = data.get("average_affinity")
            if avg_aff is not None:
                affinity_results[(strategy, pdb_id, entry_type)] = avg_aff
                print(f"  {pdb_id}/{strategy}/{entry_type}: cached affinity = {avg_aff:.2f} kcal/mol")
                continue

        print(f"  {pdb_id}/{strategy}/{entry_type}: building full-atom model...", end=" ", flush=True)

        # Step 1: Build full-atom PDB from xyz
        full_pdb = dock_dir / "full_model.pdb"
        if not _build_full_atom_pdb(str(xyz_file), str(full_pdb)):
            print("FAILED (Modeller)")
            continue
        print("OK")

        # Step 2: Convert to PDBQT
        receptor_pdbqt = dock_dir / "full_model.pdbqt"
        try:
            _pdb_to_pdbqt(full_pdb, receptor_pdbqt)
        except Exception as e:
            print(f"    obabel error: {e}")
            continue

        # Step 3: Translate ligand
        ligand_trans = dock_dir / f"{pdb_id}_ligand_trans.mol2"
        if str(ligand_mol2).endswith(".mol2"):
            _translate_mol2(str(ligand_mol2), str(ligand_trans))
        else:
            # SDF → convert to MOL2 first
            try:
                subprocess.run(
                    ["obabel", str(ligand_mol2), "-O", str(ligand_trans)],
                    check=True, capture_output=True,
                )
                _translate_mol2(str(ligand_trans), str(ligand_trans))
            except Exception as e:
                print(f"    ligand conversion error: {e}")
                continue

        # Step 4: Run docking trials
        run_affinities = []
        for trial in range(1, num_runs + 1):
            seed = random.randint(1, 1000000)
            trial_dir = str(dock_dir / f"trial_{trial}")
            aff = _run_vina_single(str(receptor_pdbqt), str(ligand_trans), trial_dir, seed)
            if aff is not None:
                run_affinities.append(aff)
            print(f"    trial {trial}/{num_runs}: affinity = {aff:.2f} kcal/mol" if aff else
                  f"    trial {trial}/{num_runs}: FAILED")

        if run_affinities:
            avg_aff = float(np.mean(run_affinities))
            affinity_results[(strategy, pdb_id, entry_type)] = avg_aff
            # Cache result
            result_json.write_text(json.dumps({
                "average_affinity": round(avg_aff, 3),
                "num_runs": len(run_affinities),
                "run_affinities": [round(a, 3) for a in run_affinities],
            }, indent=2))
            print(f"  {pdb_id}/{strategy}/{entry_type}: avg affinity = {avg_aff:.2f} kcal/mol ({len(run_affinities)} runs)")
        else:
            print(f"  {pdb_id}/{strategy}/{entry_type}: all docking trials failed")

    return affinity_results


# ─── Display ──────────────────────────────────────────────────────────────────

def print_step1_validation(qdb):
    """STEP 1: Validate RMSD pipeline against QDockBank reference JSON."""
    print("=" * 72)
    print("STEP 1: Validate QDockBank RMSD (calc vs reference JSON)")
    print("=" * 72)
    print(f"{'PDB':<8} {'Ref':>8} {'Calc':>8} {'Diff':>8} {'Status':>8} {'Affinity':>10}")
    print("-" * 56)
    ok = total = 0
    for pid in sorted(qdb):
        r = qdb[pid]
        aff_str = f"{r['avg_affinity']:.2f}" if r["avg_affinity"] is not None else "N/A"
        if r["calc_rmsd"] is None:
            print(f"{pid:<8} {r['ref_rmsd']:>8.3f} {'ERR':>8} {'':>8} {'SKIP':>8} {aff_str:>10}")
            continue
        total += 1
        diff = abs(r["calc_rmsd"] - r["ref_rmsd"])
        st = "OK" if diff < 0.01 else "FAIL"
        if st == "OK":
            ok += 1
        print(f"{pid:<8} {r['ref_rmsd']:>8.3f} {r['calc_rmsd']:>8.3f} {diff:>8.4f} {st:>8} {aff_str:>10}")
    print("-" * 56)
    print(f"Matched: {ok}/{total}\n")


def print_step2_comparison(qdb, strat_results, new_affinity=None):
    """STEP 2: RMSD + affinity comparison — new strategies vs QDockBank baseline."""
    if new_affinity is None:
        new_affinity = {}
    proteins = sorted(set(r["pdb_id"] for r in strat_results))
    strategies = sorted(set(r["strategy"] for r in strat_results))

    # Build lookups
    main_lk = {}
    best_lk = {}
    for r in strat_results:
        key = (r["strategy"], r["pdb_id"])
        if r["type"] == "main":
            main_lk[key] = r["rmsd"]
        cur = best_lk.get(key)
        if cur is None or r["rmsd"] < cur[0]:
            best_lk[key] = (r["rmsd"], r["type"])

    qdb_rmsd = {}
    qdb_aff = {}
    for pid in proteins:
        row = qdb.get(pid, {})
        if row.get("calc_rmsd") is not None:
            qdb_rmsd[pid] = row["calc_rmsd"]
        if row.get("avg_affinity") is not None:
            qdb_aff[pid] = row["avg_affinity"]

    # ── 2a: Main-structure RMSD vs QDockBank ──
    print("=" * 120)
    print("STEP 2a: Main-structure RMSD vs QDockBank baseline (Å)")
    print("         BASELINE = QDockBank (IBM Quantum VQE)")
    print("=" * 120)

    for pid in proteins:
        base_r = qdb_rmsd.get(pid)
        base_a = qdb_aff.get(pid)
        r_str = f"{base_r:.3f}" if base_r is not None else "N/A"
        a_str = f"{base_a:.2f}" if base_a is not None else "N/A"
        print(f"\n  Protein: {pid}   |   QDockBank RMSD: {r_str} Å   Affinity: {a_str} kcal/mol")
        print(f"  {'Strategy':<45} {'RMSD':>8} {'Delta':>10} {'%':>8} {'Result':>10} {'Affinity':>10}")
        print(f"  {'-'*95}")
        for s in strategies:
            v = main_lk.get((s, pid))
            na = new_affinity.get((s, pid, "main"))
            na_str = f"{na:.2f}" if na is not None else ""
            if v is None:
                print(f"  {s:<45} {'N/A':>8} {'':>10} {'':>8} {'':>10} {na_str:>10}")
                continue
            if base_r is not None:
                d = v - base_r
                pct = (d / base_r) * 100 if base_r != 0 else 0
                tag = "BETTER" if d < -0.01 else ("WORSE" if d > 0.01 else "SAME")
                print(f"  {s:<45} {v:>8.3f} {d:>+10.3f} {pct:>+7.0f}% {tag:>10} {na_str:>10}")
            else:
                print(f"  {s:<45} {v:>8.3f} {'':>10} {'':>8} {'':>10} {na_str:>10}")

    # ── 2a summary ──
    overlap = [p for p in proteins if p in qdb_rmsd]
    print("\n" + "=" * 120)
    print("STEP 2a SUMMARY: Average main-structure RMSD")
    print("=" * 120)
    print(f"  {'Method':<50} {'Avg RMSD':>10} {'vs QDockBank':>16} {'QDB Avg Aff':>14} {'New Avg Aff':>14}")
    print(f"  {'-'*106}")

    qdb_avg_r = np.mean([qdb_rmsd[p] for p in overlap]) if overlap else None
    qdb_avg_a = np.mean([qdb_aff[p] for p in overlap if p in qdb_aff]) if overlap else None
    if qdb_avg_r is not None:
        a_str = f"{qdb_avg_a:.2f}" if qdb_avg_a is not None else "N/A"
        print(f"  {'QDockBank (BASELINE)':<50} {qdb_avg_r:>10.3f} {'---':>16} {a_str:>14} {'---':>14}")

    for s in strategies:
        vals = [main_lk[(s, p)] for p in overlap if (s, p) in main_lk]
        aff_vals = [new_affinity[(s, p, "main")] for p in overlap if (s, p, "main") in new_affinity]
        if vals:
            avg = np.mean(vals)
            delta = ""
            if qdb_avg_r is not None:
                d = avg - qdb_avg_r
                delta = f"{d:+.3f} ({d / qdb_avg_r * 100:+.0f}%)"
            aff_str = f"{np.mean(aff_vals):.2f}" if aff_vals else "N/A"
            print(f"  {s:<50} {avg:>10.3f} {delta:>16} {'':>14} {aff_str:>14}")

    # ── 2b: Best RMSD across all top-K ──
    print("\n" + "=" * 110)
    print("STEP 2b: Best RMSD (main + top-K) vs QDockBank baseline (Å)")
    print("=" * 110)

    for pid in proteins:
        base_r = qdb_rmsd.get(pid)
        base_a = qdb_aff.get(pid)
        r_str = f"{base_r:.3f}" if base_r is not None else "N/A"
        a_str = f"{base_a:.2f}" if base_a is not None else "N/A"
        print(f"\n  Protein: {pid}   |   QDockBank RMSD: {r_str} Å   Affinity: {a_str} kcal/mol")
        print(f"  {'Strategy':<45} {'Best':>8} {'From':>8} {'Delta':>10} {'%':>8} {'Result':>10}")
        print(f"  {'-'*93}")
        for s in strategies:
            b = best_lk.get((s, pid))
            if b is None:
                print(f"  {s:<45} {'N/A':>8}")
                continue
            rmsd_val, src = b
            if base_r is not None:
                d = rmsd_val - base_r
                pct = (d / base_r) * 100 if base_r != 0 else 0
                tag = "BETTER" if d < -0.01 else ("WORSE" if d > 0.01 else "SAME")
                print(f"  {s:<45} {rmsd_val:>8.3f} {src:>8} {d:>+10.3f} {pct:>+7.0f}% {tag:>10}")
            else:
                print(f"  {s:<45} {rmsd_val:>8.3f} {src:>8}")

    # ── 2b summary ──
    print("\n" + "=" * 110)
    print("STEP 2b SUMMARY: Average best-candidate RMSD")
    print("=" * 110)
    print(f"  {'Method':<50} {'Avg RMSD':>10} {'vs QDockBank':>16}")
    print(f"  {'-'*78}")
    if qdb_avg_r is not None:
        print(f"  {'QDockBank (BASELINE)':<50} {qdb_avg_r:>10.3f} {'---':>16}")
    for s in strategies:
        vals = [best_lk[(s, p)][0] for p in overlap if (s, p) in best_lk]
        if vals:
            avg = np.mean(vals)
            delta = ""
            if qdb_avg_r is not None:
                d = avg - qdb_avg_r
                delta = f"{d:+.3f} ({d / qdb_avg_r * 100:+.0f}%)"
            print(f"  {s:<50} {avg:>10.3f} {delta:>16}")

    # ── 2c: Overall winner per protein ──
    print("\n" + "=" * 120)
    print("STEP 2c: Overall winner per protein vs QDockBank baseline")
    print("=" * 120)
    print(f"  {'Protein':<8} {'QDB RMSD':>9} {'QDB Aff':>9} {'Best RMSD':>10} {'Delta':>10} "
          f"{'%':>8} {'New Aff':>9} {'Winner Strategy':<45} {'From':>8}")
    print(f"  {'-'*120}")

    wins = 0
    for pid in proteins:
        base_r = qdb_rmsd.get(pid)
        base_a = qdb_aff.get(pid)
        best_all = None
        for s in strategies:
            b = best_lk.get((s, pid))
            if b and (best_all is None or b[0] < best_all[0]):
                best_all = (b[0], b[1], s)

        r_str = f"{base_r:.3f}" if base_r is not None else "N/A"
        a_str = f"{base_a:.2f}" if base_a is not None else "N/A"
        if best_all:
            d_str = pct_str = ""
            if base_r is not None:
                d = best_all[0] - base_r
                pct = (d / base_r) * 100 if base_r != 0 else 0
                d_str = f"{d:+.3f}"
                pct_str = f"{pct:+.0f}%"
                if d < -0.01:
                    wins += 1
            na = new_affinity.get((best_all[2], pid, best_all[1]))
            na_str = f"{na:.2f}" if na is not None else "N/A"
            print(f"  {pid:<8} {r_str:>9} {a_str:>9} {best_all[0]:>10.3f} {d_str:>10} "
                  f"{pct_str:>8} {na_str:>9} {best_all[2]:<45} {best_all[1]:>8}")

    print(f"\n  New framework beats QDockBank on {wins}/{len(overlap)} proteins")


def print_step3_top6(qdb, strat_results):
    """STEP 3: All top-6 candidates vs QDockBank baseline."""
    top_labels = [f"top_{i}" for i in range(1, 7)]
    top_rows = [r for r in strat_results if r["type"] in top_labels]
    if not top_rows:
        print("\n" + "=" * 110)
        print("STEP 3: No top-6 structures found.")
        print("=" * 110)
        return

    qdb_rmsd = {pid: row["calc_rmsd"] for pid, row in qdb.items() if row.get("calc_rmsd") is not None}
    qdb_aff = {pid: row["avg_affinity"] for pid, row in qdb.items() if row.get("avg_affinity") is not None}
    proteins = sorted({r["pdb_id"] for r in top_rows})
    strategies = sorted({r["strategy"] for r in top_rows})
    lookup = {(r["strategy"], r["pdb_id"], r["type"]): r for r in top_rows}

    print("\n" + "=" * 110)
    print("STEP 3: All strategies top-6 vs QDockBank baseline")
    print("=" * 110)

    for pid in proteins:
        base_r = qdb_rmsd.get(pid)
        base_a = qdb_aff.get(pid)
        r_str = f"{base_r:.3f}" if base_r is not None else "N/A"
        a_str = f"{base_a:.2f}" if base_a is not None else "N/A"
        print(f"\n  Protein: {pid}   |   QDockBank RMSD: {r_str} Å   Affinity: {a_str} kcal/mol")
        print(f"  {'Strategy':<45} {'Type':<8} {'RMSD':>8} {'Delta':>10} {'%':>8} {'Result':>10}")
        print(f"  {'-'*93}")
        for s in strategies:
            for t in top_labels:
                row = lookup.get((s, pid, t))
                if row is None:
                    continue
                v = row["rmsd"]
                if base_r is not None:
                    d = v - base_r
                    pct = (d / base_r) * 100 if base_r != 0 else 0
                    tag = "BETTER" if d < -0.01 else ("WORSE" if d > 0.01 else "SAME")
                    print(f"  {s:<45} {t:<8} {v:>8.3f} {d:>+10.3f} {pct:>+7.0f}% {tag:>10}")
                else:
                    print(f"  {s:<45} {t:<8} {v:>8.3f}")


# ─── CSV output ───────────────────────────────────────────────────────────────

def load_cached_affinities():
    """Load cached docking affinities from Result/process_data/**/Docking/<type>/docking_result.json."""
    affinities = {}
    if not PROCESS_DATA_DIR.is_dir():
        return affinities
    for strategy_dir in sorted(PROCESS_DATA_DIR.iterdir()):
        if not strategy_dir.is_dir():
            continue
        for prot_dir in sorted(strategy_dir.iterdir()):
            if not prot_dir.is_dir():
                continue
            dock_dir = prot_dir / "Docking"
            if not dock_dir.is_dir():
                continue
            # Legacy: top-level docking_result.json → treat as "main"
            legacy_json = dock_dir / "docking_result.json"
            if legacy_json.is_file():
                data = json.loads(legacy_json.read_text())
                avg_aff = data.get("average_affinity")
                if avg_aff is not None:
                    affinities[(strategy_dir.name, prot_dir.name, "main")] = avg_aff
            # Per-type subdirectories (main/, top_1/, top_2/, ...)
            for type_dir in sorted(dock_dir.iterdir()):
                if not type_dir.is_dir():
                    continue
                result_json = type_dir / "docking_result.json"
                if result_json.is_file():
                    data = json.loads(result_json.read_text())
                    avg_aff = data.get("average_affinity")
                    if avg_aff is not None:
                        affinities[(strategy_dir.name, prot_dir.name, type_dir.name)] = avg_aff
    return affinities


def save_rmsd_comparison_csv(qdb, strat_results, path):
    """Save CSV with per-row RMSD data, matching the rmsd_comparison.csv format."""
    qdb_rmsd = {pid: row["calc_rmsd"] for pid, row in qdb.items()
                if row.get("calc_rmsd") is not None}

    fieldnames = ["source", "strategy", "pdb_id", "type", "rmsd",
                  "QDockBank_baseline", "delta_vs_baseline", "pct_change", "file"]
    rows = []

    # QDockBank baseline rows
    for pid in sorted(qdb):
        r = qdb[pid]
        rows.append({
            "source": "QDockBank_BASELINE",
            "strategy": "QDockBank",
            "pdb_id": pid,
            "type": "main",
            "rmsd": r["calc_rmsd"] if r["calc_rmsd"] is not None else "",
            "QDockBank_baseline": r["calc_rmsd"] if r["calc_rmsd"] is not None else "",
            "delta_vs_baseline": 0.0 if r["calc_rmsd"] is not None else "",
            "pct_change": "0%" if r["calc_rmsd"] is not None else "",
            "file": f"{pid}.pdb",
        })

    # Strategy rows
    for r in strat_results:
        pid = r["pdb_id"]
        base = qdb_rmsd.get(pid)
        delta = ""
        pct = ""
        if base is not None and r["rmsd"] is not None:
            d = r["rmsd"] - base
            delta = round(d, 3)
            pct = f"{(d / base) * 100:+.0f}%" if base != 0 else ""
        rows.append({
            "source": "NEW_FRAMEWORK",
            "strategy": r["strategy"],
            "pdb_id": pid,
            "type": r["type"],
            "rmsd": r["rmsd"],
            "QDockBank_baseline": base if base is not None else "",
            "delta_vs_baseline": delta,
            "pct_change": pct,
            "file": r["xyz_file"],
        })

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"RMSD comparison CSV saved to {path}  ({len(rows)} rows)")


def save_affinity_comparison_csv(qdb, strat_results, new_affinity, path):
    """Save CSV with per-row average affinity data, same format as rmsd_comparison."""
    qdb_aff = {pid: row["avg_affinity"] for pid, row in qdb.items()
               if row.get("avg_affinity") is not None}

    fieldnames = ["source", "strategy", "pdb_id", "type", "average_affinity",
                  "QDockBank_baseline", "delta_vs_baseline", "pct_change", "file"]
    rows = []

    # QDockBank baseline rows
    for pid in sorted(qdb):
        r = qdb[pid]
        rows.append({
            "source": "QDockBank_BASELINE",
            "strategy": "QDockBank",
            "pdb_id": pid,
            "type": "main",
            "average_affinity": r["avg_affinity"] if r["avg_affinity"] is not None else "",
            "QDockBank_baseline": r["avg_affinity"] if r["avg_affinity"] is not None else "",
            "delta_vs_baseline": 0.0 if r["avg_affinity"] is not None else "",
            "pct_change": "0%" if r["avg_affinity"] is not None else "",
            "file": f"{pid}.pdb",
        })

    # Strategy rows — affinity is per (strategy, pdb_id, type)
    for r in strat_results:
        pid = r["pdb_id"]
        base = qdb_aff.get(pid)
        aff = new_affinity.get((r["strategy"], pid, r["type"]))
        delta = ""
        pct = ""
        if base is not None and aff is not None:
            d = aff - base
            delta = round(d, 3)
            pct = f"{(d / abs(base)) * 100:+.0f}%" if base != 0 else ""
        rows.append({
            "source": "NEW_FRAMEWORK",
            "strategy": r["strategy"],
            "pdb_id": pid,
            "type": r["type"],
            "average_affinity": round(aff, 3) if aff is not None else "",
            "QDockBank_baseline": base if base is not None else "",
            "delta_vs_baseline": delta,
            "pct_change": pct,
            "file": r["xyz_file"],
        })

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Affinity comparison CSV saved to {path}  ({len(rows)} rows)")


# ─── Strategy / entanglement parsing ─────────────────────────────────────────

KNOWN_STRATEGY_PREFIXES = [
    "recursive_forging_depth2",
    "forging_kronecker",
    "subspace_hamiltonian",
    "baseline",
]


def split_strategy_entanglement(full_name):
    """Split e.g. 'forging_kronecker_circular' -> ('forging_kronecker', 'circular')."""
    for prefix in KNOWN_STRATEGY_PREFIXES:
        if full_name.startswith(prefix):
            rest = full_name[len(prefix):]
            ent = rest.lstrip("_") if rest else ""
            return prefix, ent if ent else "default"
    return full_name, "default"


# ─── Best-per-group CSV writers ──────────────────────────────────────────────

def save_best_rmsd_comparison_csv(qdb, strat_results, path):
    """Save CSV: best RMSD per (strategy_type, pdb_id) across all entanglements
    and all top-K xyz files, in same format as rmsd_comparison.csv."""
    qdb_rmsd = {pid: row["calc_rmsd"] for pid, row in qdb.items()
                if row.get("calc_rmsd") is not None}

    # Find best RMSD per (strategy_type, pdb_id) across all entanglements + xyz types
    best = {}  # (strategy_type, pdb_id) -> (rmsd, entanglement, type, xyz_file)
    for r in strat_results:
        if r["rmsd"] is None:
            continue
        strategy_type, ent = split_strategy_entanglement(r["strategy"])
        key = (strategy_type, r["pdb_id"])
        if key not in best or r["rmsd"] < best[key][0]:
            best[key] = (r["rmsd"], ent, r["type"], r["xyz_file"])

    fieldnames = ["source", "strategy", "pdb_id", "type", "best_rmsd",
                  "QDockBank_baseline", "delta_vs_baseline", "pct_change",
                  "best_entanglement", "file"]
    rows = []

    # QDockBank baseline rows
    for pid in sorted(qdb):
        r = qdb[pid]
        rows.append({
            "source": "QDockBank_BASELINE",
            "strategy": "QDockBank",
            "pdb_id": pid,
            "type": "main",
            "best_rmsd": r["calc_rmsd"] if r["calc_rmsd"] is not None else "",
            "QDockBank_baseline": r["calc_rmsd"] if r["calc_rmsd"] is not None else "",
            "delta_vs_baseline": 0.0 if r["calc_rmsd"] is not None else "",
            "pct_change": "0%" if r["calc_rmsd"] is not None else "",
            "best_entanglement": "",
            "file": f"{pid}.pdb",
        })

    for (strategy_type, pid), (rmsd_val, ent, entry_type, xyz_file) in sorted(best.items()):
        base = qdb_rmsd.get(pid)
        delta = ""
        pct = ""
        if base is not None:
            d = rmsd_val - base
            delta = round(d, 3)
            pct = f"{(d / base) * 100:+.0f}%" if base != 0 else ""
        rows.append({
            "source": "NEW_FRAMEWORK",
            "strategy": strategy_type,
            "pdb_id": pid,
            "type": entry_type,
            "best_rmsd": round(rmsd_val, 3),
            "QDockBank_baseline": round(base, 3) if base is not None else "",
            "delta_vs_baseline": delta,
            "pct_change": pct,
            "best_entanglement": ent,
            "file": xyz_file,
        })

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Best-RMSD comparison CSV saved to {path}  ({len(rows)} rows)")


def save_best_affinity_comparison_csv(qdb, new_affinity, path):
    """Save CSV: best (most negative) affinity per (strategy_type, pdb_id) across all
    entanglements, in same format as rmsd_comparison.csv."""
    qdb_aff = {pid: row["avg_affinity"] for pid, row in qdb.items()
               if row.get("avg_affinity") is not None}

    # Collapse across entanglements: keep the most negative (best) per (strategy_type, pid)
    best = {}  # (strategy_type, pdb_id) -> (affinity, entanglement, full_strategy)
    for (full_strat, pid, _entry_type), aff_val in new_affinity.items():
        strategy_type, ent = split_strategy_entanglement(full_strat)
        key = (strategy_type, pid)
        if key not in best or aff_val < best[key][0]:
            best[key] = (aff_val, ent, full_strat)

    fieldnames = ["source", "strategy", "pdb_id", "type", "best_affinity",
                  "QDockBank_baseline", "delta_vs_baseline", "pct_change",
                  "best_entanglement"]
    rows = []

    # QDockBank baseline rows
    for pid in sorted(qdb):
        r = qdb[pid]
        rows.append({
            "source": "QDockBank_BASELINE",
            "strategy": "QDockBank",
            "pdb_id": pid,
            "type": "main",
            "best_affinity": r["avg_affinity"] if r["avg_affinity"] is not None else "",
            "QDockBank_baseline": r["avg_affinity"] if r["avg_affinity"] is not None else "",
            "delta_vs_baseline": 0.0 if r["avg_affinity"] is not None else "",
            "pct_change": "0%" if r["avg_affinity"] is not None else "",
            "best_entanglement": "",
        })

    for (strategy_type, pid), (aff_val, ent, full_strat) in sorted(best.items()):
        base = qdb_aff.get(pid)
        delta = ""
        pct = ""
        if base is not None:
            d = aff_val - base
            delta = round(d, 3)
            pct = f"{(d / abs(base)) * 100:+.0f}%" if base != 0 else ""
        rows.append({
            "source": "NEW_FRAMEWORK",
            "strategy": strategy_type,
            "pdb_id": pid,
            "type": "main",
            "best_affinity": round(aff_val, 3),
            "QDockBank_baseline": round(base, 3) if base is not None else "",
            "delta_vs_baseline": delta,
            "pct_change": pct,
            "best_entanglement": ent,
        })

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Best-affinity comparison CSV saved to {path}  ({len(rows)} rows)")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Parse args
    num_runs = 20
    for i, arg in enumerate(sys.argv):
        if arg == "--dock-runs" and i + 1 < len(sys.argv):
            num_runs = int(sys.argv[i + 1])

    load_protein_info()

    # Step 1: Load QDockBank baseline (RMSD + affinity) and validate
    qdb = load_qdockbank_baseline()
    print_step1_validation(qdb)

    # Step 2: Compute RMSD for process_data strategies
    strat = calc_strategy_rmsd()

    # Step 3: Run docking (cached results reused) and load all cached affinities
    new_affinity = run_docking_for_strategy(strat, num_runs=num_runs)
    cached = load_cached_affinities()
    for key, val in cached.items():
        if key not in new_affinity:
            new_affinity[key] = val

    # Step 4: Print comparison tables
    print_step2_comparison(qdb, strat, new_affinity)
    print_step3_top6(qdb, strat)

    # Step 5: Output 4 CSVs to Result/structure_results/
    out_dir = BASE_DIR / "Result" / "structure_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    save_rmsd_comparison_csv(qdb, strat, out_dir / "rmsd_comparison.csv")
    save_affinity_comparison_csv(qdb, strat, new_affinity,
                                 out_dir / "average_affinity_comparison.csv")
    save_best_rmsd_comparison_csv(qdb, strat,
                                   out_dir / "best_rmsd_comparison.csv")
    save_best_affinity_comparison_csv(qdb, new_affinity,
                                      out_dir / "best_affinity_comparison.csv")
