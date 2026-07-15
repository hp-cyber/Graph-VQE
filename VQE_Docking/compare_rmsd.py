#!/usr/bin/env python3
"""
Per-protein RMSD comparison: QDockBank (validated) vs Result/process_data strategies.

Uses the same compute_rmsd / ground-truth pipeline from calculate_rmsd.py for both
QDockBank PDB files and process_data .xyz files, then prints side-by-side tables.

Usage:
    python compare_rmsd.py
"""

import json
import csv
from pathlib import Path

import numpy as np

# Import everything from the original script (unchanged)
from calculate_rmsd import (
    BASE_DIR,
    QDOCKBANK_DIR,
    SELECTED_DIR,
    PROCESS_DATA_DIR,
    PROTEIN_INFO,
    load_protein_info,
    compute_rmsd,
    get_ground_truth_ca,
    extract_ca_from_predicted_pdb,
    process_xyz_structure,
    get_sequence_for_protein,
    parse_xyz,
)


# ─── QDockBank: compute RMSD with same pipeline, return per-protein dict ─────

def calc_qdockbank_rmsd():
    """
    For every QDockBank predicted PDB, compute RMSD using the same
    get_ground_truth_ca + compute_rmsd pipeline. Also load the reference
    RMSD from *_RMSD_docking_result.json for validation.

    Returns:
        dict[pdb_id] -> {"ref": float, "calc": float|None, "error": str|None}
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

        ref_rmsd = json.loads(rmsd_file.read_text())["rmsd"]

        info = PROTEIN_INFO.get(pdb_id)
        if not info:
            results[pdb_id] = {"ref": ref_rmsd, "calc": None, "error": "No metadata"}
            continue
        sequence = info[0]

        gt_coords, err = get_ground_truth_ca(pdb_id, sequence)
        if err:
            results[pdb_id] = {"ref": ref_rmsd, "calc": None, "error": err}
            continue

        pred_coords = extract_ca_from_predicted_pdb(pred_pdb)
        if len(pred_coords) != len(gt_coords):
            results[pdb_id] = {
                "ref": ref_rmsd, "calc": None,
                "error": f"Length mismatch pred={len(pred_coords)} gt={len(gt_coords)}",
            }
            continue

        calc = compute_rmsd(gt_coords, pred_coords)
        results[pdb_id] = {"ref": ref_rmsd, "calc": round(calc, 3), "error": None}

    return results


# ─── process_data: compute RMSD with same pipeline ──────────────────────────

def calc_strategy_rmsd():
    """
    For every xyz in Result/process_data, compute RMSD using the same
    get_ground_truth_ca + compute_rmsd pipeline (via process_xyz_structure).

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


# ─── Display ─────────────────────────────────────────────────────────────────

def print_validation(qdb):
    print("=" * 60)
    print("STEP 1: Validate QDockBank (calc vs reference JSON)")
    print("=" * 60)
    print(f"{'PDB':<8} {'Ref':>8} {'Calc':>8} {'Diff':>8} {'Status':>8}")
    print("-" * 44)
    ok = total = 0
    for pid in sorted(qdb):
        r = qdb[pid]
        if r["calc"] is None:
            print(f"{pid:<8} {r['ref']:>8.3f} {'ERR':>8} {'':>8} {'SKIP':>8}")
            continue
        total += 1
        diff = abs(r["calc"] - r["ref"])
        st = "OK" if diff < 0.01 else "FAIL"
        if st == "OK":
            ok += 1
        print(f"{pid:<8} {r['ref']:>8.3f} {r['calc']:>8.3f} {diff:>8.4f} {st:>8}")
    print("-" * 44)
    print(f"Matched: {ok}/{total}\n")


def _delta_str(new_val, baseline):
    """Format a delta string: negative = better (green arrow down)."""
    if baseline is None or new_val is None:
        return ""
    d = new_val - baseline
    pct = (d / baseline) * 100 if baseline != 0 else 0
    return f"{d:+.3f} ({pct:+.0f}%)"


def print_comparison(qdb, strat_results):
    """Print per-protein tables with QDockBank as the BASELINE.
    Every new strategy shows its RMSD and delta vs QDockBank.
    Negative delta = improvement over baseline."""

    proteins = sorted(set(r["pdb_id"] for r in strat_results))
    strategies = sorted(set(r["strategy"] for r in strat_results))

    # Lookups
    main_lk = {}   # (strategy, pdb_id) -> rmsd
    best_lk = {}   # (strategy, pdb_id) -> (rmsd, type)
    for r in strat_results:
        key = (r["strategy"], r["pdb_id"])
        if r["type"] == "main":
            main_lk[key] = r["rmsd"]
        cur = best_lk.get(key)
        if cur is None or r["rmsd"] < cur[0]:
            best_lk[key] = (r["rmsd"], r["type"])

    # Baseline lookup
    qdb_lk = {}  # pdb_id -> rmsd (QDockBank baseline)
    for pid in proteins:
        q = qdb.get(pid, {}).get("calc")
        if q is not None:
            qdb_lk[pid] = q

    # ── Table A: Per-protein, per-strategy main RMSD vs QDockBank baseline ──
    print("=" * 100)
    print("STEP 2a: Main-structure RMSD vs QDockBank baseline (Å)")
    print("         BASELINE = QDockBank (IBM Quantum VQE)")
    print("         Delta: negative = better than baseline, positive = worse")
    print("=" * 100)

    for pid in proteins:
        base = qdb_lk.get(pid)
        base_str = f"{base:.3f}" if base is not None else "N/A"
        print(f"\n  Protein: {pid}   |   QDockBank baseline RMSD: {base_str} Å")
        print(f"  {'Strategy':<40} {'RMSD':>8} {'Delta':>10} {'% Change':>10} {'Result':>10}")
        print(f"  {'-'*82}")

        for s in strategies:
            v = main_lk.get((s, pid))
            if v is None:
                print(f"  {s:<40} {'N/A':>8}")
                continue
            if base is not None:
                d = v - base
                pct = (d / base) * 100 if base != 0 else 0
                tag = "BETTER" if d < -0.01 else ("WORSE" if d > 0.01 else "SAME")
                print(f"  {s:<40} {v:>8.3f} {d:>+10.3f} {pct:>+9.0f}% {tag:>10}")
            else:
                print(f"  {s:<40} {v:>8.3f} {'':>10} {'':>10} {'':>10}")

    # ── Summary averages for main structures ──
    print("\n" + "=" * 100)
    print("STEP 2a SUMMARY: Average main-structure RMSD across all proteins")
    print("=" * 100)
    print(f"  {'Method':<45} {'Avg RMSD':>10} {'vs QDockBank':>14}")
    print(f"  {'-'*70}")

    # QDockBank baseline average (only for overlapping proteins)
    overlap_pids = [p for p in proteins if p in qdb_lk]
    if overlap_pids:
        qdb_avg = np.mean([qdb_lk[p] for p in overlap_pids])
        print(f"  {'QDockBank (BASELINE)' :<45} {qdb_avg:>10.3f} {'---':>14}")
    else:
        qdb_avg = None

    for s in strategies:
        vals = [main_lk[(s, p)] for p in overlap_pids if (s, p) in main_lk]
        if vals:
            avg = np.mean(vals)
            delta = ""
            if qdb_avg is not None:
                d = avg - qdb_avg
                delta = f"{d:+.3f} ({d/qdb_avg*100:+.0f}%)"
            print(f"  {s:<45} {avg:>10.3f} {delta:>14}")

    # ── Table B: Best-of-all-candidates RMSD vs QDockBank baseline ──
    print("\n" + "=" * 100)
    print("STEP 2b: Best RMSD (across main + all top-K) vs QDockBank baseline (Å)")
    print("         Picks the single best structure from each strategy for each protein")
    print("=" * 100)

    for pid in proteins:
        base = qdb_lk.get(pid)
        base_str = f"{base:.3f}" if base is not None else "N/A"
        print(f"\n  Protein: {pid}   |   QDockBank baseline RMSD: {base_str} Å")
        print(f"  {'Strategy':<40} {'Best RMSD':>10} {'From':>8} {'Delta':>10} {'% Change':>10} {'Result':>10}")
        print(f"  {'-'*92}")

        for s in strategies:
            b = best_lk.get((s, pid))
            if b is None:
                print(f"  {s:<40} {'N/A':>10}")
                continue
            rmsd_val, src = b
            if base is not None:
                d = rmsd_val - base
                pct = (d / base) * 100 if base != 0 else 0
                tag = "BETTER" if d < -0.01 else ("WORSE" if d > 0.01 else "SAME")
                print(f"  {s:<40} {rmsd_val:>10.3f} {src:>8} {d:>+10.3f} {pct:>+9.0f}% {tag:>10}")
            else:
                print(f"  {s:<40} {rmsd_val:>10.3f} {src:>8}")

    # ── Summary averages for best candidates ──
    print("\n" + "=" * 100)
    print("STEP 2b SUMMARY: Average best-candidate RMSD across all proteins")
    print("=" * 100)
    print(f"  {'Method':<45} {'Avg RMSD':>10} {'vs QDockBank':>14}")
    print(f"  {'-'*70}")
    if qdb_avg is not None:
        print(f"  {'QDockBank (BASELINE)' :<45} {qdb_avg:>10.3f} {'---':>14}")
    for s in strategies:
        vals = [best_lk[(s, p)][0] for p in overlap_pids if (s, p) in best_lk]
        if vals:
            avg = np.mean(vals)
            delta = ""
            if qdb_avg is not None:
                d = avg - qdb_avg
                delta = f"{d:+.3f} ({d/qdb_avg*100:+.0f}%)"
            print(f"  {s:<45} {avg:>10.3f} {delta:>14}")

    # ── Table C: Overall winner per protein ──
    print("\n" + "=" * 100)
    print("STEP 2c: Overall winner per protein (best candidate across all strategies)")
    print("         Compared against QDockBank baseline")
    print("=" * 100)
    print(f"  {'Protein':<8} {'QDockBank':>10} {'Best New RMSD':>14} {'Delta':>10} "
          f"{'% Change':>10} {'Winner Strategy':<40} {'From':>8}")
    print(f"  {'-'*104}")

    wins = 0
    for pid in proteins:
        base = qdb_lk.get(pid)
        best_all = None
        for s in strategies:
            b = best_lk.get((s, pid))
            if b and (best_all is None or b[0] < best_all[0]):
                best_all = (b[0], b[1], s)

        base_str = f"{base:.3f}" if base is not None else "N/A"
        if best_all:
            d_str = pct_str = ""
            if base is not None:
                d = best_all[0] - base
                pct = (d / base) * 100 if base != 0 else 0
                d_str = f"{d:+.3f}"
                pct_str = f"{pct:+.0f}%"
                if d < -0.01:
                    wins += 1
            print(f"  {pid:<8} {base_str:>10} {best_all[0]:>14.3f} {d_str:>10} "
                  f"{pct_str:>10} {best_all[2]:<40} {best_all[1]:>8}")

    print(f"\n  New framework beats QDockBank baseline on {wins}/{len(overlap_pids)} proteins")


def save_unified_csv(qdb, strat_results, path):
    """Save CSV with clear baseline (QDockBank) column and delta for every row."""
    # Build QDockBank baseline lookup
    qdb_lk = {}
    for pid, r in qdb.items():
        if r.get("calc") is not None:
            qdb_lk[pid] = r["calc"]

    fieldnames = [
        "source", "strategy", "pdb_id", "type", "rmsd",
        "QDockBank_baseline", "delta_vs_baseline", "pct_change",
        "file",
    ]
    rows = []

    # QDockBank rows (baseline itself)
    for pid in sorted(qdb):
        r = qdb[pid]
        rows.append({
            "source": "QDockBank_BASELINE",
            "strategy": "QDockBank",
            "pdb_id": pid,
            "type": "main",
            "rmsd": r["calc"],
            "QDockBank_baseline": r["calc"],
            "delta_vs_baseline": 0.0 if r["calc"] is not None else "",
            "pct_change": "0%" if r["calc"] is not None else "",
            "file": f"{pid}.pdb",
        })

    # Strategy rows (new framework)
    for r in strat_results:
        pid = r["pdb_id"]
        base = qdb_lk.get(pid)
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
    print(f"\nUnified CSV saved to {path}")


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    load_protein_info()

    # Both use the SAME compute_rmsd + get_ground_truth_ca pipeline
    qdb = calc_qdockbank_rmsd()
    strat = calc_strategy_rmsd()

    print_validation(qdb)
    print_comparison(qdb, strat)
    save_unified_csv(qdb, strat, BASE_DIR / "rmsd_comparison.csv")
