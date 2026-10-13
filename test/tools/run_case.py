"""Run one protein/shot/optimizer/entanglement case on one GPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from job_state import reserve_case
from optimization_trace import atomic
from readout import protein_probabilities

ROOT=Path(__file__).resolve().parents[1]


def case_environment(job,case,gpu):
    env={k:v for k,v in os.environ.items() if not k.startswith(('VQE_','CUDAQ_MPS_','CUDAQ_MQPU_','EQC_','EDEF_','CLUSTER_','ANALYTIC_MPS_'))}
    cap=os.environ.get('GRAPH_VQE_MAX_ITER','1000000')
    env.update(CUDA_VISIBLE_DEVICES=str(gpu),CUDAQ_MPS_NUM_GPUS='1',CUDAQ_MQPU_NGPUS='1',
        CUDAQ_MPS_HBATCH_NGPUS='1',CUDAQ_MPS_ANALYTIC_MODE='shared',CUDAQ_MPS_ABS_CUTOFF='0',CUDAQ_MPS_RELATIVE_CUTOFF='0',
        VQE_SHOTS=str(job['shots']),VQE_ENTANGLEMENT=job['entanglement'],
        ANALYTIC_MPS_OPTIMIZER=job['optimizer'],ANALYTIC_MPS_TRACE_DIR=str(case),ANALYTIC_MPS_CENTRAL_STOP='1',
        ANALYTIC_MPS_IMMEDIATE_ENERGY_TOL='1',ANALYTIC_MPS_LBFGS_REVERSE_MODE='required' if job['shots']==-1 else 'hybrid',
        ANALYTIC_MPS_LBFGS_MAX_ITER=cap,ANALYTIC_MPS_SPSA_MAX_ITER=cap,
        ANALYTIC_MPS_METRICS=str(case/'cobyla_evaluations.jsonl'),ANALYTIC_MPS_LBFGS_METRICS=str(case/'lbfgs_evaluations.jsonl'),
        ANALYTIC_MPS_SPSA_METRICS=str(case/'spsa_evaluations.jsonl'),ANALYTIC_MPS_SPSA_STATE_DIR=str(case/'spsa_states'),
        ANALYTIC_MPS_BACKEND_METRICS=str(case/'backend_evaluations.jsonl'),ANALYTIC_MPS_SPSA_CALIBRATION_PAIRS='5')
    return env


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('case',type=Path);parser.add_argument('--gpu',type=int,required=True)
    parser.add_argument('--lock-fd',type=int)
    args=parser.parse_args();case=args.case.resolve();job=json.loads((case/'job.json').read_text())
    lock,reason=reserve_case(case,args.lock_fd)
    if reason:print(f'Skipping {case}: {reason}',flush=True);return
    if job['shots'] not in (-1,1000) or job['optimizer'] not in ('cobyla','lbfgs','spsa') or job['entanglement'] not in ('linear','reverse_linear'):
        raise ValueError('Invalid matched case')
    env=case_environment(job,case,args.gpu);os.environ.clear();os.environ.update(env)
    os.chdir(case);sys.path.insert(0,str(ROOT/'sources/independent-vqe'))
    protocol=dict(job,gpu=args.gpu,target='tensornet-mps-mqpu-shots',max_iterations=int(env['GRAPH_VQE_MAX_ITER']) if 'GRAPH_VQE_MAX_ITER' in env else 1000000,
        energy_tolerance=1,patience=0,window=0,local_runs_per_partition=1,global_runs=1,
        global_ansatz='nearest_neighbor_su2_v1',parameter_order='layer/qubit/gate',
        reps=4,max_block_qubits=12,seed=0,spsa_calibration_pairs=5,
        gradient_estimator='analytic reverse mode' if job['optimizer']=='lbfgs' else job['optimizer'],
        reverse_mode=env['ANALYTIC_MPS_LBFGS_REVERSE_MODE'],readout=job['shots']==1000,added_noise_model=False,
        analytic_mode='shared',forward_bond_policy='min(2**reps,2**floor(width/2)); zero cutoffs',started=time.time())
    backend_root=Path(env.get('GRAPH_VQE_BACKEND_INSTALL',str(Path(env['CUDAQ_LOCAL'])/'cudaq')))
    protocol['backend_manifest']=json.loads((backend_root/'analytic_backend_manifest.json').read_text())
    protocol['strategy_sha256']=hashlib.sha256((ROOT/'sources/independent-vqe/Qiskit_VQE/independent_vqe.py').read_bytes()).hexdigest()
    old=case/'protocol.json'
    if old.exists():
        previous=json.loads(old.read_text())
        for k in ('sequence','protein_id','shots','optimizer','entanglement','energy_tolerance','reverse_mode','global_ansatz','parameter_order'):
            if previous.get(k)!=protocol[k]:raise ValueError(f'Incompatible resume setting: {k}')
    atomic(old,protocol)
    started=time.perf_counter()
    try:
        import cudaq
        import numpy as np
        cudaq.set_target('tensornet-mps-mqpu-shots');cudaq.set_random_seed(0)
        if job['shots']==-1:
            from analytic_execution import install
        else:
            from sampled_execution import install
        install('independent-vqe')
        from Protein_Folding import Peptide
        from Protein_Folding.interactions.miyazawa_jernigan_interaction import MiyazawaJerniganInteraction
        from Protein_Folding.penalty_parameters import PenaltyParameters
        from Protein_Folding.protein_folding_problem import ProteinFoldingProblem
        from independent_campaign import CampaignVQE
        problem=ProteinFoldingProblem(Peptide(job['sequence'],['']*len(job['sequence'])),MiyazawaJerniganInteraction(),PenaltyParameters(10,10,10))
        h=problem.qubit_op()
        if h.num_qubits!=job['simulation_qubits']:raise ValueError('Protein qubit count differs from job')
        solver=CampaignVQE(qiskit_hamiltonian=h,hamiltonian=None,num_qubits=h.num_qubits,shots=job['shots'],
            entanglement=job['entanglement'],reps=4,seed=0,energy_tol=1.,max_block_qubits=12,
            block_maxiter=protocol['max_iterations'],global_maxiter=protocol['max_iterations'],max_epochs=1,
            global_refinement=True,use_gradient_optimizer=False,enable_parallel=False,force_sequential=True,
            timing_directory=str(case/'System_Energy'))
        atomic(case/'partition.json',dict(solver.partition_summary,parameter_count=solver.num_params,
            local_parameter_counts=[model['count'] for model in solver.local_models]))
        from graph_resources import check_solver
        check_solver(solver,case,args.gpu)
        energy,params,kernel,top=solver.run_vqe()
        np.save(case/'final_parameters.npy',params)
        atomic(case/'state.json',dict(parameters=params.tolist(),parameter_count=len(params),partition=solver.partition_summary,
              entanglement=job['entanglement'],reps=solver.reps))
        atomic(case/'energy_summary.json',dict(best_energy=float(top[0][0]),warm_start_energy=solver.warm_start_energy,
            energy_history=energy,parameter_count=len(params),local_results=solver.local_results,global_results=solver.global_results,
            seconds=time.perf_counter()-started))
        if job['shots']==1000:
            counts=cudaq.sample(kernel,*params.tolist(),shots_count=1000)
            counts={str(k):int(v) for k,v in counts.items()}
            if sum(counts.values())!=1000:raise RuntimeError('Incorrect final readout shot count')
            probabilities=protein_probabilities(counts)
            atomic(case/'readout.json',dict(shots=1000,counts=counts,
                counts_bit_order='q0_left',protein_probabilities=probabilities,
                protein_bit_order='q0_right'))
            structures=case/'Structures';structures.mkdir(exist_ok=True)
            problem.interpret(probabilities).save_xyz_file(name=job['protein_id'],path=str(structures))
        atomic(case/'job_status.json',dict(status='complete',finished=time.time(),seconds=time.perf_counter()-started))
        print(f'Completed {job["key"]}: best energy={top[0][0]}',flush=True)
    except BaseException:
        atomic(case/'job_status.json',dict(status='failed',finished=time.time(),traceback=traceback.format_exc()))
        raise
    finally:lock.close()


if __name__=='__main__':main()
