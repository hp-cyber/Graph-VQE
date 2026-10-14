"""Validate 97-qubit global energy, reverse gradients and 1000-shot readout."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=2);args=p.parse_args()
os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu),CUDAQ_MQPU_NGPUS='1',CUDAQ_MPS_HBATCH_NGPUS='1',
                  CUDAQ_MPS_ANALYTIC_MODE='shared',ANALYTIC_MPS_OPTIMIZER='lbfgs',ANALYTIC_MPS_LBFGS_REVERSE_MODE='required')
out=ROOT/'validation/global_chain_97';out.mkdir(parents=True,exist_ok=True)
os.environ['ANALYTIC_MPS_BACKEND_METRICS']=str(out/'backend_evaluations.jsonl')
sys.path.insert(0,str(ROOT/'sources/independent-vqe'))
import cudaq
cudaq.set_target('tensornet-mps-mqpu-shots');cudaq.set_random_seed(534)
native_observe,native_sample=cudaq.observe,cudaq.sample
from analytic_execution import install
install('independent-vqe')
from circuit_tape import native
from Protein_Folding import Peptide,ProteinFoldingProblem,MiyazawaJerniganInteraction,PenaltyParameters
from Qiskit_VQE.independent_vqe import IndependentSubspaceVQE
from Qiskit_VQE.mps_config import configure_full_bond
from reverse_mps import make_problem
from graph_resources import check_solver
job=next(j for j in json.loads((ROOT/'protein_catalog.json').read_text()) if j['protein_id']=='3d7z')
h=ProteinFoldingProblem(Peptide(job['sequence'],['']*len(job['sequence'])),MiyazawaJerniganInteraction(),PenaltyParameters(10,10,10)).qubit_op()
rows=[]
for pattern in ('linear','reverse_linear'):
    solver=IndependentSubspaceVQE(qiskit_hamiltonian=h,hamiltonian=None,num_qubits=h.num_qubits,
        shots=-1,entanglement=pattern,reps=4,seed=0,max_epochs=1,max_block_qubits=12,enable_parallel=False,force_sequential=True)
    case=out/pattern;case.mkdir(exist_ok=True)
    report=check_solver(solver,case,args.gpu)
    assert report['global_max_bond']==16 and report['passed']
    assert solver.num_params==970 and solver.partition_summary['cross_parameter_count']==0
    assert len([op for op in solver.kernel.operations if op[0]=='cx'])==4*96
    assert all(abs(q[0]-q[1])==1 for name,q,_ in solver.kernel.operations if name=='cx')
    indices=[]
    for model in solver.local_models:
        expected=[(gate,q,layer,g) for layer in range(5) for q in model['qubits'] for g,gate in enumerate(solver.su2_gates)]
        assert [solver.param_order[i] for i in model['parameter_indices']]==expected
        indices+=model['parameter_indices']
    assert sorted(indices)==list(range(970))
    configure_full_bond(97,stages=[dict(entanglement=pattern,reps=4)])
    x=np.random.default_rng(534).uniform(-.4,.4,solver.num_params)
    reverse=make_problem(solver.kernel,solver.hamiltonian)
    start=time.perf_counter();value,gradient=reverse.value_gradient(x);gradient_seconds=time.perf_counter()-start
    def energy(v,shots=-1):
        return float(native_observe(native(solver.kernel),solver.hamiltonian,*v.tolist(),shots_count=shots).expectation())
    start=time.perf_counter();reference=energy(x);energy_seconds=time.perf_counter()-start
    np.testing.assert_allclose(value,reference,rtol=1e-9,atol=1e-7)
    assert np.isfinite(gradient).all()
    errors=[]
    for i in (0,len(x)//2,len(x)-1):
        shift=np.zeros_like(x);shift[i]=np.pi/2
        expected=(energy(x+shift)-energy(x-shift))/2
        np.testing.assert_allclose(gradient[i],expected,rtol=1e-6,atol=1e-6)
        errors.append(abs(gradient[i]-expected))
    sampled=energy(x,1000);assert np.isfinite(sampled)
    counts=dict(native_sample(native(solver.kernel),*x.tolist(),shots_count=1000).items())
    assert sum(counts.values())==1000 and all(len(k)==97 for k in counts)
    row=dict(protein_id=job['protein_id'],qubits=97,entanglement=pattern,parameters=len(x),max_bond=16,
             engine=reverse.engine,energy=reference,energy_error=abs(value-reference),energy_seconds=energy_seconds,
             gradient_seconds=gradient_seconds,max_checked_gradient_error=max(errors),sampled_energy=sampled,
             readout_shots=sum(counts.values()),passed=True)
    rows.append(row);print(json.dumps(row),flush=True)
    (out/'report.json').write_text(json.dumps(rows,indent=2)+'\n')
