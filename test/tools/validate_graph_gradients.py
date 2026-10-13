"""Compare derivatives of real local and nearest-neighbor global circuits to CUDA-Q.

Run after sourcing environment.sh, with an otherwise unused physical GPU.
Checks both entanglement orders at generic nonzero rotation angles.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from run_case import ROOT,case_environment


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=0)
    args=p.parse_args()
    output=ROOT/'validation/gradients';output.mkdir(parents=True,exist_ok=True)
    job=next(j for j in json.loads((ROOT/'campaign.json').read_text())['jobs']
             if j['protein_id']=='3ckz' and j['shots']==-1 and j['optimizer']=='lbfgs' and j['entanglement']=='linear')
    env=case_environment(job,output,args.gpu);os.environ.clear();os.environ.update(env)
    sys.path.insert(0,str(ROOT/'sources/independent-vqe'))
    import cudaq
    from analytic_execution import install
    cudaq.set_target('tensornet-mps-mqpu-shots');install('independent-vqe')
    from Protein_Folding import Peptide
    from Protein_Folding.interactions.miyazawa_jernigan_interaction import MiyazawaJerniganInteraction
    from Protein_Folding.penalty_parameters import PenaltyParameters
    from Protein_Folding.protein_folding_problem import ProteinFoldingProblem
    from Qiskit_VQE.independent_vqe import IndependentSubspaceVQE
    from Qiskit_VQE.mps_config import configure_full_bond
    from reverse_mps import make_problem
    h=ProteinFoldingProblem(Peptide(job['sequence'],['']*len(job['sequence'])),MiyazawaJerniganInteraction(),PenaltyParameters(10,10,10)).qubit_op()
    rows=[];rng=np.random.default_rng(534)
    for pattern in ('linear','reverse_linear'):
        solver=IndependentSubspaceVQE(qiskit_hamiltonian=h,hamiltonian=None,num_qubits=h.num_qubits,
            shots=-1,entanglement=pattern,reps=4,seed=0,max_epochs=1,max_block_qubits=12,
            enable_parallel=False,force_sequential=True)
        model=next(m for m in solver.local_models if not m['constant'])
        models=[('local',model),('global',dict(kernel=solver.kernel,hamiltonian=solver.hamiltonian,
                                            width=solver.num_qubits,count=solver.num_params))]
        for phase,model in models:
            configure_full_bond(model['width'],stages=[dict(entanglement=pattern,reps=4)])
            x=rng.uniform(-.4,.4,model['count'])
            def energy(v):return float(cudaq.observe(model['kernel'],model['hamiltonian'],*v.tolist(),shots_count=-1).expectation())
            reverse=make_problem(model['kernel'],model['hamiltonian'])
            started=time.perf_counter();value,gradient=reverse.value_gradient(x);seconds=time.perf_counter()-started
            reference=energy(x)
            np.testing.assert_allclose(value,reference,rtol=1e-9,atol=1e-7)
            checks=[]
            from qiskit import QuantumCircuit
            from qiskit.quantum_info import Statevector
            from reverse_tensor import angle_value
            import torch
            pauli=model.get('pauli',h)
            def dense(v):
                q=QuantumCircuit(model['width']);angles=torch.as_tensor(v,dtype=torch.float64)
                for name,qubits,expression in model['kernel'].operations:
                    if expression is None:getattr(q,name)(*qubits)
                    else:getattr(q,name)(float(angle_value(expression,angles)),*qubits)
                return float(Statevector(q).expectation_value(pauli).real)
            np.testing.assert_allclose(value,dense(x),rtol=1e-11,atol=1e-10)
            shift_errors=[]
            for index in sorted({0,len(x)//2,len(x)-1}):
                direction=np.zeros(len(x));direction[index]=1.;checks.append(direction)
                shifted=(energy(x+np.pi/2*direction)-energy(x-np.pi/2*direction))/2
                np.testing.assert_allclose(gradient[index],shifted,rtol=1e-6,atol=1e-7)
                shift_errors.append(abs(gradient[index]-shifted))
            direction=rng.normal(size=len(x));direction/=np.linalg.norm(direction);checks.append(direction)
            errors=[]
            for direction in checks:
                # Dense seven-qubit reference avoids amplifying small SVD
                # roundoff in the native MPS by division through a tiny step.
                finite=(dense(x+1e-5*direction)-dense(x-1e-5*direction))/2e-5
                actual=float(gradient@direction)
                np.testing.assert_allclose(actual,finite,rtol=1e-5,atol=2e-5)
                errors.append(abs(actual-finite))
            row=dict(entanglement=pattern,phase=phase,qubits=model['width'],parameter_count=len(x),
                     cross_edges=len(solver.cross_edges) if phase=='global' else 0,
                     engine=reverse.engine,energy_error=abs(value-reference),max_derivative_error=max(errors),
                     max_native_parameter_shift_error=max(shift_errors),
                     reverse_seconds=seconds,passed=True)
            rows.append(row);print(json.dumps(row),flush=True)
    (output/'report.json').write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':main()
