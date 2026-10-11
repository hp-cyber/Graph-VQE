"""Inspect the 97-qubit global contraction without evolving its quantum state."""
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'sources/independent-vqe'))
os.environ.update(CUDA_VISIBLE_DEVICES='0',CUDAQ_MQPU_NGPUS='1',CUDAQ_MPS_HBATCH_NGPUS='1')
import cudaq
cudaq.set_target('tensornet-mps-mqpu-shots')
from circuit_tape import install
install(cudaq)
from Protein_Folding import Peptide,ProteinFoldingProblem,MiyazawaJerniganInteraction,PenaltyParameters
from Qiskit_VQE.independent_vqe import IndependentSubspaceVQE
from reverse_tensor import TensorCircuit

job=max(json.loads((ROOT/'protein_catalog.json').read_text()),key=lambda x:x['simulation_qubits'])
h=ProteinFoldingProblem(Peptide(job['sequence'],['']*len(job['sequence'])),MiyazawaJerniganInteraction(),PenaltyParameters(10,10,10)).qubit_op()
rows=[]
for pattern in ('linear','reverse_linear'):
    begin=time.perf_counter()
    s=IndependentSubspaceVQE(qiskit_hamiltonian=h,hamiltonian=None,num_qubits=h.num_qubits,
        shots=-1,entanglement=pattern,reps=4,seed=0,max_epochs=1,max_block_qubits=12,enable_parallel=False,force_sequential=True)
    # Build only shape metadata for one term, giving a lower bound on the
    # largest intermediate of the production greedy contraction plan.
    circuit=TensorCircuit(s.kernel,['Z'+'I'*(h.num_qubits-1)],device='cpu')
    labels=circuit.labels+[[i+circuit.offset for i in label] for label in circuit.labels]
    labels += [[2*circuit.offset,q+circuit.offset,q] for q in circuit.ends]
    shapes=[(2,)*len(label) for label in circuit.labels]*2+[(1,2,2)]*h.num_qubits
    equation=','.join(''.join(circuit.oe.get_symbol(i) for i in label) for label in labels)+'->'+circuit.oe.get_symbol(2*circuit.offset)
    _,info=circuit.oe.contract_path(equation,*shapes,shapes=True,optimize='greedy')
    row=dict(protein_id=job['protein_id'],qubits=h.num_qubits,entanglement=pattern,
             blocks=s.blocks,parameter_count=s.num_params,cross_edges=len(s.cross_edges),
             hamiltonian_terms=len(h),largest_single_term_complex_intermediate_bytes=int(info.largest_intermediate)*16,
             planning_seconds=time.perf_counter()-begin,state_evolved=False)
    rows.append(row);print(json.dumps(row),flush=True)
output=ROOT/'validation/largest_graph.json';output.parent.mkdir(exist_ok=True);output.write_text(json.dumps(rows,indent=2)+'\n')
