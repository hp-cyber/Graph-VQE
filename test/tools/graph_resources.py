"""Allocation preflight for the nearest-neighbor global ansatz.

With zero cutoffs the backend retains min(m,n,maxBond) SVD columns, including
null columns. Their allocated dimensions can therefore be predicted without
evolving a state. This is an allocation bound, not a claim about Schmidt rank.
"""
import os
from worker_resources import gpu_memory
from optimization_trace import atomic


def allocation_bound(width, edges, max_bond, budget=None):
    bonds=[1]*(width+1)
    peak=32*width;largest=1;steps=0
    for a,b in edges:
        a,b=sorted((a,b))
        if not 0<=a<b<width:raise ValueError('Invalid circuit edge')
        # Matches ShardedMPS::applyTwoQubit, including restoring SWAPs.
        for cut in [*range(b-1,a,-1),a,*range(a+1,b)]:
            left,right=bonds[cut],bonds[cut+2]
            m,n=2*left,2*right
            rank=min(m,n,max_bond)
            state=32*sum(x*y for x,y in zip(bonds,bonds[1:]))
            # Matrix, direct-SVD copies/factors and state copies. Workspace
            # reserve is conservative; vendor workspace can vary by version.
            estimated=2*state+8*16*m*n
            peak=max(peak,estimated);largest=max(largest,rank);steps+=1
            if budget is not None and peak>budget:
                return dict(passed=False,estimated_peak_bytes=peak,max_allocated_bond=largest,
                            routed_two_qubit_gates=steps,first_over_budget_edge=[a,b])
            bonds[cut+1]=rank
    return dict(passed=True,estimated_peak_bytes=peak,max_allocated_bond=largest,routed_two_qubit_gates=steps)


def check_solver(solver,case,gpu):
    from Qiskit_VQE.independent_vqe import pairs_for_pattern
    # 80% of the GPU's currently free memory (CAMPAIGN_JOB_GPU_BYTES overrides);
    # the scheduler sets no per-job budget.
    budget=int(os.environ.get('CAMPAIGN_JOB_GPU_BYTES','0'))
    if not budget:budget=int(gpu_memory()[gpu][1]*.8)
    width=max(2,solver.num_qubits)
    cap=1<<min(width//2,solver.reps)
    local_width=max(m['width'] for m in solver.local_models)
    local_cap=1<<min(local_width//2,solver.reps)
    def edges():
        for layer in range(solver.reps):
            yield from pairs_for_pattern(solver.num_qubits,layer,solver.entanglement)
    report=dict(qubits=width,local_max_bond=local_cap,global_max_bond=cap,
                hamiltonian_cross_edges=len(solver.cross_edges),additional_cross_gates=0,gpu_budget_bytes=budget,
                policy='nearest_neighbor_su2_v1; zero cutoffs; predicted SVD allocations',
                **allocation_bound(width,edges(),cap,budget))
    atomic(case/'resource_preflight.json',report)
    if not report['passed']:
        raise MemoryError(f'Exact global graph circuit exceeds this job GPU budget before simulation: '
            f"{report['estimated_peak_bytes']/1024**3:.2f} GiB estimated vs {budget/1024**3:.2f} GiB budget. "
            'No reduced bond cap or approximate state was substituted. See resource_preflight.json.')
    return report
