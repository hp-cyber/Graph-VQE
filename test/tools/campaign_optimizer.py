"""Select one optimizer for this process; all solvers share its probe context."""
import importlib
import os

OPTIMIZERS={'cobyla':('COBYLA','cobyla_optimizer'),
            'lbfgs':('L-BFGS-B','lbfgs_optimizer'),
            'spsa':('SPSA','spsa_optimizer')}

def optimizer_key():
    key=os.environ.get('ANALYTIC_MPS_OPTIMIZER','cobyla').lower()
    if key not in OPTIMIZERS:raise ValueError(f'Unknown analytic MPS optimizer: {key}')
    return key

def optimizer_name():
    return OPTIMIZERS[optimizer_key()][0]

def backend():
    return importlib.import_module(OPTIMIZERS[optimizer_key()][1])

def is_gradient_probe():
    return backend().is_gradient_probe()

def minimize(*args,**kwargs):
    from optimization_trace import minimize as traced_minimize
    return traced_minimize(backend(),*args,**kwargs)

def protocol_name():
    return "analytic_mps_cobyla_v1" if optimizer_key()=="cobyla" else f"analytic_mps_{optimizer_key()}_v1"
