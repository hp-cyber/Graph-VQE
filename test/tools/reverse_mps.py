"""Exact reverse-mode gradients for registered shallow nearest-neighbor circuits.

No SVD, truncation or dense 2**n state is used. A CNOT doubles one MPS bond
using its exact two-term operator decomposition. Analytic gradients are also
available explicitly as a hybrid estimator for finite-shot objective values.
"""
import os
import numpy as np
_REGISTRY = {}


def register(kernel,n,reps,gates,pattern,initial,param_order,pairs,final):
    if pattern not in ('linear', 'reverse_linear') or reps>6 or initial not in (None,'0','1','+','-'):
        return
    if any(not isinstance(g,str) or g.lower() not in ('rx','ry','rz') for g in gates):return
    rotations={}
    for index,(_,q,layer,g) in enumerate(param_order):
        rotations.setdefault(layer,[]).append((gates[g].lower(),q,index))
    ops=[]
    for layer in range(reps+1):
        ops.extend(rotations.get(layer,[]))
        if layer<reps:
            if any(t!=c+1 for c,t in pairs[layer]):return
            ops.extend(('cx',c,t) for c,t in pairs[layer])
    _REGISTRY[id(kernel)]=(kernel,dict(n=n,ops=ops,initial=initial,count=len(param_order)))


class ReverseMPS:
    engine = 'reverse_mps'
    def __init__(self,spec,words,coeffs,device=None):
        import torch
        self.torch=torch;self.spec=spec
        self.device=device or os.environ.get('VQE_REVERSE_DEVICE','cuda')
        self.words=words;self.coeffs=np.asarray(coeffs,dtype=float)
        self.chunk=int(os.environ.get('VQE_REVERSE_TERM_BATCH','128'))
        if self.chunk<1:raise ValueError('VQE_REVERSE_TERM_BATCH must be positive')
        self.dtype=torch.complex128

    def sites(self,x):
        t=self.torch;n=self.spec['n'];dev=self.device
        initial=self.spec['initial']
        vec={'0':[1,0],None:[1,0],'1':[0,1],'+':[2**-.5,2**-.5],'-':[2**-.5,-2**-.5]}[initial]
        sites=[t.tensor(vec,dtype=self.dtype,device=dev).reshape(1,2,1) for _ in range(n)]
        for gate,q,index in self.spec['ops']:
            if gate=='cx':
                a,b=sites[q],sites[index]
                eye=t.eye(2,dtype=self.dtype,device=dev)
                left=t.einsum('aib,is->aibs',a,eye).reshape(a.shape[0],2,a.shape[2]*2)
                right=t.stack((b,b.flip(1)),dim=1).reshape(b.shape[0]*2,2,b.shape[2])
                sites[q],sites[index]=left,right
            else:
                angle=x[index];c=t.cos(angle/2);s=t.sin(angle/2)
                if gate=='ry':m=t.stack((c,-s,s,c)).reshape(2,2).to(self.dtype)
                elif gate=='rx':m=t.stack((c,-1j*s,-1j*s,c)).reshape(2,2)
                else:m=t.diag(t.stack((t.exp(-.5j*angle),t.exp(.5j*angle))))
                sites[q]=t.einsum('ij,ajb->aib',m,sites[q])
        return sites

    def energy_chunk(self,sites,words,coeffs):
        t=self.torch
        # Prefix nodes share partial contractions within each bounded chunk.
        env=t.ones((1,1,1),dtype=self.dtype,device=self.device)
        previous=[0]*len(words)
        for q,a in enumerate(sites):
            nodes={};parents=[];paulis=[];current=[]
            for prev,word in zip(previous,words):
                key=(prev,word[q])
                if key not in nodes:
                    nodes[key]=len(nodes);parents.append(prev);paulis.append(word[q])
                current.append(nodes[key])
            incoming=env[t.tensor(parents,device=self.device)]
            # E[bra,ket] * A[ket,physical,right].
            tmp=t.einsum('tab,bic->taic',incoming,a)
            # Pauli action, including Y phase, without constructing dense operators.
            idx=t.tensor([[1,0] if p in 'XY' else [0,1] for p in paulis],device=self.device)
            tmp=tmp.gather(2,idx[:,None,:,None].expand(-1,tmp.shape[1],-1,tmp.shape[3]))
            signs=t.tensor([[1,-1] if p=='Z' else [-1j,1j] if p=='Y' else [1,1] for p in paulis],dtype=self.dtype,device=self.device)
            tmp=tmp*signs[:,None,:,None]
            env=t.einsum('aib,taic->tbc',a.conj(),tmp)
            previous=current
        values=env[t.tensor(previous,device=self.device),0,0].real
        return values if coeffs is None else (values*t.as_tensor(coeffs,device=self.device)).sum()

    def value_gradient(self,parameters):
        t=self.torch
        x=t.tensor(np.asarray(parameters),dtype=t.float64,device=self.device,requires_grad=True)
        actual=self.sites(x)
        leaves=[a.detach().requires_grad_(True) for a in actual]
        gradients=[t.zeros_like(a) for a in leaves];value=0.
        # Contract/backprop chunk by chunk: bound tape memory independently of
        # Hamiltonian term count, then reverse through circuit construction once.
        for start in range(0,len(self.words),self.chunk):
            e=self.energy_chunk(leaves,self.words[start:start+self.chunk],self.coeffs[start:start+self.chunk])
            grads=t.autograd.grad(e,leaves)
            value+=float(e.detach().cpu())
            for total,g in zip(gradients,grads):total.add_(g)
        active=[(a,g) for a,g in zip(actual,gradients) if a.requires_grad]
        grad=t.autograd.grad([a for a,g in active],x,[g for a,g in active])[0] if active else t.zeros_like(x)
        return value,grad.detach().cpu().numpy()


def make_problem(kernel,hamiltonian):
    entry=_REGISTRY.get(id(kernel))
    from circuit_tape import Kernel
    spec = entry[1] if entry is not None and entry[0] is kernel else None
    if spec is None and not isinstance(kernel, Kernel): return None
    width = spec['n'] if spec is not None else kernel.width
    words=[];coeffs=[]
    def term(t):
        coeff=complex(t.get_coefficient())
        if abs(coeff.imag)>1e-12:raise ValueError('Reverse mode requires Hermitian real Pauli coefficients')
        word=t.to_string(False).strip().ljust(width,'I')
        if len(word)!=width or any(p not in 'IXYZ' for p in word):raise ValueError('Unsupported Pauli encoding')
        words.append(word);coeffs.append(coeff.real)
    if hasattr(hamiltonian, 'for_each_term'):
        hamiltonian.for_each_term(term)
    else:
        # A one-term local SIEVE Hamiltonian can be a SpinOperatorTerm.
        term(hamiltonian)
    if spec is not None: return ReverseMPS(spec,words,coeffs)
    return recorded_problem(kernel, words, coeffs)


def for_objective(fun,mode):
    if mode=='off':return None
    if mode not in ('auto','hybrid','required'):raise ValueError('Reverse mode must be off, auto, hybrid or required')
    factory = getattr(fun, 'reverse_factory', None)
    if factory is not None: return factory()
    pair=getattr(fun,'reverse_problem',None)
    owner=getattr(fun,'__self__',None)
    # SHARC optimizes its current effective Hamiltonian. Analytic CVaR is
    # supported only at alpha=1, as enforced by this campaign's adapter.
    names = ('CUDAQVQE', 'SHARCVQE', 'CVaRVQE')
    if (pair is None and owner is not None and type(owner).__name__ in names
            and getattr(fun,'__name__','')=='cost_func'):
        if type(owner).__name__ == 'CVaRVQE' and owner.alpha != 1.:
            raise ValueError('Reverse CVaR requires alpha=1')
        pair=(owner.kernel,owner.hamiltonian)
    return make_problem(*pair) if pair is not None else None

class ReverseTapeMPS(ReverseMPS):
    """Exact MPS contraction for shallow recorded tapes, including static gates.

A nonlocal CX uses its rank-two MPO across the intervening sites. The factory
only selects this path when the resulting exact bonds remain bounded.
"""
    engine = 'reverse_tape_mps'
    def __init__(self,kernel,words,coeffs=None,device=None):
        self.kernel=kernel
        super().__init__(dict(n=kernel.width),list(words),np.ones(len(words)) if coeffs is None else coeffs,device)

    def sites(self,x):
        from reverse_tensor import angle_value
        t=self.torch
        constants={name:t.tensor(m,dtype=self.dtype,device=self.device) for name,m in {
            'h':np.array([[1,1],[1,-1]])/np.sqrt(2),'x':[[0,1],[1,0]],
            'y':[[0,-1j],[1j,0]],'z':[[1,0],[0,-1]],'s':[[1,0],[0,1j]],'sdg':[[1,0],[0,-1j]]}.items()}
        sites=[t.tensor([1.,0.],dtype=self.dtype,device=self.device).reshape(1,2,1) for _ in range(self.kernel.width)]
        eye=t.eye(2,dtype=self.dtype,device=self.device)
        for name,qubits,expr in self.kernel.operations:
            if name=='cx':
                control,target=qubits;left,right=sorted(qubits)
                a,b=sites[left],sites[right]
                if control==left:
                    aa=t.einsum('aib,is->aibs',a,eye)
                    bb=t.stack((b,b.flip(1)),dim=1)
                else:
                    aa=t.stack((a,a.flip(1)),dim=-1)
                    bb=t.einsum('aib,is->asib',b,eye)
                sites[left]=aa.reshape(a.shape[0],2,a.shape[2]*2)
                sites[right]=bb.reshape(b.shape[0]*2,2,b.shape[2])
                for q in range(left+1,right):
                    a=sites[q]
                    sites[q]=t.einsum('aib,st->asibt',a,eye).reshape(a.shape[0]*2,2,a.shape[2]*2)
            else:
                q=qubits[0]
                if expr is None:m=constants[name]
                else:
                    angle=angle_value(expr,x);c=t.cos(angle/2);s=t.sin(angle/2)
                    if name=='ry':m=t.stack((c,-s,s,c)).reshape(2,2).to(self.dtype)
                    elif name=='rx':m=t.stack((c,-1j*s,-1j*s,c)).reshape(2,2)
                    else:m=t.diag(t.stack((t.exp(-.5j*angle),t.exp(.5j*angle))))
                sites[q]=t.einsum('ij,ajb->aib',m,sites[q])
        return sites

    def values(self,parameters):
        t=self.torch
        with t.no_grad():
            sites=self.sites(t.as_tensor(np.asarray(parameters),dtype=t.float64,device=self.device))
            chunks=[]
            for i in range(0,len(self.words),self.chunk):
                # energy_chunk optionally returns unweighted moments.
                chunks.append(self.energy_chunk(sites,self.words[i:i+self.chunk],None).cpu().numpy())
            return np.concatenate(chunks) if chunks else np.zeros(0)

    def value_gradient(self,parameters,coefficients=None):
        old=self.coeffs
        if coefficients is not None:self.coeffs=np.asarray(coefficients)
        try:return super().value_gradient(parameters)
        finally:self.coeffs=old

    def moments(self,x):
        from reverse_tensor import TensorCircuit
        return TensorCircuit.moments(self,x)


def recorded_problem(kernel,words,coeffs=None,device=None):
    crossings=[0]*max(0,kernel.width-1)
    for name,qs,expr in kernel.operations:
        if name=='cx':
            for q in range(min(qs),max(qs)):crossings[q]+=1
    if max(crossings,default=0)<=6:
        return ReverseTapeMPS(kernel,words,coeffs,device)
    from reverse_tensor import TensorCircuit
    return TensorCircuit(kernel,words,coeffs,device)
