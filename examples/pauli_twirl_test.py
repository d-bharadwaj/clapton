import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.append(os.path.abspath(os.path.join(script_dir, "../")))

from clapton.clapton import claptonize
from clapton.ansatzes import *
import numpy as np

# define Hamiltonian, e.g. 3q Heisenberg model with random coefficients
paulis = ["XXI", "IXX", "YYI", "IYY", "ZZI", "IZZ"]
coeffs = np.random.random(len(paulis))

from clapton.depolarization import GateGeneralDepolarizationModel #TODO: check this out 

# let's add a noise model where we specify global 1q and 2q gate errors
nm = GateGeneralDepolarizationModel(p1=0.005, p2=0.02)
rng = np.random.default_rng()

TWIRL_GATES_CX = (
        (('I', 'I'), ('I', 'I')),
        (('I', 'X'), ('I', 'X')),
        (('I', 'Y'), ('Z', 'Y')),
        (('I', 'Z'), ('Z', 'Z')),
        (('X', 'I'), ('X', 'X')),
        (('X', 'X'), ('X', 'I')),
        (('X', 'Y'), ('Y', 'Z')),
        (('X', 'Z'), ('Y', 'Y')),
        (('Y', 'I'), ('Y', 'X')),
        (('Y', 'X'), ('Y', 'I')),
        (('Y', 'Y'), ('X', 'Z')),
        (('Y', 'Z'), ('X', 'Y')),
        (('Z', 'I'), ('Z', 'I')),
        (('Z', 'X'), ('Z', 'X')),
        (('Z', 'Y'), ('I', 'Y')),
        (('Z', 'Z'), ('I', 'Z')),
    )

pauli_twirl_dict = {"I":0,"X":1,"Y":2,"Z":3}

def twirled_circular_ansatz(N, reps=1, fix_2q=False):
    pcirc = ParametrizedCliffordCircuit()
    for _ in range(reps):
        for i in range(N):
            pcirc.RY(i)
        for i in range(N):
            pcirc.RZ(i)
        for i in range(N):
            control = (i-1) % N
            target = i
            if fix_2q:
                (before0, before1), (after0, after1) = TWIRL_GATES_CX[
                    rng.integers(len(TWIRL_GATES_CX))]

                pcirc.PauliTwirl(control).fix(pauli_twirl_dict[before0])
                pcirc.PauliTwirl(control).fix(pauli_twirl_dict[before1])
                pcirc.Q2(control, target).fix(1)
                pcirc.PauliTwirl(target).fix(pauli_twirl_dict[after0])
                pcirc.PauliTwirl(target).fix(pauli_twirl_dict[after1])
            else:   
                pcirc.Q2(control, target)
    for i in range(N):
        pcirc.RY(i)
    for i in range(N):
        pcirc.RZ(i)
    return pcirc

pauli_twirl_list = [twirled_circular_ansatz(N=len(paulis[0]), reps=2, fix_2q=True) for _ in range(100)]

vqe_pcirc = circular_ansatz(N=len(paulis[0]), reps=2, fix_2q=True)
vqe_pcirc.add_depolarization_model(nm)
vqe_pcirc.add_pauli_twirl_list(pauli_twirl_list)


# we can perform nCAFQA by using the main optimization function "claptonize"
# now with the noisy circuit
# this is slower, as the noisy circuit needs to be sampled 
ks_best, energy_noisy, energy_noiseless = claptonize(
    paulis,
    coeffs,
    vqe_pcirc,
    n_proc=4,           # total number of processes in parallel
    n_starts=4,         # number of random genetic algorithm starts in parallel
    n_rounds=1,         # number of budget rounds, if None it will terminate itself
    callback=print,     # callback for internal parameter (#iteration, energies, ks) processing
    budget=20,          # budget per genetic algorithm instance
)

# differrence
print(f"Difference between Noisy and Noiseless calculation : {np.abs(energy_noisy-energy_noiseless)}")
