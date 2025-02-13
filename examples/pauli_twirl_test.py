# This script compares the results between using 'nCAFQA+VQE' vs 'nCAFQA(w Pauli Twirling)+VQE(w Pauli Twirling)'.

import sys
import os
import numpy as np
import stim

from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit import QuantumCircuit, QuantumRegister, Gate,ParameterVector
from qiskit.circuit.library import CXGate, ECRGate
from qiskit.transpiler import PassManager
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.quantum_info import Operator, pauli_basis , SparsePauliOp
from qiskit_ibm_runtime.fake_provider import FakeMumbaiV2
from qiskit_ibm_runtime import EstimatorV2 as Estimator, QiskitRuntimeService
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, QuantumError, coherent_unitary_error
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit.converters import circuit_to_dag, dag_to_circuit

import numpy as np
from typing import Iterable, Optional
from scipy.optimize import minimize

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.append(os.path.abspath(os.path.join(script_dir, "../")))

from clapton.clapton import claptonize
from clapton.ansatzes import circular_ansatz,qiskit_circular_ansatz
from clapton.depolarization import GateGeneralDepolarizationModel
from clapton.hamiltonians import ising_model
from clapton.PT_PM import PauliTwirl 

np.random.seed(0)



num_qubits = 10
reps = 1 
coeffs,paulis,_ = ising_model(N=num_qubits,Jx=0.2,Jy=0.3,h=0.4)
qc = qiskit_circular_ansatz(num_qubits,reps)


def qiskit_params_map(circ):
    dag = circuit_to_dag(circ)
    param_list = [list(node.op.params[0].parameters)[0].name for node in dag.op_nodes() if node.op.params and isinstance(node.op.params[0], ParameterExpression)]
    return {k: v for v, k in enumerate(param_list)}

def circuit_to_tableau(circuit: stim.Circuit) -> stim.Tableau:
    s = stim.TableauSimulator()
    s.do_circuit(circuit)
    return s.current_inverse_tableau() ** -1

qiskit_param_map = qiskit_params_map(qc)

# Ensure the sorted names are correct
ordered_params = [param.name for param in qc.parameters]
assert sorted(qiskit_param_map.keys()) == ordered_params

param_map = {qiskit_param_map[param]: i for i, param in enumerate(ordered_params)}

nm = GateGeneralDepolarizationModel(p1=0.005, p2=0.05)
pauli_twirl = True

def initialize_circuit_and_claptonize(pauli_twirl: bool, num_qubits: int, reps: int, nm, paulis, coeffs):
    if pauli_twirl:
        init_ansatz = circular_ansatz(N=num_qubits, reps=reps, fix_2q=True)
        # Pauli Twirl the circuit
        vqe_pcirc = init_ansatz
        pauli_twirl_list = [vqe_pcirc.add_pauli_twirl() for _ in range(100)]
        
        # Ensure Twirled Circuits are logically equal to Original Ansatz
        for i, circuit in enumerate(pauli_twirl_list):
            assert circuit_to_tableau(vqe_pcirc.stim_circuit()) == circuit_to_tableau(circuit.stim_circuit()), \
                f"Circuit Mismatch at index {i}"

        vqe_pcirc.add_depolarization_model(nm)
        pauli_twirl_list = [circuit.add_depolarization_model(nm) for circuit in pauli_twirl_list]
        vqe_pcirc.add_pauli_twirl_list(pauli_twirl_list)  # NOTE: Made major change here by adding list after adding noise

    else:
        vqe_pcirc = circular_ansatz(N=num_qubits, reps=1, fix_2q=True)
        vqe_pcirc.add_depolarization_model(nm)

    vqe_pcirc.define_parameter_map(param_map)

    # Perform nCAFQA using the main optimization function "claptonize" with the noisy circuit
    ks_best, energy_noisy, energy_noiseless = claptonize(
        paulis,
        coeffs,
        vqe_pcirc,
        n_proc=4,           # Total number of processes in parallel
        n_starts=4,         # Number of random genetic algorithm starts in parallel
        n_rounds=1,         # Number of budget rounds, if None it will terminate itself
        callback=print,     # Callback for internal parameter (#iteration, energies, ks) processing
        budget=20,          # Budget per genetic algorithm instance
    )

    #Initializing params from CAFQA 
    initial_params = [ (param * np.pi/2) for param in vqe_pcirc.internal_read()]

    return initial_params, energy_noisy, energy_noiseless

pt_cafqa = True
ks_best, energy_noisy, energy_noiseless = initialize_circuit_and_claptonize(pt_cafqa, num_qubits, reps, nm, paulis, coeffs)

# Print the energies
print(f"Energy Noisy: {energy_noisy}")
print(f"Energy Noiseless: {energy_noiseless}")

# Difference
print(f"Difference between Noisy and Noiseless calculation: {np.abs(energy_noisy - energy_noiseless)}")

# VQE 
#Initializing params from CAFQA 

initial_params = ks_best

# Defining Hamiltoninan for VQE
weights  =  coeffs
pauli_op = [([pauli,weight]) for pauli,weight in zip(paulis,weights)]
hamiltonian = SparsePauliOp.from_list([ op for op in pauli_op ])

# Calculate Exact Ground State Energy
numpy_solver = NumPyMinimumEigensolver()
ref_result = numpy_solver.compute_minimum_eigenvalue(operator=hamiltonian)
ref_value = ref_result.eigenvalue.real

# Create your custom pass
pm = PassManager([PauliTwirl()])

#Create your twirled circuits
num_twirled = 100
twirled_qcs = [pm.run(qc) for _ in range(num_twirled)]
# Device Noise Model
backend = FakeMumbaiV2() # Your quantum backend

# noise_model = NoiseModel.from_backend(backend) 

# Only Coherent Noise Model 
noise_model = NoiseModel()

# Define the over-rotation angle (in radians)
theta = 0.1  # Adjust this value to control the amount of over-rotation

# Create the over-rotated CX gate
cx_overrotated = Operator([
    [np.cos(theta/2), -1j*np.sin(theta/2), 0, 0],
    [-1j*np.sin(theta/2), np.cos(theta/2), 0, 0],
    [0, 0, np.cos(theta/2), -1j*np.sin(theta/2)],
    [0, 0, -1j*np.sin(theta/2), np.cos(theta/2)]
])

# Create a quantum error from the over-rotated CX gate
noise_model.add_all_qubit_quantum_error(coherent_unitary_error(cx_overrotated), ['cx'])

pt_vqe_energies=[]
iteration_number=0
def pt_vqe_cost_function(params, ansatz, hamiltonian, pass_manager, estimator,twirled_qcs):
    global iteration_number 

    # Calculate expectation value
    energy_values = np.array([])
    for transpiled_circuit in twirled_qcs:
        transpiled_circuit = transpiled_circuit.assign_parameters(params)
        job = estimator.run([(transpiled_circuit, hamiltonian)]) #Something wrong w parameter optim.
        result = job.result()
        energy = result[0].data.evs #NOTE: Is this actually expectation value?
        energy_values = np.append(energy_values, energy)
    
    iteration_number += 1
    # Print statement to see where we are at
    print(f"Iteration: {iteration_number}, Current energy: {np.average(energy_values)}, Current parameters: {params}")
    # return result.values[0]
    pt_vqe_energies.append(np.average(energy_values))
    return np.average(energy_values)

## Noisy Estimator
estimator = Estimator(
    mode=AerSimulator(
        noise_model=noise_model,
        coupling_map=backend.coupling_map,
        basis_gates=noise_model.basis_gates,
        device = 'GPU'
    )
)
result = minimize(
    pt_vqe_cost_function,
    initial_params,
    args=(qc, hamiltonian, pm, estimator,twirled_qcs),
    method='COBYLA',
    options={'maxiter': 100}
)

pt_optimal_params = result.x
pt_optimal_value = result.fun

print(f"Optimal value: {pt_optimal_value}")
print(f"Optimal parameters: {pt_optimal_params}")


# Normal Method
non_pt_cafqa = False
ks_best, energy_noisy, energy_noiseless = initialize_circuit_and_claptonize(non_pt_cafqa, num_qubits, reps, nm, paulis, coeffs)
initial_params = [ (param * np.pi/2) for param in ks_best]

iteration_number= 0

vqe_energies = []
def vqe_cost_function(params, ansatz, hamiltonian, pass_manager, estimator):
    global iteration_number 
    # Bind parameters to the ansatz
    bound_circuit = ansatz.assign_parameters(params)
    
    # Calculate expectation value
    job = estimator.run([(bound_circuit, hamiltonian)]) #Something wrong w parameter optim.
    result = job.result()
    energy = result[0].data.evs 
    vqe_energies.append(energy)
    
    iteration_number += 1
    # Print statement to see where we are at
    print(f"Iteration: {iteration_number}, Current energy: {energy}, Current parameters: {params}")
    return energy

result = minimize(
    vqe_cost_function,
    initial_params,
    args=(qc, hamiltonian, pm, estimator),
    method='COBYLA',
    options={'maxiter': 100}
)

optimal_params = result.x
optimal_value = result.fun

print(f"Optimal value without PT : {optimal_value}")
print(f"Optimal parameters: {optimal_params}")

import matplotlib.pyplot as plt

plt.plot(vqe_energies, label='VQE Energy')
plt.plot(pt_vqe_energies, label='Twirled CAFQA + PT_VQE Energy')
plt.axhline(y=ref_value, color='r', linestyle='--', label=f'Ground State Energy: {ref_value:.5f}')
plt.xlabel('Iteration')
plt.ylabel('VQE Energy')
plt.title(f'VQE Energy vs Iteration ; {num_qubits} qubits')
plt.legend()
plt.show()

# Save the plot
plt.savefig('plots/test.png')
