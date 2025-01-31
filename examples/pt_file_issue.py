# Script used to file issue on qiskit github.

import qiskit
import numpy as np
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
import qiskit.quantum_info as qi
from qiskit_ibm_runtime.fake_provider import FakeMumbaiV2
# Import from Qiskit Aer noise module
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (NoiseModel, coherent_unitary_error)

def test_circuit(n):
    qc = qiskit.QuantumCircuit(n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
        if i + 2 < n:
            qc.cz(i + 1, i + 2)
    return qc

num_qubits = 5
np.random.seed(0)
paulis = ["".join(np.random.choice(['I', 'X', 'Y', 'Z'], size=num_qubits)) for _ in range(num_qubits+1)]
qc = test_circuit(num_qubits)

np.random.seed(0)
coeffs = np.random.random(len(paulis))
observable = SparsePauliOp.from_list(list(zip(paulis, coeffs)))

#Aer Backend
backend = AerSimulator(method='statevector', device='GPU',seed_simulator = 0)


#Ideal Simulation 
ideal_estimator = Estimator(mode=backend)
ideal_estimator.options.seed_estimator=0
pm = generate_preset_pass_manager(backend=backend, optimization_level=1,seed_transpiler=0)
isa_circuit = pm.run(qc)
isa_observable = observable.apply_layout(isa_circuit.layout)
job = ideal_estimator.run([(isa_circuit, isa_observable)])
# Get results for the first (and only) PUB
ideal_res = job.result()[0].data.evs

#Noisy Simulation
noise_model = NoiseModel()
##Coherent Noise
epsilon = 0.1
err_cx = qiskit.QuantumCircuit(2)
err_cx.cx(0,1)
err_cx.p(epsilon, 0)
err_cx.p(epsilon, 1)
err_cx.cx(0,1)
err_cx.p(-epsilon, 0)
err_cx.p(-epsilon, 1)
err_cx = qi.Operator(err_cx)
noise_model.add_all_qubit_quantum_error(
    coherent_unitary_error(err_cx), ["cx"])

noisy_backend =  FakeMumbaiV2()
noisy_estimator = Estimator(mode=noisy_backend)
(noisy_estimator.options.simulator.noise_model) = noise_model

pm = generate_preset_pass_manager(backend=noisy_backend, optimization_level=1)
isa_circuit = pm.run(qc)
isa_observable = observable.apply_layout(isa_circuit.layout)
job = noisy_estimator.run([(isa_circuit, isa_observable)])
noisy_res = job.result()[0].data.evs

## PT Mitigation
pt_estimator = Estimator(mode=noisy_backend)
pt_estimator.options.twirling.enable_gates = True
pt_estimator.options.twirling.num_randomizations = 100
pt_estimator.options.twirling.shots_per_randomization = 1000
pt_estimator.options.simulator.noise_model = noise_model

pm = generate_preset_pass_manager(backend=noisy_backend, optimization_level=1)
isa_circuit  = pm.run(qc)
isa_observable = observable.apply_layout(isa_circuit.layout)
job = pt_estimator.run([(isa_circuit, isa_observable)])
pt_res = job.result()[0].data.evs

improvement_factor = (abs(noisy_res - ideal_res)) / (abs(pt_res - ideal_res))
print(f"Factor of improvement: {improvement_factor}")