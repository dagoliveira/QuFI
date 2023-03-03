from copy import deepcopy
from sys import exit
import numpy as np
from itertools import product
import pickle, gzip
import datetime
from math import ceil
from os.path import isdir, dirname
from os import mkdir
# Importing standard Qiskit libraries
from qiskit.circuit.quantumcircuit import QuantumCircuit as qiskitQC
from qiskit.test.mock import FakeSantiago
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer import AerSimulator
from qiskit import transpile
import pennylane as qml

import sys
sys.path.insert(0,'..')

from qiskit.quantum_info.operators.symplectic import PauliList
from qiskit.quantum_info.states.densitymatrix import DensityMatrix
from qiskit.quantum_info import Statevector
from qiskit.circuit import QuantumCircuit

file_logging = False
logging_filename = "./qufi.log"
console_logging = True

def log(content):
    """Logging wrapper, can redirect both to stdout and a file"""
    if file_logging:
        fp = open(logging_filename, "a")
        fp.write(content+'\n')
        fp.flush()
        fp.close()
    if console_logging:
        print(content)

def bloch_multivector_data(circuit):
    """Return list of Bloch vectors for each qubit

    Args:
        circuit (Pennylane qml): an N-qubit circuit.

    Returns:
        list: list of Bloch vectors (x, y, z) for each qubit.

    Raises:
        VisualizationError: if input is not an N-qubit state.
    """
    q_circuit = QuantumCircuit.from_qasm_str(circuit.qtape.to_openqasm())
    q_circuit.remove_final_measurements()
    state = Statevector(q_circuit)
    rho = DensityMatrix(state)
    num = rho.num_qubits
    # if num is None:
    #     raise VisualizationError("Input is not a multi-qubit quantum state.")
    pauli_singles = PauliList(["X", "Y", "Z"])
    bloch_data = []
    for i in range(num):
        if num > 1:
            paulis = PauliList.from_symplectic(
                np.zeros((3, (num - 1)), dtype=bool), np.zeros((3, (num - 1)), dtype=bool)
            ).insert(i, pauli_singles, qubit=True)
        else:
            paulis = pauli_singles
        bloch_state = [np.real(np.round(np.trace(np.dot(mat, rho.data)),4)) for mat in paulis.matrix_iter()]
        # bloch_state = [np.real(np.trace(np.dot(mat, rho.data))) for mat in paulis.matrix_iter()]
        bloch_data.append(bloch_state)
    return bloch_data

def probs_to_counts(probs, nwires):
    """Utility to convert pennylane result probabilities to qiskit counts"""
    res_dict = {}
    shots = 1024
    for p, t in zip(probs, list(product(['0', '1'], repeat=nwires))):
        b = ''.join(t)
        count = int(ceil(shots*float(p)))
        if count != 0:
            res_dict[b] = count
    # Debug check for ceil rounding (Still bugged somehow, sometimes off by 1-2 shots)
    #if sum(res_dict.values()) != shots:
    #    log(f"Rounding error! {sum(res_dict.values())} != {shots}")
    return res_dict

def QVF_michelson_contrast(answer_gold, answer, shots):
    """Compute Michelson contrast between gold and highest percentage fault string"""
    # Sort the answer, position 0 has the highest bitstring, position 1 the second highest
    answer_sorted  = sorted(answer, key=answer.get, reverse=True)
    gold_bitstring = sorted(answer_gold, key=answer_gold.get, reverse=True)[0]

    # print(gold_bitstring)
    # print(answer)
    # print(answer_sorted)

    # If gold bitstring is not in answer, percentage is zero
    if gold_bitstring not in answer:
        good_percent = 0
    else:
        good_percent = answer[gold_bitstring]/shots

    if len(answer_sorted) == 1:
        if answer_sorted[0] == gold_bitstring:
            qvf = 1
        else:
            qvf = -1
    else:
        if answer_sorted[0] == gold_bitstring: # gold bitstring has the highest count (max)
            # next bitstring is the second highest
            next_percent = answer[answer_sorted[1]]/shots
        else: # gold bitstring has NOT the highest count (not max)
            next_percent = answer[answer_sorted[0]]/shots
        qvf = (good_percent - next_percent) / (good_percent + next_percent)
    return 1 - (qvf+1)/2

def run_circuits(base_circuit, generated_circuits, theta_phi_pairs, device_backend=FakeSantiago()):
    """Internally called function which runs the circuits for all golden/faulty noiseless/noisy combinations"""
    # Execute golden circuit simulation without noise
    log('Running circuits')
    gold_device = qml.device('lightning.qubit', wires=base_circuit.device.num_wires)
    gold_qnode = qml.QNode(base_circuit.func, gold_device)
    answer_gold = probs_to_counts(gold_qnode(), base_circuit.device.num_wires)

    # print("Answer gold: {}".format(answer_gold))

    # Execute injection circuit simulations without noise
    qvf     = []
    answers = []
    for c in generated_circuits:
        inj_device = qml.device('lightning.qubit', wires=c.device.num_wires)
        inj_qnode = qml.QNode(c.func, inj_device)
        answer = probs_to_counts(inj_qnode(), base_circuit.device.num_wires)
        qvf.append(QVF_michelson_contrast(answer_gold, answer, 1024))
        answers.append(answer)

    return {'output_gold':answer_gold
            , 'qvf':qvf
            , 'answers':answers
            , 'noise_target':str(device_backend)
            }

def convert_qiskit_circuit(qiskit_circuit):
    """Converts a qiskit QuantumCircuit object to a pennylane QNode object"""
    shots = 1024
    measure_list = [g[1][0]._index for g in qiskit_circuit[0].data if g[0].name == 'measure']
    qregs = qiskit_circuit[0].num_qubits
    # Avoid qml.load warning on trying to convert measure operators
    qiskit_circuit[0].remove_final_measurements()
    pl_circuit = qml.load(qiskit_circuit[0], format='qiskit')
    device = qml.device("lightning.qubit", wires=qregs, shots=shots)
    @qml.qnode(device)
    def conv_circuit():
        pl_circuit(wires=range(qregs))
        return qml.probs(wires=measure_list) #[qml.expval(qml.PauliZ(i)) for i in range(qregs)]
    # Do NOT remove this evaluation, else the qnode can't bind the function before exiting convert_qiskit_circuit()'s context
    conv_circuit()
    #print(qml.draw(conv_circuit)())
    return conv_circuit

def convert_qasm_circuit(qasm_circuit):
    """Converts a QASM string to a pennylane QNode object"""
    qiskit_circuit = qiskitQC.from_qasm_str(qasm_circuit[0])
    qnode = convert_qiskit_circuit((qiskit_circuit, qasm_circuit[1]))
    return qnode

# TO DO: improve this method
@qml.qfunc_transform
def pl_insert_gate(tape, index, wire, theta=0, phi=0, lam=0):
    """Decorator qfunc_transform which inserts a single fault gate"""
    i = 0
    for gate in tape.operations + tape.measurements:
        # Ignore barriers and measurement gates
        if i == index:
            # If gate are not using a single qubit, insert one gate after each qubit
            qml.apply(gate)
            qml.U3(theta=theta, phi=phi, delta=lam, wires=wire, id="FAULT")
        else:
            qml.apply(gate)
        i = i + 1

def pl_insert(circuit, name, angle_combinations, state_circuit):
    """Wrapper for constructing the single fault circuits object"""
    output = {'name': name, 'base_circuit':circuit}
    output['pennylane_version'] = qml.version()
    output['bloch_vector'] = bloch_multivector_data(state_circuit)[0]
    generated_circuits = []
    theta_phi_pairs    = []
    # print(qml.draw(circuit)())
    # print()
    # generated_circuits, wires, indexes = pl_generate_circuits(circuit, name, angle_combinations)
    tape = circuit.tape
    for op in tape.operations:
        wire = op.wires[0]
        for angles in angle_combinations:
            theta = angles[0]
            phi = angles[1]
            lam = 0

            shots = 1024
            index = 0
            transformed_circuit = pl_insert_gate(index, wire, theta, phi, lam)(circuit.func)
            device = qml.device('lightning.qubit', wires=1, shots=shots)
            transformed_qnode = qml.QNode(transformed_circuit, device)
            # log(f'Generated single fault circuit: {name} with fault on ({op.name}, wire:{wire}), theta = {theta}, phi = {phi}')
            # print(index)
            # print(wire)
            # print(qml.draw(transformed_qnode)())
            # print()

            transformed_qnode()
            generated_circuits.append(transformed_qnode)
            theta_phi_pairs.append((theta, phi))
    output['generated_circuits'] = generated_circuits
    output['theta_phi_pairs']    = theta_phi_pairs
    return output

def pl_inject(circuitStruct):
    """Run a single/double fault circuits object"""
    circuitStruct.update(run_circuits( circuitStruct['base_circuit'], circuitStruct['generated_circuits'], circuitStruct['theta_phi_pairs'] ) )

def execute_over_range_dict(circuits,
            angles={'theta':np.arange(0, np.pi+0.01, np.pi/12), 
                    'phi':np.arange(0, 2*np.pi+0.01, np.pi/12)}, 
            results_folder="./tmp/"):
    """Given a range of angles, build all single/double fault injection circuits and run them sequentially"""
    #results = []
    results_names = []
    tstart = datetime.datetime.now()
    log(f"Start: {tstart}")
    for circuit in circuits:
        circuit_name = circuit[3]
        log(f"-"*80+"\n")
        tstartint = datetime.datetime.now()
        log(f"Circuit {circuit_name} start: {tstartint}")
        angle_combinations = product(angles['theta'], angles['phi'])
        # for angle_pair1 in angle_combinations:
        # Converting the circuit only once at the start of outer loop causes reference bugs (insight needed)
        if isinstance(circuit[0], qml.QNode):
            target_circuit = circuit[0]
        elif isinstance(circuit[0], qiskitQC):
            target_circuit = convert_qiskit_circuit(circuit)
        elif isinstance(circuit[0], str) and circuit[0].startswith("OPENQASM"):
            target_circuit = convert_qasm_circuit(circuit)
        else:
            log(f"Unsupported {type(circuit[0])} object, injection stopped.")
            exit()

        # transpiled_circuit = qml.transforms.transpile(coupling_map=coupling_map)(target_circuit)
        # device = qml.device('lightning.qubit', wires=n_qubits, shots=shots)
        # transformed_qnode = qml.QNode(transpiled_circuit, device)
        log(f"-"*80+"\n"+f"Injecting circuit: {circuit_name}")
        r = pl_insert(deepcopy(target_circuit), circuit_name, angle_combinations, circuit[1])
        r['state_vector'] = circuit[2]
        pl_inject(r)

        tmp_name = f"{results_folder}{circuit_name}.gz"
        save_results([r], tmp_name)
        results_names.append(tmp_name)  
        tendint = datetime.datetime.now()
        log(f"Done: {tendint}\nElapsed time: {tendint-tstartint}\n"+"-"*80+"\n")
    tend = datetime.datetime.now()
    log(f"Done: {tend}\nTotal elapsed time: {tend-tstart}\n")

    # return results
    return results_names

def save_results(results, filename='./results.p.gz'):
    """Save a single/double circuits results object"""
    # Temporary fix for pickle.dump
    for circuit in results:
        del circuit['base_circuit']
        del circuit['generated_circuits']
    if not isdir(dirname(filename)):
        mkdir(dirname(filename))
    pickle.dump(results, gzip.open(filename, 'w'))
    log(f"Files saved to {filename}")