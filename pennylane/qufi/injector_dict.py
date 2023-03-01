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
import networkx as nx

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

def get_qiskit_coupling_map(qnode, device_backend):
    """Get structured coupling map compatible with qufi from given qiskit backend"""
    bv4_qasm = qnode.tape.to_openqasm()
    bv4_qiskit = qiskitQC.from_qasm_str(bv4_qasm)
    simulator = AerSimulator.from_backend(device_backend)
    # Transpile the circuit for the noisy basis gates
    # If there are not enough qubits, transpile will throw an error for us
    tcirc = transpile(bv4_qiskit, simulator, optimization_level=3)
    coupling_map = {}
    coupling_map['topology'] = set(tuple(sorted(l)) for l in device_backend.configuration().to_dict()['coupling_map'])
    coupling_map['logical2physical'] = { k._index:v for (k,v) in tcirc._layout.get_virtual_bits().items() if k._register.name == 'q'}
    coupling_map['physical2logical'] = { k:v._index for (k, v) in tcirc._layout.get_physical_bits().items() if v._register.name == 'q'}

    return coupling_map

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

    if len(answer) == 1:
        qvf = 1
        next_percent = 0
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

    # Execute injection circuit simulations without noise
    qvf     = []
    answers = []
    for c, i in zip(generated_circuits, range(0, len(generated_circuits))):
        inj_device = qml.device('lightning.qubit', wires=c.device.num_wires)
        inj_qnode = qml.QNode(c.func, inj_device)
        answer = probs_to_counts(inj_qnode(), base_circuit.device.num_wires)
        qvf.append(QVF_michelson_contrast(answer_gold, answer, 1024))
        answers.append(answer)

    # # Execute injection circuit simulations with noise
    # answers_noise = []
    # for c, i in zip( , range(0, len(generated_circuits))):
    #     inj_device_noisy = qml.device('qiskit.aer', wires=c.device.num_wires, backend='aer_simulator', noise_model=noise_model)
    #     inj_qnode_noisy = qml.QNode(c.func, inj_device_noisy)
    #     answer_noise = probs_to_counts(inj_qnode_noisy(), base_circuit.device.num_wires)
    #     answers_noise.append(answer_noise)

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

# @qml.qfunc_transform
# def pl_insert_df_gate(tape, index, wire, second_theta=0, second_phi=0, lam=0):
#     """Decorator qfunc_transform which inserts a second fault gate after a given index in the QNode.tape"""
#     i = 0
#     inserted = False
#     for gate in tape.operations:
#         # Ignore barriers and measurement gates
#         if i > index and not inserted and wire in gate.wires:
#             # If gate are not using a single qubit, insert one gate after each qubit
#             qml.apply(gate)
#             qml.U3(theta=second_theta, phi=second_phi, delta=lam, wires=wire, id="FAULT")
#             inserted = True
#         else:
#             qml.apply(gate)
#         i = i + 1
#     if not inserted:
#         qml.U3(theta=second_theta, phi=second_phi, delta=lam, wires=wire, id="FAULT")
#     for meas in tape.measurements:
#         qml.apply(meas)

# def pl_generate_circuits(base_circuit, name, theta=0, phi=0, lam=0):
#     """Generate all possible fault circuits"""
#     mycircuits = []
#     inj_info = []
#     index_info = []
#     n_qubits = base_circuit.device.num_wires
#     tape = base_circuit.tape
#     index = 0
#     for op in tape.operations:
#         for wire in op.wires:
#             shots = 1024
#             transformed_circuit = pl_insert_gate(index, wire, theta, phi, lam)(base_circuit.func)
#             device = qml.device('lightning.qubit', wires=n_qubits, shots=shots)
#             transformed_qnode = qml.QNode(transformed_circuit, device)
#             # log(f'Generated single fault circuit: {name} with fault on ({op.name}, wire:{wire}), theta = {theta}, phi = {phi}')
#             # print(index)
#             # print(wire)
#             # print(qml.draw(transformed_qnode)())
#             # print()
#             transformed_qnode()
#             mycircuits.append(transformed_qnode)
#             inj_info.append(wire)
#             index_info.append(index)
#         index = index + 1    
#     # print(index_info)
#     # print(inj_info)
#     log(f"{len(mycircuits)} circuits generated\n")
#     return mycircuits, inj_info, index_info

def pl_insert(circuit, name, angle_combinations):
    """Wrapper for constructing the single fault circuits object"""
    output = {'name': name, 'base_circuit':circuit}
    generated_circuits=[]
    theta_phi_pairs   =[]
    output['pennylane_version'] = qml.version()
    #print(qml.draw(circuit)())
    # generated_circuits, wires, indexes = pl_generate_circuits(circuit, name, angle_combinations)
    index = 0
    tape = circuit.tape
    for op in tape.operations:
        for wire in op.wires:
            for angles in angle_combinations:
                theta = angles[0]
                phi = angles[1]
                lam = 0

                shots = 1024
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
        index = index + 1  
    output['generated_circuits'] = generated_circuits
    output['theta_phi_pairs']    = theta_phi_pairs
    return output

# def pl_insert_df(r, name, theta1, phi1, coupling_map):
#     """Wrapper for expanding a single fault circuits object to a double fault one"""
#     shots = 1024
#     r['theta1'] = theta1
#     r['phi1'] = phi1
#     r['second_wires'] = []
#     double_fault_circuits = []
    
#     for qnode, wire, index in zip(r['generated_circuits'], r['wires'], r['indexes']):
#         with qnode.tape as tape:
#             for gate in tape.operations:
#                 if gate.id == 'FAULT':
#                     for logical_qubit in tape.wires:
#                         physical_qubit = coupling_map['logical2physical'][logical_qubit]
#                         neighbouring_qubits = [el[1] for el in coupling_map['topology'] if el[0]==physical_qubit]
#                         # Don't loop over all logical qubits, just over the ones connected! Kinda
#                         for neighbor in neighbouring_qubits:
#                             if neighbor not in coupling_map['physical2logical'].keys() or coupling_map['physical2logical'][neighbor] not in range(len(tape.wires)):
#                                 continue
#                             else:
#                                 second_fault_wire = coupling_map['physical2logical'][neighbor]
#                                 if second_fault_wire in gate.wires:
#                                     continue
#                                 double_fault_circuit = pl_insert_df_gate(index, second_fault_wire, theta1, phi1)(deepcopy(qnode).func)
#                                 double_fault_device = qml.device('lightning.qubit', wires=len(tape.wires), shots=shots)
#                                 double_fault_qnode = qml.QNode(double_fault_circuit, double_fault_device)
#                                 double_fault_qnode()
#                                 log(f'Generated double fault circuit: {name} with faults on (wire1:{wire}, theta0:{gate.parameters[0]:.2f}, phi0:{gate.parameters[1]:.2f}) and (wire2:{second_fault_wire}, theta1:{theta1:.2f}, phi1:{phi1:.2f})')
#                                 #print(qml.draw(double_fault_qnode)())
#                                 double_fault_circuits.append(double_fault_qnode)
#                                 r['second_wires'].append(second_fault_wire)
#                                 r['wires'].append(wire)
#     log(f"{len(double_fault_circuits)} double fault circuits generated\n")
#     r['generated_circuits'] = double_fault_circuits
#     return r

def pl_inject(circuitStruct):
    """Run a single/double fault circuits object"""
    circuitStruct.update(run_circuits( circuitStruct['base_circuit'], circuitStruct['generated_circuits'], circuitStruct['theta_phi_pairs'] ) )

def execute_over_range_dict(circuits,
            angles={'theta':np.arange(0, np.pi+0.01, np.pi/12), 
                    'phi':np.arange(0, 2*np.pi+0.01, np.pi/12)}, 
            coupling_map=None,
            results_folder="./tmp/"):
    """Given a range of angles, build all single/double fault injection circuits and run them sequentially"""
    #results = []
    results_names = []
    tstart = datetime.datetime.now()
    log(f"Start: {tstart}")
    for circuit in circuits:
        circuit_name = circuit[2]
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
        r = pl_insert(deepcopy(target_circuit), circuit_name, angle_combinations)
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

# def execute(circuits,
#             angles=None, 
#             coupling_map=None,
#             results_folder="./tmp/"):
#     """Given a list of angle combinations, split them in batches, then compute all circuits and run them sequentially"""
#     results_folder = "./tmp/"
#     results_names = []
#     tstart = datetime.datetime.now()
#     log(f"Start: {tstart}")
#     for circuit in circuits:
#         log(f"-"*80+"\n")
#         tstartint = datetime.datetime.now()
#         log(f"Circuit {circuit[1]} start: {tstartint}")
#         if isinstance(circuit[0], qml.QNode):
#             target_circuit = circuit[0]
#         elif isinstance(circuit[0], qiskitQC):
#             target_circuit = convert_qiskit_circuit(circuit)
#         elif isinstance(circuit[0], str) and circuit[0].startswith("OPENQASM"):
#             target_circuit = convert_qasm_circuit(circuit)
#         else:
#             log(f"Unsupported {type(circuit[0])} object, injection stopped.")
#             exit()
        
#         for angles_batch, batch in zip(np.array_split(angles, ceil(len(angles)/500)), range(1, ceil(len(angles)/500) + 1) ):
#             results = []
#             for angles_combination, iteration in zip(angles_batch, range(1, len(angles_batch))):
#                 log(f"Executing iteration:{iteration}/{len(angles_batch)} batch:{batch}/{ceil(len(angles)/500)}")
#                 log(f"-"*80+"\n"+f"Injecting circuit: {circuit[1]} theta0: {angles_combination[0]} phi0: {angles_combination[1]}")
#                 r = pl_insert(deepcopy(target_circuit), circuit[1], theta=angles_combination[0], phi=angles_combination[1])
#                 if coupling_map != None:
#                     s = pl_insert_df(deepcopy(r), circuit[1], angles_combination[2], angles_combination[3], coupling_map)
#                     pl_inject(s)
#                     results.append(s)
#                 else:
#                     pl_inject(r)
#                     results.append(r)
#             tmp_name = f"{results_folder}{circuit[1]}_{angles_combination[0]}_{angles_combination[1]}_{angles_combination[2]}_{angles_combination[3]}.p.gz"
#             results_names.append(tmp_name)
#             save_results(results, tmp_name)
#         tendint = datetime.datetime.now()
#         log(f"Done: {tendint}\nElapsed time: {tendint-tstartint}\n"+"-"*80+"\n")
#     tend = datetime.datetime.now()
#     log(f"Done: {tend}\nTotal elapsed time: {tend-tstart}\n")

#     return results_names


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