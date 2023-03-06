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
import os
sys.path.insert(0,'..')

from qiskit.quantum_info.operators.symplectic import PauliList
from qiskit.quantum_info.states.densitymatrix import DensityMatrix
from qiskit.quantum_info import Statevector
from qiskit.circuit import QuantumCircuit, ClassicalRegister
# from .exceptions import VisualizationError

file_logging = False
logging_filename = "./qufi.log"
console_logging = True
circuit_fodler = "./circuits_fast/"
circuit_index_fodler = "./circuits_index_fast/"

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
    state = Statevector(circuit)
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

@qml.qfunc_transform
def pl_insert_gate(tape, index, wire, n_qubits):
    """Decorator qfunc_transform which inserts a single fault gate"""
    for qubit in range(n_qubits):
        qml.U3(theta=0, phi=0, delta=0, wires=qubit, id="RST")

    i = 0
    for gate in tape.operations + tape.measurements:
        qml.apply(gate)
        if i == index:
            qml.U3(theta=0, phi=0, delta=0, wires=wire, id="FAULT")
            break
        i += 1
    # qml.state()

def pl_generate_circuits(base_circuit, name):
    """Generate all possible fault circuits"""
    inj_info = []
    index_info = []
    bloch_info = []
    n_qubits = base_circuit.device.num_wires
    qubits_m = base_circuit.tape.measurements[0].wires
    len_qubits_m = len(qubits_m)
    cbit     = ClassicalRegister(len_qubits_m)
    shots = 1024
    index = 0

    tape = base_circuit.tape
    for op in tape.operations:
        for wire in op.wires:
            transformed_circuit = pl_insert_gate(index, wire, n_qubits)(base_circuit.func)
            device = qml.device('qiskit.aer', wires=n_qubits, shots=shots)
            transformed_qnode = qml.QNode(transformed_circuit, device)
            transformed_qnode()

            q_c = QuantumCircuit.from_qasm_str(str(device._circuit.qasm()))
            q_c.remove_final_measurements()
            bloch = bloch_multivector_data(q_c)
            
            q_c.add_bits(cbit)
            q_c.barrier()
            q_c.measure(qubits_m, range(len_qubits_m))
            q_c.draw(output="mpl", filename=str(circuit_index_fodler+name+"_"+str(index)+"_"+str(wire)+".pdf"))
            
            log(f'Bloch: {bloch[wire]}, wire:{wire}, index: {index}')
            # log(f'Bloch: {bloch}, wire:{wire}, index: {index}')
            print(qml.draw(transformed_qnode)())
            print()
            
            bloch_info.append(bloch[wire])
            inj_info.append(wire)
            index_info.append(index)

        index = index + 1    
    # print(bloch_info)
    # print(inj_info)
    # print(index_info)
    return bloch_info, inj_info, index_info, n_qubits, qubits_m, tape

def execute_over_range_fast(circuits, results_folder="./tmp_qvf/"):
    """Given a range of angles, build all single/double fault injection circuits and run them sequentially"""
    #results = []
    results_names = []
    tstart = datetime.datetime.now()
    log(f"Start: {tstart}")
  
    # read all qvf heatmaps
    folder_path = "tmp_state_vector"  # replace with the path to your folder
    files = os.listdir(folder_path)
    qvf_heatmaps = []
    for file in files:
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            data = pickle.load(gzip.open(file_path, 'r'))[0]
            try:
                A = data['name']
                A = data['bloch_vector']
                A = data['state_vector']
            except:
                print(data['name'], " can't be read\n")
            else:
                # qvf_heatmaps.append((data['bloch_vector'], data['state_vector'], data['qvf']))
                qvf_heatmaps.append([data['bloch_vector'], np.array(data['qvf']).round(3), data['name']])
        
    heatmap_all_0 = np.array([0] * 325 )
    # print(heatmap_all_0)

    # for k in qvf_heatmaps:
    #     print(k[0])

    for circuit in circuits:
        log(f"-"*80+"\n")
        tstartint = datetime.datetime.now()
        log(f"Circuit {circuit[1]} start: {tstartint}")
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

        name = circuit[1]
        log(f"-"*80+"\n"+f"Injecting circuit: {name}")
        # r = pl_insert(deepcopy(target_circuit), circuit[1], theta=angle_pair1[0], phi=angle_pair1[1])
        # pl_inject(r)
        #results.append(r)
        bloch_info, inj_info, index_info, n_qubits, qubits_m, tape = pl_generate_circuits(deepcopy(target_circuit), circuit[1])

        qvf_index = {}        
        qvf     = {}
        qvf     = {"QVF_circuit": np.zeros(325)}
        qvf_occ = {"QVF_circuit": 0}
        for qubit in range(n_qubits):
            qvf["QVF_qubit_{}".format(str(qubit))]     = np.zeros(325)
            qvf_occ["QVF_qubit_{}".format(str(qubit))] = 0
        
        # for i in range(len(bloch_info)):
        #     bloch = bloch_info[i]
        #     flag = False
        #     log(f'Bloch: {bloch}, wire:{inj_info[i]}, index: {index_info[i]}')
        #     for heatmap in qvf_heatmaps:
        #         if heatmap[0][0] == bloch[0] and heatmap[0][1] == bloch[1] and heatmap[0][2] == bloch[2]:
        #             flag = True
        #             break
        #     if flag:
        #         print("MATCH")
        #         print(heatmap[0])
        #     else:
        #         print("NO MATCH")
        #     print()

        for i in range(len(bloch_info)):
            bloch = bloch_info[i]
            log(f'Bloch: {bloch}, wire:{inj_info[i]}, index: {index_info[i]}')
            
            # check if there are harmless errors
            flag = True
            if inj_info[i] not in qubits_m:
                pos = index_info[i]+1
                if len(tape.operations) > pos:
                    print(tape.operations[pos:])
                    for op in tape.operations[pos:]:
                        flag = False
                        # print(f"Check if {inj_info[i]} is in {op.wires}")
                        if inj_info[i] in op.wires:
                            flag = True
                            break
                else:
                    flag = False
            
            if flag:
                flag = False
                for heatmap in qvf_heatmaps:
                    if heatmap[0][0] == bloch[0] and heatmap[0][1] == bloch[1] and heatmap[0][2] == bloch[2]:
                        flag = True
                        break
                        
                if flag:
                    print("MATCH")
                    print(heatmap[0])
                    print(heatmap[2])

                    qvf["QVF_qubit_{}".format(str(inj_info[i]))]     += heatmap[1]
                    qvf_occ["QVF_qubit_{}".format(str(inj_info[i]))] += 1
                    
                    qvf["QVF_circuit"]     += heatmap[1]
                    qvf_occ["QVF_circuit"] += 1

                    qvf_index["QVF_index_{}_qubit_{}".format(str(index_info[i]),str(inj_info[i]))] = deepcopy(heatmap[1])
                else:
                    print("NO MATCH")
            else:
                print("QVF all 0s!")

                qvf["QVF_qubit_{}".format(str(inj_info[i]))]     += heatmap_all_0
                qvf_occ["QVF_qubit_{}".format(str(inj_info[i]))] += 1
                
                qvf["QVF_circuit"]     += heatmap_all_0
                qvf_occ["QVF_circuit"] += 1

                qvf_index["QVF_index_{}_qubit_{}".format(str(index_info[i]),str(inj_info[i]))] = deepcopy(heatmap_all_0)
            print()
        
        # qvf["QVF_circuit"] += 3
        qvf["QVF_circuit"] /= (qvf_occ["QVF_circuit"])
        qvf["QVF_circuit"].round(3)
        for qubit in range(n_qubits):
            # qvf["QVF_qubit_{}".format(str(qubit))] += 1
            qvf["QVF_qubit_{}".format(str(qubit))] /= (qvf_occ["QVF_qubit_{}".format(str(qubit))])
            qvf["QVF_qubit_{}".format(str(qubit))].round(3)

        tmp_name = f"{results_folder}{circuit[1]}.p.gz"
        save_results([qvf, qvf_index, name], tmp_name)
        results_names.append(tmp_name)         

        tendint = datetime.datetime.now()
        log(f"Done: {tendint}\nElapsed time: {tendint-tstartint}\n"+"-"*80+"\n")
    tend = datetime.datetime.now()
    log(f"Done: {tend}\nTotal elapsed time: {tend-tstart}\n")

    # return results
    return results_names

def save_results(results, filename='./results.p.gz'):
    """Save a single/double circuits results object"""
    if not isdir(dirname(filename)):
        mkdir(dirname(filename))
    pickle.dump(results, gzip.open(filename, 'w'))
    log(f"Files saved to {filename}")