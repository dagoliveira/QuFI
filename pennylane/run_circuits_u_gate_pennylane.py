#%%
import numpy as np
from itertools import product
import pickle, gzip
import datetime
# Importing standard Qiskit libraries
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile, Aer, IBMQ, execute
#from qiskit.tools.jupyter import *
#from qiskit.visualization import *

from fault_injector_u_gate_pennylane import inject


fp=open("./run_circuits_u_gate_logging.txt", "a")

#%%
circuits = []

#import Grover
#grove = Grover.build_circuit()
#circuits.append( (grove, 'Grover') )

#import Bernstein_Vazirani
#bv_4 = Bernstein_Vazirani.build_circuit(3, '101')
#circuits.append( (bv_4, 'Bernstein-Vazirani_4') )
#
#bv_5 = Bernstein_Vazirani.build_circuit(4, '1010')
#circuits.append( (bv_5, 'Bernstein-Vazirani_5') )
#
#bv_6 = Bernstein_Vazirani.build_circuit(5, '10101')
#circuits.append( (bv_6, 'Bernstein-Vazirani_6') )
#
#bv_7 = Bernstein_Vazirani.build_circuit(6, '101010')
#circuits.append( (bv_7, 'Bernstein-Vazirani_7') )


#import Deutsch_Jozsa
#dj_4 = Deutsch_Jozsa.build_circuit(3, '101')
#circuits.append( (dj_4, 'Deutsch-Jozsa_4') )
#
#dj_5 = Deutsch_Jozsa.build_circuit(4, '1010')
#circuits.append( (dj_5, 'Deutsch-Jozsa_5') )
#
#dj_6 = Deutsch_Jozsa.build_circuit(5, '10101')
#circuits.append( (dj_6, 'Deutsch-Jozsa_6') )
#
#dj_7 = Deutsch_Jozsa.build_circuit(6, '101010')
#circuits.append( (dj_7, 'Deutsch-Jozsa_7') )

import inverseQFT_pennylane as inverseQFT
qft4 = inverseQFT.build_circuit(4)
circuits.append( (qft4, 'inverseQFT4') )
qft5 = inverseQFT.build_circuit(5)
circuits.append( (qft5, 'inverseQFT5') )
qft6 = inverseQFT.build_circuit(6)
circuits.append( (qft6, 'inverseQFT6') )
qft7 = inverseQFT.build_circuit(7)
circuits.append( (qft7, 'inverseQFT7') )

#%%
import pennylane as qml

#%%

for qiskit_circuit in circuits:
    qregs = len(qiskit_circuit[0].qubits)
    cregs = len(qiskit_circuit[0].clbits)
    pl_circuit = qml.load(qiskit_circuit[0], format='qiskit')

    device = qml.device("default.qubit", wires=qregs)
    @qml.qnode(device)
    def conv_circuit():
        pl_circuit(wires=range(qregs))
        return qml.expval(qml.PauliZ(0))
    
    print(qml.draw(conv_circuit)())
    

#%%
theta_values = np.arange(0, np.pi+0.01, np.pi/12) # 0 <= theta <= pi
phi_values = np.arange(0, 2*np.pi, np.pi/12) # 0 <= phi < 2pi
results = []
for circ in circuits:
    print('-'*80)
    fp.write('-'*80)
    fp.write('\n')
    print('start:',datetime.datetime.now())
    fp.write('start:'+str(datetime.datetime.now()))
    fp.write('\n')
    fp.flush()
    angle_values = product(theta_values, phi_values)
    for angles in angle_values:
        print('-'*80)
        fp.write('-'*80)
        fp.write('\n')
        print('circuit:',circ[1], 'theta:',angles[0], 'phi:',angles[1])
        fp.write('circuit: '+str(circ[1])+ ' theta: '+str(angles[0]) +' phi: '+str(angles[1]))
        fp.write('\n')
        fp.flush()
        r = inject(circ[0], circ[1], theta=angles[0], phi=angles[1])
        results.append(r)
    print('done:',datetime.datetime.now())
    fp.write('done:'+str(datetime.datetime.now()))
    fp.write('\n')
    print('-'*80)
    fp.write('-'*80)
    fp.write('\n')
#%%
filename_output = '../results/u_gate_15degrees_step_qft_4_5_6_7.p.gz'
pickle.dump(results, gzip.open(filename_output, 'w'))
print('files saved to:',filename_output)
fp.write('files saved to:'+str(filename_output))
fp.close()
