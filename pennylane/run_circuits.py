#%%
from qiskit.test.mock import FakeSantiago
import pennylane as qml
import qiskit
from qiskit import Aer
from qiskit.tools.visualization import plot_histogram
from qufi import execute_over_range, Google_3, BernsteinVazirani, Try_0, Try_1, Try_2,  read_results_directory, generate_all_statistics

# angles = {'theta':[1, 2], 'phi':[2, 3]}
circuits = []

# c = Google_3.build_circuit()
# circuits.append((c, 'Google_3'))

# bv4_p = BernsteinVazirani.build_circuit()
# circuits.append((bv4_p, 'BernsteinVazirani_4'))

# try_0 = Try_0.build_circuit()
# circuits.append((try_0, 'try_0'))

# try_1 = Try_1.build_circuit()
# circuits.append((try_1, 'try_1'))

try_2 = Try_2.build_circuit()
circuits.append((try_2, 'try_2'))

#%%
simulator = Aer.get_backend('qasm_simulator')

circuit_fodler = "circuits/"
for c in circuits:
    q_c = qiskit.circuit.QuantumCircuit.from_qasm_str(c[0].qtape.to_openqasm(rotations=False, measure_all=False))

    # Run and get counts
    result = simulator.run(q_c).result()
    counts = result.get_counts(q_c)
    plot_histogram(counts, title='Bell-State counts', filename=str(circuit_fodler+c[1]+"_plot.png"))

    q_c.draw(output="mpl", filename=str(circuit_fodler+c[1]))

#%%
results_names = execute_over_range(circuits, results_folder="./tmp_try/")

# %%
from qufi import read_results_directory, generate_all_statistics
results = read_results_directory("./tmp_try/", noise=True)

generate_all_statistics(results, savepath="./plots_try")
# %%


