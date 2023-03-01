#%%
from qiskit.test.mock import FakeSantiago
import pennylane as qml
import qiskit
from qiskit import Aer
from qiskit.tools.visualization import plot_histogram
from qufi import execute_over_range, Google_3, Google_4, Google_2, get_qiskit_coupling_map, read_results_directory, generate_all_statistics

angles = {'theta':[1, 2], 'phi':[2, 3]}
circuits = []

#%%
google = Google_3.build_circuit()
circuits.append((google, 'Google_3'))

bv4_p = Google_4.build_circuit()
circuits.append((bv4_p, 'Google_4'))
#%%
simulator = Aer.get_backend('qasm_simulator')

circuit_fodler = "circuits/"
for c in circuits:
    q_c = qiskit.circuit.QuantumCircuit.from_qasm_str(c[0].qtape.to_openqasm())

    # Run and get counts
    result = simulator.run(q_c).result()
    counts = result.get_counts(q_c)
    plot_histogram(counts, title='Bell-State counts', filename=str(circuit_fodler+c[1]+"_plot.png"))

    q_c.draw(output="mpl", filename=str(circuit_fodler+c[1]))

#%%
device_backend = FakeSantiago()
coupling_map = get_qiskit_coupling_map(circuits[0][0], device_backend)

#%%
results_names = execute_over_range(circuits, results_folder="./tmp/")

# %%
from qufi import execute_over_range, get_qiskit_coupling_map, read_results_directory, generate_all_statistics
results = read_results_directory("./tmp/", noise=True)

generate_all_statistics(results)
# %%

# %%
