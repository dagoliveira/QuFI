#%%
import numpy as np
from qiskit.test.mock import FakeSantiago
from itertools import product
from qufi import execute_over_range_dict, anglePair, anglePair_statevector, read_results_directory, generate_all_statistics

angles={'theta':np.arange(0, np.pi+0.01, np.pi/12), 
        'phi':np.arange(0, 2*np.pi+0.01, np.pi/12)}
circuits    = []
statevector = []

# Add only the angle pairs which create different statevectors

angle_combinations = product(angles['theta'], angles['phi'])
i = 0
for angle_pair in angle_combinations:
    c = anglePair.build_circuit(angle_pair[0], angle_pair[1])
    s = anglePair_statevector.build_circuit(angle_pair[0], angle_pair[1])
    if not (str(s()) in statevector):
        i +=1
        print(f"theta: {angle_pair[0]}  - phi: {angle_pair[1]}")
        print(str(s()))
        print()
        statevector.append(str(s()))
        circuits.append((c, s, 'Angle_pair_'+str(angle_pair[1])+'_'+str(angle_pair[1])))
    # break

print("Total state are: "+str(i))

results_names = execute_over_range_dict(circuits, results_folder="./tmp_state_vector/")

# %%
results = read_results_directory("./tmp/", noise=True)

generate_all_statistics(results)
# %%

# %%