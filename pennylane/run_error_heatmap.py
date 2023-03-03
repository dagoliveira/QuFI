#%%
import numpy as np
import pickle, gzip
from itertools import product
import pennylane as qml

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors

from qiskit.circuit import QuantumCircuit
from qiskit.visualization import plot_bloch_vector

from qufi import execute_over_range_dict, anglePair, anglePair_statevector

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
    kets = np.round(s().numpy(), 4)
    print(s().numpy())
    print(kets)
    flag = True
    for state in statevector:
        if np.array_equal(state, kets):
            flag = False
            break
    if flag:
        print("TAKE")
        print(f"theta: {angle_pair[0]}  - phi: {angle_pair[1]}")
        # print(str(s()))
        # print()
        statevector.append(kets)
        circuits.append((c, s, kets, 'Angle_pair_'+str(angle_pair[0])+'_'+str(angle_pair[1])))
        # print(str(s.qtape.to_openqasm()))
        q_circuit = QuantumCircuit.from_qasm_str(s.qtape.to_openqasm())
        q_circuit.draw(output="mpl", filename='heatmap_angle_pairs/{}_circuit.pdf'.format('Angle_pair_'+str(angle_pair[0])+'_'+str(angle_pair[1])))
        i +=1
        # break
    else:
        print("NOT TAKE")
    print()
print("Total state are: "+str(i))
print()
print()

#%%

results_names = execute_over_range_dict(circuits, results_folder="./tmp_state_vector/")

# %%
theta_list_tex = ['0', '', '', '$\\frac{\pi}{4}$', '', '', '$\\frac{\pi}{2}$', ''
                    , '', '$\\frac{3\pi}{4}$', '', '', '$\pi$']
phi_list_tex = ['', '', '$\\frac{7\pi}{4}$', '', ''
            , '$\\frac{6\pi}{4}$', '', '', '$\\frac{5\pi}{4}$', ''
            , '', '$\pi$', '', '',  '$\\frac{3\pi}{4}$'
            , '', '', '$\\frac{\pi}{2}$', '', '', '$\\frac{\pi}{4}$'
            , '', '', '0']
param={'label': 'QVF'}

for filename in results_names:
    print(filename)
    data = pickle.load(gzip.open(filename, 'r'))[0]

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    divnorm = colors.TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1)
    rdgn = sns.diverging_palette(h_neg=130, h_pos=10, s=200, l=55, sep=20, as_cmap=True)
    sns.set(font_scale=1.3)

    qvf_tmp = np.empty((25,13))
    index = 0
    for j in range(13):
        for i in range(25):
            qvf_tmp[i][j] = data['qvf'][index]
            index += 1
    title = "Bloch {}".format(str(data['bloch_vector']))

    ax = sns.heatmap(qvf_tmp, xticklabels=theta_list_tex, yticklabels=phi_list_tex, cmap=rdgn, cbar_kws=param, vmin=0, vmax=1).set(title=title)
    fig.savefig('heatmap_angle_pairs/{}_heatmap.pdf'.format(data['name']), bbox_inches='tight')
    plt.close()

    plot_bloch_vector(data['bloch_vector']).savefig('heatmap_angle_pairs/{}_bloch.pdf'.format(data['name']), bbox_inches='tight')
# %%
