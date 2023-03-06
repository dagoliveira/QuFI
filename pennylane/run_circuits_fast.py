#%%
from qufi import execute_over_range_fast, BernsteinVazirani, Google_3, Try_0, Try_1, read_results_directory_fast, generate_all_statistics_fast
import numpy as np
import pickle, gzip
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors

#%%

circuits = []

# c = Google_3.build_circuit()
# circuits.append((c, 'Google_3'))

bv4_p = BernsteinVazirani.build_circuit()
circuits.append((bv4_p, 'BernsteinVazirani_4'))

# try_0 = Try_0.build_circuit()
# circuits.append((try_0, 'try_0'))

# try_1 = Try_1.build_circuit()
# circuits.append((try_1, 'try_1'))

results_names = execute_over_range_fast(circuits, results_folder="./tmp_qvf/")

#%%
# read using the older method

results = read_results_directory_fast("./tmp_try/", noise=False)

qvf_old = generate_all_statistics_fast(results, savepath="./plots_try")

# qvf_old["BernsteinVazirani_4"]["QVF_index_0_qubit_3"]

#%%
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
    print()
    data = pickle.load(gzip.open(filename, 'r'))
    
    name      = data[2]
    qvf_index = data[1]
    qvf       = data[0]

    # prints qubit heatmaps
    for k in qvf.keys():
        # print(k)
        # title = "{}_QVF_{}".format(name, str(k))

        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        divnorm = colors.TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1)
        rdgn = sns.diverging_palette(h_neg=130, h_pos=10, s=200, l=55, sep=20, as_cmap=True)
        sns.set(font_scale=1.3)
        
        qvf_tmp = np.zeros((25,13))
        index = 0
        for j in range(13):
            for i in range(25):
                # print(index, qvf[k][index])
                qvf_tmp[i][j] = qvf[k][index]
                index += 1

        # ax = sns.heatmap(qvf_tmp, xticklabels=theta_list_tex, yticklabels=phi_list_tex, cmap=rdgn, cbar_kws=param, vmin=0, vmax=1).set(title=title)
        ax = sns.heatmap(qvf_tmp, xticklabels=theta_list_tex, yticklabels=phi_list_tex, cmap=rdgn, cbar_kws=param, vmin=0, vmax=1)
        fig.savefig('heatmap_fast/{}_{}_heatmap.pdf'.format(name, k), bbox_inches='tight')
        plt.close()

        # diff, Euclidean distance
        diff =np.sqrt(np.square(qvf[k] - qvf_old[name][k]))
        print(f"Diff {k}: {round(diff.sum(),3)}")
        # diff =np.sqrt(np.square(np.zeros(325) + [round(num, 3) for num in qvf[k]] - qvf_old[name][k]))
        # print(f"Diff {k}: {diff.sum()}")

    print()

    n = 0
    # prints index heatmaps
    for k in qvf_index.keys():
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        divnorm = colors.TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1)
        rdgn = sns.diverging_palette(h_neg=130, h_pos=10, s=200, l=55, sep=20, as_cmap=True)
        sns.set(font_scale=1.3)
        
        qvf_tmp = np.zeros((25,13))
        index = 0
        for j in range(13):
            for i in range(25):
                qvf_tmp[i][j] = qvf_index[k][index]
                index += 1

        title = "{}_QVF_index_{}".format(name, str(n))
        ax = sns.heatmap(qvf_tmp, xticklabels=theta_list_tex, yticklabels=phi_list_tex, cmap=rdgn, cbar_kws=param, vmin=0, vmax=1)
        fig.savefig('heatmap_fast/{}_{}_heatmap.pdf'.format(name, k), bbox_inches='tight')
        plt.close()
        n += 1

        # diff, Euclidean distance
        # print(qvf_index[k])
        # print(qvf_old[name][k])
        diff =np.sqrt(np.square(qvf_index[k] - qvf_old[name][k]))
        print(f"Index {n}, diff {k}: {round(diff.sum(),3)}")
        # diff =np.sqrt(np.square(np.zeros(325) + [round(num, 3) for num in qvf_index[k]] - qvf_old[name][k]))
        # print(f"Index {n}, diff {k}: {diff.sum()}")
