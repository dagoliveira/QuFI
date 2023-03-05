#%%
from qufi import execute_over_range_fast, BernsteinVazirani, Google_3, Try_0, Try_1
import numpy as np
import pickle, gzip
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors

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
    data = pickle.load(gzip.open(filename, 'r'))
    
    qvf_index = data[1]
    qvf       = data[0]
    name      = qvf.pop("name", None)

    for k in qvf.keys():
        print(k)
        title = "{}_QVF_{}".format(name, str(k))

        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        divnorm = colors.TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1)
        rdgn = sns.diverging_palette(h_neg=130, h_pos=10, s=200, l=55, sep=20, as_cmap=True)
        sns.set(font_scale=1.3)
        
        qvf_tmp = np.zeros((25,13))
        index = 0
        for j in range(13):
            for i in range(25):
                qvf_tmp[i][j] = qvf[k][index]
                index += 1

        # ax = sns.heatmap(qvf_tmp, xticklabels=theta_list_tex, yticklabels=phi_list_tex, cmap=rdgn, cbar_kws=param, vmin=0, vmax=1).set(title=title)
        ax = sns.heatmap(qvf_tmp, xticklabels=theta_list_tex, yticklabels=phi_list_tex, cmap=rdgn, cbar_kws=param, vmin=0, vmax=1)
        fig.savefig('heatmap_fast/{}_heatmap.pdf'.format(title), bbox_inches='tight')
        plt.close()

    n = 0
    for q in qvf_index:
        print(n)

        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        divnorm = colors.TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1)
        rdgn = sns.diverging_palette(h_neg=130, h_pos=10, s=200, l=55, sep=20, as_cmap=True)
        sns.set(font_scale=1.3)
        
        qvf_tmp = np.zeros((25,13))
        index = 0
        for j in range(13):
            for i in range(25):
                qvf_tmp[i][j] = q[index]
                index += 1

        title = "{}_QVF_index_{}".format(name, str(n))
        ax = sns.heatmap(qvf_tmp, xticklabels=theta_list_tex, yticklabels=phi_list_tex, cmap=rdgn, cbar_kws=param, vmin=0, vmax=1).set(title=title)
        fig.savefig('heatmap_fast/{}_heatmap.pdf'.format(title), bbox_inches='tight')
        plt.close()
        n += 1
# %%