import numpy as np
import matplotlib as plot
from modules import *
import argparse
import time
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns
from rdkit import Chem, RDLogger, DataStructs


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', metavar='N', type=str, nargs='+',
                    help='an integer for the accumulator')
# chembl_dopamine_d2
# chembl_factor_xa
# postera_sars_cov_2_mpro
args = parser.parse_args()


datafolder_filepath = "data/"+ args.dataset[0] +"/"
X_smiles_mmps = load_dict(datafolder_filepath + '/X_smiles_mmps.txt')
# breakpoint()
ecfp = pd.DataFrame.from_dict(load_dict(datafolder_filepath +'/smiles_ecfp_dict.pkl'))
e3fp = load_dict(datafolder_filepath +'/smiles_e3fp_dict.pkl')
for j in e3fp.keys():
    e3fp[j] = np.array(e3fp[j][0].to_rdkit())


def tanimoto(v1, v2):
    return(np.bitwise_and(v1, v2).sum() / np.bitwise_or(v1, v2).sum())

ecfp_mmps_tan = []
e3fp_mmps_tan = []

for i in range(len(X_smiles_mmps)):
    tan = tanimoto(ecfp[X_smiles_mmps[i][0]],ecfp[X_smiles_mmps[i][1]])
    ecfp_mmps_tan.append(tan)

for n in range(len(X_smiles_mmps)):
    tan = tanimoto(e3fp[X_smiles_mmps[n][0]],e3fp[X_smiles_mmps[n][1]])
    e3fp_mmps_tan.append(tan)

# breakpoint()


# e3fp = pd.DataFrame.from_dict(e3fp)


# pca = PCA(n_components=2)

# ecfp_transformed = pca.fit_transform(ecfp.values)
# # breakpoint()
# e3fp_transformed = pca.fit_transform(e3fp.values)

# plt.figure(figsize=(10, 10))
# plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

fig, ax = plt.subplots()

plt.title('Tan(ECFP) vs Tan(E3FP)')

ax.scatter(ecfp_mmps_tan,e3fp_mmps_tan, c='black')
line = mlines.Line2D([0, 1], [0, 1], color='red')

transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)

ax.set_xlabel('Tan(ECFP)')
ax.set_ylabel('Tan(E3FP)')


# plt.plot(ecfp_mmps_tan,e3fp_mmps_tan, color='blue',linestyle = 'None',marker='o')

plt.savefig(datafolder_filepath+'ecfp_e3fp_plot.png', format='png')

