import os
# os.chdir(wd)
import warnings

warnings.filterwarnings("ignore")

# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity = "all"
show_rows = 30
cover_flg = True


import numpy as np

np.set_printoptions(threshold=show_rows)

import pandas as pd

pd.options.display.max_rows = int(show_rows)
pd.options.display.max_columns = int(show_rows)

import pickle
import gc
from collections import defaultdict
import sys


# sys.exit()
def noCoverWrite(df, file, f=True):
    if (not (os.path.isfile(file))) or f:
        df.to_csv(file)
        print("wrote csv")
    else:
        print("no cover")


# %% load data
# file_path = "save_embedding/VGCN_seed=3_batch=1024_embedding.npy"
file_path = "save_embedding/Tumor2Graph_seed=5_batch=1024_embedding.npy"
df_y_ori = np.load(file_path)
# print(df_y_ori)
print(df_y_ori.shape)
df_emb_ori = pd.DataFrame(
    df_y_ori,
    columns=['col_%d' % i for i in range(np.shape(df_y_ori)[1])],
    #        index=train_data['sample']
)
# df_y_ori = df_y_ori[:6180] ##use train data
## load train data
df_all = pd.read_csv("dataset/v10/df_all_new2.csv")
# print(list(train_df.columns))
# print(list(extra_test_df.columns))
df_all = df_all[df_all["split_type"] == "primary_train"]
print(df_all[["sample", "cancer_type"]])
print(df_emb_ori.shape)
print(df_all[["sample", "cancer_type"]].shape)
df_new = pd.concat([df_emb_ori, df_all[["sample", "cancer_type", "split_type"]]], axis=1)
# df_emb_ori = df_new[df_new["split_type"]=="primary_test"]

df_emb_ori = df_new.dropna(axis=0, how='any')

# df_emb_ori = pd.DataFrame(
#         df_y_ori,
#         columns=['col_%d'% i for i in range(np.shape(df_y_ori)[1])],
# #        index=train_data['sample']
#         )
# df_emb_ori["sample"] = train_data["sample"]
# df_emb_ori["cancer_type"] = train_data["cancer_type"]
# sys.exit()
# df_emb = df_emb_ori[np.array(df_y_ori['metastatic']!=1)].copy()
# assert(np.sum((df_y['sample_id'] != df_emb.index))==0)
df_emb = df_emb_ori[[c for c in df_emb_ori.columns if c not in ["sample", "cancer_type", "split_type"]]]
print(df_emb.shape)
print(df_emb)
df_y = df_emb_ori[["cancer_type"]]
df_y.rename(columns={"cancer_type": "true_label"}, inplace=True)
# (df_emb.iloc[0,:] * df_emb.iloc[1,:]).sum()

from sklearn.preprocessing import scale

#### scale process
# df_emb_scaled = pd.DataFrame(scale(df_emb), index=df_emb.index, columns=df_emb.columns)
# df_emb_scaled = df_emb

df_emb = df_emb.applymap(lambda x: np.exp(x))
df_emb_scaled = pd.DataFrame(scale(df_emb), index=df_emb.index, columns=df_emb.columns)
print(df_emb_scaled.head())

# %% hierarchical

# import scipy
# import scipy.cluster.hierarchy as sch
from scipy.cluster.vq import vq, kmeans, whiten
import numpy as np
import matplotlib.pylab as plt

mtd = ['single', 'complete', 'average', 'weighted', 'ward']
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, distance

cancre_type = pd.DataFrame()
print(df_emb_ori['cancer_type'].unique())
x = df_emb_ori['cancer_type'].unique()
print(list(filter(lambda v: v == v, x)))
y = list(filter(lambda v: v == v, x))
cancre_type['cancer'] = np.sort(y)
cancre_type['clstr'] = 4
df_y['hierachical_separate_clstr'] = np.nan
df_y["sample_id"] = df_emb_ori["sample"]

Cohort = []
nums = []
# %%
for i in range(len(cancre_type['cancer'])):
    #    i = 0
    cancre_type_ = cancre_type['cancer'][i]
    #    n_clstrs = cluster_num[cancre_type_]
    num = (df_y['true_label'] == cancre_type_).sum()
    print(cancre_type_)
    print('num: %d' % num)
    Cohort.append(cancre_type_)
    nums.append(num)
    if num == 1:
        continue
    df_ = df_emb_scaled.loc[np.array(df_y['true_label'] == cancre_type_), :].copy()

    j = 3
    disMat = distance.pdist(df_, 'euclidean')
    Z = linkage(disMat, mtd[j])
    # f = fcluster(Z,6,'distance')

    fig = plt.figure(figsize=(30, 18))

    dn = dendrogram(Z)
    # plt.show()
    plt.savefig('cluster_ward_dna_rna_methy/%2d_%s_%s.png' % (i, cancre_type_, mtd[j]))
    plt.close()

    clstr = fcluster(Z, t=len(np.unique(dn['color_list'])) - 1, criterion='maxclust')
    # clstr= fcluster(Z, t=4, criterion='maxclust')
    #    assert(df_y.loc[df_y['true_label']==cancre_type_,'hierachical_separate_clstr'].notnull().sum()==0)

    df_y.loc[df_y['true_label'] == cancre_type_, 'hierachical_separate_clstr'] = clstr

    print(len(np.unique(dn['color_list'])))
    for n in range(2, 5):
        #        if not n!=n:
        #            clstr= fcluster(Z, t=len(np.unique(dn['color_list']))-1, criterion='maxclust')
        ##
        # clstr= fcluster(Z, t=n, criterion='maxclust')
        clstr = fcluster(Z, t=n, criterion='maxclust')
        print("cluster num:", len(set(clstr)))
        #            assert(df_y.loc[df_y['true_label']==cancre_type_,'hierachical_separate_clstr'].notnull().sum()==0)

        df_y.loc[df_y['true_label'] == cancre_type_, 'hierachical_separate_clstr_' + str(n)] = clstr

print(df_y['hierachical_separate_clstr'])
print(df_y['hierachical_separate_clstr'].isnull().sum())

dataframe = pd.DataFrame({'Cohort': Cohort, 'numbers': nums})
dataframe.to_csv("cancer.csv", index=False, sep=',')

# assert(df_y['hierachical_separate_clstr'].isnull().sum()==0)

noCoverWrite(df_y, 'output/dna_rna_methy_cluster_result.csv')

