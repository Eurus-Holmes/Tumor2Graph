import pandas as pd

cancer = pd.read_csv("cancer.csv")
cluster2 = pd.read_csv("2_pfs.csv")
cluster3 = pd.read_csv("3_pfs.csv")
cluster4 = pd.read_csv("4_pfs.csv")
cluster = pd.read_csv("clstr_pfs.csv")

df1 = pd.concat([cluster2, cluster3, cluster4, cluster], axis=1, join='inner')
df1 = df1.T.drop_duplicates().T
# print(df1)

df2 = pd.merge(cancer, df1, how='inner')
# print(df2)
print(df2.iloc[:, 2:])

a = df2.iloc[:, 1:]
df2['best_cluster'] = a.min(axis=1)
print(df2)

df2.to_csv("new/PFS_complete_new_new.csv",index=False,sep=',')