# Tumor2Graph: a novel Overall-Tumor-Profile-derived virtual graph deep learning for predicting tumor typing and subtyping.

In this research, we propose a novel multimodal graph framework, namely Tumor2Graph, to jointly model 7 types of biomarkers (including structured data and unstructured data) for predicting tumor typing and subtyping. For the structured data (CNV, SNV (gene mutations), DNA methylation, mRNA (gene expression), miRNA, protein), We use element-wise add to integrate the primary feature embedding vectors in the first and second fully connected layers and the Laplacian smoothing embedding vectors in the graph convolutional layer. For the unstructured data (pathology images), we separate their feature extraction algorithms due to their specificity. We use a neural module including a 2D convectional layer and concatenate the extracted feature embedding vectors with those of the structured data to work as patient embedding vectors. The patient embedding vectors are directly used for supervised learning to classify tumor typing and unsupervised learning to cluster tumor subtyping.


## TCGA Datasets

Download [TCGA datasets](https://gdc.cancer.gov/about-data/publications/pancanatlas) from [official portal](https://www.cancer.gov/about-nci/organization/ccg/research/structural-genomics/tcga), including the training, validation, and test dialogues and the features of Charades videos extracted using VGGish and I3D models.

All the data should be saved into folder data in the repo root folder.




`srun -p MIA -n1 -w SH-IDC1-10-5-30-204 --gres=gpu:2 --mpi=pmi2 python -u vgcn_trainer_cnn_new.py >> new_res_new.log 2>&1 &`






```
df_all = df_all.filter(regex="(?<!snv)$")
df_all = df_all.filter(regex="(?<!_rna)$")
df_all = df_all.filter(regex="(?<!methy)$")
df_all = df_all.filter(regex="(?<!cnv)$")
df_all = df_all.filter(regex="(?<!mirna)$")
df_all = df_all.filter(regex="(?<!rppa)$")
```
