# Tumor2Graph: a novel Overall-Tumor-Profile-derived virtual graph deep learning for predicting tumor typing and subtyping.

In this research, we propose a novel multimodal graph framework, namely Tumor2Graph, to jointly model 7 types of biomarkers (including structured data and unstructured data) for predicting tumor typing and subtyping, namely Tumor2Graph. For the structured data (CNV, SNV (gene mutations), DNA methylation, mRNA (gene expression), miRNA, protein), We use element-wise add to integrate the primary feature embedding vectors in the first and second fully connected layers and the Laplacian smoothing embedding vectors in the graph convolutional layer. For the unstructured data (pathology images), we separate their feature extraction algorithms due to their specificity. We use a neural module including a 2D convectional layer and concatenate the extracted feature embedding vectors with those of the structured data to work as patient embedding vectors. The patient embedding vectors are directly used for supervised learning to classify tumor typing and unsupervised learning to cluster tumor subtyping.

新数据
srun -p MIA -n1 -w SH-IDC1-10-5-30-204 --gres=gpu:2 --mpi=pmi2 python -u vgcn_trainer_cnn_new.py >> new_res_new.log 2>&1 &


去图像
srun -p MIA -n1 -w SH-IDC1-10-5-30-204 --gres=gpu:2 --mpi=pmi2 python -u vgcn_trainer_v2.py >> new_res2.log 2>&1 &


去掉snv:
srun -p MIA -n1 -w SH-IDC1-10-5-30-204 --gres=gpu:2 --mpi=pmi2 python -u vgcn_trainer_cnn_new4.py >> new_res3.log 2>&1 &


去掉rna: 
srun -p MIA -n1 -w SH-IDC1-10-5-30-207 --gres=gpu:2 --mpi=pmi2 python -u vgcn_trainer_cnn_new5.py >> new_res4.log 2>&1 &


去掉methy: 
srun -p MIA -n1 -w SH-IDC1-10-5-30-207 --gres=gpu:2 --mpi=pmi2 python -u vgcn_trainer_cnn_new6.py >> new_res5.log 2>&1 &


去掉cnv:
srun -p MIA -n1 -w SH-IDC1-10-5-30-207 --gres=gpu:2 --mpi=pmi2 python -u vgcn_trainer_cnn_new7.py >> new_res6.log 2>&1 &


去掉mirna:
srun -p MIA -n1 -w SH-IDC1-10-5-30-198 --gres=gpu:2 --mpi=pmi2 python -u vgcn_trainer_cnn_new8.py >> new_res7.log 2>&1 &


去掉rppa:
srun -p MIA -n1 -w SH-IDC1-10-5-30-201 --gres=gpu:2 --mpi=pmi2 python -u vgcn_trainer_cnn_new9.py >> new_res8.log 2>&1 &



Cluster:
srun -p MIA -n1 -w SH-IDC1-10-5-30-201 --gres=gpu:2 --mpi=pmi2 python -u vgcn_trainer_v2.py >> new_res222.log 2>&1 &




df_all = df_all.filter(regex="(?<!snv)$")
df_all = df_all.filter(regex="(?<!_rna)$")
df_all = df_all.filter(regex="(?<!methy)$")
df_all = df_all.filter(regex="(?<!cnv)$")
df_all = df_all.filter(regex="(?<!mirna)$")
df_all = df_all.filter(regex="(?<!rppa)$")
