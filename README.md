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
