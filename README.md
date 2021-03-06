# Tumor2Graph: a novel Overall-Tumor-Profile-derived virtual graph deep learning for predicting tumor typing and subtyping.

In this research, we propose a novel multimodal graph framework, namely Tumor2Graph, to jointly model 7 types of biomarkers (including structured data and unstructured data) for predicting tumor typing and subtyping. For the structured data (CNV, SNV (gene mutations), DNA methylation, mRNA (gene expression), miRNA, protein), We use element-wise add to integrate the primary feature embedding vectors in the first and second fully connected layers and the Laplacian smoothing embedding vectors in the graph convolutional layer. For the unstructured data (pathology images), we separate their feature extraction algorithms due to their specificity. We use a neural module including a 2D convectional layer and concatenate the extracted feature embedding vectors with those of the structured data to work as patient embedding vectors. The patient embedding vectors are directly used for supervised learning to classify tumor typing and unsupervised learning to cluster tumor subtyping.


## Datasets

Download [TCGA datasets](https://gdc.cancer.gov/about-data/publications/pancanatlas) from [official portal](https://www.cancer.gov/about-nci/organization/ccg/research/structural-genomics/tcga), including CNV, SNV (gene mutations), DNA methylation, mRNA (gene expression), miRNA, protein, and pathology images. The standardized, normalized, batch-corrected, and platform-corrected data matrices and mutation data were generated by [the PanCancer Atlas](https://www.cell.com/pb-assets/consortium/pancanceratlas/pancani3/index.html) and collected from [the Genomic Data Commons portal](https://gdc.cancer.gov/about-data/publications/pancanatlas) and [The Cancer Genome Atlas](https://www.cancer.gov/about-nci/organization/ccg/research/structural-genomics/tcga) using [Xena Functional Genomics Explorer](https://xenabrowser.net/), [University of California, Santa Cruz](https://www.ucsc.edu/). 

Data Preprocessing: [MultimodalTCGA](https://github.com/Eurus-Holmes/MultimodalTCGA)


All the data should be saved into folder `dataset/v10/` in the repo root folder.



## Installation

```
pip install -r requirements.txt
```


## Usage

Run the following code on the computing cluster to predict the results of tumor typing and subtyping.

```
srun -p MIA -n1 -w SH-IDC1-10-5-30-204 --gres=gpu:2 --mpi=pmi2 python -u Tumor2Graph.py >> res.log 2>&1 &
```


### Ablation Study

Just add a line of code to the `tumor_dataset.py` to get the result of the corresponding ablation study. Such as:

```python
# Without SNV:
df_all = df_all.filter(regex="(?<!snv)$")

# Without RNA:
df_all = df_all.filter(regex="(?<!_rna)$")

# Without Methylation:
df_all = df_all.filter(regex="(?<!methy)$")

# Without CNV:
df_all = df_all.filter(regex="(?<!cnv)$")

# Without miRNA:
df_all = df_all.filter(regex="(?<!mirna)$")

# Without RPPA:
df_all = df_all.filter(regex="(?<!rppa)$")
```

Without pathology images, just use `VGCN.py` to replace `Tumor2Graph.py`.

> VGCN means Virtual Graph Convolutional Networks, check [here](https://github.com/Eurus-Holmes/VGCN) for more information.


### Identifying Tumor2Graph's tumor sub-typing in OS and PFS

Run `os.sh` and `pfs.sh`.
