# -*- coding: utf-8 -*-

from lifelines.datasets import load_waltons
from lifelines import KaplanMeierFitter
from lifelines.utils import median_survival_times
from lifelines.statistics import logrank_test,multivariate_logrank_test
from lifelines import CoxPHFitter
#from lifelines.plotting import add_at_risk_counts
from my_plotting import add_at_risk_counts
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from matplotlib.backends.backend_pdf import PdfPages
#from sksurv.linear_model import CoxPHSurvivalAnalysis as cpa
#sys.exit()

#%%
cluster_result = pd.read_csv("/home/user/zyw_document/st_zyw/cancer_classifier/data/dna_rna_methy_cluster_result.csv")
data = pd.read_csv(r"/home/user/TCGA_DNA_methylation/Survival_SupplementalTable_S1_20171025_xena_sp",sep="\t")
#cluster_result["sample"] = cluster_result["sample_id"].apply(lambda x:x[:15])
df = pd.merge(cluster_result,data[["sample","OS.time","OS"]],how="inner",left_on="sample_id",right_on="sample")
df = df.loc[df["OS.time"].dropna().index]
#df_brca = df[df["true_label"] == "KIRP"]
#%%
sys.exit()
cluster_col = "hierachical_separate_clstr"
df_p = pd.DataFrame()
pdff=PdfPages(f"../data/cluster_result.pdf")
for cluster_col in ["hierachical_separate_clstr","hierachical_separate_clstr_2","hierachical_separate_clstr_3","hierachical_separate_clstr_4"]:
#firstPage = plt.figure(figsize=(30,3))


##plt.show()
#pdff.savefig()
#firstPage.clf()
#plt.close()
    fig = plt.figure(figsize=(30, 40), dpi=100)
    fig.clf()
    plt.tight_layout(pad=0.01, h_pad=0.01, w_pad=0.01,)#调整整体空白
    plt.subplots_adjust(wspace =0.5, hspace =0.7)#调整子图间距
    n = 1
    
    
    #pdff.savefig()
    #plt.close()
    #txt = 'this is an example'
    #plt.text(0.05,0.95,txt, transform=plt.transFigure, size=24)
    p_cancer_type = []
    cancer_type_list = []
    for gp in df.groupby("true_label"):
    
        
        df_cancer = gp[1]
        print(gp[0])
        if gp[0]=="KICH":
            continue
        result = multivariate_logrank_test(df_cancer["OS.time"],df_cancer[cluster_col],df_cancer["OS"])
        
        cph = CoxPHFitter()
        
        cph.fit(df_cancer[[cluster_col,"OS.time","OS"]],duration_col="OS.time",event_col="OS")
        #cph.print_summary()
        p_value = round(cph.summary["p"].values[0],5)
#        if p_value<0.05:
        p_cancer_type.append(p_value)
        cancer_type_list.append(gp[0])
        HR =  round(cph.summary["exp(coef)"][0],2)
        low_HR =  round(cph.summary["exp(coef) lower 95%"][0],2)
        upper_HR =  round(cph.summary["exp(coef) upper 95%"][0],2)
        
        
        #plt1 = plt.subplot(221)
    #    add_at_risk_counts(kmf)
    
        ax = plt.subplot(7,5,n)
#        plt.clf()
    #    ax = plt.add_subplt(6,6,n)
        
        n+=1
        #ax.legend(labels=[f"HR={HR}({low_HR}-{upper_HR})",f"p-value={p_value}"])
        #fig = plotly.tools.make_subplots(rows=2, cols=1, print_grid=False)
        #ax = kmf.plot(ci_alpha = 0,marker='*') kmf.survival_function_.
    #    fig,ax = plt.subplot(111)
        kmfl = []
        labels = []
        for gp in df_cancer.groupby(cluster_col):
            label = "C "+str(int(gp[0]))
            labels.append(label)
            kmf = KaplanMeierFitter()
            kmf.fit(gp[1]["OS.time"],event_observed = gp[1]["OS"])
            kmfl.append(kmf)
            
            x = kmf.survival_function_.index[kmf.survival_function_.index<=2000]
            y = kmf.survival_function_.values[kmf.survival_function_.index<=2000].flatten()
        #    x = kmf.survival_function_.index
        #    y = kmf.survival_function_.values.flatten()
            plt.plot(x,y,marker='x',linestyle="-",label = label)
            plt.ylim((-0.05,1.05))
            plt.xlim((-50,2050))
            my_y_ticks = np.arange(0, 1.05, 0.2)
            my_x_ticks = np.arange(0, 2050, 400)
            plt.yticks(my_y_ticks)
            plt.xticks(my_x_ticks)
            gca = plt.gca()
            gca.spines['top'].set_visible(False)
            gca.spines['right'].set_visible(False)
            
            
            
    #        ax = add_at_risk_counts(kmf,ax=ax)
        plt.legend(loc="upper right",frameon=0)
        plt.text(-3,0.18,f"HR = {HR}({low_HR} - {upper_HR})")
        plt.text(-3,0.08,f"p-value = {p_value}")
        plt.xlabel("Time(days)")
        plt.ylabel("Probability of survival")
        plt.title(df_cancer["true_label"].unique()[0])
    #    add_at_risk_counts(kmfl[1],ax=ax,labels=["Number at risk",""],rows_to_show=["At risk"])
        if n==4:
            txt = cluster_col
            plt.text(1000,2,txt,ha="center",size=24)
        y_pos = -0.2
        for i,kmf in enumerate(kmfl):
            y_pos -= 0.2
            if i==0:
                l=["Number at risk"]
            else:
                l=[""]
            ax,ticklabels = add_at_risk_counts(kmf,cluster=labels[i],ax=ax,ypos=y_pos,labels=l,rows_to_show=["At risk"])
    #        ticklabels[0] = ticklabels[0].replace("At risk",labels[i])
    #        ax.set_xticklabels(ticklabels, ha="right")
    #    ax.legend()
    #    break
    
#    plt.close()
    print(cluster_col+":"+str(len(p_cancer_type)))
    df_p[cluster_col] = np.array(p_cancer_type)
    pdff.savefig()
#pdff.savefig()
plt.savefig(f"../data/{cluster_col}_.png")

pdff.close()
plt.show()









