### Import libraries


import pandas as pd
import matplotlib.pyplot as plt
from seaborn import violinplot
import numpy as np


### Functions useful for the drug comparison


def plot_comparison(score_rf, score_vae, score_vae_highcorr, label, title, save = False, out_path = "figures/comparison_vae_rf.png"):
    """
    Plot a comparison of the random forest and vae across the three metrics.
    """
    score = [score_rf, score_vae, score_vae_highcorr]
    xlim = [(-0.5, 0.7), (0, 9), (0.35, 0.75)]

    fig = plt.figure(constrained_layout = True, figsize = (15,15))
    fig.suptitle(title, fontsize = 15, fontweight = "bold")
    subfigs = fig.subfigures(nrows = 3, ncols = 1)

    for i in range(len(subfigs)):
        subfigs[i].suptitle(label[i], fontweight = "bold")
        axs = subfigs[i].subplots(nrows = 1, ncols = 3)
        
        for j, ax in enumerate(axs.flat):

            ax.grid(True)
            ax.set_title(score[i].columns[j])
            ax.set_xlim(xlim[j])
            ax.boxplot(score[i][score[i].columns[j]], vert = False)
                        
    if save:
        plt.savefig(out_path, bbox_inches = 'tight', dpi = 300)

def plot_violin(score_RF_downsampling, score_RF_veryhigh_corr, score_RF_PCA, score_RF_embedding, score_RF, save = False, out_path = "figures/embedding_comparisons.png"):
    """
    Plot a comparison of different RF models fitted with different input data, as violin plot.
    """
    df1 = pd.DataFrame([score_RF_downsampling["Pearson's r"].tolist(), 349*["Down-sampled"]], \
                 index = ["Pearson's r", "Input variables"]).T
    df2 = pd.DataFrame([score_RF_veryhigh_corr["Pearson's r"].tolist(),349*["Drug-correlated"]],\
                 index = ["Pearson's r", "Input variables"]).T
    df3 = pd.DataFrame([score_RF_PCA["Pearson's r"].tolist(), 349*["PCA"]], \
                 index = ["Pearson's r", "Input variables"]).T
    df4 = pd.DataFrame([score_RF_embedding["Pearson's r"].tolist(), 349*["VAE embedded"]], \
                 index = ["Pearson's r", "Input variables"]).T
    df5 = pd.DataFrame([score_RF["Pearson's r"].tolist(), 349*["All proteins"]], \
                 index = ["Pearson's r", "Input variables"]).T
    df = pd.concat([df1, df2, df3, df4, df5])
    df["Pearson's r"] = df["Pearson's r"].astype(float)

    plt.figure(constrained_layout = True, figsize = (8,8))
    ax = violinplot(x = "Input variables", y = "Pearson's r", data = df)
    ax.axhline(df4["Pearson's r"].median(), ls='-.', color = "black")
    ax.axhline(df4["Pearson's r"].quantile(0.25), ls=':', color = "purple")
    ax.axhline(df4["Pearson's r"].quantile(0.75), ls=':', color = "purple")

    plt.suptitle("Comparison of Random Forest models fitted with different input data",\
                 fontsize = 15, fontweight = "bold")

    if save:
        plt.savefig(out_path, bbox_inches = 'tight', dpi = 300)

def get_drugs(score, bad_pred = True):
    """
    Get a list of drugs that are either well-predicted (bad_pred = False) according to all metrics of the score 
    (Pearson's r, RMSE, C-index), or badly-predicted (bad_pred = True).
    """
    drug_list = []
    
    if bad_pred:
        for i, elem in enumerate(score.values):
            if elem[0] < score["Pearson's r"].quantile(0.25) and elem[1] > score["RMSE"].quantile(0.75) and elem[2] < score["C-index"].quantile(0.25): # then the drug is badly predicted at all level
                drug_list.append(score.index[i])
    
    else:
        for i, elem in enumerate(score.values):
            if elem[0] > score["Pearson's r"].quantile(0.75) and elem[1] < score["RMSE"].quantile(0.25) and elem[2] > score["C-index"].quantile(0.75): # then the drug is well predicted at all level
                drug_list.append(score.index[i])
    
    return(drug_list)

def plot_drug_distrib(drug, bad_drugs, negative_skew, save = False, out_path = "figures/drugs_distrib.png"):
    """
    Plot the distributions of badly predicted drugs (bad_drugs) and well-predicted drugs with negative 
    skew (negative_skew) to see whether there are noticable differences which could explain the prediction.
    """
    fig = plt.figure(constrained_layout = True, figsize = (15,10))
    fig.suptitle("Distributions of AUC of outlier drugs", fontsize = 15, fontweight = "bold")
    subfigs = fig.subfigures(nrows = 2, ncols = 1)

    subfigs[0].suptitle("Well-predicted drugs", fontweight = "bold")
    axs = subfigs[0].subplots(nrows = 1, ncols = 3)
    for j, ax in enumerate(axs.flat):
        ax.grid(True)
        ax.set_title(negative_skew[j])
        ax.set_xlim((-0.5,20))
        ax.set_ylim((0,180))
        ax.hist(drug[negative_skew[j]])
        ax.set_xlabel("Drug sensitivity (AUC)")

    subfigs[1].suptitle("Badly-predicted drugs", fontweight = "bold")
    axs = subfigs[1].subplots(nrows = 1, ncols = 3)
    for j, ax in enumerate(axs.flat):
        ax.grid(True)
        ax.set_title(bad_drugs[j])
        ax.set_xlim((-0.5,20))
        ax.set_ylim((0,180))
        ax.hist(drug[bad_drugs[j]])
        ax.set_xlabel("Drug sensitivity (AUC)")

    if save:
        plt.savefig(out_path, bbox_inches = 'tight', dpi = 300)


### Functions useful for the deep-SHAP analysis


def convert_shap_to_gene(shap_values, gene_prot, proteins, shap_threshold = 0.9):
    """
    Order proteins by SHAP value absolute mean importance, then convert the list of proteins to set of genes.
    """
    # order proteins by shap value absolute mean importance
    df = pd.DataFrame({
        "mean_abs_shap": np.mean(np.abs(shap_values), axis=0), 
        "name": proteins
    })
    df = df.sort_values("mean_abs_shap", ascending=False)
    enriched_prot = df.loc[df.mean_abs_shap > df.mean_abs_shap.quantile(shap_threshold)].name

    # convert from protein to genes
    gene_set = gene_prot.loc[gene_prot["Uniprot"].isin(enriched_prot)].Gene_Symbol
    gene_set = set(gene_set)

    return(gene_set)












