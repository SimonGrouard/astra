### Import libraries 


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cmapPy.pandasGEXpress.parse_gct import parse
from scipy.stats import mannwhitneyu


### Useful functions to pre-process drug data


def preprocess_cancer_therapeutics(drug_meta, save = False, out_path = "data/drug_sensitivity.csv"):
	"""
	Transform the metadata file from the cancer therapeutics response portal into a useful drug sensitivity dataframe.
	"""
	drug = drug_meta.data_df.T
	drug = drug.set_index(drug_meta.col_metadata_df.cell_line_name)
	drug.columns = drug_meta.row_metadata_df.compound_name
	if not save:
		drug.columns = drug.columns.str.lower().str.replace(" ", "-")

	## replace cell line duplicates by their mean

	all_duplicates = drug.loc[drug.index.duplicated()]
	all_duplicate_names = list(set(all_duplicates.index))

	# take the mean of the duplicates
	averaged = [drug.loc[name].mean() for name in all_duplicate_names]

	# rename the duplicates
	drug.index = drug.index.where(~drug.index.duplicated(), drug.index + "_dp")

	# add the means and remove duplicates
	drug.loc[all_duplicate_names] = averaged
	drug.drop(drug.index[drug.index.str.endswith("_dp")])

	if save:
		drug.to_csv(out_path)

	return(drug)

def preprocess_genomics_drug(drug_meta1, drug_meta2):
	"""
	Transform the metadata files from the genomics of drug sensitivity cancer portal into a useful drug sensitivity dataframe.
	"""
	drug_meta = pd.concat([drug_meta1, drug_meta2], ignore_index = True, axis = 0)
	drug_meta["Cell line name"] = drug_meta["Cell line name"].str.replace("-", "")

	drug = drug_meta[["Drug name", "Cell line name", "AUC"]]
	drug = drug.pivot_table(values = "AUC", index = "Cell line name", columns = "Drug name", aggfunc = "first")
	drug.columns = drug.columns.str.lower().str.replace(" ", "-")

	return(drug)


### Functions to compare both datasets


def print_comparison(cell_filtered, drug_filtered1, drug_filtered2, common_cell1, common_cell2):
	"""
	Print comparisons between both datasets to see whether they could be individually useful, or even usefully combined.
	"""
	print(f"Number of proteomic cell lines: {len(cell_filtered)}")
	print(f"Number of drug1 cell lines: {len(drug_filtered1)}")
	print(f"Number of drug2 cell lines: {len(drug_filtered2)}\n")

	print(f"Number of drug1 and proteomic common cell lines: {len(common_cell1)}")
	print(f"Number of drug2 and proteomic common cell lines: {len(common_cell2)}\n")

	common_cell = common_cell1.union(common_cell2)
	print(f"Union of drugs and proteomic cell lines: {len(common_cell)}")
	print(f"==> Proportion of proteomic cell lines having drug results: {100*len(common_cell)/len(cell_filtered)}%")

def compare_correlation_sensitivity(drug1, drug2, drug_filtered1, drug_filtered2, verbose = True, save = False, out_path = "figures/drugs_correlation.png"):
	"""
	Compare both drug sensitivity datasets, by calculating their Pearson's r correlation for every common drug, and then boxplot these correlations.
	If most correlations are high, then we can combine both datasets. If not, they will have to be used separately.
	Spoiler from the boxplot: they are pretty low, so will unfortunately be used separately.
	"""
	drug_common_cell = drug_filtered1.intersection(drug_filtered2) 
	if verbose:
		print(f"Number of drug1 and drug2 common cell lines: {len(drug_common_cell)}")

	drug1_common = drug1.loc[drug_common_cell]
	drug2_common = drug2.loc[drug_common_cell]
	common_drug = set(drug1_common.columns).intersection(set(drug2_common.columns))
	if verbose:
		print(f"Number of common drugs: {len(common_drug)}\n")

	drug1_common = drug1_common[common_drug]
	drug2_common = drug2_common[common_drug]

	correlations = []
	for drug in drug1_common.columns:
	    both_columns = {f"{drug}_drug1": drug1_common[drug], f"{drug}_drug2": drug2_common[drug]}
	    correlation = pd.DataFrame(both_columns).corr().iloc[0, 1]
	    
	    correlations.append(correlation)

	plt.figure(constrained_layout = True, figsize = (6,4))
	plt.title("Boxplot of the Pearson's correlation between AUC values from common drugs of both datasets", fontweight = "bold")
	plt.grid(True)
	plt.boxplot(correlations, vert = False, notch = True)
	if save:
		plt.savefig(out_path, bbox_inches = 'tight', dpi = 300)

def plot_na_drugs(drug1, drug2, common_cell1, common_cell2, mannwhitney = False, save1 = False, out_path1 = "figures/drugs_distrib_NA.png", save2 = False, inout_path2 = "data/drug_sensitivity.csv"):
	"""
	Compare the distribution of number of NAs per drugs across cell lines in both datasets. 
	The dataset which has stochastically the least missing values is the one that will be used.
	"""
	drug1_reduced = drug1.loc[common_cell1]
	drug1_not_missing = drug1_reduced.count()
	drug2_reduced = drug2.loc[common_cell2]
	drug2_not_missing = drug2_reduced.count()

	# sort the data:
	drug1_not_missing_sorted = np.sort(drug1_not_missing)
	drug2_not_missing_sorted = np.sort(drug2_not_missing)
	drug_not_missing = [drug1_not_missing_sorted, drug2_not_missing_sorted]

	# calculate the proportional values of samples
	p1 = 100 * np.arange(len(drug1_not_missing_sorted)) / (len(drug1_not_missing_sorted) - 1)
	p2 = 100 * np.arange(len(drug2_not_missing_sorted)) / (len(drug2_not_missing_sorted) - 1)
	p = [p1, p2]

	# plot the CDF and boxplot of the distributions of missing values
	labels = ["Cumulative distribution function of number of non-missing values", "Boxplot of number of non-missing values"]

	fig = plt.figure(constrained_layout = True, figsize = (12,12))
	fig.suptitle("Comparison of both drug datasets distribution of NAs. \n\n           Cancer Therapeutics Response Portal.              Genomics of Drug Sensitivity in Cancer.\n", fontsize = 15, fontweight = "bold")
	subfigs = fig.subfigures(nrows = 2, ncols = 1)

	subfigs[0].suptitle(labels[0], fontweight = "bold")
	axs = subfigs[0].subplots(nrows = 1, ncols = 2)
	for i, ax in enumerate(axs.flat):
	    ax.grid(True)
	    ax.plot(drug_not_missing[i], p[i])
	    ax.set_xlim(0, 300)
	    ax.set_xlabel("Number of non missing-values in the dataset")
	    ax.set_ylabel("Proportion of drugs (%)")

	subfigs[1].suptitle(labels[1], fontweight = "bold")
	axs = subfigs[1].subplots(nrows = 1, ncols = 2)
	for i, ax in enumerate(axs.flat):
	    ax.grid(True)
	    ax.boxplot(drug_not_missing[i], notch = False)
	    ax.set_ylim(0, 300)
	    ax.set_ylabel("Number of quantified (non-missing) cell lines")

	if mannwhitney: # Unfortunately, I realised that makes no sense to use a statistical test here, since there’s nothing to infer about the total population 
		#(there is no total population, just the two datasets I already have). Therefore, the visualisations are enough.
		U1, p = mannwhitneyu(drug1_not_missing, drug2_not_missing, alternative = "greater")
		print("Mann–Whitney U test: \nNull: distribution underlying drug1 NAs is the same as the distribution underlying drug2 NAs \n\
Alternative: distribution underlying drug1 NAs is stochastically greater than distribution underlying drug2 NAs\n")
		print(f"The p-value given by the Mann–Whitney U test is: {p}")

	if save1:
		fig.savefig(out_path1, bbox_inches = "tight", dpi = 300)

	if save2: # only drug1 since it's the one I will use (because it has stochastically the least number of missing values)
		drug1 = pd.read_csv(inout_path2, index_col = 0)
		drug1_reduced = drug1.loc[common_cell1]
		drug1_reduced.to_csv(inout_path2)

		return(drug1_reduced)



if __name__ == "__main__":
	cell_line = pd.read_csv("data/cell_line_quant.csv", index_col = 0)
	cell_filtered = set(cell_line.index.str.split("_").str[0])

	## cancer therapeutics response portal
	drug1_meta = parse('data/raw_drug_sensitivities/drug_sensitivity.gct')
	drug1 = preprocess_cancer_therapeutics(drug1_meta)
	print(drug1.shape)

	drug_filtered1 = set(drug1.index)
	common_cell1 = drug_filtered1.intersection(cell_filtered)

	## genomics of drug sensitivity in cancer
	drug2_meta1 = pd.read_csv("data/raw_drug_sensitivities/drug_sensitivity1.csv")
	drug2_meta2 = pd.read_csv("data/raw_drug_sensitivities/drug_sensitivity2.csv")
	drug2 = preprocess_genomics_drug(drug2_meta1, drug2_meta2)
	print(drug2.shape)

	drug_filtered2 = set(drug2.index)
	common_cell2 = drug_filtered2.intersection(cell_filtered)

	## print comparisons between both datasets
	print_comparison(cell_filtered, drug_filtered1, drug_filtered2, common_cell1, common_cell2)

	## is the AUC from both datasets comparable? (which means we could combine both drug datasets)
	compare_correlation_sensitivity(drug1, drug2, drug_filtered1, drug_filtered2, verbose = True)
	plt.show()

	## they are not comparable ==> which one should I use?
	plot_na_drugs(drug1, drug2, common_cell1, common_cell2, mannwhitney = True)
	plt.show()






