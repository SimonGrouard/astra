### Import libraries 


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import random


### Function to create dataframe


def create_df(df, out_path = "data/cell_line_quant.csv"):
	"""
	Input: the original cell line csv file,
	ie table S2 from the 'Quantitative Proteomics of the Cancer Cell Line Encyclopedia'
	Saves the filtered dataframe with only the quantitative data
	"""
	columns = list(df.columns)
	filtered = list(filter(lambda array: "_TenPx" in array, columns)) # only the quant columns
	cell_line = df[filtered].T
	cell_line.columns = df.Uniprot
	cell_line = cell_line.rename_axis("Protein uniprot name", axis = 1)
	cell_line = cell_line.rename_axis("Cell line name", axis = 0)

	to_drop = ['SW948_LARGE_INTESTINE_TenPx11', 'CAL120_BREAST_TenPx02', 'HCT15_LARGE_INTESTINE_TenPx30']
	cell_line = cell_line.drop(to_drop)

	cell_line.to_csv(out_path, header = "Protein uniprot name")


### Functions to investigate missing values


def plot_na(cell_line, save = False, out_path  = "figures/distrib_nas_full_dataset.png"):
	"""
	Input: the quantified normalised protein expression dataframe
	Plot the distribution of NAs of every protein, and of every cell line
	"""
	na_per_prot = cell_line.isna().sum() # the number of NAs every protein has (length = number of proteins)
	na_per_line = cell_line.T.isna().sum() # the number of NAs every cell line has (length = 378)
	data = [na_per_prot, na_per_line]
	labels = ["Number of NAs of every protein", "Number of NAs of every cell line"]
	colours = ['b', 'g']

	fig, axs = plt.subplots(1, 2, figsize = (18,8), constrained_layout = True)
	fig.suptitle("Distributions of NAs across proteins and cell lines", fontweight = "bold")
	for i, ax in enumerate(axs.flat):
		ax.grid(True)
		ax.set_title(labels[i])
		ax.hist(data[i], bins = 12, facecolor = colours[i], alpha = 0.75)

	if save:
		fig.savefig(out_path, bbox_inches = 'tight', dpi = 300)

def filter_na(cell_line, threshold = 0.2, verbose = True):
	"""
	Input: the quantified normalised protein expression dataframe
	Output: the dataframe, without the protein having in proportion > threshold NAs
	"""
	na_per_prot = cell_line.isna().sum() # the number of NAs every protein has (length = number of proteins)
	protein_indexes = na_per_prot[na_per_prot < (len(cell_line)*threshold)].index # proteins having less than threshold NAs in proportion

	if verbose: 
		print(f'{len(na_per_prot)} -> {len(protein_indexes)} = reduction of {100-100*len(protein_indexes)/len(na_per_prot):.0f}% in size')

	return(cell_line[protein_indexes])

def plot_varying_size_dataset(cell_line, save = False, out_path = "figures/plot_varying_size_na.png"):
	"""
	Input: the quantified normalised protein expression dataframe
	Use filter_na for 100 threshold (from 0 to 1), and plot the varying size of the remaining dataframes after filtering for all thresholds
	"""
	thresholds = np.linspace(0, 1, 101)
	thresh_proportions = []
	for thresh in thresholds:
	    cell_line_filtered = filter_na(cell_line, verbose = False, threshold = thresh)
	    size_remaining = round(100 * cell_line_filtered.shape[1] / cell_line.shape[1], 1) # keep size_remaining % of the proteins
	    thresh_proportions.append(size_remaining)

	x_axis = [round(100*thresh) for thresh in thresholds]

	plt.figure(constrained_layout = True, figsize = (6,4))
	plt.tight_layout(rect=[0, 0.03, 1, 0.95])
	plt.suptitle("Proportion of proteins kept after filtering proteins that had more NAs than the threshold", fontweight = "bold")
	plt.xlabel("Varying threshold (%)")
	plt.ylabel("Proportion of proteins kept (%)")
	plt.grid(True)
	plt.bar(x_axis, thresh_proportions)

	if save:
		plt.savefig(out_path, bbox_inches = 'tight', dpi = 300)


### Function to investigate non-missing values


def plot_hist_values(cell_line_filtered, save = False, out_path = "figures/plot_hist_values.png"):
	"""
	Simple histogram of all the values from the dataset
	"""
	plt.figure(constrained_layout = True, figsize = (6,4))
	plt.xlabel("Normalised protein abundance")
	plt.title("Distribution of protein abundance", fontweight = "bold")
	plt.grid(True)
	plt.hist(cell_line_filtered.to_numpy().flatten(), bins = 30, range = (-5, 5))

	if save:
		plt.savefig(out_path, bbox_inches = 'tight', dpi = 300)

def plot_log_log_absolute_distrib(cell_line_filtered, verbose = True, save = False, out_path = "figures/plot_log_log_absolute_distrib.png"):
	"""
	Plot the log-log distribution of the cell line dataset, in absolute value
	It is much more understandable than the regular histogram plotted by the plot_hist_values function.
	"""
	# Define the thresholds, all powers of 10
	min_val = abs(cell_line_filtered).min().min()
	max_val = abs(cell_line_filtered.max().max())
	if verbose:
		print(f"The minimum and maximum protein absolute values of the dataset are, respectively, {format(min_val, '.2e')} and {round(max_val,1)}")

	min_10 = round(np.log10(min_val)) # round to nearest power of 10
	max_10 = round(np.log10(max_val))
	thresholds = list(reversed([10**i for i in range(min_10-1, max_10+1)]))

	# Find the proportion of values in the dataset with an absolute value lower than each threshold
	def small_value(elem, threshold):
	    if abs(elem) < threshold:
	        return(1)
	    return(0)

	data_shape = cell_line_filtered.shape
	number_nans = cell_line_filtered.isna().sum().sum()
	data_size = data_shape[0]*data_shape[1] - number_nans
	num_values = []
	for thresh in thresholds:
	    small_valued_data = cell_line_filtered.applymap(lambda elem: small_value(elem, thresh)) # is this value smaller than the threshold
	    num_value = 100 * small_valued_data.sum().sum() / data_size # proportion of the number of values lower than the threshold
	    num_values.append(num_value)

	# Plot the result
	plt.figure(constrained_layout = True, figsize = (6,4))
	plt.suptitle("Log-log distribution of protein quantification, in absolute value", fontweight = "bold")
	plt.xlabel("Threshold (log scale)")
	plt.ylabel("% (log scale)")
	plt.grid(True)
	plt.bar(range(len(thresholds)), num_values, align = "center", log = True)
	plt.xticks(range(len(thresholds)), thresholds)

	if save:
		plt.savefig(out_path, bbox_inches = 'tight', dpi = 300)

def plot_random_boxplot(cell_line_filtered, shape = (2, 2), seed = 23, save = False, out_path = "figures/plot_random_boxplot.png"):
	"""
	Subplot a certain number of boxplot of given random samples from the data
	"""
	number_boxplots = shape[0]*shape[1]
	random.seed(seed)
	random_prot = random.sample(range(cell_line_filtered.shape[0]), number_boxplots)

	fig, axs = plt.subplots(*shape, figsize = (10,10), constrained_layout = True)
	fig.suptitle("Boxplot of different cell line samples", fontweight = "bold")
	for i, ax in enumerate(axs.flat):
	    row = cell_line_filtered.iloc[random_prot[i]]
	    row_filtered = row[~np.isnan(row)]
	    
	    ax.grid(True)
	    ax.set_title(cell_line_filtered.index[random_prot[i]])
	    ax.boxplot(row_filtered)
	    ax.set_ylabel("Proteomic abundance")

	if save:
		plt.savefig(out_path, bbox_inches = 'tight', dpi = 300)


### Functions to investigate the correlation in the data


def never_run_this(self, verbose = True, save = False):
	"""
	Never run this function, unless you want to wait 30 minutes for it to run or crash
	It tries to improve the visualisation of the protein correlation heatmap, where we cannot see anything (because of the big size)
	Thus, it decomposes this 7515*7515 heatmap into 64 smaller bits (and then plots them side by side, in square)
	"""
	answer = input("Do you really want to run that?? It takes half an hour... To continue, type 'yes'.")

	if answer == "yes":

		print("You are crazy dude... But let's do it!")

		fig, axs = plt.subplots(8, 8, figsize = (40,40)) # figsize to change the whole size

		x_prev = 0
		y_prev = 0
		for i, ax in enumerate(axs.flat):
			if verbose:
				print(i)

			x = 1000*((i+1)%8) if 1000*((i+1)%8) != 0 else len(self.corr)
			y = 1000*((i//8)+1) if 1000*((i//8)+1) != 8000 else len(self.corr)

			data = self.corr.iloc[x_prev:x, y_prev:y]
			sns.heatmap(data, ax = ax)

			x_prev = x if x != len(self.corr) else 0
			y_prev = y if (i+1)//8 - i//8 == 1 else y_prev

		if save:
			fig.savefig('figures/all_heatmaps.png', bbox_inches = 'tight', dpi = 300)

def find_corr_proteins(corr, corr_thresh = 0.945, common_feat_thresh = 2, high_only = False):
	"""
	If common_feat_thresh = 2, we are sure that every protein on the reduced dataframe has at least one other protein correlated > corr_threshold.
	If common_feat_thresh > 2, proteins on the reduced dataframe have, in the entire dataset, common_feat_thresh proteins correlated > corr_threshold.
	However, nothing ensures that these common_feat_thresh proteins will also be in the reduced dataframe.
	Indeed, they may themselves have less than common_feat_thresh proteins they are correlated to, therefore not chosen for the reduced dataframe.
	"""
	corr2 = abs(corr[abs(corr) > corr_thresh]) # only keep the correlations that are above the threshold
	corr2 = corr2[corr2.count() >= common_feat_thresh] # all the proteins that have more than thresh other correlated proteins
	if high_only: # then you only show the correlations of the chosen proteins that are above corr_thresh
		corr2 = corr2[corr2.index]
	else: # then you show all the correlations of the chosen proteins
		corr2 = abs(corr.loc[corr2.index, corr2.index]) # we reduce the proteins to these ones

	assert len(corr2) > 0, "corr_thresh or/and common_feat_thresh are too high, so they return an empty reduced correlation matrix (no protein satisfies both thresholds)"

	return(corr2)

def plot_reduced_heatmap(reduced_corr, save = False, out_path = "figures/reduced_heatmap"):
	"""
	Plot a reduced heatmap of the correlation matrix. 
	Good visualisations with the proteomic data are (corr_thresh = 0.95, common_feat_thresh = 2) and (corr_thresh = 0.935, common_feat_thresh = 3).
	So, common_feat_thresh ensures not to have an empty matrix mathematically, but in practice a high common_feat_thresh gives a better dataframe.
	To understand more about the arguments, see the find_corr_proteins function.
	"""
	plt.figure(constrained_layout = True, figsize = (6,5))
	plt.suptitle("Reduced heatmap of highly correlated proteins (in absolute value)", fontweight = "bold")
	sns.heatmap(reduced_corr, cmap = "YlGnBu")

	if save:
		plt.savefig(out_path + f".png", dpi = 300, bbox_inches = 'tight')

class ReducedCorrelation:
	"""
	Investigate the correlation between the proteins that are highly correlated with other proteins
	"""
	def __init__(self, reduced_corr):
		self.reduced_corr = reduced_corr

	def add_group_number(self, groups, group_number, row, column, previous_row, previous_column, c, r, previous_c, previous_r):
	    """
	    Sub-function for the create_adjacent_groups function.
	    Choose what group this pair of proteins will be a part of.
	    Proteins are in common groups if they are adjacent in a column or in a row with another high correlated value.
	    Not when they are adjacent in both though, because this means they would not have any protein in common.
	    """
	    if previous_c > r and not np.isnan(self.reduced_corr.loc[row, previous_column]): # adjacent in column
	        groups[(row, column)] = groups[(row, previous_column)]
	        
	    elif r > 0 and c > previous_r and not np.isnan(self.reduced_corr.loc[previous_row, column]): # adjacent in row
	        groups[(row, column)] = groups[(previous_row, column)]

	    else: # new group to start
	        groups[(row, column)] = group_number
	        group_number += 1
	        
	    return(groups, group_number)

	def get_final_groups(self, groups, group_number):
	    """
	    Sub-function for the create_adjacent_groups function
	    Create a dictionary, each key being a group number and its value a set of proteins of that group
	    """
	    all_groups = {k: set() for k in range(group_number)} # don't use dict.fromkeys() as it will create sets referenced to each other
	    
	    for proteins in groups:
	        val = groups[proteins]
	        
	        if val is not None:
	            all_groups[val].update(proteins)
	    
	    return(all_groups)

	def create_visually_adjacent_groups(self):
	    """
	    Create the visually adjacent groups in a reduced correlation matrix.
	    Visually adjacent means high correlations that are adjacent either in row, or in column (not both).
	    It is entirely dependent on how the proteins are ordered in the reduced matrix.
	    """
	    # First, create a dictionnary where every pair of highly correlated proteins is mapped to their group
	    previous_c, previous_column = 0, ""
	    previous_r, previous_row = 0, ""

	    groups = dict.fromkeys(product(self.reduced_corr, self.reduced_corr))
	    group_number = 0

	    for r, row in enumerate(self.reduced_corr):
	        for c, column in enumerate(self.reduced_corr):
	            if c > r: # since it's symmetrical, run only through the strict upper diagonal
	                val = self.reduced_corr.loc[row, column]

	                if not np.isnan(val): # then correlation > threshold
	                    groups, group_number = self.add_group_number(groups, group_number, row, column, previous_row, 
	                                                            previous_column, c, r, previous_c, previous_r)

	            previous_c, previous_column = c, column
	        previous_r, previous_row = r, row
	    
	    # Return the visually adjacent protein sets
	    all_groups = self.get_final_groups(groups, group_number)
	    
	    return(all_groups)

	def compute_groups(self):
		"""
		Find the proteins that can be grouped in block together because they have high "chaining" correlations in common.
		"""
		all_groups = []

		for r, row in enumerate(self.reduced_corr):
			for c, column in enumerate(self.reduced_corr):

				val = self.reduced_corr.loc[row, column]
				if c > r and not np.isnan(val):

	                # see the different already existing sets from all_groups in which there are common proteins with either row or column
					intersect = []
					for index, prot_set in enumerate(all_groups):
						if row in prot_set or column in prot_set: # if protein row or column is in an already existing set from all_groups
							intersect.append(index)

	                # create the new set with all the common proteins from row, column and the already existing sets from all_groups
					new_group = set() 
					for i in intersect: # there can be at most 2 different sets in intersect
						new_group.update(all_groups[i])
					new_group.update([row, column])

	                # delete the previous sets from all_groups that are now merged together in new_group
					for index in sorted(intersect, reverse=True):
						del all_groups[index]

					all_groups.append(new_group)
	                
		return(all_groups)

	def plot_grouped_heatmap(self, verbose = True, random = False, square = False, all_corr = None, save = False, out_path = "figures/reduced_heatmap"):
		"""
		Plot the reduced heatmap which has been blocked by correlated protein groups with compute_group
		"""
		all_groups = self.compute_groups()
		if verbose:
			print("The correlated groups are the following:")
			print(*all_groups)

		if random:
			new_columns = [prot for group in all_groups for prot in group]

		else: # order the columns in each group from most correlation to least correlation
			nans_per_prot = self.reduced_corr.apply(np.isnan, axis = 1).sum(axis = 1)
			new_columns = []
			for group in all_groups:
				ordered_col = nans_per_prot[group].sort_values()

				if square: # visualise the correlations in a sparse square
					even_index = ordered_col.iloc[::2]
					odd_index = ordered_col.iloc[1::2].sort_values(ascending = False)
					ordered_col = even_index.append(odd_index)

				ordered_col = ordered_col.index
				new_columns.extend(ordered_col)

		if all_corr is not None: # then you show all correlations of this subset of proteins, whether they are low or high
			new_reduced_corr = abs(all_corr.loc[new_columns, new_columns])

		else:
			new_reduced_corr = self.reduced_corr.loc[new_columns, new_columns]

		plot_reduced_heatmap(new_reduced_corr, save = save, out_path = out_path)


### Class to investigate protein isoforms


class GeneIsoforms:
	"""
	Investigate the protein isoforms of the proteomic cell line dataset
	"""
	def __init__(self, df):
		self.df = df

	def find_gene_isoforms(self, column = "Gene_Symbol"):
		"""
		Input: the dataframe, and the column where there are duplicates
		Output: the list of gene containing multiple variants
		"""
		elements = set()
		duplicate = set()

		for elem in self.df[column]:
		    if elem in elements and isinstance(elem, str): # if not a string, then nan
		        duplicate.add(elem)
		    elements.add(elem)

		return(list(duplicate))

	def find_protein_isoforms(self, gene, column = None):
		"""
		Input: the list of gene isoforms
		Apply this function to every element of a list of gene isoforms to get the corresponding protein isoforms
		"""
		isoforms = self.df.loc[self.df["Gene_Symbol"] == gene]

		if column is not None:
			return(isoforms[column])

		return(isoforms)

	def count_na_single_isoform(self, gene):
		"""
		Input: the list of gene isoforms 
		Apply this function to every element of a list of gene isoforms to get for each protein the number of NAs
		"""
		isoform = self.find_protein_isoforms(gene)
		columns = self.df.columns
		filtered = list(filter(lambda array: "_TenPx" in array, columns))

		isoform = isoform[filtered] # only the protein expression columns
		nans = isoform.apply(np.isnan, axis = 1).sum(axis = 1) # number of nans for the isoforms of the given gene
		nans = nans.sort_values() # each protein isoforms of a gene are ordered by number of NAs

		return(nans)


	def count_isoforms(self, verbose = True):
		"""
		Input: the dataframe
		Output: the number of protein isoforms
		"""
		dup_list = self.find_gene_isoforms("Gene_Symbol")
		protein_variants = map(lambda gene: self.find_protein_isoforms(gene, "Protein_Id"), dup_list) # all protein isoforms are here

		count = sum(map(len, protein_variants)) # number of protein variants

		if verbose:
			print(f"Total number of protein isoforms: {count} ({100*count/self.df.shape[0]:.1f}% of the dataset)")
			print(f'Total number of genes involved: {len(dup_list)}')

		return(None)

	def count_na_isoforms(self):
		"""
		Input: the dataframe
		Output: the number of missing values per protein isoform
		"""
		dup_list = self.find_gene_isoforms("Gene_Symbol")
		nans = list(map(lambda gene: self.count_na_single_isoform(gene), dup_list))
		
		return(nans)

	def plot_isoform_gene(self, save = False, out_path = "figures/distrib_number_isoform_per_gene.png"):
		"""
		Plot the distribution of number of isoforms per gene
		"""
		nans = self.count_na_isoforms()
		num_dup = []
		for elem in nans:
		    num_dup.append(len(elem))

		plt.figure(constrained_layout = True, figsize = (6,4))
		plt.xlabel("Number of protein isoforms for a given gene")
		plt.title("Distribution of the number of protein isoforms per gene", fontweight = "bold")
		plt.grid(True)
		plt.hist(num_dup)

		if save:
			plt.savefig(out_path, bbox_inches = "tight", dpi = 300)

	def plot_na_isoforms(self, save = False, out_path = "figures/gene_variants_missing.png"):
		"""
		Plot the investigation of missing values from isoforms
		"""
		nans = self.count_na_isoforms()
		differences, mins, maxs = [], [], []
		for elem in nans:
		    differences.append(elem.iloc[-1]-elem.iloc[0])
		    mins.append(elem.iloc[0])
		    maxs.append(elem.iloc[-1])

		data = [mins, maxs, differences]
		colors = ['b', 'r', 'g']
		labels = ["Missing values of the protein isoform with the least NAs for every gene",
		          "Missing values of the protein isoform with the most NAs for every gene",
		          "Differences between both numbers (max - min)"]

		fig = plt.figure(constrained_layout = True, figsize = (12,12))
		fig.suptitle("Investigation in the gene variants difference in missing values", fontsize = 20, fontweight = "bold")
		subfigs = fig.subfigures(nrows = 3, ncols = 1)

		for i in range(3):
		    subfigs[i].suptitle(labels[i], fontweight = "bold")
		    axs = subfigs[i].subplots(nrows = 1, ncols = 2)
		    
		    axs[0].grid(True)
		    axs[0].hist(data[i], 20, facecolor = colors[i], alpha = 0.75)
		    axs[1].boxplot(data[i], notch = True, vert = False)

		if save:
			fig.savefig(out_path, bbox_inches = "tight", dpi = 300)





if __name__ == "__main__":
	
	df = pd.read_csv("data/protein_quant_normalized.csv")
	print("\n", f"Shape of the dataframe: {df.shape}", "\n\n")
	
	## create dataframe with only protein expression
	out_path  = "data/cell_line_quant.csv"
	create_df(df, out_path)

	## investigate the NAs
	cell_line = pd.read_csv(out_path, index_col = 0)
	plot_na(cell_line, save = False, out_path = "figures/distrib_nas_full_dataset.png") # before removing NAs
	cell_line_filtered = filter_na(cell_line, threshold = 0.2, verbose = True)
	plot_na(cell_line_filtered, save = False, out_path = "figures/distrib_nas_filtered_dataset.png") # after removing proteins with more than 20% NAs
	plot_varying_size_dataset(cell_line, save = False, out_path = "figures/plot_varying_size_na.png")
	plt.show()
	print("\n\n")

	## investigate the non-NA values
	plot_hist_values(cell_line_filtered, save = False, out_path = "figures/plot_hist_values.png")
	plot_log_log_absolute_distrib(cell_line_filtered, verbose = True, save = False, out_path = "figures/plot_log_log_absolute_distrib.png")
	plot_random_boxplot(cell_line_filtered, shape = (2, 2), seed = 23, save = False, out_path = "figures/plot_random_boxplot.png")
	plt.show()
	print("\n\n")

	## investigate the correlation
	corr = cell_line_filtered.corr()

	# reduce the number of proteins (correlation > 0.935, common proteins highly correlated >= 3)
	reduced_corr = find_corr_proteins(corr, corr_thresh = 0.935, common_feat_thresh = 3, high_only = True)
	out_path = "figures/heatmaps/corr_0.935_common_3/"
	print(f"There are {reduced_corr.shape[0]} proteins.")
	plot_reduced_heatmap(reduced_corr, save = False, out_path = f"{out_path}highonly")
	investigate_corr = ReducedCorrelation(reduced_corr)
	print("Visually, the correlated groups are (first heatmap):", investigate_corr.create_visually_adjacent_groups())
	print("But in reality, after ordering the proteins (second and third heatmap)")
	investigate_corr.plot_grouped_heatmap(save = False, out_path = f"{out_path}grouped_highonly")
	investigate_corr.plot_grouped_heatmap(all_corr = corr, verbose = False, random = True, save = False, out_path = f"{out_path}grouped_all")
	plt.show()
	print("\n\n")
	
	# specific groups of proteins (correlation > 0.95, common proteins highly correlated >= 2)
	reduced_corr = find_corr_proteins(corr, corr_thresh = 0.95, common_feat_thresh = 2, high_only = True)
	out_path = "figures/heatmaps/corr_0.95_common_2/"
	print(f"There are {reduced_corr.shape[0]} proteins.")
	plot_reduced_heatmap(reduced_corr, save = False, out_path = f"{out_path}highonly")
	investigate_corr = ReducedCorrelation(reduced_corr)
	investigate_corr.plot_grouped_heatmap(random = True, save = False, out_path = f"{out_path}grouped_highonly")
	investigate_corr.plot_grouped_heatmap(all_corr = corr, verbose = False, random = True, save = False, out_path = f"{out_path}grouped_all")
	plt.show()
	print("\n\n")

	# allow more proteins (correlation > 0.9, common proteins highly correlated >= 5) ==> Different parameters truly give different visualisations
	reduced_corr = find_corr_proteins(corr, corr_thresh = 0.9, common_feat_thresh = 5, high_only = True)
	out_path = "figures/heatmaps/corr_0.9_common_5/"
	print(f"There are {reduced_corr.shape[0]} proteins.")
	investigate_corr = ReducedCorrelation(reduced_corr)
	investigate_corr.plot_grouped_heatmap(verbose = True, random = True, save = False, out_path = f"{out_path}grouped1_highonly")
	investigate_corr.plot_grouped_heatmap(verbose = False, random = False, square = False, save = False, out_path = f"{out_path}grouped2_highonly")
	investigate_corr.plot_grouped_heatmap(verbose = False, random = False, square = True, save = False, out_path = f"{out_path}grouped3_highonly")
	investigate_corr.plot_grouped_heatmap(verbose = False, random = True, all_corr = corr, save = False, out_path = f"{out_path}grouped_all")
	plt.show()
	print("\n\n")

	# too many proteins (correlation > 0.94, common proteins highly correlated >= 2) ==> The graph becomes quite messy
	reduced_corr = find_corr_proteins(corr, corr_thresh = 0.94, common_feat_thresh = 2, high_only = True)
	out_path = "figures/heatmaps/corr_0.94_common_2/"
	print(f"There are {reduced_corr.shape[0]} proteins.")
	investigate_corr = ReducedCorrelation(reduced_corr)
	investigate_corr.plot_grouped_heatmap(save = False, out_path = f"{out_path}grouped_highonly")
	investigate_corr.plot_grouped_heatmap(verbose = False, all_corr = corr, random = True, save = False, out_path = f"{out_path}grouped_all")
	plt.show()
	print("\n\n")

	## investigate the protein isoforms (coming from the same genes)
	variants = GeneIsoforms(df)
	variants.count_isoforms(verbose = True)
	variants.plot_isoform_gene(save = False, out_path = "figures/distrib_number_isoform_per_gene.png")
	variants.plot_na_isoforms(save = False, out_path = "figures/gene_variants_missing.png")
	plt.show()










