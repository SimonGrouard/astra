### Import libraries


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import json
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from itertools import combinations


### Functions to have a ready-to-use dataset


def filter_na(cell_line, threshold = 0.01, verbose = False):
	"""
	Input: the quantified normalised protein expression dataframe
	Output: the dataframe, without the protein having in proportion > threshold NAs
	"""
	na_per_prot = cell_line.isna().sum() # the number of NAs every protein has (length = number of proteins)
	protein_indexes = na_per_prot[na_per_prot < (len(cell_line)*threshold)].index # proteins having less than threshold NAs in proportion

	if verbose: 
		print(f'{len(na_per_prot)} -> {len(protein_indexes)} = reduction of {100-100*len(protein_indexes)/len(na_per_prot):.0f}% in number of proteins')

	return(cell_line[protein_indexes])

def finalise_proteomic(proteomic, drug, threshold = 0.01, verbose = False):
	"""
	Output: the proteomic dataset, ready to be used for ML purposes.
	Remove proteins thanks to the filter_na function + Remove cell lines not common to the drug dataset, and rename them.
	If threshold = 0.01 (or less) in filter_na, then we remove all proteins that have more than 0.01% NA values, which means in our case remove proteins with 1 or more NA.
	"""
	# for proteomic proteins
	proteomic = proteomic.rename_axis("Protein uniprot name", axis = 1)
	proteomic = filter_na(proteomic, threshold = threshold, verbose = verbose)

	# for proteomic cell lines
	cell_filtered = proteomic.index.str.split("_").str[0]
	proteomic.index = cell_filtered
	new_proteomic = proteomic.loc[drug.index]
	other_proteomic = proteomic.loc[~cell_filtered.isin(drug.index)]

	return(new_proteomic, other_proteomic)

def finalise_drug(drug, min_nb_cell = 260, replace_na = False):
	"""
	Output: the drug dataset, ready to be used for ML purposes.
	Remove drugs which have less than min_nb_cell cell lines they have quantified data (non NA).
	min_nb_cell = 260 makes all drugs kept have between 260 and 299 cell lines, and remove less than 30% of the drugs in total.
	"""
	drug = drug.rename_axis("Drug name", axis = 1)
	drug = drug.loc[:, drug.count() > min_nb_cell]

	if replace_na: # for the deep learning part, replace na by -1 to skip gradient update when NA
		drug = drug.fillna(-1)

	return(drug)


### Functions useful for any method


def c_index(observed, predicted):
	"""
	Compute the concordent index (no censored data though) between the predicted and observed values.
	"""
	all_pairs = list(combinations(range(len(observed)), 2)) # all possible pairwise comparisons in the vector
	concordent = 0

	for i, j in all_pairs:
	    if (observed[i] > observed[j] and predicted[i] > predicted[j]) or (observed[i] < observed[j] and predicted[i] < predicted[j]):
	        concordent += 1

	    elif predicted[i] == predicted[j]:
	        concordent += 0.5

	return(concordent/len(all_pairs))

def compute_score(observed, predicted):
	"""
	Returns the Pearson's correlation, RMSE and concordent index of the observed vs predicted values for a given method.
	"""
	corr = np.corrcoef(predicted, observed)[0,1] # Pearson's correlation 
	rmse = np.sqrt(mean_squared_error(predicted, observed)) # rmse
	cindex = c_index(predicted, observed) # c-index
	all_metrics = (corr, rmse, cindex)

	return(all_metrics)


### Train and tune Random Forest models


class RandomForestReg:
	"""
	Use the Random Forest regressor model to investigate the proteomic / drug dataset.
	"""
	def __init__(self, X_train, y_train, X_test, y_test, parameters = {}):
		self.xtrain = X_train
		self.ytrain = y_train 
		self.ytrain_univ = y_train

		self.xtest = X_test 
		self.ytest = y_test 
		self.ytest_univ = y_test

		self.parameters = parameters

	def tune_univariate(self, tune = False):
		"""
		Input: The training set, one of the drug training set columns (ie one drug), set of hyperparameters to tune the RF.
		Output: A fitted RF model.
		Before fitting the RF, the algorithm removes the cell lines in X_train and y_train_univ that have NAs in y_train_univ, in accordance with min_nb_cell in finalise_drug.
		the default parameters value is an empty dictionary, making GridSearch on 3 folds of the default RF values.

		TO DO: n_jobs = -1 for both the construction of the RF and its tuning with gridsearch, but it may be better optimised than taking all processors for both tasks.
		"""    
		df_train = pd.concat([self.xtrain, self.ytrain_univ], axis = 1).dropna(axis = 0)

		if tune:
			rf_reg = GridSearchCV(RandomForestRegressor(n_jobs = -1, random_state = 42),
		                        self.parameters, cv = 3, n_jobs = -1)
		else:
			rf_reg = RandomForestRegressor(n_jobs = -1, random_state = 42, **self.parameters)
		
		rf_reg.fit(df_train.iloc[:,:-1], df_train.iloc[:,-1])
	    
		return(rf_reg)

	def predict_score(self, rf_reg):
		"""
		Before fitting the RF, the algorithm removes the cell lines in X_test and y_test_univ that have NAs in y_train_univ, in accordance with min_nb_cell in finalise_drug.
		"""
		df_test = pd.concat([self.xtest, self.ytest_univ], axis = 1).dropna(axis = 0)
		preds = rf_reg.predict(df_test.iloc[:,:-1])

		# compute the different score metrics to evaluate the test set
		all_metrics = compute_score(df_test.iloc[:,-1], preds)

		return(all_metrics)

	def fit_predict(self, nb_drugs = 1, tune = False, verbose = 2):
		"""
		Use all the functions defined above to fit/tune/score a RF model.

		TO DO: even though there's already parallelisation on the construction of the RF, and its tuning with gridsearch, 
			   parallelisation can be also applied instead of using the for loop, as all drugs are independent
		"""
		if verbose > 0:
			start_time = time.time()

		all_score = []
		best_parameters = {}
		for i in range(nb_drugs):

			if verbose > 1 and i % 50 == 0:
				print(f"Drug number: {i}")
				sub_start_time = time.time()

			self.ytrain_univ = self.ytrain.iloc[:,i]
			self.ytest_univ = self.ytest.iloc[:,i]

			rf_reg = self.tune_univariate(tune = tune)

			if tune:
				best_param = rf_reg.best_params_
				best_parameters[self.ytest.columns[i]] = best_param

			score = self.predict_score(rf_reg)
			all_score.append(score)

			if verbose > 1 and i%50 == 0:
				print("--- %s seconds ---" % (time.time() - sub_start_time))

		all_score = pd.DataFrame.from_records(all_score)
		all_score.columns = ["Pearson's r", "RMSE", "C-index"]
		all_score.index = self.ytest.columns[:nb_drugs]
		all_score = all_score.round(3)

		if verbose > 0:
			print("--- %s seconds ---" % (time.time() - start_time))

		if tune:
			return(all_score, best_parameters)

		return(all_score)

def plot_comparison_rf(score_default, score_tuning, save = False, out_path = "figures/comparison_rf.png"):
	"""
	Plot a comparison of the random forest tuned and not-tuned (fitted with default values) across the three metrics.
	"""
	label = ["Not-tuned random forest", "Tuned random forest"]
	score = [score_default, score_tuning]
	xlim = [(-0.3, 0.7), (0, 4), (0.35, 0.75)]

	fig = plt.figure(constrained_layout = True, figsize = (12,12))
	fig.suptitle("Performance comparison between the not-tuned and tuned random forest", fontsize = 15, fontweight = "bold")
	subfigs = fig.subfigures(nrows = 2, ncols = 1)

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




if __name__ == "__main__":
	proteomic = pd.read_csv("data/cell_line_quant.csv", index_col = 0)
	print(proteomic.shape)
	drug = pd.read_csv("data/drug_sensitivity.csv", index_col = 0)
	print(drug.shape, "\n")

	## finalise the proteomic and drug datasets
	proteomic = finalise_proteomic(proteomic, drug, threshold = 0.01, verbose = True)
	print(proteomic.shape)
	drug = finalise_drug(drug)
	print(drug.shape, "\n")

	## train_test_split, in accordance with my data pipeline
	X_train, X_test, y_train, y_test = train_test_split(proteomic, drug, test_size=0.2, random_state=42)
	print(X_train.shape)
	print(y_train.shape)
	#X_train.to_csv("data/train_test_datasets/xtrain.csv")
	#y_train.to_csv("data/train_test_datasets/ytrain.csv")
	print(X_test.shape)
	print(y_test.shape, "\n")
	#X_test.to_csv("data/train_test_datasets/xtest.csv")
	#y_test.to_csv("data/train_test_datasets/ytest.csv")

	## train single RF - for more, do it yourself, takes a long time though
	start_time = time.time()

	rf_reg = RandomForestReg(X_train, y_train, X_test, y_test, {})
	score = rf_reg.fit_predict(nb_drugs = 1, tune = False, verbose = 1)

	print("--- %s seconds ---" % (time.time() - start_time))
	print(score)

	## compare tuned and not-tuned RF
	score_default = pd.read_csv("ML_output/RF/score_tuning.csv", index_col = 0)
	score_tuning = pd.read_csv("ML_output/RF/score_final.csv", index_col = 0)
	plot_comparison_rf(score_default, score_tuning, save = True)
	plt.show()










