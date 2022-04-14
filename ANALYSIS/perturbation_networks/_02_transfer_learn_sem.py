#!/bin/python

# transfer-learn the twin GCN model to predict FEP SEMs.

# import code to regenerate the twin GCN.
from _01_twin_gcn import *
import tensorflow.keras.backend as K_backend

# misc imports.
import os
import pickle 

import csv
from scipy import stats
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

# PCA:
from sklearn.decomposition import PCA
from sklearn import preprocessing

from rdkit.Chem import rdmolfiles, rdMolDescriptors, AllChem

# Mordred descriptors:
from mordred import Calculator, descriptors

# base models.
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor


tf.get_logger().setLevel('ERROR')
warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

def writeFEPSpaceResults(results_path, reference_path, output_path):
	"""
	Loads FEP-Space SEMs (as computed by SOMD); retrieves the SMILES for ligand1 and ligand2.
	This could be a lot faster, of course.
	"""

	print("Writing FEP-Space results..")
	# first load the references that contain SMILES information per perturbation in FEP-Space.
	fepspace_ref = []
	with open(reference_path, "r") as fepspace_refs:
		reader = csv.reader(fepspace_refs)
		for row in reader:
			fepspace_ref.append(row)

	fail_counter = 0
	success_counter = 0
	fepspace_smiles_sems = []
	with open(results_path, "r") as fepspace_results:
		reader = csv.reader(fepspace_results)
		next(reader) # skip header
		for row in reader:
			found = False
			pert = "_".join(row[0].split("_")[:-1]).replace("lig_", "")
			tgt = row[0].split("_")[-1]

			# also get the inverse pert.
			inv_pert = f"{pert.split('~')[1]}~{pert.split('~')[0]}"

			# now find the SMILES information in ref.
			for ref in fepspace_ref:

				if ref[0].replace("lig_", "") == pert and ref[1] == tgt:
					found = True
					fepspace_smiles_sems.append([ref[3], ref[4], row[-1], pert, tgt])
					break

				elif ref[0].replace("lig_", "") == inv_pert and ref[1] == tgt:
					found = True
					# found the inverse pert. just append the inverse of the inverse.
					fepspace_smiles_sems.append([ref[4], ref[3], row[-1], pert, tgt])
					break

			if not found:
				fail_counter += 1
			elif found:
				success_counter += 1

	print(f"Found {success_counter} SMILES references; unable to find {fail_counter}.")

	# now write to file.
	pd.DataFrame.from_records(fepspace_smiles_sems, columns=[
		"ligand1_smiles", 
		"ligand2_smiles", 
		"fepspace_sem",
		"ligand1_original_pert_name",
		"target"]).to_csv(
		output_path)
	print(f"Written file: {output_path}")

def modelTransfer(base_model, n_layers_remove, n_layers_add, dropout=False):
	"""
	Removes the last n layers from an existing NN; replaces them with n new layers
	leading up to a single neuron.
	"""
	# remove n layers by selecting the layer at the correct index.
	base_output = base_model.layers[-n_layers_remove].output
	n_neurons_base = base_model.layers[-n_layers_remove].units
	n_neurons_base = int(0.8*n_neurons_base)  # slightly decrease the first added layer wrt the last layer of base.

	for num_neurons in np.linspace(n_neurons_base, 1, n_layers_add, dtype=int):
		if num_neurons == 1:
			act = "linear"
		else:
			act = "relu"

		# add  dense layer.
		new_output = tf.keras.layers.Dense(activation=act, units=num_neurons)(base_output)

		# add dropout layers after each multidim layer.
		if dropout and num_neurons != 1:
			new_output = tf.keras.layers.Dropout(dropout)(new_output)

		extended_model = tf.keras.models.Model(inputs=base_model.inputs, outputs=new_output)

		

		# get ready for next iteration.
		base_output = extended_model.output

	# freeze the layers up to the extension.
	# Check https://www.tensorflow.org/tutorials/images/transfer_learning
	# and https://stackoverflow.com/questions/57569460/freezing-keras-layer-doesnt-change-sumarry-trainable-params
	for layer in extended_model.layers[:-n_layers_add]:
		layer.trainable = False

	print(extended_model.summary())

	return extended_model



def computeStats(y_true, y_pred):
	"""Returns r, mue and tau between two input arrays of equal length."""
	r, _ = stats.pearsonr(y_true, y_pred)
	mue = mean_absolute_error(y_true, y_pred)
	tau, _ = stats.kendalltau(y_true, y_pred)

	return round(r, 2), round(mue, 2), round(tau, 2)

def normaliseFeats(feats):
	"""Given an array of features,
	Returns a normalised DataFrame and stats for test set scaling."""

	feats_df = pd.DataFrame.from_records(feats)
	stat = feats_df.describe()
	stat = stat.transpose()

	def norm(x):
		return (x - stat['mean']) / stat['std']

	# Normalise and return separately.
	normed_data = norm(feats_df).fillna(0).replace([np.inf, -np.inf], 0.0)
	
	return normed_data, stat 


def reduce_features(training_data, pca_threshold=100):
    """Returns PCA reduced DataFrame according to a pca_threshold parameter.
    Original columns with the highest contribution to PCX are written to CSV."""
    
    # Initialise PCA object, keep 100 dimensions:
    PCA.__init__
    pca = PCA(n_components=pca_threshold)

    # Fit to and transform training set.
    train_post_pca = pd.DataFrame(pca.fit_transform(training_data))

    # Reset column names to PCX.
    pca_col = np.arange(1, len(train_post_pca.columns) + 1).tolist()
    pca_col = ['PC' + str(item) for item in pca_col]
    train_post_pca.columns = pca_col
    train_post_pca.index = training_data.index


    def recovery_pc(training_data, pca_threshold):

        # Normalise data.
        data_scaled = pd.DataFrame(preprocessing.scale(training_data), columns=training_data.columns)

        # Initialise PCA object, keep components up to x% variance explained:
        PCA.__init__
        pca = PCA(n_components=pca_threshold)
        pca.fit_transform(data_scaled)

        index = list(range(1, len(train_post_pca.columns) + 1))
        index = ['PC{}'.format(x) for x in index]

        return_df = pd.DataFrame(pca.components_, columns=data_scaled.columns, index=index)

        return return_df

    # Adapted from https://stackoverflow.com/questions/22984335/recovering-features-names-of-explained-variance-ratio-in
    # -pca-with-sklearn
    recovered_pc = recovery_pc(training_data, pca_threshold)

    # List of column names with highest value in each row.
    recovered_pc_max = recovered_pc.idxmax(axis=1)

    # Recovery 'PCX' indexing.
    pc_index = recovered_pc_max.index.tolist()

    # Write feature names to list.
    pc_feature = recovered_pc_max.values.tolist()

    # Write to DataFrame.
    recovered_pc_dict = {'PCX': pc_index, 'Highest contributing feature': pc_feature}
    recovered_pc_df = pd.DataFrame(recovered_pc_dict)
    
    return train_post_pca, recovered_pc_df, pca

def computeFeats(training_set, write_path):
	"""
	Given input training set; featurise using atom-pair fingerprinting, molecular properties
	and extended connectivity FPs. Concatenate the fingerprints and subtract properties, normalise 
	and PCA on the set and return the training set as well as the 
	normalisation/PCA objects to pickle (for test sets).

	Most of this code is based on https://github.com/michellab/hybrid_FEP-ML/blob/master/datasets/backend/features_X/calc_reduced_features.py
	"""
	print("Computing FPs for base models..")
	apfps, ecfps, props = [], [], []
		
	calc = Calculator(descriptors, ignore_3D=True)

	for liga, ligb in tqdm(zip(training_set.ligand1_smiles.values, 
							training_set.ligand2_smiles.values), total=len(training_set)):
		liga = Chem.MolFromSmiles(liga)
		ligb = Chem.MolFromSmiles(ligb)

		# ATOM-PAIR FPs
		apfp = list(rdMolDescriptors.GetHashedAtomPairFingerprint(liga, 256))
		for bit in list(rdMolDescriptors.GetHashedAtomPairFingerprint(ligb, 256)):
			apfp.append(bit)
		apfps.append(apfp)

		# EXTENDED CONNECTIVITY FPs
		ecfp = list(AllChem.GetMorganFingerprintAsBitVect(liga,1,nBits=1024))
		for bit in list(AllChem.GetMorganFingerprintAsBitVect(ligb,1,nBits=1024)):
			ecfp.append(bit)
		ecfps.append(ecfp)

		# MOLECULAR PROPERTIES
		liga_props = calc(liga).fill_missing(value=0)
		ligb_props = calc(ligb).fill_missing(value=0)
		dProps = np.array(list(ligb_props.values())) - np.array(list(liga_props.values()))
		dProps_col_names = ligb_props.keys()
		props.append(dProps)

	apfps_normed, apfp_stats = normaliseFeats(apfps)
	apfps_reduced, apfps_pc_importance, apfp_pca_obj = reduce_features(apfps_normed)

	ecfps_normed, ecfp_stats = normaliseFeats(ecfps)
	ecfps_reduced, ecfps_pc_importance, ecfp_pca_obj = reduce_features(ecfps_normed)


	props_normed, props_stats = normaliseFeats(props)
	props_normed.columns = dProps_col_names
	props_reduced, props_pc_importance, props_pca_obj = reduce_features(props_normed)

	# write stats/pca imps+objs to files.
	apfp_stats.to_csv(path_or_buf=write_path+"stats_apfp.csv", index=True)
	ecfp_stats.to_csv(path_or_buf=write_path+"stats_ecfp.csv", index=True)
	props_stats.to_csv(path_or_buf=write_path+"stats_props.csv", index=True)

	apfps_pc_importance.to_csv(path_or_buf=write_path+"pc_impacts_apfp.csv", index=True)
	ecfps_pc_importance.to_csv(path_or_buf=write_path+"pc_impacts_ecfp.csv", index=True)
	props_pc_importance.to_csv(path_or_buf=write_path+"pc_impacts_props.csv", index=True)

	pickle.dump(apfp_pca_obj, open(write_path+"pca_apfp.pkl","wb"))
	pickle.dump(ecfp_pca_obj, open(write_path+"pca_ecfp.pkl","wb"))
	pickle.dump(props_pca_obj, open(write_path+"pca_props.pkl","wb"))


	return apfps_reduced, ecfps_reduced, props_reduced


def trainBaseModels(apfp_set, ecfp_set, props_set, y_labels, write_path):
	"""Given input training sets, train both support vector machine and
	Random Forest regressors. Stores the fitted models."""
	print("Training base models:")
	for feat_set, name in zip([apfp_set, ecfp_set, props_set],
						["apfp", "ecfp", "props"]):
		print(name+"..")
		# Fit an SVM
		svr = SVR(verbose=False, gamma=1e-8)
		svr.fit(feat_set, y_labels)
		pickle.dump(svr, open(write_path+"fit_svr_"+name+".pkl", "wb"))

		# Fit a RF
		rf = RandomForestRegressor(verbose=False)
		rf.fit(feat_set, y_labels)
		pickle.dump(rf, open(write_path+"fit_rf_"+name+".pkl", "wb"))


# @@@@@@@@@@@@@@@@@@@@@  



if __name__ == "__main__":
	n_epochs = 5000
	n_patience_early_stopping = int(n_epochs*0.02)+1
	n_layers_remove = 4
	n_layers_add = 4

	output_path = "process/fepspace_smiles_per_sem.csv"
	writeFEPSpaceResults(
		results_path="input/fepspace_sems_full_balanced.csv",
		reference_path="input/fepspace_perts_full_compiled.csv",
		output_path=output_path)

	# load the dataset into a dataframe.
	fepspace_df = pd.read_csv(output_path)


	######################################################################
	######################################################################
	########################  BASE MODELS   ##############################
	######################################################################
	# we run an SVM and an RF regressor. We train on the WHOLE of FEP-Space.
	# we train on ECFP2, molecular properties and on APFP features.
	apfps, ecfps, props = computeFeats(fepspace_df, "process/base_models/")

	# train and store models as pickle.
	trainBaseModels(apfps, ecfps, props,
					fepspace_df.fepspace_sem.values,
					"process/base_models/")

	#####################################################################
	#####################################################################
	#####################################################################
	################### MODEL REGENERATION###############################
	#####################################################################
	#####################################################################

	print("Retrieving FEP atom-mappings..")
	fepspace_df["atom_mappings"] = [ retrieveRGroupMappings(*a) for a in tuple(zip(
			fepspace_df["target"], fepspace_df["ligand1_original_pert_name"])) ]

	for rep in range(10):
		print("£"*100)
		print(f"REPLICATE {rep}")
		print("£"*100)
		########## CROSS VALIDATION #######
		# split into n folds. Transfer-learn, finetune and save each.
		kf = KFold(n_splits=5, shuffle=True)
		for K, (train_index, valid_index) in enumerate(kf.split(fepspace_df)):
			
			print(f"WORKING ON CV SPLIT {K}")


			############## featurise and split the input data ##################
			print("\nGenerating graphs from SMILES..")

			# Train set: 
			print("\nSetting up training set.")
			x_train_0 = graphs_from_smiles(fepspace_df.iloc[train_index].ligand1_smiles)
			x_train_1 = graphs_from_smiles(fepspace_df.iloc[train_index].ligand2_smiles)
			y_train = fepspace_df.iloc[train_index].fepspace_sem
			train_mapping_arrays = fepspace_df.iloc[train_index].atom_mappings.values.tolist()
			train_set = MPNNDataset(x_train_0, x_train_1, train_mapping_arrays, y_train)
			print("Size:",len(y_train))

			# Valid set: 
			print("\nSetting up validation set.")
			x_valid_0 = graphs_from_smiles(fepspace_df.iloc[valid_index].ligand1_smiles)
			x_valid_1 = graphs_from_smiles(fepspace_df.iloc[valid_index].ligand2_smiles)
			y_valid = fepspace_df.iloc[valid_index].fepspace_sem
			valid_mapping_arrays = fepspace_df.iloc[valid_index].atom_mappings.values.tolist()
			valid_set = MPNNDataset(x_valid_0, x_valid_1, valid_mapping_arrays, y_valid)
			print("Size:",len(y_valid))

			

			############## build MPNN MODELS ##################

			# Build the lambda 0 and 1 legs (both are individual MPNNs).
			print("\nBuilding model..")
			fepnn = MPNNModel(
				atom_dim_0=x_train_0[0][0][0].shape[0], bond_dim_0=x_train_0[1][0][0].shape[0],
				atom_dim_1=x_train_1[0][0][0].shape[0], bond_dim_1=x_train_1[1][0][0].shape[0],
				r_group_mapping_dim=valid_mapping_arrays[0].shape[0]
				)

			# now load in pre-trained weights.
			weights_path = "process/trained_model_weights/weights"
			print(f"Loading model weights from {weights_path}..")
			fepnn.load_weights(weights_path)

			######################################################################
			######################################################################
			######################################################################
			#################### TRANSFER LEARNING ###############################
			######################################################################
			######################################################################

			# try/except to catch heisenbug in tensorflow, similar to 
			# https://stackoverflow.com/questions/49951822/invalidargumenterror-concatop-dimensions-of-inputs-should-match
			try:
				# now transfer-learn. First, attach new layers to the model while removing the last few.
				print("Replacing last n layers with untrained FCNN layers..")
				transferred_fepnn = modelTransfer(base_model=fepnn, 
									n_layers_remove=n_layers_remove, n_layers_add=n_layers_add,
									dropout=False)
				transferred_fepnn.compile(
					loss=keras.losses.LogCosh(),
					optimizer=keras.optimizers.Adam(learning_rate=5e-4),
					metrics=['mae'],
					)

				print("Fitting transferred model..")
				es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_mae', 
					patience=n_patience_early_stopping, restore_best_weights=True, verbose=1)

				history = transferred_fepnn.fit(
				train_set,
					validation_data=valid_set,
					epochs=n_epochs,
					verbose=2,
					callbacks=[es_callback]
				)

				# save training information.
				print("Saving training information..")
				pd.DataFrame(history.history).to_csv(f"process/training_history_transfer_{rep}_{K}.csv")
				plt.figure()
				plt.plot(history.history['mae'], label="Train")
				plt.plot(history.history['val_mae'], label="Validation")
				plt.ylabel('Loss - mean absolute error of SEM / kcal$\cdot$mol$^{-1}$')
				plt.xlabel('Epoch')
				plt.legend(loc='upper left')
				plt.savefig(f'process/training_history_plot_transfer_{K}.png')

				# save model weights. The model can be restored in later scripts by reconstructing
				# the classes in this script and loading the weights.
				weights_path = f"process/trained_model_weights/weights_transfer_{rep}_{K}"
				print(f"Saving model weights to {weights_path}..")
				transferred_fepnn.save_weights(weights_path)

				lowest_val_mae_transfer = min(pd.DataFrame(history.history)["val_mae"])

				######################################################################
				######################################################################
				######################################################################
				######################### FINETUNING #################################
				######################################################################
				######################################################################	
				# unfreeze all layers to see if we can further push down the validation error.
				print("\n"+("@"*50))
				print("Fine-tuning.")
				# copy the model and unfreeze weights. 
				finetuned_model = tf.keras.models.clone_model(transferred_fepnn)
				finetuned_model.set_weights(transferred_fepnn.get_weights())

				for layer in finetuned_model.layers:
					layer.trainable = True

				print(finetuned_model.summary())

				# continue with training protocol as before.
				finetuned_model.compile(
					loss=keras.losses.LogCosh(),
					optimizer=keras.optimizers.Adam(learning_rate=5e-5),
					metrics=['mae'],
					)

				history = finetuned_model.fit(
				train_set,
					validation_data=valid_set,
					epochs=n_epochs,
					verbose=2,
					callbacks=[es_callback]
				)

				# save training information.
				print("Saving training information..")
				pd.DataFrame(history.history).to_csv(f"process/training_history_finetuned_{rep}_{K}.csv")

				plt.figure()
				plt.plot(history.history['mae'], label="Train")
				plt.plot(history.history['val_mae'], label="Validation")
				plt.ylabel('Loss - mean absolute error of SEM / kcal$\cdot$mol$^{-1}$')
				plt.xlabel('Epoch')
				plt.legend(loc='upper left')
				plt.savefig(f'process/training_history_plot_finetuned_{rep}_{K}.png')

				lowest_val_mae_finetune = min(pd.DataFrame(history.history)["val_mae"])

				# save model weights. The model can be restored in later scripts by reconstructing
				# the classes in this script and loading the weights.
				weights_path = f"process/trained_model_weights/weights_finetuned_{rep}_{K}"
				if lowest_val_mae_finetune < lowest_val_mae_transfer:
					print("Finetuned model performs better than transfer-learned model.")
					print(f"Saving model weights to {weights_path}..")
					finetuned_model.save_weights(weights_path)

				elif lowest_val_mae_finetune > lowest_val_mae_transfer:
					print("Finetuned model performs worse than transfer-learned model. Saving TF model instead.")
					print(f"Saving model weights to {weights_path}..")
					transferred_fepnn.save_weights(weights_path)
			except tf.errors.InvalidArgumentError as e:
				print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!! TENSORFLOW ERROR:")
				print(e)
				print("\n\n\n\n\n")
				print("Continuing with next replicate..")
				continue

	print("\nDone.")



