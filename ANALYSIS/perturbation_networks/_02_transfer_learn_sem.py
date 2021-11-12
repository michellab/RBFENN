#!/bin/python

# transfer-learn the twin GCN model to predict FEP SEMs.

# import code to regenerate the twin GCN.
from _01_twin_gcn import *

# misc imports.
import os

import csv
from scipy import stats
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

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
		for row in reader:

			found = False
			pert = "_".join(row[0].split("_")[:-1])
			tgt = row[0].split("_")[-1]

			# now find the SMILES information in ref.
			for ref in fepspace_ref:
				if ref[0] == pert and ref[1] == tgt:
					found = True
					fepspace_smiles_sems.append([ref[3], ref[4], row[-1], pert, tgt])

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
    leading up to a single linear neuron.
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

# @@@@@@@@@@@@@@@@@@@@@  



if __name__ == "__main__":
	n_epochs = 5000
	n_patience_early_stopping = int(n_epochs*0.02)+1
	n_layers_remove = 4
	n_layers_add = 4

	output_path = "process/fepspace_smiles_per_sem.csv"
	writeFEPSpaceResults(
		results_path="input/fepspace_training_set_main.csv",
		reference_path="input/fepspace_perts.csv",
		output_path=output_path)

	# load the dataset into a dataframe.
	fepspace_df = pd.read_csv(output_path)

	print("Retrieving FEP atom-mappings..")
	fepspace_df["atom_mappings"] = [ retrieveRGroupMappings(*a) for a in tuple(zip(
			fepspace_df["target"], fepspace_df["ligand1_original_pert_name"])) ]

	######################################################################
	######################################################################
	######################################################################
	#################### MODEL REGENERATION###############################
	######################################################################
	######################################################################


	########### CROSS VALIDATION #######
	# split into ten folds. Transfer-learn, finetune and save each.
	kf = KFold(n_splits=10, shuffle=True)
	for K, (train_index, valid_index) in enumerate(kf.split(fepspace_df)):
		print("£"*100)
		print(f"WORKING ON CV SPLIT {K}")

		############### featurise and split the input data ##################
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
		pd.DataFrame(history.history).to_csv(f"process/training_history_transfer_{K}.csv")
		plt.figure()
		plt.plot(history.history['mae'], label="Train")
		plt.plot(history.history['val_mae'], label="Validation")
		plt.ylabel('Loss - mean absolute error of SEM / kcal$\cdot$mol$^{-1}$')
		plt.xlabel('Epoch')
		plt.legend(loc='upper left')
		plt.savefig(f'process/training_history_plot_transfer_{K}.png')

		# save model weights. The model can be restored in later scripts by reconstructing
		# the classes in this script and loading the weights.
		weights_path = f"process/trained_model_weights/weights_transfer_{K}"
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
		pd.DataFrame(history.history).to_csv(f"process/training_history_finetuned_{K}.csv")

		plt.figure()
		plt.plot(history.history['mae'], label="Train")
		plt.plot(history.history['val_mae'], label="Validation")
		plt.ylabel('Loss - mean absolute error of SEM / kcal$\cdot$mol$^{-1}$')
		plt.xlabel('Epoch')
		plt.legend(loc='upper left')
		plt.savefig(f'process/training_history_plot_finetuned_{K}.png')

		lowest_val_mae_finetune = min(pd.DataFrame(history.history)["val_mae"])

		# save model weights. The model can be restored in later scripts by reconstructing
		# the classes in this script and loading the weights.
		weights_path = f"process/trained_model_weights/weights_finetuned_{K}"
		if lowest_val_mae_finetune < lowest_val_mae_transfer:
			print("Finetuned model performs better than transfer-learned model.")
			print(f"Saving model weights to {weights_path}..")
			finetuned_model.save_weights(weights_path)

		elif lowest_val_mae_finetune > lowest_val_mae_transfer:
			print("Finetuned model performs worse than transfer-learned model. Saving TF model instead.")
			print(f"Saving model weights to {weights_path}..")
			transferred_fepnn.save_weights(weights_path)
			
	print("\nDone.")



