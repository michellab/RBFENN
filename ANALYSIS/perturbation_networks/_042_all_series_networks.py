import BioSimSpace as BSS
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import minmax_scale
import glob
import csv
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import os
import shutil
from functools import reduce
import sys


from rdkit import Chem

def computeStats(input_df):
	"""return statistics input for table to plot."""
	col_labels = ["Random", "ML", "LOMAP"]
	row_labels = ["Pearson r", "MUE /\nkcal$\cdot$mol$^{-1}$", r"Kendall $\tau$"]
	table_vals = []
	for array in [input_df["Random\nSEM [0-1]"], 
				  input_df["ML-Predicted\nSEM [0-1]"],
				  input_df["LOMAP-Score"]
				 ]:
		

		r = round(stats.pearsonr(input_df["True SEM [0-1]"], array)[0], 2)
		mue = round(mean_absolute_error(input_df["True SEM [0-1]"], array), 2)
		tau = round(stats.kendalltau(input_df["True SEM [0-1]"], array)[0], 2)
		table_vals.append([r, mue, tau])
		
	# need to transpose table values to fit the row/col layout.
	table_vals = np.array(table_vals).T.tolist()
	return col_labels, row_labels, table_vals
	

def generateLinksFiles(preds_path):
	"""Given a csv file with SEM predictions, return links file data for LOMAP.
	Because SEMs are the inverse of LOMAP (i.e. LOMAP-score 0.0 is 'poor', but SEM 0.0 is 'good'), 
	we need to invert the predicted SEM values. Additionally, compress them to fall in the range 0-1 
	as is the case with LOMAP-scores."""
	preds_df = pd.read_csv(preds_path)


	pert_names = preds_df["pert_name"].values
	
	# for these, take inverse and scale to 0-1.
	pred_sems = minmax_scale(1 / preds_df["pred_sem_mean"].values, feature_range=(0,1))
	
	random_sems = preds_df["random_sem"].values

	simi_vals = preds_df["fp_similarity"].values

	return pert_names, pred_sems, random_sems, simi_vals
		


def writeLinksFile(pert_names, values, filename, simi=False):
	with open(filename, "w") as writefile:
		writer = csv.writer(writefile, delimiter =" ")
		for pert_name, value in zip(pert_names, values):
			writer.writerow([pert_name, value])

def runLOMAP(tgt, links_file, simi=False):
	path_to_ligands = f"/home/jscheen/projects/FEPSPACE/fep_ref_ligands/{tgt}"
	ligand_files = glob.glob(f"{path_to_ligands}/*.sdf")

	ligands = []
	ligand_names = []

	for filepath in ligand_files:
		# append the molecule object to a list.
		#ligands.append(BSS.IO.readMolecules(filepath)[0])
		ligands.append(Chem.SDMolSupplier(filepath)[0])

		# append the molecule name to another list so that we can use the name of each molecule in our workflow.
		ligand_names.append(filepath.split("/")[-1].replace(".sdf",""))

	if not links_file:   
		# standard LOMAP. Even if we have a links_file, we need to run a 'vanilla' LOMAP first to get the work dir.
		tranformations, lomap_scores = BSS.Align.generateNetwork(ligands, plot_network=True, names=ligand_names,
							work_dir="tmp/lomap_workdir") 

	else:
		# run LOMAP with alternative input scorings.
		if not simi:
			# use ecfp6 tanimoto similarity instead of LOMAP-Scores.
			links_file = f"tmp/lomap_ml_links_file_{tgt}.csv"
		else:
			# use ml-predicted SEM instead of LOMAP-Scores.
			links_file = f"tmp/lomap_simi_links_file_{tgt}.csv"


		# bit of a workaround, but we have to find the sdf file names that LOMAP uses internally.  
		# write out a second linksfile that doesn't have the tilde to denote the perturbations. Also
		# use glob in the pre-generated LOMAP work folder to find what we should call our ligands (i.e.
		# refer to the internal LOMAP file name).
		links_file_contents = pd.read_csv(links_file, sep=" ", header=None)
		lomap_internal_files = glob.glob("tmp/lomap_workdir/inputs/*.sdf")
		internal_links_file_path = links_file.replace(".csv", "_internal.csv")

		with open(internal_links_file_path, "w") as writefile:
			writer = csv.writer(writefile, delimiter =" ")
			lig1, lig2 = None, None
			for pert_name, value in zip(links_file_contents[0].values, links_file_contents[1].values):
				# find the internal path.
				for lig in lomap_internal_files:
					if pert_name.split("~")[0] in lig:
						lig1 = lig.split("/")[-1]
					elif pert_name.split("~")[1] in lig:
						lig2 = lig.split("/")[-1]
				if lig1 and lig2:
					writer.writerow([lig1, lig2, value])

		# now run LOMAP with the pre-specified edge scorings.
		tranformations, lomap_scores = BSS.Align.generateNetwork(ligands, plot_network=True, names=ligand_names,
																links_file=internal_links_file_path,
																work_dir="tmp/lomap_workdir")    
	
	pert_network_dict = {}
	transformations_named = [(ligand_names[transf[0]], ligand_names[transf[1]]) for transf in tranformations]
	for transf, score in zip(transformations_named, lomap_scores):
		transf_tilde = "~".join(transf)
		pert_network_dict[transf_tilde] = score

	return tranformations, lomap_scores, pert_network_dict

def compareNetworks(tgt_to_do, input_type):
	for sem_preds_file in glob.glob("output/series_predictions/*.csv"):
		tgt = sem_preds_file.split("/")[-1].replace(".csv","")
		if tgt == "network_comparison":
			continue

		if tgt_to_do in tgt:

			pert_names, pred_sems, random_sems, simi_vals = generateLinksFiles(sem_preds_file)

			if input_type == "naive":
				print("Running LOMAP naively..")
				tranformations, scores, pert_network_dict = runLOMAP(tgt, links_file=False)

			elif input_type == "ml":
				print("Running ML-LOMAP..")
				writeLinksFile(pert_names, pred_sems, f"tmp/lomap_ml_links_file_{tgt}.csv")
				tranformations, scores, pert_network_dict = runLOMAP(tgt, links_file=True)

			elif input_type == "tanimoto":
				print("Running SIMI-LOMAP..")
				writeLinksFile(pert_names, simi_vals, f"tmp/lomap_simi_links_file_{tgt}.csv", simi=True)
				tranformations, scores, pert_network_dict = runLOMAP(tgt, links_file=True, simi=True)

			if len(tranformations) < 2:
				#raise ValueError("No network was generated, check your inputs or ./tmp/lomap_workdir/!") 
				print("Failed.")
			else:
				# copy the network to results.
				shutil.copy("tmp/lomap_workdir/images/network.png", 
					f"output/series_predictions/{tgt}_{input_type}_network.png")

				return tranformations, scores, pert_network_dict

def computeNetworkOverlap(tranformations, tranformations_query):
	"""
	Given two sets of edges (list), count the number of common edges. Return the 
	number of overlapping edges as a % of the total number of edges in the first list (lomap).
	"""
	overlap = 0
	unique = 0

	for edge_lomap in tranformations:

		inv_edge = (edge_lomap[1], edge_lomap[0])
		if edge_lomap in tranformations_query:
			overlap += 1
		elif inv_edge in tranformations_query:
			overlap += 1
		else:
			unique += 1

	total = overlap + unique
	perc_overlap = int(overlap/total*100)
	return perc_overlap

def countLigands(edges):
	"""
	From a list of edges (consisting of tuples of atom indices), return the number of ligands.
	"""
	ligands = []
	for mol0, mol1 in edges:
		if not mol0 in ligands:
			ligands.append(mol0)
		if not mol1 in ligands:
			ligands.append(mol1)

	return len(ligands)


if __name__ == "__main__":

	[os.remove(linkspath) for linkspath in glob.glob("tmp/lomap*.csv")]
	shutil.rmtree("tmp/lomap_workdir", ignore_errors=True)

	with open("output/series_predictions/network_comparison.csv", "w") as writefile:
		writer = csv.writer(writefile)
		writer.writerow([
			"target",
			"num_ligands",
			"num_edges_naive",
			"num_edges_ml",
			"num_edges_tanimoto",
			"overlap_ml_vs_naive",
			"overlap_tanimoto_vs_naive",
			"overlap_ml_vs_tanimoto"
			])

		for sem_preds_file in glob.glob("output/series_predictions/*.csv"):
			tgt = sem_preds_file.split("/")[-1].replace(".csv","")
			if tgt == "network_comparison":
				continue
			print(tgt)

			# run vanilla LOMAP first; then alternative input LOMAPs. 
			tranformations, scores, pert_network_dict = compareNetworks(tgt, "naive")
			tranformations_ml, scores_ml, pert_network_dict_ml = compareNetworks(tgt, "ml")
			tranformations_simi, scores_simi, pert_network_dict_simi = compareNetworks(tgt, "tanimoto")

			# analyse network overlaps wrt vanilla LOMAP.
			ml_ntw_overlap = computeNetworkOverlap(tranformations, tranformations_ml)
			simi_ntw_overlap = computeNetworkOverlap(tranformations, tranformations_simi)
			ml_vs_simi_ntw_overlap = computeNetworkOverlap(tranformations_ml, tranformations_simi)

			print(ml_ntw_overlap, simi_ntw_overlap, ml_vs_simi_ntw_overlap)
			writer.writerow([
				tgt,
				countLigands(tranformations),
				len(tranformations),
				len(tranformations_ml),
				len(tranformations_simi),
				ml_ntw_overlap,
				simi_ntw_overlap,
				ml_vs_simi_ntw_overlap
				])

			# clear all links files from ./tmp.
			[os.remove(linkspath) for linkspath in glob.glob("tmp/lomap*.csv")]
			shutil.rmtree("tmp/lomap_workdir", ignore_errors=True)
			print("\n")












