#!/bin/python
# does the same as other script but instead does free - bound per repl.
import os
import csv
import glob
from scipy import stats
from tqdm import tqdm
from pathlib import Path
import sys
import numpy as np

def readMBARFile(path):
	freenrg = None
	# retrieve the MBAR freenrg for this perturbation.
	try:
		with open(str(path), "r") as readfile:
			for row in readfile:
				if row.startswith("#MBAR"):
					freenrg_line = next(readfile)
					freenrg = float(freenrg_line.rsplit(",")[0])
	except FileNotFoundError:
		return None
	###############################
	# append to compiled list.
	if freenrg:
		return freenrg

def propErrors(err1, err2):
	"""Returns the propagated error"""
	squared_sum = err1**2 + err2**2
	return np.sqrt(squared_sum)
	
# if prop, compute SEM per leg, then propagate SEM between legs. If not prop, compute ddG per leg 
# per replicate, then compute SEM of ddGs.
propagate = True


if not propagate:
	with open("sems_ddgs_cresset_quintuplicates.csv", "w") as writefile:
		writer = csv.writer(writefile)
		writer.writerow(["tgt", "pert", "SEM", "freenrg", "std"])
		for tgt in ["BACE", "CDK2", "JNK1", "MCL", "P38", "PTP1B", "THROMBIN", "TYK2"]:

			perts_quintup_dict = {}
			for repl in range(1, 6):

				handled_perts = []
				perts_fullpaths = glob.glob(f"{tgt}/{repl}/*")

				perts = [path.split("/")[-1] for path in perts_fullpaths]

				freenrgs = [ readMBARFile(f"{path}/freenrg-MBAR.dat") for path in perts_fullpaths ]

				perts_comb = {}
				for pert, freenrg in zip(perts, freenrgs):
					pert_name = "_".join(pert.split("_")[:-1])
					pert_phase = pert.split("_")[-1]

					if pert_name in perts_comb:
						perts_comb[pert_name].append([pert_phase, freenrg])
					else:
						perts_comb[pert_name] = [[pert_phase, freenrg]]
				

				for pert, values in perts_comb.items():
					dg_free = [val[1] for val in values if val[0] == 'Free']
					dg_bound = [val[1] for val in values if val[0] == 'Bound']
					try:
						ddg = float(dg_free[0]) - float(dg_bound[0])
					except (TypeError, IndexError) as e:
						# either free or bound phase is missing; discard.
						continue

					# assemble the ddGs per perturbation into a dict for easy averaging.
					if pert in perts_quintup_dict:
						perts_quintup_dict[pert].append(ddg)
					else:
						perts_quintup_dict[pert] = [ddg]

			# compute SEM for each pert. Allow triplicates and up.
			for pert, ddgs in perts_quintup_dict.items():
				if len(ddgs) < 2:
					continue
				sem = stats.sem(ddgs)
				freenrg = np.mean(ddgs)
				std = np.std(ddgs)

				writer.writerow([tgt, pert, sem, freenrg, std])

elif propagate:
	with open("propd_sems_ddgs_cresset_quintuplicates.csv", "w") as writefile:
		writer = csv.writer(writefile)
		writer.writerow(["tgt", "pert", "SEM"])
		for tgt in ["BACE", "CDK2", "JNK1", "MCL", "P38", "PTP1B", "THROMBIN", "TYK2"]:

			free_perts_quintup_dict = {}
			bound_perts_quintup_dict = {}
			all_freenrgs = []
			for repl in range(1, 6):

				handled_perts = []
				perts_fullpaths = glob.glob(f"{tgt}/{repl}/*")

				perts = [path.split("/")[-1] for path in perts_fullpaths]

				freenrgs = [ readMBARFile(f"{path}/freenrg-MBAR.dat") for path in perts_fullpaths ]
				
				for freenrg in freenrgs:
					if freenrg:
						all_freenrgs.append(float(freenrg))

				for pert, freenrg in zip(perts, freenrgs):
					pert_name = "_".join(pert.split("_")[:-1])
					pert_phase = pert.split("_")[-1]

					if pert_phase == "Free":

						if pert_name in free_perts_quintup_dict:
							free_perts_quintup_dict[pert_name].append(freenrg)
						else:
							free_perts_quintup_dict[pert_name] = [freenrg]
					elif pert_phase == "Bound":

						if pert_name in bound_perts_quintup_dict:
							bound_perts_quintup_dict[pert_name].append(freenrg)
						else:
							bound_perts_quintup_dict[pert_name] = [freenrg]
			free_perts = {}
			bound_perts = {}

			for k, v in free_perts_quintup_dict.items():
				if not len(v) < 3:
					sem = stats.sem(np.array(v).astype(float), nan_policy='omit')
					free_perts[k] = sem

			for k, v in bound_perts_quintup_dict.items():
				if not len(v) < 3:
					sem = stats.sem(np.array(v).astype(float), nan_policy='omit')
					bound_perts[k] = sem
					
			# propagate and write.
			for k_f, v_f in free_perts.items():
				try:
					pert_name = k_f
					free_sem = v_f
					bound_sem = bound_perts[pert_name]
					writer.writerow([tgt, pert_name, propErrors(free_sem, bound_sem)])
				except KeyError:
					# either free or bound leg is missing; set to have a high SEM.
					writer.writerow([tgt, pert_name, max(all_freenrgs)])


		
