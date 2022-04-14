#!/bin/python


"""
Given a SEMs csv, ensure that all edges in the network are present in the file.
Compute both the ddG (solvated - bound) and SEM (propagated) for each perturbation.
Make the file bidirectional by inverting the freenrg sign per edge.
"""

import csv
import numpy as np 
from glob import glob
import itertools


##################################################################
# make dicts with all available freenrg data.

freenrg_dict_free = {}
freenrg_dict_bound = {}

with open("compiled_mbar_freenrgs_std.csv", "r") as readfile:
	reader = csv.reader(readfile)
	next(reader) # skip header

	for row in reader:
		pertname = "_".join(row[0].split("_")[:-1])
		inv_pertname = "~".join([pertname.split("~")[1], pertname.split("~")[0]])


		phase = row[0].split("_")[-1]

		freenrgs = [ float(val) for val in row[1:-1] if not val == 'fail' ]
		
		if phase == "free":
			freenrg_dict_free[pertname] = np.mean(freenrgs)
			freenrg_dict_free[inv_pertname] = np.mean(freenrgs)*-1

		elif phase == "bound":
			freenrg_dict_bound[pertname] = np.mean(freenrgs)
			freenrg_dict_bound[inv_pertname] = np.mean(freenrgs)*-1


##################################################################
# make dict with all available SEM data by propagating.

sems_dict_free = {}
sems_dict_bound = {}

all_sems = []

with open("compiled_mbar_freenrgs_std.csv", "r") as readfile:
	reader = csv.reader(readfile)
	next(reader) # skip header

	for row in reader:
		pertname = "_".join(row[0].split("_")[:-1])
		inv_pertname = "~".join([pertname.split("~")[1], pertname.split("~")[0]])
		phase = row[0].split("_")[-1]

		
		if phase == "free":
			sems_dict_free[pertname] = float(row[-1])
			sems_dict_free[inv_pertname] = float(row[-1])

		elif phase == "bound":
			sems_dict_bound[pertname] = float(row[-1])
			sems_dict_bound[inv_pertname] = float(row[-1])

		all_sems.append(float(row[-1]))


sems_dict = {}
for pert, sem in sems_dict_free.items():


	# propagate the error between free and bound.
	try:
		prop_err = np.sqrt(sem**2 + sems_dict_bound[pert]**2)
	except KeyError:
		# this SEM is missing - some replicates must have failed.
		prop_err = max(all_sems)
	sems_dict[pert] = prop_err

##################################################################
# Write all data to a single file containing rows of [pert, freenrg, prop_err].
with open("fep_data_tyk2_fc_std.csv", "w") as writefile:
	writer = csv.writer(writefile)

	writer.writerow(["lig_1","lig_2","freenrg","error"])

	for pert, freenrg_free in freenrg_dict_free.items():

		lig_1 = "lig_"+pert.split("~")[0] 
		lig_2 = "lig_"+pert.split("~")[1]
		try:
			freenrg_bound = freenrg_dict_bound[pert]
		except KeyError:
			# this freenrg is missing - some replicates must have failed.
			freenrg_bound = 5000
		prop_sem = sems_dict[pert]

		ddg = freenrg_bound - freenrg_free # sign has been inverted before.
		writer.writerow([lig_1, lig_2, ddg, prop_sem])






