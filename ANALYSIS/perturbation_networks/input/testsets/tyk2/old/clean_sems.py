#!/bin/python


"""
Given a SEMs csv, ensure that all edges in the network are present in the file.
in some cases edges might be missing - SEMs and freenrgs can either be derived 
from the inverse perturbation data or set to a high value and nan, respectively.
"""

import csv
import numpy as np 
from glob import glob
import itertools

##################################################################
# make a dict with all available freenrg/SEM data.

fep_dict = {}
ddg_sems, bound_sems, free_sems = [], [], []
with open("tyk2_fc_prop_sems_equil.csv", "r") as readfile:
	reader = csv.reader(readfile)
	next(reader) # skip header

	for row in reader:
		fep_dict[row[0]] = row[1:]

		# also append sems to lists so that we can get max(sems) in next loop.
		ddg_sems.append(float(row[2].replace("--", "nan")))
		bound_sems.append(float(row[3].replace("--", "nan")))
		free_sems.append(float(row[4].replace("--", "nan")))


##################################################################
# get all possible perturbations for the ligand series,
# get all inverse perturbations and store information in a dict.

ligands = [ lig.replace(".sdf", "").replace("lig_", "").split("/")[-1] for \
			lig in glob("ligands/*.sdf") ]

write_dict = {}			
for pert in itertools.combinations(ligands, 2):
	inv_pert = "~".join([pert[1], pert[0]])
	pert = "~".join(pert)

	if "intermediate" in pert:
		continue			# removes intermediate because we 
							# haven't simulated all edges for this.

	# try to grab data for the forward perturbation.
	try:
		write_dict[pert] = fep_dict[pert]

	except KeyError:
		# this perturbation is missing. Use the inverse edge info instead,
		# while making sure to invert the freenrg sign.
		inv_pert_data = fep_dict[inv_pert]
		inv_pert_data[0] = -float(inv_pert_data[0])
		write_dict[pert] = inv_pert_data

	# now do the same for the backward perturbation.
	try:
		write_dict[inv_pert] = fep_dict[inv_pert]

	except KeyError:
		# this perturbation is missing. Use the inverse edge info instead,
		# while making sure to invert the freenrg sign.
		inv_pert_data = fep_dict[pert]
		inv_pert_data[0] = -float(inv_pert_data[0])
		write_dict[pert] = inv_pert_data	


##################################################################
# clean the write dict, replacing nans etc.

for k, v in write_dict.items():
	
	# clean up data. 
	v = [ str(val).replace("--", "nan") for val in v ]

	# - Replace freenrg nans with 5000 (handled by freenrgworkflows)
	# - Replace SEM nans with max(sems) of the corresponding error type
	if v[0] == "nan":
		v[0] = 5000

	if v[1] == "nan":
		v[1] = max(ddg_sems)
		
	if v[2] == "nan":
		v[2] = max(bound_sems)

	if v[3] == "nan":
		v[3] = max(free_sems)

	# update the dict.
	write_dict[k] = v


# write the output files, one per SEM type.
with open("cleaned_tyk2_fc_prop_sems_ddg_equil.csv", "w") as writefile:
	writer = csv.writer(writefile)
	writer.writerow(["lig_1", "lig_2","freenrg","error"])

	for k, v in write_dict.items():
		lig1 = "lig_"+k.split("~")[0]
		lig2 = "lig_"+k.split("~")[1]

		freenrg = v[0]
		sem = v[1]  # ddG SEM
		writer.writerow([lig1, lig2, freenrg, sem])

with open("cleaned_tyk2_fc_prop_sems_bound_equil.csv", "w") as writefile:
	writer = csv.writer(writefile)
	writer.writerow(["lig_1", "lig_2","freenrg","error"])

	for k, v in write_dict.items():
		lig1 = "lig_"+k.split("~")[0]
		lig2 = "lig_"+k.split("~")[1]

		freenrg = v[0]
		sem = v[2]  # bound SEM
		writer.writerow([lig1, lig2, freenrg, sem])


with open("cleaned_tyk2_fc_prop_sems_free_equil.csv", "w") as writefile:
	writer = csv.writer(writefile)
	writer.writerow(["lig_1", "lig_2","freenrg","error"])

	for k, v in write_dict.items():
		lig1 = "lig_"+k.split("~")[0]
		lig2 = "lig_"+k.split("~")[1]

		freenrg = v[0]
		sem = v[3]  # free SEM
		writer.writerow([lig1, lig2, freenrg, sem])








