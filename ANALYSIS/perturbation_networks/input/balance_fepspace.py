#!/bin/python

"""
Quick script to balance FEP-Space SEMs. In its native form, the FEP-Space training set
(fepspace_sems_full.csv) contains A->B and B->A for most perturbations, but SEM values
can be (usually are) not the same in both directions, whereas this should be true in 
theory. 

This script takes each perturbation, finds both directions, computes the mean SEM and 
assigns the mean SEM to both directions. This training set balancing will reinforce
model signal that SEM A->B = SEM B->A.
"""

import csv
import numpy as np 

fepspace_entries = {}
with open("fepspace_sems_full.csv", "r") as readfile:
	reader = csv.reader(readfile)
	next(reader)  # skip header.
	for row in reader:
		if row[0] == "pert":
			continue # skips some artefact headers in the file.

		fepspace_entries[row[0]] = row[1:]



fepspace_pert_smiles = {}
with open("fepspace_perts_full.csv", "r") as readfile:
	reader = csv.reader(readfile)
	next(reader)  # skip header.
	for row in reader:
		# make a pert_name that matches the style of fepspace_entries
		pert_name = f"{row[0]}_{row[1]}"
		fepspace_pert_smiles[pert_name] = row[3:5]

# now that we have 1) fepspace sems and 2) smiles for each pert, balance the sems.
balanced_fepspace = {}

for pert_name, freenrgs in fepspace_entries.items():
	# get the inverse pert.
	liga, ligb = "_".join(pert_name.split("_")[:-1]).split("~")
	inv_pert_name = f"{ligb}~{liga}_{pert_name.split('_')[-1]}"

	# skip already handled entries.
	if pert_name in balanced_fepspace:
		continue

	# two possible scenarios to deal with.
	# 1): there is no reverse of the perturbation in FEP-Space yet. 
	# take the single SEM as mean SEM, create both entries. n=586.
	if not inv_pert_name in fepspace_entries:
		inv_freenrgs = [f"{float(freenrg)*-1}" for freenrg in freenrgs[:5]]
		inv_freenrgs.append(freenrgs[-1])

		# add to balanced dataset.
		balanced_fepspace[pert_name] = freenrgs
		balanced_fepspace[inv_pert_name] = inv_freenrgs

	# 2): there is a reverse of the perturbation in FEP-Space.
	# compute the mean SEM between the two; add both to the balanced set.
	# n=3842.
	else:
		sem_1 = freenrgs[-1]
		sem_2 = fepspace_entries[inv_pert_name][-1]
		mean_sem = np.mean([float(sem_1), float(sem_2)])

		# replace SEMs with mean sem.
		freenrgs[-1] = mean_sem
		inv_freenrgs = fepspace_entries[inv_pert_name]
		inv_freenrgs[-1] = mean_sem

		# add to balanced dataset.
		balanced_fepspace[pert_name] = freenrgs
		balanced_fepspace[inv_pert_name] = inv_freenrgs

# by balancing and mirroring perturbations, we may have created duplicates, i.e. different pert_names
# but the chemical transformation is the same (there are many overlapping perturbations in the FEP-Space source).
# remove these.
handled_pert_smiles = []
perts_to_remove = []
for i, (pert_name, freenrgs) in enumerate(balanced_fepspace.items()):

	# first grab the pert_smiles as a list of ["smiles_lig_a", "smiles_lig_b"]
	try:
		pert_smiles = fepspace_pert_smiles[pert_name]
	except KeyError:
		try:
			pert_smiles = fepspace_pert_smiles[inv_pert_name]
		except KeyError:
			# perturbation is not in the reference. Instead, use NaN as smiles.
			pert_smiles = [f"NaN_{i}_A", f"NaN_{i}_B"]

	# delete duplicates from dict.
	if not pert_smiles in handled_pert_smiles:
		handled_pert_smiles.append(pert_smiles)
	else:
		perts_to_remove.append(pert_name)

for pert in perts_to_remove:
	del balanced_fepspace[pert]

# now write the balanced fepspace to file.
with open("fepspace_sems_full_balanced.csv", "w") as writefile:
	writer = csv.writer(writefile)
	writer.writerow(["pert","freenrg_1","freenrg_2","freenrg_3","freenrg_4","freenrg_5","SEM"])
	for pert, freenrgs in balanced_fepspace.items():
		data = []
		data.append(pert)
		[data.append(val) for val in freenrgs]
		writer.writerow(data)











