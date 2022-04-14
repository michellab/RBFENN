import csv
import sys
import numpy as np
from math import exp, log
from scipy import stats
from glob import glob 
from itertools import combinations

# FEPs on the FEPNN and LOMAP networks were run in the same campaign on a HPC, so
# need to split them up into separate summary files.

# find out which edges are in the FEPNN network.
fepnn_feps = []
with open("tnks2_fepnn_perts.csv", "r") as readfile:
	reader = csv.reader(readfile)
	for row in reader:
		inv_row = "~".join([row[0].split("~")[1], row[0].split("~")[0]])
		fepnn_feps.append(row[0])
		fepnn_feps.append(inv_row)

# find out which edges are in the LOMAP network.
lomap_feps = []
with open("tnks2_lomap_perts.csv", "r") as readfile:
	reader = csv.reader(readfile)
	for row in reader:
		inv_row = "~".join([row[0].split("~")[1], row[0].split("~")[0]])
		lomap_feps.append(row[0])
		lomap_feps.append(inv_row)


for i in range(1, 6):

	# get average FE predictions per replicate.
	summary_fes = {}
	summary_errs = {}

	# first read FEP data.
	with open(f"summaries/summary_rep_{i}.csv", "r") as readfile:
		reader = csv.reader(readfile)
		
		for row in reader:
			if row[0] == "lig_1":
				continue # skip headers

			pert = f"{row[0]}~{row[1]}"
			summary_fes[pert] = float(row[2])
			summary_errs[pert] = float(row[3])


	# write out an individual summary file for the FEPNN network.
	with open(f"summary_fepnn_{i}.csv", "w") as writefile:
		writer = csv.writer(writefile)
		writer.writerow(['lig_1', 'lig_2', 'freenrg', 'error', 'engine'])

		for pert in fepnn_feps:
			# get the data for this edge (and the inverse)
			liga, ligb = pert.split("~")

			try:
				fe = summary_fes[pert]
				err = summary_errs[pert]
			except KeyError:
				# if the pert failed for this run, grab the inverse.
				inv_pert = f"{ligb}~{liga}"
				fe = summary_fes[inv_pert]*-1
				err = summary_errs[inv_pert]
			engine = "SOMD"

			# write out the lines with data.
			writer.writerow([liga, ligb, fe, err, engine])


	# write out an individual summary file for the FEPNN network.
	with open(f"summary_lomap_{i}.csv", "w") as writefile:
		writer = csv.writer(writefile)
		writer.writerow(['lig_1', 'lig_2', 'freenrg', 'error', 'engine'])

		for pert in lomap_feps:
			# get the data for this edge (and the inverse)
			liga, ligb = pert.split("~")

			try:
				fe = summary_fes[pert]
				err = summary_errs[pert]
			except KeyError:
				# if the pert failed for this run, grab the inverse.
				inv_pert = f"{ligb}~{liga}"
				fe = summary_fes[inv_pert]*-1
				err = summary_errs[inv_pert]
			engine = "SOMD"

			# write out the lines with data.
			writer.writerow([liga, ligb, fe, err, engine])

for performed_feps, ntwk_type in zip([fepnn_feps, lomap_feps], ["fepnn", "lomap"]):

	for i in range(1, 6):
		# make all possible combinations of the 5 replicates.
		for cnts in combinations([1,2,3,4,5], i):

			# get average FE predictions per replicate.
			summary_fes = {}
			summary_errs = {}

			for content in cnts:
				with open(f"summary_{ntwk_type}_{content}.csv", "r") as readfile:
					reader = csv.reader(readfile)

					for row in reader:
						if row[0] == "lig_1":
							continue # skip header
						pert = f"{row[0]}~{row[1]}"
						if not pert in summary_fes:
							summary_fes[pert] = [float(row[2])]
							summary_errs[pert] = [float(row[3])]
						else:
							summary_fes[pert].append(float(row[2]))
							summary_errs[pert].append(float(row[3]))
			

			# write out an individual summary file for the FEPNN network.
			cnts = [ str(cnt) for cnt in cnts]
			with open(f"processed_summaries/summary_{ntwk_type}_{''.join(cnts)}.csv", "w") as writefile:
				writer = csv.writer(writefile)
				writer.writerow(['lig_1', 'lig_2', 'freenrg', 'error', 'engine'])

				for pert in performed_feps:
					# get the data for this edge (and the inverse)
					liga, ligb = pert.split("~")

					try:
						fe = summary_fes[pert]
						err = summary_errs[pert]
					except KeyError:
						try:
							# if the pert failed for this run, grab the inverse.
							inv_pert = f"{ligb}~{liga}"
							fe = summary_fes[inv_pert]*-1
							err = summary_errs[inv_pert]
						except KeyError:
							pass # pert is missing from this replicate.


					# get mean fes and sems.

					if len(fe) == 1:
						fe = fe[0]
						err = err[0]
					else:
						err = stats.sem(fe)
						fe = np.mean(fe)



					engine = "SOMD"

					# write out the lines with data.
					writer.writerow([liga, ligb, fe, err, engine])




























