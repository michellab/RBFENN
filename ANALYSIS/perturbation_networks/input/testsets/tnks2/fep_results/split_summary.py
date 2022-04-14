import csv
import sys
import numpy as np
from math import exp, log
from scipy import stats

# FEPs on the FEPNN and LOMAP networks were run in the same campaign on a HPC, so
# need to split them up into separate summary files.


# get average FE predictions across replicates.
summary_fes = {}
summary_errs = {}

for i in range(1, 11):
	# first read FEP data.
	with open(f"summary_{i}.csv", "r") as readfile:
		reader = csv.reader(readfile)
		next(reader) # skip header
		for row in reader:
			pert = f"{row[0]}~{row[1]}"
			if not pert in summary_fes:
				summary_fes[pert] = [float(row[2])]
				summary_errs[pert] = [float(row[3])]
			else:
				summary_fes[pert].append(float(row[2]))
				summary_errs[pert].append(float(row[3]))

# get mean FEs and propagated errors.
for pert, fes in summary_fes.items():
	mean_fe = np.mean(fes)
	summary_fes[pert] = mean_fe

	summary_errs[pert] = stats.sem(fes)


# find out which edges are in the FEPNN network.
fepnn_feps = []
with open("tnks2_fepnn_perts.csv", "r") as readfile:
	reader = csv.reader(readfile)
	for row in reader:
		inv_row = "~".join([row[0].split("~")[1], row[0].split("~")[0]])
		if row[0] in summary_fes:
			fepnn_feps.append(row[0])
		if inv_row in summary_fes:
			fepnn_feps.append(inv_row)
		else:
			pass

# find out which edges are in the LOMAP network.
lomap_feps = []
with open("tnks2_lomap_perts.csv", "r") as readfile:
	reader = csv.reader(readfile)
	for row in reader:
		inv_row = "~".join([row[0].split("~")[1], row[0].split("~")[0]])

		if row[0] in summary_fes:
			lomap_feps.append(row[0])
		if inv_row in summary_fes:
			lomap_feps.append(inv_row)
		else:
			pass

# write out an individual summary file for the FEPNN network.
with open(f"summary_fepnn_mean.csv", "w") as writefile:
	writer = csv.writer(writefile)
	writer.writerow(['lig_1', 'lig_2', 'freenrg', 'error', 'engine'])

	for pert in fepnn_feps:
		# get the data for this edge (and the inverse)
		liga, ligb = pert.split("~")
		fe = summary_fes[pert]
		err = summary_errs[pert]
		engine = "SOMD"

		# write out the lines with data.
		writer.writerow([liga, ligb, fe, err, engine])
		# writer.writerow([ligb, liga, fe*-1, err, engine])


# write out an individual summary file for the LOMAP network.
with open(f"summary_lomap_mean.csv", "w") as writefile:
	writer = csv.writer(writefile)
	writer.writerow(['lig_1', 'lig_2', 'freenrg', 'error', 'engine'])

	for pert in lomap_feps:
		# get the data for this edge (and the inverse)
		liga, ligb = pert.split("~")
		fe = summary_fes[pert]
		err = summary_errs[pert]
		engine = "SOMD"

		# write out the lines with data.
		writer.writerow([liga, ligb, fe, err, engine])
		# writer.writerow([ligb, liga, fe*-1, err, engine])


