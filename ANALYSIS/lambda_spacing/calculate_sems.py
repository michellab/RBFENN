#!/bin/python

# take summaries.csv, compute quintuplicate freenrgs and SEMs. Write to a new file.

import csv
import numpy as np 
from scipy import stats

perts = {}

with open("summaries.csv", "r") as readfile:
	reader = csv.reader(readfile)
	for row in reader:
		if not row[0] == "lig_1":

			pert = f"{row[0]}~{row[1]}"
			if not pert in perts:
				perts[pert] = [float(row[2])]
			elif pert in perts:
				perts[pert].append(float(row[2]))
i = 0
with open("tyk2_freenrgs.csv", "w") as writefile:
	writer = csv.writer(writefile)
	for pert, freenrgs in perts.items():
		if len(freenrgs) > 2:
			writer.writerow([pert, stats.sem(freenrgs)])
			i+=1


print(f"Wrote {i} SEM values to 'tyk2_freenrgs.csv'.")