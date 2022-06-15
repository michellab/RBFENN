#!/bin/python

"""
Creates example files that tell the SF predictor CLI which input files to grab.
"""

import glob # for searching through dirs
import csv # for writing csv files
from itertools import combinations # handy function to create pairs of items in lists.

ligand_file_paths = glob.glob("ClusterMolecules10/*.sdf")

# first example: make a file with 5 pairs of ligands.
#with open("example_parse_file_small.csv", "w") as writefile:
#	writer = csv.writer(writefile)
#
#	writer.writerow([ligand_file_paths[1], ligand_file_paths[4]])
#	writer.writerow([ligand_file_paths[4], ligand_file_paths[2]])
#	writer.writerow([ligand_file_paths[12], ligand_file_paths[0]])
#	writer.writerow([ligand_file_paths[6], ligand_file_paths[8]])
#	writer.writerow([ligand_file_paths[7], ligand_file_paths[5]])

# second example: make a fileall possible pairs of ligands.
with open("cluster_pairs.csv", "w") as writefile:
	writer = csv.writer(writefile)

	# first get all combinations of ligands paths.
	all_pairs = list(combinations(ligand_file_paths, 2))

	# loop over the pairs while writing out lines.
	for a, b in all_pairs:
		writer.writerow([a, b])

		# also write the inverse pair, itertools.combinations only grabs A->B.
		writer.writerow([b, a])
