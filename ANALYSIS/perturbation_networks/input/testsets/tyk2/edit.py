import csv


print(f"lig_1,lig_2,freenrg,error")
with open("sems_tmp", "r") as readfile:
	# file made by cat freenrg* > sems_tmp

	reader = csv.reader(readfile)
	for row in reader:
		if row[0] == 'pert':
			# skip headers.
			continue

		lig1, lig2 = row[0].split("~")

		if lig1 == "jul_01":
			lig1 = "intermediate_01"
		if lig2 == "jul_01":
			lig2 = "intermediate_01"
		lig1_newname = "lig_"+lig1
		lig2_newname = "lig_"+lig2

		if lig1_newname == "lig_intermediate_01":
			lig1_newname = lig1_newname.replace("lig_", "")
		if lig2_newname == "lig_intermediate_01":
			lig2_newname = lig2_newname.replace("lig_", "")

		print(f"{lig1_newname},{lig2_newname},{row[2]},{row[1]}")
		print(f"{lig2_newname},{lig1_newname},{(float(row[2]))*-1},{row[1]}")