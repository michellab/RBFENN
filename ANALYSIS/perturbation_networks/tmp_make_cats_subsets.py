from rdkit import Chem
from rdkit import DataStructs
import shutil
import os
import glob
import itertools

lig_1 = None
lig_1 = None
remaining_ligs = []

mols = []

for lig_path in glob.glob("/home/jscheen/projects/FEPSPACE/fep_ref_ligands/cats/*.sdf"):
	mol = Chem.SDMolSupplier(lig_path)[0]
	lig_name = lig_path.split("/")[-1].replace(".sdf","")

	mols.append([mol, lig_name])

mol_simis = []
for lig_1, lig_2 in list(itertools.combinations(mols, 2)):
	fps = [Chem.RDKFingerprint(x) for x in [lig_1[0], lig_2[0]]]
	simi_1 = DataStructs.FingerprintSimilarity(fps[0],fps[1])

	mol_simis.append([lig_1[1], lig_2[1], simi_1, lig_1[0], lig_2[0]])

mol_simis.sort(key = lambda x: x[2], reverse=True)

cluster_center_1_name = mol_simis[0][0]
cluster_center_2_name = mol_simis[0][1]
cluster_center_1_mol = mol_simis[0][3]
cluster_center_2_mol = mol_simis[0][4]
 
for lig_path in glob.glob("/home/jscheen/projects/FEPSPACE/fep_ref_ligands/cats/*.sdf"):
	mol = Chem.SDMolSupplier(lig_path)[0]
	lig_name = lig_path.split("/")[-1].replace(".sdf","")

	# if Chem.rdmolops.GetFormalCharge(mol) != 0:
	# 	continue

	if lig_name == cluster_center_1_name:
		continue
	if lig_name == cluster_center_2_name:
		continue

	else:
		fps = [Chem.RDKFingerprint(x) for x in [cluster_center_1_mol, cluster_center_2_mol, mol]]
		simi_1 = DataStructs.FingerprintSimilarity(fps[0],fps[2])
		simi_2 = DataStructs.FingerprintSimilarity(fps[1],fps[2])
		avg_simi = (simi_1+simi_2)/2

		remaining_ligs.append([lig_name, avg_simi])

remaining_ligs.sort(key = lambda x: x[1], reverse=True)	


for i in range(5,len(remaining_ligs)):
	core_set = [x[0] for x in remaining_ligs[:i]]
	core_set.append(cluster_center_1_name) 
	core_set.append(cluster_center_2_name) 

	if os.path.exists(f"tmp/cats_{i}"):
		shutil.rmtree(f"tmp/cats_{i}", ignore_errors=True)
	os.mkdir(f"tmp/cats_{i}")
		
	for lig_name in core_set:
		# copy to folder.
		shutil.copyfile(f"/home/jscheen/projects/FEPSPACE/fep_ref_ligands/cats/{lig_name}.sdf",
			f"tmp/cats_{i}/{lig_name}.sdf")

		shutil.copyfile(f"/home/jscheen/projects/FEPSPACE/fep_ref_ligands/cats/{lig_name}.mol2",
			f"tmp/cats_{i}/{lig_name}.mol2")