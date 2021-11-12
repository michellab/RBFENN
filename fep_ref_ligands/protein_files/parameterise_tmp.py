import BioSimSpace as BSS 

print("Reading..")
protein = BSS.IO.readPDB("p38.pdb", pdb4amber=False)[0]

print("Paramaterising..")
protein_p = BSS.Parameters.ff99SB(protein, work_dir="tmp").getMolecule()