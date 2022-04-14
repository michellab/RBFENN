import shutil
import BioSimSpace as BSS
import os

if os.path.exists("poep"):
	shutil.rmtree("poep")

lig = BSS.IO.readPDB("protein.pdb", pdb4amber=True)[0]
protein = BSS.Parameters.ff14SB(lig, water_model="tip3p", work_dir="poep").getMolecule()
print(protein)

BSS.IO.saveMolecules("protein", protein, ["prm7", "rst7"])
