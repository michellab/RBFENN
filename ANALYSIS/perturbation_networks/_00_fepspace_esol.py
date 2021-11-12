#!/bin/python

# Takes FEPspace transformations, makes all possible transformations, estimates the relative solubility
# for each pair and writes to file. 

import csv
import tqdm
import itertools
import random

import sys
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski
from rdkit.Chem import PandasTools
import pandas as pd
from sklearn.linear_model import LinearRegression
from collections import namedtuple

class ESOLCalculator:
	# adopted from https://github.com/PatWalters/solubility/blob/master/esol.py
    def __init__(self):
        self.aromatic_query = Chem.MolFromSmarts("a")
        self.Descriptor = namedtuple("Descriptor", "mw logp rotors ap")

    def calc_ap(self, mol):
        """
        Calculate aromatic proportion #aromatic atoms/#atoms total
        :param mol: input molecule
        :return: aromatic proportion
        """
        matches = mol.GetSubstructMatches(self.aromatic_query)
        return len(matches) / mol.GetNumAtoms()

    def calc_esol_descriptors(self, mol):
        """
        Calcuate mw,logp,rotors and aromatic proportion (ap)
        :param mol: input molecule
        :return: named tuple with descriptor values
        """
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        rotors = Lipinski.NumRotatableBonds(mol)
        ap = self.calc_ap(mol)
        return self.Descriptor(mw=mw, logp=logp, rotors=rotors, ap=ap)

    def calc_esol(self, mol):
        """
        Calculate ESOL based on descriptors in the Delaney paper, coefficients refit for the RDKit using the
        routine refit_esol below
        :param mol: input molecule
        :return: predicted solubility
        """
        intercept = 0.26121066137801696
        coef = {'mw': -0.0066138847738667125, 'logp': -0.7416739523408995, 'rotors': 0.003451545565957996, 'ap': -0.42624840441316975}
        desc = self.calc_esol_descriptors(mol)
        esol = intercept + coef["logp"] * desc.logp + coef["mw"] * desc.mw + coef["rotors"] * desc.rotors \
               + coef["ap"] * desc.ap
        return esol

if __name__ == "__main__":
    esol_calculator = ESOLCalculator()

    fepspace_mols = []
    # read the perturbations file, create per-ligand solubility estimates.
    with open("input/fepspace_perts.csv", "r") as fepspace_perts:
        reader = csv.reader(fepspace_perts)
        next(reader) # skip header.
        for row in tqdm.tqdm(reader, total=3144):
            lig1 = Chem.MolFromSmiles(row[3])
            lig2 = Chem.MolFromSmiles(row[4])
            if lig1:
                lig1_solv = esol_calculator.calc_esol(lig1)
                fepspace_mols.append([row[3], lig1_solv, row[1], row[0]])
            if lig2:
                lig2_solv = esol_calculator.calc_esol(lig2)
                fepspace_mols.append([row[4], lig2_solv, row[1], row[0]])  


    # divide into train and validation sets here before we inflate the data. Do 80/20.
    random.shuffle(fepspace_mols) # is in place.
    train_mols = fepspace_mols[:int(len(fepspace_mols) * .8)]
    valid_mols = fepspace_mols[int(len(fepspace_mols) * .8):]


    for dataset, name in zip([train_mols, valid_mols], ["train", "valid"]):
        rsolv_entries = []
        # now generate all possible pairs of ligands.
        # during combination, compute the relative solubility by subtraction.
        print(f"\nGenerating combinations and computing relative solubility estimates: {name}..")
        for comb in tqdm.tqdm(itertools.combinations(dataset, 2)):
            lig1_smi = comb[0][0]
            lig2_smi = comb[1][0]
            rel_sol = comb[1][1] - comb[0][1]

            lig1_tgt = comb[0][2]
            lig1_ori_pert = comb[0][3]
       
            rsolv_entries.append([lig1_smi, lig2_smi, rel_sol, lig1_tgt, lig1_ori_pert])
        # shuffle so that we're sure that data is OK for ML input.
        print("\nShuffling data..")
        random.shuffle(rsolv_entries) # takes about 10sec on our system.

        print("\nWriting to file..")
        with open(f"process/inputs/rel_esol_fepspace_{name}.csv", "w") as solv_file:
            writer = csv.writer(solv_file)
            writer.writerow(["ligand1_smiles", "ligand2_smiles", "relative_solubility", "target", "ligand1_original_pert_name"])
            for entry in tqdm.tqdm(rsolv_entries):
                writer.writerow(entry)    

    print("Done.")















