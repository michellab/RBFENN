#!/bin/python

# Create the FEPSpace variant of all ligands in an external test set (i.e. congeneric series). 
# Load the transfer-learned model, predict SEMs. Compare to TRUE SEMs. 
# Write predictions to file to generate networks with in _04.

import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import rdmolops, rdMolAlign
from rdkit.Chem import Draw, rdFMCS, AllChem, rdmolfiles, Descriptors, rdchem, rdMolDescriptors
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem import Draw

from rdkit.Chem.Draw import IPythonConsole

import BioSimSpace as BSS

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import re

import time
from collections import deque
import subprocess
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from openbabel import pybel

import glob
import csv
import copy
import itertools
import random
# import code to regenerate the twin GCN.
import sys
sys.path.insert(1, '../ANALYSIS/perturbation_networks/')
from _01_twin_gcn import *
from _02_transfer_learn_sem import *


import argparse


def CountHAChange(fragment1_mol, fragment2_mol):
    """Takes in two rdkit fragment molecules, counts heavy atom changes and returns the number."""
    fragA_smiles = Chem.MolToSmiles(fragment1_mol)
    fragB_smiles = Chem.MolToSmiles(fragment2_mol)  
    
    double_letter_elements = ["Cl", "Br", "Si"]

    # assign a score based on n_ha transformed:
    transform_score = 0
    for frag_A, frag_B in itertools.zip_longest(fragA_smiles.split("."), fragB_smiles.split(".")):

        # clean up the strings by just retaining letters for easier counting:
        if frag_A:
            fragA_atoms = ''.join(x for x in frag_A if x.isalpha())
        else:
            fragA_atoms = "X"
        if frag_B:
            fragB_atoms = ''.join(x for x in frag_B if x.isalpha())
        else:
            fragB_atoms = "X"
            
        
        # a substitution counts as a single-atom change:
        if len(fragA_atoms) == len(fragB_atoms):
            transform_score += 1
        
        elif len(fragA_atoms) != len(fragB_atoms):
            # add number of heavy atoms changed.
            if len(fragA_atoms) > len(fragB_atoms):
                transform_score += len(fragA_atoms)
            else:
                transform_score += len(fragB_atoms)
        
        # correct for double-letter elements by subtracting 1.
        for elem in double_letter_elements:
            if elem in fragA_atoms:
                transform_score -= 1
            if elem in fragB_atoms:
                transform_score -= 1
            

    return transform_score, fragA_smiles, fragB_smiles

def constructSmarts(lig_mol, mcs_object):
    """
    Given a ligand and MCS (generated with a second ligand), construct an alternative SMARTS that contains
    information on the anchor atom (i.e. the atom in the MCS the perturbed R-group is attached to.)
    
    Get all neighbour indices of the fragment atoms in the original molecule.  
    The (single) index that is in the neighbour indices but not in the fragment 
    indices (set difference) is the atom we want. Anchor atoms and fragments are 
    in the same order because of consistent RDKit indexing.
    """
    # get the fragments by subtracting MCS from ligand.
    lig_fragments = Chem.ReplaceCore(lig_mol, Chem.MolFromSmarts(mcs_object.smartsString))
       
    # get the atom indices for the MCS object.
    mcs_indices = lig_mol.GetSubstructMatch(Chem.MolFromSmarts(mcs_object.smartsString))

    # get all the indices for the ligand.
    ligand_indices = set([x for x in range(0, lig_mol.GetNumAtoms())])

    # get all the fragment indices.
    non_mcs_indices = set(ligand_indices) - set(mcs_indices)

    new_smarts = None
    anchor_atoms = []
    anchor_atoms_idcs = []

    for frag_idx in non_mcs_indices:
        # get the neighbours for this fragment atom.
        nghbrs = lig_mol.GetAtomWithIdx(frag_idx).GetNeighbors()

        nghbr_ats = [ lig_mol.GetAtomWithIdx(nghbr.GetIdx()).GetSmarts() for nghbr in nghbrs ] 

        for nghbr in nghbrs:
            # find the set difference.
            if not nghbr.GetIdx() in non_mcs_indices:
                anchor_atoms.append(lig_mol.GetAtomWithIdx(nghbr.GetIdx()).GetSmarts())
                anchor_atoms_idcs.append(nghbr.GetIdx())

    # correct an issue with MCS, where e.g. an isopropyl is grafted as two methyls because the MCS 
    # extends into the base carbon of the isopropyl.
    molfrags = Chem.GetMolFrags(lig_fragments, asMols=True, sanitizeFrags=False)

    if len(anchor_atoms_idcs) == 3:
        if "." in Chem.MolToSmiles(lig_fragments): # exclude fused rings.

            if anchor_atoms_idcs[0] == anchor_atoms_idcs[1] == anchor_atoms_idcs[2]:
                # this will be an isobutyl (-type) R-group. Correct the lig_fragments.
                anchor_atom = lig_mol.GetAtomWithIdx(anchor_atoms_idcs[0]).GetSmarts()

                fr_0 = Chem.MolToSmiles(lig_fragments).split(".")[0].partition("]")[-1]
                fr_1 = Chem.MolToSmiles(lig_fragments).split(".")[1].partition("]")[-1]
                try:
                    fr_2 = Chem.MolToSmiles(lig_fragments).split(".")[2].partition("]")[-1]
                except IndexError:
                    new_smarts = f"{anchor_atom}[1*]({fr_0})({fr_1})"
                    return new_smarts # skips the third fragment if there is none.


                new_smarts = f"{anchor_atom}[1*]({fr_0})({fr_1}){fr_2}"
                return new_smarts
    elif len(anchor_atoms_idcs) == 2:
        if "." in Chem.MolToSmiles(lig_fragments): # exclude fused rings.

            if anchor_atoms_idcs[0] == anchor_atoms_idcs[1]:
                # this will be an isopropyl (-type) R-group. Correct the lig_fragments.
                anchor_atom = lig_mol.GetAtomWithIdx(anchor_atoms_idcs[0]).GetSmarts()
                fr_0 = Chem.MolToSmiles(lig_fragments).split(".")[0].partition("]")[-1]
                fr_1 = Chem.MolToSmiles(lig_fragments).split(".")[1].partition("]")[-1]

                new_smarts = f"{anchor_atom}[1*]({fr_0}){fr_1}"
                return new_smarts            

    # if no need to correct, just continue as normal, i.e. loop over R-groups.
    for anchor, molfrag in zip(anchor_atoms, Chem.GetMolFrags(lig_fragments, asMols=True, sanitizeFrags=False)):
        # clean up anchor. We really only care about aromatic vs non-aromatic etc.
        anchor = anchor.replace("@","").replace("[","").replace("]","")
    
        # for each fragment, we construct the smarts as [anchor*]atoms which fits with SMARTS logic.
        # Check https://www.daylight.com/dayhtml/doc/theory/theory.smarts.html
        # frag_smiles sometimes contains two dummy attachment points (in cases of ring fragments),
        # but for our purposes it should be OK to only denote the first dummy with the anchor.
        frag_smarts = Chem.MolToSmiles(molfrag)

        # we paste the anchor atom in front of the dummy notation in the SMARTS string. We need to retain the dummy
        # because this can come in handy when creating R groups on a scaffold (e.g. in case of fused ring perts).
        frag_smarts_anchored = anchor+frag_smarts

        # build the new SMARTS string. Insert a "." when there is already a fragment in the new string.
        if not new_smarts:
            new_smarts = frag_smarts_anchored
        else:
            new_smarts += "."+frag_smarts_anchored

    # sometimes the picked ligand is the actual MCS so there are no pert SMARTS.
    if not new_smarts:
        new_smarts = ""
        
    return new_smarts

def rewriteSMARTS(smarts_string):
    """Given a SMARTS string with possible multiple fragments, return a valid string where the anchor atom 
    is the first atom in each fragment, instead of denoted as [x*] (which is not parsable)."""
    
    frags_1, frags_2 = smarts_string.split("~")
    
    def constructPerFrag(frags_smarts):
        """Splits a ligand's fragments and processes each; puts anchor atom at base of each fragment."""
        fused_ring = False
        frags_whole_rewritten = None

        ####### MANUAL REPLACEMENTS
        # based on visualising specific outlier perturbations.

        # remove some stereo information that we don't need which just makes parsing 
        # these fragments even more complicated.
        if not ":" in frags_smarts and "[2*]" in frags_smarts:
            # dealing with a non-fused ring structure. We can replace the attachment
            # point with a carbon; just need to determine whether it should be aromatic
            # or aliphatic.
            if frags_smarts.count("c") < frags_smarts.count("C"):
                # there's an edge case here; [2*] is not always a ring structure. these perts are failing now in some cases.
                # frags_smarts = frags_smarts.replace("[2*]", "C1")
                # frags_smarts = frags_smarts[:2]+frags_smarts[2:].replace("C", "C1", 1) # set the first-occurring carbon as ring root.
                frags_smarts = frags_smarts.replace("[2*]", "C1")
                frags_smarts = frags_smarts[:2]+frags_smarts[2:].replace("C", "C1", 1) # set the first-occurring carbon as ring root.

            else:
                frags_smarts = frags_smarts.replace("[2*]", "c1")
                frags_smarts = frags_smarts[:2]+frags_smarts[2:].replace("c", "c1", 1) # set the first-occurring carbon as ring root.


        replace_queries = [
                            ["CH", "C"],
                            ["C@H", "C"],
                            ["[C@H]", "C"],
                            ["[C]", "C"],
                            ]
        for source, target in replace_queries:
            frags_smarts = frags_smarts.replace(source, target)


        # if trihalo, we can merge that into a single R group (i.e. fragment). These are dirty patches, 
        # protocol should be adjusted in FEP-Space generation.
        if "[C*]F.[C*]F.[C*]F" in frags_smarts:
            frags_smarts = frags_smarts.replace("[C*]F.[C*]F.[C*]F", "C[1*](F)(F)F")
        elif "[C*]Cl.[C*]Cl.[C*]Cl" in frags_smarts:
            frags_smarts = frags_smarts.replace("[C*]Cl.[C*]Cl.[C*]Cl", "C[1*](Cl)(Cl)Cl")

        for idx_a, idx_b, idx_c in itertools.permutations(["1", "2", "3"], 3):
            # indices are random; cover all possible orders and replace to isopropyl.
            if f"C[{idx_a}*]C.C[{idx_b}*]C.C[{idx_c}*]C" in frags_smarts:
                frags_smarts = frags_smarts.replace(f"C[{idx_a}*]C.C[{idx_b}*]C.C[{idx_c}*]C", 
                                                        "C[1*]C(C)(C)C")
        # NB: above conditionals may be obsolete due to isopropyl-type bugfixes in constructSmarts().
        
        # deal with a special case; R-group is a phenyl. This messes with the grafting algorithm,
        # (as the common scaffold is also benzene) so has to be set manually.  
        if frags_smarts == "c[1*]:ccccc:[2*]":
            frags_smarts = "c[1*]c1ccccc1"

        ####### END OF MANUAL REPLACEMENTS


        # now rewrite each fragment.
        for frag in frags_smarts.split("."):
            frag_parsed = None
            if len(frag) == 0:
                frag_parsed = anchor_atom = r_group_smarts = ""
            else: 
                anchor_atom = frag[0]
                r_group_smarts = frag[1:]
    
            if anchor_atom == "n":
                # use non-aromatic nitrogen instead.
                if frag_parsed:
                    frag_parsed += ".N"+r_group_smarts
                else:
                    frag_parsed = "N"+r_group_smarts 
            else:
                # anchor atom is used between R group and scaffold.
                if frag_parsed:
                    frag_parsed += "."+anchor_atom+r_group_smarts
                else:
                    frag_parsed = anchor_atom+r_group_smarts 

            # add the rewritten SMARTS string to the ligand SMARTS string (potentially >1 fragments).      
            if frags_whole_rewritten:
                frags_whole_rewritten += "."+frag_parsed
            else:
                frags_whole_rewritten = frag_parsed
            # record if this fragment contains a fused ring (multiple attachment points).  
            if frag_parsed.count(":") == 2:
                fused_ring = True

        # in case the fragments for this ligand contain a fused ring (multiple wildcards), reorder such
        # that the fused ring information comes first (simplifies grafting the fragments onto scaffold).
        if fused_ring:
            # get number of wildcards per sub-fragment.
            num_wildcards = [frag_str.count("*") for frag_str in frags_whole_rewritten.split(".")]
            
            # reorder the subfragments by descending number of wildcards (i.e. make fused ring come first).
            reordered = [x for _, x in sorted(zip(num_wildcards, frags_whole_rewritten.split(".")), reverse=True)]
            frags_whole_rewritten = ".".join(reordered)
                  
        return frags_whole_rewritten

    return constructPerFrag(frags_1), constructPerFrag(frags_2)

def graftToScaffold(frag):
    """Given a SMARTS pattern, create a benzene with R groups corresponding to the fragments in the SMARTS"""
    # start with regular benzene.
    main_scaffold = "c1ccccc1"
    scaffold_mol = Chem.MolFromSmiles(main_scaffold)
    
    # abort this perturbation if there are two or more fused rings being perturbed.
    if frag.count(".") == 1 and frag.count("*]") == 4:
        #print("Aborting this pert --> two or more fused rings being perturbed.")
        return None

    # abort this perturbation if there are too many R groups such that the 6 carbons on benzene are not enough.
    if frag.count(".") >= 5:
        #print("Aborting this pert --> more than 6 R groups.")
        return None       

    # loop over the molecules in this fragment.
    for fr in frag.split("."):
        # if the fragment is empty, this side of the perturbation is empty. Exit the loop to just keep benzene.
        if len(fr) == 0:
            break

        if fr[0] == "[":
            # catch rare cases where anchor atom had stereo information.
            fr = fr[1]+fr[2:]
            
        # count the number of wildcards in this set of fragments.
        if fr.count("*") <= 1:
            # in this case we can simply graft the structure onto benzene.
            anchor = fr[0]


            if anchor == "c":
                # removing aromatic anchor will make r_group graft onto benzene directly (making it use the aromatic 
                # benzene carbon as anchor).
                anchor = ""
            r_group = fr[5:]
            
            try:
                r_group_mol = Chem.MolFromSmiles("C"+anchor+r_group)
                scaffold_mol = Chem.ReplaceSubstructs(scaffold_mol, 
                                         Chem.MolFromSmiles('C'), 
                                         r_group_mol,
                                         replaceAll=False,
                                        )[0]
            # if no failed r_group, return as None. Next loop iteration will succeed in grafting.
            except:
                return None


        else:
            # if >1 wildcards are in the string, we are dealing with a fused ring structure.
            # this is considerably more complicated to graft onto benzene. We will rewrite 
            # the fragment such that it contains the benzene as well; then instead of replacing 
            # a single carbon in the scaffold we will replace it with the newly generated scaffold.            
            new_scaffold = createFusedRingScaffold(fr)
            
            # several forms of fused rings are being excluded (see createFusedRingScaffold() for rationales).
            if new_scaffold is None:
                return None
            else:
                try:
                    scaffold_mol = Chem.ReplaceSubstructs(scaffold_mol, 
                                 Chem.MolFromSmiles("c1ccccc1"), 
                                 Chem.MolFromSmiles(new_scaffold),
                                 replaceAll=False,
                                )[0]
                except:
                    # final error catches. If at this point the fused ring can not be grafted onto the scaffold
                    # it can be discarded as brute-forcing it in would substantially alter the chemistry of 
                    # the benzene scaffold.
                    #print("Aborting this pert --> miscellaneous fused ring issue.")
                    return None
        
        # after grafting we need to sanitize.
        try:
            Chem.SanitizeMol(scaffold_mol)
        except:
            pass
    
    return scaffold_mol    

def createFusedRingScaffold(input_frag):
    """Given a fragment SMARTS that describes a fused ring, create a canonical smiles notation for benzene
    that contains the fused ring.
    """
    # first, verify that the input is indeed a fused ring SMARTS.
    if not input_frag.count("*") > 1:
        raise Exception("Input fragment is not a fused ring SMARTS!", input_frag)
        
    # in some cases, fused rings with double bonds will mess with scaffold aromaticity to such a degree that 
    # building them into our training set becomes complex and introduces noise.
    if "C=[" in input_frag or "]=C" in input_frag:
        #print("Aborting this pert --> fused ring double bond interrupts scaffold aromaticity.")
        return None
    
    ########## handle non-aromatic fused rings:
    if not input_frag.count(":") == 2:
        # For our structure, we need everything that is contained within the parentheses of the SMARTS notation.
        fused_ring_atoms = input_frag[input_frag.find("]")+len("["):input_frag.rfind("[")]
        
        # in case of 5 main cycle atoms, remove 1 'C'. This is due to SMARTS grammar and fixes kekulization errors.
        # count the number of atoms in the main cycle of the fused ring.
        fused_ring_string = re.sub(r"\([^()]*\)", "", fused_ring_atoms)
        fused_ring_size = len(fused_ring_string.replace("[", "").replace("]", "").replace("H",""))
        if fused_ring_size >4:
            if "CCC" in fused_ring_atoms:
                fused_ring_atoms = fused_ring_atoms.replace("CCC", "CC", 1)
            elif "CC" in fused_ring_atoms:
                fused_ring_atoms = fused_ring_atoms.replace("CC", "C", 1)
        
        # create the new scaffold string.
        new_scaffold = f"c1cccc2c1{fused_ring_atoms}2"
        
        return new_scaffold
    ########## handle aromatic fused rings:
    # this is much more complex.
    else:

        # we can ignore the anchor atom (assume aromaticity here, accounting for non-aromaticity here would be exceedingly
        # complex). For our structure, we need everything that is contained within the colons of the SMARTS notation.
        fused_ring_atoms = input_frag[input_frag.find(":")+len(":"):input_frag.rfind(":")]

        ##### clean up structure; 
        #in some cases there will be a trailing closing '('.
        if fused_ring_atoms[-1] == "(":
            fused_ring_atoms = fused_ring_atoms[:-1]
        elif fused_ring_atoms[-2:] == "(2":
            fused_ring_atoms = fused_ring_atoms[:-2]
        elif fused_ring_atoms[-3:] == "(C2":
            fused_ring_atoms = fused_ring_atoms[:-3]
        #####
        # in case branch notation for this ring is in the shape of [n*]:ring_atoms(:[n*])branch_atoms,
        # we have to swap ':[n*]' and 'branch_atoms'. The way this is written depends mostly on stereochemistry 
        # and messes up the way we parse.
        attch_pnt_strings = [f"(:[{i}*])" for i in range(10)]

        # count the number of atoms in the main cycle of the fused ring.
        fused_ring_string = re.sub(r"\([^()]*\)", "", fused_ring_atoms)
        fused_ring_size = len(fused_ring_string.replace("[", "").replace("]", "").replace("H",""))
        
        # in case of 5 main cycle atoms, remove 1 'c'. This is due to SMARTS grammar and fixes kekulization errors.
        if fused_ring_size >4:
            if "ccc" in fused_ring_atoms:
                fused_ring_atoms = fused_ring_atoms.replace("ccc", "cc", 1)
            elif "cc" in fused_ring_atoms:
                fused_ring_atoms = fused_ring_atoms.replace("cc", "c", 1)
         

        if any(c.split("])")[-1].isalpha() for c in input_frag) and any(x in input_frag for x in attch_pnt_strings):
            branch_structure = input_frag.split("])")[-1]
            fused_ring_atoms += f"({input_frag.split('])')[-1]})"
            
        # create the new scaffold string.
        new_scaffold = f"c1cccc2c1{fused_ring_atoms}2"

        return new_scaffold
               
def loadEnsemble(len_k, path_to_weights, input_model):
    """Loads k weights into k copies of input tf.keras model given weights path."""
    models_collections = []
    model_paths = glob(path_to_weights+"*.data*")

    replicates = [ int(path.split("_")[3]) for path in model_paths ]
    n_replicates = max(replicates)

    for rep in range(n_replicates):
        rep_model_basepath = f"{path_to_weights}_{rep}_*"

        models_collection = []
        
        for k in range(len_k):
            k_model_path = rep_model_basepath.replace("*", str(k))
                   
            k_model = tf.keras.models.clone_model(input_model)
            try:
                k_model.load_weights(k_model_path)
                models_collection.append(k_model)
            except tf.errors.NotFoundError:
                pass # in rare cases some model training fails; see training code.


        models_collections.append(models_collection)

    #print(f"Loaded a total of {len(models_collection)} models; {len_k} CV folds.")

    return models_collections

def predictEnsemble(ensemble, test_dataset):
    """Given an ensemble of tf.keras models with loaded weights and a test set, 
    make k predictions and return mean and std of the predictions."""
    preds = []

    for model in ensemble:

        pred_sem_values = np.ravel(model.predict(test_dataset))
        preds.append(pred_sem_values)

    # return the mean and std for each perturbation.
    return np.mean(preds, axis=0), np.std(preds, axis=0)

def RDKitToBSSMol(mol):
    """Uses a .pdb intermediate to convert an rdkit molecule object to a BSS mol object.
    """
    #mol = Chem.AddHs(mol) 
    AllChem.EmbedMolecule(mol) # need 3D coordinates for BSS to align molecules better.
    Chem.MolToPDBFile(mol, "tmp_mol.pdb")
    bss_mol = BSS.IO.readPDB("tmp_mol.pdb")[0]
    os.remove("tmp_mol.pdb")  
    return bss_mol  

def reloadLigand(ligand_smiles):
    """Loads a SMILES entry into RDKit, attempts to clean the structure and returns 
    a re-shuffled SMILES equivalent of the processed input. 
    """
    # use rdkit to write alternative SMILES.
    tmpmol = Chem.MolFromSmiles(ligand_smiles)

    # attempt to clean the structure.
    Chem.SanitizeMol(tmpmol)
    tmpmol.ClearComputedProps()
    tmpmol.UpdatePropertyCache()

    # reload through PDB.
    AllChem.EmbedMolecule(tmpmol) # adding 3D coordinates to the ligand helps with OFF processing.
    Chem.MolToPDBFile(tmpmol, "tmp/mol.pdb")
    bss_mol = BSS.IO.readPDB("tmp/mol.pdb")[0]

    return Chem.MolToSmiles(tmpmol, doRandom=True), bss_mol



def parameteriseLigand(input_ligand):
    """Parameterise an input BSS ligand structure with GAFF2 from SMILES input. returns the parameterised ligand
    and the used SMILES."""
    
    try:
        input_ligand_p = BSS.Parameters.gaff2(input_ligand, charge_method="GAS").getMolecule()
    
    except BSS._Exceptions.ParameterisationError:
        # introduce stereochemistry, see https://github.com/openforcefield/openff-toolkit/issues/146
        try:
            input_ligand_p = BSS.Parameters.gaff2(input_ligand.replace("[C]", "[C@H]"), charge_method="GAS").getMolecule()


        except BSS._Exceptions.ParameterisationError:
            # if it fails again, OFF might be struggling with the input SMILES. For these edge-cases 
            # sometimes it helps shuffling the order of SMILES. Try a few times.
            for attempt in range(15):
                try:
                    newsmiles, bss_mol = reloadLigand(input_ligand)
                    input_ligand_p = BSS.Parameters.gaff2(bss_mol, 
                            charge_method="GAS").getMolecule()
                    # return the new smiles as well. 
                    input_ligand = newsmiles
                    break
                    
                except BSS._Exceptions.ParameterisationError as e:
                    err_msg = e
                    input_ligand_p = None 

        
    if input_ligand_p == None:
        print(f"Bad input {err_msg}, returning None:", input_ligand)
        
    return input_ligand_p, input_ligand


def mapAtoms(mol1, mol2, forced_mcs_mapp=False):
    """
    Aligns and merges two BSS molecules; returns the atom mapping, merged molecule and a nested
    list describing the atom type changes.
    """
    if forced_mcs_mapp:
        mapp = BSS.Align.matchAtoms(mol1, mol2, prematch=forced_mcs_mapp)
    else:
        mapp = BSS.Align.matchAtoms(mol1, mol2)

    try:
        merged = BSS.Align.merge(mol1, mol2, mapp,
                                allow_ring_breaking=True,
                                allow_ring_size_change=True,
                                )
    except BSS._Exceptions.IncompatibleError:
        try:
            merged = BSS.Align.merge(mol1, mol2, mapp,
                                allow_ring_breaking=True,
                                allow_ring_size_change=True,
                                force=True
                                )
        except BSS._Exceptions.IncompatibleError:      
            # this mapping creates a very funky perturbation; discard.
            return {}, None, [[]]

    # # Get indices of perturbed atoms.
    idxs = merged._getPerturbationIndices()

    # For each atom in the merged molecule, get the lambda 0 and 1 amber atom type.
    atom_type_changes = [[merged.getAtoms()[idx]._sire_object.property("ambertype0"),  \
                 merged.getAtoms()[idx]._sire_object.property("ambertype1")] \
                 for idx in idxs]

    # Keep only changing atoms.
    atom_type_changes = [at_ch for at_ch in atom_type_changes if at_ch[0] != at_ch[1] ]

    return mapp, merged, atom_type_changes

def getBenzeneScaffoldIndices(molecule):
    """For a given RDKit molecule, return the atom indices for a benzene scaffold"""
    patt = Chem.MolFromSmiles('c1ccccc1')
    hit_ats = list(molecule.GetSubstructMatch(patt))
    return hit_ats, Chem.AddHs(molecule)

def rotate_values(my_dict):
    values_deque = deque(my_dict.values())
    values_deque.rotate(1)
    return dict(zip(my_dict.keys(), values_deque))

def getMapping(ligA, ligB, ori_mcs, abstract_mol_1, abstract_mol_2):
    """
    Given input (parameterised) original ligands A and B, their MCSresult and 
    the generated FEP-Space derivatives 1/2 in RDKit molecule object format,
    find the FEP-Space derivative atom-mapping that matches the original ligands' MCS.

    In its current form this step is prohibitively rate-limiting. Even with gasteiger-charge based
    parameterisation this protocol requires further speed-up. See https://github.com/michellab/BioSimSpace/issues/249.
    Potentially this could be solved by refactoring FEP-Space training set setup:
    - robustly graft R-groups onto benzene such that we keep track of the R-group indices that are perturbed
    - insert a reference R-group at e.g. idx 5 (e.g. -cc(At)c-)
    - featurise with standard mapping {0:0, 1:1, ..}
    - remove reference R-group

    alternatively, instead of feeding a mapping array into the NN, the amber atom-type changes 
    could be featurised directly through e.g. one-hot encoding.
    """
    """
    !!!!!!!!!!!!!!
    FOR THE CLI PROTOTYPE NO MAPPING CALCULATION IS DONE AS IT IS TOO EXPENSIVE
    RETURNING THE DEFAULT MAPPING INSTEAD
    THIS FUNCTION IS NOT EXECUTED
    IF AT SOME POINT THIS FUNCTION HAS BE EXECUTED THE INPUT LIGANDS MUST BE 
    PARAMETERISED BSS OBJECTS!!
    """
    return {0:0, 1:1, 2:2, 3:3, 4:4, 5:5} # remove this line if want to unlock mapping computation.


    # get the atom type changes for the original perturbation (i.e. input ligand).
    _, _, ori_atom_type_changes = mapAtoms(ligA, ligB)


    abs_lig_1, _ = parameteriseLigand(Chem.MolToSmiles(abstract_mol_1))
    abs_lig_2, _ = parameteriseLigand(Chem.MolToSmiles(abstract_mol_2))

    if not abs_lig_1 or not abs_lig_2:
        # ligands are not parameterisable due to SMILES inconsistencies. Will be discarded downstream.
        return {0:0, 1:1, 2:2, 3:3, 4:4, 5:5}

    # use RDKit to find the benzene atom indices. These are conserved between RDKit and BSS.
    lig_1_benzene_indices, lig_1_rdkit = getBenzeneScaffoldIndices(abstract_mol_1)
    lig_2_benzene_indices, lig_2_rdkit = getBenzeneScaffoldIndices(abstract_mol_2)


    # with these benzene indices make an initial forced mapping. 
    forced_mapping = {}
    for at1, at2 in zip(lig_1_benzene_indices, lig_2_benzene_indices):
        forced_mapping[at1] = at2

    # now find the correct mapping for the FEP-Space derivatives.
    correct_mapping = None
    mapping_highscore = -1

    for rotat in range(6):
        # rotate along the benzene MCS. 
        forced_mapping = rotate_values(forced_mapping)

        # use BSS to map the ligands together using the forced mapping.
        mapping, _, abs_atom_type_changes = mapAtoms(abs_lig_1, abs_lig_2, forced_mcs_mapp=forced_mapping)

        # compare the original and the current abstract atom type change lists.
        # the mapping with the highest number of matches will be the correct mapping.
        counter = 0
        for at_ch in abs_atom_type_changes:
            if at_ch in ori_atom_type_changes:
                counter += 1

        # set the new mapping as the correct mapping if it outperforms previous ones.
        if counter > mapping_highscore:
            correct_mapping = mapping
            mapping_highscore = counter

    return correct_mapping


def calcFPSimilarity(lig1, lig2):
    """computes molecular similarity using RDKit's standard approach"""

    # Compute ECFP6 (6 is diameter, in rdkit radius 3).
    fps = [ AllChem.GetMorganFingerprintAsBitVect(m,3,nBits=1024) for m in [lig1, lig2] ]

    # compute tanimoto similarity.
    fp_simi = DataStructs.FingerprintSimilarity(fps[0],fps[1], metric=DataStructs.DiceSimilarity)
    return fp_simi
    

def featurisePerturbations(perts, param_dict):
    """Loads ligands for a list of perturbations, uses a param_dict that is a {} that contains the 
    parameterised (GAFF2) molecule object value for each input file key.

    returns data for each in shape [[lig_1, lig2], [],[]]"""
    pert_to_investigate = None # enter name of perturbation here if investigating faulty featurisation. 

    pert_features, pert_names, pert_mappings, ha_change_counts, fp_simis = [], [], [], [], []

    
    for pert in tqdm(perts, total=len(perts)):
      
      pert_name = pert[0].split("/")[-1].replace(".sdf", "")+"~"+pert[1].split("/")[-1].replace(".sdf", "")
      
      if pert_to_investigate: # for testing certain perturbations.
        if pert_name != pert_to_investigate:
            continue

      ####################################################################
      ####################################################################
      # now find the FEP-Space derivative of these ligands given their MCS.
      ligA = Chem.SDMolSupplier(pert[0])[0]
      ligB = Chem.SDMolSupplier(pert[1])[0]


      # compute the molecular fingerprint similarity.
      fp_simis.append(calcFPSimilarity(ligA, ligB))

      # get MCS and fragments (i.e. perturbed R-groups). 1 second timeout is enough for drug-like ligands.
      mcs = rdFMCS.FindMCS([ligA, ligB], timeout=1, ringMatchesRingOnly=True, completeRingsOnly=True)

      
      # subtract the MCS to get the perturbed (i.e. non-mcs) atoms.
      fragA = Chem.ReplaceCore(ligA, Chem.MolFromSmarts(mcs.smartsString))
      fragB = Chem.ReplaceCore(ligB, Chem.MolFromSmarts(mcs.smartsString))

      # count the number of perturbed heavy atoms.
      ha_change_count, fragA_smiles, fragB_smiles = CountHAChange(fragA, fragB)

            # get the alternative perturbation SMARTS and graft to the FEP-Space scaffold.
      fail = False
      for _ in range(15):
          pert_smartsA = constructSmarts(ligA, mcs)
          pert_smartsB = constructSmarts(ligB, mcs)
          pert_smarts = pert_smartsA+"~"+pert_smartsB
          frags_1, frags_2 = rewriteSMARTS(pert_smarts) 

          # we've excluded charge perturbations from the training set as these were deemed out of scope.
          # for this test set just neutralise the charges. 
          frags_1 = frags_1.replace("+", "")
          frags_2 = frags_2.replace("+", "")
          frags_1 = frags_1.replace("-", "")
          frags_2 = frags_2.replace("-", "")

          abstract_mol_1 = graftToScaffold(frags_1)
          abstract_mol_2 = graftToScaffold(frags_2)


          if abstract_mol_1 and abstract_mol_2:
            fail = False
            break # when grafting has succeeded, continue on with the workflow instead of retrying.

          else:
            fail = True # in next iteration the SMILES of pert_smarts will have been shuffled.
          
      if pert_to_investigate: # for testing certain perturbations.
        if pert_name == pert_to_investigate:
            print(f"FEP-Space deriv. of {pert_to_investigate} (Fail:{fail}):")
            print("Pert-Smarts:", pert_smarts)
            print("Rewritten Smarts:", frags_1, frags_2)
            print(Chem.MolToSmiles(abstract_mol_1), Chem.MolToSmiles(abstract_mol_2))
            print("Quitting.")
            sys.exit()     

      if not fail:
      # MAPPING INFORMATION INJECTION
        # get the mapping for this perturbation.
        try:
          mapping_array = getMapping(param_dict[pert[0]], param_dict[pert[1]], mcs, abstract_mol_1, abstract_mol_2)

        except BSS._Exceptions.AlignmentError:
          # in very rare cases BSS alignment fails for very large perturbations; for 
          # these cases return a standard mapping. 
          mapping_array = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5}

      else:
        # if after 15 attempts the grafting still failed, flag the perturbation for failure.
          abstract_mol_1 = abstract_mol_2 = Chem.MolFromSmiles("c1ccccc1")
          ha_change_count = str(ha_change_count)+"_fail"

          # failed featurisations we can just set to default mapping as well.
          mapping_array = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5}


      # featurise the mapping.
      pert_mappings.append(featuriseMapping(mapping_array))

      # Get the SMILES for each FEP-Space derivative.
      pert_features.append([Chem.MolToSmiles(abstract_mol_1), Chem.MolToSmiles(abstract_mol_2)])

      pert_names.append(pert_name)
      ha_change_counts.append(ha_change_count)

    return pert_features, pert_names, pert_mappings, ha_change_counts, fp_simis   



def createTestDataset(perts_featurised, pert_mappings):
    """Creates a tf.dataset from a nested list of perturbations containing the smiles 
    for each ligand in shape [[lig_1, lig2], [],[]]"""
    
    ligA_features = [pert[0] for pert in perts_featurised]
    ligB_features = [pert[1] for pert in perts_featurised]

    # we have to use a placeholder y label. Model.predict() will predict the actual y_test values.
    y_test = list(range(len(perts_featurised)))

    ligA_graph = graphs_from_smiles(ligA_features)
    ligB_graph = graphs_from_smiles(ligB_features)

    # make a 'batch graph' of this perturbation. Use a batch size that fits (number doesn't really matter
    # in testing phase).
    series_dataset = MPNNDataset(ligA_graph, ligB_graph, pert_mappings, y_test, batch_size=1)

    return series_dataset

def loadPostProcessors(basepath_to_input):
    """
    Loads statistical summaries to postprocess features for test datapoints. Any test feature array
    must be normalized/PCAd in the same way that the training data was in order for predictions to 
    be sensible.
    """
    apfp_stats = pd.read_csv(basepath_to_input+"stats_apfp.csv")
    ecfp_stats = pd.read_csv(basepath_to_input+"stats_ecfp.csv")
    props_stats = pd.read_csv(basepath_to_input+"stats_props.csv")

    apfp_pca_obj = pickle.load(open(basepath_to_input+"pca_apfp.pkl","rb"))
    ecfp_pca_obj = pickle.load(open(basepath_to_input+"pca_ecfp.pkl","rb"))
    props_pca_obj = pickle.load(open(basepath_to_input+"pca_props.pkl","rb"))

    rf_apfp = pickle.load(open(basepath_to_input+"fit_rf_apfp.pkl","rb")) 
    rf_ecfp = pickle.load(open(basepath_to_input+"fit_rf_ecfp.pkl","rb")) 
    rf_props = pickle.load(open(basepath_to_input+"fit_rf_props.pkl","rb")) 

    svr_apfp = pickle.load(open(basepath_to_input+"fit_svr_apfp.pkl","rb")) 
    svr_ecfp = pickle.load(open(basepath_to_input+"fit_svr_ecfp.pkl","rb")) 
    svr_props = pickle.load(open(basepath_to_input+"fit_svr_props.pkl","rb"))

    # return a dict to simplify parsing this function's output.
    return_dict = {
                "apfp_stats" : apfp_stats,
                "ecfp_stats" : ecfp_stats,
                "props_stats" : props_stats,
                "apfp_pca_obj" : apfp_pca_obj,
                "ecfp_pca_obj" : ecfp_pca_obj,
                "props_pca_obj" : props_pca_obj,
                "rf_apfp" : rf_apfp,
                "rf_ecfp" : rf_ecfp,
                "rf_props" : rf_props,
                "svr_apfp" : svr_apfp,
                "svr_ecfp" : svr_ecfp,
                "svr_props" : svr_props
                }
    return return_dict

def normaliseTestFeats(feats, stat):
    """Given an array of features,
    Returns a normalised DataFrame and stats for test set scaling."""

    feats_df = pd.DataFrame.from_records(feats)

    def norm(x):
        return (x - stat['mean']) / stat['std']

    # Normalise and return separately.
    normed_data = norm(feats_df).fillna(0).replace([np.inf, -np.inf], 0.0)
    
    return normed_data 

def reduceTestFeatures(feats, pca):
    """Given a pd dataframe of normalised features, reduce
    to 100 dimensions using the provided PCA object that has been
    pre-fit."""
    return pca.transform(feats)

def predictWithModel(feats, model1, model2):
    """Given an array of pre-processed features, predict the y label using the
    provided pre-trained models"""
    return model1.predict(feats)[0], model2.predict(feats)[0]

def predictBaseModels(perts):
    """
    Given a list of perturbations where each item in each list is a path to an SDF file,
    - load both ligands as rdkit mol objects
    - featurise as during training
    - predict the SEM using each method & return arrays
    """
    print("Predicting using base models..")
    calc = Calculator(descriptors, ignore_3D=True)

    # load all statistics and PCA objects required for pre-processing the fingerprints.
    pp_dict = loadPostProcessors("process/base_models/")

    apfp_rf_preds, apfp_svr_preds,  ecfp_rf_preds, ecfp_svr_preds, \
    props_rf_preds, props_svr_preds = [], [], [], [], [], []

    for liga, ligb in tqdm(perts):
        liga = Chem.SDMolSupplier(liga)[0]
        ligb = Chem.SDMolSupplier(ligb)[0]

        ####### ATOM-PAIR FPs
        apfp = list(rdMolDescriptors.GetHashedAtomPairFingerprint(liga, 256))
        for bit in list(rdMolDescriptors.GetHashedAtomPairFingerprint(ligb, 256)):
            apfp.append(bit)
        apfp_normed = normaliseTestFeats([apfp], pp_dict["apfp_stats"])
        apfp_postprocessed = reduceTestFeatures(apfp_normed, pp_dict["apfp_pca_obj"])
        rf_pred, svr_pred = predictWithModel(apfp_postprocessed, 
                        pp_dict["rf_apfp"], pp_dict["svr_apfp"])
        apfp_rf_preds.append(rf_pred)
        apfp_svr_preds.append(svr_pred)


        ####### EXTENDED CONNECTIVITY FPs
        ecfp = list(AllChem.GetMorganFingerprintAsBitVect(liga,1,nBits=1024))
        for bit in list(AllChem.GetMorganFingerprintAsBitVect(ligb,1,nBits=1024)):
            ecfp.append(bit)
        ecfp_normed = normaliseTestFeats([ecfp], pp_dict["ecfp_stats"])
        ecfp_postprocessed = reduceTestFeatures(ecfp_normed, pp_dict["ecfp_pca_obj"])
        rf_pred, svr_pred = predictWithModel(ecfp_postprocessed, 
                        pp_dict["rf_ecfp"], pp_dict["svr_ecfp"])
        ecfp_rf_preds.append(rf_pred)
        ecfp_svr_preds.append(svr_pred)


        ####### MOLECULAR PROPERTIES
        liga_props = calc(liga).fill_missing(value=0)
        ligb_props = calc(ligb).fill_missing(value=0)
        dProps = np.array(list(ligb_props.values())) - np.array(list(liga_props.values()))
        #dProps_col_names = ligb_props.keys()
        props_normed = normaliseTestFeats([dProps], pp_dict["props_stats"])
        props_postprocessed = reduceTestFeatures(props_normed, pp_dict["props_pca_obj"])
        rf_pred, svr_pred = predictWithModel(props_postprocessed, 
                        pp_dict["rf_props"], pp_dict["svr_props"])
        props_rf_preds.append(rf_pred)
        props_svr_preds.append(svr_pred)

    return apfp_rf_preds, apfp_svr_preds,  ecfp_rf_preds, ecfp_svr_preds, props_rf_preds, props_svr_preds

def writeBaselModelPreds(preds, pert_paths, output_pathbase):
    """
    writes a nested list of SEM predictions to files. See the return format in predictBaseModels().
    """
    print(f"Writing predictions to {output_pathbase}..")
    pred_methods = ["apfp_rf_preds", "apfp_svr_preds",  
                    "ecfp_rf_preds", "ecfp_svr_preds", 
                    "props_rf_preds", "props_svr_preds"]
    pert_names = [ pert[0].split("/")[-1].replace(".sdf","")+"~"+pert[1].split("/")[-1].replace(".sdf","") \
                    for pert in pert_paths ]

    for pred, method in zip(preds, pred_methods):

        # write the predictions file for parsing during analysis.
        with open(output_pathbase+"_"+method, "w") as writefile:
            writer = csv.writer(writefile)
            writer.writerow(["pert_name", "pred_sem_base"])

            for pred_sem, pert_name in zip(pred, pert_names):
                inv_pert_name = f"{pert_name.split('~')[1]}~{pert_name.split('~')[0]}"
                writer.writerow([pert_name, pred_sem])
                writer.writerow([inv_pert_name, pred_sem])


if __name__ == "__main__":

    ##############################################
    ##############################################
    ##############################################
    # catch input and output file paths from CLI.


    # catch the CLI flags. This is a quick argparse fix, could do with optimisation of course.
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", help="Path to input CSV file containing input pairs of ligands to predict SEMs on. Example line: input_files_example/jmc_27.sdf,input_files_example/ejm_31.sdf",
                    type=str)

    parser.add_argument("-o", help="Path to output CSV file that will contain pairs of ligands with predicted SEMs.",
                    type=str)

    args = parser.parse_args()

    input_file = args.i
    output_file = args.o

    if input_file == None or output_file == None:
        raise Exception("Provide both -i and -o flags to the CLI. See python predict_sd.py -h for info.")

    ##############################################
    ##############################################
    ##############################################
    # with the input files, make predictions.

    ##### MODEL LOADING  
    # 1) Load SEM predictor.
    # First, build the network architecture based on reference graph inputs from training.
    fepspace_df = pd.read_csv("../ANALYSIS/perturbation_networks/process/fepspace_smiles_per_sem.csv", nrows=1)
    x_ref_0 = graphs_from_smiles(fepspace_df.ligand1_smiles)
    x_ref_1 = graphs_from_smiles(fepspace_df.ligand2_smiles)

    # 2) Build the lambda 0 and 1 legs (both are individual MPNNs).
    print("\nBuilding model..")
    fepnn = MPNNModel(
        atom_dim_0=x_ref_0[0][0][0].shape[0], bond_dim_0=x_ref_0[1][0][0].shape[0],
        atom_dim_1=x_ref_1[0][0][0].shape[0], bond_dim_1=x_ref_1[1][0][0].shape[0],
        r_group_mapping_dim=50 # fixed value
        )

    # 3) extend the model with the transfer-learned weight architecture (_02).
    fepnn = modelTransfer(fepnn, 4, 4) # adjust parameters if changes them in _02.

    # 4) now load in pre-trained weights for all five folds of all 10 replicates.
    weights_path = "rbfenn_weights/weights_finetuned"
    print(f"Loading model weights from {weights_path}_*..")
    fepnn_ensembles = loadEnsemble(5, weights_path, fepnn) 

    # -> steps 1-3 basically reproduce steps taken during training; step 4 adds trained weights.

    #### LOAD TRANSFORMATIONS AND LIGANDS
    query_transformations = []
    input_ligands = {}
    with open(input_file, "r") as readfile:
        reader = csv.reader(readfile)
        for row in reader:
            query_transformations.append(row)

            # also load ligands as SMILES
            for lig_path in row:
                if not lig_path in input_ligands:
                    input_ligands[lig_path] = Chem.MolToSmiles(
                        Chem.SDMolSupplier(lig_path)[0])

    #get the original smiles of the transformations too.
    query_smiles = []
    for a, b in query_transformations:
        a_smiles = Chem.MolToSmiles(Chem.SDMolSupplier(a)[0])
        b_smiles = Chem.MolToSmiles(Chem.SDMolSupplier(b)[0])
        query_smiles.append([a_smiles, b_smiles])

    # # CLI NOTE: we don't parameterise because we don't attempt to get the atom mapping.
    # # these two steps are currently very time-consuming and not fit for the current project.
    # # here is the original code though:
    # param_dict = {}
    # for lig_path in tqdm(ligs, total=len(ligs)):
    #     #lig = BSS.IO.readMolecules(lig_path.replace(".sdf", ".mol2"))[0]
    #     lig = Chem.MolToSmiles(Chem.SDMolSupplier(lig_path)[0])
    #     #lig, _ = parameteriseLigand(lig)
    #     param_dict[lig_path] = lig


    # #### PREDICTIONS    
    print("\nComputing features for all perturbations in the set (MCS + graphs).") 
    # 2) graft all requested transformations onto benzene       
    perts_featurised, perts_names, pert_mappings, \
    ha_change_counts, fp_simis = featurisePerturbations(query_transformations, 
                                        input_ligands)
    
    # 3) create a test set, i.e. transform the input features such that it has
    # the training set format.
    perts_dataset = createTestDataset(perts_featurised, pert_mappings)

    # 4) make predictions per ensemble.
    sf_predictions = []
    print(f"\nPredicting SFs per siamese GNN model in the ensemble (n={len(fepnn_ensembles)}).")
    for fepnn_ensemble in tqdm(fepnn_ensembles):
          #make the SEM predictions for this replicate.
          pred_sem_values_mean, pred_sem_values_std = predictEnsemble(fepnn_ensemble, perts_dataset)
          sf_predictions.append(pred_sem_values_mean)

    # get the mean prediction (i.e. mean of mean predictions as each ensemble is 5 
    # model predictions because of cross validation scheme)
    sf_predictions = np.mean(sf_predictions, axis=0)
    
    #5) write out the mean ensemble prediction to the requested output file.
    with open(output_file, "w") as writefile:
        writer = csv.writer(writefile)

        writer.writerow(["lig_1_path", "lig_2_path", "lig_1_smiles", "lig_2_smiles",
                        "lig_1_benzene_deriv", "lig_2_benzene_deriv", "sf_prediction"])

        for (a,b), (a_ori, b_ori), \
            (a_rbfespace, b_rbfe_space), pred_sf in zip(
            query_transformations, query_smiles, perts_featurised, sf_predictions):

            # write out the resulting data. 
            writer.writerow([a, b, a_ori, b_ori, \
                            a_rbfespace, b_rbfe_space, pred_sf])














