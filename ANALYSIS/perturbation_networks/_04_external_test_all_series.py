#!/bin/python

# Create the FEPSpace variant of all ligands in an external test set (i.e. congeneric series). 
# Load the transfer-learned model, predict SEMs. Compare to TRUE SEMs. 
# Write predictions to file to generate networks with in _04.

# This code is largely copied from 

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

import glob
import csv
import copy
import itertools
import random
# import code to regenerate the twin GCN.
from _01_twin_gcn import *
from _02_transfer_learn_sem import modelTransfer


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

    for frag_idx in non_mcs_indices:
        # get the neighbours for this fragment atom.
        nghbrs = lig_mol.GetAtomWithIdx(frag_idx).GetNeighbors()

        for nghbr in nghbrs:
            # find the set difference.
            if not nghbr.GetIdx() in non_mcs_indices:
                anchor_atoms.append(lig_mol.GetAtomWithIdx(nghbr.GetIdx()).GetSmarts())

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
        
        # if trihalo, we can merge that into a single R group (i.e. fragment).
        if "[C*]F.[C*]F.[C*]F" in frags_smarts:
            frags_smarts = frags_smarts.replace("[C*]F.[C*]F.[C*]F", "[C*](F)(F)F")
        elif "[C*]Cl.[C*]Cl.[C*]Cl" in frags_smarts:
            frags_smarts = frags_smarts.replace("[C*]Cl.[C*]Cl.[C*]Cl", "[C*](Cl)(Cl)Cl")
        
        # replace SMARTS notation style for CH with C (will protonate later in workflow anyway)
        frags_smarts = frags_smarts.replace("CH", "C")
        frags_smarts = frags_smarts.replace("C@H", "C")
    
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
            
            # record if this fragment contains a fused ring (multiple wildcards).  
            if frag_parsed.count("*") == 2:
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
            
        # count the number of wildcards in this set of fragments.
        if fr.count("*") == 1:
            # in this case we can simply graft the structure onto benzene.
            anchor = fr[0]
            if anchor == "c":
                # removing aromatic anchor will make r_group graft onto benzene directly (making it use the aromatic 
                # benzene carbon as anchor).
                anchor = ""
            r_group = fr[5:]

            scaffold_mol = Chem.ReplaceSubstructs(scaffold_mol, 
                                     Chem.MolFromSmiles('C'), 
                                     Chem.MolFromSmiles("C"+anchor+r_group),
                                     replaceAll=False,
                                    )[0]


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
    # all done. Checks?
    
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
    models_collection = []

    for k in range(len_k):
        k_model = tf.keras.models.clone_model(input_model)
        k_model.load_weights(f"{weights_path}_{k}")
        models_collection.append(k_model)

    return models_collection

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


def parameteriseLigand(input_ligand):
    """Parameterise an input BSS ligand structure with GAFF2 from SMILES input. returns the parameterised ligand
    and the used SMILES."""
    if os.path.exists("tmp_setup"):
        shutil.rmtree("tmp_setup")
    
    try:
        input_ligand_p = BSS.Parameters.parameterise(input_ligand, forcefield="GAFF2").getMolecule()
    
    except BSS._Exceptions.ParameterisationError:
        # introduce stereochemistry, see https://github.com/openforcefield/openff-toolkit/issues/146
        try:
            input_ligand_p = BSS.Parameters.parameterise(input_ligand.replace("[C]", "[C@H]"), forcefield="GAFF2").getMolecule()
    
        except BSS._Exceptions.ParameterisationError:
            # if it fails again, OFF might be struggling with the input SMILES. For these edge-cases 
            # sometimes it helps shuffling the order of SMILES. Try a few times.
            for attempt in range(5):
                try:
                    # use rdkit to write alternative SMILES.
                    tmpmol = Chem.MolFromSmiles(input_ligand)
                    newsmiles = Chem.MolToSmiles(tmpmol, doRandom=True)
                    print("Retrying with SMILES shuffle:", newsmiles)
                    input_ligand_p = BSS.Parameters.parameterise(newsmiles, forcefield="GAFF2").getMolecule()
                    print("Success!")
                    # return the new smiles as well. 
                    input_ligand = newsmiles
                    break
                    
                except BSS._Exceptions.ParameterisationError:
                    input_ligand_p = None 

        
    if input_ligand_p == None:
        print("Bad input, returning None:", input_ligand)
        
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
        # this mapping creates a very funky perturbation; discard.
        return {}, None, [[]]

    # Get indices of perturbed atoms.
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

    In its current form this step is prohibitively rate-limiting. For a usable implementation
    this will need to be refactored to not require parameterisation for both the input 
    and fep-space ligands.
    """

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
    pert_features, pert_names, pert_mappings, ha_change_counts, fp_simis = [], [], [], [], []

    
    for pert in tqdm(perts, total=len(perts)):
      pert_name = pert[0].split("/")[-1].replace(".sdf", "")+"~"+pert[1].split("/")[-1].replace(".sdf", "")

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

      # get the alternative perturbation SMARTS
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

      # graft the R groups to a benzene scaffold.
      try:
          abstract_mol_1 = graftToScaffold(frags_1)
          abstract_mol_2 = graftToScaffold(frags_2)
      except:
          # fails because of charge jump. instead, denote as just benzene abstracts and catch
          # later using ha_change_count; this perturbation can be set to have a high SEM.
          abstract_mol_1 = abstract_mol_2 = Chem.MolFromSmiles("c1ccccc1")
          ha_change_count = str(ha_change_count)+"_fail"

      if not abstract_mol_1 or not abstract_mol_2:
          # fails because of complex (or too many) fused ring jumps. handle as above.
          abstract_mol_1 = abstract_mol_2 = Chem.MolFromSmiles("c1ccccc1")
          ha_change_count = str(ha_change_count)+"_fail"


      if not "_fail" in str(ha_change_count):
          # get the mapping for this perturbation.
          try:
            mapping_array = getMapping(param_dict[pert[0]], param_dict[pert[1]], mcs, abstract_mol_1, abstract_mol_2)

          except BSS._Exceptions.AlignmentError:
            # in very rare cases BSS alignment fails for very large perturbations; for 
            # these cases return a standard mapping. 
            mapping_array = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5}
      else:
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


if __name__ == "__main__":
    ####################################################################
    ####################################################################    
    ## Load SEM predictor.
    # First, build the network architecture based on reference graph inputs from training.
    fepspace_df = pd.read_csv("process/fepspace_smiles_per_sem.csv", nrows=1)
    x_ref_0 = graphs_from_smiles(fepspace_df.ligand1_smiles)
    x_ref_1 = graphs_from_smiles(fepspace_df.ligand2_smiles)

    # Build the lambda 0 and 1 legs (both are individual MPNNs).
    print("\nBuilding model..")
    fepnn = MPNNModel(
        atom_dim_0=x_ref_0[0][0][0].shape[0], bond_dim_0=x_ref_0[1][0][0].shape[0],
        atom_dim_1=x_ref_1[0][0][0].shape[0], bond_dim_1=x_ref_1[1][0][0].shape[0],
        r_group_mapping_dim=50 # fixed value
        )

    # extend the model with the transfer-learned weight architecture (_02).
    fepnn = modelTransfer(fepnn, 4, 4) # adjust parameters if changes them in _02.

    # now load in pre-trained weights.
    weights_path = "process/trained_model_weights/weights_finetuned"
    print(f"Loading model weights from {weights_path}_*..")
    fepnn_ensemble = loadEnsemble(10, weights_path, fepnn) 

    # write files with SEM predictions per perturbation, per target in all available benchmarking series.
    walltime_storage = []
    for tgt in glob("/home/jscheen/projects/FEPSPACE/fep_ref_ligands/*"):

      if "bace" in tgt:
        # Exclude BACE as too many perturbations have double fused rings.
        continue
      start = time.time()

      ligs = glob(f"{tgt}/*.sdf")
      if len(ligs) >= 2:
          print("\n"+"$"*50)
          print(tgt)
          perts = list(itertools.combinations(ligs, 2))
      else:
        # don't work on this target if only two ligands are in the set.
        continue

      # create a dictionary that contains all parameterised ligands.
      print("Parameterising..")
      param_dict = {}
      for lig_path in tqdm(ligs, total=len(ligs)):
        lig = BSS.IO.readMolecules(lig_path.replace(".sdf", ".mol2"))[0]
        lig_p, _ = parameteriseLigand(lig)
        param_dict[lig_path] = lig_p

      with open(f"output/series_predictions/{tgt.split('/')[-1]}.csv", "w") as preds_file:
          writer = csv.writer(preds_file)
          writer.writerow(["pert_name", "pred_sem_mean", "pred_sem_std", 
                        "random_sem", "num_ha_change", "fp_similarity"])

          print("\nComputing features for all possible perturbations in the set (MCS + graphs).")
          
          perts_featurised, perts_names, pert_mappings, \
                    ha_change_counts, fp_simis = featurisePerturbations(perts, param_dict)

          perts_dataset = createTestDataset(perts_featurised, pert_mappings)
            
          print("\nPredicting SEM per input perturbation.")
          # make the SEM predictions.
          pred_sem_values_mean, pred_sem_values_std = predictEnsemble(fepnn_ensemble, perts_dataset)
          
          if not len(pred_sem_values_mean) == len(perts_names) == len(ha_change_counts):
            raise ValueError(f"SEM predictions, perturbation names and num_ha_change " +\
                f"variables do not match in length! {(pred_sem_values_mean)}, {(perts_names)}, {(ha_change_counts)}")
          
          print(f"\nDone. Writing to output/series_predictions/{tgt.split('/')[-1]}.csv.")
          fail_counter = 0

          for pred_sem_mean, pred_sem_std, pert_name, ha_change_count, fp_simi in zip(
                                                pred_sem_values_mean,
                                                pred_sem_values_std,
                                                perts_names,
                                                ha_change_counts,
                                                fp_simis):

            # also make a random SEM prediction to use as negative control later on.
            random_sem_value = random.uniform(0, 1)

            # replace SEM preds and restore ha_change variable if failed previously during featurisation
            # (formal charge perturbation issue.)
            if "_fail" in str(ha_change_count):
                ha_change_count = int(ha_change_count.replace("_fail",""))
                pred_sem_mean = max(pred_sem_values_mean)
                pred_sem_std = max(pred_sem_values_std)
                fail_counter += 1
            


            # write results to file. These will be analysed in _04.
            writer.writerow([
                              pert_name,
                              pred_sem_mean,
                              pred_sem_std,
                              random_sem_value,
                              ha_change_count,
                              fp_simi
                              ])
          if fail_counter > 0:
            print(f"{fail_counter} perturbations failed because of complex fused rings or a change in formal charge. SEM_pred for these was set to {round(max(pred_sem_values_mean), 3)} kcal/mol.")
          end = time.time()
          walltime_storage.append([tgt, end-start])





    print("DONE.")
    with open("process/walltime_ml_pred_per_tgt.csv", "w") as writefile:
        writer = csv.writer(writefile)
        for row in walltime_storage:
            writer.writerow(row)













