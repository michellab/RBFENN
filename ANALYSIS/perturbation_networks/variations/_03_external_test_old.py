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


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import re

import subprocess

import os
import glob
import csv
from tqdm.notebook import tqdm
import itertools
import random
random.uniform(0, 1)
# import code to regenerate the twin GCN.
from _01_twin_gcn import *

def CountHAChange(fragment1_mol, fragment2_mol):
    """Takes in two rdkit fragment molecules, counts heavy atom changes and returns the number."""
    fragA_smiles = Chem.MolToSmiles(fragA)
    fragB_smiles = Chem.MolToSmiles(fragB)  
    
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
        print("Aborting this pert --> two or more fused rings being perturbed.")
        return None

    # abort this perturbation if there are too many R groups such that the 6 carbons on benzene are not enough.
    if frag.count(".") >= 5:
        print("Aborting this pert --> more than 6 R groups.")
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
                    print("Aborting this pert --> miscellaneous fused ring issue.")
                    return None
        
        # after grafting we need to sanitize.
        try:
            Chem.SanitizeMol(scaffold_mol)
        except:
            print("Aborting this pert --> miscellaneous ring issue.")
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
        print("Aborting this pert --> fused ring double bond interrupts scaffold aromaticity.")
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
	mpnn_lam0 = MPNNModel(
		atom_dim=x_ref_0[0][0][0].shape[0], bond_dim=x_ref_0[1][0][0].shape[0],
		lambda_val=0
	)
	mpnn_lam1 = MPNNModel(
		atom_dim=x_ref_1[0][0][0].shape[0], bond_dim=x_ref_1[1][0][0].shape[0],
		lambda_val=1
	)

	# concatenate them (i.e. merge).
	combined = tf.keras.layers.Concatenate()([mpnn_lam0.output, mpnn_lam1.output])

	# make some more FCNN layers after the concatenation that will learn the delta property.
	# note: these layers should match the last n layers that were attached during transfer learning!
	# to find these, check either print(transfer_learned_model.summary()) in _02, or add layers 
	# based on tf exceptions thrown when loading weights below.
	head_model = layers.Dense(700, activation="relu")(combined)
	head_model = layers.Dense(450, activation="relu")(head_model)
	head_model = layers.Dense(360, activation="relu")(head_model)
	head_model = layers.Dense(240, activation="relu")(head_model)
	head_model = layers.Dense(120, activation="relu")(head_model)
	head_model = layers.Dense(1, activation="linear")(head_model)

	# build up the twin model.
	sem_predictor = keras.Model(inputs=[mpnn_lam0.input, mpnn_lam1.input], outputs=head_model)

	# now load in pre-trained weights.
	weights_path = "process/trained_model_weights/weights_finetuned"
	print(f"Loading model weights from {weights_path}..")
	sem_predictor.load_weights(weights_path)

	# # predict on JNK1 as an external test set. We first load the ligands, then generate each perturbation's
	# # FEP-Space derivatives, then featurise and predict. We store the (predicted) SEMs in a CSV that
	# # we will analyse in _04.
	# sem_collector = []
	# with open("input/JNK1_FULLY_CONNECTED_TESTSET/freenrg_data/sems_for_lomap_bi.csv", "r") as sems_file, \
	# 	open("output/sem_predictions_jnk1.csv", "w") as preds_file:
	# 	reader = csv.reader(sems_file)
	# 	writer = csv.writer(preds_file)
	# 	writer.writerow(["pert_name", "true_sem", "pred_sem", "random_sem", "num_ha_change"])

	# 	print("\nPredicting SEM per input perturbation.")
	# 	for row in tqdm(reader, total=420):
	# 		# get the SEM value as calculated from a quintuplicate SOMD run (original ligand in bound phase).
	# 		true_sem_value = float(row[2])

	# 		####################################################################
	# 		####################################################################
	# 		# now find the FEP-Space derivative of these ligands given their MCS.
	# 		ligA = Chem.SDMolSupplier(f"input/JNK1_FULLY_CONNECTED_TESTSET/ligand_files/{row[0]}.sdf")[0]
	# 		ligB = Chem.SDMolSupplier(f"input/JNK1_FULLY_CONNECTED_TESTSET/ligand_files/{row[1]}.sdf")[0]

	# 		# get MCS and fragments (i.e. perturbed R-groups).
	# 		mcs = rdFMCS.FindMCS([ligA, ligB], ringMatchesRingOnly=True, completeRingsOnly=True)

	# 		# subtract the MCS to get the perturbed (i.e. non-mcs) atoms.
	# 		fragA = Chem.ReplaceCore(ligA, Chem.MolFromSmarts(mcs.smartsString))
	# 		fragB = Chem.ReplaceCore(ligB, Chem.MolFromSmarts(mcs.smartsString))

	# 		# count the number of perturbed heavy atoms.
	# 		ha_change_count, fragA_smiles, fragB_smiles = CountHAChange(fragA, fragB)

	# 		# get the alternative perturbation SMARTS
	# 		pert_smartsA = constructSmarts(ligA, mcs)
	# 		pert_smartsB = constructSmarts(ligB, mcs)
	# 		pert_smarts = pert_smartsA+"~"+pert_smartsB

	# 		frags_1, frags_2 = rewriteSMARTS(pert_smarts) 

	# 		# we've excluded charge perturbations from the training set as these were deemed out of scope.
	# 		# for this test set just neutralise the charges. 
	# 		frags_1 = frags_1.replace("+[", "[")
	# 		frags_2 = frags_2.replace("+[", "[")

	# 		# graft the R groups to a benzene scaffold. 
	# 		abstract_mol_1 = graftToScaffold(frags_1)
	# 		abstract_mol_2 = graftToScaffold(frags_2)

	# 		# Get the SMILES for each FEP-Space derivative.
	# 		ligA_fepspace = Chem.MolToSmiles(abstract_mol_1)
	# 		ligB_fepspace = Chem.MolToSmiles(abstract_mol_2)

	# 		####################################################################
	# 		####################################################################
	# 		## Predict SEM.
	# 		# Make graphs from this perturbation to predict on.
	# 		ligA_graph = graphs_from_smiles([ligA_fepspace])
	# 		ligB_graph = graphs_from_smiles([ligB_fepspace])

	# 		# make a 'batch graph' of this perturbation, for the sake of simplicity provide the function
	# 		# with a y value of 0. Currently this step is a significant bottle-neck in terms of computing time, 
	# 		# it would be well worth optimising this. Would potentially require some updates to the tensorflow
	# 		# API wrt how it generates ragged tensors.
	# 		pert_dataset = MPNNDataset(ligA_graph, ligB_graph, [0], batch_size=1)

	# 		# make the SEM prediction.
	# 		pred_sem_value = sem_predictor.predict(pert_dataset)[0][0]

	# 		# also make a random SEM prediction to use as negative control later on.
	# 		random_sem_value = random.uniform(0, 1)

	# 		# write results to file. These will be analysed in _04.
	# 		writer.writerow([
	# 							row[0]+"~"+row[1],
	# 							true_sem_value,
	# 							pred_sem_value,
	# 							random_sem_value,
	# 							ha_change_count
	# 							])
    # predict on a collection of SEMs kindly provided by Cresset as an external test set. We first 
    #load the ligands, then generate each perturbation's FEP-Space derivatives, then featurise and 
    #predict. We store the (predicted) SEMs in a CSV that we will analyse in _04.
	sem_collector = []
	with open("input/testsets/cresset_quintuplicates_jacs/sems_ddgs_cresset_quintuplicates.csv", "r") as sems_file, \
	open("output/sem_predictions_cresset_quintup.csv", "w") as preds_file:
		reader = csv.reader(sems_file)
		next(reader)
		writer = csv.writer(preds_file)
		writer.writerow(["tgt","pert_name", "true_sem", "pred_sem", "random_sem", "num_ha_change"])

		tmp_counter = 0
		print("\nPredicting SEM per input perturbation.")
		for row in tqdm(reader, total=214):
			# get the SEM value as calculated from a quintuplicate SOMD run (original ligand in bound phase).
			true_sem_value = float(row[2])
			lig_1, lig_2 = row[1].split("~")

			tgt = row[0].lower()
			# fix some name discrepancies. 
			if tgt == "mcl":
				tgt = "mcl1"
			####################################################################
			####################################################################
			# now find the FEP-Space derivative of these ligands given their MCS.
			try:
				ligA = Chem.SDMolSupplier(f"../../fep_ref_ligands/{tgt}/lig_{lig_1}.sdf")[0]
				ligB = Chem.SDMolSupplier(f"../../fep_ref_ligands/{tgt}/lig_{lig_2}.sdf")[0]
			except OSError:
				# one of the ligands is an intermediate - discard this perturbation.
				continue

			# get MCS and fragments (i.e. perturbed R-groups).
			mcs = rdFMCS.FindMCS([ligA, ligB], ringMatchesRingOnly=True, completeRingsOnly=True)

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
			frags_1 = frags_1.replace("+[", "[")
			frags_2 = frags_2.replace("+[", "[")

			# graft the R groups to a benzene scaffold. 
			abstract_mol_1 = graftToScaffold(frags_1)
			abstract_mol_2 = graftToScaffold(frags_2)

			# grafting is canceled in cases of a double fused ring perturbation.
			if abstract_mol_1 and abstract_mol_2:
					
				# Get the SMILES for each FEP-Space derivative.
				ligA_fepspace = Chem.MolToSmiles(abstract_mol_1)
				ligB_fepspace = Chem.MolToSmiles(abstract_mol_2)

				####################################################################
				####################################################################
				## Predict SEM.
				# Make graphs from this perturbation to predict on.
				ligA_graph = graphs_from_smiles([ligA_fepspace])
				ligB_graph = graphs_from_smiles([ligB_fepspace])

				# make a 'batch graph' of this perturbation, for the sake of simplicity provide the function
				# with a y value of 0. Currently this step is a significant bottle-neck in terms of computing time, 
				# it would be well worth optimising this. Would potentially require some updates to the tensorflow
				# API wrt how it generates ragged tensors.
				pert_dataset = MPNNDataset(ligA_graph, ligB_graph, [0], batch_size=1)

				# make the SEM prediction.
				pred_sem_value = sem_predictor.predict(pert_dataset)[0][0]

				# also make a random SEM prediction to use as negative control later on.
				random_sem_value = random.uniform(0, 1)

				# write results to file. These will be analysed in _04.
				writer.writerow([
								row[0],row[1],
								true_sem_value,
								pred_sem_value,
								random_sem_value,
								ha_change_count
								])
	print("\nDone.")











