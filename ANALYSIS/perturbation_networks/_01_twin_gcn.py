#!/bin/python

# pre-processes molecule data in FEPspace and port into a GCN model; train.

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.Draw import MolsToGridImage
import logging
from tqdm import tqdm
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from glob import glob
import csv
import sys
import ast

tf.get_logger().setLevel('ERROR')
warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

np.random.seed(42)
tf.random.set_seed(42)


# code adopted mainly from https://keras.io/examples/graph/mpnn-molecular-graphs/
#################################################################################
#################################################################################
########################### DEFINE FEATURES #####################################

class Featurizer:
    def __init__(self, allowable_sets):
        self.dim = 0
        self.features_mapping = {}
        for k, s in allowable_sets.items():
            s = sorted(list(s))
            self.features_mapping[k] = dict(zip(s, range(self.dim, len(s) + self.dim)))
            self.dim += len(s)

    def encode(self, inputs):
        output = np.zeros((self.dim,))
        for name_feature, feature_mapping in self.features_mapping.items():
            feature = getattr(self, name_feature)(inputs)
            if feature not in feature_mapping:
                continue
            output[feature_mapping[feature]] = 1.0
        return output


class AtomFeaturizer(Featurizer):
    def __init__(self, allowable_sets):
        super().__init__(allowable_sets)

    def symbol(self, atom):
        return atom.GetSymbol()

    def n_valence(self, atom):
        return atom.GetTotalValence()

    def n_hydrogens(self, atom):
        return atom.GetTotalNumHs()

    def hybridization(self, atom):
        return atom.GetHybridization().name.lower()


class BondFeaturizer(Featurizer):
    def __init__(self, allowable_sets):
        super().__init__(allowable_sets)
        self.dim += 1

    def encode(self, bond):
        output = np.zeros((self.dim,))
        if bond is None:
            output[-1] = 1.0
            return output
        output = super().encode(bond)
        return output

    def bond_type(self, bond):
        return bond.GetBondType().name.lower()

    def conjugated(self, bond):
        return bond.GetIsConjugated()

#################################################################################
#################################################################################
########################### GENERATE GRAPHS #####################################

def molecule_from_smiles(smiles):
    # MolFromSmiles(m, sanitize=True) should be equivalent to
    # MolFromSmiles(m, sanitize=False) -> SanitizeMol(m) -> AssignStereochemistry(m, ...)
    molecule = Chem.MolFromSmiles(smiles, sanitize=False)

    # If sanitization is unsuccessful, catch the error, and try again without
    # the sanitization step that caused the error
    flag = Chem.SanitizeMol(molecule, catchErrors=True)
    if flag != Chem.SanitizeFlags.SANITIZE_NONE:
        Chem.SanitizeMol(molecule, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ flag)

    Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
    return molecule


def graph_from_molecule(molecule):
    # Initialize graph
    atom_features = []
    bond_features = []
    pair_indices = []

    atom_featurizer = AtomFeaturizer(
        allowable_sets={
            "symbol": {"B", "Br", "C", "Ca", "Cl", "F", "H", "I", "N", "Na", "O", "P", "S"},
            "n_valence": {0, 1, 2, 3, 4, 5, 6},
            "n_hydrogens": {0, 1, 2, 3, 4},
            "hybridization": {"s", "sp", "sp2", "sp3"},
        }
    )

    bond_featurizer = BondFeaturizer(
        allowable_sets={
            "bond_type": {"single", "double", "triple", "aromatic"},
            "conjugated": {True, False},
        }
    )

    for atom in molecule.GetAtoms():
        atom_features.append(atom_featurizer.encode(atom))

        # Add self-loop. Notice, this also helps against some edge cases where the
        # last node has no edges. Alternatively, if no self-loops are used, for these
        # edge cases, zero-padding on the output of the edge network is needed.
        pair_indices.append([atom.GetIdx(), atom.GetIdx()])
        bond_features.append(bond_featurizer.encode(None))

        atom_neighbors = atom.GetNeighbors()

        for neighbor in atom_neighbors:
            bond = molecule.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
            pair_indices.append([atom.GetIdx(), neighbor.GetIdx()])
            bond_features.append(bond_featurizer.encode(bond))

    return np.array(atom_features), np.array(bond_features), np.array(pair_indices)


def graphs_from_smiles(smiles_list):
    # Initialize graphs
    atom_features_list = []
    bond_features_list = []
    pair_indices_list = []

    for smiles in smiles_list:
        molecule = molecule_from_smiles(smiles)
        atom_features, bond_features, pair_indices = graph_from_molecule(molecule)

        atom_features_list.append(atom_features)
        bond_features_list.append(bond_features)
        pair_indices_list.append(pair_indices)

    # Convert lists to ragged tensors for tf.data.Dataset later on
    return (
        tf.ragged.constant(atom_features_list, dtype=tf.float32),
        tf.ragged.constant(bond_features_list, dtype=tf.float32),
        tf.ragged.constant(pair_indices_list, dtype=tf.int64),
    )

#################################################################################
#################################################################################
########################### GENERATE TENSORFLOW DATASETS ########################

def prepare_batch(x_batch_0, x_batch_1, x_mapping_batch, y_batch):
    """Merges (sub)graphs of batch into a single global (disconnected) graph
    
    NB: this function was adjusted to cope with dual inputs! see:
    https://github.com/tensorflow/tensorflow/issues/34912
    https://github.com/tensorflow/tensorflow/issues/47965

    bit odd, but need this workaround to get the model to take in the 
    two inputs. This will likely be streamlined in future versions of tf.

    """
    def prepper(x_batch):
        atom_features, bond_features, pair_indices = x_batch

        # Obtain number of atoms and bonds for each graph (molecule)
        num_atoms = atom_features.row_lengths()
        num_bonds = bond_features.row_lengths()

        # Obtain partition indices. atom_partition_indices will be used to
        # gather (sub)graphs from global graph in model later on
        molecule_indices = tf.range(len(num_atoms))
        atom_partition_indices = tf.repeat(molecule_indices, num_atoms)
        bond_partition_indices = tf.repeat(molecule_indices[:-1], num_bonds[1:])

        # Merge (sub)graphs into a global (disconnected) graph. Adding 'increment' to
        # 'pair_indices' (and merging ragged tensors) actualizes the global graph
        increment = tf.cumsum(num_atoms[:-1])
        increment = tf.pad(
            tf.gather(increment, bond_partition_indices), [(num_bonds[0], 0)]
        )
        pair_indices = pair_indices.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
        pair_indices = pair_indices + increment[:, tf.newaxis]
        atom_features = atom_features.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
        bond_features = bond_features.merge_dims(outer_axis=0, inner_axis=1).to_tensor()

        return (atom_features, bond_features, pair_indices, atom_partition_indices)

    x_batch_0 = prepper(x_batch_0)
    x_batch_1 = prepper(x_batch_1)


    return (x_batch_0, x_batch_1, x_mapping_batch), y_batch


def MPNNDataset(X_0, X_1, X_map, y, batch_size=50, shuffle=False):
    dataset = tf.data.Dataset.from_tensor_slices((X_0, X_1, X_map, (y)))

    if shuffle:
        dataset = dataset.shuffle(1024)
    return dataset.batch(batch_size, drop_remainder=True).map(prepare_batch, -1)
#################################################################################
#################################################################################
################################### MPNN BUILDING ###############################

class EdgeNetwork(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.atom_dim = input_shape[0][-1]
        self.bond_dim = input_shape[1][-1]
        self.kernel = self.add_weight(
            shape=(self.bond_dim, self.atom_dim * self.atom_dim),
            trainable=True,
            initializer="glorot_uniform",
            name="edge_network_bias"
        )
        self.bias = self.add_weight(
            shape=(self.atom_dim * self.atom_dim), trainable=True, initializer="zeros",
            name="edge_network_weight"
        )
        self.built = True

    def call(self, inputs):
        atom_features, bond_features, pair_indices = inputs

        # Apply linear transformation to bond features
        bond_features = tf.matmul(bond_features, self.kernel) + self.bias

        # Reshape for neighborhood aggregation later
        bond_features = tf.reshape(bond_features, (-1, self.atom_dim, self.atom_dim))

        # Obtain atom features of neighbors
        atom_features_neighbors = tf.gather(atom_features, pair_indices[:, 1])
        atom_features_neighbors = tf.expand_dims(atom_features_neighbors, axis=-1)

        # Apply neighborhood aggregation
        transformed_features = tf.matmul(bond_features, atom_features_neighbors)
        transformed_features = tf.squeeze(transformed_features, axis=-1)
        aggregated_features = tf.math.segment_sum(
            transformed_features, pair_indices[:, 0]
        )
        return aggregated_features

class MessagePassing(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.units = 64
        self.steps = 4

    def build(self, input_shape):
        self.atom_dim = input_shape[0][-1]
        self.message_step = EdgeNetwork()
        self.pad_length = max(0, self.units - self.atom_dim)
        self.update_step = layers.GRUCell(self.atom_dim + self.pad_length)
        self.built = True

    def get_config(self):
        config = super(MessagePassing, self).get_config()
        return config

    def call(self, inputs):
        atom_features, bond_features, pair_indices = inputs

        # Pad atom features if number of desired units exceeds atom_features dim
        atom_features_updated = tf.pad(atom_features, [(0, 0), (0, self.pad_length)])

        # Perform a number of steps of message passing
        for i in range(self.steps):
            # Aggregate atom_features from neighbors
            atom_features_aggregated = self.message_step(
                [atom_features_updated, bond_features, pair_indices]
            )

            # Update aggregated atom_features via a step of GRU
            atom_features_updated, _ = self.update_step(
                atom_features_aggregated, atom_features_updated
            )
        return atom_features_updated

class PartitionPadding(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = 50

    def get_config(self):
        config = super(PartitionPadding, self).get_config()
        return config

    def call(self, inputs):
        atom_features, atom_partition_indices = inputs

        # Obtain subgraphs
        atom_features = tf.dynamic_partition(
            atom_features, atom_partition_indices, self.batch_size
        )

        # Pad and stack subgraphs
        num_atoms = [tf.shape(f)[0] for f in atom_features]
        max_num_atoms = tf.reduce_max(num_atoms)
        atom_features_padded = tf.stack(
            [
                tf.pad(f, [(0, max_num_atoms - n), (0, 0)])
                for f, n in zip(atom_features, num_atoms)
            ],
            axis=0,
        )

        # Remove empty subgraphs (usually for last batch)
        nonempty_examples = tf.where(tf.reduce_sum(atom_features_padded, (1, 2)) != 0)
        nonempty_examples = tf.squeeze(nonempty_examples, axis=-1)

        return tf.gather(atom_features_padded, nonempty_examples, axis=0)

class TransformerEncoder(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.attention = layers.MultiHeadAttention(8, 64)
        self.dense_proj = keras.Sequential(
            [layers.Dense(512, activation="relu"), layers.Dense(64),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.supports_masking = True

    def get_config(self):
        config = super(TransformerEncoder, self).get_config()
        return config

    def call(self, inputs, mask=None):
        attention_mask = mask[:, tf.newaxis, :] if mask is not None else None
        attention_output = self.attention(inputs, inputs, attention_mask=attention_mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        return self.layernorm_2(proj_input + self.dense_proj(proj_input))

def MPNNModel(
    atom_dim_0,
    bond_dim_0,
    atom_dim_1,
    bond_dim_1,
    r_group_mapping_dim,
    batch_size=50,
    message_units=64,
    message_steps=4,
    num_attention_heads=8,
    dense_units=512,
):
    atom_features_0 = layers.Input((atom_dim_0), dtype="float32", name=f"atom_features_0")
    bond_features_0 = layers.Input((bond_dim_0), dtype="float32", name=f"bond_features_0")
    pair_indices_0 = layers.Input((2), dtype="int32", name=f"pair_indices_0")
    atom_partition_indices_0 = layers.Input(
        (), dtype="int32", name=f"atom_partition_indices_0"
    )

    atom_features_1 = layers.Input((atom_dim_1), dtype="float32", name=f"atom_features_1")
    bond_features_1 = layers.Input((bond_dim_1), dtype="float32", name=f"bond_features_1")
    pair_indices_1 = layers.Input((2), dtype="int32", name=f"pair_indices_1")
    atom_partition_indices_1 = layers.Input(
        (), dtype="int32", name=f"atom_partition_indices_1"
    )

    r_group_mapping = layers.Input((r_group_mapping_dim), dtype="int32", name=f"r_group_mapping")

    #############
    # Create the layers first; then call them twise to make layers share weights.
    # see https://keras.io/guides/functional_api/#shared-layers

    # custom:
    l_mp = MessagePassing()
    l_pp = PartitionPadding()
    l_ms = layers.Masking()
    l_te = TransformerEncoder()

    # out of the box:
    l_gap = layers.GlobalAveragePooling1D()
    l_d1 = layers.Dense(dense_units, activation="relu")
    l_d2 = layers.Dense(450, activation="relu")

    ##############
    # now call layers to encode both inputs by calling each layer twice.
    x_0 = l_d2(l_d1(l_gap(
        l_te(l_ms(l_pp(
            [l_mp([atom_features_0, bond_features_0, pair_indices_0]), atom_partition_indices_0]))))))

    x_1 = l_d2(l_d1(l_gap(
        l_te(l_ms(l_pp(
            [l_mp([atom_features_1, bond_features_1, pair_indices_1]), atom_partition_indices_1]))))))

    ##############
    # create a model subsection made of FCNN layers that encodes the R-group mapping.
    """A simple FCNN that encodes atom mapping between benzene derivates. The array
    contains 6 integers, where the index of the integer refers to the index of the R-group in
    ligand 1, whereas the value of the integer refers to the index of the R-group in ligand 2."""
    x_rgm = layers.Dense(10, activation="relu")(r_group_mapping)
    x_rgm = layers.Dense(10, activation="relu")(x_rgm)
    outputs_rgm = layers.Dense(5)(x_rgm)

    ##############
    # concatenate all layers together.
    # note the use of underscore concatenate!
    combined = tf.keras.layers.concatenate([x_0, x_1, outputs_rgm])

    # make some more FCNN layers after the concatenation that will primarily learn the delta property.
    head_model = layers.Dense(700, activation="relu")(combined)
    head_model = layers.Dense(450, activation="relu")(head_model)
    head_model = layers.Dense(100, activation="relu")(head_model)
    head_model = layers.Dense(1, activation="linear")(head_model)

    model = tf.keras.Model(
    inputs=[atom_features_0, bond_features_0, pair_indices_0, atom_partition_indices_0,
    atom_features_1, bond_features_1, pair_indices_1, atom_partition_indices_1, r_group_mapping],
    outputs=[head_model],
    )

    return model

def featuriseMapping(mapping):
    """Given a dictionary mapping, create an array of length 50, where the index 
    of the integer refers to the index of the R-group in ligand 1, whereas the 
    value of the integer refers to the index of the R-group in ligand 2.
    """
    # make an empty array - during testing we found atom idx 39 to be the highest number, so
    # take 50 to be safe.
    mapping_array = np.ones(50)*99

    # populate the array with data.
    if mapping:
        for k, v in mapping.items():
            mapping_array[k] = v

    return mapping_array

def retrieveRGroupMappings(tgt, pert):
    """For a given perturbation, find the R group mapping in the original
    FEP-Space setup files."""
    path_to_mappings = "../../FEPSPACE_TRAIN/MAIN/"

    info_path = f"{path_to_mappings}{tgt}/{pert}/pert_info.txt"
    try:
        with open(info_path, "r") as info_file:
            reader = csv.reader(info_file)

            for row in reader:
                if "MAPPING" in row[0]:
                    g = ast.literal_eval(" ".join(row[0].split(" ")[1:]))
    except FileNotFoundError:
        g = None

        
    return featuriseMapping(g)
    




# @@@@@@@@@@@@@@@@@@@@@  


if __name__ == "__main__":
    print("Loading input data..")

    train_input = pd.read_csv("process/inputs/rel_esol_fepspace_train.csv", nrows=800000)
    valid_input = pd.read_csv("process/inputs/rel_esol_fepspace_valid.csv", nrows=200000)

    train_mapping_arrays = [ retrieveRGroupMappings(*a) for a in tuple(zip(
                        train_input["target"], train_input["ligand1_original_pert_name"])) ]

    valid_mapping_arrays = [ retrieveRGroupMappings(*a) for a in tuple(zip(
                        valid_input["target"], valid_input["ligand1_original_pert_name"])) ]

    ############### featurise and split the input data ##################
    print("\nGenerating graphs from SMILES..")
    # Train set: 80 % of data. These have been split in the previous script before pairing, 
    # such that there is no ligand overlap between train/valid.
    print("Training set..")
    x_train_0 = graphs_from_smiles(train_input.ligand1_smiles)
    x_train_1 = graphs_from_smiles(train_input.ligand2_smiles)
    y_train = train_input.relative_solubility
    train_set = MPNNDataset(x_train_0, x_train_1, train_mapping_arrays, y_train)

    print("Validation set..")
    x_valid_0 = graphs_from_smiles(valid_input.ligand1_smiles)
    x_valid_1 = graphs_from_smiles(valid_input.ligand2_smiles)
    y_valid = valid_input.relative_solubility
    valid_set = MPNNDataset(x_valid_0, x_valid_1, valid_mapping_arrays, y_valid)

    # # can omit test set generation because we only care about learning molecular structures.
   
    # ############## build MPNN MODELS ##################
    # Build the lambda 0 and 1 legs (both are individual MPNNs).
    print("\nBuilding model..")
    fepnn = MPNNModel(
        atom_dim_0=x_train_0[0][0][0].shape[0], bond_dim_0=x_train_0[1][0][0].shape[0],
        atom_dim_1=x_train_1[0][0][0].shape[0], bond_dim_1=x_train_1[1][0][0].shape[0],
        r_group_mapping_dim=valid_mapping_arrays[0].shape[0]
        )


    print("\nCompiling..")
    fepnn.compile(
        loss=keras.losses.LogCosh(),
        optimizer=keras.optimizers.Adam(learning_rate=5e-7),
        metrics=['mae'],
    )

    # save a rough sketch of the model. Will manually draw a schematic for the paper from this.
    keras.utils.plot_model(fepnn, to_file='process/twin_model.png', show_dtype=True, show_shapes=True)
    print(fepnn.summary())


    # $$$$$$$$$$$$$$$$$$ TRAINING $$$$$$$$$$$$$$$$$$$$$

    print("\nFitting..")
    es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_mse', 
        patience=5, restore_best_weights=True, verbose=1)


    history = fepnn.fit(
        train_set,
        validation_data=valid_set,
        epochs=98,
        verbose=2,
        callbacks=[es_callback]
    )

    # save training information.
    print("Saving training information..")
    pd.DataFrame(history.history).to_csv("process/pretraining_history.csv")

    plt.plot(history.history['loss'], label="Train")
    plt.plot(history.history['val_loss'], label="Validation")
    plt.ylabel('Loss - LogCosh')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.savefig('process/pretraining_history_plot.png')

    # save model weights. The model can be restored in later scripts by reconstructing
    # the classes in this script and loading the weights.
    weights_path = "process/trained_model_weights/weights"
    print(f"Saving model weights to {weights_path}..")
    fepnn.save_weights(weights_path)

    print("\nDone.")
    
    



