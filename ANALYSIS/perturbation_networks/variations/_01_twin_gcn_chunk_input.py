#!/bin/python

# pre-processes molecule data in FEPspace and port into a GCN model; train.
# variation: load input data as chunks, train sequentially.
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

def graphs_from_smiles_batch(input_data_paths):
    """Creates batches of tensor data and saves as .npy
    This allows feeding large (>1M) datasets into the model.
    """
    savepath = input_data_paths.replace("smiles*","tensors")
    # check if we can abort this function because tensors have already been written to the 
    # target path.
    if len(glob(savepath+"/*.npy")) > 10:
        print(f"Tensor batch files already written to {savepath}. Skipping this step.")

        return None


    all_paths = glob(input_data_paths)
    for i, data_path in tqdm(enumerate(all_paths), total=len(all_paths)):
        input_df = pd.read_csv(data_path)
        smiles_0 = input_df["ligand1_smiles"]
        smiles_1 = input_df["ligand2_smiles"]
        y_vals = input_df["relative_solubility"]


        for smiles_list, lambda_val in zip([smiles_0, smiles_1],
                                           ["lam_0", "lam_1"]):
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
            tensors_atoms = tf.ragged.constant(atom_features_list, dtype=tf.float32).numpy()
            tensors_bonds = tf.ragged.constant(bond_features_list, dtype=tf.float32).numpy()
            tensors_pairs = tf.ragged.constant(pair_indices_list, dtype=tf.int64).numpy()

            # save to numpy files.
            np.save(f"{savepath}/batch_{i}_{lambda_val}_atoms", tensors_atoms)
            np.save(f"{savepath}/batch_{i}_{lambda_val}_bonds", tensors_bonds)
            np.save(f"{savepath}/batch_{i}_{lambda_val}_pairs", tensors_pairs)
            np.save(f"{savepath}/batch_{i}_y_labels", y_vals)



#################################################################################
#################################################################################
########################### GENERATE TENSORFLOW DATASETS ########################

def prepare_batch(x_batch_0, x_batch_1, y_batch):
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


    return (x_batch_0, x_batch_1), y_batch

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

#################################################################################
#################################################################################
################################### MPNN BUILDING ###############################
@tf.keras.utils.register_keras_serializable()
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

def MPNNDataset(X_0, X_1, y, batch_size=56, shuffle=False):
    dataset = tf.data.Dataset.from_tensor_slices((X_0, X_1, (y)))

    if shuffle:
        dataset = dataset.shuffle(1024)
    return dataset.batch(batch_size).map(prepare_batch, -1)

@tf.keras.utils.register_keras_serializable()
class PartitionPadding(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = 56

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

@tf.keras.utils.register_keras_serializable()
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
    atom_dim,
    bond_dim,
    lambda_val,
    batch_size=56,
    message_units=64,
    message_steps=4,
    num_attention_heads=8,
    dense_units=512,
):
    atom_features = layers.Input((atom_dim), dtype="float32", name=f"atom_features_{lambda_val}")
    bond_features = layers.Input((bond_dim), dtype="float32", name=f"bond_features_{lambda_val}")
    pair_indices = layers.Input((2), dtype="int32", name=f"pair_indices_{lambda_val}")
    atom_partition_indices = layers.Input(
        (), dtype="int32", name=f"atom_partition_indices_{lambda_val}"
    )

    # removed __init__ arguments from the class layers to allow tensorflow to save the model.
    # custom:
    x = MessagePassing()(
        [atom_features, bond_features, pair_indices]
    )
    x = PartitionPadding()([x, atom_partition_indices])
    x = layers.Masking()(x)
    x = TransformerEncoder()(x)

    # out of the box:
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dense(250, activation="relu")(x)

    model = tf.keras.Model(
        inputs=[atom_features, bond_features, pair_indices, atom_partition_indices],
        outputs=[x],
    )
    return model


def cluster_batches(path_to_batches):
    """For a folder containing numpy files, create a nested list where each list contains
    paths to the components of the batch (i.e. all individual numpy files per tensor)"""
    clustered_batch_paths = []


    file_paths = glob(path_to_batches)
    batch_indices = [ path.split("_")[2] for path in file_paths ]
    for idx in batch_indices:
        # get all files for this batch.
        batch_files = glob(path_to_batches.replace("*", f"batch_{idx}_*"))
        clustered_batch_paths.append(batch_files)

    return clustered_batch_paths

def generate_batch(batch_cluster, return_dims=False):
    # load the correct files to reconstruct the tensors.

    def unfold(input_obj):
        if isinstance(input_obj, list):
            return input_obj[0]
        else:
            return input_obj

    y_labels = [ unfold(f) for f in batch_cluster if "y_labels.npy" in f ]
    y_labels_array = np.load(unfold(y_labels))


    lam_0_atoms = [ f for f in batch_cluster if "lam_0_atoms.npy" in f ]
    lam_0_bonds = [ f for f in batch_cluster if "lam_0_bonds.npy" in f ]
    lam_0_pairs = [ f for f in batch_cluster if "lam_0_pairs.npy" in f ]

    lam_1_atoms = [ f for f in batch_cluster if "lam_1_atoms.npy" in f ]
    lam_1_bonds = [ f for f in batch_cluster if "lam_1_bonds.npy" in f ]
    lam_1_pairs = [ f for f in batch_cluster if "lam_1_pairs.npy" in f ]

    x_lam_0 = (
                tf.ragged.constant(np.load(unfold(lam_0_atoms), allow_pickle=True), dtype=tf.float32),
                tf.ragged.constant(np.load(unfold(lam_0_bonds), allow_pickle=True), dtype=tf.float32),
                tf.ragged.constant(np.load(unfold(lam_0_pairs), allow_pickle=True), dtype=tf.int64)
                )
    x_lam_1 = (
                tf.ragged.constant(np.load(unfold(lam_1_atoms), allow_pickle=True), dtype=tf.float32),
                tf.ragged.constant(np.load(unfold(lam_1_bonds), allow_pickle=True), dtype=tf.float32),
                tf.ragged.constant(np.load(unfold(lam_1_pairs), allow_pickle=True), dtype=tf.int64)
                )

    if return_dims:  
        return x_lam_0[0][0][0].shape[0], x_lam_0[1][0][0].shape[0], x_lam_1[0][0][0].shape[0], x_lam_1[1][0][0].shape[0]
    else:
        # create the dataset for this batch.
        dataset = MPNNDataset(x_lam_0, x_lam_1, y_labels_array)
      
        return dataset
# @@@@@@@@@@@@@@@@@@@@@  


if __name__ == "__main__":
    print("Loading input data..")

    ############### featurise and split the input data ##################
    print("\nGenerating graphs from SMILES..")
    # Train set: 80 % of data. These have been split in the previous script before pairing, 
    # such that there is no ligand overlap between train/valid.

    print("Training set:")
    graphs_from_smiles_batch("process/data_batches/train/smiles*")

    print("Validation set:")
    graphs_from_smiles_batch("process/data_batches/valid/smiles*")

    print("Done.")
    # can omit test set generation because we only care about learning molecular structures.

   
    ############## build MPNN MODELS ##################
    # retrieve atom dimensions from the first batch to build the model with.
    print("Retrieving model input dimensions..")
    lam_0_atom_dims, lam_0_bond_dims, lam_1_atom_dims, lam_1_bond_dims, =  generate_batch(
                            cluster_batches("process/data_batches/train/tensors/*")[0], return_dims=True)

    # Build the lambda 0 and 1 legs (both are individual MPNNs).
    print("\nBuilding model..")
    mpnn_lam0 = MPNNModel(
        atom_dim=lam_0_atom_dims, bond_dim=lam_0_bond_dims,
        lambda_val=0
    )
    mpnn_lam1 = MPNNModel(
        atom_dim=lam_1_atom_dims, bond_dim=lam_1_bond_dims,
        lambda_val=1
    )

    # concatenate them (i.e. merge).
    combined = tf.keras.layers.Concatenate()([mpnn_lam0.output, mpnn_lam1.output])

    # make some more FCNN layers after the concatenation that will learn the delta property.
    head_model = layers.Dense(300, activation="relu")(combined)
    head_model = layers.Dense(150, activation="relu")(head_model)
    head_model = layers.Dense(1, activation="linear")(head_model)

    # build up the twin model.
    twin_model = keras.Model(inputs=[mpnn_lam0.input, mpnn_lam1.input], outputs=head_model)


    print("\nCompiling..")
    twin_model.compile(
        loss=keras.losses.LogCosh(),
        optimizer=keras.optimizers.Adam(learning_rate=5e-4),
        metrics=['mse'],
    )

    # save a rough sketch of the model. Will manually draw a schematic for the paper from this.
    keras.utils.plot_model(twin_model, to_file='process/twin_model.png', show_dtype=True, show_shapes=True)
    print(twin_model.summary())


    train_cluster_batches = cluster_batches("process/data_batches/train/tensors/*")
    valid_cluster_batches = cluster_batches("process/data_batches/valid/tensors/*")


    """
    $$$$$$$$$$$$$$$$$$ TRAINING $$$$$$$$$$$$$$$$$$$$$

    Need to do a manual loop over the data chunks (NOT BATCHES!) because the current 
    method of generating tf.datasets, then using a generator and fit_generator() isn't implemented yet. 
    Similarly, model.fit() can take in a generator, but the generated data can not yet be in the form 
    of a tf dataset, as is the case with ours."""
    if not len(train_cluster_batches) == len(valid_cluster_batches):
        print("WARNING: number of data chunks are not equal between train and valid:", len(train_cluster_batches),len(valid_cluster_batches), "- leaking some data.")
    
    for batch_train, batch_valid in zip(train_cluster_batches, valid_cluster_batches):
        print("Creating datasets for next chunk..")
        train_set = generate_batch(batch_train)
        valid_set = generate_batch(batch_valid)

        print("\nFitting chunk..")
        es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_mse', 
            patience=5, restore_best_weights=True, verbose=1)


        history = twin_model.fit(
            train_set,
            validation_data=valid_set,
            epochs=400,
            verbose=2,
            callbacks=[es_callback]
        )

    # save training information.
    print("Saving training information..")
    pd.DataFrame(history.history).to_csv("process/training_history.csv")

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('Loss - mean squared error of $\Delta$ESOL')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig('process/training_history_plot.png')

    # save model weights. The model can be restored in later scripts by reconstructing
    # the classes in this script and loading the weights.
    weights_path = "process/trained_model_weights/weights"
    print(f"Saving model weights to {weights_path}..")
    twin_model.save_weights(weights_path)

    print("\nDone.")
    
    



