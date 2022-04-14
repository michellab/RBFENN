# Machine-learned Predictions of Statistical Fluctuations in Relative Binding Free Energy Calculations and Data-driven Perturbation Network Generation

![alt text](https://github.com/michellab/data_driven_fep_reliabilities/blob/master/toc_figure.png)

Source code to reproduce [arxiv link](https://github.com/MobleyLab/Lomap). 

----------------------------------------------------------------

Given all publicly available RBFE benchmarking datasets, we have created a training domain ('RBFE-Space') that contains a representation of all perturbations present in these datasets by grafting them onto a common benzene scaffold. Then, after running all RBFE simulations for this novel set, we have used this training domain to train ML models to predict the quintuplicate standard error of the mean free energy (SEM). We have adjusted [LOMAP](https://github.com/MobleyLab/Lomap) to ingest these predicted SEM values to use instead of the native LOMAP-score, thereby producing a data-driven method of producing RBFE networks.

----------------------------------------------------------------

To reproduce, install the provided conda environment on a linux machine with at least one GPU (cuda). Main dependencies:
- [BioSimSpace](https://github.com/michellab/BioSimSpace)
- [TensorFlow-gpu >= 2.6.0](https://pypi.org/project/tensorflow-gpu/)

Main steps to reproduce:
1) Run `_01_SETUP_BENZENE_TRAINSET.ipynb` to get the list of transformations in RBFE-Space

2) Run `_02_SETUP_BSS_FOLDERS_TRAINSET.ipynb` to set up RBFE input files using BioSimSpace. BSS can set up simulations for SOMD, Amber and Gromacs or export the files needed for other RBFE implementations.

3) Run all RBFE simulations on a cluster

4) Collect SEM values from simulations in the format of `ANALYSIS/perturbation_networks/input/fepspace_sems_full_balanced.csv`

5) Sequentially run all python scripts/notebooks in `ANALYSIS/perturbation_networks/` to reproduce the RBFE network generation figures used in the paper.

6) Other figures can be reproduced using the notebooks found in `ANALYSIS/fepspace_vs_free_vs_bound/` and `ANALYSIS/lambda_spacing/`

----------------------------------------------------------------

Please note that some files (e.g. simulation outputs) were not included in this repository due to github memory restrictions. Feel free to post an issue with any questions regarding this work.

Authors:

- J. Scheen
- M. Mackey
- J. Michel
