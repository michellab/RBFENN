# A DATA-DRIVEN APPROACH TO RELATIVE FREE ENERGY PERTURBATION RELIABILITY PREDICTIONS FOR ALCHEMICAL FREE ENERGY CALCULATIONS IN DRUG DESIGN

![alt text](https://github.com/michellab/data_driven_fep_reliabilities/blob/master/ddfr_abstract_fig.png)

Source code to reproduce this work. Given all publicly available FEP benchmarking datasets, we have created a training domain ('FEP-Space') that contains a representation of all perturbations present in these datasets by grafting them onto a common benzene scaffold. Then, after running all FEP simulations for this novel set, we have used this training domain to train ML models to predict the quintuplicate standard error of the mean free energy (SEM). We have adjusted [LOMAP](https://github.com/MobleyLab/Lomap) to ingest these predicted SEM values to use instead of the native LOMAP-score, thereby producing a data-driven method of producing FEP networks.

To reproduce, install the provided conda environment on a linux machine with at least one GPU (cuda). Main dependencies:
- [BioSimSpace](https://github.com/michellab/BioSimSpace)
- [TensorFlow-gpu >= 2.6.0](https://pypi.org/project/tensorflow-gpu/)
