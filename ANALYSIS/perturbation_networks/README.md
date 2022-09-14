## Main analyses presented in manuscript that involve perturbation networks.

The majority of this code is presented in jupyter notebooks - all files are named by chronological order. 

_00_fepspace_esol.py: creates pretraining dataset of 1M relative estimated solubility points.

_01_twin_gcn.py: creates and pretrains the RBFENN.

_02_transfer_learn_sem.py: transfer learns the RBFENN to predict SEMs.

_03_analyse_training_losses.ipynb: analyse training progress, primarily loss curves.

_04_external_test_all_series.py: (outdated) script to run SEM predictor on all RBFE benchmarking series.

_05_check_tyk2_sf_correlations.ipynb: for TYK2 test set, compare SEMs with offsets and LOMAP-Scores.

_06_tyk2_network_analyses.ipynb: TYK2 network planning analyses. Compare suggested networks with optimal network.

_07_compare_ntwx_all_series.ipynb: Generate network planning for all RBFE benchmarking series to record network sizes and overlaps.

_08_tnks2_network_analyses.ipynb: TNKS2 network analyses.

_09_compare_galectin_network.ipynb: Generation of a simple Galectin network using the SEM predictor and LOMAP-Score.
