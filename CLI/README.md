Workspace for the command-line interface (CLI) of the SF predictor presented in this work. See the main folder of this repository for a conda environment to use with this CLI. 

----------------------

Check `create_example` for inspiration on how to create parse files given a collection of ligand files (SDF) in `input_files_example`. The parse files tell the CLI which transformations to predict the SF on. `rbfenn_weights` contains trained weights of the siamese GNN. 

The reason the CLI was designed using this input parsing file is that it takes \~ 15 seconds to load the siamese GNN weights and create the model ensemble which would introduce too much overhead if it had to be loaded per ligand pair.

----------------------
Steps:

[WIP] Add explanations for following:
- CLI -h
- running example
- checking output example (notebook)
- checking distribution plot (notebook)

Note: SF predicted is SEM, so the larger the predicted value the higher the predicted SF.

