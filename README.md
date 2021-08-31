# FFgenius
This repository will contain machine learning models meant to help rank players to help with your fantasy football draft

## Important Files
`models.py` runs a basic comparison between random forest, multiple linear regression, and neural nets for each position tested. Simply run the program and it will generate matrices for each position showing important evaluation scores for each model.

`nn_tuning.py` runs a randomized or grid search CV on a position of players (based on what is commented and uncommented out) in order to find the ideal hyperparameters for each model by position.

`random_forest_tuning.py` essentially does the same thing as `nn_tuning.py`, but with random forest regressors.

`relationships.py` shows a sorted heatmap for each position showing the relationship between all of the independent variables and the dependent variable (average points per game)

`parse.py` is run to create the filtered CSVs used for the model tuning. This program takes the general files and creates optimized CSVs that can be used more easily to streamline the ML model creation process.

The `stats` directory contains all of the files used by the machine learning models.