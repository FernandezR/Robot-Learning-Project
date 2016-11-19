#!/bin/bash

FOLDS=5

NNEIGHBORS=1

START=50
END=350
INCREMENT=50

BASELINE="/home/ghostman/Git/Robot-Learning-Project/scripts/baseline.py"

dataset_directory="/home/ghostman/Git/Robot-Learning-Project/robobarista_dataset/dataset/"
pickle_directory="/home/ghostman/Git/Robot-Learning-Project/Models/Baseline/"
meteor_directory="/home/ghostman/Git/Robot-Learning-Project/meteor-1.5"
test_data_directory="/home/ghostman/Git/Robot-Learning-Project/Test-Data"
validation_data_directory="/home/ghostman/Git/Robot-Learning-Project/Test-Data/validation"

PYPATH=/home/ghostman/Git/Robot-Learning-Project:/home/ghostman/Git/Robot-Learning-Project/scripts:/home/ghostman/Git/Robot-Learning-Project/scripts/dtw_mt:$PYTHONPATH

OUTPUT="$(PYTHONPATH=$PYPATH python3.4 $BASELINE calcCluster $FOLDS $validation_data_directory $START $END $INCREMENT $NNEIGHBORS)"

bash baseline-meteor.sh $OUTPUT $BASELINE $dataset_directory $pickle_directory $meteor_directory $test_data_directory $PYPATH $FOLDS 10
