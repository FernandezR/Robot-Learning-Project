#!/bin/bash

FOLDS=5

meteor_directory="/home/ghostman/Git/Robot-Learning-Project/meteor-1.5"
test_data_directory="/home/ghostman/Git/Robot-Learning-Project/Test-Data"

for i in $(seq 1 $FOLDS); do
    cd $test_data_directory/baseline-fold_$i/
    echo "Evalualating Fold ${i} with Meteor"
    java -Xmx2G -jar $meteor_directory/meteor-*.jar $test_data_directory/baseline-fold_$i/test_fold_$i $test_data_directory/baseline-fold_$i/gold_reference_fold_$i -norm -writeAlignments -f fold_$i > "fold_${i}_score"
    python $meteor_directory/xray/xray.py -p fold_$i fold_$i-align.out
    echo "Completed Evalualation of Fold ${i} with Meteor"; done
