#!/bin/bash

NCLUSTERS=$1
NNEIGHBORS=10

BASELINE=$2

dataset_directory=$3
pickle_directory=$4
meteor_directory=$5
test_data_directory=$6
PYPATH=$7

FOLDS=$8
NNEIGHBORS=$9

for i in $(seq 1 $FOLDS); do
    echo "Training Fold ${i} Models with ${NCLUSTERS} clusters and ${NNEIGHBORS} neighbors"
    PYTHONPATH=$PYPATH python3.4 $BASELINE train $i $NCLUSTERS $NNEIGHBORS $dataset_directory $pickle_directory
done

for i in $(seq 1 $FOLDS); do
    echo "Testing Fold ${i} Data with ${NCLUSTERS} clusters and ${NNEIGHBORS} neighbors"
    PYTHONPATH=$PYPATH python3.4 $BASELINE test $i $NCLUSTERS $NNEIGHBORS $dataset_directory $pickle_directory $test_data_directory/
done

for i in $(seq 1 $FOLDS); do
    dir=$test_data_directory/baseline-fold_${i}_for_${NCLUSTERS}_clusters_and_${NNEIGHBORS}_neighbors
    cd $dir
    echo "Evalualating Fold ${i} with Meteor for ${NCLUSTERS} clusters and ${NNEIGHBORS} neighbors"
    java -Xmx2G -jar $meteor_directory/meteor-*.jar $dir/test_reference $dir/gold_reference -norm -writeAlignments -f fold_$i > "fold_${i}_score"
    python $meteor_directory/xray/xray.py -p fold_$i fold_$i-align.out
    echo "Completed Evalualation of Fold ${i} with Meteor for ${NCLUSTERS} clusters and ${NNEIGHBORS} neighbors"
done
