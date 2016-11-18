#!/bin/bash

FOLDS=5

NCLUSTERS=50
NNEIGHBORS=1

BASELINE="/home/ghostman/Git/Robot-Learning-Project/baseline/baseline.py"

dataset_directory="/home/ghostman/Git/Robot-Learning-Project/robobarista_dataset/dataset/"
pickle_directory="/home/ghostman/Git/Robot-Learning-Project/Models/Baseline/"
meteor_directory="/home/ghostman/Git/Robot-Learning-Project/meteor-1.5"
test_data_directory="/home/ghostman/Git/Robot-Learning-Project/Test-Data"

for i in $(seq 1 $FOLDS); do
    echo "Training Fold ${i} Models"
    PYTHONPATH=/home/ghostman/Git/Robot-Learning-Project python3.4 $BASELINE train $i $NCLUSTERS $NNEIGHBORS $dataset_directory $pickle_directory
done

for i in $(seq 1 $FOLDS); do
    echo "Testing Fold ${i} Data"
    PYTHONPATH=/home/ghostman/Git/Robot-Learning-Project python3.4 $BASELINE test $i $NCLUSTERS $NNEIGHBORS $dataset_directory $pickle_directory $test_data_directory/
done

for i in $(seq 1 $FOLDS); do
    cd $test_data_directory/baseline-fold_$i/
    echo "Evalualating Fold ${i} with Meteor"
    java -Xmx2G -jar $meteor_directory/meteor-*.jar $test_data_directory/baseline-fold_$i_for_$NCLUSTERS_clusters_and_$NNEIGHBORS_neighbors/test_reference $test_data_directory/baseline-fold_$i_for_$NCLUSTERS_clusters_and_$NNEIGHBORS_neighbors/gold_reference -norm -writeAlignments -f fold_$i > "fold_${i}_score"
    python $meteor_directory/xray/xray.py -p fold_$i fold_$i-align.out
    echo "Completed Evalualation of Fold ${i} with Meteor"
done

# NCLUSTERS=(50)
# for item in ${NCLUSTERS[*]}; do
#     echo $item
# done

# NCLUSTERS=50
# for counter in {1..10}; do
#    echo $counter. "\$n_clusters = " $SIZE
#    SIZE=$(($SIZE+10))
# done
