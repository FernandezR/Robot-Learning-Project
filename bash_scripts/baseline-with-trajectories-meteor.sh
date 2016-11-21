#!/bin/bash

FOLDS=5

NCLUSTERS=50
NNEIGHBORS=10


BASELINE="/home/ghostman/Git/Robot-Learning-Project/scripts/baseline_with_trajectories.py"

dataset_directory="/home/ghostman/Git/Robot-Learning-Project/robobarista_dataset/dataset/"
pickle_directory="/home/ghostman/Git/Robot-Learning-Project/Models/Baseline/"
meteor_directory="/home/ghostman/Git/Robot-Learning-Project/meteor-1.5"
test_data_directory="/home/ghostman/Git/Robot-Learning-Project/Test-Data"
dtw_script_path="/home/ghostman/Git/Robot-Learning-Project/scripts/dtw_mt/compare_two_trajectories.py"

PYPATH=/home/ghostman/Git/Robot-Learning-Project:/home/ghostman/Git/Robot-Learning-Project/scripts:/home/ghostman/Git/Robot-Learning-Project/scripts/dtw_mt:$PYTHONPATH

echo $PYPATH

for i in $(seq 1 $FOLDS); do
    echo "Testing Fold ${i} Data with ${NCLUSTERS} clusters and ${NNEIGHBORS} neighbors"
    PYTHONPATH=$PYPATH python3.4 $BASELINE test $i $NCLUSTERS $NNEIGHBORS $dataset_directory $pickle_directory $test_data_directory/ $dtw_script_path
done

for i in $(seq 1 $FOLDS); do
    dir=$test_data_directory/kneighbors/baseline-with-traj-fold_${i}_for_${NCLUSTERS}_clusters_and_${NNEIGHBORS}_neighbors
    cd $dir
    echo "Evalualating Fold ${i} with Meteor for ${NCLUSTERS} clusters and ${NNEIGHBORS} neighbors"
    java -Xmx2G -jar $meteor_directory/meteor-*.jar $dir/test_reference $dir/gold_reference -norm -writeAlignments -f fold_$i > "fold_${i}_score"
    python $meteor_directory/xray/xray.py -p fold_$i fold_$i-align.out
    echo "Completed Evalualation of Fold ${i} with Meteor for ${NCLUSTERS} clusters and ${NNEIGHBORS} neighbors"
done

for i in $(seq 1 $FOLDS); do
    dir=$test_data_directory/predict/baseline-with-traj-fold_${i}_for_${NCLUSTERS}_clusters_and_${NNEIGHBORS}_neighbors
    cd $dir
    echo "Evalualating Fold ${i} with Meteor for ${NCLUSTERS} clusters and ${NNEIGHBORS} neighbors"
    java -Xmx2G -jar $meteor_directory/meteor-*.jar $dir/test_reference $dir/gold_reference -norm -writeAlignments -f fold_$i > "fold_${i}_score"
    python $meteor_directory/xray/xray.py -p fold_$i fold_$i-align.out
    echo "Completed Evalualation of Fold ${i} with Meteor for ${NCLUSTERS} clusters and ${NNEIGHBORS} neighbors"
done
