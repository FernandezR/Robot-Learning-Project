#!/bin/bash

FOLDS=5

NCLUSTERS=$1
NNEIGHBORS=1

BASELINE="/home/ghostman/Git/Robot-Learning-Project/scripts/baseline.py"

dataset_directory="/home/ghostman/Git/Robot-Learning-Project/robobarista_dataset/dataset/"
pickle_directory="/home/ghostman/Git/Robot-Learning-Project/Models/Baseline/"
meteor_directory="/home/ghostman/Git/Robot-Learning-Project/meteor-1.5"
test_data_directory="/home/ghostman/Git/Robot-Learning-Project/Test-Data"
validation_data_directory="/home/ghostman/Git/Robot-Learning-Project/Test-Data/validation"

PYPATH=/home/ghostman/Git/Robot-Learning-Project:/home/ghostman/Git/Robot-Learning-Project/scripts:/home/ghostman/Git/Robot-Learning-Project/scripts/dtw_mt:$PYTHONPATH

echo $PYPATH


for i in $(seq 1 $FOLDS); do
   echo "Training Fold ${i} Models with ${NCLUSTERS} clusters and ${NNEIGHBORS} neighbors"
   PYTHONPATH=$PYPATH python3.4 $BASELINE train $i $NCLUSTERS $NNEIGHBORS $dataset_directory $pickle_directory
done

for i in $(seq 1 $FOLDS); do
   echo "Validating Fold ${i} Data with ${NCLUSTERS} clusters and ${NNEIGHBORS} neighbors"
   PYTHONPATH=$PYPATH python3.4 $BASELINE validation $i $NCLUSTERS $NNEIGHBORS $dataset_directory $pickle_directory $validation_data_directory/
done

for i in $(seq 1 $FOLDS); do
    dir=$validation_data_directory/baseline-fold_${i}_for_${NCLUSTERS}_clusters_and_${NNEIGHBORS}_neighbors
    cd $dir
    echo "Evalualating Fold ${i} with Meteor for ${NCLUSTERS} clusters and ${NNEIGHBORS} neighbors"
    java -Xmx2G -jar $meteor_directory/meteor-*.jar $dir/validation_reference $dir/gold_reference -norm -writeAlignments -f fold_$i > "fold_${i}_score"
    python $meteor_directory/xray/xray.py -p fold_$i fold_$i-align.out
    echo "Completed Evalualation of Fold ${i} with Meteor for ${NCLUSTERS} clusters and ${NNEIGHBORS} neighbors"
done
