#!/bin/bash

FOLDS=5

NCLUSTERS=50
NNEIGHBORS=1

START=$NCLUSTERS
INCREMENT=50

BASELINE="/home/ghostman/Git/Robot-Learning-Project/scripts/baseline.py"
BASHSCRIPTS="/home/ghostman/Git/Robot-Learning-Project/bash_scripts"

dataset_directory="/home/ghostman/Git/Robot-Learning-Project/robobarista_dataset/dataset/"
pickle_directory="/home/ghostman/Git/Robot-Learning-Project/Models/Baseline/"
meteor_directory="/home/ghostman/Git/Robot-Learning-Project/meteor-1.5"
test_data_directory="/home/ghostman/Git/Robot-Learning-Project/Test-Data"
validation_data_directory="/home/ghostman/Git/Robot-Learning-Project/Test-Data/validation"

PYPATH=/home/ghostman/Git/Robot-Learning-Project:/home/ghostman/Git/Robot-Learning-Project/scripts:/home/ghostman/Git/Robot-Learning-Project/scripts/dtw_mt:$PYTHONPATH

echo $PYPATH

for counter in {1..7}; do
    for i in $(seq 1 $FOLDS); do
        echo "Training Fold ${i} Models with ${NCLUSTERS} clusters and ${NNEIGHBORS} neighbors"
        PYTHONPATH=$PYPATH python3.4 $BASELINE train $i $NCLUSTERS $NNEIGHBORS $dataset_directory $pickle_directory
    done

    for i in $(seq 1 $FOLDS); do
        echo "Validating Fold ${i} Data with ${NCLUSTERS} clusters and ${NNEIGHBORS} neighbors"
        PYTHONPATH=$PYPATH python3.4 $BASELINE validation $i $NCLUSTERS $NNEIGHBORS $dataset_directory $pickle_directory $validation_data_directory/
    done

    for i in $(seq 1 $FOLDS); do
        dir=$validation_data_directory/kneighbors/baseline-fold_${i}_for_${NCLUSTERS}_clusters_and_${NNEIGHBORS}_neighbors
        cd $dir
        echo "Evalualating Fold ${i} with Meteor for ${NCLUSTERS} clusters and ${NNEIGHBORS} neighbors"
        java -Xmx2G -jar $meteor_directory/meteor-*.jar $dir/validation_reference $dir/gold_reference -norm -writeAlignments -f fold_$i > "fold_${i}_score"
        python $meteor_directory/xray/xray.py -p fold_$i fold_$i-align.out
        echo "Completed Evalualation of Fold ${i} with Meteor for ${NCLUSTERS} clusters and ${NNEIGHBORS} neighbors"
    done

    for i in $(seq 1 $FOLDS); do
        dir=$validation_data_directory/predict/baseline-fold_${i}_for_${NCLUSTERS}_clusters_and_${NNEIGHBORS}_neighbors
        cd $dir
        echo "Evalualating Fold ${i} with Meteor for ${NCLUSTERS} clusters and ${NNEIGHBORS} neighbors"
        java -Xmx2G -jar $meteor_directory/meteor-*.jar $dir/validation_reference $dir/gold_reference -norm -writeAlignments -f fold_$i > "fold_${i}_score"
        python $meteor_directory/xray/xray.py -p fold_$i fold_$i-align.out
        echo "Completed Evalualation of Fold ${i} with Meteor for ${NCLUSTERS} clusters and ${NNEIGHBORS} neighbors"
    done
    NCLUSTERS=$(($NCLUSTERS+$INCREMENT))
done

OUTPUT="$(PYTHONPATH=$PYPATH python3.4 $BASELINE calcCluster $FOLDS $validation_data_directory $START $NCLUSTERS $INCREMENT $NNEIGHBORS predict)"
echo "------------------------------"
echo "Best Cluster Size with Neighbors predict"
echo $OUTPUT

OUTPUT="$(PYTHONPATH=$PYPATH python3.4 $BASELINE calcCluster $FOLDS $validation_data_directory $START $NCLUSTERS $INCREMENT $NNEIGHBORS)"
echo "------------------------------"
echo "Best Cluster Size with Neighbors kneighbors"
echo $OUTPUT

echo "------------------------------"
echo "Echo using Kneighbors best cluster size to Train Final Model"
echo "------------------------------"

cd $BASHSCRIPTS
bash baseline-meteor.sh $OUTPUT $BASELINE $dataset_directory $pickle_directory $meteor_directory $test_data_directory $PYPATH $FOLDS 10

# NCLUSTERS=(50)
# for item in ${NCLUSTERS[*]}; do
#     echo $item
# done

# NCLUSTERS=50
# for counter in {1..7}; do
#    echo $counter. "\$n_clusters = " $NCLUSTERS
#    NCLUSTERS=$(($NCLUSTERS+10))
# done
