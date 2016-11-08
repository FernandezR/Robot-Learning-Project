#!/usr/bin/env python3.4
'''
Baseline Test Evaluation Script

@author: Rolando Fernandez <rfernandez@utexas.edu>
'''
import csv
import os
import pickle
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
import sys

from baseline import extractPointCloudKeyPointsFromCSV
import numpy as np
import scripts.preprocess as preprocess


if __name__ == '__main__':

    fold_number = int( sys.argv[1] )

    n_clusters = 50

    dataset_directory = "/home/ghostman/Git/Robot-Learning-Project/robobarista_dataset/dataset/"
    pickle_directory = "/home/ghostman/Git/Robot-Learning-Project/Models/Baseline"
    test_data_directory = "/home/ghostman/Git/Robot-Learning-Project/Test-Data/baseline-fold_{}/".format( fold_number )
    folds_file = dataset_directory + "folds.json"

    folds_dictionary = pickle.load( open( pickle_directory + "folds_dictionary.p", "rb" ) )
    training_data = pickle.load( open( pickle_directory + "training_data_fold_{}.p".format( fold_number ), "rb" ) )
    kmeans = pickle.load( open( pickle_directory + "kmeans_fold_{}.p".format( fold_number ), "rb" ) )
    knneigh = pickle.load( open( pickle_directory + "knn_fold_{}.p".format( fold_number ), "rb" ) )

    print( "Loading Test Point Cloud Filenames for Fold {}".format( fold_number ) )

    test_data = preprocess.load_data_set( dataset_directory, folds_dictionary[fold_number]['test'] )
    point_cloud_files = test_data[1]

    #######################################################################
    #               Create Gold Standard Reference File                   #
    #######################################################################

    for sentence in test_data[2]:
        with open( test_data_directory + 'gold_reference_fold_{}'.format( fold_number ), 'a' ) as file:
            file.write( sentence + '\n' )

    for point_cloud in test_data[1]:

        #######################################################################
        #                  Extract Key Points from File                       #
        #######################################################################

        point_cloud_key_points = extractPointCloudKeyPointsFromCSV( point_cloud )

        #######################################################################
        #               Predict Features Vector for Point Cloud               #
        #######################################################################

        cluster_index_prediction = kmeans.predict( point_cloud_key_points )
        cluster_index_prediction = cluster_index_prediction.tolist()

        point_cloud_features_vector = []
        for cluster in range( n_clusters ):
            point_cloud_features_vector.append( cluster_index_prediction.count( cluster ) )

        #######################################################################
        #              Predict Nearest Neighbor for Point Cloud               #
        #######################################################################

        nearest_neighbor_index = knneigh.predict( [point_cloud_features_vector] )

        with open( test_data_directory + 'test_fold_{}'.format( fold_number ), 'a' ) as file:
            file.write( training_data[2][nearest_neighbor_index[0]] + '\n' )
