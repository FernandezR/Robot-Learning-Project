#!/usr/bin/env python3.4
'''
Baseline with Trajectories

@author: Rolando Fernandez <rfernandez@utexas.edu>
'''
import csv
import os
import pickle
import shutil
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
import subprocess
import sys

import numpy as np
import preprocess


def extractPointCloudKeyPointsFromCSV( filename ):
    '''
    Extracts the features from a sgemented point cloud stored as a CSV file

    args:
        filename: A segmented point cloud stored as a CSV [x,y,z,r,g,b]

    returns:
        A list of fetaure lists for a segmented point cloud
    '''
    point_cloud = []
    with open( filename, newline = '' ) as csvfile:
        csvreader = csv.reader( csvfile, delimiter = ',', quoting = csv.QUOTE_NONNUMERIC )
        for row in csvreader:
            point_cloud.append( [row[0], row[1], row[2], row[3], row[4], row[5]] )
    return point_cloud

def testBaseline( fold_number, n_clusters, n_neighbors, dataset_directory, pickle_directory, test_data_directory, dtw_script_path, folds_file ):
    '''
    '''
    if not os.path.exists( test_data_directory ):
        os.makedirs( test_data_directory )

    if not os.path.exists( test_data_directory + 'predict/' ):
        os.mkdir( test_data_directory + 'predict/' )

    if not os.path.exists( test_data_directory + 'kneighbors/' ):
        os.mkdir( test_data_directory + 'kneighbors/' )

    predict_test_data_directory = test_data_directory + "predict/baseline-with-traj-fold_{}_for_{}_clusters_and_{}_neighbors/".format( fold_number, n_clusters, n_neighbors )
    kneighbors_test_data_directory = test_data_directory + "kneighbors/baseline-with-traj-fold_{}_for_{}_clusters_and_{}_neighbors/".format( fold_number, n_clusters, n_neighbors )

    if not  os.path.exists( predict_test_data_directory ):
        os.mkdir( predict_test_data_directory )

    if not  os.path.exists( kneighbors_test_data_directory ):
        os.mkdir( kneighbors_test_data_directory )

    folds_dictionary = pickle.load( open( pickle_directory + "folds_dictionary_ttv.p", "rb" ) )
    training_data = pickle.load( open( pickle_directory + "training_data_fold_{}_ttv.p".format( fold_number ), "rb" ) )
    kmeans = pickle.load( open( pickle_directory + "kmeans_fold_{}_for_{}_clusters.p".format( fold_number, n_clusters ), "rb" ) )
    knneigh = pickle.load( open( pickle_directory + "knn_fold_{}_for_{}_clusters_and_{}_neighbors.p".format( fold_number, n_clusters, n_neighbors ), "rb" ) )

    print( "Loading Test Point Cloud Filenames for Fold {}".format( fold_number ) )

    test_data = preprocess.load_data_set( dataset_directory, folds_dictionary[fold_number]['test'] )

    #######################################################################
    #               Create Gold Standard Reference File                   #
    #######################################################################

    print( "Creating Gold Reference for Fold {} with {} clusters and {} neighbors".format( fold_number, n_clusters, n_neighbors ) )

    if os.path.exists( predict_test_data_directory + 'gold_reference' ):
        os.remove( predict_test_data_directory + 'gold_reference' )

    if os.path.exists( kneighbors_test_data_directory + 'gold_reference' ):
        os.remove( kneighbors_test_data_directory + 'gold_reference' )

    for sentence in test_data[2]:

        with open( predict_test_data_directory + 'gold_reference', 'a' ) as f:
            f.write( sentence + '\n' )

    shutil.copy( predict_test_data_directory + 'gold_reference', kneighbors_test_data_directory )

    print( "Creating Test Reference for Fold {} with {} clusters and {} neighbors".format( fold_number, n_clusters, n_neighbors ) )

    if os.path.exists( predict_test_data_directory + 'test_reference' ):
        os.remove( predict_test_data_directory + 'test_reference' )

    if os.path.exists( kneighbors_test_data_directory + 'test_reference' ):
        os.remove( kneighbors_test_data_directory + 'test_reference' )

    for i in range( len( test_data[1] ) ):

        #######################################################################
        #                  Extract Key Points from File                       #
        #######################################################################

        point_cloud_key_points = extractPointCloudKeyPointsFromCSV( test_data[1][i] )

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

        nearest_neighbor_predict_index = knneigh.predict( [point_cloud_features_vector] )
        nearest_neighbor_index = knneigh.kneighbors( np.array( point_cloud_features_vector ).reshape( 1, -1 ), n_neighbors )[1][0]

        with open( predict_test_data_directory + 'test_reference', 'a' ) as f:
            f.write( training_data[2][nearest_neighbor_predict_index[0]] + '\n' )

        nearest_neighbor_indicies = knneigh.kneighbors( np.array( point_cloud_features_vector ).reshape( 1, -1 ), n_neighbors )[1][0]

        dtw_scores = []
        for nearest_neighbor_index in nearest_neighbor_indicies:
            cmd = ['python', dtw_script_path, test_data[3][i], training_data[3][nearest_neighbor_index] ]
            process = subprocess.Popen( cmd, stdout = subprocess.PIPE, stderr = subprocess.PIPE )
            process.communicate()
            dtw_value = pickle.load( open( "dtw_value.p", "rb" ), encoding = 'bytes' )
            os.remove( "dtw_value.p" )
            dtw_scores.append( ( nearest_neighbor_index, dtw_value ) )

        ranked_neighbors = sorted( dtw_scores, key = lambda x: x[1] )

        with open( kneighbors_test_data_directory + 'test_reference', 'a' ) as f:
            f.write( training_data[2][ranked_neighbors[0][0]] + '\n' )

    print( "Done running test for Fold {} with {} clusters and {} neighbors".format( fold_number, n_clusters, n_neighbors ) )

def printUsage():
    '''
    Helper method for displaying required usage parameters
    '''
    print( 'Run baseline testing: python3.4 baseline.py test [fold_number] [number_of_clusters] [number_of_neighbors] [dataset_directory] [pickle_directory] [test_data_directory] [dtw_script_path]' )
    print( '' )
    print( 'Use absolute path names' )

if __name__ == '__main__':
    if not len( sys.argv ) >= 2:
        printUsage()

    # dataset_directory = "/home/ghostman/Git/Robot-Learning-Project/robobarista_dataset/dataset/"
    # pickle_directory = "/home/ghostman/Git/Robot-Learning-Project/Models/Baseline/"

    elif sys.argv[1] == 'test':
        if not len( sys.argv ) == 9:
            printUsage()
        else:
            fold_number = int( sys.argv[2] )
            n_clusters = int( sys.argv[3] )
            n_neighbors = int( sys.argv[4] )
            dataset_directory = sys.argv[5]
            pickle_directory = sys.argv[6]
            test_data_directory = sys.argv[7]
            dtw_script_path = sys.argv[8]
            folds_file = dataset_directory + "folds.json"
            testBaseline( fold_number, n_clusters, n_neighbors, dataset_directory, pickle_directory, test_data_directory, dtw_script_path, folds_file )

    else:
        printUsage()
