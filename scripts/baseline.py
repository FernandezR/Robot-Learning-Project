#!/usr/bin/env python3.4
'''
Baseline

@author: Rolando Fernandez <rfernandez@utexas.edu>
'''
import csv
from functools import reduce
from operator import add
import os
import pickle
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
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

def createKMeansModel( dataset, pickle_directory, n_clusters, fold_number ):
    '''
    Creates a KMeans Model using the provided Dataset and saves it as a pickle file

    args:
        dataset:          A list of segmented point clouds filenames [stored as a CSV [x,y,z,r,g,b]]
        pickle_directory: Path to save pickled KMeans Model
        n_cluster:        Number of clusters to use for KMeans Model
        fold_number:      Fold number to use when saving pickled Kmeans Model

    returns:
        point_clouds: A list of Segmented Point Clouds
        kmeans:       A KMeans Model
    '''
    #######################################################################
    #                  Extract Key Points from Dataset                    #
    #######################################################################

    point_clouds = []
    point_clouds_key_points = []
    point_cloud_names = []
    for data in dataset:
        key_points = extractPointCloudKeyPointsFromCSV( data )
        point_clouds.append( key_points )
        if data not in point_cloud_names:
            point_clouds_key_points.extend( key_points )
            point_cloud_names.append( data )

    #######################################################################
    #              Create KMeans Model for Dataset KeyPoints              #
    #######################################################################

    kmeans = KMeans( n_clusters = n_clusters, precompute_distances = False, n_jobs = -1 ).fit( np.array( point_clouds_key_points ) )

    if os.path.exists( pickle_directory + "kmeans_fold_{}_for_{}_clusters.p".format( fold_number, n_clusters ) ):
        os.remove( pickle_directory + "kmeans_fold_{}_for_{}_clusters.p".format( fold_number, n_clusters ) )

    pickle.dump( kmeans, open( pickle_directory + "kmeans_fold_{}_for_{}_clusters.p".format( fold_number, n_clusters ), "wb" ) )

    return point_clouds, kmeans

def createPCFeatureVectors( kmeans_model, point_clouds, pickle_directory, fold_number, n_clusters ):
    '''
    Creates a Feature Vector list for a Point Cloud Key Point List using the provided KMeans Model and saves it as a pickle file

    args:
        kmeans_model:     A KMeans Model
        point_clouds:     A list of Segmented Point Clouds
        pickle_directory: Path to save pickled Feature Vector
        fold_number:      Fold number to use when saving pickled KNN Model
        n_cluster:        Number of clusters to used for KMeans Model

    returns:
        Feature Vector lists of Point Cloud Feature Vectors
    '''
    #######################################################################
    #                Predict Feature Vectors for Dataset                  #
    #######################################################################

    point_clouds_feature_vectors = []
    for point_cloud in point_clouds:
        cluster_index_prediction = kmeans_model.predict( point_cloud )
        cluster_index_prediction = cluster_index_prediction.tolist()

        feature_vector = []
        for cluster in range( n_clusters ):
            feature_vector.append( cluster_index_prediction.count( cluster ) )

        point_clouds_feature_vectors.append( feature_vector )

    if os.path.exists( pickle_directory + "pc_feature_vectors_fold_{}_for_{}_clusters.p".format( fold_number, n_clusters ) ):
        os.remove( pickle_directory + "pc_feature_vectors_fold_{}_for_{}_clusters.p".format( fold_number, n_clusters ) )

    pickle.dump( point_clouds_feature_vectors, open( pickle_directory + "pc_feature_vectors_fold_{}_for_{}_clusters.p".format( fold_number, n_clusters ), "wb" ) )

    return point_clouds_feature_vectors

def createKNNModel( kmeans_model, point_clouds, pickle_directory, fold_number, n_clusters, n_neighbors ):
    '''
    Creates a K-Nearest Neighbor Model using the provided KMeans Model and saves it as a pickle file

    args:
        kmeans_model:     A KMeans Model
        point_clouds:     A list of Segmented Point Clouds
        pickle_directory: Path to save pickled KNN Model
        fold_number:      Fold number to use when saving pickled KNN Model
        n_cluster:        Number of clusters to used for KMeans Model
        n_neighbors:        Number of neighbors to used for KNN Model

    returns:
        Nothing
    '''
    #######################################################################
    #                Predict Feature Vectors for Dataset                  #
    #######################################################################

    point_clouds_feature_vectors = createPCFeatureVectors( kmeans_model, point_clouds, pickle_directory, fold_number, n_clusters )

    #######################################################################
    #    Create K-Nearest Neighbors Model for Dataset Feature Vectors     #
    #######################################################################

    n_samples = len( point_clouds_feature_vectors )

    targets = list( range( n_samples ) )

    knneigh = KNeighborsClassifier( n_neighbors = n_neighbors, weights = 'distance' )
    knneigh.fit( point_clouds_feature_vectors, targets )

    if os.path.exists( pickle_directory + "knn_fold_{}_for_{}_clusters_and_{}_neighbors.p".format( fold_number, n_clusters, n_neighbors ) ):
        os.remove( pickle_directory + "knn_fold_{}_for_{}_clusters_and_{}_neighbors.p".format( fold_number, n_clusters, n_neighbors ) )

    pickle.dump( knneigh, open( pickle_directory + "knn_fold_{}_for_{}_clusters_and_{}_neighbors.p".format( fold_number, n_clusters, n_neighbors ), "wb" ) )


def trainBaseline( pickle_directory, folds_file, fold_number, dataset_directory, n_clusters, n_neighbors ):
    '''
    '''
    if not os.path.exists( pickle_directory + "folds_dictionary.p" ):
        print( "Creating Folds Dictionary" )
        folds_dictionary = preprocess.get_folds_dictionary( folds_file )
        pickle.dump( folds_dictionary, open( pickle_directory + "folds_dictionary.p", "wb" ) )
    else:
        folds_dictionary = pickle.load( open( pickle_directory + "folds_dictionary.p", "rb" ) )

    print( "Loading Training Point Cloud Filenames for Fold {}".format( fold_number ) )
    training_data = preprocess.load_data_set( dataset_directory, folds_dictionary[fold_number]['train'] )

    if os.path.exists( pickle_directory + "training_data_fold_{}.p".format( fold_number ) ):
        os.remove( pickle_directory + "training_data_fold_{}.p".format( fold_number ) )

    pickle.dump( training_data, open( pickle_directory + "training_data_fold_{}.p".format( fold_number ), "wb" ) )
    point_cloud_files = training_data[1]

    print( "Creating KMeans Model for Fold {}".format( fold_number ) )
    point_clouds, kmeans_model = createKMeansModel( point_cloud_files, pickle_directory, n_clusters, fold_number )

    print( "Creating KNN Model for Fold {}".format( fold_number ) )
    createKNNModel( kmeans_model, point_clouds, pickle_directory, fold_number, n_clusters, n_neighbors )

    print( "Done training models for Fold {}".format( fold_number ) )

def testBaseline( fold_number, n_clusters, n_neighbors, dataset_directory, pickle_directory, test_data_directory, folds_file, validation = False ):
    '''
    '''
    test_data_directory = test_data_directory + "baseline-fold_{}_for_{}_clusters_and_{}_neighbors/".format( fold_number, n_clusters, n_neighbors )

    if not  os.path.exists( test_data_directory ):
        os.mkdir( test_data_directory )

    folds_dictionary = pickle.load( open( pickle_directory + "folds_dictionary.p", "rb" ) )
    training_data = pickle.load( open( pickle_directory + "training_data_fold_{}.p".format( fold_number ), "rb" ) )
    kmeans = pickle.load( open( pickle_directory + "kmeans_fold_{}_for_{}_clusters.p".format( fold_number, n_clusters ), "rb" ) )
    knneigh = pickle.load( open( pickle_directory + "knn_fold_{}_for_{}_clusters_and_{}_neighbors.p".format( fold_number, n_clusters, n_neighbors ), "rb" ) )

    print( "Loading Test Point Cloud Filenames for Fold {}".format( fold_number ) )


    if( validation ):
        test_data = preprocess.load_data_set( dataset_directory, folds_dictionary[fold_number]['validate'] )
        test_file_name = "validation_reference"
    else:
        test_data = preprocess.load_data_set( dataset_directory, folds_dictionary[fold_number]['test'] )
        test_file_name = "test_reference"

    point_cloud_files = test_data[1]

    #######################################################################
    #               Create Gold Standard Reference File                   #
    #######################################################################

    print( "Creating Gold Reference for Fold {} with {} clusters and {} neighbors".format( fold_number, n_clusters, n_neighbors ) )

    if os.path.exists( test_data_directory + 'gold_reference' ):
        os.remove( test_data_directory + 'gold_reference' )

    for sentence in test_data[2]:

        with open( test_data_directory + 'gold_reference', 'a' ) as file:
            file.write( sentence + '\n' )

    print( "Creating Test Reference for Fold {} with {} clusters and {} neighbors".format( fold_number, n_clusters, n_neighbors ) )

    if os.path.exists( test_data_directory + test_file_name ):
        os.remove( test_data_directory + test_file_name )

    for point_cloud in point_cloud_files:

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

        with open( test_data_directory + test_file_name, 'a' ) as file:
            file.write( training_data[2][nearest_neighbor_index[0]] + '\n' )

    print( "Done running test for Fold {} with {} clusters and {} neighbors".format( fold_number, n_clusters, n_neighbors ) )


def calculateBestNumClusters( n_folds, validation_data_directory, cluster_range, n_neighbors ):
    '''
    '''
    average_scores = []
    for k_clusters in cluster_range:
        scores = []
        for fold in range( 1, n_folds + 1 ):
            with open( validation_data_directory + "/baseline-fold_{}_for_{}_clusters_and_{}_neighbors/fold_{}_score".format( fold, k_clusters, n_neighbors, fold ), 'r' ) as f:
                last = f.readlines()[-1]
            score = float( last.split( ':' )[1].strip() )
            scores.append( score )
        average_scores.append( ( k_clusters, ( reduce( add, scores ) / len( scores ) ) ) )

    ranked_scores = sorted( average_scores, reverse = True, key = lambda x: x[1] )

    return ranked_scores[0][0]


def printUsage():
    '''
    Helper method for displaying required usage parameters
    '''
    print ( 'Run baseline training: python3.4 baseline.py train [fold_number] [number_of_clusters] [number_of_neighbors] [dataset_directory] [pickle_directory]' )
    print ( '' )
    print ( 'Run baseline testing: python3.4 baseline.py test [fold_number] [number_of_clusters] [number_of_neighbors] [dataset_directory] [pickle_directory] [test_data_directory]' )
    print ( '' )
    print ( 'Run baseline testing: python3.4 baseline.py validation [fold_number] [number_of_clusters] [number_of_neighbors] [dataset_directory] [pickle_directory] [test_data_directory]' )
    print ( '' )
    print ( 'Run baseline testing: python3.4 baseline.py calcCluster [number_of_folds] [validation_data_directory] [starting_cluster_size] [ending_cluster_size_[plus_1]] [increment_value] [number_of_neighbors]' )
    print ( '' )
    print ( 'Use absolute path names' )

if __name__ == '__main__':
    if not len( sys.argv ) >= 2:
        printUsage()

    # dataset_directory = "/home/ghostman/Git/Robot-Learning-Project/robobarista_dataset/dataset/"
    # pickle_directory = "/home/ghostman/Git/Robot-Learning-Project/Models/Baseline/"

    elif sys.argv[1] == 'train':
        if not len( sys.argv ) == 7:
            printUsage()
        else:
            fold_number = int( sys.argv[2] )
            n_clusters = int( sys.argv[3] )
            n_neighbors = int( sys.argv[4] )
            dataset_directory = sys.argv[5]
            pickle_directory = sys.argv[6]
            folds_file = dataset_directory + "folds.json"
            trainBaseline( pickle_directory, folds_file, fold_number, dataset_directory, n_clusters, n_neighbors )

    elif sys.argv[1] == 'test':
        if not len( sys.argv ) == 8:
            printUsage()
        else:
            fold_number = int( sys.argv[2] )
            n_clusters = int( sys.argv[3] )
            n_neighbors = int( sys.argv[4] )
            dataset_directory = sys.argv[5]
            pickle_directory = sys.argv[6]
            test_data_directory = sys.argv[7]
            folds_file = dataset_directory + "folds.json"
            testBaseline( fold_number, n_clusters, n_neighbors, dataset_directory, pickle_directory, test_data_directory, folds_file )

    elif sys.argv[1] == 'validation':
        if not len( sys.argv ) == 8:
            printUsage()
        else:
            fold_number = int( sys.argv[2] )
            n_clusters = int( sys.argv[3] )
            n_neighbors = int( sys.argv[4] )
            dataset_directory = sys.argv[5]
            pickle_directory = sys.argv[6]
            test_data_directory = sys.argv[7]
            folds_file = dataset_directory + "folds.json"
            testBaseline( fold_number, n_clusters, n_neighbors, dataset_directory, pickle_directory, test_data_directory, folds_file, True )

    elif sys.argv[1] == 'calcCluster':
        if not len( sys.argv ) == 8:
            printUsage()
        else:
            n_folds = int( sys.argv[2] )
            validation_data_directory = int( sys.argv[3] )
            start_k = int( sys.argv[4] )
            stop_k = sys.argv[5]
            incr_k = sys.argv[6]
            n_neighbors = sys.argv[7]
            print( calculateBestNumClusters( n_folds, validation_data_directory, range( start_k, stop_k, incr_k ), n_neighbors ) )

    else:
        printUsage()
