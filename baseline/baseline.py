#!/usr/bin/env python3.4
'''
Baseline

@author: Rolando Fernandez <rfernandez@utexas.edu>
'''
import csv
import os
import pickle
# from scipy.sparse import lil_matrix
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
import sys

import numpy as np
import scripts.preprocess as preprocess


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

    kmeans = KMeans( n_clusters = n_clusters, precompute_distances = False ).fit( np.array( point_clouds_key_points ) )

    if os.path.exists( pickle_directory + "kmeans_fold_{}.p".format( fold_number ) ):
        os.remove( pickle_directory + "kmeans_fold_{}.p".format( fold_number ) )

    pickle.dump( kmeans, open( pickle_directory + "kmeans_fold_{}.p".format( fold_number ), "wb" ) )

    return point_clouds, kmeans

def createKNNModel( kmeans_model, point_clouds, pickle_directory, fold_number ):
    '''
    Creates a K-Nearest Neighbor Model using the provided KMeans Model and saves it as a pickle file

    args:
        kmeans_model:     A KMeans Model
        point_clouds:     A list of Segmented Point Clouds
        pickle_directory: Path to save pickled KNN Model
        fold_number:      Fold number to use when saving pickled KNN Model

    returns:
        Nothing
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

    #######################################################################
    #    Create K-Nearest Neighbors Model for Dataset Feature Vectors     #
    #######################################################################

    n_samples = len( point_clouds_feature_vectors )

    targets = list( range( n_samples ) )

    knneigh = KNeighborsClassifier( n_neighbors = 1, weights = 'distance' )
    knneigh.fit( point_clouds_feature_vectors, targets )

    if os.path.exists( pickle_directory + "knn_fold_{}.p".format( fold_number ) ):
        os.remove( pickle_directory + "knn_fold_{}.p".format( fold_number ) )

    pickle.dump( knneigh, open( pickle_directory + "knn_fold_{}.p".format( fold_number ), "wb" ) )

if __name__ == '__main__':

    fold_number = int( sys.argv[1] )

    n_clusters = 50

    dataset_directory = "/home/ghostman/Git/Robot-Learning-Project/robobarista_dataset/dataset/"
    pickle_directory = "/home/ghostman/Git/Robot-Learning-Project/Models/Baseline/"
    folds_file = dataset_directory + "folds.json"

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
    createKNNModel( kmeans_model, point_clouds, pickle_directory, fold_number )
