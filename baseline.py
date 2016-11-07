#!/usr/bin/env python
'''
Baseline

@author: Rolando Fernandez <rfernandez@utexas.edu>
'''
import csv
import os
import pickle

import numpy as np
import scripts.preprocess as preprocess
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier


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

    #######################################################################
    #                  Extract Key Points from Dataset                    #
    #######################################################################

    point_clouds = []
    point_clouds_key_points = []
    for data in dataset:
        key_points = extractPointCloudKeyPointsFromCSV( data )
        point_clouds.append( key_points )
        point_clouds_key_points.extend( key_points )

    #######################################################################
    #              Create KMeans Model for Dataset KeyPoints              #
    #######################################################################

    kmeans = KMeans( n_clusters = n_clusters, copy_x = True ).fit( np.array( point_clouds_key_points ) )

    pickle.dump( kmeans, open( pickle_directory + "kmeans_fold_{}.p".format( fold_number ), "wb" ) )

    return point_clouds, kmeans

def createKNNModel( kmeans_model, point_clouds, pickle_directory, fold_number ):

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

    knneigh = KNeighborsClassifier( n_neighbors = 1 )
    knneigh.fit( point_clouds_feature_vectors, targets )

    pickle.dump( knneigh, open( pickle_directory + "knn_fold_{}.p".format( fold_number ), "wb" ) )

if __name__ == '__main__':

    dataset_directory = "/home/ghostman/Git/Robot-Learning-Project/robobarista_dataset/dataset/"
    pickle_directory = "/home/ghostman/Git/Robot-Learning-Project/Models/"
    folds_file = dataset_directory + "folds.json"

    n_clusters = 50

    folds_dictionary = preprocess.get_folds_dictionary( folds_file )

    for key in folds_dictionary:

        point_cloud_files = preprocess.load_dataset( dataset_directory, folds_dictionary[key] )[1]

        point_clouds, kmeans_model = createKMeansModel( point_cloud_files, pickle_directory, n_clusters, key )

        createKNNModel( kmeans_model, point_clouds, pickle_directory, key )
