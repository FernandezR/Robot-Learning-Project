#!/usr/bin/python

import glob
import json
import os
import sys
import yaml
import nltk
import numpy as np

"""
Gets dictionary of folds containing
the test and train data files for
each fold
"""
def get_folds_dictionary( folds_file ):
    with open( folds_file ) as json_data:
        dictionary = json.load( json_data )

    folds_dictionary = {1:{'test':[], 'train':[]}, 2:{'test':[], 'train':[]}, 3:{'test':[], 'train':[]}, 4:{'test':[], 'train':[]}, 5:{'test':[], 'train':[]}}

    for key in dictionary:
        folds_dictionary[dictionary[key]]['test'].append( key )

    for fold_number in range( 1, 6 ):
        for key in ( key for key in folds_dictionary if key != fold_number ):
            folds_dictionary[fold_number]['test'].sort()
            folds_dictionary[fold_number]['train'].extend( folds_dictionary[key]['test'] )
        folds_dictionary[fold_number]['train'].sort()

    return folds_dictionary

"""
Gets part names from the
info.yaml file in the given
object directory.
"""
def get_part_names( obj_dir ):
        part_list = []

        yaml_file = open( obj_dir + '/info.yaml' )

        return [part for part in yaml.load( yaml_file )['parts']]


"""
Reads a trajectory file and returns it as
an array of waypoints.
"""
def get_trajectory_from_file( traj_file_name ):
    trajectory = []
    traj_file = open( traj_file_name, 'r' )

    # Assigns an integer to grip type.
    types = {'hold': 1.0, 'open': 2.0, 'close': 3.0}

    for line in traj_file:
        if line.startswith( '"' ):
            traj_type = types[line.strip().strip( '"' )]

        # Waypoint line.
        else:
            # Reads in parameter values.
            params = [float( param.strip() ) for param in line.strip().strip( '[' ).strip( ']' ).split( ',' )]

            # Formats waypoint and appends it to trajectory.
            waypoint = [traj_type] + params

            trajectory.append( waypoint )

    return trajectory


"""
Loads training set specified by a list
of object directories into arrays
that may be used by our models.
"""
def load_data_set( root_dir, obj_dir_list ):
    trajectories = []
    point_clouds = []
    nl_descriptions = []

    # Compiles all triplets (t, p, l) into arrays from a given object list.
    for obj_dir in obj_dir_list:
        path = root_dir + obj_dir

        # Gets parts and their names from info file.
        parts = get_part_names( path )
        parts.sort()

        # Get all natural language descriptions.
        descriptions = {}

        for manual_file in sorted( glob.glob( path + '/manual*' ) ):
            for part, desc in yaml.load( open( manual_file, 'r' ) )['steps']:

                # Gets description for part.
                descriptions[part] = desc

        # Create data points.
        for part in parts:
            part_num = part.split( '_' )[1]

            # Gets natural language description for part.
            if part in descriptions:
                description = descriptions[part]
            else:
                # Don't add point if there is no natural language descrption for this part.
                continue

            # Gets point cloud name for part.
            point_cloud_name = 'pointcloud_' + obj_dir + '_' + part
            point_cloud_path = path + '/' + point_cloud_name

            # Make sure it exists in case names aren't completely standardized.
            if not os.path.isfile( point_cloud_path ):
                print ( 'Point cloud path: "' + point_cloud_path + '" does not specify a file, verify it exists and that code is correct!' )
                sys.exit()

            # Gets all trajectories for a part.
            traj_path = path + '/user_input/'

            for traj_file in sorted( glob.glob( traj_path + '*_' + str( part_num ) ) ):
                traj = get_trajectory_from_file( traj_file )

                # Add triplet to lists at same index.
                trajectories.append( traj )
                point_clouds.append( point_cloud_path )
                nl_descriptions.append( description )

    # Make sure that everything is of the same dimensionality.
    assert ( len( trajectories ) == len( point_clouds ) ) and ( len( trajectories ) == len( nl_descriptions ) )

    return [trajectories, point_clouds, nl_descriptions]

"""
Takes a list of phrases and creates a
vocabulary dictionary which it returns.
The dictionary may be used to
generate one-hot vectors.
"""
def create_vocabulary( phrase_list ):
    # Start at one
    index = 1
    vocab = {}

    for phrase in phrase_list:
        # Tokenize phrase.
        words = nltk.tokenize.word_tokenize( phrase )

        # Looks for new words in phrase.
        for word in words:
            if not word in vocab:
                vocab[word] = index
                index += 1

    # Adds index for unknown token.
    vocab['UNK'] = index

    return vocab

"""
Given a vocabulary dictionary and
a list of phrases, this will return
a list of one hot vector matrices
representing each phrase.
"""
def to_one_hot( vocab, phrase_list, max_len ):
    # New list to contain one-hot vectors.
    new_list = np.zeros( ( len( phrase_list ), max_len, len( vocab.keys() ) ) )

    for i in range( len( phrase_list ) ):
        words = nltk.tokenize.word_tokenize( phrase_list[i] )

        # Sets word index to 1.
        for j in range( len( words ) ):
            new_list[i][j][vocab[words[j]]] = 1

    return new_list

"""
Converts vectors of classes to
phrases using the given vocabulary.
"""
def class_vectors_to_phrases( predictions, vocab ):
    phrases = []

    # Vocab maps words to indeces, we need the inverse.
    inverse_vocab = {value: key for key, value in vocab.iteritems()}

    for pred_vector in predictions:
        words = []

        for label in pred_vector:
            if label == 0:
                word = 'UNK'
            else:
                word = inverse_vocab[label]

            words.append( word )

            # If period, then sentence ended, so exit loop.
            if word == '.':
                break

        phrases.append( ' '.join( words ) )

    return phrases

"""
Takes all trajectories and pads
them with zero vector waypoints
in order to have trajectory sequences
of the same length.
"""
def pad_trajectory_sequences( traj_list, seq_len ):
    padded_trajectories = []

    # Padds each trajectory.
    for i in range( len( traj_list ) ):
        traj_dim = len( traj_list[i][0] )
        zero_vec = np.zeros( traj_dim )

        # New trajectory that will be padded.
        new_traj = []

        for j in range( seq_len ):
            if j >= len( traj_list[i] ):
                new_traj.append( zero_vec )
            else:
                new_traj.append( np.array( traj_list[i][j] ) )

        # Appends padded trajectory to list.
        padded_trajectories.append( np.array( new_traj ) )

    return np.array( padded_trajectories )


if __name__ == '__main__':
    dirs = set( [directory for directory in os.listdir( '../dataset/robobarista_dataset/dataset/' )] )
    dirs.remove( 'folds.json' )
    dirs = list( dirs )

    dataset = load_data_set( '../dataset/robobarista_dataset/dataset/', dirs )
    print ( dataset )
