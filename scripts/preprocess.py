#!/usr/bin/python

import sys
import os
import yaml
import glob


"""
Gets part names from the 
info.yaml file in the given 
object directory. 
"""
def get_part_names(obj_dir):
        part_list = []

        yaml_file = open(obj_dir + '/info.yaml')

        return [part for part in yaml.load(yaml_file)['parts']]

"""
Reads a trajectory file and returns it as
an array of waypoints. 
"""
def get_trajectory_from_file(traj_file_name):
    trajectory = []
    traj_file = open(traj_file_name, 'r')

    #Assigns an integer to grip type. 
    types = {'hold': 1.0, 'open': 2.0, 'close': 3.0}

    for line in traj_file:
        if line.startswith('"'):
            traj_type = types[line.strip().strip('"')]

        #Waypoint line. 
        else:
            #Reads in parameter values. 
            params = [float(param.strip()) for param in line.strip().strip('[').strip(']').split(',')]
          
            #Formats waypoint and appends it to trajectory. 
            waypoint = [traj_type] + params

            trajectory.append(waypoint)

    return trajectory

"""
Loads training set specified by a list
of object directories into arrays
that may be used by our models. 
"""
def load_data_set(root_dir, obj_dir_list):
    trajectories = []
    point_clouds = []
    nl_descriptions = []

    #Compiles all triplets (t, p, l) into arrays from a given object list.
    for obj_dir in obj_dir_list:
        path = root_dir + obj_dir

        #Gets parts and their names from info file. 
        parts = get_part_names(path)

        #Get all natural language descriptions. 
        descriptions = {}

        for manual_file in glob.glob(path + '/manual*'):
            for part, desc in yaml.load(open(manual_file, 'r'))['steps']:
                
                #Gets description for part. 
                descriptions[part] = desc

        #Create data points. 
        for part in parts:
            part_num = part.split('_')[1]

            #Gets natural language description for part.
            if part in descriptions:
                description = descriptions[part]
            else:
                #Don't add point if there is no natural language descrption for this part. 
                continue

            #Gets point cloud name for part.
            point_cloud_name = 'pointcloud_' + obj_dir + '_' + part
            point_cloud_path = path + '/' + point_cloud_name

            #Make sure it exists in case names aren't completely standardized. 
            if not os.path.isfile(point_cloud_path):
                print 'Point cloud path: "' + point_cloud_path + '" does not specify a file, verify it exists and that code is correct!'
                sys.exit()

            #Gets all trajectories for a part.  
            traj_path = path + '/user_input/'

            for traj_file in glob.glob(traj_path + '*_' + str(part_num)):
                traj = get_trajectory_from_file(traj_file)

                #Add triplet to lists at same index. 
                trajectories.append(traj)
                point_clouds.append(point_cloud_path)
                nl_descriptions.append(description)

    #Make sure that everything is of the same dimensionality. 
    assert (len(trajectories) == len(point_clouds)) and (len(trajectories) == len(nl_descriptions))

    return [trajectories, point_clouds, nl_descriptions]

if __name__ == '__main__':
    dirs = set([directory for directory in os.listdir('../dataset/robobarista_dataset/dataset/')])
    dirs.remove('folds.json')
    dirs = list(dirs)

    dataset = load_data_set('../dataset/robobarista_dataset/dataset/', dirs)
    print dataset
