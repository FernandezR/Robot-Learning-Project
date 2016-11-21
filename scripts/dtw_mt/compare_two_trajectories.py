#!/usr/bin/env python

import optparse
import os
import sys
import cPickle as pickle

from dtw_mt import compute_dtw_mt
from trajectory import Trajectory


# compute DTW-measure between two traj files
def compute_metric( traj_filename1, traj_filename2 ):
    print "loading %s" % traj_filename1
    traj1 = Trajectory()
    traj1.load_from_file( traj_filename1 )

    print "loading %s" % traj_filename2
    traj2 = Trajectory()
    traj2.load_from_file( traj_filename2 )

    dtw_value = compute_dtw_mt( traj1, traj2 )

    print ( "" )
    print ( "DTW-MT between two trajectories: %f" % dtw_value )

    pickle.dump( dtw_value, open( "dtw_value.p", "wb" ) )



def main():
    parser = optparse.OptionParser( usage = "Usage: %prog [options] traj_file1 traj_file2" )
    parser.add_option( '-d', '--debug', help = 'show debug info',
                    action = "store_true", default = False,
                    dest = "debug" )
    ( opts, args ) = parser.parse_args()

    DEBUG = opts.debug

    if len( args ) != 2:
        print "Give trajectory files!!"
        parser.print_help()
        exit( -1 )

    traj_filename1 = args[0]
    traj_filename2 = args[1]

    compute_metric( traj_filename1, traj_filename2 )

if __name__ == "__main__":
    main()

