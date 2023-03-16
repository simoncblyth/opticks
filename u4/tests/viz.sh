#!/bin/bash -l 

usage(){ cat << EOU
viz.sh
========


Use qi first indices of each history as APID/BPID indices to viz::

    In [1]: a.qtab_,a.qtab
    Out[1]: 
    ('np.c_[qn,qi,qu][quo]',
     array([[b'900', b'0', b'TO BT SR BT SA                                                                                  '],
            [b'93', b'12', b'TO BT SA                                                                                        '],
            [b'2', b'600', b'TO BT SR BT AB                                                                                  '],
            [b'2', b'239', b'TO BR SA                                                                                        '],
            [b'1', b'484', b'TO BT SR BR SR BT SA                                                                            '],
            [b'1', b'938', b'TO BT SR AB                                                                                     '],
            [b'1', b'964', b'TO BT AB                                                                                        ']], dtype='|S96'))

    epsilon:tests blyth$ APID=484 ./viz.sh 


Photons with "SC" scatter out of the plane, use pyvista 3D plotting with::

   MODE=3 BPID=... ./viz.sh 


EOU
}


DIR=$(dirname $BASH_SOURCE)

defarg="ana"
arg=${1:-$defarg}
script=$DIR/U4SimtraceTest.sh

if [ -n "$APID" -a -n "$BPID" ]; then 
    N=1 APID=$APID BPID=$BPID $script $arg
elif [ -n "$APID" ]; then
    N=0 APID=$APID AOPT=idx $script $arg
elif [ -n "$BPID" ]; then
    N=1 BPID=$BPID BOPT=idx $script $arg
elif [ "$arg" == "runboth" ]; then
    N=0 $script run
    N=1 $script run
else
    N=${N:-1} $script $arg
fi 






