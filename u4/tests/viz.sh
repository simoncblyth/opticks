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

Find some big bouncer BPID to look at::

    In [4]: np.c_[np.where(b.n>5)[0], b.q[b.n>5]]
    Out[4]: 
    array([[b'56', b'TO BT BR BR BR BT SA                                                                            '],
           [b'865', b'TO BT BT BR BR SR SR SR SR SR SR SR SA                                                          '],
           [b'877', b'TO BT BT BR BR SR SR SA                                                                         '],
           [b'896', b'TO BT BT BR SR SR SR SR SR SR SA                                                                '],
           [b'943', b'TO BT BT BR SR SR SR SR SR SA                                                                   '],
           ...,
           [b'9057', b'TO BT BT BR SR SR SR SR SR SA                                                                   '],
           [b'9073', b'TO BT BT BR SR SR SR SR SR BR SA                                                                '],
           [b'9076', b'TO BT BT BR SR SR SR SR SR SR SR SR BT BT SA                                                    '],
           [b'9092', b'TO BT BT BR SR SR SR SR SR SR SR BR BR BR BR SA                                                 '],
           [b'9107', b'TO BT BT BT BT SA                                                                               ']], dtype='|S96')




Photons with "SC" scatter out of the plane, use pyvista 3D plotting with::

   MODE=3 BPID=... ./viz.sh 


Adjusting view and annotation for viaibility::

    APID=10 FOCUS=0,0,260 TIGHT=1 ./viz.sh 

    BPID=9092 BOPT=idx,pdy FOCUS=0,0,280 TIGHT=1 HV_THIRDLINE=0.5,0.97  ./viz.sh 



EOU
}


DIR=$(dirname $BASH_SOURCE)

defarg="ana"
arg=${1:-$defarg}
script=$DIR/U4SimtraceTest.sh
aopt=idx
bopt=idx

if [ -n "$APID" -a -n "$BPID" ]; then 
    N=1 APID=$APID BPID=$BPID $script $arg
elif [ -n "$APID" ]; then
    N=0 APID=$APID AOPT=${AOPT:-$aopt} $script $arg
elif [ -n "$BPID" ]; then
    N=1 BPID=$BPID BOPT=${BOPT:-$bopt} $script $arg
elif [ "$arg" == "runboth" ]; then
    N=0 $script run
    N=1 $script run
else
    N=${N:-1} $script $arg
fi 






