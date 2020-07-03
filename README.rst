
Opticks : A GPU Accelerated Optical Photon Simulation using NVIDIA OptiX  
==========================================================================

For presentations and videos about Opticks:

* http://simoncblyth.bitbucket.io

For instructions on building Opticks and externals: 

* http://simoncblyth.bitbucket.io/opticks/

Related repositories:

* https://bitbucket.org/simoncblyth/env
* https://bitbucket.org/simoncblyth/g4dae


Bitbucket and Github repositories

* https://bitbucket.org/simoncblyth/opticks
* https://github.com/simoncblyth/opticks

Currently the bitbucket opticks repository is used 
for day to day pushes with the github repository only 
being pushed to infrequently when making releases 
that are provided as github releases.

Installation instructions start with a clone::

    cd $HOME ;
    git clone http://bitbucket.org/simoncblyth/opticks  
    git clone http://github.com/simoncblyth/opticks  

For commit access to *opticks*, you need to use SSH::

    cd $HOME ;
    git clone git@bitbucket.org:simoncblyth/opticks.git


Source opticks from your bash shell profile with::

    export LOCAL_BASE=/usr/local       # folder beneath which opticks will be installed
    export OPTICKS_HOME=$HOME/opticks
    opticks-(){  [ -r $OPTICKS_HOME/opticks.bash ] && . $OPTICKS_HOME/opticks.bash && opticks-env $* ; } 
    opticks-
    o(){ cd $OPTICKS_HOME ; hg st ; } 




