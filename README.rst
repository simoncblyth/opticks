
Opticks : A GPU Accelerated Optical Photon Simulation using NVIDIA OptiX  
==========================================================================

For presentations and videos about Opticks:

* http://simoncblyth.bitbucket.org

For instructions on building Opticks and externals: 

* http://simoncblyth.bitbucket.org/opticks/

Related repositories:

* https://bitbucket.org/simoncblyth/env
* https://bitbucket.org/simoncblyth/g4dae



Installation instructions start with a clone::

    cd $HOME ;
    hg clone http://bitbucket.org/simoncblyth/opticks  

For commit access to *opticks*, you need to use SSH::

    cd $HOME ;
    hg clone ssh://hg@bitbucket.org/simoncblyth/opticks   

Source opticks from your bash shell profile with::

    export LOCAL_BASE=/usr/local       # folder beneath which opticks will be installed
    export OPTICKS_HOME=$HOME/opticks
    opticks-(){  [ -r $OPTICKS_HOME/opticks.bash ] && . $OPTICKS_HOME/opticks.bash && opticks-env $* ; } 
    opticks-
    o(){ cd $OPTICKS_HOME ; hg st ; } 




