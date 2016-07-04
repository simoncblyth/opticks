
Opticks : A GPU Accelerated Optical Photon Simulation using NVIDIA OptiX  
==========================================================================

This *opticks* repository was spawned from its cradle *env* repository on July 4th, 2016.

Clone *opticks* with::

    cd $HOME ;
    hg clone http://bitbucket.org/simoncblyth/opticks  

If you need to commit to *opticks*, you need to use SSH::

    cd $HOME ;
    hg clone ssh://hg@bitbucket.org/simoncblyth/opticks   


Source opticks from your bash shell profile with::

    export OPTICKS_HOME=$HOME/opticks
    opticks-(){  [ -r $OPTICKS_HOME/opticks.bash ] && . $OPTICKS_HOME/opticks.bash && opticks-env $* ; } 
    opticks-
    o(){ cd $OPTICKS_HOME ; hg st ; } 

Transitionally it may be necessary for developers to also clone the *env* repository
and similarly source *env*. If so that should be done prior to opticks.
More details about bitbucket/mercurial setup can be found on the *env* frontpage:

* https://bitbucket.org/simoncblyth/env




