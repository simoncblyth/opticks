
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
The Github repo is usually several months behind bitbucket 
so you are advised NOT to use it.


Installation instructions start with a clone::

    cd $HOME
    git clone http://bitbucket.org/simoncblyth/opticks  

To update an existing clone::

    cd ~/opticks
    git remote -v   # should list bitbucket.org urls 
    git status
    git pull 

Setup opticks by copying *~/opticks/example.opticks_config* to your 
HOME directory and customizing it as instructed therein::

    cp ~/opticks/example.opticks_config ~/.opticks_config
    vi ~/.opticks_config    # adapt PREFIX paths 
    echo "source ~/.opticks_config" >> .bashrc 

Then after starting a new bash session you can proceed with::

    opticks-info   # check bash hookup 
    opticks-full   # download and build externals and opticks

If you have commit access to *opticks*, you need to use SSH::

    cd $HOME ;
    git clone git@bitbucket.org:simoncblyth/opticks.git


