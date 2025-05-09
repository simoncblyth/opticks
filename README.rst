
Opticks : A GPU Accelerated Optical Photon Simulation using NVIDIA OptiX  
==========================================================================

For presentations and videos about Opticks:

* http://simoncblyth.bitbucket.io

For instructions on building Opticks and externals: 

* http://simoncblyth.bitbucket.io/opticks/docs/opticks.html
* http://simoncblyth.bitbucket.io/opticks/docs/examples.html
* http://simoncblyth.bitbucket.io/opticks/

Related repositories:

* https://bitbucket.org/simoncblyth/env

Bitbucket and Github repositories

* https://bitbucket.org/simoncblyth/opticks
* https://github.com/simoncblyth/opticks

Currently the bitbucket opticks repository is used 
for day to day pushes with the github repository only 
being pushed to infrequently when making releases 
that are provided as github releases.
The Github repo is usually several months behind bitbucket 
so you are advised NOT to use it.


.. table::
    :align: center

    +----------------------------------------------+---------------------------------------------------------+
    | https://bitbucket.org/simoncblyth/opticks    | very latest code repository, unstable, breakage common  |     
    +----------------------------------------------+---------------------------------------------------------+
    | https://github.com/simoncblyth/opticks       | "releases" weeks/months behind, more stable             |     
    +----------------------------------------------+---------------------------------------------------------+
    | https://simoncblyth.bitbucket.io             | presentations and videos                                |    
    +----------------------------------------------+---------------------------------------------------------+
    | https://groups.io/g/opticks                  | forum/mailing list archive                              |    
    +----------------------------------------------+---------------------------------------------------------+
    | email:opticks+subscribe@groups.io            | subscribe to mailing list                               |    
    +----------------------------------------------+---------------------------------------------------------+ 



Installation instructions start with a clone::

    cd $HOME
    git clone http://bitbucket.org/simoncblyth/opticks  

If you have commit access to *opticks*, you need to use SSH::

    cd $HOME ;
    git clone git@bitbucket.org:simoncblyth/opticks.git

To update an existing clone::

    cd ~/opticks
    git remote -v   # should list bitbucket.org urls 
    git status
    git pull 

Setup opticks by copying *~/opticks/example.opticks_config* to your 
HOME directory and customizing it as instructed by the links therein::

    cp ~/opticks/example.opticks_config ~/.opticks_config
    vi ~/.opticks_config    # adapt PREFIX paths 
    echo "source ~/.opticks_config" >> .bashrc 

Then after starting a new bash session you can proceed with::

    opticks-info   # check bash hookup 
    opticks-full   # download and build externals and opticks



Overview of Opticks installation steps
----------------------------------------------

A high level overview of the sequence of steps to install Opticks are listed below.
For details see http://simoncblyth.bitbucket.io/opticks/


0. install "system" externals : NVIDIA GPU Driver, CUDA, OptiX 7, 7.5 or 8.0 (6.5 not supported)  
   following instructions from NVIDIA. Check they are working.
1. use git to clone opticks bitbucket repository to your home directory, creating ~/opticks
2. hookup the opticks bash functions to your bash shell 
  
   * cp ~/opticks/example.opticks_config ~/.opticks_config
   * ensure that your .bash_profile sources .bashrc
   * add line to .bashrc "source ~/.opticks_config"

3. start a new session and check the bash functions are hooked up correctly with:

   * **opticks-info**
   * bash -lc **"opticks-info"**

4. install the foreign externals OR use preexisting installs of boost,clhep,xercesc,g4

   * **opticks-foreign**     # lists them 
   * **opticks-foreign-install**    # installs them 

5. edit ~/.opticks_config setting the paths appropriately for the 
   prefixes of the "system" and "foreign" externals and setting 
   the prefix for the opticks install (eg /usr/local/opticks)

6. install the "automated" externals and opticks itself with **opticks-full**
7. translate a geometry using for example **g4cx/tests/G4CXOpticks_setGeometry_Test.sh**, see :doc:`docs/testing`
8. test the opticks build with **opticks-t** 



Orientation documents for developing an Understanding of the Opticks codebase
--------------------------------------------------------------------------------

* https://simoncblyth.bitbucket.io/opticks/docs/orientation.html

The orientation documentation seeks to highlight the Geant4+Opticks classes/functions 
that you need to be familiar with to understand how Opticks 
takes a Geant4 geometry and converts that into an  
NVIDIA OptiX 7+ geometry suitable for optical photon simulation.

By design the orientation is a far from complete guide to the codebase, I 
just focus on classes/functions that you should look at first when 
building your understanding.

The html is generated by Sphinx from readable .rst sources
with includes from comments in the source files themselves. 
The html orientation docs are best read whilst also 
looking at the sources, get those with::

   git clone https://bitbucket.org/simoncblyth/opticks

Feel free to ask for further things for me to add to these docs.



