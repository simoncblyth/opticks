procedure_for_binary_release
============================


Overview
---------

* do the build 
* create tarball 
* transfer tarball 
* check can use the release, first via ctests 

  * suspect the python included in install is deficient
    (relying on stuff from source tree that is not yet installed)
  * TODO: add failing test and fix by including all thats needed
    into the install 


Details
----------

N:workstation/blyth update::
 
   N
   jre
   o
   git pull    ## for real labelled releases (as opposed to tests) would need to checkout a tag 
  

R:workstation/simon build standalone opticks (shares same opticks working copy as N)::

   R
   vip                # switch .bashrc to .opticks_build_config for source tree build
   ./fresh_build.sh    # delete local/opticks and recreate and make tarball and extract it


fresh_build.sh from R::

    #!/bin/bash -l 

    rm -rf local/opticks  

    opticks-full   # recreate local/opticks (takes around 5 min), now installs ctests and cmake modules

    #opticks-t     # standard ctest 

    opticks-tar    # create tarball with okdist--


::


   ## NOTE LOCATION : /tmp/simon/opticks/okdist/Opticks-0.0.1_alpha.tar

   vip  # in .bashrc switch mode from "build" to "usage" for local binary release testing 
   x
   R   # exit and reconnect 

   ort    # cd $OPTICKS_PREFIX/tests
   ctest -N 
   ctest 

   scp -4 /tmp/simon/opticks/okdist/Opticks-0.0.1_alpha.tar L:g/local/     # copy tarball to L 
   scp -4 $(okdist-path) L:g/local/      # (from standard env can use the func)   

L:gateway/blyth::

   L
   cd g/local

   rm -rf Opticks-0.0.1_alpha         # remove the old expanded archive 
   tar xvf Opticks-0.0.1_alpha.tar    # extract the new one 

   vip   # check .bashrc and .opticks_usage_config 

   x
   L   # fresh session 

   ort         # cd $OPTICKS_RELEASE_PREFIX/tests
   ctest -N    # list the tests

   ctest       # expect around 19/205 FAILs for lack of GPU   

   sf # list the slurm related functions 
   sj # review the GPU job 


   /hpcfs/juno/junogpu/blyth/j/okjob.sh   # test run of script before submission on lxslc7 



Added custom4 to R:.opticks_build_config
---------------------------------------------

::

    # config system level pre-requisites 
    export OPTICKS_CUDA_PREFIX=/usr/local/cuda-11.7
    #export OPTICKS_OPTIX_PREFIX=/home/blyth/local/opticks/externals/OptiX_750
    export OPTICKS_OPTIX_PREFIX=/cvmfs/opticks.ihep.ac.cn/external/OptiX_750
    export OPTICKS_COMPUTE_CAPABILITY=70

    # PATH envvars control the externals that opticks/CMake will build against 
    unset CMAKE_PREFIX_PATH
    unset PKG_CONFIG_PATH
    source /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/Pre-Release/J23.1.x/ExternalLibs/Boost/1.78.0/bashrc
    source /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/Pre-Release/J23.1.x/ExternalLibs/Xercesc/3.2.3/bashrc
    source /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/Pre-Release/J23.1.x/ExternalLibs/CLHEP/2.4.1.0/bashrc
    source /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/Pre-Release/J23.1.x/ExternalLibs/Geant4/10.04.p02.juno/bashrc 
    source /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/Pre-Release/J23.1.0-rc3.dc1/ExternalLibs/custom4/0.1.8/bashrc

    # sourcing the bashrc and not opticks-prepend-prefix allows more intuitive ordering 

    # config opticks build : wheres the source, where to install etc..
    export OPTICKS_DOWNLOAD_CACHE=/data/opticks_download_cache
    export OPTICKS_HOME=$HOME/opticks
    export OPTICKS_PREFIX=/data/simon/local/opticks  
    export PYTHONPATH=$(dirname $OPTICKS_HOME)     ## HMM FIX: SOURCE TREE?, STOMPING,  TUCK AWAY 

    opticks-(){  [ -r $OPTICKS_HOME/opticks.bash ] && . $OPTICKS_HOME/opticks.bash && opticks-env $* ; } 
    opticks-

    o(){ opticks- ; cd $(opticks-home) ; git status ; } 
    oo(){ opticks- ; cd $(opticks-home) ; om- ; om-- ;  }




Packaging .opticks
--------------------

::

     N
     jre
     cd ~/.opticks
     ~/opticks/bin/oktar.py /tmp/tt/dot_opticks.tar create --prefix dot_opticks/v0 --mode CACHE

     N[blyth@localhost .opticks]$ scp -4 /tmp/tt/dot_opticks.tar L:g/.opticks/

Extract that archive with the two element prefix stripped:: 

     tar tvf dot_opticks.tar  # check the explosion
     
     L7[blyth@lxslc711 .opticks]$ rm -rf GEOM InputPhotons flight precooked rngcache  
         # clean ahead to avoid mixing 
      
     L7[blyth@lxslc711 .opticks]$ tar xvf dot_opticks.tar --strip-components=2



