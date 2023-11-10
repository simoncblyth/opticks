procedure_for_binary_release
============================

N:workstation/blyth update::
 
   N
   jre
   o
   git pull 
  
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



