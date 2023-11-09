procedure_for_binary_release
============================

N:workstation/blyth update::
 
   N
   jre
   o
   git pull 
  
R:workstation/simon build standalone opticks (shares same opticks working copy as N)::

   R
   vip  # switch .bashrc to .opticks_standard_config for source tree build

   ./fresh_build.sh    # delete local/opticks

   opticks-full        # recreate local/opticks (takes around 5 min)

   opticks-t           # standard install ctest 

   okdist-
   okdist--            # create tarball and extract it into opticks_releases

   ## HMM : SOME OF WHAT okdist-- DOES (metadata + collecting ctest files) 
   ## COULD BE ADDED TO opticks-full ? 

   ## NOTE LOCATION : /tmp/simon/opticks/okdist/Opticks-0.0.1_alpha.tar
   ## binary release extracted to /data/simon/local/opticks_release/Opticks-0.0.1_alpha/x86_64-CentOS7-gcc1120-geant4_10_04_p02-dbg

   vip  # in .bashrc switch from .opticks_standard_config to  .opticks_release_config for local binary release testing 
   x
   R   # exit and reconnect 

   ort    # cd $OPTICKS_RELEASE_PREFIX/tests
   ctest -N 
   ctest 

   scp -4 /tmp/simon/opticks/okdist/Opticks-0.0.1_alpha.tar L:g/local/     # copy tarball to L 
   scp -4 $(okdist-path) L:g/local/      # (from standard env can use the func)   

L:gateway/blyth::

   L
   cd g/local

   rm -rf Opticks-0.0.1_alpha         # remove the old expanded archive 
   tar xvf Opticks-0.0.1_alpha.tar    # extract the new one 

   vip   # check .opticks_release_config OPTICKS_RELEASE_PREFIX is correct 

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



Potential issues with binary release running
----------------------------------------------

1. Lack of AFS permissions to create $HOME/.opticks : causes all SysRap tests to fail 
2. Lack of $HOME/.opticks/InputPhotons 





