opnovice-source(){ echo $BASH_SOURCE ; }
opnovice-vi(){ vi $(opnovice-source) ; }
opnovice-env(){  olocal- ; opticks- ; }
opnovice-usage(){ cat << EOU
OpNovice : Changes needed to apply Opticks to OpNovice example
================================================================

CMakeLists.txt 
---------------

Need to adopt Opticks CMake machinery to bring in 
the Opticks G4OK package and the Opticks G4 external.

Note that the Opticks external Geant4 is found via:: 

  -DCMAKE_PREFIX_PATH=$(opticks-prefix)/externals 

There is no need (so long as only one installed G4 version in externals) 
to use::

  -DGeant4_DIR=$(g4-cmake-dir)


1. need newer CMake 3.5 (actually may be able to live with 2.6? 
   but will requires some changes as BCM requires that)

2. adopt modern CMake config style (based on targets rather than variables)  


Minor particle iterator change
--------------------------------

Attempt to use the version of the example from geant4_10_04_p01
with the current version of the Opticks external geant4_10_02_p01
did not compile due to a change with particle iteration, returned
to old style.


Compare with .out
-------------------

::

    epsilon:OpNovice blyth$ OpNovice -m OpNovice.in > OpNovice.out.now
    epsilon:OpNovice blyth$ diff OpNovice.out OpNovice.out.now


Not the same (but not very different), 
but then I had to make a code change regarding water surface property.
So lets start again with the version that matches the Opticks G4 external version.

::

    epsilon:OpNovice blyth$ hg st .
    M CMakeLists.txt
    M src/OpNoviceDetectorConstruction.cc
    M src/OpNovicePhysicsList.cc
    epsilon:OpNovice blyth$ 
  

Keep copy::

    cp CMakeLists.txt ~/


Bring in the Opticks current version
--------------------------------------

::

    epsilon:Geant4 blyth$ cp -r $(g4x-curdir) .
    epsilon:Geant4 blyth$ g4x-curdir
    /usr/local/opticks/externals/g4/geant4_10_02_p01/examples/extended/optical/OpNovice
    epsilon:Geant4 blyth$ 

::

    cp ~/CMakeLists.txt .
    opnovice-;opnovice-conf clean
    opnovice-make
        ## notice the reason for theParticleIterator change : a warning 

    opnovice-run 
        ## now no substantive difference



EOU
}

opnovice-diff()
{
    g4x-
    G4X_NAME=extended/optical/OpNovice g4x-diff 
}

opnovice-sdir(){ echo $(opticks-home)/examples/Geant4/OpNovice ; }
opnovice-cd(){   cd $(opnovice-sdir) ; } 

opnovice-bdir(){ echo /tmp/$USER/opticks/examples/Geant4/OpNovice ; }
opnovice-bcd(){   cd $(opnovice-bdir) ; } 

opnovice-conf(){
   local sdir=$(opnovice-sdir)
   local bdir=$(opnovice-bdir)

   if [ "$1" == "clean" ]; then 
       echo $msg remove bdir $bdir
       rm -rf $bdir 
   fi
   mkdir -p $bdir && cd $bdir && pwd 

   cmake $sdir -DCMAKE_BUILD_TYPE=Debug \
            -DCMAKE_PREFIX_PATH=$(opticks-prefix)/externals \
            -DCMAKE_INSTALL_PREFIX=$(opticks-prefix) \
            -DCMAKE_MODULE_PATH=$(opticks-home)/cmake/Modules    

}

opnovice-make()
{
   local iwd=$(pwd)
   opnovice-bcd
   pwd
   make ${1:-install}
   cd $iwd
}


opnovice-run()
{
   g4-
   g4-export

   opnovice-cd

   $(opticks-prefix)/lib/OpNovice -m OpNovice.in > OpNovice.out.now

   diff OpNovice.out OpNovice.out.now 

}

opnovice-run-note(){ cat << EON

epsilon:OpNovice blyth$ diff OpNovice.out OpNovice.out.now 
2,5d1
<         ############################################
<         !!! WARNING - FPE detection is activated !!!
<         ############################################
< 
300,306d295
< OpenGLImmediateQt (OGLIQt, OGLI)
< OpenGLStoredQt (OGLSQt, OGL, OGLS)
< OpenGLImmediateXm (OGLIXm, OGLIQt_FALLBACK)
< OpenGLStoredXm (OGLSXm, OGLSQt_FALLBACK)
< OpenGLImmediateX (OGLIX, OGLIQt_FALLBACK, OGLIXm_FALLBACK)
< OpenGLStoredX (OGLSX, OGLSQt_FALLBACK, OGLSXm_FALLBACK)
< RayTracerX (RayTracerX)
504c493
<       Sampling table 17x1001 from 1 GeV to 10 TeV 
---
>       Sampling table 17x1001; from 1 GeV to 10 TeV 
531c520
<       Sampling table 17x1001 from 1 GeV to 10 TeV 
---
>       Sampling table 17x1001; from 1 GeV to 10 TeV 
583c572
< number of event = 1 User=0.01s Real=0.01s Sys=0s
---
> number of event = 1 User=0s Real=0s Sys=0s
epsilon:OpNovice blyth$ 


EON
}



