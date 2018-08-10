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


Thoughts on the approach to take
------------------------------------

* aim to minimize code changes within the example

* instead place everything that can live in G4OK there, as this avoids 
  repetition for every usage of Opticks 


Hmm how to handle Opticks logging in embedded mode ? 
--------------------------------------------------------

* PLOG wants access to argc, argv : but Opticks is embedded so it aint really appropriate ?
  the example or the users code is in control of such things ... 

  * perhaps move to envvar OPTICKS_ARGV for arguments directed to the embedded Opticks 

* where to do the logging setup that Opticks executables usually do first thing in main ?
  G4OpticksManager ctor ? Will that work being done in a lib ? Hmm could require an 
  initialization in main ? Actually probably better to make it clear that Opticks is in use. 

  Need a way to shorten the excessively long list of logging includes and macro invokations.

* see sysrap/PLOG_review.rst for reminders on the setup which uses macros a lot because
  some things must be executed in main, but they need to be defined for every package 


world-pv to Geometry/Material/Surface digests to see if already have a cached geometry
---------------------------------------------------------------------------------------

Presumably a G4 equivalent of GGeo::getProgenyDigest in g4ok would be the starting point, but thats
based in a GGeo GNode tree. Perhaps can use cfg4 CTraverser ?  That needs an Opticks instance...
So how to instanciate Opticks in embedded mode ?

Current OpMgr assumes the geocache is present and correct and immediately loads it::

     78 OpMgr::OpMgr(int argc, char** argv, const char* argforced )
     79     :
     80     m_log(new SLog("OpMgr::OpMgr")),
     81     m_ok(new Opticks(argc, argv, argforced)),
     82     m_hub(new OpticksHub(m_ok)),            // immediate configure and loadGeometry 
     83     m_idx(new OpticksIdx(m_hub)),
     84     m_num_event(m_ok->getMultiEvent()),     // after hub instanciation, as that configures Opticks
     85     m_gen(m_hub->getGen()),
     86     m_run(m_hub->getRun()),
     87     m_propagator(new OpPropagator(m_hub, m_idx)),
     88     m_count(0),
     89     m_opevt(NULL)
     90 {


Even though may end up doing the geocache check inside OpticksHub tis 
convenient to have the Opticks instance outside OpMgr 




 

EOU
}

opnovice-diff()
{
    g4x-
    G4X_NAME=extended/optical/OpNovice g4x-diff 
}

opnovice-sdir(){ echo $(opticks-home)/examples/Geant4/OpNovice ; }
opnovice-cd(){   cd $(opnovice-sdir) ; } 
opnovice-c(){    cd $(opnovice-sdir) ; } 

opnovice-bdir(){ echo /tmp/$USER/opticks/examples/Geant4/OpNovice ; }
opnovice-bcd(){   cd $(opnovice-bdir) ; } 
opnovice-b(){     cd $(opnovice-bdir) ; } 

opnovice-info(){ cat << EOI

   opnovice-sdir : $(opnovice-sdir)
   opnovice-bdir : $(opnovice-bdir)


EOI
}



opnovice-conf()
{
   local iwd=$(pwd)
   local sdir=$(opnovice-sdir)
   local bdir=$(opnovice-bdir)

   if [ "$1" == "clean" ]; then 
       echo $msg remove bdir $bdir
       rm -rf $bdir 
   fi
   mkdir -p $bdir && cd $bdir && pwd 

   cmake $sdir \
         -DCMAKE_BUILD_TYPE=Debug \
         -DCMAKE_PREFIX_PATH=$(opticks-prefix)/externals \
         -DCMAKE_INSTALL_PREFIX=$(opticks-prefix) \
         -DCMAKE_MODULE_PATH=$(opticks-home)/cmake/Modules    

   cd $iwd
}

opnovice-make()
{
   local msg="=== $FUNCNAME :"
   local iwd=$(pwd)
   local bdir=$(opnovice-bdir)

   [ ! -d "$bdir" ] && echo $msg build dir $bdir does not exist : run opnovice-conf to create it && return 

   opnovice-bcd
   pwd
   make ${1:-install}
   cd $iwd
}

opnovice--(){ opnovice-make ; }


opnovice-run()
{
   g4-
   g4-export

   opnovice-cd

   #export OPTICKS_ARGS="--ggeo trace" 
   unset OPTICKS_ARGS
   
   lldb $(opticks-prefix)/lib/OpNovice --  -m OpNovice.in #> OpNovice.out.now

   #diff OpNovice.out OpNovice.out.now 

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



opnovice-cls () 
{ 
    local iwd=$PWD;
    opnovice-cd;
    opnovice-cls- . $*;
    cd $iwd
}

opnovice-cls- () 
{ 
    local base=${1:-.};
    local name=${2:-OpNovice};
    local hh=$(find $base -name "$name.hh");
    local cc=$(find $base -name "$name.cc");
    local vcmd="vi $hh $cc ";
    echo $vcmd;
    eval $vcmd
}

opnovice-lldb-notes(){ cat << EON

(lldb) b G4Cerenkov::GetMeanFreePath(G4Track const&, double, G4ForceCondition*)
Breakpoint 2: 2 locations.
(lldb) b G4Cerenkov::PostStepDoIt(G4Track const&, G4Step const&) 
Breakpoint 3: 2 locations.
(lldb) 

(lldb) b G4Cerenkov::PostStepGetPhysicalInteractionLength(G4Track const&, double, G4ForceCondition*)
(lldb) b G4Cerenkov::GetAverageNumberOfPhotons(double, double, G4Material const*, G4PhysicsOrderedFreeVector*) const 


(lldb) b G4SteppingManager::Stepping()


EON

}


