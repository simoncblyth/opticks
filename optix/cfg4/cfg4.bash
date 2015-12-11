cfg4-src(){      echo optix/cfg4/cfg4.bash ; }
cfg4-source(){   echo ${BASH_SOURCE:-$(env-home)/$(cfg4-src)} ; }
cfg4-vi(){       vi $(cfg4-source) ; }
cfg4-usage(){ cat << EOU

Comparisons against Geant4
===========================

Objectives
------------

* Construct Geant4 test geometries and light sources from the same commandline
  arguments as ggv invokations like ggv-rainbow, ggv-prism.

* Add requisite step action(?) to record photon steps in the same format as
  optixrap- using NPY 


1st approach : try to follow Chroma g4py use of Geant4 
---------------------------------------------------------

* /usr/local/env/chroma_env/src/chroma/chroma/generator
* ~/env/chroma/chroma_geant4_integration.rst


2nd approach : C++ following Geant4 examples 
----------------------------------------------

* reuse ggeo- machinery as much as possible


Geant4 Stepping Action
------------------------

Coordination with EventAction extended/electromagnetic/TestEm4/src/SteppingAction.cc::

     55 void SteppingAction::UserSteppingAction(const G4Step* aStep)
     56 {
     57  G4double EdepStep = aStep->GetTotalEnergyDeposit();
     58  if (EdepStep > 0.) fEventAction->addEdep(EdepStep);

With detector extended/polarisation/Pol01/src/SteppingAction.cc




EOU
}

cfg4-env(){  
   elocal- 
   g4-
}


cfg4-name(){ echo cfg4test ; }
cfg4-bin(){ echo ${CFG4_BINARY:-$(cfg4-idir)/bin/$(cfg4-name)} ; }

cfg4-idir(){ echo $(local-base)/env/optix/cfg4; } 
cfg4-bdir(){ echo $(local-base)/env/optix/cfg4.build ; }
cfg4-sdir(){ echo $(env-home)/optix/cfg4 ; }

cfg4-icd(){  cd $(cfg4-idir); }
cfg4-bcd(){  cd $(cfg4-bdir); }
cfg4-scd(){  cd $(cfg4-sdir); }

cfg4-dir(){  echo $(cfg4-sdir) ; }
cfg4-cd(){   cd $(cfg4-dir); }



cfg4-wipe(){
    local bdir=$(cfg4-bdir)
    rm -rf $bdir
}

cfg4-cmake(){
   local iwd=$PWD
   local bdir=$(cfg4-bdir)
   mkdir -p $bdir
   cfg4-bcd

  # -DWITH_GEANT4_UIVIS=OFF \

   cmake \
         -DGeant4_DIR=$(g4-cmake-dir) \
         -DCMAKE_INSTALL_PREFIX=$(cfg4-idir) \
         -DCMAKE_BUILD_TYPE=Debug  \
         $(cfg4-sdir)
   cd $iwd 
}

cfg4-make(){
    local iwd=$PWD
    cfg4-bcd
    make $*
    cd $iwd 
}

cfg4-install(){
   cfg4-make install
}

cfg4--(){
   cfg4-wipe
   cfg4-cmake
   cfg4-make
   cfg4-install
}

cfg4-export()
{
   g4-export
}

cfg4-run(){
   local bin=$(cfg4-bin)
   cfg4-export
   $bin $*
}



