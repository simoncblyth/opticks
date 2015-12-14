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


Torch via General Particle Source
-------------------------------------

* /usr/local/env/g4/geant4.10.02/source/event/include/G4SingleParticleSource.hh
* https://geant4.web.cern.ch/geant4/UserDocumentation/UsersGuides/ForApplicationDeveloper/html/ch02s07.html

Creates OpSource based on G4SingleParticleSource as need to pick position from generator.


[AVOIDED] Pushing to 1M photons get segv at cleanup
-----------------------------------------------------

**AVOIDED** by splitting propagation into separate "events" of 10k photons each, 
Geant4 doesnt handle large numbers of primaries.

Binary search to find max can propagate as ~65k only::

    cfg4 65485 # succeeds
    cfg4 65486 # segments

After switch to GPS manage a few less::

    cfg4 65483 # succeeds 

Ginormous dtor stack from chain of primary vertices::

    frame #196447: 0x0000000105416029 libG4particles.dylib`G4PrimaryVertex::~G4PrimaryVertex(this=0x000000010b5c7f90) + 25 at G4PrimaryVertex.cc:68
    frame #196448: 0x0000000105415f8c libG4particles.dylib`G4PrimaryVertex::~G4PrimaryVertex(this=0x000000010b5c7f30) + 156 at G4PrimaryVertex.cc:74
    frame #196449: 0x0000000105416005 libG4particles.dylib`G4PrimaryVertex::~G4PrimaryVertex(this=0x000000010b5c7f30) + 21 at G4PrimaryVertex.cc:68
    frame #196450: 0x0000000105416029 libG4particles.dylib`G4PrimaryVertex::~G4PrimaryVertex(this=0x000000010b5c7f30) + 25 at G4PrimaryVertex.cc:68
    frame #196451: 0x0000000105415f8c libG4particles.dylib`G4PrimaryVertex::~G4PrimaryVertex(this=0x000000010b5c7ed0) + 156 at G4PrimaryVertex.cc:74
    frame #196452: 0x0000000105416005 libG4particles.dylib`G4PrimaryVertex::~G4PrimaryVertex(this=0x000000010b5c7ed0) + 21 at G4PrimaryVertex.cc:68
    frame #196453: 0x0000000105416029 libG4particles.dylib`G4PrimaryVertex::~G4PrimaryVertex(this=0x000000010b5c7ed0) + 25 at G4PrimaryVertex.cc:68
    frame #196454: 0x0000000105415f8c libG4particles.dylib`G4PrimaryVertex::~G4PrimaryVertex(this=0x000000010b5c7e70) + 156 at G4PrimaryVertex.cc:74
    frame #196455: 0x0000000105416005 libG4particles.dylib`G4PrimaryVertex::~G4PrimaryVertex(this=0x000000010b5c7e70) + 21 at G4PrimaryVertex.cc:68
    frame #196456: 0x0000000105416029 libG4particles.dylib`G4PrimaryVertex::~G4PrimaryVertex(this=0x000000010b5c7e70) + 25 at G4PrimaryVertex.cc:68
    frame #196457: 0x0000000105415f8c libG4particles.dylib`G4PrimaryVertex::~G4PrimaryVertex(this=0x000000010b5c7e10) + 156 at G4PrimaryVertex.cc:74
    frame #196458: 0x0000000105416005 libG4particles.dylib`G4PrimaryVertex::~G4PrimaryVertex(this=0x000000010b5c7e10) + 21 at G4PrimaryVertex.cc:68
    frame #196459: 0x0000000105416029 libG4particles.dylib`G4PrimaryVertex::~G4PrimaryVertex(this=0x000000010b5c7e10) + 25 at G4PrimaryVertex.cc:68
    frame #196460: 0x0000000105415f8c libG4particles.dylib`G4PrimaryVertex::~G4PrimaryVertex(this=0x000000010b5c7db0) + 156 at G4PrimaryVertex.cc:74
    frame #196461: 0x0000000105416005 libG4particles.dylib`G4PrimaryVertex::~G4PrimaryVertex(this=0x000000010b5c7db0) + 21 at G4PrimaryVertex.cc:68
    frame #196462: 0x0000000105416029 libG4particles.dylib`G4PrimaryVertex::~G4PrimaryVertex(this=0x000000010b5c7db0) + 25 at G4PrimaryVertex.cc:68
    frame #196463: 0x0000000102264c88 libG4event.dylib`G4Event::~G4Event(this=0x000000010b5c7690) + 72 at G4Event.cc:64
    frame #196464: 0x0000000102264e55 libG4event.dylib`G4Event::~G4Event(this=0x000000010b5c7690) + 21 at G4Event.cc:63
    frame #196465: 0x0000000102194eb8 libG4run.dylib`G4RunManager::StackPreviousEvent(this=0x0000000107e12ee0, anEvent=0x000000010b5c7690) + 152 at G4RunManager.cc:546
    frame #196466: 0x0000000102194df0 libG4run.dylib`G4RunManager::TerminateOneEvent(this=0x0000000107e12ee0) + 32 at G4RunManager.cc:407
    frame #196467: 0x0000000102194ac2 libG4run.dylib`G4RunManager::DoEventLoop(this=0x0000000107e12ee0, n_event=1, macroFile=0x0000000000000000, n_select=-1) + 114 at G4RunManager.cc:368
    frame #196468: 0x00000001021938e4 libG4run.dylib`G4RunManager::BeamOn(this=0x0000000107e12ee0, n_event=1, macroFile=0x0000000000000000, n_select=-1) + 196 at G4RunManager.cc:273
    frame #196469: 0x000000010000c96e cfg4test`main(argc=3, argv=0x00007fff5fbfea48) + 702 at cfg4.cc:47
    frame #196470: 0x00007fff87c165fd libdyld.dylib`start + 1
    frame #196471: 0x00007fff87c165fd libdyld.dylib`start + 1



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

cfg4---(){
   cfg4-wipe
   cfg4-cmake
   cfg4-make
   cfg4-install
}

cfg4--(){
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

cfg4-dbg(){
   local bin=$(cfg4-bin)
   cfg4-export
   lldb $bin -- $*
}




