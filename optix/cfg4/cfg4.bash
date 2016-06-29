cfg4-rel(){      echo optix/cfg4 ; }
cfg4-src(){      echo $(cfg4-rel)/cfg4.bash ; }
cfg4-source(){   echo ${BASH_SOURCE:-$(env-home)/$(cfg4-src)} ; }
cfg4-vi(){       vi $(cfg4-source) ; }
cfg4-usage(){ cat << EOU

Comparisons of Opticks against Geant4 for simple test geometries
==================================================================

Features
------------

* Construct Geant4 test geometries and light sources from the same commandline
  arguments as ggv invokations like ggv-rainbow, ggv-prism.

* G4 stepping action Recorder that records photon steps in Opticks format 


See Also
--------

* cg4- for notes on development of full geometry Opticks/Geant4 integration/comparison


TODO
----

* light source config, blackbody


macOS : 2 test fails
-----------------------

::

    simon:cfg4 blyth$ CPropLibTest
    2016-06-29 14:53:38.811 INFO  [13144929] [main@18] CPropLibTest
    2016-06-29 14:53:38.820 WARN  [13144929] [CPropLib::init@84] CPropLib::init surface lib sensor_surface NULL 
    2016-06-29 14:53:38.820 INFO  [13144929] [CPropLib::checkConstants@122] CPropLib::checkConstants mm 1 MeV 1 nanosecond 1 ns 1 nm 1e-06 GC::nanometer 1e-06 h_Planck 4.13567e-12 GC::h_Planck 4.13567e-12 c_light 299.792 GC::c_light 299.792 dscale 0.00123984
    2016-06-29 14:53:38.820 INFO  [13144929] [main@26] CPropLibTest convert 
    2016-06-29 14:53:38.821 WARN  [13144929] [CPropLib::addConstProperty@401] CPropLib::addConstProperty OVERRIDE GdDopedLS.SCINTILLATIONYIELD from 11522 to 10
    2016-06-29 14:53:38.821 WARN  [13144929] [CPropLib::addConstProperty@401] CPropLib::addConstProperty OVERRIDE LiquidScintillator.SCINTILLATIONYIELD from 11522 to 10
    2016-06-29 14:53:38.821 FATAL [13144929] [*CPropLib::makeMaterialPropertiesTable@317] CPropLib::makeMaterialPropertiesTable material with SENSOR_MATERIAL name Bialkali but no sensor_surface 
    Assertion failed: (surf), function makeMaterialPropertiesTable, file /Users/blyth/env/optix/cfg4/CPropLib.cc, line 322.
    Abort trap: 6
    simon:cfg4 blyth$ CG4Test
    2016-06-29 14:54:11.079 INFO  [13145139] [main@18] CG4Test
      0 : CG4Test
    CG4::init opticks summary
    valid    :valid
    envprefix: OPTICKS_
    geokey   : DAE_NAME_DYB
    daepath  : /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.dae
    gdmlpath : /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.gdml
    metapath : /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.ini
    query    : range:3153:12221
    ctrl     : volnames
    digest   : 96ff965744a2f6b78c24e33c80d3a4cd
    idpath   : /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae
    idfold   : /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300
    idbase   : /usr/local/opticks/opticksdata/export
    detector : dayabay
    detector_name : DayaBay
    detector_base : /usr/local/opticks/opticksdata/export/DayaBay
    getPmtPath(0) : /usr/local/opticks/opticksdata/export/DayaBay/GPmt/0
    meshfix  : NULL
    2016-06-29 14:54:11.082 INFO  [13145139] [main@32]   CG4 ctor DONE 
    2016-06-29 14:54:11.082 INFO  [13145139] [CG4::configure@129] CG4::configure g4ui 0

    -------- EEEE ------- G4Exception-START -------- EEEE -------
    *** G4Exception : PART70000
          issued by : G4NuclideTable
    G4ENSDFSTATEDATA environment variable must be set
    *** Fatal Exception *** core dump ***
    -------- EEEE -------- G4Exception-END --------- EEEE -------


    *** G4Exception: Aborting execution ***
    Abort trap: 6
    simon:cfg4 blyth$ echo $?
    134





Windows : Resource Issue ?
-----------------------------

::

    2016-06-28 21:00:05.591 DEBUG [13284] [CSteppingAction::UserSteppingAction@199]     (step) event_id 33 track_id 2902 track_step_count 1 step_id 0 trackStatus fAlive
    2016-06-28 21:00:05.591 DEBUG [13284] [CSteppingAction::UserSteppingAction@199]     (step) event_id 33 track_id 2902 track_step_count 2 step_id 1 trackStatus fStopAndKill
    2016-06-28 21:00:05.591 DEBUG [13284] [CSteppingAction::UserSteppingAction@162] CSA (trak) event_id 33 track_id 2901 parent_id 0 event_track_count 7099 pdg_encoding 0 optical 1 particle_name opticalphoton steprec_store_count 0
    2016-06-28 21:00:05.591 DEBUG [13284] [CSteppingAction::UserSteppingActionOptical@243]     (opti) photon_id 2900 step_id 0 record_id 332900 record_max 500000
    2016-06-28 21:00:05.591 DEBUG [13284] [CSteppingAction::UserSteppingAction@199]     (step) event_id 33 track_id 2901 track_step_count 1 step_id 0 trackStatus fStopAndKill
    WARNING - Attempt to delete the physical volume store while geometry closed !
    WARNING - Attempt to delete the logical volume store while geometry closed !
    WARNING - Attempt to delete the solid store while geometry closed !
    WARNING - Attempt to delete the region store while geometry closed !

    ntuhep@ntuhep-PC MINGW64 ~/env/optix/cfg4



Plumbing Classes
-------------------

CG4 
     geant4 singleton (guest from cg4-)

CCfG4
     high level control, app umbrella, bringing together Opticks and G4 
     constituents include: CTestDetector, Recorder and Rec

CPrimaryGeneratorAction
     G4VUserPrimaryGeneratorAction subclass, "shell" class just 
     providing GeneratePrimaries(G4Event*)
     which passes through to CSource::GeneratePrimaryVertex

CSteppingAction
     G4UserSteppingAction subclass, which feeds G4Step to the recorders
     in method CSteppingAction::UserSteppingAction(G4Step*)

ActionInitialization
     G4VUserActionInitialization subclass, providing UserAction plumbing 
     for CPrimaryGeneratorAction and CSteppingAction

PhysicsList
     G4VModularPhysicsList subclass, follow chroma : registered just 

     G4OpticalPhysics() 
     G4EmPenelopePhysics(0) 


Domain Classes
---------------

CSource
     G4VPrimaryGenerator subclass, with GeneratePrimaryVertex(G4Event *evt)
     Provides TorchStepNPY configurable optical photon squadrons just like the GPU eqivalent.
     Implemented using distribution generators from SingleParticleSource: 

     G4SPSPosDistribution
     G4SPSAngDistribution
     G4SPSEneDistribution

CTestDetector
     G4VUserDetectorConstruction subclass, converts simple test geometries
     commandline configured using GGeoTestConfig into G4 geometries
     with help from constituents: CMaker and CPropLib

CGDMLDetector
     G4VUserDetectorConstruction subclass, loads GDML persisted G4 geometries
     with help from constituent: CPropLib

CMaker
     Constitent of CTestDetector used to convert GCSG geometry 
     into G4 geometry in G4VPhysicalVolume* CTestDetector::Construct() 

CPropLib  
     CPropLib is a constituent of CTestDetector that converts
     GGeo materials and surfaces into G4 materials and surfaces

OpStatus
     G4 status/enum code formatters and translation of G4 codes to Opticks flags 

Recorder
     Collects G4Step/G4StepPoint optical photon data  
     into NPY arrays in Opticks array format
     which are persisted to .npy  within a OpticksEvent

     *RecordStep* is called for all G4Step
     each of which is comprised of *pre* and *post* G4StepPoint, 
     as a result the same G4StepPoint are "seen" twice, 
     thus *RecordStep* only records the 1st of the pair 
     (the 2nd will come around as the first at the next call)
     except for the last G4Step pair where both points are recorded

Rec 
     Alternative implementation of Recorder using a vector of State instances

State 
     holds copy of G4Step together with G4OpBoundaryProcessStatus, 
     a vector of State instances is held by Rec



Other Classes
---------------

CTraverser
     G4 geometry tree traverser, used for debugging material properties

Format
     G4 object string formatters for debugging 

SteppingVerbose
     Not currently used





[ABANDONED AFTER 1HR] 1st approach : try to follow Chroma g4py use of Geant4 
-------------------------------------------------------------------------------

* /usr/local/env/chroma_env/src/chroma/chroma/generator
* ~/env/chroma/chroma_geant4_integration.rst

Too complicated an environment to work with (python/numpy/pyublas/g4py/g4/chroma/..)  
for little gain over my bog standard G4 C++ examples approach in cfg4-
with NPY persisting to for python analysis


[PURSUING] 2nd approach : C++ following Geant4 examples 
--------------------------------------------------------

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
   opticks-
}



cfg4-bin(){ echo ${CFG4_BINARY:-$(cfg4-idir)/bin/$(cfg4-name)Test} ; }
cfg4-tbin(){ echo $(cfg4-idir)/bin/$1 ; }

cfg4-idir(){ echo $(local-base)/env/optix/cfg4; } 
cfg4-bdir(){ echo $(opticks-bdir)/$(cfg4-rel) ; }
cfg4-sdir(){ echo $(env-home)/optix/cfg4 ; }
cfg4-tdir(){ echo $(env-home)/optix/cfg4/tests ; }

cfg4-icd(){  cd $(cfg4-idir); }
cfg4-bcd(){  cd $(cfg4-bdir); }
cfg4-scd(){  cd $(cfg4-sdir); }
cfg4-tcd(){  cd $(cfg4-tdir); }

cfg4-dir(){  echo $(cfg4-sdir) ; }
cfg4-cd(){   cd $(cfg4-dir); }


cfg4-name(){ echo cfg4 ; }
cfg4-tag(){  echo CFG4 ; }

cfg4-wipe(){    local bdir=$(cfg4-bdir) ; rm -rf $bdir ; } 

cfg4--(){       opticks-- $(cfg4-bdir) ; } 
cfg4-ctest(){   opticks-ctest $(cfg4-bdir) $* ; } 
cfg4-genproj(){ cfg4-scd ; opticks-genproj $(cfg4-name) $(cfg4-tag) ; } 
cfg4-gentest(){ cfg4-tcd ; opticks-gentest ${1:-CExample} $(cfg4-tag) ; } 
cfg4-txt(){     vi $(cfg4-sdir)/CMakeLists.txt $(cfg4-tdir)/CMakeLists.txt ; } 




############### old funcs predating SUPERBUILD approach  #################


cfg4-cmake-standalone(){
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


cfg4-dpib()
{
   local msg="=== $FUNCNAME "

   export-

   local base=$(export-base dpib)
   local path=$base.dae
   [ -f "$path" ] && echo $msg path $path exists already : delete and rerun to recreate && return 

   ggv-;ggv-pmt-test --cdetector --export --exportconfig $path
}


