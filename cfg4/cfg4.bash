cfg4-src(){      echo cfg4/cfg4.bash ; }
cfg4-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(cfg4-src)} ; }
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


Xcode 9.2
------------

::

    In file included from /usr/local/opticks/externals/include/Geant4/G4Region.hh:278:
    /usr/local/opticks/externals/include/Geant4/G4Region.icc:247:3: warning: instantiation of variable 'G4GeomSplitter<G4RegionData>::offset' required here, but no definition is available
          [-Wundefined-var-template]
      G4MT_fsmanager = fsm;
      ^
    /usr/local/opticks/externals/include/Geant4/G4Region.hh:102:45: note: expanded from macro 'G4MT_fsmanager'
    #define G4MT_fsmanager ((subInstanceManager.offset[instanceID]).fFastSimulationManager)
                                                ^
    /usr/local/opticks/externals/include/Geant4/G4GeomSplitter.hh:193:40: note: forward declaration of template entity is here
        G4GEOM_DLL static G4ThreadLocal T* offset;
                                           ^
    /usr/local/opticks/externals/include/Geant4/G4Region.icc:247:3: note: add an explicit instantiation declaration to suppress this warning if 'G4GeomSplitter<G4RegionData>::offset' is explicitly
          instantiated in another translation unit
      G4MT_fsmanager = fsm;
      ^
    /usr/local/opticks/externals/include/Geant4/G4Region.hh:102:45: note: expanded from macro 'G4MT_fsmanager'
    #define G4MT_fsmanager ((subInstanceManager.offset[instanceID]).fFastSimulationManager)
                                                ^


Clean Build Link Issue
-----------------------

::

    [ 71%] Linking CXX shared library libcfg4.dylib
    Undefined symbols for architecture x86_64:
      "xercesc_2_8::DTDEntityDecl::serialize(xercesc_2_8::XSerializeEngine&)", referenced from:
          vtable for xercesc_2_8::DTDEntityDecl in CGDMLDetector.cc.o
    ...
      "typeinfo for xercesc_2_8::SAXParseException", referenced from:
          xercesc_2_8::HandlerBase::fatalError(xercesc_2_8::SAXParseException const&) in CGDMLDetector.cc.o
    ld: symbol(s) not found for architecture x86_64
    clang: error: linker command failed with exit code 1 (use -v to see invocation)
    make[2]: *** [cfg4/libcfg4.dylib] Error 1
    make[1]: *** [cfg4/CMakeFiles/cfg4.dir/all] Error 2
    make: *** [all] Error 2


::

    simon:tests blyth$ clang XercescCTest.cc -I$(xercesc-include-dir) -L/usr/local/opticks/externals/lib 
    XercescCTest.cc:11:5: error: use of undeclared identifier 'XMLPlatformUtils'; did you mean 'xercesc_3_1::XMLPlatformUtils'?
        XMLPlatformUtils::Initialize();
        ^~~~~~~~~~~~~~~~
        xercesc_3_1::XMLPlatformUtils
    /usr/local/opticks/externals/include/xercesc/util/PlatformUtils.hpp:68:22: note: 'xercesc_3_1::XMLPlatformUtils' declared here
    class XMLUTIL_EXPORT XMLPlatformUtils
                         ^
    1 error generated.




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


CG4Test fails for lack of g4 envvars. After g4-export CG4Test gets further but misses GDML, 
after update opticksdata clone get to the same error as CPropLib above.


Rearrange ggeo test to check back in ggeo. Plenty of surfaces, but no sensor surfaces.

Huh old GSurfaceLib.npy::

    simon:assimprap blyth$ ll /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GSurfaceLib/
    total 128
    drwxr-xr-x   4 blyth  staff    136 Jun  8 18:37 .
    -rw-r--r--   1 blyth  staff    816 Jun 15 19:08 GSurfaceLibOptical.npy
    -rw-r--r--   1 blyth  staff  57488 Jun 15 19:08 GSurfaceLib.npy
    drwxr-xr-x  72 blyth  staff   2448 Jun 30 16:28 ..

Wrong cache, pilot error from stale envvar::

    simon:tests blyth$ GGeoViewTest -G
    2016-06-30 17:24:58.392 INFO  [13430829] [Timer::operator@38] Opticks:: START
    2016-06-30 17:24:58.393 INFO  [13430829] [Opticks::Summary@339] App::init OpticksResource::Summary sourceCode 4096 sourceType torch mode Interop
    App::init OpticksResource::Summary
    valid    :valid
    envprefix: OPTICKS_
    geokey   : DAE_NAME_DYB
    daepath  : /usr/local/env/geant4/geometry/export/DayaBay_VGDX_20140414-1300/g4_00.dae
    gdmlpath : /usr/local/env/geant4/geometry/export/DayaBay_VGDX_20140414-1300/g4_00.gdml
    metapath : /usr/local/env/geant4/geometry/export/DayaBay_VGDX_20140414-1300/g4_00.ini
    query    : range:3153:12221
    ctrl     : volnames

CG4Test needs below envvars::

    simon:~ blyth$ env | grep G4
    simon:~ blyth$ g4-
    simon:~ blyth$ g4-export
    simon:~ blyth$ env | grep G4
    G4LEVELGAMMADATA=/usr/local/opticks/externals/share/Geant4-10.2.1/data/PhotonEvaporation3.2
    G4NEUTRONXSDATA=/usr/local/opticks/externals/share/Geant4-10.2.1/data/G4NEUTRONXS1.4
    G4LEDATA=/usr/local/opticks/externals/share/Geant4-10.2.1/data/G4EMLOW6.48
    G4NEUTRONHPDATA=/usr/local/opticks/externals/share/Geant4-10.2.1/data/G4NDL4.5
    G4ENSDFSTATEDATA=/usr/local/opticks/externals/share/Geant4-10.2.1/data/G4ENSDFSTATE1.2.1
    G4RADIOACTIVEDATA=/usr/local/opticks/externals/share/Geant4-10.2.1/data/RadioactiveDecay4.3.1
    G4ABLADATA=/usr/local/opticks/externals/share/Geant4-10.2.1/data/G4ABLA3.0
    G4PIIDATA=/usr/local/opticks/externals/share/Geant4-10.2.1/data/G4PII1.3
    G4SAIDXSDATA=/usr/local/opticks/externals/share/Geant4-10.2.1/data/G4SAIDDATA1.1
    G4REALSURFACEDATA=/usr/local/opticks/externals/share/Geant4-10.2.1/data/RealSurface1.0
    simon:~ blyth$ 





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
   olocal- 
   g4-
   opticks-
}



cfg4-idir(){ echo $(opticks-idir); } 
cfg4-bdir(){ echo $(opticks-bdir)/cfg4 ; }
cfg4-sdir(){ echo $(opticks-home)/cfg4 ; }
cfg4-tdir(){ echo $(opticks-home)/cfg4/tests ; }

cfg4-icd(){  cd $(cfg4-idir); }
cfg4-bcd(){  cd $(cfg4-bdir); }
cfg4-scd(){  cd $(cfg4-sdir); }
cfg4-tcd(){  cd $(cfg4-tdir); }

cfg4-dir(){  echo $(cfg4-sdir) ; }
cfg4-cd(){   cd $(cfg4-dir); }
cfg4-c(){    cd $(cfg4-dir); }


cfg4-name(){ echo cfg4 ; }
cfg4-tag(){  echo CFG4 ; }

cfg4-apihh(){  echo $(cfg4-sdir)/$(cfg4-tag)_API_EXPORT.hh ; }
cfg4---(){     touch $(cfg4-apihh) ; cfg4--  ; }



cfg4-wipe(){    local bdir=$(cfg4-bdir) ; rm -rf $bdir ; } 

#cfg4--(){       opticks-- $(cfg4-bdir) ; } 
#cfg4-t(){       opticks-t $(cfg4-bdir) $* ; } 

cfg4--(){       cfg4-scd ; om- ; om-make ;  } 
cfg4-t(){       cfg4-scd ; om- ; om-test ;  } 


#cfg4-ts(){      opticks-ts $(cfg4-bdir) $* ; } 
#cfg4-tl(){      opticks-tl $(cfg4-bdir) $* ; } 

cfg4-genproj(){ cfg4-scd ; opticks-genproj $(cfg4-name) $(cfg4-tag) ; } 
cfg4-gentest(){ cfg4-tcd ; opticks-gentest ${1:-CExample} $(cfg4-tag) ; } 
cfg4-txt(){     vi $(cfg4-sdir)/CMakeLists.txt $(cfg4-tdir)/CMakeLists.txt ; } 


cfg4-g4lldb-(){ echo /tmp/g4lldb.txt ; }

cfg4-g4lldb()
{
    local path=$($FUNCNAME-)
    $OPTICKS_HOME/cfg4/g4lldb.py > $path  
    cat $path

}


############### old funcs predating SUPERBUILD approach  #################


cfg4-dpib()
{
   local msg="=== $FUNCNAME "

   export-

   local base=$(export-base dpib)
   local path=$base.dae
   [ -f "$path" ] && echo $msg path $path exists already : delete and rerun to recreate && return 

   ggv-;ggv-pmt-test --cdetector --export --exportconfig $path
}





