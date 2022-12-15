Hans_QSim_segv_with_CaTS
==========================


::

    Hi Simon, 

    with your latest  commit I still have the same seg.  fault problem. Different
    things I tried (e.g. using a singlethreaded version of Geant4, doing a
    cudaDeviceReset();, switching off scintillation,....) had no effect. I also
    checked the code and executable are in sync. I  have included the backtrace and
    the run log.  overall there are only a few files that changed:

    include/PhotonSD.hh:#ifdef WITH_CXG4OPTICKS
    src/MCEventAction.cc:#ifdef WITH_CXG4OPTICKS
    src/EventAction.cc:#ifdef WITH_CXG4OPTICKS
    src/PhotonSD.cc:#ifdef WITH_CXG4OPTICKS
    src/MyG4Cerenkov.cc:#ifdef WITH_CXG4OPTICKS
    src/DetectorConstruction.cc:#ifdef WITH_CXG4OPTICKS
    src/MyG4Scintillation.cc:#ifdef WITH_CXG4OPTICKS
    CaTS.cc:#ifdef WITH_CXG4OPTICKS

    everything is in 
    https://github.com/hanswenzel/CaTS

    BTW. do I have to worry about the message: 

    NP::load Failed to load from path /home/wenzel/.opticks/geocache/CaTS_World_PV_g4live/g4ok_gltf/6a511c07e6d72b5e4d71b74bd548e8fd/1/GScintillatorLib/LS_ori/RINDEX.npy



Suggest to Check the QEvent pointer held by QSim
----------------------------------------------------

Observation:

* QEvent pointer 0x312d202a2a2a4c looks very different to other pointers like 0x7fffffffb9f0 and 0x7fffe40a3030 
  so looks like an overwrite issue somehow ? 

* Debug that by dumping the QEvent pointer at instantiation and at usage
* As I added a little debug dumping you can do that (after updating) with::

   export QSim=INFO 


Backtrace
------------

::

    0x00007ffff2d27d03 in QEvent::setGenstep (this=0x312d202a2a2a4c, gs_=0x7fffe6aabfc0) at /data3/wenzel/newopticks_dev5/opticks/qudarap/QEvent.cc:169
    169	    gs = gs_ ; 
    gdb$ bt
    #0  0x00007ffff2d27d03 in QEvent::setGenstep (this=0x312d202a2a2a4c, gs_=0x7fffe6aabfc0) at /data3/wenzel/newopticks_dev5/opticks/qudarap/QEvent.cc:169
    #1  0x00007ffff2d27c84 in QEvent::setGenstep (this=0x312d202a2a2a4c) at /data3/wenzel/newopticks_dev5/opticks/qudarap/QEvent.cc:153
    #2  0x00007ffff2cfda8f in QSim::simulate (this=0x7fffffffba70) at /data3/wenzel/newopticks_dev5/opticks/qudarap/QSim.cc:296
    #3  0x00007ffff7f677d3 in G4CXOpticks::simulate (this=0x7fffffffb9f0) at /data3/wenzel/newopticks_dev5/opticks/g4cx/G4CXOpticks.cc:354
    #4  0x00005555555db65b in MCEventAction::EndOfEventAction (this=0x7fffe40a3030, event=0x7fffe6aa8aa0) at /data3/wenzel/newopticks_dev5/CaTS/src/MCEventAction.cc:158
    #5  0x00007ffff643a809 in G4EventManager::DoProcessing(G4Event*) () from /data3/wenzel/geant4-v11.0.3-install/lib/libG4event.so
    #6  0x00007ffff7506cdc in G4WorkerTaskRunManager::ProcessOneEvent(int) () from /data3/wenzel/geant4-v11.0.3-install/lib/libG4tasking.so
    #7  0x00007ffff75069b6 in G4WorkerTaskRunManager::DoEventLoop(int, char const*, int) () from /data3/wenzel/geant4-v11.0.3-install/lib/libG4tasking.so
    #8  0x00007ffff7506b9d in G4WorkerTaskRunManager::DoWork() () from /data3/wenzel/geant4-v11.0.3-install/lib/libG4tasking.so


::

    319 void G4CXOpticks::simulate()
    320 {
    ....
    354     qs->simulate();   // GPU launch doing generation and simulation here 
    355 
    356     sev->gather();   // downloads components configured by SEventConfig::CompMask 
    ...


    0211 QSim::QSim()
     212     :
     213     base(QBase::Get()),
     214     event(new QEvent),
     215     rng(QRng::Get()),
     216     scint(QScint::Get()),
     217     cerenkov(QCerenkov::Get()),
     218     bnd(QBnd::Get()),
     219     prd(QPrd::Get()),
     220     debug_(QDebug::Get()),
     221     prop(QProp<float>::Get()),
     222     multifilm(QMultiFilm::Get()),
     223     sim(nullptr),
     224     d_sim(nullptr),
     225     dbg(debug_ ? debug_->dbg : nullptr),
     226     d_dbg(debug_ ? debug_->d_dbg : nullptr),
     227     cx(nullptr)
     228 {
     229 
     230     init();
     231 }


    0290 double QSim::simulate()
     291 {
     292    LOG_IF(error, event == nullptr) << " event null " << desc()  ;
     293    if( event == nullptr ) std::raise(SIGINT) ;
     294    if( event == nullptr ) return -1. ;
     295 
     296    int rc = event->setGenstep() ;
     297    LOG_IF(error, rc != 0) << " QEvent::setGenstep ERROR : have event but no gensteps collected : will skip cx.simulate " ;
     298    double dt = rc == 0 && cx != nullptr ? cx->simulate() : -1. ;
     299    return dt ;
     300 }


    147 int QEvent::setGenstep()  // onto device
    148 {
    149     NP* gs = SEvt::GatherGenstep(); // TODO: review memory handling  
    150     SEvt::Clear();   // clear the quad6 vector, ready to collect more genstep
    151     LOG_IF(fatal, gs == nullptr ) << "Must SEvt::AddGenstep before calling QEvent::setGenstep " ;
    152     //if(gs == nullptr) std::raise(SIGINT); 
    153     return gs == nullptr ? -1 : setGenstep(gs) ;
    154 }
    155 



Observations on CaTS 
-------------------------


Opticks sets "WITH_G4CXOPTICKS" not "WITH_CXG4OPTICKS" in cmake/Modules/FindOpticks.cmake::

    epsilon:CaTS blyth$ opticks-f WITH_G4CXOPTICKS 
    ./cmake/Modules/FindOpticks.cmake:    add_compile_definitions(WITH_G4CXOPTICKS)
    ./cmake/Modules/FindOpticks.cmake:    add_compile_definitions(WITH_G4CXOPTICKS_DEBUG)

* not necessarily an issue, but need to be aware of that 


::

    epsilon:CaTS blyth$ find . -type f -exec grep -H WITH_G4CXOPTICKS {} \;
    ./include/mySensitiveDetector.hh:  //#ifdef WITH_G4CXOPTICKS
    epsilon:CaTS blyth$ 

    epsilon:CaTS blyth$ find . -type f -exec grep -H WITH_CXG4OPTICKS {} \;
    ./CaTS.cc:#ifdef WITH_CXG4OPTICKS
    ./CaTS.cc:#ifdef WITH_CXG4OPTICKS
    ./CMakeLists.txt:if(WITH_G4OPTICKS AND WITH_CXG4OPTICKS)
    ./CMakeLists.txt:  message(FATAL_ERROR "CaTS cmake illegal set of parameters. One can't use opticks and legacy opticks at the same time. Either -DWITH_G4OPTICKS or -DWITH_CXG4OPTICKS must be set to off and can't be set on at the same time.
    ./CMakeLists.txt:-DWITH_G4OPTICKS=OFF -DWITH_CXG4OPTICKS=OFF : no opticks just Geant4 optical physics
    ./CMakeLists.txt:-DWITH_G4OPTICKS=ON  -DWITH_CXG4OPTICKS=OFF : build using legacy opticks based on NVIDIA OptiX 6
    ./CMakeLists.txt:-DWITH_G4OPTICKS=OFF -DWITH_CXG4OPTICKS=ON  : build using current opticks based on NVIDIA OptiX > 7
    ./CMakeLists.txt:option(WITH_CXG4OPTICKS "Build example with OPTICKS" OFF)
    ./CMakeLists.txt:elseif(WITH_CXG4OPTICKS)
    ./CMakeLists.txt:  message(STATUS "WITH_CXG4OPTICKS is set")
    ./CMakeLists.txt:elseif(WITH_CXG4OPTICKS)
    ./CMakeLists.txt:  target_compile_definitions(${name} PRIVATE WITH_CXG4OPTICKS WITH_ROOT)
    ./CMakeLists.txt:message(STATUS "WITH_CXG4OPTICKS: ${WITH_CXG4OPTICKS}")
    ./include/PhotonSD.hh:#ifdef WITH_CXG4OPTICKS
    ./src/PhotonSD.cc:#ifdef WITH_CXG4OPTICKS
    ./src/PhotonSD.cc:#ifdef WITH_CXG4OPTICKS
    ./src/EventAction.cc:#ifdef WITH_CXG4OPTICKS
    ./src/MyG4Scintillation.cc:#ifdef WITH_CXG4OPTICKS
    ./src/MyG4Scintillation.cc:#ifdef WITH_CXG4OPTICKS
    ./src/DetectorConstruction.cc:#ifdef WITH_CXG4OPTICKS
    ./src/DetectorConstruction.cc:#ifdef WITH_CXG4OPTICKS
    ./src/MyG4Cerenkov.cc:#ifdef WITH_CXG4OPTICKS
    ./src/MyG4Cerenkov.cc:#ifdef WITH_CXG4OPTICKS
    ./src/MCEventAction.cc:#ifdef WITH_CXG4OPTICKS
    ./src/MCEventAction.cc:#ifdef WITH_CXG4OPTICKS
    ./src/MCEventAction.cc:#ifdef WITH_CXG4OPTICKS
    ./src/MCEventAction.cc:#endif  // end   WITH_CXG4OPTICKS
    epsilon:CaTS blyth$ 



Suggestions for CaTS src/DetectorConstruction.cc
----------------------------------------------------

::

    421 void DetectorConstruction::ReadGDML()
    422 {
    423   fReader = new ColorReader;
    424   parser  = new G4GDMLParser(fReader);
    425   parser->Read(gdmlFile, false);
    426   G4VPhysicalVolume* World = parser->GetWorldVolume();
    427   //----- GDML parser makes world invisible, this is a hack to make it
    428   // visible again...
    429   G4LogicalVolume* pWorldLogical = World->GetLogicalVolume();
    430 #ifdef WITH_CXG4OPTICKS

    431   G4CXOpticks gx;  // Simulate is the default RGMode
    432   gx.setGeometry(World);
    ^^^^^^^^^^^^^^^^^^^^^^^^  THIS WILL NO LONGER COMPILE

    433   SEventConfig::SetMaxPhoton(10000000);
    ^^^^^^^^^^^^^^^^^^^^^^^^  ALL SEventConfig MUST BE DONE BEFORE SetGeometry AS IT TAKES EFFECT AT SEvt INSTANCIATION 

    434 #endif
    435   pWorldLogical->SetVisAttributes(0);
    436   if(verbose)


G4CXOpticks on the stack is not going to work, it will get cleaned up, 
before you are finished with it. Instead use::

     G4CXOpticks::SetGeometry(World) 

To avoid others making this mistake, I have made the ctor private. 
And changed the G4CXOpticks::SetGeometry static method to
return the pointer::

     61 G4CXOpticks* G4CXOpticks::SetGeometry(const G4VPhysicalVolume* world)
     62 {
     63     G4CXOpticks* g4cx = new G4CXOpticks ;
     64     g4cx->setGeometry(world);
     65     return g4cx ;
     66 }

Regarding the SEventConfig static methods such as::

    SEventConfig::SetMaxPhoton(10000000); 

you need to call those before SEvt gets instanciated : so that means 
you need to do it first (eg in the main) or you can effectively do it 
automatically by defining envvar OPTICKS_MAX_PHOTON to control this::

     SEventConfig::SetMaxPhoton(10000000);
     G4CXOpticks::SetGeometry(World) 




Suggestions for CaTS src/EventAction.cc
----------------------------------------------------

You include loadsa Opticks headers and dont use them.
Suggest you remove::

    074 #include <istream>
     75 #ifdef WITH_G4OPTICKS
     76 #  include "OpticksFlags.hh"
     77 #  include "G4Opticks.hh"
     78 #  include "G4OpticksHit.hh"
     79 #endif
     80 #ifdef WITH_CXG4OPTICKS
     81 #  include "SLOG.hh"
     82 #  include "G4Step.hh"
     83 #  include "scuda.h"
     84 #  include "sqat4.h"
     85 #  include "sframe.h"
     86 
     87 #  include "SSys.hh"
     88 #  include "SEvt.hh"
     89 #  include "SSim.hh"
     90 #  include "SGeo.hh"
     91 #  include "SOpticksResource.hh"
     92 #  include "SFrameGenstep.hh"
     93 
     94 #  include "U4VolumeMaker.hh"
     95 
     96 #  include "SEventConfig.hh"
     97 #  include "U4GDML.h"
     98 #  include "U4Tree.h"
     99 
    100 #  include "CSGFoundry.h"
    101 #  include "CSG_GGeo_Convert.h"
    102 #  include "CSGOptiX.h"
    103 #  include "QSim.hh"
    104 
    105 #  include "U4Hit.h"
    106 
    107 #  include "U4.hh"
    108 #  include "G4CXOpticks.hh"
    109 // #include "G4Opticks.hh"
    110 #endif



Suggestions for CaTS src/MCEventAction.cc
----------------------------------------------------

Again loads Opticks headers you dont use. You probably only need the two indented ones::

    72 #ifdef WITH_G4OPTICKS
     73 #  include "OpticksFlags.hh"
     74 #  include "G4Opticks.hh"
     75 #  include "G4OpticksHit.hh"
     76 #endif
     77 #ifdef WITH_CXG4OPTICKS
     78 #  include "SLOG.hh"

     
     79 #  include "G4Step.hh"
     80 #  include "scuda.h"
     81 #  include "sqat4.h"
     82 #  include "sframe.h"
     83 
     84 #  include "SSys.hh"

                                               85 #  include "SEvt.hh"


     86 #  include "SSim.hh"
     87 #  include "SGeo.hh"
     88 #  include "SOpticksResource.hh"
     89 #  include "SFrameGenstep.hh"
     90 
     91 #  include "U4VolumeMaker.hh"
     92 
     93 #  include "SEventConfig.hh"
     94 #  include "U4GDML.h"
     95 #  include "U4Tree.h"
     96 
     97 #  include "CSGFoundry.h"
     98 #  include "CSG_GGeo_Convert.h"
     99 #  include "CSGOptiX.h"
    100 #  include "QSim.hh"
    101 
    102 #  include "U4Hit.h"
    103 
    104 #  include "U4.hh"
                                              105 #  include "G4CXOpticks.hh"
    106 // #include "G4Opticks.hh"
    107 #endif

