
g4ok-planning-for-ease-of-use-examples
==========================================

DONE : Ease of configuration 
-------------------------------

* CMake target export/import makes it trivial
  to build against any Opticks sub-project with dependencies
  automatically configured 

  * Opticks flags follow Geant4 : so should be no conflicts 


Example Context to work with
-------------------------------

g4x-
   building G4 examples like LXe 


OKOP.OpMgr ? Embedded Opticks as used by JUNO G4OpticksAnaMgr
----------------------------------------------------------------

Detector simulation framework plumbing can be brought in 
via minimal wrapper classes that hold the generatic Gean4-Opticks manager class.  


G4OpticksAnaMgr : JUNO specific example that directly holds OpMgr without intended G4OpMgr intermediary 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://juno.ihep.ac.cn/trac/browser/offline/trunk/Simulation/DetSimV2/G4Opticks/src/G4OpticksAnaMgr.cc

Note that OKOP has no G4 dependency, G4 dependency only in the wrapper class that holds on to OKOP.


G4OK.G4OKMgr : planned experiment agnostic Opticks embedded inside G4 steering class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Decide not to live in OKG4(OK,CFG4) 

* OKG4 is based on "Opticks-containing-Geant4" in the form of CG4 
  but G4OKMgr requires the converse "Geant4-containing-Opticks" 

  * once the G4 geometry is available and converted/cached into OpticksHub, 
    the containment order ceases to be relevant. It is however relevant to the 
    for high level steering public API thinking. 
    Internals can use whatever is required from the existing classes of CFG4 

* also OKG4 brings in a boatload of deps for visualization

::

    epsilon:issues blyth$ opticks-deps
    INFO:__main__:root /Users/blyth/opticks-cmake-overhaul 
     10               okconf :               OKConf : BCM OptiX G4  
     20               sysrap :               SysRap : BCM PLog  
     30             boostrap :             BoostRap : BCM Boost PLog SysRap  
     40                  npy :                  NPY : BCM PLog GLM OpenMesh BoostRap YoctoGL ImplicitMesher DualContouringSample  
     50          optickscore :          OpticksCore : BCM OKConf NPY  
     60                 ggeo :                 GGeo : BCM OpticksCore  
     70            assimprap :            AssimpRap : BCM OpticksAssimp GGeo  
     80          openmeshrap :          OpenMeshRap : BCM GGeo OpticksCore  
     90           opticksgeo :           OpticksGeo : BCM OpticksCore AssimpRap OpenMeshRap  
    100              cudarap :              CUDARap : BCM SysRap OpticksCUDA  
    110            thrustrap :            ThrustRap : BCM OpticksCore CUDARap  
    120             optixrap :             OptiXRap : BCM OptiX OpticksGeo ThrustRap  
    130                 okop :                 OKOP : BCM OptiXRap  
    140               oglrap :               OGLRap : BCM ImGui OpticksGLEW OpticksGLFW OpticksGeo  
    150            opticksgl :            OpticksGL : BCM OGLRap OKOP  
    160                   ok :                   OK : BCM OpticksGL  
    170                 cfg4 :                 CFG4 : BCM G4 OpticksXercesC OpticksGeo  
    180                 okg4 :                 OKG4 : BCM OK CFG4  
    epsilon:issues blyth$ 


G4OK new package dedicated to Opticks embedded inside Geant 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* very little code should be needed, just high level steering : functionality 
  should be inplemented in dependees for clarity of the steering

* public dependencies: G4
* private dependencies: 

  * OKOP (Opticks Compute Only) **NB no G4 dependency**
  * G4DAE
  * CFG4? lower level classes maybe : perhaps split out low level G4 wrapper
    classes into separate C4 package  
  

G4OKMgr requirements
~~~~~~~~~~~~~~~~~~~~~~~

* (at BeginOfRun) convert geometry world volume pointer into a string digest, 
  for checking with OpMgr if the geometry has already been converted/cached  


Pulling the geometry out of that context into G4DAE COLLADA, Opticks CSG
---------------------------------------------------------------------------

g4d-
   G4DAE builder

   NEXT:

   * export CMake targets from G4DAE so can easily configure against it 
   * bring g4dae back as an Opticks external

   * make Opticks example of exporting a geometry using capabilities of the
     G4DAE and G4 externals 

     * think about automation of this, via checking cache location 
       to see if was done already at each initialization 

     * also the analytic conversion process 



What about cfg4/CDetector ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There is this::

    312 void CDetector::export_dae(const char* dir, const char* name)
    313 {
    314     std::string path_ = BFile::FormPath(dir, name);
    315 
    316     const G4String path = path_ ;
    317     LOG(info) << "export to " << path_ ;
    318 
    319     G4VPhysicalVolume* world_pv = getTop();
    320     assert( world_pv  );
    321 
    322 #ifdef WITH_G4DAE 
    323     G4DAEParser* g4dae = new G4DAEParser ;
    324 
    325     G4bool refs = true ;
    326     G4bool recreatePoly = false ;
    327     G4int nodeIndex = -1 ;   // so World is volume 0 
    328 
    329     g4dae->Write(path, world_pv, refs, recreatePoly, nodeIndex );
    330 #else
    331     LOG(warning) << " export requires WITH_G4DAE " ;
    332 #endif
    333 }


BUT CDetector looks to be very embedded inside Opticks, need a more external approach
(eventually to become a pure G4 operation) for the first geometry export 



High Level Steering : to accelerate minimally evasively  
-----------------------------------------------------------

* how to minimise detector specifics ? so can reuse most of the steering ?

  * approach : just try an do it any old how, and then rejig 


 


