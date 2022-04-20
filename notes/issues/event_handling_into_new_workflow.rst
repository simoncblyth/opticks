event_handling_into_new_workflow
====================================

Summary
---------

Old way has an excess of middle-management (eg OKOP) and 
gets embroiled into OptiX and even OpenGL dependency for interop running.  


What does event handling plug into with new workflow
--------------------------------------------------------

::

    CSG/CSGFoundry 
        geometry 

    QUDARap/QSim,QEvent

    CSGOptiX/CSGOptiX
        render, simtrace, simulate 



TODO : cxsim.sh 
-------------------

* arrange raindrop simulation to give rainbow histories, so have some physical histories 
  for developing the compressed record

  * need container box with Rock//perfectAbsorbSurface/Air using new QBnd::Add capabilities

    * GeoChain geometry needs a Geant4 volume as starting point : 
    * needs some improvements to X4VolumeMaker ?
    * DONE : pulled eg P4Volume::Transverse out of j/PMTSim : 
    * this code needs to be relocated as j/PMTSim not standardly available ? 
    * seems like need a G4 utilities package 

    * HMM: getting involved just to make a simple box and sphere : 
      suggests do it at CSG level instead with CSGMaker::makeLayered


  * will need to bring over some of the old TORCH genstep generation as
    need more stats than with the simple duplicating PHOTON_CARRIER

* compressed sequence recording (seqhis seqmat) is needed for OpticksEvent 
  as the full record is only appropriate for debugging with small numbers of photons 

* once compressed record operational switch off full recording and bump up stats

* photon indexing (using thrust sorting) needs to be ported over : probably this can be done in sysrap/SU
  together with the already ported stream compaction for hit downloading

* thence will need OpticksEvent to start becoming complete

* then try switching the G4Opticks engine to use CSGOptiX either directly or via some slim middleware "Engine" 
 
* get generation going in new workflow to fully check the QEvent/qevent design  

* review old and new looking for other aspects that need to be ported over 

* OpticksEvent components and hookup to allow ab.py validation machinery to work with new workflow

  * move OpticksEvent down to sysrap : to keep simple primary dependency chain sysrap-qudarap-csgoptix



DONE: QBnd::Add QBnd::GetPerfectValues sysrap/NPDigest NP::itembytes QBnd::DescDigest 
----------------------------------------------------------------------------------------
 
* the old perfect surfaces are useful for these kind of tests ? what happened to those ? they were in GGeo ? 
* GSurfaceLib::addPerfectSurfaces
* X4PhysicalVolume::convertSurfaces calls m_slib->addPerfectSurfaces(); 
* BUT: vaguely recall that only used materials and surfaces are actually converted

  * it feels kinda dirty and cheating to do things back at GGeo level like that 

* perhaps could add extra_bnd.npy at QBnd level ? DONE THIS

  * actually starting from a simple Geant4 geometry and doing a conversion on it 
    is the cleanest way if want to compare the simple simulation with Geant4 anyhow

  * YES: but having a shortcut way to add simple boundaries also useful, as that just 
    needs some interesting NP gymastics to load the base bnd and create a compatibly shaped 5D array 
    with the added boundaries followed by NP::Concatenate

    * QBndTest is the natural place to develop this 
    * hmm would be good to be able to put together any boundary using the material and surface props 
      already present : so that needs to extract QMaterial and QSurf : DID THIS WITHOUT ANY NEW STRUCTS 
    * obvious way to test that capability is to pull apart the QBnd into QMateral and QSurf 
      and then put it back together again and verify get perfect match  

      * DID NOT DO THAT AS THE SUB-ITEM DIGESTS MAKE THE VERACITY PLAIN TO SEE WITH QBnd::DescDigest 


::

    1281 void GGeo::prepareMaterialLib()
    1282 {
    1283     LOG(verbose) ;
    1284 
    1285     GMaterialLib* mlib = getMaterialLib() ;
    1286 
    1287     mlib->addTestMaterials();
    1288 }
    1289 
    1290 void GGeo::prepareSurfaceLib()
    1291 {
    1292     LOG(verbose) ;
    1293 
    1294     GSurfaceLib* slib = getSurfaceLib() ;
    1295 
    1296     slib->addPerfectSurfaces();
    1297 }



  * probably the perfect surfaces were skipped from the conversion due to not being used in the standard geometry
  * HMM: shortcut : artificially make a MISS result in surface absorb 
  * then can develop the compressed history recording so can switch off the expensive full step record
    and then do some high stats testing with raindrop geometry 
 


DONE: in CSGOptiX/cxsim.sh check new CSGOptiXSimulateTest with OpticksGenstep_PHOTON_CARRIER and a simple geometry
----------------------------------------------------------------------------------------------------------------------

:: 

    QSim<float>::UploadComponents(fd->icdf, fd->bnd, fd->optical, rindexpath );

* simpler to reuse standard CSGFoundry components together with simple non-standard geometry for the test
* arrange for CSGOptiXSimulateTest to combine:

1. "basis" standard CSGFoundry components (eg bnd, bndname etc) 
2. simple GeoChain geometry from another CFBase with boundaries configured with CSGFoundary::setPrimBoundary 


DONE : boundary mechanics in CSGFoundry
-----------------------------------------

* DONE : need way get the boundary index by a string spec lookup 
* DONE : also need API to set the boundary onto the CSGNode tree prior to upload 
* DONE : CSGFoundry::setPrimBoundary as need to iterate over all CSGNode of the CSGPrim 
 
``OptiX7Test.cu:__intersection__is`` gets boundary from CSGNode::

    489     float4 isect ; // .xyz normal .w distance 
    490     if(intersect_prim(isect, node, plan, itra, t_min , ray_origin, ray_direction ))
    491     {
    492         const unsigned hitKind = 0u ;            // only 8bit : could use to customize how attributes interpreted
    493         const unsigned boundary = node->boundary() ;  // all nodes of tree have same boundary 
    494 #ifdef WITH_PRD
    495         if(optixReportIntersection( isect.w, hitKind))
    496         {
    497             quad2* prd = getPRD<quad2>();
    498             prd->q0.f = isect ;
    499             prd->set_boundary(boundary) ;
    500         }

* added bndname handling to CSGFoundry analogous to meshname
* added CSGFoundry::setPrimBoundary 
* added boundary dumping CSGFoundry::detailPrim which is used from CSG/CSGPrimTest.cc 



DONE : split off cxs 2D as simtrace running
-----------------------------------------------

* cxs_geochain.sh running with simple geometry 


DONE : reviewing CSGOptiX and Six backwards compat
----------------------------------------------------

* CSGOptix currently depends on OpticksCore

  * see if can move it down to sysrap-qudarap ?
  * CONCLUDED : USE OF Composition PREVENTS THIS CURRENTLY 

* CSGOptiX with pre-7 : *Six* 

  * review *Six* and its tests : add more tests using very simple geometry if necessary 

    * DONE : added minimal CSGOptiXTest 

  * update *Six* backwards compat machinery to accomodate recent QUDARap developments 

    * CONCLUDED : EFFORT NOT WORTHY OF THE BENEFIT 
    * **END OF THE LINE FOR OptiX < 7 SIM : OTHER THAN RENDERING**

  * arrange for the two "branches" to share more code, eg 
 
    * can more use of OptiX 6/CUDA interop be made : using alt view of same CUDA buffers  
    * DONE: now using Frame with both branches 


DONE : incorporate SU stream compaction into QEvent::getHits 
----------------------------------------------------------------

* QEvent/qevent needs hit buffer handling integrating SU stream compaction SU::select_copy_device_to_host_presized
  
  * developed this at small scale using mock_propagate with mock_prd 
  * holding the selector functor in QEvent


DONE : incorporate QEvent/qevent into QSim/qsim
---------------------------------------------------

* incorporate QEvent/qevent into QSim/qsim and test utility of qevent encapsulated buffer handling with QSimTest, 
  if the design is appropriate this should significantly simplify and remove duplication of buffer handling in QSimTest 
  and become the basis for real event handling  

  * hmm many tests are photon level, with no gensteps so need to check QEvent::setNumPhotons  
  * actually the main benefit of QEvent/qevent comes when actually generating photons on device
    which requires use of QEvent::setGensteps with seeding etc.. 
  * photon level tests are sufficiently different from standard running 
    that they will not benefit much from QEvent. 
  * HMM: looking at CSGOptiX/OptiX7Test.cu:simulate the qevent and qsim instances 
    are kept separate and both come in from params 


   

Review Progress already in new workflow
------------------------------------------

qudarap/tests/QSimWithEventTest.cc 
     much more direct approach than old way revolving around QEvent/qevent 

     * this can act as nucleus for bringing over functionality

QEvent.hh/qevent.h
     moved QSeed into QEvent for clarity 

What about dependencies:

* qudarap can almost go down to depending on sysrap (not optickscore)
* would like to stay with that by moving OpticksEvent down to sysrap  


How to migrate from old to new workflow ? What level to make switch over ?
----------------------------------------------------------------------------

* SUSPECT QUICKER (AND BETTER) TO START WITH FRESH DESIGN, 
  AND GRAB PIECES FROM OLD WORKFLOW THAT CAN BE REUSED AS NEEDED

  * qudarap/tests/QSimWithEventTest.cc can act as nucleus for development 


* want to come up with something much simpler than old way 
* needs to be testable with CUDA only (no OptiX)  

* fundamentals (OpticksEvent) can be reused mostly intact, all the 
  middle management needs to be scrapped 

* OpticksEvent format can stay almost exactly the same, just with NPY replaced by NP
* G4Opticks interface can stay almost exactly the same, just with NPY replaced by NP

  * what about internals okop/OpMgr ? 

* does okop stay or go ?  clearly it must GO, its too embroiled in 
  OptiXRap and is far too middle management style to be usable 


g4ok/G4Opticks 
    top level : depending on okop/OpMgr 
         
okop/OpMgr : not doing much itself 

    * coordinates OpticksRun m_run and OpPropagator m_propagator 
    * OpticksEvent coordination
    * OpMgr::propagate uses OpticksRun m_run to create OpticksEvent from gensteps 

okop/OpPropagator : again not doing much itself      

    * holds m_engine:OpEngine m_tracer:OpTracer  
    * (CSGOptiX::render CSGOptiX::simulate are different methods of same CSGOptiX instance) 

okop/OpEngine : using OptiXRap OConfig/OContext/OEvent/OPropagator/OScene and okop OpSeeder/OpZeroer/OpIndexer

    * m_oevt:OEvent
    * m_propagator:OPropagator
    * m_seeder:OpSeeder
    * m_zeroer:OpZeroer
    * m_indexer:OpIndexer

opticksgeo/OpticksHub
   acted as intermediary on top of GGeo : given the move to new CSG geometry this has lost its reason to live      

oxrap/OEvent
    OEvent::createBuffers(OpticksEvent* evt)
        functionality clearly needed in QUDARap going from the CPU side OpticksEvent to GPU side buffers
        but the way of doing that will be very different (plain CUDA, no OptiX) 



All Packages : Thinking of their future (or not)
-------------------------------------------------

::

    epsilon:qudarap blyth$ opticks-deps
    [2022-04-09 14:45:58,096] p99829 {/Users/blyth/opticks/bin/CMakeLists.py:170} INFO - home /Users/blyth/opticks 
              API_TAG :        reldir :         bash- :     Proj.name : dep Proj.names  
     10        OKCONF :        okconf :        okconf :        OKConf : OpticksCUDA OptiX G4  
     20        SYSRAP :        sysrap :        sysrap :        SysRap : OKConf NLJSON PLog OpticksCUDA  

             GROWING BASIS

     30          BRAP :      boostrap :          brap :      BoostRap : Boost BoostAsio NLJSON PLog SysRap Threads  
     40           NPY :           npy :           npy :           NPY : PLog GLM BoostRap  
     50        OKCORE :   optickscore :           okc :   OpticksCore : NPY  
              
            LONGTERM : ELIMINATE BRAP, NPY, REPLACE boost:program_options with something else   
            SO OKCORE CAN SINK TO JUST ABOVE SYSRAP 


     60          GGEO :          ggeo :          ggeo :          GGeo : OpticksCore  
    165            X4 :         extg4 :            x4 :         ExtG4 : G4 GGeo OpticksXercesC CLHEP PMTSim  
    170          CFG4 :          cfg4 :          cfg4 :          CFG4 : G4 ExtG4 OpticksXercesC OpticksGeo ThrustRap  

            VERY LONGTERM : REPLACE GGEO WITH G4->CSG DIRECT WORKFLOW 
            THIS WILL NEED TO HANDLE THE NPY PRIM AND THE VITAL GGEO GInstancer FACTORIZATION


     90         OKGEO :    opticksgeo :           okg :    OpticksGeo : OpticksCore GGeo  
    100       CUDARAP :       cudarap :       cudarap :       CUDARap : SysRap OpticksCUDA  
    110         THRAP :     thrustrap :         thrap :     ThrustRap : OpticksCore CUDARap  
    120         OXRAP :      optixrap :         oxrap :      OptiXRap : OKConf OptiX OpticksGeo ThrustRap  
    130          OKOP :          okop :          okop :          OKOP : OptiXRap  

              SHORTTERM : ELIMINATE ALL THESE 

    140        OGLRAP :        oglrap :        oglrap :        OGLRap : ImGui OpticksGLEW BoostAsio OpticksGLFW OpticksGeo  
    150          OKGL :     opticksgl :          okgl :     OpticksGL : OGLRap OKOP  
    160            OK :            ok :            ok :            OK : OpticksGL  
    180          OKG4 :          okg4 :          okg4 :          OKG4 : OK CFG4  

              GRAPHICS RELATED DEVELOPMENT ON HOLD AS DIFFICULT TO DO INTEROP IN REMOTE WORKING MODE

    190          G4OK :          g4ok :          g4ok :          G4OK : CFG4 ExtG4 OKOP  

               SHORTTERM : SWITCH OKOP -> CSGOptiX

    200          None :   integration :   integration :   Integration :   

    300           CSG :           CSG :          None :           CSG : CUDA SysRap  
    310      CSG_GGEO :      CSG_GGeo :          None :      CSG_GGeo : CUDA CSG GGeo  
    320      GEOCHAIN :      GeoChain :          None :      GeoChain : CUDA CSG_GGeo ExtG4 PMTSim jPMTSim  
    330       QUDARAP :       qudarap :       qudarap :       QUDARap : OpticksCore OpticksCUDA  
    340      CSGOPTIX :      CSGOptiX :       resolut :      CSGOptiX : CUDA OpticksCore QUDARap CSG OpticksOptiX  
    epsilon:qudarap blyth$ 

