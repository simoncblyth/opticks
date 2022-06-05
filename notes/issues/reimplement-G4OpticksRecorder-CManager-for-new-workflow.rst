reimplement-G4OpticksRecorder-CManager-for-new-workflow
==========================================================

Overview
---------

Fundamentally Geant4 recording is "mocking" the recording done by Opticks 
CSGOptiX7.cu:simulate and using the same event array handling machinery 
to facilitate direct comparison. 

The main complication is handling reemission bookkeeping
to make it directly comparable.  

Task
-----

* machinery for Opticks vs Geant4 comparison
* Geant4 CFG4/CRecorder needs reworking to to write SEvt
* python analysis comparison machinery will need update for new arrays and NPFold layout

Preliminaries : DONE
---------------------

* moved event array saving from QEvent down to SEvt/NPFold, 
  so that the new U4Recorder can depend only on sysrap, G4 
  (as it makes no sense for U4 to depend on qudarap/QEvent)

How to proceed
-----------------

1. DONE : setup test environment in which to develop and exercise the U4Recorder in standalone way 
   for fast development cycle on laptop.  

   * DONE : created u4/tests/U4RecorderTest that combines stuff from ckm:CerenkovMinimal and sgs:ScintGenStandalone 
     to run Geant4 generating run/evt/track/step objects to exercise the recorder with 

2. review CManager/CRecorder/... and document them, see if can find simpler way for U4 

   * can skip non-optical CStepRec recording 
   * central CCtx record of Geant4 state updated by the Geant4 messages is convenient
   * better to keep saving very separate, handled by SEvt/NPFold 
   * also better to keep random engine control separate 
   * genstep signally should use the new SEvt::AddGenstep machinery 
   * TODO: SEvt::SetIndex(int idx) for 1,2,3,-1,-2,-3 event tags for saving 

   * TODO: contrast with qsim.h to see what can be shared : sphoton.h srec.h    
   * recall that U4 genstep collection holds state in translation-unit-local static variables

   * NB: for now are not collecting Cerenkov gensteps 
  
 
U4.cc::

    110 static spho ancestor = {} ;     // updated by U4::GenPhotonAncestor prior to the photon generation loop(s)
    111 static sgs gs = {} ;            // updated by eg U4::CollectGenstep_DsG4Scintillation_r4695 prior to each photon generation loop 
    112 static spho pho = {} ;          // updated by U4::GenPhotonBegin at start of photon generation loop
    113 static spho secondary = {} ;    // updated by U4::GenPhotonEnd   at end of photon generation loop 

cfg4/CCtx.hh::

    133     // CCtx::setTrackOptical
    134     CPhotonInfo* _cpui ;  
    135     CPho     _pho ;
    136     CGenstep _gs ; 


* standard U4 genstep bookkeping is duplicating some of what was done by recording,  
  specifically the pairs CPho/spho and CGenstep/sgs are doing the same thing  

* DONE: SEvt::AddPho collects into SEvt::pho SEvt::pho0 vectors of sphoton 

  * pho0: simple push_back
  * pho: using resize at each genstep and slotting in using spho.id 

* TODO: check that works with both C+S gensteps and when reemission enabled 
  
  * with reemission, the appropriate thing to do is replace the spho with the reemission one in same slot    

* TODO: exapnd to collecting records and rec 
 

3. check that can follow reemission lineage, reusing the functionality 



U4RecorderTest with S+C being collected and reemission disabled
------------------------------------------------------------------

* FIXED : assert due to cerenkov photon not being labelled because 
  the ancestor was not being set for cerenkov by a call to U4::GenPhotonAncestor
  prior to the generation loop(s)

::

    237 /**
    238 U4::GenPhotonAncestor
    239 ----------------------
    240 
    241 NB calling this prior to generation loops to get the ancestor 
    242 is needed for BOTH Scintillation and Cerenkov in order for photon G4Track 
    243 labelling done by U4::GenPhotonEnd to work. 
    244 
    245 **/
    246 
    247 void U4::GenPhotonAncestor( const G4Track* aTrack )
    248 {
    249     ancestor = U4PhotonInfo::Get(aTrack) ;
    250     if(dump) std::cout << "U4::GenPhotonAncestor " << ancestor.desc() << std::endl ;
    251 }

::

    271 void U4::GenPhotonEnd( int genloop_idx, G4Track* aSecondaryTrack )
    272 {
    273     assert(genloop_idx > -1);
    274     secondary = gs.MakePho(genloop_idx, ancestor) ;
    275 
    276     assert( secondary.isIdentical(pho) );
    277 
    278 #ifdef DEBUG
    279     if(dump) std::cout << "U4::GenPhotonEnd " << secondary.desc() << std::endl ;
    280 #endif
    281 
    282     U4PhotonInfo::Set(aSecondaryTrack, secondary );
    283 }





Test Environmnent : u4/tests/U4RecorderTest 
------------------------------------------------

Depends on: 

1. sysrap/SEvt+NPFold+NP array holding and persisting 
2. u4/U4Recorder Geant4 object collecting 
3. DONE : migrated some U4VolumeMaker from X4
4. quite a few classes from CFG4 need to be migrated to U4 to do the Opticks mocking 


New Approach : Whats different
-------------------------------

0. much stricter dependency control, shallow dependency tree : avoiding Tower of Babel effect 
1. new U4 + SEvt approach to genstep collection
2. SEvt/NPfold array holding and persisting 
3. NP (not NPY) arrays : NP does not yet have an extend method 
4. replace Opticks instance for config with SGeoConfig SEventConfig and others if needed
5. populate exact same structs used by qsim.h : sphoton.h srec.h (rather than writing directly into arrays)
6. would be good to follow qsim.h but Opticks and Geant4 models are so different that is probably not realistic 

   * true at high level, but at low level can reuse exactly the same struct methods that qsim.h uses
   * nevetherless the point is to match qsim.h so have to keep in firmly in mind


Old Approach : how G4OpticksRecorder was hooked up to Geant4 within JUNO framework
------------------------------------------------------------------------------------

Using optional G4OpticksAnaMgr within JUNO code.::

    epsilon:g4ok blyth$ jcv G4OpticksAnaMgr
    2 files to edit
    ./Simulation/DetSimV2/AnalysisCode/include/G4OpticksAnaMgr.hh
    ./Simulation/DetSimV2/AnalysisCode/src/G4OpticksAnaMgr.cc
    epsilon:offline blyth$ 

Looks straightforward for that class to be adapted to work with U4Recorder. 


Old Approach : using Opticks, OpticksEvent
----------------------------------------------


g4ok/G4OpticksRecorder 
    G4 object collector passing thru to cfg4/CManager

cfg4/CManager
    G4 lifecycle API : G4Run, G4Event, G4Track, G4Step

    * invoke methods of CCtx to keep an updated geant4 state

cfg4/CCtx
    * aspects close to Geant4 can be migrated verbatim 
    
cfg4/CRecorder
    * RE-joining : reemission photon history 

cfg4/CWriter
    * CWriter::expand invoked by CWriter::BeginOfGenstep extends the NPY arrays by gs_photons
    * HMM: maybe can do this more simply with std::vector push_back, or concatenating sub NP arrays
      for the photons, records from each genstep 


