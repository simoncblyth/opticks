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


Switching on reemission triggers the assert again with pho0, not with pho
----------------------------------------------------------------------------

* YEP: the assert is expected with pho0 (push_back labels) when reemission is enabled, 
  as without the reemission "re-join" bookkeeping it looks like have more photons than genstep slots.

* with the slotted-in persisting the constraints are expected to be followed, they are currently


::

    AssertionError                            Traceback (most recent call last)
    ~/opticks/u4/tests/U4RecorderTest.py in <module>
         50 
         51      # pho: labels are collected within U4Recorder::PreUserTrackingAction
    ---> 52      check_pho_labels(t.pho0)
         53      check_pho_labels(t.pho)
         54 

    ~/opticks/u4/tests/U4RecorderTest.py in check_pho_labels(l)
         31 
         32      id_u, id_c = np.unique( id_, return_counts=True  )
    ---> 33      assert np.all( id_c == 1 )
         34      # expecting the photon identity index to be unique within event, so these should all be 1
         35      # if not then that points to problem with offsetting ?

     




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
    * lifecycle 
    * RE-joining : reemission photon history 
    * CRecorder::Record

cfg4/CWriter
    * CWriter::expand invoked by CWriter::BeginOfGenstep extends the NPY arrays by gs_photons
    * HMM: maybe can do this more simply with std::vector push_back, or concatenating sub NP arrays
      for the photons, records from each genstep 


cfg4/CRec
    holds the CCtx (like many others)

    * CRec::add collects CStp


cfg4/CStp
    takes a copy of G4Step


General re-implementation approach
-------------------------------------

* arrays -> resized at genstep vectors of structs : sphoton, srec, sseq
* spho::id mimicking CUDA photon index 


Flag setting is involved in the old way
-------------------------------------------

* CRecorder::postTrackWriteSteps looks ahead to status of next step... so have to collect steps ?

* maybe treat BULK_REEMIT like CERENKOV and SCINTILLATION generation flags for step zero 
  then can avoid the stage argument 

* Q: where does initial flag get recorded ?


::

    345 unsigned int OpStatus::OpPointFlag(const G4StepPoint* point, const G4OpBoundaryProcessStatus bst, CStage::CStage_t stage)
    346 #endif
    347 {
    348     G4StepStatus status = point->GetStepStatus()  ;
    349     // TODO: cache the relevant process objects, so can just compare pointers ?
    350     const G4VProcess* process = point->GetProcessDefinedStep() ;
    351     const G4String& processName = process ? process->GetProcessName() : "NoProc" ;
    352 
    353     bool transportation = strcmp(processName,"Transportation") == 0 ;
    354     bool scatter = strcmp(processName, "OpRayleigh") == 0 ;
    355     bool absorption = strcmp(processName, "OpAbsorption") == 0 ;
    356 
    357     unsigned flag(0);
    358 
    359     // hmm stage and REJOINing look kinda odd here, do elsewhere ?
    360     // moving it first, breaks seqhis matching for multi-RE lines 
    361 
    362     if(absorption && status == fPostStepDoItProc )
    363     {
    364         flag = BULK_ABSORB ;
    365     }
    366     else if(scatter && status == fPostStepDoItProc )
    367     {
    368         flag = BULK_SCATTER ;
    369     }
    370     else if( stage == CStage::REJOIN )
    371     {
    372         flag = BULK_REEMIT ;
    373     }
    374     else if(transportation && status == fGeomBoundary )
    375     {



Q: Where does initial genflag come from ?
-------------------------------------------

::

    epsilon:sysrap blyth$ opticks-f GentypeToPhotonFlag
    ./cfg4/CGenstep.cc:    return OpticksGenstep_::GentypeToPhotonFlag(gentype); 
    ./cfg4/CCtx.cc:    return OpticksGenstep_::GentypeToPhotonFlag(_gentype); 
    ./sysrap/OpticksGenstep.h:    static unsigned GentypeToPhotonFlag(char gentype); // 'C' 'S' 'T' -> CK, SI, TO
    ./sysrap/OpticksGenstep.h:inline unsigned OpticksGenstep_::GentypeToPhotonFlag(char gentype)  // static

::

    337 void CCtx::setGentype(char gentype)
    338 {
    339     _gentype = gentype ;
    340 }
    341 
    342 unsigned CCtx::getGenflag() const
    343 {
    344     return OpticksGenstep_::GentypeToPhotonFlag(_gentype);
    345 }
    346 

    epsilon:opticks blyth$ opticks-f getGenflag 
    ./cfg4/CGenstep.cc:unsigned CGenstep::getGenflag() const
    ./cfg4/CCtx.cc:unsigned CCtx::getGenflag() const
    ./cfg4/CRecorder.cc:        unsigned preFlag = first ? m_ctx._gs.getGenflag() : OpStatus::OpPointFlag(pre,  prior_boundary_status, stage) ;
    ./cfg4/CCtx.hh:    unsigned  getGenflag() const ;
    ./cfg4/CGenstep.hh:    unsigned getGenflag() const ;  // SI CK TO from gentype 'C' 'S' 'T'
    ./cfg4/CRec.cc:                                                 m_ctx._gs.getGenflag()
    epsilon:opticks blyth$ 

::

    479 void CRecorder::postTrackWriteSteps()
    480 {
    ...
    632 
    633         unsigned preFlag = first ? m_ctx._gs.getGenflag() : OpStatus::OpPointFlag(pre,  prior_boundary_status, stage) ;
    634 


::

    np.unique(t.p.view(np.uint32)[:,3,3] , return_counts=True )  


Hmm all flags are scintillation when running with both S+C::

    In [1]: np.unique(t.p.view(np.uint32)[:,3,3] , return_counts=True )                                                                                                                                     
    Out[1]: (array([2], dtype=uint32), array([23548]))



Looks like the C current_gs gets stomped on by S::

    2022-06-06 18:34:25.598 INFO  [16587114] [U4Recorder::BeginOfRunAction@31] 
    2022-06-06 18:34:25.598 INFO  [16587114] [U4Recorder::BeginOfEventAction@39] 
    2022-06-06 18:34:25.598 INFO  [16587114] [SEvt::addGenstep@99]  s.desc sgs: idx   0 pho    62 off      0 typ G4Cerenkov_modified
    2022-06-06 18:34:25.599 INFO  [16587114] [SEvt::addGenstep@99]  s.desc sgs: idx   1 pho     1 off     62 typ DsG4Scintillation_r4695
    2022-06-06 18:34:25.599 INFO  [16587114] [SEvt::addGenstep@99]  s.desc sgs: idx   2 pho     1 off     63 typ DsG4Scintillation_r4695
    2022-06-06 18:34:25.599 INFO  [16587114] [SEvt::addGenstep@99]  s.desc sgs: idx   3 pho     1 off     64 typ DsG4Scintillation_r4695
    2022-06-06 18:34:25.599 INFO  [16587114] [SEvt::addGenstep@99]  s.desc sgs: idx   4 pho     1 off     65 typ DsG4Scintillation_r4695
    2022-06-06 18:34:25.599 INFO  [16587114] [SEvt::beginPhoton@143]  gentype 5 current_gs sgs: idx   4 pho     1 off     65 typ DsG4Scintillation_r4695
    2022-06-06 18:34:25.599 INFO  [16587114] [SEvt::beginPhoton@143]  gentype 5 current_gs sgs: idx   4 pho     1 off     65 typ DsG4Scintillation_r4695
    2022-06-06 18:34:25.599 INFO  [16587114] [SEvt::beginPhoton@143]  gentype 5 current_gs sgs: idx   4 pho     1 off     65 typ DsG4Scintillation_r4695
    2022-06-06 18:34:25.599 INFO  [16587114] [SEvt::beginPhoton@143]  gentype 5 current_gs sgs: idx   4 pho     1 off     65 typ DsG4Scintillation_r4695
    2022-06-06 18:34:25.599 INFO  [16587114] [SEvt::beginPhoton@143]  gentype 5 current_gs sgs: idx   4 pho     1 off     65 typ DsG4Scintillation_r4695
    2022-06-06 18:34:25.599 INFO  [16587114] [SEvt::beginPhoton@143]  gentype 5 current_gs sgs: idx   4 pho     1 off     65 typ DsG4Scintillation_r4695
    2022-06-06 18:34:25.599 INFO  [16587114] [SEvt::beginPhoton@143]  gentype 5 current_gs sgs: idx   4 pho     1 off     65 typ DsG4Scintillation_r4695
    2022-06-06 18:34:25.599 INFO  [16587114] [SEvt::beginPhoton@143]  gentype 5 current_gs sgs: idx   4 pho     1 off     65 typ DsG4Scintillation_r4695
    2022-06-06 18:34:25.600 INFO  [16587114] [SEvt::beginPhoton@143]  gentype 5 current_gs sgs: idx   4 pho     1 off     65 typ DsG4Scintillation_r4695
    2022-06-06 18:34:25.600 INFO  [16587114] [SEvt::beginPhoton@143]  gentype 5 current_gs sgs: idx   4 pho     1 off     65 typ DsG4Scintillation_r4695

Seems cannot rely on current_gs, so instead use spho::gs index to access the genstep corresponding to the photon. 


HMM: how to scrub BULK_ABSORB and replace with BULK_REEMIT ?
----------------------------------------------------------------

::

    epsilon:cfg4 blyth$ grep BULK_ABSORB *.*
    CPhoton.cc:    if(flag == BULK_REEMIT) scrub_mskhis(BULK_ABSORB)  ;
    CPhoton.cc:    if(flag == BULK_REEMIT) scrub_mskhis(BULK_ABSORB)  ;
    CPhoton.cc:so need to scrub the AB (BULK_ABSORB) when a RE (BULK_REEMIT) from rejoining
    CPhoton.cc:    bool flag_done = ( _flag & (BULK_ABSORB | SURFACE_ABSORB | SURFACE_DETECT | MISS)) != 0 ;
    CPhoton.cc:        if(_state._topslot_rewrite == 1 && _flag == BULK_REEMIT && _flag_prior  == BULK_ABSORB)
    CRecorder.cc:        bool lastPost = (postFlag & (BULK_ABSORB | SURFACE_ABSORB | SURFACE_DETECT | MISS )) != 0 ;
    CRecorder.cc:             m_state.decrementSlot();   // this allows REJOIN changing of a slot flag from BULK_ABSORB to BULK_REEMIT 
    CRecorderLive.cc:        decrementSlot();    // this allows REJOIN changing of a slot flag from BULK_ABSORB to BULK_REEMIT 
    CRecorderLive.cc:    bool lastPost = (postFlag & (BULK_ABSORB | SURFACE_ABSORB | SURFACE_DETECT | MISS)) != 0 ;
    CWriter.cc:    if( flag == BULK_ABSORB )
    CWriter.cc:   a some photons that previously ended with an "AB" BULK_ABSORB to ones with 
    OpStatus.cc:    return (flag & (BULK_ABSORB | SURFACE_ABSORB | SURFACE_DETECT | MISS )) != 0 ;
    OpStatus.cc:        flag = BULK_ABSORB ;
    epsilon:cfg4 blyth$ 


::

    175 void SEvt::continuePhoton(const spho& sp)
    176 {   
    177     unsigned id = sp.id ; 
    178     assert( id < pho.size() );
    ...
    200     // HMM: could directly change photon[id] via ref ? 
    201     // But are here taking a copy to current_photon, and relying on copyback at SEvt::endPhoton
    202     current_photon = photon[id] ; 
    203     current_photon.flagmask &= ~BULK_ABSORB  ; // scrub BULK_ABSORB from flagmask
    204     current_photon.set_flag(BULK_REEMIT) ;     // gets OR-ed into flagmask 
    205 }





Thinking about step point recording U4Recorder/SEvt needs the event config limits
-----------------------------------------------------------------------------------

* this are currently held in qevent.h, BUT there is not need for that 
  to be in QUDARap 

* so to avoid duplication need to migrate QUDARap/qevent.h down to sysrap/sevent.h ?

* also the compressed record domains are common to Opticks and U4Recorder/Geant4 running  
  and those are imp in qevent.h : which is another reason to  migrate it down to sysrap


::

    epsilon:qudarap blyth$ opticks-f qevent.h 
    ./ana/feature.py:        qudarap/qevent.h::
    ./CSGOptiX/CSGOptiX6.cu:#include "qevent.h"
    ./CSGOptiX/CSGOptiX7.cu:#include "qevent.h"
    ./CSGOptiX/CSGOptiX7.cu:* CPU side params including qsim.h qevent.h pointers instanciated in CSGOptiX::CSGOptiX 
    ./CSGOptiX/CSGOptiX.cc:HMM: get d_sim (qsim.h) now holds d_evt (qevent.h) but this is getting evt again rom QEvent ?
    ./sysrap/srec.h:domains, see qevent.h 
    ./qudarap/CMakeLists.txt:    qevent.h
    ./qudarap/QEvent.cu:#include "qevent.h"
    ./qudarap/tests/qevent_test.cc:#include "qevent.h"
    ./qudarap/QSim.cu:#include "qevent.h"
    ./qudarap/QU.cc:#include "qevent.h"
    ./qudarap/QEvent.hh:    // should reside inside the qevent.h instance not up here in QEvent.hh
    ./qudarap/qsim.h:#include "qevent.h"
    ./qudarap/QEvent.cc:#include "qevent.h"
    epsilon:opticks blyth$ 



How to support torch gensteps and input photons with U4Recorder ?
--------------------------------------------------------------------

* :doc:`torch-gensteps-with-new-workflow-U4Recorder`




