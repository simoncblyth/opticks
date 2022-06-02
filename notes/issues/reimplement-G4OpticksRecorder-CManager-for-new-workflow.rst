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

1. setup test environment in which to develop and exercise the U4Recorder in standalone way 
   for fast development cycle on laptop.  

   * ScintStandalone (starting from sgs:ScintGenStandalone) with some simple geometry 
   * (eg big sphere of scintillator and a single standalone PMT, from PMTSim)
   * take look at ckm CerenkovMinimal

   * HMM: actually thats too involved for initial test...
   * start dev with U4RecorderTest 


Test Environmnent : ScintStandalone
---------------------------------------

Depends on: 

1. sysrap/SEvt+NPFold+NP array holding and persisting 
2. u4/U4Recorder Geant4 object collecting 
3. need to migrate some volume setup machinery from X4 to U4, eg X4VolumeMaker 
4. quite a few classes from CFG4 need to be migrated to U4 to do the Opticks mocking 


New Approach : Whats different
-------------------------------

0. much stricter dependency control, shallow dependency tree : avoiding Tower of Babel effect 
1. new U4 + SEvt approach to genstep collection
2. SEvt/NPfold array holding and persisting 
3. NP (not NPY) arrays : NP does not yet have an extend method 
4. replace Opticks instance for config with SGeoConfig SEventConfig and others if needed
5. populate sphoton.h struct rather than writing directly into arrays
6. would be good to follow qsim.h but Opticks and Geant4 models
   are so different that is probably not realistic 

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


