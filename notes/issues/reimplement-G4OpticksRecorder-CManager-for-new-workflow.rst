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

Preliminaries
---------------

* move event array saving from QEvent down to SEvt/NPFold, then 
  U4Recorder can depend only on sysrap, G4 
  (as it makes no sense for U4 to depend on qudarap/QEvent)

How to proceed
-----------------

1. setup test environment to exercise the U4Recorder in standalone way 
   for fast development cycle on laptop.  

   * ScintStandalone (starting from sgs:ScintGenStandalone) with some simple geometry 
   * (eg big sphere of scintillator and a single standalone PMT, from PMTSim)


New Approach : Whats different
-------------------------------

1. new U4 + SEvt approach to genstep collection
2. SEvt/NPfold array holding and persisting 
3. NP (not NPY) arrays : NP does not yet have an extend method 
4. replace Opticks instance for config with SGeoConfig SEventConfig and others if needed
5. sphoton.h  (not directly into arrays)


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



