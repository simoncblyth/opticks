High Level Refactor
=====================

Objective
------------

Op and G4 pathways with the same high-level API 

Approach
---------

* common base OpticksEngine
* consistent externalized event handling, allow multiple g4gun firing yielding multiple events 

* keep in mind : what needs to be done only at startup, and only at each event

* streamlining to make cfg4-/CG4Test clean and simple
* streamlining to make ggv-/GGeoView clean and simple
* most of GGeoView is geometry setup, need a GGeoManager to 
  mop this up

* remove duplication between pathways
* move infrastructure like GCache/OpticksResource into Opticks




Tests Commands To Run Whilst Refactoring
------------------------------------------


G4 running::

    ## G4 test running starting from genstep, which means non-dynamic record/photon/sequence collection

    ggv-pmt-test --cfg4  --dbg                ## geant4 with test PMT in box geometry

    ggv-pmt-test --cfg4  --dbg --load         ## loading and vizualizing into GGeoView

    
    ## G4 gdml geometry running starting from nothing, dynamic collection

    ggv-g4gun --dbg 

    ggv-g4gun --dbg  --load

        ## photon selection menu item doesnt show up in GUI
 

Op running::

    ## Op full geometry running from genstep 

    op 




