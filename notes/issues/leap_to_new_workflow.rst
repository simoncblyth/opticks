leap_to_new_workflow
=======================

* prior :doc:`CSGFoundry_CreateFromSim_shakedown_now_with_flexible_sn`

* CONCLUDED TOO MUCH EFFORT TO BRING OSUR IMPLICITS TO THE OLD WORKFLOW : LEAPING TO NEW WORKFLOW


LEAP ENCOMPASSES

1. hide old world GGeo etc.. behind WITH_GGEO
2. remove old world packages from om-subs--all
3. arrange G4CXOpticks::setGeometry to skip GGeo going
   instead to SSim/stree and then CSGFoundry::CreateFromSim 


SAME COMMAND FROM PREVIOUS NOW USES NEW WORKFLOW
-------------------------------------------------

::

     NEW    U4Tree             CSGImport
     Geant4 -----> SSim/stree ----->  CSGFoundry 
                         
     ~/opticks/g4cx/tests/G4CXOpticks_setGeometry_Test.sh   


NEXT
-----

* peel back to Geant4 A/B comparison was doing previously : with 3inch PMT discrep
  motivating the addition of OSUR implicits to NEW workflow


