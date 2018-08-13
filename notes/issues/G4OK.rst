G4OK : interface for Opticks embedded inside Geant4
=======================================================

Testing with examples/Geant4/CerenkovMinimal : ckm--
-------------------------------------------------------

::

    ckm--


G4Opticks::setGeometry
--------------------------

1. DONE : bring in the X4 direct translation


DONE : examples/Geant4/CerenkovMinimal
----------------------------------------

* :doc:`G4OK_SensitiveDetector` 

  * DONE : example app with SD as testing ground to develop G4OK 


DONE : automated material mapping
-------------------------------------

* :doc:`G4OK_MaterialIndex2TextureLine_NLookup`

In CCollector the NLookup translates raw G4 materialIdx into a GBndLib texture line 
this is definitely needed : but it should be possible to entirely automate it 
from the direct geometry and thus get it out of the interface


NEXT : Review OpticksHub usage in OKOP
-----------------------------------------

OpticksHub is used throughout OKOP, OpMgr OpPropagator, OpEngine
but it does too much that is not relevant to pure-compute embedded running, 
so... 

Investigate how its used in OKOP and see if a lite version (perhaps OpticksCtx) 
with just the essentials replace it.  The ctx can then be a constituent of OpticksHub.


OpEngine
~~~~~~~~~

::

     39 OpEngine::OpEngine(OpticksHub* hub)
     40      :
     41       m_log(new SLog("OpEngine::OpEngine")),
     42       m_hub(hub),
     43       m_ok(m_hub->getOpticks()),
     44       m_scene(new OScene(m_hub)),
     45       m_ocontext(m_scene->getOContext()),
     46       m_entry(NULL),
     47       m_oevt(NULL),
     48       m_propagator(NULL),



