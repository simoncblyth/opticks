G4/Op Integration Overview
============================

Objectives
-----------

Full geometry Geant4 and Opticks integrated, enabling:

* full geometry testing/development/comparison
* Geant4 particle gun control from within interactive Opticks (GGeoView) 
* operational without CUDA capable GPU, using Geant4 simulation and OpenGL viz
* drastically faster operation when have  CUDA capable GPU 

See Also
---------

* cfg4- partial integration for small test geometries only
* export- geometry exporter


DONE : High Level Cleanup/Refactor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* main: migrate from App to ggeoview-/OpticksMgr
* config/control/event: handling into okg-/OpticksHub 
* visualization: into oglrap-/OpticksViz 

TODO : Testing untested aspects of cleanup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Aspects included in cleanup, but as yet un-tested 

* clearer distinction between once-only geometry initialization 
  within OptiX and otherwise per-event actions (in response to input gensteps) 

* multi-event handling ... gensteps (eg G4gun derived or from multi-event file or over network ZMQ) 
  need to be repeatedly passed to OpEngine and the OpticksHub, 
  OpticksViz etc.. 
  

 
DONE : Optical Step Collection
--------------------------------

* :doc:`optical_step_collection`


DONE : Material Code Mapping Generalization
----------------------------------------------

* :doc:`material_code_mapping_generalization`


WIP : Integrated running
---------------------------

::

    OKG4MgrTest              
       # default torch step running, produces nothing visible in integrated running 
       # after trying to hand off generated torch gensteps to opticks 
       # just see axis and no geometry, and the index looks like all photons are missing
       # (targetting issue) 

    OKG4MgrTest --g4gun      
       # integrated g4gun running,  produces a visible propagation

    OKG4MgrTest --g4gun --save   
       #   now saves both g4 and ok evt with same parameter dir timestamp
       #         /tmp/blyth/opticks/evt/dayabay/g4gun/100/
       #         /tmp/blyth/opticks/evt/dayabay/g4gun/-100/


NEXT : G4/Op Comparison of generation distribs
-------------------------------------------------

Integrated is special as are doing generation and propagation with both G4 and Op
from the same single executable 

::

    ipython -i $(which tevt.py) --  --src g4gun --tag 100 


NEXT : event handling in integrated mode
-------------------------------------------

Attempting to re-use the G4 created evt for the Opticks propagation
in order to visualize the nopsteps results in a hard CUDA copy crash on launch, 
requiring a reboot.  Maybe this is because are attempting to upload buffers
which are normally produced GPU side like the records, photons and sequence
which are all mimicked CPU side by CG4.

In retrospect its the wrong thing to do anyhow, integrated mode
is effectively producing two events.  Instead just copy the 
G4 nopsteps (and of course the gensteps) into the Opticks evt.

So do the negated tag for G4 trick previously did via 
arguments in OKG4Mgr ? 

::

    084 void OKG4Mgr::propagate()
     85 {
     86     m_g4->propagate();
     87 
     88     NPY<float>* gs = m_g4->getGensteps();
    ...
    101     m_hub->translateGensteps(gs);     // relabel and apply lookup
    102 
    103     OpticksEvent* evt = m_hub->createEvent(); // make a new evt 
    104     //OpticksEvent* evt = m_hub->getEvent();      // use the evt created by CG4 
    105 
    106     evt->setGenstepData(gs);


NEXT : move CG4 event creation later for multi-event
-------------------------------------------------------------



Approach
---------

Geant4 and Opticks need to be using the same geometry...
 
* G4DAE for Opticks
* GDML for Geant4 

Standard export- controlled geometry exports include the .gdml
and .dae when they have a "G" and "D" in the path like the 
current standard::

  /usr/local/env/geant4/geometry/export/DayaBay_VGDX_20140414-1300/


ggeoview-/App
~~~~~~~~~~~~~~~

Far to much stuff in "global" app scope.  Need to partition 
off functionality into other classes, with eye to G4 integration.


OpticksEngine base class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    class CFG4_API CG4 : public OpticksEngine

Currently *OpticksEngine* is a rather minimal base class of *CG4* 
but its intended to mop up common aspects between Opticks and Geant4
simulators.  Stuff in ggeoview-/App that is common ?

::

    simon:opticks blyth$ opticks-lfind OpticksEngine
    ./cfg4/CG4.cc
    ./cfg4/CG4.hh
    ./optickscore/OpticksEngine.cc
    ./optickscore/OpticksEngine.hh


* cfg4-/tests/CG4Test.cc is very simple (high level steering only)
* ggv-/tests/GGeoViewTest.cc can that be similarly simplified ?


* Op and G4 are not really peers, Op can only do a subset of what G4 does


OpticksApp 
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Bad name, too generic. 

* OpticksCtx ?
* OpticksSim ?
* OpticksGUI/Viz ?  manage frame, window etc.. 

* What are the responsibilities of an OpticksApp ? 
* How does that fit in with CG4, OpticksEngine, OpEngine ?
* where does it belong ? OpenGL dependencies ? Or keep it abstract ?


How does Opticks and G4 need to interface ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* G4 geometry needs to be used by Op
* G4 produced gensteps need to be provided to Op
* Op produced hits need to be given back for G4 collection


DONE
-----

* OpticksResource .gdml path handling 
* Break off a CG4 singleton class from cfg4- to hold common G4 components, runmanager etc.. 
* move ggv- tests out of ggeoview- into separate .bash, check the cfg4 tests following refactor 
* add GDML loading 
* workaround lack of MPT in ancient g4 GDML export by converting from the G4DAE export  
* collect other(non-photon producing processes) particle step tree into nopstep buffers

* split G4 geometry handling into TEST and FULL using a CDetector based specialized with:

  * CTestDetector for simple partial geometries
  * CGDMLDetector for full GDML loaded geometries 

* pmt test broken by g4gun generalizations, fixed up to the groupvel issue
* CPU indexing, to support non-CUDA capable nodes 




DEBUGGING
----------

* nopstep visualization 

TODO
----

* workout where/how best to do the with/without CUDA split, 
  
  * currently done very coarsely in App with preprocessor macro WITH_OPTIX

* where to slot in CG4/CGDMLDetector into the machinery, cli, options, config ?

  * ggv-/App needs overhaul/simplification before attempting to bring in CG4
  * CG4 similarly needs cleanup, especially re event handling 

  * need to arrange CG4 and OpEngine to have a common 
    high level OpticksEngine API 

    * common aspects are: event handling/saving 
    * see :doc:`high_level_refactor`
 

* bring over, cleanup, simplify G4DAEChroma gdc- (no need for ZMQ) 
  with the customized step collecting Cerenkov and Scintillation processes

* gun control interface, ImGui?  particle palette, shooter mode

* updated JUNO export, both DAE and GDML 



