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


WIP : Integrated G4GUN running
---------------------------------

::

    OKG4Test --g4gun      
       # Thu: integrated g4gun running,  produces a visible propagation
       #
       # Fri: NOT DOING SO WELL ANYMORE FOLLOWING CHANGES FOR TORCH
       #
       #      FIXED, HAD MISSED TRANSLATION OF THE G4 LIVE COLLECTED GENSTEPS
       #

    OKG4Test --g4gun --save   
       #   now saves both g4 and ok evt with same parameter dir timestamp
       #         /tmp/blyth/opticks/evt/dayabay/g4gun/100/
       #         /tmp/blyth/opticks/evt/dayabay/g4gun/-100/
       #
       #  Fri: still saving OK
       #  


Compare events in 2 ipython sessions::


     ipython -i $(which tevt.py) --  --src g4gun --tag -100 

     ipython -i $(which tevt.py) --  --src g4gun --tag  100 


The G4 evt looks reasonable::

     In [3]: evt.history_table()
                      -100:dayabay 
                  4f        0.921           1310       [2 ] G4GUN AB
           4cccccccf        0.042             60       [9 ] G4GUN BT BT BT BT BT BT BT AB
          cccbcccccf        0.011             15       [10] G4GUN BT BT BT BT BT BR BT BT BT
                4ccf        0.005              7       [4 ] G4GUN BT BT AB
          4cc00cc0cf        0.004              5       [10] G4GUN BT ?0? BT BT ?0? ?0? BT BT AB


OK one looks wrong, maybe not labelled::

    In [8]: evt.history_table()
                          100:dayabay 
                       3        0.999           1421       [1 ] MI
                   45552        0.001              1       [5 ] SI RE RE RE AB
                                1422         1.00 


Raw pre-labels were not converted into CERENKOV, SCINTILLATION codes::

    In [11]: evt.gs.shape
    Out[11]: (100, 6, 4)


    In [14]: evt.gs[:,0].view(np.int32)
    Out[14]: 
    A()sliced
    A([[  -1,    1,    7,   23],        ##  sid/parentId/materialIndex/numPhotons
           [  -2,    1,    7,   18],
           [  -3,    1,    7,   22],
           [  -4,    1,    7,   32],
           [  -5,    1,    7,   27],
           [   1,    1,    7,    0],
           [   2,    1,    7,    1],
           ...
           [   3,    1,    7,    0],
           [   4,    1,    7,    1],
           [ -19,    1,    7,   18],
           [ -20,    1,    7,   19],
           [ -21,    1,    7,   21],
           [ -22,    1,    7,   20],
           [ -23,    1,    7,   18],
           [ -24,    1,    7,   26],
           ...
           [ -64,    1,    7,    6],
           [ -65,    1,    7,    7],
           [  11,    1,    7,    1],
           [  12,    1,    7,    1],
           [  13, 1327,    7,    1],
           [  14, 1327,    7,    1],
           [  15, 1345,    7,    0],
           [  16, 1345,    7,    1],
           ...
           [  25, 1328,    7,    0],
           [  26, 1328,    7,    1],
           [ -69, 1328,    7,   23],
           [ -70, 1328,    7,   14],
           [ -71, 1328,    7,   12],
           [ -72, 1328,    7,   13],
           [  27, 1328,    7,    1],
           [  28, 1328,    7,    1]], dtype=int32)

    In [16]: evt.gs[:,0,3].view(np.int32).sum()
    Out[16]: 
    A()sliced
    A(1422)


After apply the translation, the code and material lines are corrected::

    In [1]: evt.gs[:,0].view(np.int32)
    Out[1]: 
    A()sliced
    A([[   1,    1,   95,   23],
           [   1,    1,   95,   18],
           [   1,    1,   95,   22],
           [   1,    1,   95,   32],
           [   1,    1,   95,   27],
           [   2,    1,   95,    0],
           [   2,    1,   95,    1],
           [   1,    1,   95,   14],
           [   1,    1,   95,   19],


And now history looks better, but scintillation is missing::

     100:dayabay 
                  41        0.526            748       [2 ] CK AB
             8cccc51        0.074            105       [7 ] CK RE BT BT BT BT SA
                 451        0.063             89       [3 ] CK RE AB
            8cccc551        0.038             54       [8 ] CK RE RE BT BT BT BT SA
                4551        0.030             43       [4 ] CK RE RE AB
           8cccc5551        0.022             31       [9 ] CK RE RE RE BT BT BT BT SA
              8cccc1        0.015             22       [6 ] CK BT BT BT BT SA
               45551        0.014             20       [5 ] CK RE RE RE AB
          ccacccccc1        0.013             18       [10] CK BT BT BT BT BT BT SR BT BT
          cacccccc51        0.011             15       [10] CK RE BT BT BT BT BT BT SR BT
          cbccccc551        0.009             13       [10] CK RE RE BT BT BT BT BT BR BT


Checking consistency between input steps and output sequence, looks OK, there are a few SI (probably so few due to scintillator dial down)::

    OKG4Test --g4gun --save  


    2016-09-05 13:43:44.547 INFO  [591752] [G4StepNPY::checkCounts@100] OpticksIdx::indexSeqHost checkCounts compare *seqCounts* (actual photon counts from propagation sequence data SeqNPY )  with *stepCounts* (expected photon counts from input G4StepNPY )  
     bpos(hex)          0 seqCounts          0 flagLabel          0 stepCounts          0
     bpos(hex)          1 seqCounts       1405 flagLabel          1 stepCounts       1405
     bpos(hex)          2 seqCounts         17 flagLabel          2 stepCounts         17
     bpos(hex)          3 seqCounts          0 flagLabel          4 stepCounts          0
     bpos(hex)          4 seqCounts          0 flagLabel          8 stepCounts          0
     bpos(hex)          5 seqCounts          0 flagLabel         16 stepCounts          0
     bpos(hex)          6 seqCounts          0 flagLabel         32 stepCounts          0
     bpos(hex)          7 seqCounts          0 flagLabel         64 stepCounts          0
     bpos(hex)          8 seqCounts          0 flagLabel        128 stepCounts          0
     bpos(hex)          9 seqCounts          0 flagLabel        256 stepCounts          0
     bpos(hex)          a seqCounts          0 flagLabel        512 stepCounts          0
     bpos(hex)          b seqCounts          0 flagLabel       1024 stepCounts          0
     bpos(hex)          c seqCounts          0 flagLabel       2048 stepCounts          0
     bpos(hex)          d seqCounts          0 flagLabel       4096 stepCounts          0
     bpos(hex)          e seqCounts          0 flagLabel       8192 stepCounts          0
     bpos(hex)          f seqCounts          0 flagLabel      16384 stepCounts          0



DONE : genstep handling rationalize
------------------------------------

* translateGensteps invoked from multiple places
* genstep handoff from G4 to OK is messy 
* avoid duplication between OKMgr and OKG4Mgr ?
* targetting configuration in Scene (which is not always available) is messy, move to Composition ? 

::

    simon:opticks blyth$ opticks-find translateGen
    ./cfg4/CG4.cc:        m_hub->translateGensteps(gs);
    ./okg4/OKG4Mgr.cc:        m_hub->translateGensteps(gsrec);
    ./okg4/OKG4Mgr.cc:    // m_hub->translateGensteps(gs);     
    ./opticksgeo/OpticksHub.cc:void OpticksHub::translateGensteps(NPY<float>* gs)
    ./opticksgeo/OpticksHub.cc:            m_lookup->close("OpticksHub::translateGensteps");
    ./opticksgeo/OpticksHub.cc:    ss << "OpticksHub::translateGensteps " 
    ./opticksgeo/OpticksHub.hh:       G4StepNPY*           getG4Step();    // created in translateGenstep
    ./opticksgeo/OpticksHub.hh:       void                 translateGensteps(NPY<float>* gs);  // into Opticks lingo
    simon:opticks blyth$ 


Perhaps can avoid translation by applying the 
lookup translation at collection.  


NEXT : OKG4Mgr propagation multi-event test
----------------------------------------------

Tidy up propagation. 

Avoid duplication between OKMgr and OKG4Mgr, 
probably using separate high level propagation class.

Clearly split:

* once-only setup, all the way to pre-launch 
* per-event just final launch 

Do multi-event propagations to test the split.


NEXT : OKG4 genstep shakedown 
------------------------------

* compare distribs as implement standard G4 process GPU ports 
  (suspect that Cerenkov is already almost there, Scintillation
   needs some porting) 


WIP : Integrated Torch running debug
---------------------------------------

::

    OKG4Test          
       #
       # Thu: default torch step running, produces nothing visible in integrated running 
       #      after trying to hand off generated torch gensteps to opticks 
       #      just see axis and no geometry, and the index looks like all photons are missing
       #      (targetting issue) 
       #
       # Fri: THIS IS NOW WORKING, AFTER GENSTEP AND MATERIAL LOOKUP REJIG

    OKTest
       # still operational 
       #
       # Fri: STILL OK


These two should show exactly the same thing, only difference is the integrated
one runs the G4 propagation in addition to the Opticks one.

* Fri: now looking the same


Arranged plogging to use simple formatter so can compare logs without times
or process identity differences. 

Difference was with composition targetting, 
due to failing to set the frame transform for the gensteps.





FIXED : OKG4 : Material Map chicken/egg problem
----------------------------------------------------

Prior to genstep material index to texture line translation 
need to get the A and B mappings and do lookup crossReference. 

That doesnt fit in with current early gensteps creation in hub.


Mapping A
~~~~~~~~~~

* G4 material name to geant4 materialIndex 
* defaults to the json, which is valid for loaded from file gensteps
  but not live created ones

* available after geant4 run manager initializes and the materials
  come into existance

Mapping B
~~~~~~~~~~~~

* OK material name to GPU texLine
* available after GBndLib is loaded

Fix
~~~~~

* maybe, moving G4 geometry loading first ?

  * didnt do that, instead just deferred doing cross referencing/translation
    until just before setting into OpticksEvent and allowing 
    the A lookup to be overrided once the G4 materials are available




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



