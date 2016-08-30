
NEXT : G4 Optical Step Collection
======================================

Objective
-----------

CG4 steered simulation running from GDML geometry with natural 
optical steps propagated with Opticks (using OpticksMgr) and hits
passed back to CG4 

Stages
-------

* Revisit optical step collection within cfg4-, add natural collection
  of both Cerenkov and Scintillation process steps into one buffer

* The OpticksMgr integration  will need to happen in a new package, but 
  the G4 level step collection can be developed first with cfg4- and
  tested on genstep file basis with OpticksMgrTest 


Review old collection with G4DAEChroma
------------------------------------------

* http://simoncblyth.bitbucket.org/env/notes/chroma/G4DAEChroma/
* 

* *gdc-* developed in ~/env/chroma/G4DAEChroma 
* some version was copied into Dayabay NuWa svn Utilities
* Tao also integrated some version into JUNO  ~/offline/Simulation/DetSimV2/G4DAEChroma

* expect should be straightforward enough to incorporate into Opticks cfg4- ? 

G4DAEChroma Observations
~~~~~~~~~~~~~~~~~~~~~~~~~~

* was developed in an earlier (more complicated) era when was using python Chroma, 
  so had to handle distributed operation over ZMQ, metadata, database !

* had not settled on step transport 

* much of gdc- array handling, serialization is obsolete, 
  now that have opticks npy-/NPY and lots of experience using it 

* yuck : even some ROOT TObject serialization stuff in G4DAEChromaPhotonList
  which is totally obsolete


G4DAEChroma Things to plunder : looks like very little, NPY handles most of it  
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* G4 workflow : G4DAEChroma::BeginOfRun EndOfRun BeginOfEvent ...

* G4DAESensDet::MakeTrojanSensDet G4 trojan SD to sneak hits into G4 hit collections en-masse


Approach to reviving optical step collection
-----------------------------------------------

* raid G4DAEChroma for the minimum necessary amount of code to 
  do optical step collection within a new class for this in opticks cfg4- 

  * initially avoid all the bells and whistles, no ZMQ, no metadata, ...
    that complexity can hopefully be totally avoided with the C++ opticks


