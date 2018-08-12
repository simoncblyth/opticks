G4OK_MaterialIndex2TextureLine_NLookup : Automate it 
========================================================

* DONE : in G4Opticks 


opticks-findl NLookup
-----------------------

::

    epsilon:opticks blyth$ opticks-findl NLookup
    ./ggeo/ggeo.bash
    ./opticksgeo/OpticksHub.cc
    ./cfg4/CCollector.cc
    ./cfg4/CG4.cc
    ./g4ok/G4Opticks.cc
    ./ggeo/GGeo.cc
    ./ggeo/tests/NLookupTest.cc
    ./optickscore/OpticksEvent.cc
    ./optickscore/OpticksRun.cc
    ./npy/tests/NLookupTest.cc
    ./opticksgeo/OpticksHub.hh
    ./opticksgeo/OpticksGen.hh
    ./cfg4/CCollector.hh
    ./g4ok/G4Opticks.hh
    ./ggeo/GGeo.hh
    ./npy/G4StepNPY.cpp
    ./npy/NPYBase.cpp
    ./npy/NLookup.cpp
    ./npy/NLookup.hpp
    ./npy/NPYBase.hpp
    ./npy/G4StepNPY.hpp
    ./ggeo/tests/CMakeLists.txt
    ./npy/CMakeLists.txt
    epsilon:opticks blyth$ 




material indices to GBndLib texture lines
---------------------------------------------

* material indices are converted into GBndLib texture lines at input, prior to getting to GPU,  
  so they can be directly used against the boundary texture GPU side
* the properties of materials and surfaces are interleaved into the boundary texture, this
  means that the properties for a particular material or surface will often be repeated multiple
  times within the texture 
* every texture line corresponds to the properties of a single material or surface 
* each boundary has four texture lines for omat/osur/isur/imat so to get a texture line 
  from a zero based boundary index:: 

        bndIndex*4 + 0   omat
        bndIndex*4 + 1   osur
        bndIndex*4 + 2   isur
        bndIndex*4 + 3   imat
 
* the first texture line for a material is used 


NLookup
----------

Provides A2B or B2A int->int mappings between two indexing schemes.
The A and B indexing schemes being represented by name to index maps
and the correspondence being established based on the names. 
After instanciating an NLookup the A and B are set either from 
loaded json or from std::map and only on closing the NLookup
are the A2B and B2A cross referencing maps formed.

ggeo/tests/NLookupTest.cc
---------------------------

Demonstrates index mapping between ChromaMaterialMap.json used in some 
old gensteps and the material texture line obtained from the 
geocache loaded GBndLib. 

CFG4.CCollector and NLookup
----------------------------

Used from CCollector::collectScintillationStep and CCollector::collectCerenkovStep to 
translate Geant4 raw material indices into texture lines.::

   
     55 int CCollector::translate(int acode) // raw G4 materialId translated into GBndLib material line for GPU usage 
     56 {
     57     if(!m_lookup)
     58     {
     59         LOG(fatal) << " no lookup " ;
     60         return acode ;
     61     }
     62 
     63     int bcode = m_lookup->a2b(acode) ;
     64     return bcode ;
     65 }


G4OK NLookup
--------------

X4 direct conversion from a Geant4 geometry populates a GGeo and GBndLib, so 
the material name to texline mapping can 



