Material Code Mapping Generalization
=======================================

Persisted gensteps contain material indices, in order to 
map these to actual materials it is necessary to have 
a code to material name mapping. 

Legacy Approach
----------------

Translate on load
~~~~~~~~~~~~~~~~~~~

Genstep material indices are translated into GPU material lines on loading the file,
to keep things simple GPU side.

`NPY<float>* OpticksHub::loadGenstepFile()`::

    389     G4StepNPY* g4step = new G4StepNPY(gs);
    390     g4step->relabel(CERENKOV, SCINTILLATION);
    391     // which code is used depends in the sign of the pre-label 
    392     // becomes the ghead.i.x used in cu/generate.cu
    393 
    394     if(m_opticks->isDayabay())
    395     {
    396         // within GGeo this depends on GBndLib
    397         NLookup* lookup = m_ggeo ? m_ggeo->getLookup() : NULL ;
    398         if(lookup)
    399         {
    400             g4step->setLookup(lookup);
    401             g4step->applyLookup(0, 2);  // jj, kk [1st quad, third value] is materialIndex
    402             //
    403             // replaces original material indices with material lines
    404             // for easy access to properties using boundary_lookup GPU side
    405             //
    406         }
    407         else
    408         {
    409             LOG(warning) << "OpticksHub::loadGenstepFile not applying lookup" ;
    410         }
    411     }
    412     return gs ;
         

* with in memory gensteps direct from G4, need to do the 
  same thing but with the lookup will need to be different


Lookups
~~~~~~~~~

* npy-/NLookup does the mapping

::

     /// setupLookup is invoked by GGeo::loadGeometry

     620 void GGeo::setupLookup()
     621 {
     622     //  maybe this belongs in GBndLib ?
     623     //
     624     m_lookup = new NLookup() ;
     625 
     626     const char* cmmd = m_opticks->getDetectorBase() ;
     627 
     628     m_lookup->loadA( cmmd, "ChromaMaterialMap.json", "/dd/Materials/") ;
     629 
     630     std::map<std::string, unsigned int>& msu  = m_lookup->getB() ;
     631 
     632     m_bndlib->fillMaterialLineMap( msu ) ;
     633 
     634     m_lookup->crossReference();
     635 
     636     //m_lookup->dump("GGeo::setupLookup");  
     637 }



ggeo-/tests/NLookupTest.cc::

    GBndLib* blib = GBndLib::load(m_opticks, true );

    NLookup* m_lookup = new NLookup();

    const char* cmmd = m_opticks->getDetectorBase() ;

    m_lookup->loadA( cmmd , "ChromaMaterialMap.json", "/dd/Materials/") ;

    std::map<std::string, unsigned int>& msu = m_lookup->getB() ;

    blib->fillMaterialLineMap( msu ) ;     // shortname eg "GdDopedLS" to material line mapping 

    m_lookup->crossReference();

    m_lookup->dump("ggeo-/NLookupTest");



ChromaMaterialMap.json contains name to code mappings used 
for a some very old gensteps that were produced by G4DAEChroma
and which are still in use.
As the assumption of all gensteps being produced the same
way and with the same material mappings will soon become 
incorrect, need a more flexible way.

Perhaps a sidecar file to the gensteps .npy should
contain material mappings, and if it doesnt exist then 
defaults are used ?

::

    simon:DayaBay blyth$ cat ChromaMaterialMap.json | tr "," "\n"
    {"/dd/Materials/OpaqueVacuum": 18
     "/dd/Materials/Pyrex": 21
     "/dd/Materials/PVC": 20
     "/dd/Materials/NitrogenGas": 16
     "/dd/Materials/Teflon": 24
     "/dd/Materials/ESR": 9
     "/dd/Materials/MineralOil": 14


Changes
---------

* move NLookup to live up in OpticksHub in order to 
  configure it from the hub prior to geometry loading 
  when the lookup cross referencing is done
 


