review_geometry_translation : G4VPhysicalVolume -> GGeo -> CSGFoundry
=========================================================================

Objectives
------------

* three full geometries models is one too many, but an intermediary model 
  is needed for factorization 

  * how minimal could the middle GGeo model become ? 

  * material and surface properties could be simplified straightforwardly, 
    effectively direct G4 -> NP/NPFold/SSim which then get used by QSim 

* look for areas of simplification+code reduction 

  * consider NPY->NP transition 
  * BRAP dependency removal from GGeo  

* look for causes of ellipsoid transform fragility 

* better control of complex parts of translation for debugging issues

  * uncoincidence nudging : a source of fragility 
  * tree balancing : another source of fragility

* better mapping between source G4 volumes and final CF instances

  * because of the factorization this is not an easy task 
  * particularly during live translation  
  * want to be able to fully mockup Opticks flags (like boundary) during 
    B-side running : that means the CF geometry needs to be usable from the B-side too  

* full precision transforms 


ideas for ellipsoid transform debugging
------------------------------------------


Characteristics of the issue:

* difficult to reproduce : simple geom + simple geom instanced does not show the issue
* what could possibly be wrong ?

  * factorization transform rearrangement seems most likely point  
  * so: test with repeat_candidate cut upped to 10e6 so the entire geometry is global 
  





G4CXOpticks::setGeometry
---------------------------

::

    111 void G4CXOpticks::setGeometry(const G4VPhysicalVolume* world)
    112 {   
    114     wd = world ;
    118     GGeo* gg_ = X4Geo::Translate(wd) ;
    119     setGeometry(gg_); 
    120 }   
    121 void G4CXOpticks::setGeometry(const GGeo* gg_)
    122 {       
    124     gg = gg_ ; 
    125     CSGFoundry* fd_ = CSG_GGeo_Convert::Translate(gg) ;
    126     setGeometry(fd_); 
    127 }       



X4Geo::Translate : G4VPhysicalVolume -> GGeo : instanciating X4PhysicalVolume populates GGeo
-------------------------------------------------------------------------------------------------

::

     19 GGeo* X4Geo::Translate(const G4VPhysicalVolume* top)  // static 
     20 {
     21     bool live = true ;
     22     
     23     GGeo* gg = new GGeo( nullptr, live );   // picks up preexisting Opticks::Instance
     24     
     25     X4PhysicalVolume xtop(gg, top) ;
     26     
     27     gg->postDirectTranslation();
     28     
     29     return gg ;
     30 }   


::

     191 void X4PhysicalVolume::init()
     192 {
     197     convertWater();       // special casing in Geant4 forces special casing here
     198     convertMaterials();   // populate GMaterialLib
     199     convertScintillators();
     201 
     202     convertSurfaces();    // populate GSurfaceLib
     203     closeSurfaces();
     204     convertSolids();      // populate GMeshLib with GMesh converted from each G4VSolid (postorder traverse processing first occurrence of G4LogicalVolume)  
     205     convertStructure();   // populate GNodeLib with GVolume converted from each G4VPhysicalVolume (preorder traverse) 
     206     convertCheck();       // checking found some nodes
     207 
     208     postConvert();        // just reporting 
     211 }


* convertSolids and convertStructure are much more involved than material/surface handling 

::

    export X4PhysicalVolume=INFO





