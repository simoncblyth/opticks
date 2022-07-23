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

* difficult to reproduce 

  * simple geom + simple geom instanced does not show the issue
  * also rerunning the GDML conversion not showing the issue 
  * makes it look like pilot error ?  

* what could possibly be wrong ?

  * factorization transform rearrangement seems most likely point  
  * TODO: test with repeat_candidate cut upped to 10e6 so the entire geometry is global 

* problem is from an old geometry conversion : done around 1st June 2022

  * problem with specific CFBASE directory 


The below problem of dynamic skipping does not explain the bad translation however, 
as it just means the event arrays did not match the geometry : 
it does not explain how the ellipsoid scale transforms managed to be skipped. 


Commits from around 1st June
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Was working on SGeoConfig::GeometrySpecificSetup and ELV selection for skipping virtuals, 
initially with dynamic prim selection
    
* subsequently I realised that using dynamic CF level selection is 
  just not appropriate for making skips as it means the persisted CFBase geometry 
  does not match that used by simulation running

  * when did I realize that ? 2022-06-01

2022-05-31
    ELVSelection succeeds to skip the virtual jackets as visible in simtrace
    plotting but observe prim mis-naming at python level, presumably because the
    python naming is based on the copied geocache which is unchanged by dynamic
    prim selection

2022-06-01
    generalize the IDXList from names conversion so it can be used for CXSkips as
    well as ELVString, switch to doing skips at translation time in
    CSG_GGeo_Convert rather than CSGFoundry::Load : as changing geometry is a
    bookkeeping problem so better to do it less often

2022-06-01
    remove the nasty mixed CFBase kludge now that have moved the virtual
    Water///Water skips to translation time instead of load time

2022-06-01
    propagate prd.iindex from optixGetInstanceIndex into sphoton replacing weight
    which was unused, add prim name dumping to p.npy


From notes/issues/primIdx-and-skips.rst

CONCLUDED BY MOVING THE LONG LIVED SKIPS TO TRANSLATON TIME NOT LOAD TIME

See:

1. SGeoConfig::IsCXSkipLV
2. SGeoConfig::GeometrySpecificSetup 


* https://bitbucket.org/simoncblyth/opticks/annotate/master/sysrap/SGeoConfig.cc?at=master


* https://bitbucket.org/simoncblyth/opticks/commits/a49f8ab4fa88a39ae217648f1fde9057bd9aa8dc
* https://bitbucket.org/simoncblyth/opticks/commits/be94991b615b6c7fca1d00b473ab1d3d52350dcd


Current
~~~~~~~~~

::

    2316 CSGFoundry* CSGFoundry::Load() // static
    2317 {   
    2318     LOG(LEVEL) << " argumentless " ; 
    2319     CSGFoundry* src = CSGFoundry::Load_() ;
    2320     if(src == nullptr) return nullptr ;
    2321     
    2322     SGeoConfig::GeometrySpecificSetup(src->id);
    2323     
    2324     const SBitSet* elv = ELV(src->id); 
    2325     CSGFoundry* dst = elv ? CSGFoundry::CopySelect(src, elv) : src  ;
    2326     
    2327     if( elv != nullptr && Load_saveAlt)
    2328     {   
    2329         LOG(error) << " non-standard dynamic selection CSGFoundry_Load_saveAlt " ;
    2330         dst->saveAlt() ;
    2331     }
    2332     return dst ;
    2333 }
    2334 
    2335 CSGFoundry* CSGFoundry::CopySelect(const CSGFoundry* src, const SBitSet* elv )
    2336 {
    2337     assert(elv);
    2338     LOG(info) << elv->desc() << std::endl << src->descELV(elv) ;
    2339     CSGFoundry* dst = CSGCopy::Select(src, elv );
    2340     dst->setOrigin(src);
    2341     dst->setElv(elv);
    2342     dst->setOverrideSim(src->sim);
    2343     // pass the SSim pointer from the loaded src instance, 
    2344     // overriding the empty dst SSim instance 
    2345     return dst ;
    2346 }


    107 /**
    108 SGeoConfig::IsCXSkipLV
    109 ------------------------
    110 
    111 This controls mesh/solid skipping during GGeo to CSGFoundry 
    112 translation as this is called from:
    113 
    114 1. CSG_GGeo_Convert::CountSolidPrim
    115 2. CSG_GGeo_Convert::convertSolid
    116 
    117 For any skips to be applied the below SGeoConfig::GeometrySpecificSetup 
    118 must have been called. 
    119 
    120 For example this is used for long term skipping of Water///Water 
    121 virtual solids that are only there for Geant4 performance reasons, 
    122 and do nothing useful for Opticks. 
    123 
    124 Note that ELVSelection does something similar to this, but 
    125 that is applied at every CSGFoundry::Load providing dynamic prim selection. 
    126 As maintaining consistency between results and geometry is problematic
    127 with dynamic prim selection it is best to only use the dynamic approach 
    128 for geometry render scanning to find bottlenecks. 
    129 
    130 When creating longer lived geometry for analysis with multiple executables
    131 it is more appropriate to use CXSkipLV to effect skipping at translation. 
    132 
    133 **/
    134 
    135 bool SGeoConfig::IsCXSkipLV(int lv) // static
    136 {
    137     if( _CXSkipLV_IDXList == nullptr ) return false ;
    138     std::vector<int> cxskip ;
    139     SStr::ISplit(_CXSkipLV_IDXList, cxskip, ',');
    140     return std::count( cxskip.begin(), cxskip.end(), lv ) == 1 ;
    141 }



X4Solid::convertEllipsoid : G4Ellipsoid -> nzsphere with associated scale transform
-------------------------------------------------------------------------------------

::

    1480 void X4Solid::convertEllipsoid()
    1481 { 
    1482     const G4Ellipsoid* const solid = static_cast<const G4Ellipsoid*>(m_solid);
    1483     assert(solid);
    1484 
    1485     // G4GDMLWriteSolids::EllipsoidWrite
    1486 
    1487     float ax = solid->GetSemiAxisMax(0)/mm ;
    1488     float by = solid->GetSemiAxisMax(1)/mm ;
    1489     float cz = solid->GetSemiAxisMax(2)/mm ;
    1490 
    1491     glm::vec3 scale( ax/cz, by/cz, 1.f) ;
    1492     // unity scaling in z, so z-coords are unaffected  
    ...
    1532     nnode* cn = zslice ?
    1533                           (nnode*)nzsphere::Create( 0.f, 0.f, 0.f, cz, z1, z2 )
    1534                        :
    1535                           (nnode*)nsphere::Create( 0.f, 0.f, 0.f, cz )
    1536                        ;
    1537 
    1538     cn->label = BStr::concat(m_name, "_ellipsoid", NULL) ;
    1539     cn->transform = nmat4triple::make_scale( scale );
    1540    





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



Compare CF geometries with CSG/tests/CSGFoundryAB.sh 
--------------------------------------------------------


* :doc:`ellipsoid_transform_compare_two_geometries`



