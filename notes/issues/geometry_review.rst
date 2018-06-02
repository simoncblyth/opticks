Geometry Review
==================

Issues
--------

* going via .dae and .gdml files is historical too : predating the OpticksCSG 
  .gltf analytic approach which was grafted on 


See also 
---------

* :doc:`scene` early thoughts on analytic geometry 


Direct from live G4 to geocache/gltf : how difficult ? 
-------------------------------------------------------- 

* :doc:`direct_to_gltf_feasibility`


Control Layers for geometry loading
--------------------------------------

okc.Opticks
      * commandline control, resource management
      * currently OpticksResource/BOpticksResource requires a geometry cache 
        which makes no sense in general, ie prior to exporting geometry 
        ... need to split off the code that makes this assumption 
        into a separate class "OpticksDetectorResource" ?

okg.OpticksHub   
      very high level steering


okg.OpticksGeometry   
      middle management

ggeo.GGeo
      worker


Worker Classes
----------------

AssimpGGeo
    importer that traverses the assimp (COLLADA) G4DAE geometry 
    populating GGeo as it goes 

    static int AssimpGGeo::load(GGeo* ggeo)  // GLoaderImpFunctionPtr


GBndLib
GMaterialLib
GSurfaceLib
GScintillatorLib
GSourceLib
    All above are GPropLib subclass constituents of GGeo     

GGeoLib 
    holder of GMergedMesh 

GGeoBase
    protocol pure virtual base guaranteeing that subclasses 
    provide accessors to the libs


GScene(GGeoBase)
    somewhat expediently GScene is held by GGeo 
    (cannot fly analytic only yet)

GGeo(GGeoBase)
    central structure holding libs of geomety objects, mesh-centric 

Nd
    mimimalistic structural nodes from glTF,
    most use via NScene, populated from the glTF by NScene::import_r

NCSG
    coordinator for NPY arrays of small numbers of nodes, transforms, planes for 
    CSG solid shapes. Must be small as uses complete binary tree serialization.

    Created in python from the source GDML, csg.py does tree manipulations 
    to avoid deep trees.

    analytic/sc.py 
    analytic/csg.py 

NGLTF
    holds the underlying ygltf tree (YoctoGL)
    ... currently assumes that always loads from file 

    See examples/UseYoctoGL/UseYoctoGL_Write.cc for a brief look
    at C++ construction of gltf structure.

    Of course constructing a gltf in memory structure does 
    not necessitate writing it to file, although thats very handy for 
    debugging.

NScene(NGLTF)
    NScene based on NGLTF 

    Comprises index keyed maps of Nd and NCSG 

    loads the gltf written by bin/gdml2gltf.py, so the route is
    ::
 
        LiveG4 -> GDMLfile -> bin/gdml2gltf.py -> NScene 

    This is essentially a GLTF exporter for G4 (if you decide to 
    write the gltf to file), but with my extras for Opticks.




Analytic GScene uses the GGeo proplibs for material/surface props...
------------------------------------------------------------------------

* unified analytic-triangulated gltf geometry would need to include all these

::

      46       
      47 // for some libs there is no analytic variant 
      48 GMaterialLib*     GScene::getMaterialLib() {     return m_ggeo->getMaterialLib(); }
      49 GSurfaceLib*      GScene::getSurfaceLib() {      return m_ggeo->getSurfaceLib(); }
      50 GBndLib*          GScene::getBndLib() {          return m_ggeo->getBndLib(); }
      51 GPmtLib*          GScene::getPmtLib() {          return m_ggeo->getPmtLib(); }
      52 GScintillatorLib* GScene::getScintillatorLib() { return m_ggeo->getScintillatorLib(); }
      53 GSourceLib*       GScene::getSourceLib() {       return m_ggeo->getSourceLib(); }
      54 



Geometry consumers : what is actually needed ?
------------------------------------------------

oxrap.OGeo


oxrap.OScene
--------------

Canonical m_scene instance resides in okop-/OpEngine 

OScene::init creates the OptiX context and populates
it with geometry, boundary etc.. info 



oxrap.OGeo : operates from analytic or triangulated 
----------------------------------------------------------

* GParts associated with each GMergedMesh hold the analytic geometry

::

     614 optix::Geometry OGeo::makeAnalyticGeometry(GMergedMesh* mm, unsigned lod)
     615 {
     616     if(m_verbosity > 2)
     617     LOG(warning) << "OGeo::makeAnalyticGeometry START"
     618                  << " verbosity " << m_verbosity
     619                  << " lod " << lod
     620                  << " mm " << mm->getIndex()
     621                  ;
     622 
     623     // when using --test eg PmtInBox or BoxInBox the mesh is fabricated in GGeoTest
     624 
     625     GParts* pts = mm->getParts(); assert(pts && "GMergedMesh with GEOCODE_ANALYTIC must have associated GParts, see GGeo::modifyGeometry ");
     626 
     627 





Questions
------------

* How difficult to create NScene direct from live G4 ?



NScene(NGLTF)
----------------

Used by GGeo::loadFromGLTF and GScene, GGeo.cc::

     658     m_nscene = new NScene(gltfbase, gltfname, gltfconfig);
     659     m_gscene = new GScene(this, m_nscene );

Scene files in glTF format are created by opticks/analytic/sc.py 
which parses the input GDML geometry file and writes the mesh (ie solid 
shapes) as np ncsg and the tree structure as json/gltf.

NScene imports the gltf using its NGLTF based (YoctoGL external)
creating a nd tree. The small CSG node trees for each solid
are polygonized on load in NScene::load_mesh_extras.

* somehere the Geant4 polygonizations are swapped in 


opticksgeo.OpticksHub (okg-)
-----------------------------

Starts out with most things NULL, populated in init::

    138 OpticksHub::OpticksHub(Opticks* ok)
    139    :
    140    m_log(new SLog("OpticksHub::OpticksHub")),
    141    m_ok(ok),
    142    m_gltf(-1),        // m_ok not yet configured, so defer getting the settings
    143    m_run(m_ok->getRun()),
    144    m_geometry(NULL),
    145    m_ggeo(NULL),
    146    m_gscene(NULL),
    147    m_composition(new Composition),
    148 #ifdef OPTICKS_NPYSERVER
    149    m_delegate(NULL),
    150    m_server(NULL)
    151 #endif
    152    m_cfg(new BCfg("umbrella", false)),
    153    m_fcfg(m_ok->getCfg()),
    154    m_state(NULL),
    155    m_lookup(new NLookup()),
    156    m_bookmarks(NULL),
    157    m_gen(NULL),
    158    m_gun(NULL),
    159    m_aim(NULL),
    160    m_geotest(NULL),
    161    m_err(0)
    162 {
    163    init();
    164    (*m_log)("DONE");
    165 }

    167 void OpticksHub::init()
    168 {
    169     add(m_fcfg);
    170 
    171     configure();
    172     configureServer();
    173     configureCompositionSize();
    174     configureLookupA();
    175 
    176     m_aim = new OpticksAim(this) ;
    177 
    178     loadGeometry() ;
    179     if(m_err) return ;
    180 
    181     configureGeometry() ;
    182 
    183     m_gen = new OpticksGen(this) ;
    184     m_gun = new OpticksGun(this) ;
    185 }

    208 void OpticksHub::configure()
    209 {   
    210     m_composition->addConfig(m_cfg);
    211     //m_cfg->dumpTree();
    212     
    213     int argc    = m_ok->getArgc();
    214     char** argv = m_ok->getArgv();
    215     
    216     LOG(debug) << "OpticksHub::configure " << argv[0] ;
    217     
    218     m_cfg->commandline(argc, argv);
    219     m_ok->configure();
    220     
    221     if(m_fcfg->hasError())
    222     {   
    223         LOG(fatal) << "OpticksHub::config parse error " << m_fcfg->getErrorMessage() ;
    224         m_fcfg->dump("OpticksHub::config m_fcfg");
    225         m_ok->setExit(true);
    226         return ;
    227     }
    228     
    229     m_gltf =  m_ok->getGLTF() ;
    230     LOG(info) << "OpticksHub::configure"
    231               << " m_gltf " << m_gltf
    232               ;
    233     
    234     bool compute = m_ok->isCompute();
    235     bool compute_opt = hasOpt("compute") ;
    236     if(compute && !compute_opt)
    237         LOG(warning) << "OpticksHub::configure FORCED COMPUTE MODE : as remote session detected " ;
    238     
    239     
    240     if(hasOpt("idpath")) std::cout << m_ok->getIdPath() << std::endl ;
    241     if(hasOpt("help"))   std::cout << m_cfg->getDesc()     << std::endl ;
    242     if(hasOpt("help|version|idpath"))
    243     {   
    244         m_ok->setExit(true);
    245         return ;
    246     }
    247     
    248     
    249     if(!m_ok->isValid())
    250     {   
    251         // defer death til after getting help
    252         LOG(fatal) << "OpticksHub::configure OPTICKS INVALID : missing envvar or geometry path ?" ;
    253         assert(0);
    254     }
    255 }


     



okg-.OpticksHub::loadGeometry
-------------------------------

::

    356 void OpticksHub::loadGeometry()
    357 {   
    358     assert(m_geometry == NULL && "OpticksHub::loadGeometry should only be called once");
    359     
    360     LOG(info) << "OpticksHub::loadGeometry START" ;
    361     
    362     
    363     m_geometry = new OpticksGeometry(this);   // m_lookup is set into m_ggeo here 
    364     
    365     m_geometry->loadGeometry();
    366     
    367     m_ggeo = m_geometry->getGGeo();
    368     
    369     m_gscene = m_ggeo->getScene();
    370     
    371     
    372     //   Lookup A and B are now set ...
    373     //      A : by OpticksHub::configureLookupA (ChromaMaterialMap.json)
    374     //      B : on GGeo loading in GGeo::setupLookup
    375     
    ...     skip test geometry handling 
    ...
    399     registerGeometry();
    400     
    401     
    402     m_ggeo->setComposition(m_composition);
    403     
    404     LOG(info) << "OpticksHub::loadGeometry DONE" ;
    405 }   



okg-.OpticksGeometry::loadGeometry
-----------------------------------

::

     77 void OpticksGeometry::init()
     78 {
     79     bool geocache = !m_fcfg->hasOpt("nogeocache") ;
     80     bool instanced = !m_fcfg->hasOpt("noinstanced") ; // find repeated geometry 
     81 
     82     LOG(debug) << "OpticksGeometry::init"
     83               << " geocache " << geocache
     84               << " instanced " << instanced
     85               ;
     86 
     87     m_ok->setGeocache(geocache);
     88     m_ok->setInstanced(instanced); // find repeated geometry 
     89 
     90     m_ggeo = new GGeo(m_ok);
     91     m_ggeo->setLookup(m_hub->getLookup());
     92 }
     93 


     117 // setLoaderImp : sets implementation that does the actual loading
     118 // using a function pointer to the implementation 
     119 // avoids ggeo-/GLoader depending on all the implementations
     120 
     121 void GGeo::setLoaderImp(GLoaderImpFunctionPtr imp)
     122 {   
     123     m_loader_imp = imp ;
     124 }


::

    132 void OpticksGeometry::loadGeometryBase()
    133 {
    134     LOG(error) << "OpticksGeometry::loadGeometryBase START " ;
    135     OpticksResource* resource = m_ok->getResource();
    136 
    137     if(m_ok->hasOpt("qe1"))
    138         m_ggeo->getSurfaceLib()->setFakeEfficiency(1.0);
    139 
    140 
    141     m_ggeo->setLoaderImp(&AssimpGGeo::load);    // setting GLoaderImpFunctionPtr
    142 
    143 
    144     m_ggeo->setMeshJoinImp(&MTool::joinSplitUnion);
    145     m_ggeo->setMeshVerbosity(m_fcfg->getMeshVerbosity());
    146     m_ggeo->setMeshJoinCfg( resource->getMeshfix() );
    147 
    148     std::string meshversion = m_fcfg->getMeshVersion() ;;
    149     if(!meshversion.empty())
    150     {
    151         LOG(warning) << "OpticksGeometry::loadGeometry using debug meshversion " << meshversion ;
    152         m_ggeo->getGeoLib()->setMeshVersion(meshversion.c_str());
    153     }
    154 
    155     m_ggeo->loadGeometry();   // potentially from cache : for gltf > 0 loads both tri and ana geometry 
    156 
    157     if(m_ggeo->getMeshVerbosity() > 2)
    158     {
    159         GMergedMesh* mesh1 = m_ggeo->getMergedMesh(1);
    160         if(mesh1)
    161         {
    162             mesh1->dumpSolids("OpticksGeometry::loadGeometryBase mesh1");
    163             mesh1->save("$TMP", "GMergedMesh", "baseGeometry") ;
    164         }
    165     }
    166 
    167     LOG(error) << "OpticksGeometry::loadGeometryBase DONE " ;
    168     TIMER("loadGeometryBase");
    169 }




When running precache GGeo::init creates the various libs in 
preparation to be populated during the traverse.::

     336 void GGeo::init()
     337 {
     338    const char* idpath = m_ok->getIdPath() ;
     339    LOG(trace) << "GGeo::init"
     340               << " idpath " << ( idpath ? idpath : "NULL" )
     341               ;  
     342               
     343    assert(idpath && "GGeo::init idpath is required" );
     344    
     345    fs::path geocache(idpath);
     346    bool cache_exists = fs::exists(geocache) && fs::is_directory(geocache) ;
     347    bool cache_requested = m_ok->isGeocache() ; 
     348    
     349    m_loaded = cache_exists && cache_requested ;
     350    
     351    LOG(trace) << "GGeo::init"
     352              << " idpath " << idpath
     353              << " cache_exists " << cache_exists
     354              << " cache_requested " << cache_requested
     355              << " m_loaded " << m_loaded 
     356              ;
     357              
     358    if(m_loaded) return ;
     359    
     360    //////////////  below only when operating pre-cache //////////////////////////
     361    
     362    m_bndlib = new GBndLib(m_ok);
     363    m_materiallib = new GMaterialLib(m_ok);
     364    m_surfacelib  = new GSurfaceLib(m_ok);
     365    
     366    m_bndlib->setMaterialLib(m_materiallib);
     367    m_bndlib->setSurfaceLib(m_surfacelib);
     368    
     369    // NB this m_analytic is always false
     370    //    the analytic versions of these libs are born in GScene
     371    assert( m_analytic == false );  
     372    bool testgeo = false ;  
     373    
     374    m_meshlib = new GMeshLib(m_ok, m_analytic);
     375    m_geolib = new GGeoLib(m_ok, m_analytic, m_bndlib );
     376    m_nodelib = new GNodeLib(m_ok, m_analytic, testgeo );
     377    
     378    m_treecheck = new GTreeCheck(m_geolib, m_nodelib, m_ok->getSceneConfig() ) ;
     379    
     380    
     381    GColorizer::Style_t style = GColorizer::PSYCHEDELIC_NODE ;
     382    OpticksColors* colors = getColors();
     383    
     384    m_colorizer = new GColorizer( m_nodelib, m_geolib, m_bndlib, colors, style ); // colorizer needs full tree, so pre-cache only 
     385 
     386 
     387    m_scintillatorlib  = new GScintillatorLib(m_ok);
     388    m_sourcelib  = new GSourceLib(m_ok);
     389 
     390    m_pmtlib = NULL ;
     391 
     392    LOG(trace) << "GGeo::init DONE" ;
     393 }



::

     503 void GGeo::loadGeometry()
     504 {
     505     bool loaded = isLoaded() ;
     506 
     507     int gltf = m_ok->getGLTF();
     508 
     509     LOG(info) << "GGeo::loadGeometry START"
     510               << " loaded " << loaded
     511               << " gltf " << gltf
     512               ;
     513 
     514     if(!loaded)
     515     {
     516         loadFromG4DAE();
     517         save();
     518 
     519         if(gltf > 0 && gltf < 10)
     520         {
     521             loadAnalyticFromGLTF();
     522             saveAnalytic();
     523         }
     524     }
     525     else
     526     {
     527         loadFromCache();
     528         if(gltf > 0 && gltf < 10)
     529         {
     530             loadAnalyticFromCache();
     531         }
     532     }
     533 
     534 
     535     if(m_ok->isAnalyticPMTLoad())
     536     {
     537         m_pmtlib = GPmtLib::load(m_ok, m_bndlib );
     538     }
     539 
     540     if( gltf >= 10 )
     541     {
     542         LOG(info) << "GGeo::loadGeometry DEBUGGING loadAnalyticFromGLTF " ;
     543         loadAnalyticFromGLTF();
     544     }
     545 
     546     setupLookup();
     547     setupColors();
     548     setupTyp();
     549     LOG(info) << "GGeo::loadGeometry DONE" ;
     550 }



The current standard loader in the assimp loader.  


::


     552 void GGeo::loadFromG4DAE()
     553 {
     554     LOG(error) << "GGeo::loadFromG4DAE START" ;
     555 
     556     int rc = (*m_loader_imp)(this);   //  imp set in OpticksGeometry::loadGeometryBase, m_ggeo->setLoaderImp(&AssimpGGeo::load); 
     557 
     558     if(rc != 0)
     559         LOG(fatal) << "GGeo::loadFromG4DAE"
     560                    << " FAILED : probably you need to download opticksdata "
     561                    ;
     562 
     563     assert(rc == 0 && "G4DAE geometry file does not exist, try : opticksdata- ; opticksdata-- ") ;
     564 
     565     prepareScintillatorLib();
     566 
     567     prepareMeshes();
     568 
     569     prepareVertexColors();
     570 
     571     LOG(error) << "GGeo::loadFromG4DAE DONE" ;
     572 }


