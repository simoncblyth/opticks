Opticks Launch Review
=======================

High level review of the launching of an Opticks executable
such as "OKTest" identifying the packages and primary 
classes driving the action.


Issues 
--------

* OpticksResource (resident of Opticks) currently requires a geometry source .dae file 
  to exist (opticksdata) 

  * with new geometry that will not be the case, so currently an unrelated
    other geometry needs to fill the spot before the export can be done
    (historically export was done separately : as so it was always available)
    
    * so Opticks cannot currently run without opticksdata (even when it doesnt need to use it)

* to change from the dayabay default required envvar gymnastics 
  
* OpticksResource assumes a small number of harcoded geometries for naming purposes
 

Fix? How ? 
~~~~~~~~~~~

* split off specific geometry aspects of OpticksResource (the installpath related setup, eg installcache, can remain) 
  into a separate class that gets instanciated later (perhaps with OpticksHub?)

  * started this by pulling BResource out of BOpticksResource


OKTest
--------

OKTest::

    OKMgr ok(argc, argv);
    ok.propagate();         // m_run, m_propagator
    ok.visualize();         // m_viz


OKMgr instanciation
------------------------

::

    OKMgr::OKMgr(int argc, char** argv, const char* argforced ) 
        :
        m_log(new SLog("OKMgr::OKMgr")),
        m_ok(new Opticks(argc, argv, argforced)),         
        m_hub(new OpticksHub(m_ok)),            // immediate configure and loadGeometry 
        m_idx(new OpticksIdx(m_hub)),
        m_num_event(m_ok->getMultiEvent()),     // after hub instanciation, as that configures Opticks
        m_gen(m_hub->getGen()),
        m_run(m_hub->getRun()),
        m_viz(m_ok->isCompute() ? NULL : new OpticksViz(m_hub, m_idx, true)),
        m_propagator(new OKPropagator(m_hub, m_idx, m_viz)),
        m_count(0)
    {
        init();
        (*m_log)("DONE");
    }


okc.Opticks::Opticks
----------------------

::

     410 void Opticks::init()
     411 {
     ...
     426     m_resource = new OpticksResource(this, m_envprefix, m_lastarg);
     427 
     428     setDetector( m_resource->getDetector() );
     ...
     431 }
    

Opticks validity from m_resource::

    1821 bool Opticks::isValid() {   return m_resource->isValid(); }
  
     // setInvalid is called when no daepath 

::

      77 OpticksResource::OpticksResource(Opticks* opticks, const char* envprefix, const char* lastarg)
      78     :
      79        BOpticksResource(envprefix),
      80        m_opticks(opticks),
      81        m_lastarg(lastarg ? strdup(lastarg) : NULL),
      82 
      83        m_geokey(NULL),
      84 
      85        m_query_string(NULL),

::

     254 void OpticksResource::init()
     255 {
     256    LOG(trace) << "OpticksResource::init" ;
     257 
     258    BStr::split(m_detector_types, "GScintillatorLib,GMaterialLib,GSurfaceLib,GBndLib,GSourceLib", ',' );
     259    BStr::split(m_resource_types, "GFlags,OpticksColors", ',' );
     260 
     261    readG4Environment();
     262    readOpticksEnvironment();
     263    readEnvironment();
     264 
     265    readMetadata();
     266    identifyGeometry();   // this assumes a small number hardcoded geometries 
     267    assignDetectorName();
     268    assignDefaultMaterial();
     269 
     270    LOG(trace) << "OpticksResource::init DONE" ;
     271 }
     272 


OpticksResource::readEnvironment asserts when no daepath is configured via the envvar mechanism::

     496 void OpticksResource::readEnvironment()
     497 {
     ...
     521     m_geokey = SSys::getenvvar(m_envprefix, "GEOKEY", DEFAULT_GEOKEY);
     522     const char* daepath = SSys::getenvvar(m_geokey);
     ...
     565     assert(daepath);
     566 
     567     setupViaSrc(daepath, query_digest.c_str());  // this sets m_idbase, m_idfold, m_idname done in base BOpticksResource
     568 
     569     assert(m_idpath) ;
     570     assert(m_idname) ;
     571     assert(m_idfold) ;
     572 }


     



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


     



