#include <stdlib.h>  //exit()
#include <stdio.h>

#include "OptiXUtil.hh"
#include "define.h"

// oglrap-  Frame brings in GL/glew.h GLFW/glfw3.h gleq.h
#include "Frame.hh"

// ggeoview-
//#define INTEROP 1
#ifdef INTEROP
#include "CUDAInterop.hh"
#endif

#define OPTIX 1

// oglrap-
#define GUI_ 1
#ifdef GUI_
#include "GUI.hh"
#endif

#include "FrameCfg.hh"
#include "Scene.hh"
#include "SceneCfg.hh"
#include "Renderer.hh"
#include "RendererCfg.hh"
#include "Interactor.hh"
#include "InteractorCfg.hh"

#include "Bookmarks.hh"
#include "Composition.hh"
#include "Rdr.hh"
#include "Texture.hh"
#include "Photons.hh"
#include "DynamicDefine.hh"


// TODO numpyserver-
#ifdef NPYSERVER
#include "numpydelegate.hpp"
#include "numpydelegateCfg.hpp"
#include "numpyserver.hpp"
#endif

// npy-
#include "NPY.hpp"
#include "GLMPrint.hpp"
#include "NumpyEvt.hpp"
#include "ViewNPY.hpp"
#include "MultiViewNPY.hpp"
#include "Lookup.hpp"
// #include "Sensor.hpp"
#include "G4StepNPY.hpp"
#include "PhotonsNPY.hpp"
#include "RecordsNPY.hpp"
#include "BoundariesNPY.hpp"
#include "SequenceNPY.hpp"
#include "Types.hpp"
#include "Index.hpp"
#include "stringutil.hpp"

#include "Timer.hpp"
#include "Times.hpp"
#include "Parameters.hpp"
#include "Report.hpp"

// bregex-
#include "regexsearch.hh"

// glm-
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// ggeo-
#include "GGeo.hh"
#include "GMergedMesh.hh"
#include "GBoundaryLibMetadata.hh"
#include "GLoader.hh"
#include "GCache.hh"
#include "GMaterialIndex.hh"

// assimpwrap
#include "AssimpGGeo.hh"



#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;


#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/trivial.hpp>
//#include <boost/log/utility/setup/file.hpp>
#include "boost/log/utility/setup.hpp"
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal

// optixrap-
#include "OEngine.hh"
#include "RayTraceConfig.hh"

// thrustrap-
#include "ThrustIdx.hh"
#include "ThrustHistogram.hh"
#include "ThrustArray.hh"

#define GLMVEC4(g) glm::vec4((g).x,(g).y,(g).z,(g).w) 


void dump(float* f, const char* msg)
{
    if(!f) return ;

    printf("%s\n", msg);
    for(unsigned int i=0 ; i < 16 ; i++)
    {   
        if(i%4 == 0) printf("\n");
        printf(" %10.4f ", f[i] );
    }   
    printf("\n");
}


void logging_init(const char* ldir, const char* lname)
{
   // see blogg-

    fs::path logdir(ldir);
    if(!fs::exists(logdir))
    {
        if (fs::create_directories(logdir))
        {
            printf("logging_init: created directory %s \n", ldir) ;
        }
    }

    fs::path logpath(logdir / lname );

    const char* path = logpath.string().c_str(); 

    boost::log::add_file_log(path);

    boost::log::core::get()->set_filter
    (
        boost::log::trivial::severity >= boost::log::trivial::info
    );

    boost::log::add_console_log(
        std::cerr, 
        boost::log::keywords::format = "[%TimeStamp%]: %Message%",
        boost::log::keywords::auto_flush = true
    );  

    boost::log::add_common_attributes();

    LOG(info) << "logging_init " << path ; 
}



class App {
  public:
       App(const char* prefix="GGEOVIEW_", const char* logname="ggeoview.log");
  private:
       void init();
       void wiring();
  public:
       int config(int argc, char** argv);
       void prepareContext();
  public:
       int  loadGeometry();
       void uploadGeometry();
  public:
       void loadGenstep();
       void uploadEvt();
  public:
       void prepareEngine();
       void propagate();
       void downloadEvt();
       void indexEvt();
  public:
       void prepareGUI();
       void makeReport();
       void renderLoop();
       void cleanup();

  public:
       GCache* getCache(); 
       FrameCfg<Frame>* getFrameCfg();
       bool hasOpt(const char* name);

  private:
       const char*  m_prefix ; 
       Parameters*  m_parameters ; 
       Timer*       m_timer ; 
       GCache*      m_cache ; 
       Scene*       m_scene ; 
       Composition* m_composition ;
       Frame*       m_frame ;
       GLFWwindow*  m_window ; 
       Bookmarks*   m_bookmarks ;
       Interactor*  m_interactor ;
#ifdef NPYSERVER
       numpydelegate* m_delegate ; 
       numpyserver<numpydelegate>* m_server ;
#endif
       NumpyEvt*        m_evt ;
       Cfg*             m_cfg ;
       FrameCfg<Frame>* m_fcfg ; 
       Types*           m_types ; 
       Index*           m_flags ; 
       GLoader*         m_loader ; 
       GGeo*            m_ggeo ; 
       GBoundaryLib*    m_blib ; 
       GBoundaryLibMetadata*    m_meta; 
       GMergedMesh*     m_mesh0 ;  
       Lookup*          m_lookup ;
       OEngine*         m_engine ; 
       BoundariesNPY*   m_bnd ; 
       PhotonsNPY*      m_pho ; 
       RecordsNPY*      m_rec ; 
       GItemIndex*      m_seqhis ; 
       GItemIndex*      m_seqmat ; 
       Photons*         m_photons ; 
       GUI*             m_gui ; 
   private:
       std::map<int, std::string> m_boundaries ;
       glm::uvec4       m_size ;
       bool             m_gpu_resident_evt ; 

};



App::App(const char* prefix, const char* logname)
   : 
      m_prefix(strdup(prefix)),
      m_parameters(NULL),
      m_timer(NULL),
      m_cache(NULL),
      m_scene(NULL),
      m_composition(NULL),
      m_frame(NULL),
      m_window(NULL),
      m_bookmarks(NULL),
      m_interactor(NULL),
#ifdef NPYSERVER
      m_delegate(NULL),
      m_server(NULL),
#endif
      m_evt(NULL), 
      m_cfg(NULL),
      m_fcfg(NULL),
      m_types(NULL),
      m_flags(NULL),
      m_loader(NULL),
      m_ggeo(NULL),
      m_blib(NULL),
      m_meta(NULL),
      m_mesh0(NULL),
      m_lookup(NULL),
      m_engine(NULL),
      m_bnd(NULL),
      m_pho(NULL),
      m_rec(NULL),
      m_seqhis(NULL),
      m_seqmat(NULL),
      m_photons(NULL),
      m_gui(NULL),
      m_gpu_resident_evt(true)
{
    m_cache     = new GCache(m_prefix);

    const char* idpath = m_cache->getIdPath();
    logging_init(idpath, logname);

    init();
    wiring();
}

GCache* App::getCache()
{
    return m_cache ; 
}


FrameCfg<Frame>* App::getFrameCfg()
{
    return m_fcfg ; 
}

bool App::hasOpt(const char* name)
{
    return m_fcfg->hasOpt(name);
}



void App::init()
{
    m_parameters = new Parameters ; 
    m_timer      = new Timer("main");
    m_timer->setVerbose(true);
    m_timer->start();


    const char* shader_dir = getenv("SHADER_DIR"); 
    const char* shader_incl_path = getenv("SHADER_INCL_PATH"); 
    const char* shader_dynamic_dir = getenv("SHADER_DYNAMIC_DIR"); 
    // dynamic define for use by GLSL shaders

    m_scene      = new Scene(shader_dir, shader_incl_path, shader_dynamic_dir ) ;

 
    m_composition = new Composition ; 
    m_frame       = new Frame ; 
    m_bookmarks   = new Bookmarks ; 
    m_interactor  = new Interactor ; 
#ifdef NPYSERVER
    m_delegate    = new numpydelegate ; 
#endif
}


void App::wiring()
{
    m_interactor->setFrame(m_frame);
    m_interactor->setScene(m_scene);
    m_interactor->setComposition(m_composition);
    m_interactor->setBookmarks(m_bookmarks);

    m_composition->setScene(m_scene);

    m_bookmarks->setComposition(m_composition);
    m_bookmarks->setScene(m_scene);


    m_frame->setInteractor(m_interactor);      
    m_frame->setComposition(m_composition);
    m_frame->setScene(m_scene);
}


int App::config(int argc, char** argv)
{
    m_cfg  = new Cfg("unbrella", false) ; 
    m_fcfg = new FrameCfg<Frame>("frame", m_frame,false);
    m_cfg->add(m_fcfg);
#ifdef NPYSERVER
    m_cfg->add(new numpydelegateCfg<numpydelegate>("numpydelegate", m_delegate, false));
#endif

    m_cfg->add(new SceneCfg<Scene>(           "scene",       m_scene,                      true));
    m_cfg->add(new RendererCfg<Renderer>(     "renderer",    m_scene->getGeometryRenderer(), true));
    m_cfg->add(new InteractorCfg<Interactor>( "interactor",  m_interactor,                 true));

    m_composition->addConfig(m_cfg); 

    m_cfg->commandline(argc, argv);

    const std::string cmdline = m_cfg->getCommandLine(); 
    m_timer->setCommandLine(cmdline);
    LOG(info) << argv[0] << " " << cmdline ; 

    const char* idpath = m_cache->getIdPath();

    if(m_fcfg->hasOpt("idpath")) std::cout << idpath << std::endl ;
    if(m_fcfg->hasOpt("help"))   std::cout << m_cfg->getDesc()     << std::endl ;

    if(m_fcfg->isAbort()) return 1 ; 

    bool fullscreen = m_fcfg->hasOpt("fullscreen");

    if(m_fcfg->hasOpt("size")) m_size = m_frame->getSize() ;
    else if(fullscreen)        m_size = glm::uvec4(2880,1800,2,0) ;
    else                       m_size = glm::uvec4(2880,1704,2,0) ;  // 1800-44-44px native height of menubar  

    m_composition->setSize( m_size );

    m_bookmarks->load(idpath); 
    m_frame->setTitle("GGeoView");
    m_frame->setFullscreen(fullscreen);

    m_evt  = hasOpt("noevent") ? NULL : new NumpyEvt ; 
    m_scene->setNumpyEvt(m_evt);

#ifdef NPYSERVER
    m_delegate->liveConnect(m_cfg); // hookup live config via UDP messages
    m_delegate->setNumpyEvt(m_evt); // allows delegate to update evt when NPY messages arrive
    m_server = new numpyserver<numpydelegate>(m_delegate); // connect to external messages 
#endif

    m_types = new Types ;  
    m_types->readFlags("$ENV_HOME/graphics/ggeoview/cu/photon.h");
    m_flags = m_types->getFlagsIndex(); 
    m_flags->setExt(".ini");
    //m_flags->save("/tmp");


    return 0 ; 
}


void App::prepareContext()
{
    DynamicDefine dd ;
    dd.add("MAXREC",m_fcfg->getRecordMax());    
    dd.add("MAXTIME",m_fcfg->getTimeMax());    
    dd.add("PNUMQUAD", 4);  // quads per photon
    dd.add("RNUMQUAD", 2);  // quads per record 


    m_scene->write(&dd);

    m_scene->initRenderers();  // reading shader source and creating renderers

    m_frame->init();  // creates OpenGL context
    m_window = m_frame->getWindow();

    m_scene->setComposition(m_composition);     // defer until renderers are setup 

    (*m_timer)("prepareContext"); 
    LOG(info) << "App::prepareScene DONE ";
} 


int App::loadGeometry()
{
    m_loader = new GLoader ;
    m_loader->setInstanced(true); // find repeated geometry 
    m_loader->setRepeatIndex(m_fcfg->getRepeatIndex()); // --repeatidx
    m_loader->setTypes(m_types);
    m_loader->setCache(m_cache);
    m_loader->setImp(&AssimpGGeo::load);    // setting GLoaderImpFunctionPtr

    bool nogeocache = m_fcfg->hasOpt("nogeocache");
    m_loader->load(nogeocache);

    m_parameters->add<int>("repeatIdx", m_loader->getRepeatIndex() );


    GItemIndex* materials = m_loader->getMaterials();
    m_types->setMaterialsIndex(materials->getIndex());

    GBuffer* colorbuffer = m_loader->getColorBuffer();  // composite buffer 0+:materials,  32+:flags
    m_scene->uploadColorBuffer(colorbuffer);   // oglrap-/Colors preps texture, available to shaders as "uniform sampler1D Colors"
   
    m_ggeo = m_loader->getGGeo();
    m_blib = m_loader->getBoundaryLib();
    m_lookup = m_loader->getMaterialLookup();
    m_meta = m_loader->getMetadata(); 
    m_boundaries =  m_meta->getBoundaryNames();
 
    m_composition->setTimeDomain( gfloat4(0.f, m_fcfg->getTimeMax(), 0.f, 0.f) );  
    m_composition->setColorDomain( gfloat4(0.f, colorbuffer->getNumItems(), 0.f, 0.f));

    m_parameters->add<float>("timeMax",m_composition->getTimeDomain().y  ); 

    m_mesh0 = m_ggeo->getMergedMesh(0); 
    assert(m_mesh0->getTransformsBuffer() == NULL && "expecting first mesh to be global, not instanced");

    gfloat4 ce0 = m_mesh0->getCenterExtent(0);  // 0 : all geometry of the mesh, >0 : specific volumes
    m_composition->setDomainCenterExtent(ce0);  // define range in compressions etc.. 

    LOG(info) << "loadGeometry ce0: " 
                      << " x " << ce0.x
                      << " y " << ce0.y
                      << " z " << ce0.z
                      << " w " << ce0.w
                      ;
 
    (*m_timer)("loadGeometry"); 

    if(nogeocache){
        LOG(info) << "App::loadGeometry early exit due to --nogeocache/-G option " ; 
        return 1 ; 
    }
    return 0 ; 
}


void App::uploadGeometry()
{
    m_scene->setGeometry(m_ggeo);
    m_scene->uploadGeometry();
    bool autocam = true ; 
    m_scene->setTarget(0, autocam);
 
    (*m_timer)("uploadGeometry"); 
}


void App::loadGenstep()
{
    if(hasOpt("nooptix|noevent")) 
    {
        LOG(warning) << "App::loadGenstep skip due to --nooptix/--noevent " ;
        return ;
    }

    const char* typ ; 
    if(     m_fcfg->hasOpt("cerenkov"))      typ = "cerenkov" ;
    else if(m_fcfg->hasOpt("scintillation")) typ = "scintillation" ;
    else                                     typ = "cerenkov" ;

    std::string tag_ = m_fcfg->getEventTag();
    const char* tag = tag_.empty() ? "1" : tag_.c_str()  ; 

    const char* det = m_cache->getDetector();
    bool juno       = m_cache->isJuno();
    
    int modulo = m_fcfg->getModulo();

    m_parameters->add<std::string>("Type", typ );
    m_parameters->add<std::string>("Tag", tag );
    m_parameters->add<std::string>("Detector", det );
    m_parameters->add<int>("Modulo", modulo );

    NPY<float>* npy = NPY<float>::load(typ, tag, det ) ;
    m_parameters->add<std::string>("genstepAsLoaded",   npy->getDigestString()  );
   
    if(modulo > 0)
    {
        LOG(warning) << "App::loadGenstep applying modulo scaledown " << modulo ; 
        npy = NPY<float>::make_modulo(npy, modulo); 
        m_parameters->add<std::string>("genstepModulo",   npy->getDigestString()  );
    }

    (*m_timer)("loadGenstep"); 

    G4StepNPY genstep(npy);    
    genstep.setLookup(m_loader->getMaterialLookup()); 
   
    if(juno)
    {
        LOG(warning) << "App::loadGenstep skip genstep.applyLookup for JUNO " ;
    }
    else
    {   
        genstep.applyLookup(0, 2);   // translate materialIndex (1st quad, 3rd number) from chroma to GGeo 
    }
    (*m_timer)("applyLookup"); 
 
    m_parameters->add<std::string>("genstepAfterLookup",   npy->getDigestString()  );

    m_evt->setMaxRec(m_fcfg->getRecordMax());          // must set this before setGenStepData to have effect

    bool nooptix    = m_fcfg->hasOpt("nooptix");
    bool geocenter  = m_fcfg->hasOpt("geocenter");

    m_evt->setOptix(!nooptix);
    m_evt->setAllocate(!m_gpu_resident_evt); // switch off host allocation when "GPU residency" is enabled

    m_evt->setGenstepData(npy);         // CAUTION : KNOCK ON ALLOCATES FOR PHOTONS AND RECORDS  

    (*m_timer)("hostEvtAllocation"); 

    glm::vec4 mmce = GLMVEC4(m_mesh0->getCenterExtent(0)) ;
    glm::vec4 gsce = (*m_evt)["genstep.vpos"]->getCenterExtent();
    glm::vec4 uuce = geocenter ? mmce : gsce ;
    print(mmce, "loadGenstep mmce");
    print(gsce, "loadGenstep gsce");
    print(uuce, "loadGenstep uuce");

    bool autocam = true ; 
    m_composition->setCenterExtent( uuce , autocam );

    m_scene->setRecordStyle( m_fcfg->hasOpt("alt") ? Scene::ALTREC : Scene::REC );    

    m_parameters->add<unsigned int>("NumGensteps", m_evt->getNumGensteps());
    m_parameters->add<unsigned int>("NumPhotons", m_evt->getNumPhotons());
    m_parameters->add<unsigned int>("NumRecords", m_evt->getNumRecords());

}


void App::uploadEvt()
{
    if(hasOpt("nooptix|noevent")) 
    {
        LOG(warning) << "App::uploadEvt skip due to --nooptix/--noevent " ;
        return ;
    }








 
#ifdef INTEROP
    // signal Rdr to use GL_DYNAMIC_DRAW
    CUDAInterop<unsigned char>* c_psel = new CUDAInterop<unsigned char>(m_evt->getPhoselData());
    CUDAInterop<unsigned char>* c_rsel = new CUDAInterop<unsigned char>(m_evt->getRecselData());
#endif

    m_composition->update();
    //m_composition.dumpAxisData("main:dumpAxisData");
    m_scene->uploadAxis();

    m_scene->uploadEvt();  // Scene, Rdr uploads orchestrated by NumpyEvt/MultiViewNPY

    (*m_timer)("uploadEvt"); 
    LOG(info) << "main: scene.uploadEvt DONE "; 

#ifdef INTEROP
    // recsel handled separately to the rest due to interop complications
    // as it needs Thrust indexing 
    m_scene->uploadSelection();  
    //c_psel->registerBuffer();
    //c_rsel->registerBuffer();
#else
    // non-interop workaround: defer uploadSelection until after indexing 
#endif

}


void App::prepareEngine()
{
    bool compute    = m_fcfg->hasOpt("compute"); 
    bool nooptix    = m_fcfg->hasOpt("nooptix");
    bool noevent    = m_fcfg->hasOpt("noevent");
    bool trivial    = m_fcfg->hasOpt("trivial");
    int  override   = m_fcfg->getOverride();
    int  debugidx   = m_fcfg->getDebugIdx();

    OEngine::Mode_t mode = compute ? OEngine::COMPUTE : OEngine::INTEROP ; 

    const char* idpath = m_cache->getIdPath();

    assert( mode == OEngine::INTEROP && "OEngine::COMPUTE mode not operational"); 
    m_engine = new OEngine("GGeoView", mode) ;       
    m_interactor->setTouchable(m_engine);

    m_engine->setFilename(idpath);
    m_engine->setOverride(override);
    m_engine->setDebugPhoton(debugidx);
    m_engine->setGGeo(m_ggeo);   
    m_engine->setBoundaryLib(m_blib);   
    m_engine->setNumpyEvt(noevent ? NULL : m_evt);
    m_engine->setComposition(m_composition);                 
    m_engine->setEnabled(!nooptix);
    m_engine->setTrivial(trivial);

    if(!noevent)
    {
        m_engine->setBounceMax(m_fcfg->getBounceMax());  // 0:prevents any propagation leaving generated photons
        m_engine->setRecordMax(m_evt->getMaxRec());       // 1:to minimize without breaking machinery 
        int rng_max = getenvint("CUDAWRAP_RNG_MAX",-1);
        assert(rng_max >= 1e6); 
        m_engine->setRngMax(rng_max);

        m_parameters->add<unsigned int>("RngMax",    m_engine->getRngMax() );
        m_parameters->add<unsigned int>("BounceMax", m_engine->getBounceMax() );
        m_parameters->add<unsigned int>("RecordMax", m_engine->getRecordMax() );
    }

    LOG(info)<< " ******************* main.OptiXEngine::init creating OptiX context, when enabled *********************** " ;
    m_engine->init();  
    m_engine->initRenderer(m_scene->getShaderDir(), m_scene->getShaderInclPath());

    (*m_timer)("initOptiX"); 
    LOG(info) << "App::prepareEngine DONE "; 
}


void App::propagate()
{
    if(hasOpt("nooptix|noevent|nopropagate")) 
    {
        LOG(warning) << "App::propagate skip due to --nooptix/--noevent/--nopropagate " ;
        return ;
    }

    LOG(info)<< " ******************* (main) OptiXEngine::generate + propagate  *********************** " ;
    m_engine->generate();     
    (*m_timer)("propagate"); 
}



void App::downloadEvt()
{
    if(!m_evt) return ; 

    NPY<float>* dpho = m_evt->getPhotonData();
    Rdr::download(dpho);

    NPY<short>* drec = m_evt->getRecordData();
    Rdr::download(drec);

    NPY<unsigned long long>* dhis = m_evt->getSequenceData();
    Rdr::download(dhis);

    (*m_timer)("evtDownload"); 

    m_parameters->add<std::string>("photonData",   dpho->getDigestString()  );
    m_parameters->add<std::string>("recordData",   drec->getDigestString()  );
    m_parameters->add<std::string>("sequenceData", dhis->getDigestString()  );

    (*m_timer)("checkDigests"); 

    const char* typ = m_parameters->getStringValue("Type").c_str();
    const char* tag = m_parameters->getStringValue("Tag").c_str();
    const char* det = m_parameters->getStringValue("Detector").c_str();

    // app.saveEvt
    dpho->setVerbose();
    dpho->save("ox%s", typ,  tag, det);
    drec->setVerbose();
    drec->save("rx%s", typ,  tag, det);
    dhis->setVerbose();
    dhis->save("ph%s", typ,  tag, det);

    (*m_timer)("evtSave"); 
}



void App::indexEvt()
{
    if(!m_evt) return ; 
    NPY<float>* dpho = m_evt->getPhotonData();
    m_bnd = new BoundariesNPY(dpho); 

    m_bnd->setTypes(m_types);
    m_bnd->setBoundaryNames(m_boundaries);
    m_bnd->indexBoundaries();

    m_pho = new PhotonsNPY(dpho);
    m_pho->setTypes(m_types);


    NPY<short>* drec = m_evt->getRecordData();

    m_rec = new RecordsNPY(drec, m_evt->getMaxRec());
    m_rec->setTypes(m_types);
    NPYBase* domain = m_engine->getDomain(); 
    m_rec->setDomains((NPY<float>*)domain);

    (*m_timer)("boundaryIndex"); 


    LOG(warning) << "main: hardcode noindex as not working" ;
    bool noindex = true ; 
    if(!noindex)
    {
            optix::Buffer& sequence_buffer = m_engine->getSequenceBuffer() ;
            unsigned int num_elements = OptiXUtil::getBufferSize1D( sequence_buffer );  assert(num_elements == 2*m_evt->getNumPhotons());
            unsigned int device_number = 0 ;  // maybe problem with multi-GPU
            unsigned long long* d_seqn = OptiXUtil::getDevicePtr<unsigned long long>( sequence_buffer, device_number ); 
#ifdef INTEROP
            // attempt to give CUDA access to mapped OpenGL buffer
            //unsigned char*      d_psel = c_psel->GL_to_CUDA();
            //unsigned char*      d_rsel = c_rsel->GL_to_CUDA();
            // attempt to give CUDA access to OptiX buffers which in turn are connected to OpenGL buffers
            unsigned char* d_psel = OptiXUtil::getDevicePtr<unsigned char>( m_engine->getPhoselBuffer(), device_number ); 
            unsigned char* d_rsel = OptiXUtil::getDevicePtr<unsigned char>( m_engine->getRecselBuffer(), device_number ); 
#else
            LOG(info)<< "main: non interop allocating new device buffers with ThrustArray " ;
            unsigned char*      d_psel = NULL ;    
            unsigned char*      d_rsel = NULL ;    
#endif

            unsigned int sequence_itemsize = m_evt->getSequenceData()->getShape(2) ; assert( 2 == sequence_itemsize );
            unsigned int phosel_itemsize   = m_evt->getPhoselData()->getShape(2)   ; assert( 4 == phosel_itemsize );
            unsigned int recsel_itemsize   = m_evt->getRecselData()->getShape(2)   ; assert( 4 == recsel_itemsize );
            unsigned int maxrec = m_evt->getMaxRec();
         
            LOG(info) << "main: ThrustIndex ctor " 
                      << " num_elements " << num_elements 
                      << " sequence_itemsize " << sequence_itemsize 
                      << " phosel_itemsize " << phosel_itemsize 
                      << " recsel_itemsize " << recsel_itemsize 
                      ; 
            ThrustArray<unsigned long long> pseq(d_seqn, num_elements       , sequence_itemsize );   // input flag/material sequences
            ThrustArray<unsigned char>      psel(d_psel, num_elements       , phosel_itemsize   );   // output photon selection
            ThrustArray<unsigned char>      rsel(d_rsel, num_elements*maxrec, recsel_itemsize   );   // output record selection

            ThrustIdx<unsigned long long, unsigned char> idx(&psel, &pseq);

            idx.makeHistogram(0, "FlagSequence");   
            idx.makeHistogram(1, "MaterialSequence");   

            psel.repeat_to( maxrec, rsel );
            cudaDeviceSynchronize();

            (*m_timer)("sequenceIndex"); 

#ifdef INTEROP
            // declare that CUDA finished with buffers 
            c_rsel->CUDA_to_GL();
            c_psel->CUDA_to_GL();
#else
            // non-interop workaround download the Thrust created buffers into NPY, then copy them back to GPU with uploadSelection
            psel.download( m_evt->getPhoselData() );  
            rsel.download( m_evt->getRecselData() ); 

            //m_evt->getPhoselData()->save("phosel_%s", typ, tag, det);
            //m_evt->getRecselData()->save("recsel_%s", typ, tag, det);

            m_scene->uploadSelection();                 // upload NPY into OpenGL buffer, duplicating recsel on GPU

            (*m_timer)("selectionDownloadUpload"); 
#endif

            m_seqhis = new GItemIndex(idx.getHistogramIndex(0)) ;  
            m_seqmat = new GItemIndex(idx.getHistogramIndex(1)) ;  
            //seqhis->save(idpath);
            //seqmat->save(idpath);

            m_seqhis->setTitle("Photon Flag Sequence Selection");
            m_seqhis->setTypes(m_types);
            m_seqhis->setLabeller(GItemIndex::HISTORYSEQ);
            m_seqhis->formTable();

            m_seqmat->setTitle("Photon Material Sequence Selection");
            m_seqmat->setTypes(m_types);
            m_seqmat->setLabeller(GItemIndex::MATERIALSEQ);
            m_seqmat->formTable();
    
            m_photons = new Photons(m_pho, m_bnd, m_seqhis, m_seqmat ) ; // GUI jacket 
        }


    m_scene->setPhotons(m_photons);
}



void App::makeReport()
{
    m_timer->stop();

    m_parameters->dump();
    m_timer->dump();

    Report r ; 
    r.add(m_parameters->getLines()); 
    r.add(m_timer->getLines()); 

    const char* typ = m_parameters->getStringValue("Type").c_str();
    const char* tag = m_parameters->getStringValue("Tag").c_str();
    //const char* det = m_parameters->getStringValue("Detector").c_str();

    Times* ts = m_timer->getTimes();
    ts->save("$IDPATH/times", Times::name(typ, tag).c_str());

    char rdir[128];
    snprintf(rdir, 128, "$IDPATH/report/%s/%s", tag, typ ); 
    r.save(rdir, Report::name(typ, tag).c_str());  // with timestamp prefix


}




void App::prepareGUI()
{

#ifdef GUI_
    m_gui = new GUI ;
    m_gui->setScene(m_scene);
    m_gui->setPhotons(m_photons);
    m_gui->setComposition(m_composition);
    m_gui->setBookmarks(m_bookmarks);
    m_gui->setInteractor(m_interactor);   // status line
    m_gui->setLoader(m_loader);           // access to Material / Surface indices
    
    m_gui->init(m_window);
    m_gui->setupHelpText( m_cfg->getDescString() );

    // TODO: use GItemIndex ? for stats to make it persistable
    m_gui->setupStats(m_timer->getStats());
    m_gui->setupParams(m_parameters->getLines());

#endif

}

void App::renderLoop()
{
    bool noviz      = m_fcfg->hasOpt("noviz");
    if(noviz)
    {
        LOG(info) << "ggeoview/main.cc early exit due to --noviz/-V option " ; 
        return ;
    }
    LOG(info) << "enter runloop "; 


    bool* show_gui_window = m_interactor->getGuiModeAddress();
    glm::ivec4& recsel = m_composition->getRecSelect();

    //m_frame->toggleFullscreen(true); causing blankscreen then segv
    m_frame->hintVisible(true);
    m_frame->show();
    LOG(info) << "after frame.show() "; 

    unsigned int count ; 

    while (!glfwWindowShouldClose(m_window))
    {
        m_frame->listen(); 
#ifdef NPYSERVER
        m_server->poll_one();  
#endif
        m_frame->render();

#ifdef OPTIX
        if(m_interactor->getOptiXMode()>0)
        { 
             m_engine->trace();
             m_engine->render();
        }
        else
#endif
        {
            m_scene->render();
        }

        count = m_composition->tick();

#ifdef GUI_
        m_gui->newframe();
        if(*show_gui_window)
        {
            m_gui->show(show_gui_window);

            if(m_photons)
            {
                glm::ivec4 sel = m_photons->getBoundaries()->getSelection() ;
                m_composition->setSelection(sel); 
                m_composition->getPick().y = sel.x ;   //  1st boundary 

                recsel.x = m_seqhis ? m_seqhis->getSelected() : 0 ; 
                recsel.y = m_seqmat ? m_seqmat->getSelected() : 0 ; 

                m_composition->setFlags(m_types->getFlags()); 
            }
            // maybe imgui edit selection within the composition imgui, rather than shovelling ?
            // BUT: composition feeds into shader uniforms which could be reused by multiple classes ?
        }
        m_gui->render();
#endif

        glfwSwapBuffers(m_window);
    }
}



void App::cleanup()
{

#ifdef OPTIX
    if(m_engine)
    {
        m_engine->cleanUp();
    }
#endif
#ifdef NPYSERVER
    if(m_server)
    {
        m_server->stop();
    }
#endif
#ifdef GUI_
    if(m_gui)
    {
        m_gui->shutdown();
    }
#endif
    if(m_frame)
    {
        m_frame->exit();
    }
}





int main(int argc, char** argv)
{
    const char* logname = "ggeoview.log" ; 
    for(unsigned int i=1; i < argc ; i++)
    {
       if( strcmp(argv[i],"-G")==0  || strcmp(argv[i],"--nogeocache")==0) logname = "ggeoview.nogeocache.log" ; 
    } 

    int rc ; 

    App app("GGEOVIEW_", logname); 

    rc = app.config(argc, argv) ;
    if(rc) exit(EXIT_SUCCESS);

    app.prepareContext();

    rc = app.loadGeometry();
    if(rc) exit(EXIT_SUCCESS);

    app.uploadGeometry();

    bool nooptix = app.hasOpt("nooptix");
    if(!nooptix)
    {

        app.loadGenstep();

        app.uploadEvt();    // allocates GPU buffers with OpenGL glBufferData

        app.prepareEngine();

        app.propagate();

        app.downloadEvt();

        app.indexEvt();

        app.makeReport();
    }

    app.prepareGUI();

    app.renderLoop();

    app.cleanup();


    exit(EXIT_SUCCESS);

}



