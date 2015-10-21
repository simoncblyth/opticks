#include "App.hh"

#include <unistd.h>
#include <stdio.h>
#include <algorithm>

#include <boost/algorithm/string.hpp>  
#include <boost/lexical_cast.hpp>

#include "OptiXUtil.hh"
#include "define.h"

// oglrap-  Frame brings in GL/glew.h GLFW/glfw3.h gleq.h
#include "Frame.hh"

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


// numpyserver-
#ifdef NPYSERVER
#include "numpydelegate.hpp"
#include "numpydelegateCfg.hpp"
#include "numpyserver.hpp"
#endif

// npy-
#include "NPY.hpp"
#include "GLMPrint.hpp"
#include "GLMFormat.hpp"
#include "NumpyEvt.hpp"
#include "ViewNPY.hpp"
#include "MultiViewNPY.hpp"
#include "Lookup.hpp"
// #include "Sensor.hpp"
#include "G4StepNPY.hpp"
#include "TorchStepNPY.hpp"
#include "PhotonsNPY.hpp"
#include "HitsNPY.hpp"
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
#include "NSlice.hpp"

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
#include "GBoundaryLib.hh"
#include "GBoundaryLibMetadata.hh"
#include "GLoader.hh"
#include "GCache.hh"
#include "GMaterialIndex.hh"

// assimpwrap
#include "AssimpGGeo.hh"


// openmeshrap-
#include "MTool.hh"

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
#include "OContext.hh"
#include "OFrame.hh"
#include "ORenderer.hh"
#include "OGeo.hh"
#include "OBoundaryLib.hh"
#include "OBuf.hh"
#include "OConfig.hh"
#include "OTracer.hh"
#include "OPropagator.hh"
#include "cu/photon.h"

// optix-
#include <optixu/optixpp_namespace.h>



// cudawrap-
#include "CResource.hh"
#include "CBufSpec.hh"

// thrustrap-
#include "ThrustIdx.hh"
#include "ThrustHistogram.hh"
#include "ThrustArray.hh"
#include "TBuf.hh"
#include "TBufPair.hh"
#include "TSparse.hh"

#define GLMVEC4(g) glm::vec4((g).x,(g).y,(g).z,(g).w) 



bool App::hasOpt(const char* name)
{
    return m_fcfg->hasOpt(name);
}


void logging_init(const char* ldir, const char* lname, const char* level_)
{
   // see blogg-

    unsigned int ll = boost::log::trivial::info ;
 
    std::string level(level_);
    if(level.compare("trace") == 0) ll = boost::log::trivial::trace ; 
    if(level.compare("debug") == 0) ll = boost::log::trivial::debug ; 
    if(level.compare("info") == 0)  ll = boost::log::trivial::info ; 
    if(level.compare("warning") == 0)  ll = boost::log::trivial::warning ; 
    if(level.compare("error") == 0)  ll = boost::log::trivial::error ; 
    if(level.compare("fatal") == 0)  ll = boost::log::trivial::fatal ; 

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
        boost::log::trivial::severity >= ll
    );

    boost::log::add_common_attributes();

    boost::log::register_simple_formatter_factory< boost::log::trivial::severity_level, char >("Severity");  

    boost::log::add_console_log(
        std::cerr, 
        boost::log::keywords::format = "[%TimeStamp%]:%Severity%: %Message%",
        boost::log::keywords::auto_flush = true
    );  


    LOG(info) << "logging_init " << path ; 
}



void App::loggingConfig(int argc, char** argv)
{
    // full argument parsing is done in App::config, 
    // but logging setup needs to happen before that 

    const char* logname = "ggeoview.log" ; 
    const char* loglevel = "info" ; 

    for(unsigned int i=1 ; i < argc ; ++i )
    {
        if(strcmp(argv[i], "-G")==0)        logname = "ggeoview.nogeocache.log" ;
        if(strcmp(argv[i], "--trace")==0)   loglevel = "trace" ;
        if(strcmp(argv[i], "--debug")==0)   loglevel = "debug" ;
        if(strcmp(argv[i], "--info")==0)    loglevel = "info" ;
        if(strcmp(argv[i], "--warning")==0) loglevel = "warning" ;
        if(strcmp(argv[i], "--error")==0)   loglevel = "error" ;
        if(strcmp(argv[i], "--fatal")==0)   loglevel = "fatal" ;
    }

    // dont print anything here, it messes with --idp
    //printf(" logname: %s loglevel: %s\n", logname, loglevel );

    m_logname = strdup(logname);
    m_loglevel = strdup(loglevel);
}


void App::init(int argc, char** argv)
{
    loggingConfig(argc, argv);

    m_cache     = new GCache(m_prefix);

    const char* idpath = m_cache->getIdPath();
    logging_init(idpath, m_logname, m_loglevel);

    m_parameters = new Parameters ; 
    m_timer      = new Timer("main");
    m_timer->setVerbose(true);
    m_timer->start();

    // the envvars are normally not defined, using 
    // cmake configure_file values instead
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

    wiring();

    int rc = config(argc, argv) ;
    if(rc) m_exit = true ; 
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

    try { 
        m_server = new numpyserver<numpydelegate>(m_delegate); // connect to external messages 
    } 
    catch( const std::exception& e)
    {
        LOG(fatal) << "App::config EXCEPTION " << e.what() ; 
        LOG(fatal) << "App::config FAILED to instanciate numpyserver : probably another instance is running : check debugger sessions " ;
    }
#endif

    m_types = new Types ;  
    m_types->readFlags("$ENV_HOME/graphics/optixrap/cu/photon.h");  
    // TODO: avoid hardcoding (or at least duplication of) paths, grab via cmake configure_file ?
    //       here optixrap lib should have a compiled in constant for this
    //
    m_flags = m_types->getFlagsIndex(); 
    m_flags->setExt(".ini");
    m_flags->save(idpath);


    return 0 ; 
}


void App::prepareScene()
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

    (*m_timer)("prepareScene"); 
    LOG(info) << "App::prepareScene DONE ";
} 


void App::loadGeometry()
{
    // func pointer shenanigans allows GGeo to use AssimpWrap functionality 
    // without depending on assimpwrap- 
    //
    // BUT that also means that CMake dependency tracking 
    // will not do appropriate rebuilds, if get perplexing fails
    // try wiping and rebuilding assimpwrap- and ggeo-


    m_cache->setGeocache(!m_fcfg->hasOpt("nogeocache"));

    m_ggeo = new GGeo(m_cache);

    if(m_fcfg->hasOpt("qe1"))
    {
        GBoundaryLib* blib = m_ggeo->getBoundaryLib();
        blib->setFakeEfficiency(1.0);
    }


    m_ggeo->setLoaderImp(&AssimpGGeo::load);    // setting GLoaderImpFunctionPtr

    m_ggeo->setMeshJoinImp(&MTool::joinSplitUnion);

    m_ggeo->setMeshJoinCfg( m_cache->getMeshfix() );

    std::string meshversion = m_fcfg->getMeshVersion() ;;
    if(!meshversion.empty())
    {
        LOG(warning) << "App::loadGeometry using debug meshversion " << meshversion ;  
        m_ggeo->setMeshVersion(meshversion.c_str());
    }
    


    m_loader = new GLoader(m_ggeo) ;

    m_loader->setInstanced( !m_fcfg->hasOpt("noinstanced")  ); // find repeated geometry 
    m_loader->setRepeatIndex(m_fcfg->getRepeatIndex()); // --repeatidx
    m_loader->setTypes(m_types);
    m_loader->setCache(m_cache);

    m_loader->load();



    m_parameters->add<int>("repeatIdx", m_loader->getRepeatIndex() );

    GItemIndex* materials = m_loader->getMaterials();
    ////materials->dump("App::loadGeometry materials from m_loader");

    m_types->setMaterialsIndex(materials->getIndex());

    GBuffer* colorbuffer = m_loader->getColorBuffer();  // composite buffer 0+:materials,  32+:flags
    m_composition->setColorDomain( m_loader->getColorDomain() );
    m_scene->uploadColorBuffer(colorbuffer);   // oglrap-/Colors preps texture, available to shaders as "uniform sampler1D Colors"
   

    m_ggeo->dumpStats("App::loadGeometry");
    //m_ggeo->dumpTree("App::loadGeometry");

    for(unsigned int i=1 ; i < m_ggeo->getNumMergedMesh() ; i++)
        m_ggeo->dumpNodeInfo(i);


    checkGeometry();


    m_blib = m_ggeo->getBoundaryLib();
    m_lookup = m_loader->getMaterialLookup();
    m_meta = m_loader->getMetadata(); 
    m_boundaries =  m_meta->getBoundaryNames();
 
    m_composition->setTimeDomain( gfloat4(0.f, m_fcfg->getTimeMax(), m_fcfg->getAnimTimeMax(), 0.f) );  

    m_parameters->add<float>("timeMax",m_composition->getTimeDomain().y  ); 

    m_mesh0 = m_ggeo->getMergedMesh(0); 

   
    bool zexplode = m_fcfg->hasOpt("zexplode");
    if(zexplode)
    {
       // for --jdyb --idyb --kdyb testing : making the cleave OR the mend obvious
        glm::vec4 zexplodeconfig = gvec4(m_fcfg->getZExplodeConfig());
        print(zexplodeconfig, "zexplodeconfig");
        m_mesh0->explodeZVertices(zexplodeconfig.y, zexplodeconfig.x ); 
    }


    gfloat4 ce0 = m_mesh0->getCenterExtent(0);  // 0 : all geometry of the mesh, >0 : specific volumes
    m_composition->setDomainCenterExtent(ce0);  // define range in compressions etc.. 

    LOG(info) << "loadGeometry ce0: " 
                      << " x " << ce0.x
                      << " y " << ce0.y
                      << " z " << ce0.z
                      << " w " << ce0.w
                      ;
 
    (*m_timer)("loadGeometry"); 

    if(!m_cache->isGeocache())
    {
        LOG(info) << "App::loadGeometry early exit due to --nogeocache/-G option " ; 
        setExit(true); 
    }
}

void App::checkGeometry()
{
    if(m_ggeo->isLoaded())
    {
        LOG(info) << "App::checkGeometry needs to be done precache " ;
        return ; 
    }

    MTool mtool ; 

    unsigned int nso = m_ggeo->getNumSolids();
    unsigned int nme = m_ggeo->getNumMeshes();

    LOG(info) << "App::checkGeometry " 
              << " nso " << nso  
              << " nme " << nme 
              ; 

    typedef std::map<unsigned int, unsigned int> MUU ; 
    typedef MUU::const_iterator MUUI ; 

    typedef std::vector<unsigned int> VU ; 
    typedef std::map<unsigned int, VU > MUVU ; 

    MUU& mesh_usage = m_ggeo->getMeshUsage();
    MUVU& mesh_nodes = m_ggeo->getMeshNodes();

    for(MUUI it=mesh_usage.begin() ; it != mesh_usage.end() ; it++)
    {    
        unsigned int meshIndex = it->first ; 
        unsigned int nodeCount = it->second ; 

        VU& nodes = mesh_nodes[meshIndex] ;
        assert(nodes.size() == nodeCount );

        std::stringstream nss ; 
        for(unsigned int i=0 ; i < std::min( nodes.size(), 5ul ) ; i++) nss << nodes[i] << "," ;


        GMesh* mesh = m_ggeo->getMesh(meshIndex);
        gfloat4 ce = mesh->getCenterExtent(0);

        const char* shortName = mesh->getShortName();

        bool join = m_ggeo->shouldMeshJoin(mesh);

        unsigned int nv = mesh->getNumVertices() ; 
        unsigned int nf = mesh->getNumFaces() ; 
        unsigned int tc = mtool.countMeshComponents(mesh); // topological components

        assert( tc >= 1 );  // should be 1, some meshes have topological issues however

        std::string& out = mtool.getOut();
        std::string& err =  mtool.getErr();
        unsigned int noise = mtool.getNoise();

        const char* highlight = join ? "**" : "  " ; 

        bool dump = noise > 0 || tc > 1 || join ;
        //bool dump = true ; 

        if(dump)
            printf("  %4d (v%5d f%5d )%s(t%5d oe%5u) : x%10.3f : n%6d : n*v%7d : %40s : %s \n", meshIndex, nv, nf, highlight, tc, noise, ce.w, nodeCount, nodeCount*nv, shortName, nss.str().c_str() );

        if(noise > 0)
        {
            if(out.size() > 0 ) LOG(debug) << "out " << out ;  
            if(err.size() > 0 ) LOG(debug) << "err " << err ;  
        }

   }    


    for(MUUI it=mesh_usage.begin() ; it != mesh_usage.end() ; it++)
    {
        unsigned int meshIndex = it->first ; 
        GMesh* mesh = m_ggeo->getMesh(meshIndex);
        bool join = m_ggeo->shouldMeshJoin(mesh);
        if(join)
        {
             mesh->Summary("App::checkGeometry joined mesh");
        }
    }


}



void App::uploadGeometry()
{
    m_scene->setGeometry(m_ggeo);
    m_scene->uploadGeometry();
    bool autocam = true ; 

    // handle commandline --target option that needs loaded geometry 
    unsigned int target = m_scene->getTargetDeferred();   // default to 0 
    LOG(info) << "App::uploadGeometry setting target " << target ; 

    m_scene->setTarget(target, autocam);
 
    (*m_timer)("uploadGeometry"); 
}


TorchStepNPY* App::makeSimpleTorchStep()
{
    TorchStepNPY* torchstep = new TorchStepNPY(TORCH, 1);

    std::string config = m_fcfg->getTorchConfig() ;
    if(!config.empty()) torchstep->configure(config.c_str());

    m_ggeo->targetTorchStep(torchstep);

    bool verbose = true ; 
    torchstep->addStep(verbose);  // copyies above configured step settings into the NPY and increments the step index, ready for configuring the next step 

    return torchstep ; 
}


TorchStepNPY* App::makeCalibrationTorchStep(unsigned int imesh)
{
    assert(0);
    // TODO: need way to get the volume indices of the instances for this to work.. 

    assert(imesh > 0);
    GMergedMesh* mmi = m_ggeo->getMergedMesh(imesh);
    unsigned int nti = mmi->getNumTransforms();

    // need to restrict to same AD instances, simply dividing by 2 doesnt work
    // TODO: make targetted instance ranges or lists configurable 

    LOG(info) << "App::makeCalibrationTorchStep " 
              <<  " imesh " << imesh 
              <<  " nti " << nti 
              ; 

    TorchStepNPY* torchstep = new TorchStepNPY(TORCH, nti);
    std::string config = m_fcfg->getTorchConfig() ;
    if(!config.empty()) torchstep->configure(config.c_str());

    torchstep->setNumPhotons(100);
    torchstep->setRadius(100);


    for(unsigned int i=0 ; i < nti ; i++)
    {
        torchstep->setFrame(i);  // this needs the volume index 
 
        m_ggeo->targetTorchStep(torchstep); // uses above set frame index to set the frame transform

        torchstep->addStep(); 

        // copyies above configured step settings into the NPY and increments the step index, ready for configuring the next step 
    }
 
    return torchstep ; 
}


void App::loadGenstep()
{
    if(hasOpt("nooptix|noevent")) 
    {
        LOG(warning) << "App::loadGenstep skip due to --nooptix/--noevent " ;
        return ;
    }
    unsigned int code ; 
    if(     m_fcfg->hasOpt("cerenkov"))      code = CERENKOV ;
    else if(m_fcfg->hasOpt("scintillation")) code = SCINTILLATION ;
    else if(m_fcfg->hasOpt("torch"))         code = TORCH ;
    else                                     code = TORCH ;

    std::string typ = photon_enum_label(code) ; 
    boost::algorithm::to_lower(typ);
    std::string tag = m_fcfg->getEventTag();
    if(tag.empty()) tag = "1" ; 

    std::string det = m_cache->getDetector();

    m_parameters->add<std::string>("Type", typ );
    m_parameters->add<std::string>("Tag", tag );
    m_parameters->add<std::string>("Detector", det );


    NPY<float>* npy = NULL ; 
    if( code == CERENKOV || code == SCINTILLATION )
    {
      

        npy = loadGenstepFromFile(typ, tag, det ); 

        m_g4step = new G4StepNPY(npy);    
        m_g4step->relabel(code); // becomes the ghead.i.x used in cu/generate.cu

        if(m_cache->isDayabay())
        {   
            m_g4step->setLookup(m_loader->getMaterialLookup()); 
            m_g4step->applyLookup(0, 2);      
            // translate materialIndex (1st quad, 3rd number) from chroma to GGeo 
            m_parameters->add<std::string>("genstepAfterLookup",   npy->getDigestString()  );
        }
    }
    else if(code == TORCH)
    {
        m_torchstep = makeSimpleTorchStep();
        //m_torchstep = makeCalibrationTorchStep(1);
        npy = m_torchstep->getNPY(); 
    }
    

    (*m_timer)("loadGenstep"); 

 

    m_evt->setMaxRec(m_fcfg->getRecordMax());          // must set this before setGenStepData to have effect

    bool nooptix    = m_fcfg->hasOpt("nooptix");
    bool geocenter  = m_fcfg->hasOpt("geocenter");

    m_evt->setOptix(!nooptix);
    m_evt->setAllocate(false);   

    m_evt->setGenstepData(npy);         // CAUTION : KNOCK ON ALLOCATES FOR PHOTONS AND RECORDS  

    (*m_timer)("hostEvtAllocation"); 




    glm::vec4 mmce = GLMVEC4(m_mesh0->getCenterExtent(0)) ;
    glm::vec4 gsce = (*m_evt)["genstep.vpos"]->getCenterExtent();
    glm::vec4 uuce = geocenter ? mmce : gsce ;

    print(mmce, "loadGenstep mmce");
    print(gsce, "loadGenstep gsce");
    print(uuce, "loadGenstep uuce");

    if(m_scene->getTarget() == 0)
    {
        // only pointing based in genstep if not already targetted
        bool autocam = true ; 
        m_composition->setCenterExtent( uuce , autocam );
    }


    m_scene->setRecordStyle( m_fcfg->hasOpt("alt") ? Scene::ALTREC : Scene::REC );    

    m_parameters->add<unsigned int>("NumGensteps", m_evt->getNumGensteps());
    m_parameters->add<unsigned int>("NumPhotons", m_evt->getNumPhotons());
    m_parameters->add<unsigned int>("NumRecords", m_evt->getNumRecords());

}


NPY<float>* App::loadGenstepFromFile(const std::string& typ, const std::string& tag, const std::string& det)
{
    LOG(info) << "App::loadGenstepFromFile  " 
              << " typ " << typ
              << " tag " << tag 
              << " det " << det
              ; 

    NPY<float>* npy = NPY<float>::load(typ.c_str(), tag.c_str(), det.c_str() ) ;

    m_parameters->add<std::string>("genstepAsLoaded",   npy->getDigestString()  );

    int modulo = m_fcfg->getModulo();
    m_parameters->add<int>("Modulo", modulo );

    if(modulo > 0)
    {
        LOG(warning) << "App::loadGenstepFromFile applying modulo scaledown " << modulo ; 
        npy = NPY<float>::make_modulo(npy, modulo); 
        m_parameters->add<std::string>("genstepModulo",   npy->getDigestString()  );
    }
    return npy ; 
}






void App::uploadEvt()
{
    if(hasOpt("nooptix|noevent")) 
    {
        LOG(warning) << "App::uploadEvt skip due to --nooptix/--noevent " ;
        return ;
    }
 
    m_composition->update();

    m_scene->uploadAxis();

    m_scene->uploadEvt();  // Scene, Rdr uploads orchestrated by NumpyEvt/MultiViewNPY

    m_scene->uploadSelection();   // recsel upload

    (*m_timer)("uploadEvt"); 
}



void App::seedPhotonsFromGensteps()
{
    //  Distributes unsigned int genstep indices 0:m_num_gensteps-1 into the first 
    //  4 bytes of the 4*float4 photon record in the photon buffer 
    //  using the number of photons per genstep obtained from the genstep buffer 
    //  
    //  Note that this is done almost entirely on the GPU, only the num_photons reduction
    //  needs to come back to CPU in order to allocate an appropriately sized OptiX photon 
    //  buffer on GPU.
    //  
    //  This per-photon genstep index is used by OptiX photon propagation 
    //  program cu/generate.cu to access the appropriate values from the genstep buffer
    //
    //
    //  TODO: make this operational in COMPUTE as well as INTEROP modes without code duplication ?
    //


    LOG(info)<<"App::seedPhotonsFromGensteps" ;

    NPY<float>* gensteps =  m_evt->getGenstepData() ;

    NPY<float>* photons  =  m_evt->getPhotonData() ;    // NB has no allocation and "uploaded" with glBufferData NULL

    unsigned int nv0 = gensteps->getNumValues(0) ; 

    // interop specific, but the result of the mapping 
    // hmm... does OptiX expose buffer id ? 
    // 
    // this is done prior to OptiX involvement in INTEROP mode
    // hmm could keep that in COMPUTE mode by 
    // creating buffers in Thrust and then doing gloptixthrust- CUDAToOptiX setDevicePointer 
    //

    CResource rgs( gensteps->getBufferId(), CResource::R );
    CResource rph( photons->getBufferId(), CResource::RW );

    TBuf tgs("tgs", rgs.mapGLToCUDA<unsigned int>() );
    TBuf tph("tph", rph.mapGLToCUDA<unsigned int>() );
    
    //tgs.dump<unsigned int>("App::seedPhotonsFromGensteps tgs", 6*4, 3, nv0 ); // stride, begin, end 

    unsigned int num_photons = tgs.reduce<unsigned int>(6*4, 3, nv0 );

    assert(num_photons == m_evt->getNumPhotons() && "FATAL : mismatch between CPU and GPU photon counts from the gensteps") ;   

    CBufSlice src = tgs.slice(6*4,3,nv0) ;
    CBufSlice dst = tph.slice(4*4,0,num_photons*4*4) ;

    TBufPair<unsigned int> tgp(src, dst);
    tgp.seedDestination();

    rgs.unmapGLToCUDA(); 
    rph.unmapGLToCUDA(); 

    (*m_timer)("seedPhotonsFromGensteps"); 

    // below approach in OEngine failed due to getting bad pointer from OptiX::  
    //
    //     OBuf gs("gs", m_genstep_buffer);
    //     unsigned int num_photons = gs.reduce<unsigned int>(6*4, 3) ;  // stride, offset
    //     assert(num_photons == m_evt->getNumPhotons());
    //     LOG(info)<<"OEngine::seedPhotonsFromGensteps num_photons " << num_photons ;
    //
    //     OBuf ph("ph", m_photon_buffer) ;
    //     OBufPair<unsigned int> bp(gs.slice(6*4,3,0), ph.slice(4*4,0,0));
    //     bp.seedDestination();
    //
}

void App::initRecords()
{
    // TODO: find an OpenGL way to zero a VBO, here resort to CUDA
    LOG(info)<<"App::initRecords" ;

    NPY<short>* records =  m_evt->getRecordData() ;
    CResource rec( records->getBufferId(), CResource::W );

    TBuf trec("trec", rec.mapGLToCUDA<short>() );
    trec.zero();

    rec.unmapGLToCUDA(); 

    (*m_timer)("initRecords"); 
}


void App::configureGeometry()
{
    int restrict_mesh = m_fcfg->getRestrictMesh() ;  
    int analytic_mesh = m_fcfg->getAnalyticMesh() ; 
    std::string islice = m_fcfg->getISlice() ;;
    NSlice* slice = !islice.empty() ? new NSlice(islice.c_str()) : NULL ; 

    unsigned int nmm = m_ggeo->getNumMergedMesh();
    for(unsigned int i=0 ; i < nmm ; i++)
    {
        GMergedMesh* mm = m_ggeo->getMergedMesh(i);
        if(restrict_mesh > -1 && i != restrict_mesh ) mm->setGeoCode('K');      
        if(analytic_mesh > -1 && i == analytic_mesh && i > 0) mm->setGeoCode('S');      
        if(i>0) mm->setSlice(slice);
    }
}

void App::prepareOptiX()
{
    bool compute  = m_fcfg->hasOpt("compute"); 
    int  debugidx = m_fcfg->getDebugIdx();
    int  stack    = m_fcfg->getStack();

    LOG(info) << "App::prepareOptiX stack " << stack ;  

    OContext::Mode_t mode = compute ? OContext::COMPUTE : OContext::INTEROP ; 

    assert( mode == OContext::INTEROP && "COMPUTE mode not operational"); 

    optix::Context context = optix::Context::create();

    m_ocontext = new OContext(context, mode); 

    m_ocontext->setStackSize(stack);

    m_ocontext->setDebugPhoton(debugidx);

    m_olib = new OBoundaryLib(context,m_ggeo->getBoundaryLib());

    m_olib->convert(); 

    std::string builder_   = m_fcfg->getBuilder(); 
    std::string traverser_ = m_fcfg->getTraverser(); 
    const char* builder   = builder_.empty() ? NULL : builder_.c_str() ;
    const char* traverser = traverser_.empty() ? NULL : traverser_.c_str() ;

    m_ogeo = new OGeo(m_ocontext, m_ggeo, builder, traverser);

    m_ogeo->setTop(m_ocontext->getTop());

    m_ogeo->convert(); 

    unsigned int width  = m_composition->getPixelWidth();
    unsigned int height = m_composition->getPixelHeight();
    m_oframe = new OFrame(context, width, height );
    context["output_buffer"]->set( m_oframe->getOutputBuffer() );

    m_interactor->setTouchable(m_oframe);

    m_orenderer = new ORenderer(m_oframe, m_scene->getShaderDir(), m_scene->getShaderInclPath());

    m_otracer = new OTracer(m_ocontext, m_composition);

    LOG(info) << m_ogeo->description("App::prepareOptiX ogeo");

    (*m_timer)("prepareOptiX"); 
    LOG(info) << "App::prepareOptiX DONE "; 

    m_ocontext->dump("App::prepareOptiX");
}

void App::preparePropagator()
{
    bool noevent    = m_fcfg->hasOpt("noevent");
    bool trivial    = m_fcfg->hasOpt("trivial");
    int  override   = m_fcfg->getOverride();

    assert(!noevent);

    m_opropagator = new OPropagator(m_ocontext, m_composition);

    m_opropagator->setBounceMax(m_fcfg->getBounceMax());  // 0:prevents any propagation leaving generated photons
    m_opropagator->setRecordMax(m_evt->getMaxRec());       // 1:to minimize without breaking machinery 

    m_opropagator->setNumpyEvt(m_evt);

    m_opropagator->setTrivial(trivial);
    m_opropagator->setOverride(override);

    int rng_max = getenvint("CUDAWRAP_RNG_MAX",-1);   // TODO: avoid envvar 
    assert(rng_max >= 1e6); 

    m_opropagator->setRngMax(rng_max);

    m_parameters->add<unsigned int>("RngMax",    m_opropagator->getRngMax() );
    m_parameters->add<unsigned int>("BounceMax", m_opropagator->getBounceMax() );
    m_parameters->add<unsigned int>("RecordMax", m_opropagator->getRecordMax() );

    m_opropagator->initRng();
    m_opropagator->initEvent();

    (*m_timer)("preparePropagator"); 
    LOG(info) << "App::preparePropagator DONE "; 
}






void App::propagate()
{
    if(hasOpt("nooptix|noevent|nopropagate")) 
    {
        LOG(warning) << "App::propagate skip due to --nooptix/--noevent/--nopropagate " ;
        return ;
    }

    LOG(info)<< " ******************* (main) OptiXEngine::generate + propagate  *********************** " ;

    m_opropagator->propagate();     

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

    NPY<short>* daux = m_evt->getAuxData();
    Rdr::download(daux);


    (*m_timer)("evtDownload"); 

    m_parameters->add<std::string>("photonData",   dpho->getDigestString()  );
    m_parameters->add<std::string>("recordData",   drec->getDigestString()  );
    m_parameters->add<std::string>("sequenceData", dhis->getDigestString()  );
    m_parameters->add<std::string>("auxData",      daux->getDigestString()  );

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
    daux->setVerbose();
    daux->save("au%s", typ,  tag, det);


    (*m_timer)("evtSave"); 
}


void App::indexSequence()
{
    if(!m_evt) return ; 

    OBuf* seq = m_opropagator->getSequenceBuf();

    // NB hostside allocation deferred for these
    NPY<unsigned char>* phosel_data = m_evt->getPhoselData(); 
    NPY<unsigned char>* recsel_data = m_evt->getRecselData();
    unsigned int maxrec = m_evt->getMaxRec(); // 10 

    // note the layering here, this pattern will hopefully facilitate moving 
    // from OpenGL backed to OptiX backed for COMPUTE mode
    // although this isnt a good example as indexSequence is not 
    // necessary in COMPUTE mode 

    CResource rphosel( phosel_data->getBufferId(), CResource::W );
    CResource rrecsel( recsel_data->getBufferId(), CResource::W );
    {
        TBuf tphosel("tphosel", rphosel.mapGLToCUDA<unsigned char>() );
        tphosel.zero();
        TBuf trecsel("trecsel", rrecsel.mapGLToCUDA<unsigned char>() );

#ifdef DEBUG
        unsigned int nphosel = tphosel.getSize() ; 
        unsigned int npsd = std::min(nphosel,100u) ;
        unsigned int nsqa = seq->getNumAtoms(); 
        unsigned int nsqd = std::min(nsqa,100u); 
        assert(nphosel == 2*nseqa);
        unsigned int nrecsel = trecsel.getSize() ; 
        assert(nrecsel == maxrec*2*nseqa);
        LOG(info) << "App::indexSequence "
                  << " nsqa (2*num_photons)" << nsqa 
                  << " nphosel " << nphosel
                  << " nrecsel " << nrecsel
                  ; 
#endif

        TSparse<unsigned long long> seqhis("History Sequence", seq->slice(2,0)); // stride,begin 
        seqhis.make_lookup();
        m_seqhis = new GItemIndex(seqhis.getIndex()) ;  
        seqhis.apply_lookup<unsigned char>( tphosel.slice(4,0));  // stride, begin

#ifdef DEBUG
        seq->dump<unsigned long long>("App::indexSequence OBuf seq.dump", 2, 0, nsqd);
        tphosel.dumpint<unsigned char>("tphosel.dumpint<unsigned char>(4,0)", 4,0, npsd) ;
        seqhis.dump("App::indexSequence seqhis");
#endif

        TSparse<unsigned long long> seqmat("Material Sequence", seq->slice(2,1)); // stride,begin 
        seqmat.make_lookup();
        m_seqmat = new GItemIndex(seqmat.getIndex()) ;  
        seqmat.apply_lookup<unsigned char>( tphosel.slice(4,1));

#ifdef DEBUG
        seq->dump<unsigned long long>("App::indexSequence OBuf seq.dump", 2, 1, nsqd);
        seqmat.dump("App::indexSequence seqmat");
        tphosel.dumpint<unsigned char>("tphosel.dumpint<unsigned char>(4,1)", 4,1, npsd) ;
#endif

        tphosel.repeat_to<unsigned char>( &trecsel, 4, 0, tphosel.getSize(), maxrec );  // other, stride, begin, end, repeats

#ifdef DEBUG
        tphosel.download<unsigned char>( phosel_data );  // cudaMemcpyDeviceToHost
        phosel_data->save("/tmp/phosel.npy");  
        trecsel.download<unsigned char>( recsel_data );
        recsel_data->save("/tmp/recsel.npy");  
#endif

    }
    rphosel.unmapGLToCUDA(); 
    rrecsel.unmapGLToCUDA(); 

    m_seqhis->setTitle("Photon Flag Sequence Selection");
    m_seqhis->setTypes(m_types);
    m_seqhis->setLabeller(GItemIndex::HISTORYSEQ);
    m_seqhis->formTable();

    m_seqmat->setTitle("Photon Material Sequence Selection");
    m_seqmat->setTypes(m_types);
    m_seqmat->setLabeller(GItemIndex::MATERIALSEQ);
    m_seqmat->formTable();

    (*m_timer)("indexSequence"); 
}


void App::indexBoundaries()
{
    if(!m_evt) return ; 

    OBuf* pho = m_opropagator->getPhotonBuf();


    pho->setHexDump(false);

/*
    unsigned int npha = pho->getNumAtoms(); 
    unsigned int nphd  = std::min(npha,4*4*100u); 
    pho->dump<int>("App::indexBoundaries pho->dump<int>", 4*4, 4*3+0, nphd);
*/

    TSparse<int> boundaries("Boundaries", pho->slice(4*4,4*3+0)); // stride,begin 
    boundaries.setHexDump(false);
    boundaries.make_lookup();
    //boundaries.dump("App::indexBoundaries");


    (*m_timer)("indexBoundaries"); 


    NPY<float>* dpho = m_evt->getPhotonData();

    if(dpho->hasData())
    {
        // host based indexing of unique material codes, requires downloadEvt to pull back the photon data
        LOG(info) << "App::indexBoundaries host based " ;
        m_bnd = new BoundariesNPY(dpho); 
        m_bnd->setTypes(m_types);
        m_bnd->setBoundaryNames(m_boundaries); // map<int,string>
        m_bnd->indexBoundaries();     
    } 

    (*m_timer)("indexBoundariesOld"); 


/*

App::indexBoundaries : num_unique 31 
               0         0      ## hmm this is an empty slot ...
              18    179870
               0    158960
              13     56510
            ....
             -55        29
              20        26
              22         1

Comparing with np the dump matches except for the zero slot artifact::

    In [1]: p = np.load("/usr/local/env/dayabay/oxcerenkov/1.npy")

    In [2]: b = p[:,3,0].view(np.int32)

    In [3]: c = count_unique(b)

    In [5]: c[c[:,1].argsort()[::-1]]
    Out[5]: 
    array([[    18, 179870],
           [     0, 158960],
           [    13,  56510],
           ...
           [   -55,     29],
           [    20,     26],
           [    22,      1]])

*/

}


void App::indexEvt()
{

   /*

       INTEROP mode GPU buffer access C:create R:read W:write
       ----------------------------------------------------------

                     OpenGL     OptiX              Thrust 

       gensteps       CR       R (gen/prop)       R (seeding)

       photons        CR       W (gen/prop)       W (seeding)
       sequence                W (gen/prop)
       phosel         CR                          W (indexing) 

       records        CR       W  
       recsel         CR                          W (indexing)


       OptiX has no business with phosel and recsel 
   */

    if(!m_evt) return ; 
   
    indexSequence();

    indexBoundaries();


    (*m_timer)("indexEvt"); 
}


void App::indexEvtOld()
{
    if(!m_evt) return ; 

    NPY<float>* dpho = m_evt->getPhotonData();

    if(dpho->hasData())
    {
        m_pho = new PhotonsNPY(dpho);   // a detailed photon/record dumper : looks good for photon level debug 
        m_pho->setTypes(m_types);

        m_hit = new HitsNPY(dpho, m_ggeo->getSensorList());
        m_hit->debugdump();

    }

    // hmm thus belongs in NumpyEvt rather than here
    NPY<short>* drec = m_evt->getRecordData();

    if(drec->hasData())
    {
        m_rec = new RecordsNPY(drec, m_evt->getMaxRec());
        m_rec->setTypes(m_types);
        m_rec->setDomains((NPY<float>*)m_opropagator->getDomain());

        if(m_pho)
        {
            m_pho->setRecs(m_rec);
            if(m_torchstep) m_torchstep->dump("App::indexEvtOld TorchStepNPY");

            m_pho->dump(0  ,  "App::indexEvtOld dpho 0");
            m_pho->dump(100,  "App::indexEvtOld dpho 100" );
            m_pho->dump(1000, "App::indexEvtOld dpho 1000" );

        }
        m_evt->setRecordsNPY(m_rec);
        m_evt->setPhotonsNPY(m_pho);
    }

    (*m_timer)("indexEvtOld"); 
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
 
    m_photons = new Photons(m_types, m_pho, m_bnd, m_seqhis, m_seqmat ) ; // GUI jacket : m_pho seems unused 
    m_scene->setPhotons(m_photons);


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

void App::render()
{
    m_frame->viewport();
    m_frame->clear();

#ifdef OPTIX
    if(m_interactor->getOptiXMode()>0 && m_otracer && m_orenderer)
    { 
        unsigned int scale = m_interactor->getOptiXResolutionScale() ; 
        m_otracer->setResolutionScale(scale) ;
        m_otracer->trace();
        LOG(info) << m_ogeo->description("App::render ogeo");

        m_orenderer->render();
    }
    else
#endif
    {
        m_scene->render();
    }

#ifdef GUI_
    m_gui->newframe();
    bool* show_gui_window = m_interactor->getGuiModeAddress();
    if(*show_gui_window)
    {
        m_gui->show(show_gui_window);
        if(m_photons)
        {
            if(m_bnd)
            {
                glm::ivec4 sel = m_bnd->getSelection() ;
                m_composition->setSelection(sel); 
                m_composition->getPick().y = sel.x ;   //  1st boundary 
            }
            glm::ivec4& recsel = m_composition->getRecSelect();
            recsel.x = m_seqhis ? m_seqhis->getSelected() : 0 ; 
            recsel.y = m_seqmat ? m_seqmat->getSelected() : 0 ; 
            m_composition->setFlags(m_types->getFlags()); 
        }
        // maybe imgui edit selection within the composition imgui, rather than shovelling ?
        // BUT: composition feeds into shader uniforms which could be reused by multiple classes ?
    }
    m_gui->render();
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
        count = m_composition->tick();

        if( m_composition->hasChanged() || m_interactor->hasChanged() || count == 1)  
        {
            render();
            glfwSwapBuffers(m_window);

            m_interactor->setChanged(false);  
            m_composition->setChanged(false);   // sets camera, view, trackball dirty status 
        }
    }
}



void App::cleanup()
{

#ifdef OPTIX
    if(m_ocontext)
    {
        m_ocontext->cleanUp();
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



