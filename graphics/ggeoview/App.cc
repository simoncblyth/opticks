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

#include "OpticksCfg.hh"
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
#include "NLog.hpp"
#include "NPY.hpp"
#include "GLMPrint.hpp"
#include "GLMFormat.hpp"
#include "NumpyEvt.hpp"
#include "ViewNPY.hpp"
#include "MultiViewNPY.hpp"
#include "Lookup.hpp"
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

//opticks-
#include "Opticks.hh"
#include "OpticksCfg.hh"
#include "OpticksPhoton.h"

// glm-
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// ggeo-
#include "GCache.hh"
#include "GGeo.hh"
#include "GMergedMesh.hh"

#include "GGeoLib.hh"
#include "GBndLib.hh"
#include "GMaterialLib.hh"
#include "GSurfaceLib.hh"
#include "GFlags.hh"

#include "GColors.hh"
#include "GItemIndex.hh"
#include "GAttrSeq.hh"
#include "GTestBox.hh"


// assimpwrap
#include "AssimpGGeo.hh"


// openmeshrap-
#include "MTool.hh"

#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;


// optixrap-
#include "OContext.hh"
#include "OColors.hh"
#include "OFrame.hh"
#include "ORenderer.hh"
#include "OGeo.hh"
#include "OBndLib.hh"
#include "OScintillatorLib.hh"
#include "OSourceLib.hh"
#include "OBuf.hh"
#include "OConfig.hh"
#include "OTracer.hh"
#include "OPropagator.hh"


//#include "cu/photon.h"

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


void App::init(int argc, char** argv)
{
    m_opticks = new Opticks(); 

    m_cache     = new GCache(m_prefix, "ggeoview.log", "info");
    m_cache->configure(argc, argv);  // logging setup needs to happen before below general config

    m_parameters = new Parameters ; 
    m_timer      = new Timer("App::");
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
    for(unsigned int i=1 ; i < argc ; i++ ) LOG(debug) << "App::config " << "[" << std::setw(2) << i << "]" << argv[i] ;

    m_cfg  = new Cfg("umbrella", false) ; 

    // TODO: extracate configuration (a very low dependency thing that needs to be highly portable, eg for use from cfg4-)
    //       from high dependency oglrap-/Frame and other OpenGL level objects
    //
    //       config and oglrap- objects got entangled in order to support live UDP config of high 
    //       level objects via boost bind 
    //
    //       need a way to disentangle whilst retaining this messaging functionality 
    //       split up config options according to need (eg low level things that 
    //       apply wherever file tag etc.. and those that apply to high level objects)
    //

    m_fcfg = new OpticksCfg<Opticks>("opticks", m_opticks,false);

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

    LOG(debug) << argv[0] << " " << cmdline ; 

    const char* idpath = m_cache->getIdPath();

    if(m_fcfg->hasOpt("idpath")) std::cout << idpath << std::endl ;
    if(m_fcfg->hasOpt("help"))   std::cout << m_cfg->getDesc()     << std::endl ;
    if(m_fcfg->hasOpt("help|version|idpath")) return 1 ; 

    bool fullscreen = m_fcfg->hasOpt("fullscreen");

    if(m_fcfg->hasOpt("size")) m_size = m_frame->getSize() ;
    else if(fullscreen)        m_size = glm::uvec4(2880,1800,2,0) ;
    else                       m_size = glm::uvec4(2880,1704,2,0) ;  // 1800-44-44px native height of menubar  


    m_composition->setSize( m_size );

    m_bookmarks->load(idpath); 
    m_frame->setTitle("GGeoView");
    m_frame->setFullscreen(fullscreen);

    m_dd = new DynamicDefine();   // configuration used in oglrap- shaders
    m_dd->add("MAXREC",m_fcfg->getRecordMax());    
    m_dd->add("MAXTIME",m_fcfg->getTimeMax());    
    m_dd->add("PNUMQUAD", 4);  // quads per photon
    m_dd->add("RNUMQUAD", 2);  // quads per record 
    m_dd->add("MATERIAL_COLOR_OFFSET", (unsigned int)GColors::MATERIAL_COLOR_OFFSET );
    m_dd->add("FLAG_COLOR_OFFSET", (unsigned int)GColors::FLAG_COLOR_OFFSET );
    m_dd->add("PSYCHEDELIC_COLOR_OFFSET", (unsigned int)GColors::PSYCHEDELIC_COLOR_OFFSET );
    // TODO: add spectral colors in wavelength bins to color texture

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

    return 0 ; 
}


void App::prepareScene()
{

    m_scene->write(m_dd);

    m_scene->initRenderers();  // reading shader source and creating renderers

    m_frame->init();  // creates OpenGL context

    m_window = m_frame->getWindow();

    m_scene->setComposition(m_composition);     // defer until renderers are setup 

    (*m_timer)("prepareScene"); 
    LOG(debug) << "App::prepareScene DONE ";
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
    m_cache->setInstanced( !m_fcfg->hasOpt("noinstanced")  ); // find repeated geometry 

    m_ggeo = new GGeo(m_cache);

    if(m_fcfg->hasOpt("qe1"))
        m_ggeo->getSurfaceLib()->setFakeEfficiency(1.0);

    m_ggeo->setLoaderImp(&AssimpGGeo::load);    // setting GLoaderImpFunctionPtr
    m_ggeo->setMeshJoinImp(&MTool::joinSplitUnion);
    m_ggeo->setMeshJoinCfg( m_cache->getMeshfix() );

    std::string meshversion = m_fcfg->getMeshVersion() ;;
    if(!meshversion.empty())
    {
        LOG(warning) << "App::loadGeometry using debug meshversion " << meshversion ;  
        m_ggeo->getGeoLib()->setMeshVersion(meshversion.c_str());
    }

    m_ggeo->loadGeometry();

    (*m_timer)("loadGeometry"); 

    if(m_fcfg->hasOpt("test"))
    {
        std::string testconf = m_fcfg->getTestConfig();
        m_ggeo->modifyGeometry( testconf.empty() ? NULL : testconf.c_str() );
        (*m_timer)("modifyGeometry"); 
    }

    checkGeometry();

    registerGeometry();

    if(!m_cache->isGeocache())
    {
        LOG(info) << "App::loadGeometry early exit due to --nogeocache/-G option " ; 
        setExit(true); 
    }
}



void App::registerGeometry()
{
    // TODO: replace these with equivalents from GPropertyLib subclasses
    //Index* matidx = materials->getIndex() ;
    //m_cache->getTypes()->setMaterialsIndex(matidx); 

    ////////////////////////////////////////////////////////

    GColors* colors = m_cache->getColors();

    m_composition->setColorDomain( colors->getCompositeDomain() ); 

    m_scene->uploadColorBuffer( colors->getCompositeBuffer() );  //     oglrap-/Colors preps texture, available to shaders as "uniform sampler1D Colors"

    //m_ggeo->dumpStats("App::registerGeometry dumpStats");
    //m_ggeo->dumpTree("App::registerGeometry");

    for(unsigned int i=1 ; i < m_ggeo->getNumMergedMesh() ; i++) m_ggeo->dumpNodeInfo(i);
    m_mesh0 = m_ggeo->getMergedMesh(0); 

 
    m_composition->setTimeDomain( gfloat4(0.f, m_fcfg->getTimeMax(), m_fcfg->getAnimTimeMax(), 0.f) );  

    m_parameters->add<float>("timeMax",m_composition->getTimeDomain().y  ); 
  
    gfloat4 ce0 = m_mesh0->getCenterExtent(0);  // 0 : all geometry of the mesh, >0 : specific volumes

    m_composition->setDomainCenterExtent(ce0);  // define range in compressions etc.. 

    LOG(debug) << "App::registerGeometry ce0: " 
                      << " x " << ce0.x
                      << " y " << ce0.y
                      << " z " << ce0.z
                      << " w " << ce0.w
                      ;
 
}

void App::checkGeometry()
{
    if(m_ggeo->isLoaded())
    {
        LOG(debug) << "App::checkGeometry needs to be done precache " ;
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


 
    bool zexplode = m_fcfg->hasOpt("zexplode");
    if(zexplode)
    {
       // for --jdyb --idyb --kdyb testing : making the cleave OR the mend obvious
        glm::vec4 zexplodeconfig = gvec4(m_fcfg->getZExplodeConfig());
        print(zexplodeconfig, "zexplodeconfig");

        GMergedMesh* mesh0 = m_ggeo->getMergedMesh(0);
        mesh0->explodeZVertices(zexplodeconfig.y, zexplodeconfig.x ); 
    }
}



void App::uploadGeometry()
{
    m_scene->setGeometry(m_ggeo);
    m_scene->uploadGeometry();
    bool autocam = true ; 

    // handle commandline --target option that needs loaded geometry 
    unsigned int target = m_scene->getTargetDeferred();   // default to 0 
    LOG(debug) << "App::uploadGeometry setting target " << target ; 

    m_scene->setTarget(target, autocam);
 
    (*m_timer)("uploadGeometry"); 
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

    unsigned int code = m_opticks->getSourceCode();
    std::string typ = Opticks::SourceTypeLowercase(code);
    std::string tag = m_fcfg->getEventTag();
    std::string cat = m_fcfg->getEventCat();

    std::string det = m_cache->getDetector();

    m_parameters->add<std::string>("Type", typ );
    m_parameters->add<std::string>("Tag", tag );
    m_parameters->add<std::string>("Cat", cat );
    m_parameters->add<std::string>("Detector", det );


    Lookup* lookup = m_ggeo->getLookup();

    NPY<float>* npy = NULL ; 
    if( code == CERENKOV || code == SCINTILLATION )
    {
        npy = loadGenstepFromFile(typ, tag, det ); 

        m_g4step = new G4StepNPY(npy);    
        m_g4step->relabel(code); // becomes the ghead.i.x used in cu/generate.cu

        if(m_cache->isDayabay())
        {   
            m_g4step->setLookup(lookup);   
            m_g4step->applyLookup(0, 2);      
            // translate materialIndex (1st quad, 3rd number) from chroma to GGeo 
            m_parameters->add<std::string>("genstepAfterLookup",   npy->getDigestString()  );
        }
    }
    else if(code == TORCH)
    {
        m_torchstep = m_opticks->makeSimpleTorchStep();

        m_ggeo->targetTorchStep(m_torchstep);

        const char* material = m_torchstep->getMaterial() ;
        unsigned int matline = m_ggeo->getMaterialLine(material);
        m_torchstep->setMaterialLine(matline);  

        LOG(debug) << "App::makeSimpleTorchStep"
                  << " config " << m_torchstep->getConfig() 
                  << " material " << material 
                  << " matline " << matline
                  ;

        bool verbose = hasOpt("torchdbg");
        m_torchstep->addStep(verbose);  // copyies above configured step settings into the NPY and increments the step index, ready for configuring the next step 

        npy = m_torchstep->getNPY(); 
    }
    

    (*m_timer)("loadGenstep"); 

 

    m_evt->setMaxRec(m_fcfg->getRecordMax());          // must set this before setGenStepData to have effect

    bool geocenter  = m_fcfg->hasOpt("geocenter");


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



void App::loadEvtFromFile()
{
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

    std::string instance_slice = m_fcfg->getISlice() ;;
    std::string face_slice = m_fcfg->getFSlice() ;;
    std::string part_slice = m_fcfg->getPSlice() ;;

    NSlice* islice = !instance_slice.empty() ? new NSlice(instance_slice.c_str()) : NULL ; 
    NSlice* fslice = !face_slice.empty() ? new NSlice(face_slice.c_str()) : NULL ; 
    NSlice* pslice = !part_slice.empty() ? new NSlice(part_slice.c_str()) : NULL ; 

    unsigned int nmm = m_ggeo->getNumMergedMesh();
    for(unsigned int i=0 ; i < nmm ; i++)
    {
        GMergedMesh* mm = m_ggeo->getMergedMesh(i);
        if(restrict_mesh > -1 && i != restrict_mesh ) mm->setGeoCode('K');      
        if(analytic_mesh > -1 && i == analytic_mesh && i > 0) mm->setGeoCode('S');      
        if(i>0) mm->setInstanceSlice(islice);

        // restrict to non-global for now
        if(i>0) mm->setFaceSlice(fslice);   
        if(i>0) mm->setPartSlice(pslice);   
    }
}

void App::prepareOptiX()
{
    bool compute  = m_fcfg->hasOpt("compute"); 
    int  debugidx = m_fcfg->getDebugIdx();
    int  stack    = m_fcfg->getStack();

    LOG(debug) << "App::prepareOptiX stack " << stack ;  

    OContext::Mode_t mode = compute ? OContext::COMPUTE : OContext::INTEROP ; 

    assert( mode == OContext::INTEROP && "COMPUTE mode not operational"); 

    // TODO: move inside OGeo ? 

    optix::Context context = optix::Context::create();

    m_ocontext = new OContext(context, mode); 
    m_ocontext->setStackSize(stack);
    m_ocontext->setPrintIndex(m_fcfg->getPrintIndex().c_str());
    m_ocontext->setDebugPhoton(debugidx);

    m_ocolors = new OColors(context, m_cache->getColors() );
    m_ocolors->convert();

    m_olib = new OBndLib(context,m_ggeo->getBndLib());
    m_olib->convert(); 

    m_oscin = new OScintillatorLib(context, m_ggeo->getScintillatorLib());
    m_oscin->convert(); 

    m_osrc = new OSourceLib(context, m_ggeo->getSourceLib());
    m_osrc->convert(); 



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

    LOG(debug) << m_ogeo->description("App::prepareOptiX ogeo");

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

    // TODO: move into NumpyEvt 

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
    const char* cat = m_parameters->getStringValue("Cat").c_str();
    const char* det = m_parameters->getStringValue("Detector").c_str();

    const char* udet = strlen(cat) > 0 ? cat : det ; 

    LOG(info) << "App::downloadEvt"
              << " typ: " << typ
              << " tag: " << tag
              << " cat: " << cat
              << " det: " << det
              << " udet: " << udet
              ;

    // app.saveEvt
    dpho->setVerbose();
    dpho->save("ox%s", typ,  tag, udet);

    drec->setVerbose();
    drec->save("rx%s", typ,  tag, udet);

    dhis->setVerbose();
    dhis->save("ph%s", typ,  tag, udet);

    daux->setVerbose();
    daux->save("au%s", typ,  tag, udet);


    NPY<float>* fdom = m_opropagator->getDomain();
    NPY<int>*   idom = m_opropagator->getIDomain();
    fdom->save("fdom%s", typ,  tag, udet);
    idom->save("idom%s", typ,  tag, udet);


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
        assert(nphosel == 2*nsqa);
        unsigned int nrecsel = trecsel.getSize() ; 
        assert(nrecsel == maxrec*2*nsqa);
        LOG(info) << "App::indexSequence "
                  << " nsqa (2*num_photons)" << nsqa 
                  << " nphosel " << nphosel
                  << " nrecsel " << nrecsel
                  ; 
#endif

        TSparse<unsigned long long> seqhis("History_Sequence", seq->slice(2,0)); // stride,begin 
        seqhis.make_lookup();
        m_seqhis = new GItemIndex(seqhis.getIndex()) ;  
        seqhis.apply_lookup<unsigned char>( tphosel.slice(4,0));  // stride, begin

#ifdef DEBUG
        seq->dump<unsigned long long>("App::indexSequence OBuf seq.dump", 2, 0, nsqd);
        tphosel.dumpint<unsigned char>("tphosel.dumpint<unsigned char>(4,0)", 4,0, npsd) ;
        LOG(info) << seqhis.dump_("App::indexSequence seqhis (dump_)");
#endif

        TSparse<unsigned long long> seqmat("Material_Sequence", seq->slice(2,1)); // stride,begin 
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


    // perhaps can simplify this stuff... with new GPropertyLib approach 
    // so write out the indices, to base some tests on
    m_seqhis->getIndex()->save(m_cache->getIdPath());
    m_seqmat->getIndex()->save(m_cache->getIdPath());


    //Types* types = m_cache->getTypes();   
    GAttrSeq* qflg = m_cache->getFlags()->getAttrIndex();
    GAttrSeq* qmat = m_ggeo->getMaterialLib()->getAttrNames(); 


    qflg->setCtrl(GAttrSeq::SEQUENCE_DEFAULTS);
    qflg->dumpTable(m_seqhis->getIndex(), "App::indexSequence m_seqhis"); 
    qmat->setCtrl(GAttrSeq::SEQUENCE_DEFAULTS);
    qmat->dumpTable(m_seqmat->getIndex(), "App::indexSequence m_seqmat"); 

    // looks like types only used by GItemIndex for the labels
    // by formTable/getLabel which is invoked by the labeller 
    // on all the keys of the index
    // which are then access in oglrap-/GUI by getLabels()
    //
    // but for GUI selection cannot just create vectors of labels
    // need pointers for radio selects etc..

    m_seqhis->setTitle("Photon Flag Sequence Selection");
    m_seqhis->setHandler(qflg);
    m_seqhis->formTable();

    m_seqmat->setTitle("Photon Material Sequence Selection");
    m_seqmat->setHandler(qmat);
    m_seqmat->formTable();


    (*m_timer)("indexSequence"); 
}




void App::indexBoundaries()
{
    /*
    Indexing the signed integer boundary code, from optixrap-/cu/generate.cu::

         208      p.flags.i.x = prd.boundary ;

    */
    if(!m_evt) return ; 

    OBuf* pho = m_opropagator->getPhotonBuf();
    pho->setHexDump(false);

    bool hexkey = false ; 
    TSparse<int> boundaries("indexBoundaries", pho->slice(4*4,4*3+0), hexkey); // stride,begin  hexkey effects Index and dumping only 
    boundaries.make_lookup();
    boundaries.dump("App::indexBoundaries TSparse<int>::dump");


    GBndLib* blib = m_ggeo->getBndLib();
    blib->close();                         // cannot add new boundaries after a close
    GAttrSeq* qbnd = blib->getAttrNames();
    qbnd->setCtrl(GAttrSeq::VALUE_DEFAULTS);

    std::map<unsigned int, std::string> boundary_names = qbnd->getNamesMap(GAttrSeq::ONEBASED) ;
        
    m_boundaries = new GItemIndex(boundaries.getIndex()) ;  
    m_boundaries->setTitle("Photon Termination Boundaries");
    m_boundaries->setHandler(qbnd);

    m_boundaries->getIndex()->save(m_cache->getIdPath());

    m_boundaries->formTable();

    (*m_timer)("indexBoundaries"); 



    NPY<float>* dpho = m_evt->getPhotonData();
    if(dpho->hasData())
    {
        // host based indexing of unique material codes, requires downloadEvt to pull back the photon data
        LOG(info) << "App::indexBoundaries host based " ;
        m_bnd = new BoundariesNPY(dpho); 
        m_bnd->setBoundaryNames(boundary_names); 
        m_bnd->indexBoundaries();     
    } 

    (*m_timer)("indexBoundariesOld"); 


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

    // TODO: wean this off use of Types, for the new way (GFlags..)
    Types* types = m_cache->getTypes();
    Typ* typ = m_cache->getTyp();

    NPY<float>* dpho = m_evt->getPhotonData();


    if(dpho->hasData())
    {
        m_pho = new PhotonsNPY(dpho);   // a detailed photon/record dumper : looks good for photon level debug 
        m_pho->setTypes(types);
        m_pho->setTyp(typ);

        m_hit = new HitsNPY(dpho, m_ggeo->getSensorList());
        m_hit->debugdump();
    }

    // hmm thus belongs in NumpyEvt rather than here
    NPY<short>* drec = m_evt->getRecordData();

    if(drec->hasData())
    {
        m_rec = new RecordsNPY(drec, m_evt->getMaxRec());
        m_rec->setTypes(types);
        m_rec->setTyp(typ);

        NPY<float>* fdom = m_opropagator->getDomain() ;
        m_rec->setDomains(fdom);

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

    m_types = m_cache->getTypes();  // needed for each render
    m_photons = new Photons(m_types, m_boundaries, m_seqhis, m_seqmat ) ; // GUI jacket 
    m_scene->setPhotons(m_photons);

    m_gui = new GUI(m_ggeo) ;
    m_gui->setScene(m_scene);
    m_gui->setPhotons(m_photons);
    m_gui->setComposition(m_composition);
    m_gui->setBookmarks(m_bookmarks);
    m_gui->setInteractor(m_interactor);   // status line
    
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
        //LOG(debug) << m_ogeo->description("App::render ogeo");

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
            if(m_boundaries)
            {
                //glm::ivec4 sel = m_bnd->getSelection() ;
                //m_composition->setSelection(sel); 
                //m_composition->getPick().y = sel.x ;   //  1st boundary 

                m_composition->getPick().y = m_boundaries->getSelected() ;   //  1st boundary 
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



