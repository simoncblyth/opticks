#include "App.hh"

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

// opticks-
#include "Opticks.hh"
#include "OpticksCfg.hh"
#include "OpticksResource.hh"


// oglrap-
#include "StateGUI.hh"
#include "Scene.hh"
#include "SceneCfg.hh"
#include "Renderer.hh"
#include "RendererCfg.hh"
#include "Interactor.hh"
#include "InteractorCfg.hh"

#include "Bookmarks.hh"
#include "InterpolatedView.hh"
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
#include "NState.hpp"
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
#include "TimesTable.hpp"
#include "Parameters.hpp"
#include "Report.hpp"
#include "NSlice.hpp"

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

// assimpwrap
#include "AssimpGGeo.hh"

// openmeshrap-
#include "MFixer.hh"
#include "MTool.hh"

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

// opop-
#include "OpIndexer.hh"
#include "OpSeeder.hh"
#include "OpZeroer.hh"

#define GLMVEC4(g) glm::vec4((g).x,(g).y,(g).z,(g).w) 

#define TIMER(s) \
    { \
       (*m_timer)((s)); \
       if(m_evt)\
       {\
          Timer& t = *(m_evt->getTimer()) ;\
          t((s)) ;\
       }\
    }


void App::init(int argc, char** argv)
{
    m_opticks = new Opticks(argc, argv);
    m_resource = m_opticks->getResource();
    m_resource->Summary("App::init OpticksResource::Summary");

    m_cache = new GCache(m_opticks);

    m_parameters = new Parameters ;  // favor evt params over these, as evt params are persisted with the evt
    m_timer      = new Timer("App::");
    m_timer->setVerbose(true);
    m_timer->start();

    m_cfg  = new Cfg("umbrella", false) ; 
    m_fcfg = m_opticks->getCfg();

    m_cfg->add(m_fcfg);

#ifdef NPYSERVER
    m_delegate    = new numpydelegate ; 
    m_cfg->add(new numpydelegateCfg<numpydelegate>("numpydelegate", m_delegate, false));
#endif
}

void App::initViz()
{
    if(m_opticks->isCompute()) return ; 

    // the envvars are normally not defined, using 
    // cmake configure_file values instead
    const char* shader_dir = getenv("SHADER_DIR"); 
    const char* shader_incl_path = getenv("SHADER_INCL_PATH"); 
    const char* shader_dynamic_dir = getenv("SHADER_DYNAMIC_DIR"); 
    // dynamic define for use by GLSL shaders

    m_scene      = new Scene(shader_dir, shader_incl_path, shader_dynamic_dir ) ;

    m_composition = new Composition ; 
    m_frame       = new Frame ; 
    m_interactor  = new Interactor ; 


    m_interactor->setFrame(m_frame);
    m_interactor->setScene(m_scene);
    m_interactor->setComposition(m_composition);

    m_composition->setScene(m_scene);



    m_frame->setInteractor(m_interactor);      
    m_frame->setComposition(m_composition);
    m_frame->setScene(m_scene);

    m_cfg->add(new SceneCfg<Scene>(           "scene",       m_scene,                      true));
    m_cfg->add(new RendererCfg<Renderer>(     "renderer",    m_scene->getGeometryRenderer(), true));
    m_cfg->add(new InteractorCfg<Interactor>( "interactor",  m_interactor,                 true));

    m_composition->addConfig(m_cfg); 
}


void App::configure(int argc, char** argv)
{
    LOG(debug) << "App:configure " << argv[0] ; 
    //m_cfg->dumpTree();

    m_cfg->commandline(argc, argv);
    m_opticks->configure();        // hmm: m_cfg should live inside Opticks


    if(m_fcfg->hasError())
    {
        LOG(fatal) << "App::config parse error " << m_fcfg->getErrorMessage() ; 
        m_fcfg->dump("App::config m_fcfg");
        setExit(true);
        return ; 
    }

    bool compute = hasOpt("compute") ;
    assert(compute == m_opticks->isCompute() && "App::configure compute mismatch between GCache pre-configure and configure"  ); 

    if(hasOpt("idpath")) std::cout << m_cache->getIdPath() << std::endl ;
    if(hasOpt("help"))   std::cout << m_cfg->getDesc()     << std::endl ;
    if(hasOpt("help|version|idpath"))
    {
        setExit(true);
        return ; 
    }


    m_state = m_opticks->getState();
    m_state->setVerbose(false);

    LOG(info) << "App::configure " << m_state->description();

    if(m_composition)
        m_composition->setupConfigurableState(m_state);

    m_bookmarks   = new Bookmarks(m_state->getDir()) ; 
    m_bookmarks->setState(m_state);
    m_bookmarks->setVerbose();

    if(m_interactor)
    {
        m_interactor->setBookmarks(m_bookmarks);
    }

    if(!hasOpt("noevent"))
    {
        // TODO: try moving event creation after geometry is loaded, to avoid need to update domains 
        m_evt = m_opticks->makeEvt() ; 
        m_evt->setFlat(true);

        Parameters* params = m_evt->getParameters() ;
        params->add<std::string>("cmdline", m_cfg->getCommandLine() ); 
    } 

#ifdef NPYSERVER
    if(!hasOpt("nonet"))
    {
        m_delegate->liveConnect(m_cfg); // hookup live config via UDP messages
        m_delegate->setNumpyEvt(m_evt); // allows delegate to update evt when NPY messages arrive, hmm locking needed ?

        try { 
            m_server = new numpyserver<numpydelegate>(m_delegate); // connect to external messages 
        } 
        catch( const std::exception& e)
        {
            LOG(fatal) << "App::config EXCEPTION " << e.what() ; 
            LOG(fatal) << "App::config FAILED to instanciate numpyserver : probably another instance is running : check debugger sessions " ;
        }
    }
#endif

    TIMER("configure");
}




void App::prepareViz()
{
    if(m_opticks->isCompute()) return ; 

    m_size = m_opticks->getSize();

    LOG(debug) << "App::prepareViz"
              << " size " << gformat(m_size);

    m_scene->setNumpyEvt(m_evt);
    if(m_resource->isJuno())
    {
        LOG(warning) << "App::prepareViz disable GeometryStyle  WIRE for JUNO as too slow " ;

        m_scene->setNumGeometryStyle(Scene::WIRE); 
        m_scene->setNumGlobalStyle(Scene::GVISVEC); 

        m_scene->setRenderMode("bb0,bb1,-global");
        std::string rmode = m_scene->getRenderMode();
        LOG(info) << "App::prepareViz " << rmode ; 
    }


    m_composition->setSize( m_size );

    m_frame->setTitle("GGeoView");
    m_frame->setFullscreen(hasOpt("fullscreen"));

    m_dd = new DynamicDefine();   // configuration used in oglrap- shaders
    m_dd->add("MAXREC",m_fcfg->getRecordMax());    
    m_dd->add("MAXTIME",m_fcfg->getTimeMax());    
    m_dd->add("PNUMQUAD", 4);  // quads per photon
    m_dd->add("RNUMQUAD", 2);  // quads per record 
    m_dd->add("MATERIAL_COLOR_OFFSET", (unsigned int)GColors::MATERIAL_COLOR_OFFSET );
    m_dd->add("FLAG_COLOR_OFFSET", (unsigned int)GColors::FLAG_COLOR_OFFSET );
    m_dd->add("PSYCHEDELIC_COLOR_OFFSET", (unsigned int)GColors::PSYCHEDELIC_COLOR_OFFSET );
    m_dd->add("SPECTRAL_COLOR_OFFSET", (unsigned int)GColors::SPECTRAL_COLOR_OFFSET );


    m_scene->write(m_dd);

    m_scene->initRenderers();  // reading shader source and creating renderers

    m_frame->init();           // creates OpenGL context

    m_window = m_frame->getWindow();

    m_scene->setComposition(m_composition);     // defer until renderers are setup 



    InterpolatedView* iv = m_bookmarks->getInterpolatedView() ; // creates the interpolation based on initial bookmarks

    m_composition->setAltView(iv);

    //iv->Summary("App::prepareViz setting composition.altview to InterpolatedView");


    TIMER("prepareScene");

    LOG(debug) << "App::prepareScene DONE ";
} 




void App::loadGeometry()
{
    LOG(info) << "App::loadGeometry START" ; 

    loadGeometryBase();

    if(!m_ggeo->isValid())
    {
        LOG(warning) << "App::loadGeometry finds invalid geometry, try creating geocache with --nogeocache/-G option " ; 
        setExit(true); 
        return ; 
    }

    if(hasOpt("test")) modifyGeometry() ;

    fixGeometry();

    registerGeometry();

    if(!m_opticks->isGeocache())
    {
        LOG(info) << "App::loadGeometry early exit due to --nogeocache/-G option " ; 
        setExit(true); 
    }

    configureGeometry();

    LOG(info) << "App::loadGeometry DONE" ; 
}


void App::loadGeometryBase()
{
    // hmm funny placement, move this just after config 
    m_opticks->setGeocache(!m_fcfg->hasOpt("nogeocache"));
    m_opticks->setInstanced( !m_fcfg->hasOpt("noinstanced")  ); // find repeated geometry 

    m_ggeo = new GGeo(m_cache);

    if(hasOpt("qe1"))
        m_ggeo->getSurfaceLib()->setFakeEfficiency(1.0);

    m_ggeo->setLoaderImp(&AssimpGGeo::load);    // setting GLoaderImpFunctionPtr
    m_ggeo->setMeshJoinImp(&MTool::joinSplitUnion);
    m_ggeo->setMeshJoinCfg( m_resource->getMeshfix() );

    std::string meshversion = m_fcfg->getMeshVersion() ;;
    if(!meshversion.empty())
    {
        LOG(warning) << "App::loadGeometry using debug meshversion " << meshversion ;  
        m_ggeo->getGeoLib()->setMeshVersion(meshversion.c_str());
    }

    m_ggeo->loadGeometry();

    TIMER("loadGeometryBase");
}

void App::modifyGeometry()
{
    assert(hasOpt("test"));
    LOG(debug) << "App::modifyGeometry" ;

    std::string testconf = m_fcfg->getTestConfig();
    m_ggeo->modifyGeometry( testconf.empty() ? NULL : testconf.c_str() );

    TIMER("modifyGeometry"); 
}


void App::fixGeometry()
{
    if(m_ggeo->isLoaded())
    {
        LOG(debug) << "App::fixGeometry needs to be done precache " ;
        return ; 
    }
    LOG(info) << "App::fixGeometry" ; 

    MFixer* fixer = new MFixer(m_ggeo);
    fixer->setVerbose(hasOpt("meshfixdbg"));
    fixer->fixMesh();
 
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



void App::configureGeometry()
{
    int restrict_mesh = m_fcfg->getRestrictMesh() ;  
    int analytic_mesh = m_fcfg->getAnalyticMesh() ; 
    unsigned int nmm = m_ggeo->getNumMergedMesh();

    LOG(info) << "App::configureGeometry" 
              << " restrict_mesh " << restrict_mesh
              << " analytic_mesh " << analytic_mesh
              << " nmm " << nmm
              ;

    std::string instance_slice = m_fcfg->getISlice() ;;
    std::string face_slice = m_fcfg->getFSlice() ;;
    std::string part_slice = m_fcfg->getPSlice() ;;

    NSlice* islice = !instance_slice.empty() ? new NSlice(instance_slice.c_str()) : NULL ; 
    NSlice* fslice = !face_slice.empty() ? new NSlice(face_slice.c_str()) : NULL ; 
    NSlice* pslice = !part_slice.empty() ? new NSlice(part_slice.c_str()) : NULL ; 

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

    TIMER("configureGeometry"); 
}




void App::registerGeometry()
{
    LOG(info) << "App::registerGeometry" ; 

    //for(unsigned int i=1 ; i < m_ggeo->getNumMergedMesh() ; i++) m_ggeo->dumpNodeInfo(i);

    m_mesh0 = m_ggeo->getMergedMesh(0); 

    gfloat4 ce0 = m_mesh0->getCenterExtent(0);  // 0 : all geometry of the mesh, >0 : specific volumes
    m_opticks->setSpaceDomain( glm::vec4(ce0.x,ce0.y,ce0.z,ce0.w) );

    // treat opticks as the common authority 

    if(m_evt)
    {
       // TODO: migrate npy-/NumpyEvt to opop-/OpEvent so this can happen at more specific level 
        m_opticks->dumpDomains("App::registerGeometry copy Opticks domains to m_evt");
        m_evt->setSpaceDomain(m_opticks->getSpaceDomain());
    }

    LOG(debug) << "App::registerGeometry ce0: " 
                      << " x " << ce0.x
                      << " y " << ce0.y
                      << " z " << ce0.z
                      << " w " << ce0.w
                      ;
 
}




void App::uploadGeometryViz()
{
    if(m_opticks->isCompute()) return ; 




    GColors* colors = m_cache->getColors();

    m_composition->setColorDomain( colors->getCompositeDomain() ); 

    m_scene->uploadColorBuffer( colors->getCompositeBuffer() );  //     oglrap-/Colors preps texture, available to shaders as "uniform sampler1D Colors"

    m_composition->setTimeDomain( m_opticks->getTimeDomain() );
    m_composition->setDomainCenterExtent(m_opticks->getSpaceDomain());


    m_scene->setGeometry(m_ggeo);

    m_scene->uploadGeometry();

    bool autocam = true ; 

    // handle commandline --target option that needs loaded geometry 
    unsigned int target = m_scene->getTargetDeferred();   // default to 0 
    LOG(debug) << "App::uploadGeometryViz setting target " << target ; 

    m_scene->setTarget(target, autocam);
 
    TIMER("uploadGeometryViz"); 
}









void App::loadGenstep()
{
    if(hasOpt("nooptix|noevent")) 
    {
        LOG(warning) << "App::loadGenstep skip due to --nooptix/--noevent " ;
        return ;
    }

    unsigned int code = m_opticks->getSourceCode();
    Lookup* lookup = m_ggeo->getLookup();


    NPY<float>* npy = NULL ; 
    if( code == CERENKOV || code == SCINTILLATION )
    {
        int modulo = m_fcfg->getModulo();
        npy = m_evt->loadGenstepFromFile(modulo);

        m_g4step = new G4StepNPY(npy);    
        m_g4step->relabel(code); // becomes the ghead.i.x used in cu/generate.cu

        if(m_resource->isDayabay())
        {   
            m_g4step->setLookup(lookup);   
            m_g4step->applyLookup(0, 2);      
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
    

    TIMER("loadGenstep"); 

    m_evt->setGenstepData(npy); 

    TIMER("setGenstepData"); 
}



void App::targetViz()
{
    if(m_opticks->isCompute()) return ; 

    glm::vec4 mmce = GLMVEC4(m_mesh0->getCenterExtent(0)) ;
    glm::vec4 gsce = (*m_evt)["genstep.vpos"]->getCenterExtent();
    bool geocenter  = m_fcfg->hasOpt("geocenter");
    glm::vec4 uuce = geocenter ? mmce : gsce ;

    //print(mmce, "loadGenstep mmce");
    //print(gsce, "loadGenstep gsce");
    //print(uuce, "loadGenstep uuce");

    if(m_scene->getTarget() == 0)
    {
        // only pointing based in genstep if not already targetted
        bool autocam = true ; 
        m_composition->setCenterExtent( uuce , autocam );
    }

    m_scene->setRecordStyle( m_fcfg->hasOpt("alt") ? Scene::ALTREC : Scene::REC );    

    TIMER("targetViz"); 
}


void App::loadEvtFromFile()
{
    m_evt->load(true);

    if(m_evt->isNoLoad())
        LOG(warning) << "App::loadEvtFromFile LOAD FAILED " ;
}



void App::uploadEvtViz()
{
    if(m_opticks->isCompute()) return ; 

    if(hasOpt("nooptix|noevent")) 
    {
        LOG(warning) << "App::uploadEvtViz skip due to --nooptix/--noevent " ;
        return ;
    }
 
    LOG(info) << "App::uploadEvtViz START " ;

    m_composition->update();

    m_scene->upload();

    m_scene->uploadSelection();

    TIMER("uploadEvtViz"); 
}




void App::prepareOptiX()
{
    // TODO: move inside OGeo or new opop-/OpEngine ? 

    LOG(debug) << "App::prepareOptiX" ;  

    OContext::Mode_t mode = m_opticks->isCompute() ? OContext::COMPUTE : OContext::INTEROP ; 

    optix::Context context = optix::Context::create();

    m_ocontext = new OContext(context, mode); 
    m_ocontext->setStackSize(m_fcfg->getStack());
    m_ocontext->setPrintIndex(m_fcfg->getPrintIndex().c_str());
    m_ocontext->setDebugPhoton(m_fcfg->getDebugIdx());

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

    LOG(debug) << m_ogeo->description("App::prepareOptiX ogeo");

    TIMER("prepareOptiX"); 
}


void App::prepareOptiXViz()
{
    if(m_opticks->isCompute()) return ; 

    unsigned int width  = m_composition->getPixelWidth();
    unsigned int height = m_composition->getPixelHeight();

    optix::Context context = m_ocontext->getContext();

    m_oframe = new OFrame(context, width, height);

    context["output_buffer"]->set( m_oframe->getOutputBuffer() );

    m_interactor->setTouchable(m_oframe);

    Renderer* rtr = m_scene->getRaytraceRenderer();

    m_orenderer = new ORenderer(rtr, m_oframe, m_scene->getShaderDir(), m_scene->getShaderInclPath());

    m_otracer = new OTracer(m_ocontext, m_composition);

    LOG(info) << "App::prepareOptiXViz DONE "; 

    m_ocontext->dump("App::prepareOptiX");

    TIMER("prepareOptiXViz"); 
}


void App::preparePropagator()
{
    bool noevent    = m_fcfg->hasOpt("noevent");
    bool trivial    = m_fcfg->hasOpt("trivial");
    int  override   = m_fcfg->getOverride();

    assert(!noevent);

    m_opropagator = new OPropagator(m_ocontext, m_opticks);

    m_opropagator->setNumpyEvt(m_evt);

    m_opropagator->setTrivial(trivial);
    m_opropagator->setOverride(override);

    m_opropagator->initRng();
    m_opropagator->initEvent();

    LOG(info) << "App::preparePropagator DONE "; 

    TIMER("preparePropagator"); 
}


void App::seedPhotonsFromGensteps()
{
    if(!m_evt) return ; 

    OpSeeder* seeder = new OpSeeder(m_ocontext) ; 

    seeder->setEvt(m_evt);
    seeder->setPropagator(m_opropagator);  // only used in compute mode

    seeder->seedPhotonsFromGensteps();
}


void App::initRecords()
{
    if(!m_evt) return ; 

    if(!m_evt->isStep())
    {
        LOG(info) << "App::initRecords --nostep mode skipping " ;
        return ; 
    }

    OpZeroer* zeroer = new OpZeroer(m_ocontext) ; 

    zeroer->setEvt(m_evt);
    zeroer->setPropagator(m_opropagator);  // only used in compute mode

    zeroer->zeroRecords();
}



void App::propagate()
{
    if(hasOpt("nooptix|noevent|nopropagate")) 
    {
        LOG(warning) << "App::propagate skip due to --nooptix/--noevent/--nopropagate " ;
        return ;
    }

    LOG(info)<< "App::propagate" ;

    m_opropagator->prelaunch();     
    TIMER("prelaunch"); 

    m_opropagator->launch();     
    TIMER("propagate"); 

    m_opropagator->dumpTimes("App::propagate");
}



void App::saveEvt()
{
    if(!m_evt) return ; 

    if(m_opticks->isCompute())
    {
        m_opropagator->downloadEvent();
    }
    else
    {
        Rdr::download(m_evt);
    }

    TIMER("downloadEvt"); 

    m_evt->dumpDomains("App::saveEvt dumpDomains");
    m_evt->save(true);
 
    TIMER("saveEvt"); 
}


void App::indexSequence()
{
    if(!m_evt) return ; 
    if(!m_evt->isStep())
    {
        LOG(info) << "App::indexSequence --nostep mode skipping " ;
        return ; 
    }

    OpIndexer* indexer = new OpIndexer(m_ocontext);
    indexer->setVerbose(hasOpt("indexdbg"));
    indexer->setEvt(m_evt);
    indexer->setPropagator(m_opropagator);

    indexer->indexSequence();
    indexer->indexBoundaries();

    TIMER("indexSequence"); 
}


void App::indexPresentationPrep()
{
    LOG(info) << "App::indexPresentationPrep" ; 

    if(!m_evt) return ; 

    Index* seqhis = m_evt->getHistorySeq() ;
    Index* seqmat = m_evt->getMaterialSeq();
    Index* bndidx = m_evt->getBoundaryIdx();


    if(!seqhis)
    {
         LOG(warning) << "App::indexPresentationPrep NULL seqhis" ;
    }
    else
    {
        GAttrSeq* qflg = m_cache->getFlags()->getAttrIndex();
        qflg->setCtrl(GAttrSeq::SEQUENCE_DEFAULTS);
        //qflg->dumpTable(seqhis, "App::indexPresentationPrep seqhis"); 
        m_seqhis = new GItemIndex(seqhis) ;  
        m_seqhis->setTitle("Photon Flag Sequence Selection");
        m_seqhis->setHandler(qflg);
        m_seqhis->formTable();
    }


    if(!seqmat)
    {
         LOG(warning) << "App::indexPresentationPrep NULL seqmat" ;
    }
    else
    {
        GAttrSeq* qmat = m_ggeo->getMaterialLib()->getAttrNames(); 
        qmat->setCtrl(GAttrSeq::SEQUENCE_DEFAULTS);
        //qmat->dumpTable(seqmat, "App::indexPresentationPrep seqmat"); 
        m_seqmat = new GItemIndex(seqmat) ;  
        m_seqmat->setTitle("Photon Material Sequence Selection");
        m_seqmat->setHandler(qmat);
        m_seqmat->formTable();
    }


    if(!bndidx)
    {
         LOG(warning) << "App::indexPresentationPrep NULL bndidx" ;
    }
    else
    {
        GBndLib* blib = m_ggeo->getBndLib();
        GAttrSeq* qbnd = blib->getAttrNames();
        if(!qbnd->hasSequence())
        {
            blib->close();
            assert(qbnd->hasSequence());
        }
        qbnd->setCtrl(GAttrSeq::VALUE_DEFAULTS);
        //qbnd->dumpTable(bndidx, "App::indexPresentationPrep bndidx"); 

        m_boundaries = new GItemIndex(bndidx) ;  
        m_boundaries->setTitle("Photon Termination Boundaries");
        m_boundaries->setHandler(qbnd);
        m_boundaries->formTable();
    } 


    TIMER("indexPresentationPrep"); 
}


void App::indexBoundariesHost()
{
    // Indexing the final signed integer boundary code (p.flags.i.x = prd.boundary) from optixrap-/cu/generate.cu
    // see also opop-/OpIndexer::indexBoundaries for GPU version of this indexing 

    if(!m_evt) return ; 

    GBndLib* blib = m_ggeo->getBndLib();
    GAttrSeq* qbnd = blib->getAttrNames();
    if(!qbnd->hasSequence())
    {
         blib->close();
         assert(qbnd->hasSequence());
    }

    std::map<unsigned int, std::string> boundary_names = qbnd->getNamesMap(GAttrSeq::ONEBASED) ;

    NPY<float>* dpho = m_evt->getPhotonData();
    if(dpho->hasData())
    {
        // host based indexing of unique material codes, requires downloadEvt to pull back the photon data
        LOG(info) << "App::indexBoundaries host based " ;
        m_bnd = new BoundariesNPY(dpho); 
        m_bnd->setBoundaryNames(boundary_names); 
        m_bnd->indexBoundaries();     
    } 

    TIMER("indexBoundariesHost"); 
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
  

    if(m_evt->isIndexed())
    {
        LOG(info) << "App::indexEvt" 
                  << " skip as already indexed "
                  ;
    }
 
    indexSequence();

    indexBoundariesHost();

    TIMER("indexEvt"); 
}


void App::indexEvtOld()
{
    if(!m_evt) return ; 

    // TODO: wean this off use of Types, for the new way (GFlags..)
    Types* types = m_cache->getTypes();
    Typ* typ = m_cache->getTyp();

    NPY<float>* ox = m_evt->getPhotonData();


    if(ox && ox->hasData())
    {
        m_pho = new PhotonsNPY(ox);   // a detailed photon/record dumper : looks good for photon level debug 
        m_pho->setTypes(types);
        m_pho->setTyp(typ);

        m_hit = new HitsNPY(ox, m_ggeo->getSensorList());
        //m_hit->debugdump();
    }

    // hmm thus belongs in NumpyEvt rather than here
    NPY<short>* rx = m_evt->getRecordData();

    if(rx && rx->hasData())
    {
        m_rec = new RecordsNPY(rx, m_evt->getMaxRec(), m_evt->isFlat());
        m_rec->setTypes(types);
        m_rec->setTyp(typ);
        m_rec->setDomains(m_evt->getFDomain()) ;

        if(m_pho)
        {
            m_pho->setRecs(m_rec);
            //if(m_torchstep) m_torchstep->dump("App::indexEvtOld TorchStepNPY");

            m_pho->dump(0  ,  "App::indexEvtOld dpho 0");
            m_pho->dump(100,  "App::indexEvtOld dpho 100" );
            m_pho->dump(1000, "App::indexEvtOld dpho 1000" );

        }
        m_evt->setRecordsNPY(m_rec);
        m_evt->setPhotonsNPY(m_pho);
    }

    TIMER("indexEvtOld"); 
}




void App::prepareGUI()
{
    if(m_opticks->isCompute()) return ; 

    m_bookmarks->create(0);

#ifdef GUI_

    m_types = m_cache->getTypes();  // needed for each render
    m_photons = new Photons(m_types, m_boundaries, m_seqhis, m_seqmat ) ; // GUI jacket 
    m_scene->setPhotons(m_photons);

    m_gui = new GUI(m_ggeo) ;
    m_gui->setScene(m_scene);
    m_gui->setPhotons(m_photons);
    m_gui->setComposition(m_composition);
    m_gui->setBookmarks(m_bookmarks);
    m_gui->setStateGUI(new StateGUI(m_state));
    m_gui->setInteractor(m_interactor);   // status line
    
    m_gui->init(m_window);
    m_gui->setupHelpText( m_cfg->getDescString() );

    TimesTable* tt = m_evt ? m_evt->getTimesTable() : NULL ; 
    if(tt)
    {
        m_gui->setupStats(tt->getLines());
    }
    else
    {
        LOG(warning) << "App::prepareGUI NULL TimesTable " ; 
    }  

    Parameters* parameters = m_evt ? m_evt->getParameters() : m_parameters ; 

    m_gui->setupParams(parameters->getLines());

#endif

}


void App::renderGUI()
{
#ifdef GUI_
    m_gui->newframe();
    bool* show_gui_window = m_interactor->getGUIModeAddress();
    if(*show_gui_window)
    {
        m_gui->show(show_gui_window);
        if(m_photons)
        {
            if(m_boundaries)
            {
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

    bool* show_scrub_window = m_interactor->getScrubModeAddress();
    if(*show_scrub_window)
        m_gui->show_scrubber(show_scrub_window);

    m_gui->render();
#endif
}




void App::render()
{
    if(m_opticks->isCompute()) return ; 

    m_frame->viewport();
    m_frame->clear();

#ifdef OPTIX
    if(m_scene->isRaytracedRender() || m_scene->isCompositeRender())
    {
        if(m_otracer && m_orenderer)
        {
            if(m_composition->hasChangedGeometry())
            {
                unsigned int scale = m_interactor->getOptiXResolutionScale() ; 
                m_otracer->setResolutionScale(scale) ;
                m_otracer->trace();
                m_oframe->push_PBO_to_Texture();           
            }
            else
            {
                // dont bother tracing when no change in geometry
            }
        }
    }
#endif
    m_scene->render();
}



void App::renderLoop()
{
    if(m_opticks->isCompute()) return ; 

    if(hasOpt("noviz"))
    {
        LOG(info) << "App::renderLoop early exit due to --noviz/-V option " ; 
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
        if(m_server) m_server->poll_one();  
#endif
        count = m_composition->tick();

        if( m_composition->hasChanged() || m_interactor->hasChanged() || count == 1)  
        {
            render();
            renderGUI();

            glfwSwapBuffers(m_window);

            m_interactor->setChanged(false);  
            m_composition->setChanged(false);   // sets camera, view, trackball dirty status 
        }
    }
}



void App::cleanup()
{

#ifdef OPTIX
    if(m_ocontext) m_ocontext->cleanUp();
#endif
#ifdef NPYSERVER
    if(m_server) m_server->stop();
#endif
#ifdef GUI_
    if(m_gui) m_gui->shutdown();
#endif
    if(m_frame) m_frame->exit();
}


bool App::hasOpt(const char* name)
{
    return m_fcfg->hasOpt(name);
}

