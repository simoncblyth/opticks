#include <cstring>
#include "BStr.hh"

// npy-
#include "NGLM.hpp"
#include "NState.hpp"
#include "NPY.hpp"
#include "GLMPrint.hpp"
#include "GLMFormat.hpp"
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
#include "Timer.hpp"
#include "Times.hpp"
#include "TimesTable.hpp"
#include "Parameters.hpp"
#include "Report.hpp"
#include "NSlice.hpp"
#include "NQuad.hpp"

// numpyserver-
#ifdef WITH_NPYSERVER
#include "numpydelegate.hpp"
#include "numpydelegateCfg.hpp"
#include "numpyserver.hpp"
#endif

// okc-
#include "Opticks.hh"
#include "OpticksFlags.hh"
#include "OpticksColors.hh"
#include "OpticksAttrSeq.hh"
#include "OpticksCfg.hh"
#include "OpticksEvent.hh"
#include "OpticksPhoton.h"
#include "OpticksResource.hh"
#include "Bookmarks.hh"
#include "Composition.hh"
#include "InterpolatedView.hh"

// ggeo-
#include "GGeo.hh"
#include "GItemIndex.hh"

// opticksgeo-
#include "OpticksGeometry.hh"


// windows headers from PLOG need to be before glfw 
// http://stackoverflow.com/questions/3927810/how-to-prevent-macro-redefinition
#include "PLOG.hh"

// oglrap-  Frame brings in GL/glew.h GLFW/glfw3.h gleq.h
#include "Frame.hh"

#define GUI_ 1
#ifdef GUI_
#include "GUI.hh"
#endif

#include "StateGUI.hh"
#include "Scene.hh"
#include "SceneCfg.hh"
#include "Renderer.hh"
#include "RendererCfg.hh"
#include "Interactor.hh"
#include "InteractorCfg.hh"
#include "Rdr.hh"
#include "Texture.hh"
#include "Photons.hh"
#include "DynamicDefine.hh"



#ifdef WITH_OPTIX
// optixgl-
#include "OpViz.hh"
// opticksop-
#include "OpEngine.hh"
#endif

#include "App.hh"
#include "GGV_BODY.hh"


#define TIMER(s) \
    { \
       if(m_evt)\
       {\
          Timer& t = *(m_evt->getTimer()) ;\
          t((s)) ;\
       }\
       else if(m_opticks) \
       {\
          Timer& t = *(m_opticks->getTimer()) ;\
          t((s)) ;\
       }\
    }




App::App(const char* prefix, int argc, char** argv )
   : 
      m_opticks(NULL),
      m_prefix(strdup(prefix)),
      m_parameters(NULL),
      m_timer(NULL),
      m_cache(NULL),
      m_dd(NULL),
      m_state(NULL),
      m_scene(NULL),
      m_composition(NULL),
      m_frame(NULL),
      m_window(NULL),
      m_bookmarks(NULL),
      m_interactor(NULL),
#ifdef WITH_NPYSERVER
      m_delegate(NULL),
      m_server(NULL),
#endif
      m_evt(NULL), 
      m_cfg(NULL),
      m_fcfg(NULL),
      m_types(NULL),
      m_geometry(NULL),
      m_ggeo(NULL),

#ifdef WITH_OPTIX
      m_ope(NULL),
      m_opv(NULL),
#endif

      m_bnd(NULL),
      m_pho(NULL),
      m_hit(NULL),
      m_rec(NULL),
      m_seqhis(NULL),
      m_seqmat(NULL),
      m_boundaries(NULL),
      m_photons(NULL),
      m_gui(NULL),
      m_g4step(NULL),
      m_torchstep(NULL)
{
    init(argc, argv);
}

GCache* App::getCache()
{
    return m_cache ; 
}

OpticksCfg<Opticks>* App::getOpticksCfg()
{
    return m_fcfg ; 
}



void App::init(int argc, char** argv)
{
    m_opticks = new Opticks(argc, argv);
    m_opticks->Summary("App::init OpticksResource::Summary");


    m_composition = new Composition ;   // Composition no longer Viz only

    // TODO: review BCfg machinery and relocate into Opticks
    //       .. nope it needs to live mostly at App level
    //       .. due to templated tree of BCfg objects approach 

    m_cfg  = new BCfg("umbrella", false) ; 
    m_fcfg = m_opticks->getCfg();

    m_cfg->add(m_fcfg);

#ifdef WITH_NPYSERVER
    m_delegate    = new numpydelegate ; 
    m_cfg->add(new numpydelegateCfg<numpydelegate>("numpydelegate", m_delegate, false));
#endif

    TIMER("init");
}

bool App::isCompute()
{
    //return m_opticks->isCompute() || m_opticks->isRemoteSession() ;
    return m_opticks->isCompute() ;
    // hmm : this info is needed elsewhere, so better to make the decision inside Opticks
}

void App::initViz()
{
    if(isCompute()) return ; 

    // perhaps a VizManager to contain this lot 

    // envvars normally not defined, using cmake configure_file values instead
    const char* shader_dir = getenv("OPTICKS_SHADER_DIR"); 
    const char* shader_incl_path = getenv("OPTICKS_SHADER_INCL_PATH"); 
    const char* shader_dynamic_dir = getenv("OPTICKS_SHADER_DYNAMIC_DIR"); 

    m_scene      = new Scene(shader_dir, shader_incl_path, shader_dynamic_dir ) ;
    m_frame       = new Frame ; 
    m_interactor  = new Interactor ; 

    m_interactor->setFrame(m_frame);
    m_interactor->setScene(m_scene);
    m_interactor->setComposition(m_composition);

    m_scene->setInteractor(m_interactor);      

    m_frame->setInteractor(m_interactor);      
    m_frame->setComposition(m_composition);
    m_frame->setScene(m_scene);

    m_cfg->add(new SceneCfg<Scene>(           "scene",       m_scene,                      true));
    m_cfg->add(new RendererCfg<Renderer>(     "renderer",    m_scene->getGeometryRenderer(), true));
    m_cfg->add(new InteractorCfg<Interactor>( "interactor",  m_interactor,                 true));

    TIMER("initViz");
}


void App::configure(int argc, char** argv)
{
    LOG(debug) << "App:configure " << argv[0] ; 

    m_composition->addConfig(m_cfg); 
    //m_cfg->dumpTree();

    m_cfg->commandline(argc, argv);
    m_opticks->configure();        // hmm: m_cfg should live inside Opticks


    if(m_fcfg->hasError())
    {
        LOG(fatal) << "App::config parse error " << m_fcfg->getErrorMessage() ; 
        m_fcfg->dump("App::config m_fcfg");
        m_opticks->setExit(true);
        return ; 
    }


    bool compute = m_opticks->isCompute();
    bool interop = m_opticks->isInterop();
    bool compute_requested = m_opticks->isComputeRequested();
    bool compute_opt = hasOpt("compute") ;

    assert(compute_opt == compute_requested && "App::configure compute_requested mismatch between pre-configure and configure"  ); 
    assert(compute != interop);

    if(compute && !compute_requested)
        LOG(warning) << "App::configure FORCED COMPUTE MODE : as remote session detected " ;  


    if(hasOpt("idpath")) std::cout << m_opticks->getIdPath() << std::endl ;
    if(hasOpt("help"))   std::cout << m_cfg->getDesc()     << std::endl ;
    if(hasOpt("help|version|idpath"))
    {
        m_opticks->setExit(true);
        return ; 
    }

    if(!m_opticks->isValid())
    {
        // defer death til after getting help
        LOG(fatal) << "App::configure OPTICKS INVALID : missing envvar or geometry path ?" ;
        assert(0);
    }

    if(!hasOpt("noevent"))
    {
        // TODO: try moving event creation after geometry is loaded, to avoid need to update domains 
        // TODO: organize wrt event loading, currently loading happens latter and trumps this evt ?
        m_evt = m_opticks->makeEvent() ; 
    } 

#ifdef WITH_NPYSERVER
    if(!hasOpt("nonet"))
    {
        m_delegate->liveConnect(m_cfg); // hookup live config via UDP messages
        m_delegate->setEvent(m_evt); // allows delegate to update evt when NPY messages arrive, hmm locking needed ?

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

    configureViz();

    TIMER("configure");
}


bool App::isExit()
{
    return m_opticks->isExit() ; 
}


void App::configureViz()
{
    if(isCompute()) return ; 

    m_state = m_opticks->getState();
    m_state->setVerbose(false);

    LOG(info) << "App::configureViz " << m_state->description();

    assert(m_composition);

    m_state->addConfigurable(m_scene);
    m_composition->addConstituentConfigurables(m_state); // constituents: trackball, view, camera, clipper

    m_composition->setOrbitalViewPeriod(m_fcfg->getOrbitalViewPeriod()); 
    m_composition->setAnimatorPeriod(m_fcfg->getAnimatorPeriod()); 

    if(m_evt)
    { 
        m_composition->setEvt(m_evt);
        m_composition->setTrackViewPeriod(m_fcfg->getTrackViewPeriod()); 

        bool quietly = true ; 
        NPY<float>* track = m_evt->loadGenstepDerivativeFromFile("track", quietly);
        m_composition->setTrack(track);
    }

    LOG(info) << "App::configureViz m_setup bookmarks" ;  

    m_bookmarks   = new Bookmarks(m_state->getDir()) ; 
    m_bookmarks->setState(m_state);
    m_bookmarks->setVerbose();
    m_bookmarks->setInterpolatedViewPeriod(m_fcfg->getInterpolatedViewPeriod());

    if(m_interactor)
    {
        m_interactor->setBookmarks(m_bookmarks);
    }

    LOG(info) << "App::configureViz m_setup bookmarks DONE" ;  

    TIMER("configureViz");
}



void App::prepareViz()
{
    if(isCompute()) return ; 

    m_size = m_opticks->getSize();
    glm::uvec4 position = m_opticks->getPosition() ;

    LOG(info) << "App::prepareViz"
              << " size " << gformat(m_size)
              << " position " << gformat(position)
              ;

    m_scene->setEvent(m_evt);
    if(m_opticks->isJuno())
    {
        LOG(warning) << "App::prepareViz disable GeometryStyle  WIRE for JUNO as too slow " ;

        if(!hasOpt("jwire")) // use --jwire to enable wireframe with JUNO, do this only on workstations with very recent GPUs
        { 
            m_scene->setNumGeometryStyle(Scene::WIRE); 
        }

        m_scene->setNumGlobalStyle(Scene::GVISVEC); // disable GVISVEC, GVEC debug styles

        m_scene->setRenderMode("bb0,bb1,-global");
        std::string rmode = m_scene->getRenderMode();
        LOG(info) << "App::prepareViz " << rmode ; 
    }
    else if(m_opticks->isDayabay())
    {
        m_scene->setNumGlobalStyle(Scene::GVISVEC);   // disable GVISVEC, GVEC debug styles
    }


    m_composition->setSize( m_size );
    m_composition->setFramePosition( position );

    m_frame->setTitle("GGeoView");
    m_frame->setFullscreen(hasOpt("fullscreen"));

    m_dd = new DynamicDefine();   // configuration used in oglrap- shaders
    m_dd->add("MAXREC",m_fcfg->getRecordMax());    
    m_dd->add("MAXTIME",m_fcfg->getTimeMax());    
    m_dd->add("PNUMQUAD", 4);  // quads per photon
    m_dd->add("RNUMQUAD", 2);  // quads per record 
    m_dd->add("MATERIAL_COLOR_OFFSET", (unsigned int)OpticksColors::MATERIAL_COLOR_OFFSET );
    m_dd->add("FLAG_COLOR_OFFSET", (unsigned int)OpticksColors::FLAG_COLOR_OFFSET );
    m_dd->add("PSYCHEDELIC_COLOR_OFFSET", (unsigned int)OpticksColors::PSYCHEDELIC_COLOR_OFFSET );
    m_dd->add("SPECTRAL_COLOR_OFFSET", (unsigned int)OpticksColors::SPECTRAL_COLOR_OFFSET );


    m_scene->write(m_dd);

    m_scene->initRenderers();  // reading shader source and creating renderers

    m_frame->init();           // creates OpenGL context

    m_window = m_frame->getWindow();

    m_scene->setComposition(m_composition);     // defer until renderers are setup 

    // defer creation of the altview to Interactor KEY_U so newly created bookmarks are included
    m_composition->setBookmarks(m_bookmarks);


    TIMER("prepareViz");
} 



void App::loadGeometry()
{
    m_geometry = new OpticksGeometry(m_opticks);

    m_geometry->loadGeometry();

    m_ggeo = m_geometry->getGGeo();


    //// hmm placement ? these are refugees from the OpticksGeometry::registerGeometry

    m_ggeo->setComposition(m_composition);

    if(m_evt)
    {   
       // TODO: profit from migrated OpticksEvent 
        LOG(info) << "OpticksGeometry::registerGeometry " << m_opticks->description() ;
        m_evt->setSpaceDomain(m_opticks->getSpaceDomain());
    }   
}


void App::uploadGeometryViz()
{
    if(isCompute()) return ; 


    OpticksColors* colors = m_opticks->getColors();

    nuvec4 cd = colors->getCompositeDomain() ; 
    glm::uvec4 cd_(cd.x, cd.y, cd.z, cd.w );
  
    m_composition->setColorDomain(cd_); 

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

        if(m_opticks->isDayabay())
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

        bool torchdbg = hasOpt("torchdbg");
        m_torchstep->addStep(torchdbg);  // copyies above configured step settings into the NPY and increments the step index, ready for configuring the next step 

        npy = m_torchstep->getNPY();

        if(torchdbg)
        {
             npy->save("$TMP/torchdbg.npy");
        }
 
    }
    

    TIMER("loadGenstep"); 

    m_evt->setGenstepData(npy); 

    TIMER("setGenstepData"); 
}



void App::targetViz()
{
    if(isCompute()) return ; 

    glm::vec4 mmce = m_geometry->getCenterExtent();
    glm::vec4 gsce = (*m_evt)["genstep.vpos"]->getCenterExtent();
    bool geocenter  = m_fcfg->hasOpt("geocenter");
    glm::vec4 uuce = geocenter ? mmce : gsce ;

    unsigned int target = m_scene->getTarget() ;

    LOG(info) << "App::targetViz"
              << " target " << target      
              << " geocenter " << geocenter      
              << " mmce " << gformat(mmce)
              << " gsce " << gformat(gsce)
              << " uuce " << gformat(uuce)
              ;

    if(target == 0)
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
    LOG(info) << "App::loadEvtFromFile START" ;
   
    bool verbose ; 
    m_evt->loadBuffers(verbose=false);

    if(m_evt->isNoLoad())
        LOG(warning) << "App::loadEvtFromFile LOAD FAILED " ;

    TIMER("loadEvtFromFile"); 
}


void App::uploadEvtViz()
{
    if(isCompute()) return ; 

    if(hasOpt("nooptix|noevent")) 
    {
        LOG(warning) << "App::uploadEvtViz skip due to --nooptix/--noevent " ;
        return ;
    }
 
    LOG(info) << "App::uploadEvtViz START " ;

    m_composition->update();

    m_scene->upload();

    m_scene->uploadSelection();

    m_scene->dump_uploads_table("App::uploadEvtViz");


    TIMER("uploadEvtViz"); 
}



void App::indexPresentationPrep()
{
    LOG(info) << "App::indexPresentationPrep" ; 

    if(!m_evt) return ; 

    Index* seqhis = m_evt->getHistoryIndex() ;
    Index* seqmat = m_evt->getMaterialIndex();
    Index* bndidx = m_evt->getBoundaryIndex();

    if(!seqhis)
    {
         LOG(warning) << "App::indexPresentationPrep NULL seqhis" ;
    }
    else
    {
        OpticksAttrSeq* qflg = m_opticks->getFlagNames();
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
        OpticksAttrSeq* qmat = m_geometry->getMaterialNames();
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
        OpticksAttrSeq* qbnd = m_geometry->getBoundaryNames();
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
    // also see optickscore-/Indexer for another CPU version 

    if(!m_evt) return ; 

    std::map<unsigned int, std::string> boundary_names = m_geometry->getBoundaryNamesMap();

    NPY<float>* dpho = m_evt->getPhotonData();
    if(dpho && dpho->hasData())
    {
        // host based indexing of unique material codes, requires downloadEvt to pull back the photon data
        LOG(info) << "App::indexBoundaries host based " ;
        m_bnd = new BoundariesNPY(dpho); 
        m_bnd->setBoundaryNames(boundary_names); 
        m_bnd->indexBoundaries();     
    } 
    else
    {
        LOG(warning) << "App::indexBoundaries dpho NULL or no data " ;
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
        return ; 
    }


#ifdef WITH_OPTIX 
    LOG(info) << "App::indexEvt WITH_OPTIX" ; 

    indexSequence();

    LOG(info) << "App::indexEvt WITH_OPTIX DONE" ; 
#endif

    indexBoundariesHost();

    TIMER("indexEvt"); 
}


void App::indexEvtOld()
{
    if(!m_evt) return ; 

    // TODO: wean this off use of Types, for the new way (GFlags..)
    Types* types = m_opticks->getTypes();
    Typ* typ = m_opticks->getTyp();

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

            // BELOW NEEDS REVISITING POST ADOPTION OF STRUCTURED RECORDS
            //m_pho->dump(0  ,  "App::indexEvtOld dpho 0");
            //m_pho->dump(100,  "App::indexEvtOld dpho 100" );
            //m_pho->dump(1000, "App::indexEvtOld dpho 1000" );

        }
        m_evt->setRecordsNPY(m_rec);
        m_evt->setPhotonsNPY(m_pho);
    }

    TIMER("indexEvtOld"); 
}




void App::prepareGUI()
{
    if(isCompute()) return ; 

    m_bookmarks->create(0);

#ifdef GUI_

    m_types = m_opticks->getTypes();  // needed for each render
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

    TIMER("prepareGUI"); 
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
    if(isCompute()) return ; 

    m_frame->viewport();
    m_frame->clear();

#ifdef WITH_OPTIX
    if(m_scene->isRaytracedRender() || m_scene->isCompositeRender())
    {
        if(m_opv) m_opv->render();
    }
#endif
    m_scene->render();
}



void App::renderLoop()
{
    if(isCompute()) return ; 

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
#ifdef WITH_NPYSERVER
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
#ifdef WITH_OPTIX
    if(m_ope) m_ope->cleanup();
#endif

#ifdef WITH_NPYSERVER
    if(m_server) m_server->stop();
#endif
#ifdef GUI_
    if(m_gui) m_gui->shutdown();
#endif
    if(m_frame) m_frame->exit();


    m_opticks->cleanup(); 
}


bool App::hasOpt(const char* name)
{
    return m_fcfg->hasOpt(name);
}



#ifdef WITH_OPTIX
void App::prepareOptiX()
{
    LOG(info) << "App::prepareOptiX create OpEngine " ; 
    m_ope = new OpEngine(m_opticks, m_ggeo);
    m_ope->prepareOptiX();
}

void App::prepareOptiXViz()
{
    if(!m_ope) return ; 
    m_opv = new OpViz(m_ope, m_scene); 
}

void App::setupEventInEngine()
{
    if(!m_ope) return ; 
    m_ope->setEvent(m_evt);  // without this cannot index
}

void App::preparePropagator()
{
    if(!m_ope) return ; 
    m_ope->preparePropagator();
}

void App::seedPhotonsFromGensteps()
{
    if(!m_ope) return ; 
    m_ope->seedPhotonsFromGensteps();
    if(hasOpt("dbgseed"))
    {
        dbgSeed();
    }
}

void App::dbgSeed()
{
    OpticksEvent* evt = m_ope->getEvent();    
    NPY<float>* ox = evt->getPhotonData();
    assert(ox);

    if(!isCompute()) 
    { 
        LOG(info) << "App::debugSeed (interop) download photon seeds " ;
        Rdr::download<float>(ox);
        ox->save("$TMP/dbgseed_interop.npy");
    }
    else
    {
        LOG(info) << "App::debugSeed (compute) download photon seeds " ;
        m_ope->downloadPhotonData();  
        ox->save("$TMP/dbgseed_compute.npy");
    }  
}


void App::initRecords()
{
    if(!m_ope) return ; 
    m_ope->initRecords();
}

void App::propagate()
{
    if(hasOpt("nooptix|noevent|nopropagate")) 
    {
        LOG(warning) << "App::propagate skip due to --nooptix/--noevent/--nopropagate " ;
        return ;
    }
    if(!m_ope) return ; 
    m_ope->propagate();
}

void App::saveEvt()
{
    if(!m_ope) return ; 
    if(!isCompute()) 
    {
        Rdr::download(m_evt);
    }
    m_ope->saveEvt();
}

void App::indexSequence()
{
    if(!m_ope)
    {
        LOG(warning) << "App::indexSequence NULL OpEngine " ;
        return ; 
    }

    //m_evt->prepareForIndexing();  // stomps on prior recsel phosel buffers, causes CUDA error with Op indexing, but needed for G4 indexing  
    LOG(info) << "App::indexSequence evt shape " << m_evt->getShapeString() ;

    m_ope->indexSequence();
    LOG(info) << "App::indexSequence DONE" ;
}

#endif




