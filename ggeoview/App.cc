#include <cstring>

// sysrap-
#include "SSys.hh"

// brap-
#include "BStr.hh"

// npy-

class NLookup ; 

#include "NGLM.hpp"
#include "NState.hpp"
#include "NPY.hpp"
#include "GLMPrint.hpp"
#include "GLMFormat.hpp"
#include "ViewNPY.hpp"
#include "MultiViewNPY.hpp"



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
#include "OpticksHub.hh"


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
      m_hub(NULL),
      m_prefix(strdup(prefix)),
      m_parameters(NULL),
      m_timer(NULL),
      m_scene(NULL),
      m_composition(NULL),
      m_frame(NULL),
      m_window(NULL),
      m_interactor(NULL),

      m_evt(NULL), 
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
      m_gui(NULL)
{
    init(argc, argv);
}




void App::init(int argc, char** argv)
{
    m_opticks = new Opticks(argc, argv);
    m_opticks->Summary("App::init OpticksResource::Summary");


    m_hub = new OpticksHub(m_opticks) ;

    // TRANSITIONAL
    m_composition = m_hub->getComposition(); 
    m_evt = m_hub->getEvent(); 


    TIMER("init");
}

bool App::isCompute()
{
    return m_opticks->isCompute() ;
}

bool App::isExit()
{
    return m_opticks->isExit() ; 
}

void App::initViz()
{
    if(m_opticks->isCompute()) return ; 

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
    m_interactor->setComposition(m_hub->getComposition());

    m_scene->setInteractor(m_interactor);      

    m_frame->setInteractor(m_interactor);      
    m_frame->setComposition(m_hub->getComposition());
    m_frame->setScene(m_scene);

    m_hub->add(new SceneCfg<Scene>(           "scene",       m_scene,                      true));
    m_hub->add(new RendererCfg<Renderer>(     "renderer",    m_scene->getGeometryRenderer(), true));
    m_hub->add(new InteractorCfg<Interactor>( "interactor",  m_interactor,                 true));

    TIMER("initViz");
}


void App::configure(int argc, char** argv)
{
    LOG(debug) << "App:configure " << argv[0] ; 

    m_hub->configure(argc, argv); 

    configureViz();

    TIMER("configure");
}



void App::configureViz()
{
    if(isCompute()) return ; 

    m_hub->configureViz(m_scene) ;

    if(m_interactor)
    {
        m_interactor->setBookmarks(m_hub->getBookmarks());
    }

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

    m_scene->setEvent(m_hub->getEvent());
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


    // hmm these shoud be inside hub
    Composition* composition = m_hub->getComposition();
    composition->setSize( m_size );
    composition->setFramePosition( position );

    m_frame->setTitle("GGeoView");
    m_frame->setFullscreen(hasOpt("fullscreen"));

    BDynamicDefine* dd = m_opticks->makeDynamicDefine(); 
    m_scene->write(dd);

    if(hasOpt("dbginterop"))
    {
        m_scene->initRenderersDebug();  // reading shader source and creating subset of renderers
    }
    else
    {
        m_scene->initRenderers();  // reading shader source and creating renderers
    }

    m_frame->init();           // creates OpenGL context

    m_window = m_frame->getWindow();

    m_scene->setComposition(composition);     // defer until renderers are setup 

    // defer creation of the altview to Interactor KEY_U so newly created bookmarks are included
    // m_composition->setBookmarks(m_bookmarks);


    TIMER("prepareViz");
} 



void App::loadGeometry()
{
    m_hub->loadGeometry();
}

void App::loadGenstep()
{
    m_hub->loadGenstep();
}


void App::uploadGeometryViz()
{
    if(isCompute()) return ; 


    OpticksColors* colors = m_opticks->getColors();

    nuvec4 cd = colors->getCompositeDomain() ; 
    glm::uvec4 cd_(cd.x, cd.y, cd.z, cd.w );
  
    m_composition->setColorDomain(cd_); 

    m_scene->uploadColorBuffer( colors->getCompositeBuffer() );  //     oglrap-/Colors preps texture, available to shaders as "uniform sampler1D Colors"


    // where does this info come from, where is most appropriate to set it ?
    m_composition->setTimeDomain(        m_opticks->getTimeDomain() );
    m_composition->setDomainCenterExtent(m_opticks->getSpaceDomain());

    m_scene->setGeometry(m_hub->getGGeo());

    m_scene->uploadGeometry();

    bool autocam = true ; 

    // handle commandline --target option that needs loaded geometry 
    unsigned int target = m_scene->getTargetDeferred();   // default to 0 
    LOG(debug) << "App::uploadGeometryViz setting target " << target ; 

    m_scene->setTarget(target, autocam);
 
    TIMER("uploadGeometryViz"); 
}






void App::targetViz()
{
    if(isCompute()) return ; 

    unsigned int target = m_scene->getTarget() ;

    // only pointing based on genstep if not already targetted
    if(target == 0)
    {
        m_hub->targetGenstep();
    }

    m_scene->setRecordStyle( m_hub->hasOpt("alt") ? Scene::ALTREC : Scene::REC );    

    TIMER("targetViz"); 
}


void App::loadEvtFromFile()
{
    m_hub->loadEvent();
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

    if(hasOpt("dbguploads"))
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

        GGeo* ggeo = m_hub->getGGeo();

        m_hit = new HitsNPY(ox, ggeo->getSensorList());
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

    Bookmarks* bookmarks=m_hub->getBookmarks();

    bookmarks->create(0);

#ifdef GUI_

    m_types = m_opticks->getTypes();  // needed for each render
    m_photons = new Photons(m_types, m_boundaries, m_seqhis, m_seqmat ) ; // GUI jacket 
    m_scene->setPhotons(m_photons);

    m_gui = new GUI(m_hub->getGGeo()) ;
    m_gui->setScene(m_scene);
    m_gui->setPhotons(m_photons);
    m_gui->setComposition(m_hub->getComposition());
    m_gui->setBookmarks(bookmarks);
    m_gui->setStateGUI(new StateGUI(m_hub->getState()));
    m_gui->setInteractor(m_interactor);   // status line
    
    m_gui->init(m_window);
    m_gui->setupHelpText( m_hub->getCfgString() );

    OpticksEvent* evt = m_hub->getEvent();

    TimesTable* tt = evt ? evt->getTimesTable() : NULL ; 
    if(tt)
    {
        m_gui->setupStats(tt->getLines());
    }
    else
    {
        LOG(warning) << "App::prepareGUI NULL TimesTable " ; 
    }  

    Parameters* parameters = evt ? evt->getParameters() : m_parameters ; 

    m_gui->setupParams(parameters->getLines());

#endif

    TIMER("prepareGUI"); 
}


void App::renderGUI()
{
#ifdef GUI_
    m_gui->newframe();
    bool* show_gui_window = m_interactor->getGUIModeAddress();
    Composition* composition = m_hub->getComposition();
    if(*show_gui_window)
    {
        m_gui->show(show_gui_window);
        if(m_photons)
        {
            if(m_boundaries)
            {
                m_composition->getPick().y = m_boundaries->getSelected() ;   //  1st boundary 
            }
            glm::ivec4& recsel = composition->getRecSelect();
            recsel.x = m_seqhis ? m_seqhis->getSelected() : 0 ; 
            recsel.y = m_seqmat ? m_seqmat->getSelected() : 0 ; 
            composition->setFlags(m_types->getFlags()); 
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
    
    bool noviz = hasOpt("noviz") ;
    int interactivity = SSys::GetInteractivityLevel() ;
    if(noviz || interactivity == 0 )
    {
        LOG(info) << "App::renderLoop early exit due to --noviz/-V option OR SSys::GetInteractivityLevel 0  " ; 
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


#ifdef GUI_
    if(m_gui) m_gui->shutdown();
#endif
    if(m_frame) m_frame->exit();

    m_hub->cleanup();
    m_opticks->cleanup(); 
}


bool App::hasOpt(const char* name)
{
    return m_hub->hasOpt(name);
}



#ifdef WITH_OPTIX
void App::prepareOptiX()
{
    LOG(info) << "App::prepareOptiX create OpEngine " ; 
    GGeo* ggeo = m_hub->getGGeo();
    m_ope = new OpEngine(m_opticks, ggeo);
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
    OpticksEvent* evt = m_hub->getEvent();
    m_ope->setEvent(evt);  // without this cannot index
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
        OpticksEvent* evt = m_hub->getEvent();
        Rdr::download(evt);
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

    OpticksEvent* evt = m_hub->getEvent(); 
    LOG(info) << "App::indexSequence evt shape " << evt->getShapeString() ;

    m_ope->indexSequence();
    LOG(info) << "App::indexSequence DONE" ;
}

#endif




