// brap-
#include "BCfg.hh"
#include "BMap.hh"

#include "NState.hpp"
#include "NLookup.hpp"
#include "NPY.hpp"
#include "NGLM.hpp"
#include "GLMPrint.hpp"
#include "GLMFormat.hpp"

// npy-
#include "Timer.hpp"
#include "TorchStepNPY.hpp"
#include "G4StepNPY.hpp"
#include "Index.hpp"


// numpyserver-
#ifdef WITH_NPYSERVER
#include "numpydelegate.hpp"
#include "numpydelegateCfg.hpp"
#include "numpyserver.hpp"
#endif

// ggeo-
#include "GItemIndex.hh"
#include "GGeo.hh"

// okc-
#include "Bookmarks.hh"
#include "OpticksPhoton.h"
#include "Opticks.hh"
#include "OpticksCfg.hh"
#include "OpticksEvent.hh"
#include "OpticksColors.hh"
#include "Composition.hh"

// opticksgeo-
#include "OpticksHub.hh"
#include "OpticksGeometry.hh"

#include "PLOG.hh"


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


//
// **OpticksHub**
//    Non-viz, hostside intersection of config, geometry and event
//    
//    this means is usable from anywhere, so can mop up config
//

OpticksHub::OpticksHub(Opticks* opticks) 
   :
   m_opticks(opticks),
   m_geometry(NULL),
   m_ggeo(NULL),
   m_composition(new Composition),
   m_evt(NULL),
   m_g4evt(NULL),
   m_okevt(NULL),
   m_nopsteps(NULL),
#ifdef WITH_NPYSERVER
   m_delegate(NULL),
   m_server(NULL)
#endif
   m_cfg(NULL),
   m_fcfg(NULL),
   m_state(NULL),
   m_lookup(new NLookup()),
   m_bookmarks(NULL)
{
   init();
}



void OpticksHub::init()
{
    m_cfg  = new BCfg("umbrella", false) ; 
    m_fcfg = m_opticks->getCfg();
    add(m_fcfg);

#ifdef WITH_NPYSERVER
    m_delegate    = new numpydelegate ; 
    add(new numpydelegateCfg<numpydelegate>("numpydelegate", m_delegate, false));
#endif

}

bool OpticksHub::hasOpt(const char* name)
{
    return m_fcfg->hasOpt(name);
}
bool OpticksHub::isCompute()
{
    return m_opticks->isCompute();
}
std::string OpticksHub::getCfgString()
{
    return m_cfg->getDescString();
}
OpticksCfg<Opticks>* OpticksHub::getCfg()
{
    return m_fcfg ; 
}
GGeo* OpticksHub::getGGeo()
{
    return m_ggeo ; 
}
NState* OpticksHub::getState()
{
    return m_state ; 
}
NLookup* OpticksHub::getLookup()
{
    return m_lookup ; 
}


Opticks* OpticksHub::getOpticks()
{
    return m_opticks ; 
}
Composition* OpticksHub::getComposition()
{
    return m_composition ;  
}
Bookmarks* OpticksHub::getBookmarks()
{
    return m_bookmarks ; 
}
Timer* OpticksHub::getTimer()
{
    return m_evt ? m_evt->getTimer() : m_opticks->getTimer() ; 
}




void OpticksHub::add(BCfg* cfg)
{
    m_cfg->add(cfg); 
}


void OpticksHub::configure()
{
    m_composition->addConfig(m_cfg); 
    //m_cfg->dumpTree();

    int argc    = m_opticks->getArgc();
    char** argv = m_opticks->getArgv();

    LOG(debug) << "OpticksHub::configure " << argv[0] ; 

    m_cfg->commandline(argc, argv);
    m_opticks->configure();      

    if(m_fcfg->hasError())
    {
        LOG(fatal) << "OpticksHub::config parse error " << m_fcfg->getErrorMessage() ; 
        m_fcfg->dump("OpticksHub::config m_fcfg");
        m_opticks->setExit(true);
        return ; 
    }


    bool compute = m_opticks->isCompute();
    bool compute_opt = hasOpt("compute") ;
    if(compute && !compute_opt)
        LOG(warning) << "OpticksHub::configure FORCED COMPUTE MODE : as remote session detected " ;  


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
        LOG(fatal) << "OpticksHub::configure OPTICKS INVALID : missing envvar or geometry path ?" ;
        assert(0);
    }
#ifdef WITH_NPYSERVER
    configureServer();
#endif

    configureCompositionSize();

    configureLookup();

    TIMER("configure");
}


void OpticksHub::configureLookup()
{
    const char* path = m_opticks->getMaterialMap(); 
    const char* prefix = m_opticks->getMaterialPrefix(); 

    LOG(info) << "OpticksHub::configureLookup"
              << " loading genstep material index map "
              << " path " << path
              << " prefix " << prefix
              ;

    std::map<std::string, unsigned> A ; 
    BMap<std::string, unsigned int>::load(&A, path ); 
    setMaterialMap(A, prefix);
}

void OpticksHub::setMaterialMap( std::map<std::string, unsigned>& materialMap, const char* prefix )
{
   // this must be done prior to loading geometry to take effect
    m_lookup->setA(materialMap, prefix ); 
}


#ifdef WITH_NPYSERVER
void OpticksHub::configureServer()
{
    if(!hasOpt("nonet"))
    {
      // MAYBE liveConnect should happen in initialization, not here now that event creation happens latter 
        m_delegate->liveConnect(m_cfg); // hookup live config via UDP messages

        try { 
            m_server = new numpyserver<numpydelegate>(m_delegate); // connect to external messages 
        } 
        catch( const std::exception& e)
        {
            LOG(fatal) << "OpticksHub::configureServer EXCEPTION " << e.what() ; 
            LOG(fatal) << "OpticksHub::configureServer FAILED to instanciate numpyserver : probably another instance is running : check debugger sessions " ;
        }
    }
}
#endif


void OpticksHub::configureCompositionSize()
{
    glm::uvec4 size = m_opticks->getSize();
    glm::uvec4 position = m_opticks->getPosition() ;

    LOG(info) << "OpticksHub::configureCompositionSize"
              << " size " << gformat(size)
              << " position " << gformat(position)
              ;

    m_composition->setSize( size );
    m_composition->setFramePosition( position );
}


void OpticksHub::configureState(NConfigurable* scene)
{
    // NState manages the state (in the form of strings) of a collection of NConfigurable objects
    // this needs to happen after configuration and the scene is created

    m_state = m_opticks->getState();  
    m_state->setVerbose(false);

    LOG(info) << "OpticksHub::configureViz " << m_state->description();

    m_state->addConfigurable(scene);
    m_composition->addConstituentConfigurables(m_state); // constituents: trackball, view, camera, clipper

    m_bookmarks   = new Bookmarks(m_state->getDir()) ; 
    m_bookmarks->setState(m_state);
    m_bookmarks->setVerbose();
    m_bookmarks->setInterpolatedViewPeriod(m_fcfg->getInterpolatedViewPeriod());

    m_composition->setBookmarks(m_bookmarks);

    m_composition->setOrbitalViewPeriod(m_fcfg->getOrbitalViewPeriod()); 
    m_composition->setAnimatorPeriod(m_fcfg->getAnimatorPeriod()); 
}


NPY<unsigned char>* OpticksHub::getColorBuffer()
{
    OpticksColors* colors = m_opticks->getColors();

    nuvec4 cd = colors->getCompositeDomain() ; 
    glm::uvec4 cd_(cd.x, cd.y, cd.z, cd.w );
  
    m_composition->setColorDomain(cd_); 

    return colors->getCompositeBuffer() ;
}


OpticksEvent* OpticksHub::initOKEvent(NPY<float>* gs)
{
    // Opticks OK events are created with gensteps (Scintillation+Cerenkov) 
    // from a G4 event (the G4 event can either be loaded from file 
    // or directly obtained from live G4)
    //
    bool ok = true ; 
    createEvent(ok); 
    m_okevt->setGenstepData(gs);
    assert(m_evt == m_okevt);
    return m_okevt ; 
}


OpticksEvent* OpticksHub::loadPersistedEvent()
{
    // should this handle both G4 and OK evts ?

    bool ok = true ; 
    createEvent(ok);
    loadEventBuffers();
    assert(m_evt == m_okevt);
    return m_okevt ; 
}


void OpticksHub::loadEventBuffers()
{
    LOG(info) << "OpticksHub::loadEventBuffers START" ;
   
    bool verbose ; 
    m_evt->loadBuffers(verbose=false);

    if(m_evt->isNoLoad())
        LOG(warning) << "OpticksHub::loadEventBuffers LOAD FAILED " ;

    TIMER("loadEvent"); 
}


OpticksEvent* OpticksHub::createG4Event()
{
    return createEvent(false);
}
OpticksEvent* OpticksHub::createOKEvent()
{
    return createEvent(true);
}


OpticksEvent* OpticksHub::createEvent(bool ok)
{
    m_evt = m_opticks->makeEvent(ok) ; 
    if(ok)
    {
        delete m_okevt ;
        m_okevt = NULL ; 

        m_okevt = m_evt ; 
        assert(m_okevt->isOK());
    }
    else
    {
        delete m_g4evt ;
        m_g4evt = NULL ; 
        m_nopsteps = NULL ; 

        m_g4evt = m_evt ;
        m_nopsteps = m_g4evt->getNopstepData(); 
        assert(m_g4evt->isG4());
    }
    configureEvent(m_evt);
    return m_evt ; 
}

NPY<float>* OpticksHub::getNopsteps()
{
    return m_nopsteps ; 
}

OpticksEvent* OpticksHub::getG4Event()
{
    return m_g4evt ; 
}
OpticksEvent* OpticksHub::getOKEvent()
{
    return m_okevt ; 
}
OpticksEvent* OpticksHub::getEvent()
{
    return m_evt ; 
}





void OpticksHub::configureEvent(OpticksEvent* evt)
{
    if(!evt) return 

#ifdef WITH_NPYSERVER
    if(m_delegate)
    {
        m_delegate->setEvent(evt); // allows delegate to update evt when NPY messages arrive, hmm locking needed ?
    }
#endif

    m_composition->setEvt(evt);  // look like used only for Composition::setPickPhoton  TODO: reposition this 
    m_composition->setTrackViewPeriod(m_fcfg->getTrackViewPeriod()); 

    bool quietly = true ;       
    NPY<float>* track = evt->loadGenstepDerivativeFromFile("track", quietly);
    m_composition->setTrack(track);
}








void OpticksHub::loadGeometry()
{
    m_geometry = new OpticksGeometry(this);

    m_geometry->loadGeometry();

    m_ggeo = m_geometry->getGGeo();

    m_ggeo->setComposition(m_composition);

}


NPY<float>* OpticksHub::loadGenstep()
{
    if(hasOpt("nooptix|noevent")) 
    {
        LOG(warning) << "OpticksHub::loadGenstep skip due to --nooptix/--noevent " ;
        return NULL ;
    }

    unsigned int code = m_opticks->getSourceCode();

    NPY<float>* gs = NULL ; 
    if( code == CERENKOV || code == SCINTILLATION || code == NATURAL )
    {
        gs = loadGenstepFile(); 
    }
    else if(code == TORCH)
    {
        gs = loadGenstepTorch(); 
    }

    TIMER("loadGenstep"); 

    return gs ; 
}


void OpticksHub::translateGensteps(NPY<float>* gs)
{
    G4StepNPY* g4step = new G4StepNPY(gs);    
    g4step->relabel(CERENKOV, SCINTILLATION); 

    // which code is used depends in the sign of the pre-label 
    // becomes the ghead.i.x used in cu/generate.cu

    if(m_opticks->isDayabay())
    {   
        // within GGeo this depends on GBndLib
        //NLookup* lookup = m_ggeo ? m_ggeo->getLookup() : NULL ;

        if(m_lookup)
        {  
            g4step->setLookup(m_lookup);   
            g4step->applyLookup(0, 2);  // jj, kk [1st quad, third value] is materialIndex
            //
            // replaces original material indices with material lines
            // for easy access to properties using boundary_lookup GPU side
            //
        }
        else
        {
            LOG(warning) << "OpticksHub::translateGensteps not applying lookup" ;
        } 
    }
}


NPY<float>* OpticksHub::loadGenstepFile()
{
    NPY<float>* gs = m_opticks->loadGenstep();
    if(gs == NULL) LOG(fatal) << "OpticksHub::loadGenstepFile FAILED" ;
    assert(gs);

    int modulo = m_fcfg->getModulo();

    //m_parameters->add<std::string>("genstepOriginal",   gs->getDigestString()  );
    //m_parameters->add<int>("Modulo", modulo );

    if(modulo > 0) 
    {    
        LOG(warning) << "OptickHub::loadGenstepFile applying modulo scaledown " << modulo ;
        gs = NPY<float>::make_modulo(gs, modulo);
        //m_parameters->add<std::string>("genstepModulo",   genstep->getDigestString()  );
    }    

    translateGensteps(gs) ;
    return gs ; 
}


NPY<float>* OpticksHub::loadGenstepTorch()
{
    TorchStepNPY* torchstep = m_opticks->makeSimpleTorchStep();

    if(m_ggeo)
    {
        m_ggeo->targetTorchStep(torchstep);
        const char* material = torchstep->getMaterial() ;
        unsigned int matline = m_ggeo->getMaterialLine(material);
        torchstep->setMaterialLine(matline);  

        LOG(debug) << "OpticksHub::loadGenstepTorch"
                   << " config " << torchstep->getConfig() 
                   << " material " << material 
                   << " matline " << matline
                         ;
    }
    else
    {
        LOG(warning) << "OpticksHub::loadGenstepTorch no ggeo, skip setting torchstep material line " ;
    } 

    bool torchdbg = hasOpt("torchdbg");
    torchstep->addStep(torchdbg);  // copyies above configured step settings into the NPY and increments the step index, ready for configuring the next step 

    NPY<float>* gs = torchstep->getNPY();
    if(torchdbg)
    {
        gs->save("$TMP/torchdbg.npy");
    }
    return gs ; 
}


void OpticksHub::targetGenstep()
{
    LOG(fatal) << "OpticksHub::targetGenstep"
               << " m_evt " << m_evt 
               ; 

    bool geocenter  = hasOpt("geocenter");
    bool autocam = true ; 
    if(geocenter && m_geometry != NULL )
    {
        glm::vec4 mmce = m_geometry->getCenterExtent();
        m_composition->setCenterExtent( mmce , autocam );
        LOG(info) << "OpticksHub::targetGenstep (geocenter) mmce " << gformat(mmce) ; 
    }
    else if(m_evt)
    {
        glm::vec4 gsce = m_evt->getGenstepCenterExtent();  // need to setGenStepData before this will work 
        m_composition->setCenterExtent( gsce , autocam );
        LOG(info) << "OpticksHub::targetGenstep (!geocenter) gsce " << gformat(gsce) ; 
    }
}





void OpticksHub::cleanup()
{
#ifdef WITH_NPYSERVER
    if(m_server) m_server->stop();
#endif
}


OpticksAttrSeq* OpticksHub::getFlagNames()
{
    return m_opticks->getFlagNames();
}
OpticksAttrSeq* OpticksHub::getMaterialNames()
{
    return m_geometry->getMaterialNames();
}
OpticksAttrSeq* OpticksHub::getBoundaryNames()
{
    return m_geometry->getBoundaryNames();
}
std::map<unsigned int, std::string> OpticksHub::getBoundaryNamesMap()
{
    return m_geometry->getBoundaryNamesMap();
}




