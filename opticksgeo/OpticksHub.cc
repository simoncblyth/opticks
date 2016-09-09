// sysrap-
#include "SLog.hh"

// brap-
#include "BCfg.hh"
#include "BMap.hh"

#include "NState.hpp"
#include "NLookup.hpp"
#include "NPY.hpp"
#include "NGLM.hpp"
#include "GLMPrint.hpp"
#include "GLMFormat.hpp"
#include "Parameters.hpp"

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
#include "OpticksActionControl.hh"
#include "Composition.hh"

// opticksgeo-
#include "OpticksHub.hh"
#include "OpticksGen.hh"
#include "OpticksGun.hh"
#include "OpticksRun.hh"
#include "OpticksGeometry.hh"

#include "PLOG.hh"


#define TIMER(s) \
    { \
       if(m_evt)\
       {\
          Timer& t = *(m_evt->getTimer()) ;\
          t((s)) ;\
       }\
       else if(m_ok) \
       {\
          Timer& t = *(m_ok->getTimer()) ;\
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
   m_log(new SLog("OpticksHub::OpticksHub")),
   m_ok(opticks),
   m_geometry(NULL),
   m_ggeo(NULL),
   m_composition(new Composition),
#ifdef WITH_NPYSERVER
   m_delegate(NULL),
   m_server(NULL)
#endif
   m_cfg(new BCfg("umbrella", false)),
   m_fcfg(m_ok->getCfg()),
   m_state(NULL),
   m_lookup(new NLookup()),
   m_bookmarks(NULL),
   m_gen(NULL),
   m_gun(NULL),
   m_run(NULL)
{
   init();
   (*m_log)("DONE");
}


void OpticksHub::init()
{
    add(m_fcfg);

    configure();
    configureServer();
    configureCompositionSize();
    configureLookupA();
    loadGeometry() ;    

    m_gen = new OpticksGen(this) ;
    m_gun = new OpticksGun(this) ;
    m_run = new OpticksRun(this) ;
}

void OpticksHub::configure()
{
    m_composition->addConfig(m_cfg); 
    //m_cfg->dumpTree();

    int argc    = m_ok->getArgc();
    char** argv = m_ok->getArgv();

    LOG(debug) << "OpticksHub::configure " << argv[0] ; 

    m_cfg->commandline(argc, argv);
    m_ok->configure();      

    if(m_fcfg->hasError())
    {
        LOG(fatal) << "OpticksHub::config parse error " << m_fcfg->getErrorMessage() ; 
        m_fcfg->dump("OpticksHub::config m_fcfg");
        m_ok->setExit(true);
        return ; 
    }

    bool compute = m_ok->isCompute();
    bool compute_opt = hasOpt("compute") ;
    if(compute && !compute_opt)
        LOG(warning) << "OpticksHub::configure FORCED COMPUTE MODE : as remote session detected " ;  


    if(hasOpt("idpath")) std::cout << m_ok->getIdPath() << std::endl ;
    if(hasOpt("help"))   std::cout << m_cfg->getDesc()     << std::endl ;
    if(hasOpt("help|version|idpath"))
    {
        m_ok->setExit(true);
        return ; 
    }

    if(!m_ok->isValid())
    {
        // defer death til after getting help
        LOG(fatal) << "OpticksHub::configure OPTICKS INVALID : missing envvar or geometry path ?" ;
        assert(0);
    }
}


void OpticksHub::configureServer()
{
#ifdef WITH_NPYSERVER

    m_delegate    = new numpydelegate ; 
    add(new numpydelegateCfg<numpydelegate>("numpydelegate", m_delegate, false));

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
#endif
}

void OpticksHub::configureCompositionSize()
{
    glm::uvec4 size = m_ok->getSize();
    glm::uvec4 position = m_ok->getPosition() ;

    LOG(debug) << "OpticksHub::configureCompositionSize"
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

    m_state = m_ok->getState();  
    m_state->setVerbose(false);

    LOG(trace) << "OpticksHub::configureState " << m_state->description();

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

void OpticksHub::configureLookupA()
{
    const char* path = m_ok->getMaterialMap(); 
    const char* prefix = m_ok->getMaterialPrefix(); 

    LOG(debug) << "OpticksHub::configureLookupA"
              << " loading genstep material index map "
              << " path " << path
              << " prefix " << prefix
              ;

    std::map<std::string, unsigned> A ; 
    BMap<std::string, unsigned int>::load(&A, path ); 

    m_lookup->setA(A, prefix, path);
}

void OpticksHub::overrideMaterialMapA(const std::map<std::string, unsigned>& A, const char* msg)
{
   // Used from OKG4Mgr to override the default mapping when using G4 steps directly 
    m_lookup->setA( A, "", msg);
}

void OpticksHub::loadGeometry()
{
    assert(m_geometry == NULL && "OpticksHub::loadGeometry should only be called once");

    LOG(debug) << "OpticksHub::loadGeometry" ; 

    m_geometry = new OpticksGeometry(this);   // m_lookup is set into m_ggeo here 

    m_geometry->loadGeometry();   

    //   Lookup A and B are now set ...
    //      A : by OpticksHub::configureLookupA (ChromaMaterialMap.json)
    //      B : on GGeo loading in GGeo::setupLookup

    m_ggeo = m_geometry->getGGeo();

    m_ggeo->setComposition(m_composition);

    LOG(debug) << "OpticksHub::loadGeometry DONE" ; 
}


std::string OpticksHub::getG4GunConfig()
{
    return m_gun->getConfig();
}

TorchStepNPY* OpticksHub::getTorchstep()   // needed by CGenerator
{
    return m_gen->getTorchstep() ; 
}


bool OpticksHub::hasOpt(const char* name)
{
    return m_fcfg->hasOpt(name);
}
bool OpticksHub::isCompute()
{
    return m_ok->isCompute();
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
    return m_ok ; 
}
Composition* OpticksHub::getComposition()
{
    return m_composition ;  
}
OpticksGeometry* OpticksHub::getGeometry()
{
    return m_geometry ;  
}
Bookmarks* OpticksHub::getBookmarks()
{
    return m_bookmarks ; 
}

Timer* OpticksHub::getTimer()
{
    OpticksEvent* evt = m_run->getEvent();
    return evt ? evt->getTimer() : m_ok->getTimer() ; 
}


OpticksGen* OpticksHub::getGen()
{
    return m_gen ; 
}
OpticksRun* OpticksHub::getRun()
{
    return m_run ; 
}



void OpticksHub::add(BCfg* cfg)
{
    m_cfg->add(cfg); 
}



NPY<unsigned char>* OpticksHub::getColorBuffer()
{
    OpticksColors* colors = m_ok->getColors();

    nuvec4 cd = colors->getCompositeDomain() ; 
    glm::uvec4 cd_(cd.x, cd.y, cd.z, cd.w );
  
    m_composition->setColorDomain(cd_); 

    return colors->getCompositeBuffer() ;
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

    NPY<float>* track = evt->loadGenstepDerivativeFromFile("track");
    m_composition->setTrack(track);
}





OpticksEvent* OpticksHub::getG4Event()
{
    return m_run->getG4Event() ; 
}
OpticksEvent* OpticksHub::getEvent()
{
    return m_run->getEvent() ; 
}








unsigned OpticksHub::getTarget()
{
   return m_geometry->getTarget();
}
void OpticksHub::setTarget(unsigned target, bool aim)
{
    m_geometry->setTarget(target, aim );
}


void OpticksHub::target()
{
    int target = m_geometry ? m_geometry->getTarget() : 0 ;
    bool geocenter  = hasOpt("geocenter");
    bool autocam = true ; 

    OpticksEvent* evt = m_run->getEvent();

    if(target != 0)
    {
        LOG(info) << "OpticksHub::target SKIP as geometry target already set  " << target ; 
    }
    else if(geocenter && m_geometry != NULL )
    {
        glm::vec4 mmce = m_geometry->getCenterExtent();
        m_composition->setCenterExtent( mmce , autocam );
        LOG(info) << "OpticksHub::target (geocenter) mmce " << gformat(mmce) ; 
    }
    else if(evt && evt->hasGenstepData())
    {
        glm::vec4 gsce = evt->getGenstepCenterExtent();  // need to setGenStepData before this will work 
        m_composition->setCenterExtent( gsce , autocam );
        LOG(info) << "OpticksHub::target"
                  << " evt " << evt->brief()
                  << " gsce " << gformat(gsce) 
                  ; 
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
    return m_ok->getFlagNames();
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




