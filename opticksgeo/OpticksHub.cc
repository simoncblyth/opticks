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

OpticksHub::OpticksHub(Opticks* opticks, bool immediate) 
   :
   m_log(new SLog("OpticksHub::OpticksHub")),
   m_ok(opticks),
   m_immediate(immediate),
   m_geometry(NULL),
   m_ggeo(NULL),
   m_composition(new Composition),
   m_evt(NULL),
   m_g4evt(NULL),
   m_okevt(NULL),
   m_nopsteps(NULL),
   m_gensteps(NULL),
   m_torchstep(NULL),
   m_g4step(NULL),
   m_zero(NULL),
#ifdef WITH_NPYSERVER
   m_delegate(NULL),
   m_server(NULL)
#endif
   m_cfg(new BCfg("umbrella", false)),
   m_fcfg(m_ok->getCfg()),
   m_state(NULL),
   m_lookup(new NLookup()),
   m_bookmarks(NULL)
{
   init();
   (*m_log)("DONE");
}


void OpticksHub::init()
{
    add(m_fcfg);

#ifdef WITH_NPYSERVER
    m_delegate    = new numpydelegate ; 
    add(new numpydelegateCfg<numpydelegate>("numpydelegate", m_delegate, false));
#endif

   if(m_immediate) 
   {
       configure();
       loadGeometry() ;    

       setupInputGensteps();
   }
}


void OpticksHub::loadGeometry()
{
    if(m_geometry) 
    {
        LOG(warning) << "OpticksHub::loadGeometry ALREADY LOADED "   ;
        return ; 
    }

    LOG(debug) << "OpticksHub::loadGeometry" ; 

    m_geometry = new OpticksGeometry(this);   // m_lookup is set into m_ggeo here 

    m_geometry->loadGeometry();   

    //   lookup A and B are now set ...
    //
    //   A : by OpticksHub::configureLookup (ChromaMaterialMap.json)
    //   B : on GGeo loading in GGeo::setupLookup
    //

    m_ggeo = m_geometry->getGGeo();

    m_ggeo->setComposition(m_composition);

    LOG(debug) << "OpticksHub::loadGeometry DONE" ; 

}



void OpticksHub::setupInputGensteps()
{
    LOG(debug) << "OpticksHub::setupInputGensteps" ; 

    m_zero = m_ok->makeEvent(true) ;  // needs to be after configure for spec to be defined

    unsigned int code = m_ok->getSourceCode();

    NPY<float>* gs = NULL ; 

    if(code == TORCH)
    {
        m_torchstep = makeTorchstep() ;
        gs = m_torchstep->getNPY();
        gs->addActionControl(OpticksActionControl::Parse("GS_FABRICATED,GS_TORCH"));
        setGensteps(gs);
    }
    else if( code == CERENKOV || code == SCINTILLATION || code == NATURAL )
    {
        gs = loadGenstepFile();
        gs->addActionControl(OpticksActionControl::Parse("GS_LOADED,GS_LEGACY"));
        setGensteps(gs);
    }
    else if( code == G4GUN  )
    {
        if(m_ok->isIntegrated())
        {
             LOG(info) << " integrated G4GUN running, gensteps will be collected from G4 directly " ;  
        }
        else
        {
             LOG(info) << " non-integrated G4GUN running, attempt to load gensteps from file " ;  
             gs = loadGenstepFile();
             gs->addActionControl(OpticksActionControl::Parse("GS_LOADED"));
             setGensteps(gs);
        }
    }

   if(gs)
   {
       m_zero->setGenstepData(gs);
   }
}



std::string OpticksHub::getG4GunConfig()
{
    std::string config ; 
    int itag = m_ok->getEventITag();

    if( itag == 1 )
         config.assign(
    "comment=default-config-comment-without-spaces-_"
    "particle=mu-_"
    "frame=3153_"
    "position=0,0,-1_"
    "direction=0,0,1_"
    "polarization=1,0,0_"
    "time=0.1_"
    "energy=1000.0_"
    "number=1_")
    ;  // mm,ns,MeV 

    else if(itag == 100)
         config.assign(
    "comment=default-config-comment-without-spaces-_"
    "particle=mu-_"
    "frame=3153_"
    "position=0,0,-1_"
    "direction=0,0,1_"
    "polarization=1,0,0_"
    "time=0.1_"
    "energy=100000.0_"
    "number=1_")
    ;  // mm,ns,MeV 




    LOG(info) << "OpticksHub::getG4GunConfig"
              << " itag : " << itag 
              << " config : " << config 
              ; 

    return config ; 
}


NPY<float>* OpticksHub::loadGenstepFile()
{
    NPY<float>* gs = m_ok->loadGenstep();
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

    return gs ; 
}




NPY<float>* OpticksHub::getGensteps()
{
    return m_gensteps ; 
}
TorchStepNPY* OpticksHub::getTorchstep()
{
    return m_torchstep ; 
}
G4StepNPY* OpticksHub::getG4Step()
{
    return m_g4step ;  
}
OpticksEvent* OpticksHub::getZeroEvent()
{
    return m_zero ;  
}




// hmm confusing to have this here, rather than in OpticksEvent ?
//
void OpticksHub::setGensteps(NPY<float>* gs)
{
    gs->setBufferSpec(OpticksEvent::GenstepSpec());

    m_gensteps = gs ; 
    m_g4step = new G4StepNPY(gs);    

    OpticksActionControl oac(gs->getActionControlPtr());
    bool gs_torch = oac.isSet("GS_TORCH") ; 
    bool gs_legacy = oac.isSet("GS_LEGACY") ; 

    LOG(fatal) << "OpticksHub::setGensteps"
               << " shape " << gs->getShapeString()
               << " " << oac.description("oac")
               ;


    if(gs_legacy)
    {
        assert(!gs_torch); // there are no legacy torch files ?

        m_g4step->relabel(CERENKOV, SCINTILLATION); 
        // CERENKOV or SCINTILLATION codes are used depending on 
        // the sign of the pre-label 
        // this becomes the ghead.i.x used in cu/generate.cu
        // which dictates what to generate

        assert(m_lookup);
        m_lookup->close("OpticksHub::setGensteps GS_LEGACY");

        m_g4step->setLookup(m_lookup);   
        m_g4step->applyLookup(0, 2);  // jj, kk [1st quad, third value] is materialIndex
        // replaces original material indices with material lines
        // for easy access to properties using boundary_lookup GPU side

    }
    else if(gs_torch)
    {
        LOG(debug) << " checklabel of torch steps  " << oac.description("oac") ; 
        m_g4step->checklabel(TORCH); 
    }
    else
    {
        // non-legacy gensteps (ie collected direct from G4) already 
        // have proper labelling and lookup applied during collection

        m_g4step->checklabel(CERENKOV, SCINTILLATION);
    }

    m_g4step->countPhotons();

    LOG(info) 
         << " Keys "
         << " TORCH: " << TORCH 
         << " CERENKOV: " << CERENKOV 
         << " SCINTILLATION: " << SCINTILLATION  
         << " G4GUN: " << G4GUN  
         ;

     LOG(info) 
         << " counts " 
         << m_g4step->description()
         ;
 
}



TorchStepNPY* OpticksHub::makeTorchstep()
{
    TorchStepNPY* torchstep = m_ok->makeSimpleTorchStep();

    if(m_ggeo)
    {
        m_ggeo->targetTorchStep(torchstep);   // sets frame transform of the torchstep

        // translation from a string name from config into a mat line
        // only depends on the GBndLib being loaded, so no G4 complications
        // just need to avoid trying to translate the matline later

        const char* material = torchstep->getMaterial() ;
        unsigned int matline = m_ggeo->getMaterialLine(material);
        torchstep->setMaterialLine(matline);  

        LOG(debug) << "OpticksHub::makeGenstepTorch"
                   << " config " << torchstep->getConfig() 
                   << " material " << material 
                   << " matline " << matline
                         ;
    }
    else
    {
        LOG(warning) << "OpticksHub::makeTorchstep no ggeo, skip setting torchstep material line " ;
    } 

    bool torchdbg = hasOpt("torchdbg");
    torchstep->addStep(torchdbg);  // copyies above configured step settings into the NPY and increments the step index, ready for configuring the next step 

    if(torchdbg)
    {
        NPY<float>* gs = torchstep->getNPY();
        gs->save("$TMP/torchdbg.npy");
    }

    return torchstep ; 
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
    return m_evt ? m_evt->getTimer() : m_ok->getTimer() ; 
}




void OpticksHub::add(BCfg* cfg)
{
    m_cfg->add(cfg); 
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
#ifdef WITH_NPYSERVER
    configureServer();
#endif

    configureCompositionSize();

    configureLookupA();

    TIMER("configure");
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
   // Used from OKG4Mgr to override the default mapping 
   // when using G4 steps directly 

    m_lookup->setA( A, "", msg);
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


NPY<unsigned char>* OpticksHub::getColorBuffer()
{
    OpticksColors* colors = m_ok->getColors();

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

    assert(gs && "OpticksHub::initOKEvent gs NULL");

    LOG(info) << "OpticksHub::initOKEvent "
              << " gs " << gs->getShapeString()
              ;
 
    setGensteps(gs);  

    bool ok = true ; 
    createEvent(ok); 

    m_okevt->setGenstepData(gs);
    assert(m_evt == m_okevt);

    if(m_g4evt)   // if there is a preexisting G4 event use the same timestamp for the OK event
    {
       assert(m_g4evt->isG4());
       assert(m_okevt->isOK());

       std::string tstamp = m_g4evt->getTimeStamp();
       m_okevt->setTimeStamp( tstamp.c_str() );      
    }

    LOG(info) << "OpticksHub::initOKEvent "
              << " gensteps " << gs->getShapeString()
              << " tagdir " << m_okevt->getTagDir() 
              ;

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

void OpticksHub::save()
{
    if(m_g4evt)
    {
        m_g4evt->dumpDomains("OpticksHub::save g4evt domains");
        m_g4evt->save();
    } 
    if(m_okevt)
    {
        m_okevt->dumpDomains("OpticksHub::save okevt domains");
        m_okevt->save();
    } 
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
    m_evt = m_ok->makeEvent(ok) ; 
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

    NPY<float>* track = evt->loadGenstepDerivativeFromFile("track");
    m_composition->setTrack(track);
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
    else if(m_evt)
    {
        glm::vec4 gsce = m_evt->getGenstepCenterExtent();  // need to setGenStepData before this will work 
        m_composition->setCenterExtent( gsce , autocam );
        LOG(info) << "OpticksHub::target"
                  << " evt " << m_evt->brief()
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




