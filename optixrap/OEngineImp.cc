#include "Timer.hpp"


#include "OXPPNS.hh"

#include "Opticks.hh"
#include "OpticksEvent.hh"
#include "OpticksCfg.hh"

#include "GGeo.hh"

// optixrap-
#include "OContext.hh"
#include "OColors.hh"
#include "OGeo.hh"
#include "OBndLib.hh"
#include "OScintillatorLib.hh"
#include "OSourceLib.hh"
#include "OBuf.hh"
#include "OConfig.hh"
#include "OTracer.hh"
#include "OPropagator.hh"

#include "OEngineImp.hh"


#include "PLOG.hh"


#define TIMER(s) \
    { \
       (*m_timer)((s)); \
       if(m_evt)\
       {\
          Timer& t = *(m_evt->getTimer()) ;\
          t((s)) ;\
       }\
    }



OEngineImp::OEngineImp(Opticks* opticks, GGeo* ggeo) 
     :   
      m_opticks(opticks),
      m_fcfg(NULL),
      m_ggeo(ggeo),
      m_evt(NULL),

      m_ocontext(NULL),
      m_ocolors(NULL),
      m_ogeo(NULL),
      m_olib(NULL),
      m_oscin(NULL),
      m_osrc(NULL),
      m_otracer(NULL),
      m_opropagator(NULL)
{
      init();
}


void OEngineImp::setEvent(OpticksEvent* evt)
{
    m_evt = evt ;
}

OpticksEvent* OEngineImp::getEvent()
{
    return m_evt ; 
}



OContext* OEngineImp::getOContext()
{
    return m_ocontext ; 
}
OPropagator* OEngineImp::getOPropagator()
{
    return m_opropagator ; 
}


void OEngineImp::init()
{
    m_fcfg = m_opticks->getCfg();

    m_timer      = new Timer("OEngineImp::");
    m_timer->setVerbose(true);
    m_timer->start();
}


void OEngineImp::prepareOptiX()
{
    LOG(info) << "OEngineImp::prepareOptiX START" ; 

    std::string builder_   = m_fcfg->getBuilder();
    std::string traverser_ = m_fcfg->getTraverser();
    const char* builder   = builder_.empty() ? NULL : builder_.c_str() ;
    const char* traverser = traverser_.empty() ? NULL : traverser_.c_str() ;


    OContext::Mode_t mode = m_opticks->isCompute() ? OContext::COMPUTE : OContext::INTEROP ;

    optix::Context context = optix::Context::create();

    LOG(debug) << "OEngineImp::prepareOptiX (OContext)" ;
    m_ocontext = new OContext(context, mode);
    m_ocontext->setStackSize(m_fcfg->getStack());
    m_ocontext->setPrintIndex(m_fcfg->getPrintIndex().c_str());
    m_ocontext->setDebugPhoton(m_fcfg->getDebugIdx());


    if(m_ggeo == NULL)
    {
        LOG(warning) << "OEngineImp::prepareOptiX EARLY EXIT AS no geometry " ; 
        return ; 
    }


    LOG(debug) << "OEngineImp::prepareOptiX (OColors)" ;
    m_ocolors = new OColors(context, m_opticks->getColors() );
    m_ocolors->convert();

    // formerly did OBndLib here, too soon

    LOG(debug) << "OEngineImp::prepareOptiX (OSourceLib)" ;
    m_osrc = new OSourceLib(context, m_ggeo->getSourceLib());
    m_osrc->convert();


    const char* slice = "0:1" ;
    LOG(debug) << "OEngineImp::prepareOptiX (OScintillatorLib) slice " << slice  ;
    m_oscin = new OScintillatorLib(context, m_ggeo->getScintillatorLib());
    m_oscin->convert(slice);


    LOG(debug) << "OEngineImp::prepareOptiX (OGeo)" ;
    m_ogeo = new OGeo(m_ocontext, m_ggeo, builder, traverser);
    LOG(debug) << "OEngineImp::prepareOptiX (OGeo) -> setTop" ;
    m_ogeo->setTop(m_ocontext->getTop());
    LOG(debug) << "OEngineImp::prepareOptiX (OGeo) -> convert" ;
    m_ogeo->convert();
    LOG(debug) << "OEngineImp::prepareOptiX (OGeo) done" ;


    LOG(debug) << "OEngineImp::prepareOptiX (OBndLib)" ;
    m_olib = new OBndLib(context,m_ggeo->getBndLib());
    m_olib->convert();
    // this creates the BndLib dynamic buffers, which needs to be after OGeo
    // as that may add boundaries when using analytic geometry


    LOG(debug) << m_ogeo->description("OEngineImp::prepareOptiX ogeo");
    LOG(info) << "OEngineImp::prepareOptiX DONE" ;


}




void OEngineImp::preparePropagator()
{
    bool noevent    = m_fcfg->hasOpt("noevent");
    bool trivial    = m_fcfg->hasOpt("trivial");
    bool seedtest   = m_fcfg->hasOpt("seedtest");
    int  override_   = m_fcfg->getOverride();

    if(!m_evt) return ;

    assert(!noevent);

    LOG(info) << "OEngineImp::preparePropagator" 
              << ( trivial ? " TRIVIAL TEST" : "NORMAL" )
              << " override_ " << override_
              ;  

    unsigned int entry ;

    bool defer = true ; 

    if(trivial)
    {
        entry = m_ocontext->addEntry("generate.cu.ptx", "trivial", "exception", defer);
    }
    else if(seedtest)
    {
        entry = m_ocontext->addEntry("seedTest.cu.ptx", "seedTest", "exception", defer);
    }
    else
    {
        entry = m_ocontext->addEntry("generate.cu.ptx", "generate", "exception", defer);
    }



    m_opropagator = new OPropagator(m_ocontext, m_opticks, override_);

    m_opropagator->setEvent(m_evt);
    m_opropagator->setEntry(entry);


    m_opropagator->initRng();
    m_opropagator->initEvent();

    LOG(info) << "OEngineImp::preparePropagator DONE ";
}


void OEngineImp::propagate()
{
    LOG(info)<< "OEngineImp::propagate" ;

    m_opropagator->prelaunch();
    TIMER("prelaunch");

    m_opropagator->launch();
    TIMER("propagate");

    m_opropagator->dumpTimes("OEngineImp::propagate");
}


void OEngineImp::downloadPhotonData()
{
    if(!m_evt) return ;

    if(m_opticks->isCompute())
    {
        m_opropagator->downloadPhotonData();
    }
}

void OEngineImp::saveEvt()
{
    if(!m_evt) return ;

    if(m_opticks->isCompute())
    {
        m_opropagator->downloadEvent();
    }
    else
    {
        //Rdr::download(m_evt);   
        //     interop download now done from App::saveEvt 
        //     just prior to this being called
    }

    TIMER("downloadEvt");

    m_evt->dumpDomains("OEngineImp::saveEvt dumpDomains");
    m_evt->save(true);

    TIMER("saveEvt");
}


void OEngineImp::cleanup()
{
   if(m_ocontext) m_ocontext->cleanUp();
}



