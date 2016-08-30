#include "Timer.hpp"


#include "OXPPNS.hh"

#include "Opticks.hh"
#include "OpticksEvent.hh"
#include "OpticksCfg.hh"

#include "GGeo.hh"

// opticksgeo-
#include "OpticksHub.hh"


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
       if(m_hub)\
       {\
          Timer& t = *(m_hub->getTimer()) ;\
          t((s)) ;\
       }\
    }



OEngineImp::OEngineImp(OpticksHub* hub) 
     :   
      m_timer(NULL),
      m_hub(hub),
      m_opticks(hub->getOpticks()),
      m_fcfg(NULL),
      m_ggeo(NULL),  // defer to avoid order brittleness

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

void OEngineImp::init()
{
    m_fcfg = m_opticks->getCfg();

    m_timer      = new Timer("OEngineImp::");
    m_timer->setVerbose(true);
    m_timer->start();
}


OContext* OEngineImp::getOContext()
{
    return m_ocontext ; 
}
OPropagator* OEngineImp::getOPropagator()
{
    return m_opropagator ; 
}



void OEngineImp::prepareOptiX()
{
    LOG(trace) << "OEngineImp::prepareOptiX START" ; 

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

    m_ggeo = m_hub->getGGeo();

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
    LOG(trace) << "OEngineImp::prepareOptiX DONE" ;


}




void OEngineImp::preparePropagator()
{
    bool noevent    = m_fcfg->hasOpt("noevent");
    bool trivial    = m_fcfg->hasOpt("trivial");
    bool seedtest   = m_fcfg->hasOpt("seedtest");
    int  override_   = m_fcfg->getOverride();

    OpticksEvent* evt = m_hub->getEvent(); 
    if(!evt) return ;

    assert(!noevent);

    LOG(trace) << "OEngineImp::preparePropagator" 
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


    m_opropagator = new OPropagator(m_ocontext, m_hub, override_);

    m_opropagator->setEntry(entry);
    m_opropagator->initRng();
    m_opropagator->initEvent();

    LOG(trace) << "OEngineImp::preparePropagator DONE ";
}


void OEngineImp::propagate()
{
    LOG(trace)<< "OEngineImp::propagate" ;

    m_opropagator->prelaunch();
    TIMER("prelaunch");

    m_opropagator->launch();
    TIMER("propagate");

    m_opropagator->dumpTimes("OEngineImp::propagate");
}


void OEngineImp::downloadPhotonData()
{
    OpticksEvent* evt = m_hub->getEvent(); 
    if(!evt) return ;

    if(m_opticks->isCompute())
    {
        m_opropagator->downloadPhotonData();
    }
}

void OEngineImp::saveEvt()
{
    OpticksEvent* evt = m_hub->getEvent(); 
    if(!evt) return ;

    // note that "interop" download with Rdr::download(evt);   
    // is now done from App::saveEvt just prior to this being called

    m_opropagator->downloadEvent();  
    // formerly did OPropagator::downloadEvt only in compute mode
    // but now that interop/compute are blurring have to check each buffer
    // as some may be compute mode even whilst running interop

    TIMER("downloadEvt");

    evt->dumpDomains("OEngineImp::saveEvt dumpDomains");
    evt->save();  // TODO: this should happen at higher level, not buried here ?

    TIMER("saveEvt");
}


void OEngineImp::cleanup()
{
   if(m_ocontext) m_ocontext->cleanUp();
}



