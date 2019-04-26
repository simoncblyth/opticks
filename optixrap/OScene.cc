

#include "OKConf_Config.hh"

#include "BTimeKeeper.hh"

#include "SSys.hh"
#include "SLog.hh"
#include "OXPPNS.hh"
#include "OError.hh"

#include "Opticks.hh"
#include "OpticksEvent.hh"
#include "OpticksCfg.hh"

// okg-
#include "OpticksHub.hh"
#include "GScintillatorLib.hh"

// oxrap-
#include "OContext.hh"
#include "OFunc.hh"
#include "OColors.hh"
#include "OGeo.hh"
#include "OBndLib.hh"
#include "OScintillatorLib.hh"
#include "OSourceLib.hh"
#include "OBuf.hh"
#include "OConfig.hh"

#include "OScene.hh"


#include "PLOG.hh"


#define TIMER(s) \
    { \
       (*m_timer)((s)); \
    }



const plog::Severity OScene::LEVEL = debug ; 

OContext* OScene::getOContext()
{
    return m_ocontext ; 
}

OBndLib*  OScene::getOBndLib()
{
    return m_olib ; 
}


OScene::OScene(OpticksHub* hub) 
    :   
    m_log(new SLog("OScene::OScene","", LEVEL)),
    m_timer(new BTimeKeeper("OScene::")),
    m_hub(hub),
    m_ok(hub->getOpticks()),
    m_ocontext(NULL),
    m_osolve(NULL),
    m_ocolors(NULL),
    m_ogeo(NULL),
    m_olib(NULL),
    m_oscin(NULL),
    m_osrc(NULL),
    m_verbosity(m_ok->getVerbosity()),
    m_use_osolve(false)
{
    init();
    (*m_log)("DONE");
}


/**

OScene::Init
---------------

1. creates OptiX context
2. instanciates the O*Libs which populate the OptiX context 
   from the corresponding libs provided by OpticksHub accessors
   (NB not directly from GGeo or GScene, the Hub mediates)
::

    OColors 
    OSourceLib
    OScintillatorLib
    OGeo
    OBndLib 

**/


void OScene::initRTX()
{
    //const char* key = "OPTICKS_RTX" ;
    //int rtx = SSys::getenvint(key, -1 ); 

    int rtxmode = m_ok->getRTX();

    if(rtxmode == -1)
    {
        LOG(fatal) << " --rtx " << rtxmode << " leaving ASIS "  ;   
    }
    else
    { 
        int rtx0(-1) ;
        RT_CHECK_ERROR( rtGlobalGetAttribute(RT_GLOBAL_ATTRIBUTE_ENABLE_RTX, sizeof(rtx0), &rtx0) );
        assert( rtx0 == 0 );  // despite being zero performance suggests it is enabled

        int rtx = rtxmode > 0 ? 1 : 0 ;       
        LOG(fatal) << " --rtx " << rtxmode << " setting  " << ( rtx == 1 ? "ON" : "OFF" )  ; 
        RT_CHECK_ERROR( rtGlobalSetAttribute(RT_GLOBAL_ATTRIBUTE_ENABLE_RTX, sizeof(rtx), &rtx));

        int rtx2(-1) ; 
        RT_CHECK_ERROR(rtGlobalGetAttribute(RT_GLOBAL_ATTRIBUTE_ENABLE_RTX, sizeof(rtx2), &rtx2));
        assert( rtx2 == rtx );
    }
}


void OScene::init()
{
    LOG(info) << "[" ; 

    plog::Severity level = LEVEL ; 

    m_timer->setVerbose(true);
    m_timer->start();

    initRTX();

    LOG(verbose) << "optix::Context::create() START " ; 
    optix::Context context = optix::Context::create();
    LOG(verbose) << "optix::Context::create() DONE " ; 

    m_ocontext = new OContext(context, m_ok);


    // solvers despite being used for geometry intersects have no dependencies
    // as just pure functions : so place them accordingly 
    if(m_use_osolve)
    {  
        m_osolve = new OFunc(m_ocontext, "solve_callable.cu", "solve_callable", "SolveCubicCallable" ) ; 
        m_osolve->convert();
    }

    LOG(LEVEL) 
          << " ggeobase identifier : " << m_hub->getIdentifier()
          ;

    LOG(level) << "(OColors)" ;
    m_ocolors = new OColors(context, m_ok->getColors() );
    m_ocolors->convert();

    // formerly did OBndLib here, too soon

    LOG(level) << "(OSourceLib)" ;
    m_osrc = new OSourceLib(context, m_hub->getSourceLib());
    m_osrc->convert();


    GScintillatorLib* sclib = m_hub->getScintillatorLib() ;
    unsigned num_scin = sclib->getNumScintillators(); 
    const char* slice = "0:1" ;

    LOG(level) << "(OScintillatorLib)"
               << " num_scin " << num_scin 
               << " slice " << slice  
               ;

    // a placeholder reemission texture is created even when no scintillators
    m_oscin = new OScintillatorLib(context, sclib );
    m_oscin->convert(slice);


    LOG(level) << "(OGeo)" ;
    m_ogeo = new OGeo(m_ocontext, m_ok, m_hub->getGeoLib() );
    LOG(level) << "(OGeo) convert" ;
    m_ogeo->convert();
    LOG(level) << "(OGeo) done" ;


    LOG(level) << "(OBndLib)" ;
    m_olib = new OBndLib(context,m_hub->getBndLib());
    m_olib->convert();
    // this creates the BndLib dynamic buffers, which needs to be after OGeo
    // as that may add boundaries when using analytic geometry


    LOG(debug) << m_ogeo->description();

    LOG(info) << "]" ;

}


void OScene::cleanup()
{
   if(m_ocontext) m_ocontext->cleanUp();
}


