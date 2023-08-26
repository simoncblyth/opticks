/**
CSGOptiX.cc
============

This code contains two branches for old (OptiX < 7) and new (OptiX 7+) API

Branched aspects:

1. CSGOptiX7.cu vs CSGOptiX6.cu  
2. Ctx/SBT/PIP  vs Six
3. new workflow uses uploaded params extensively from CUDA device code, 
   old workflow with *Six* uses hostside params to populate the optix context 
   which then gets used on device

   CSGOptiX::prepareParam (called just before launch)

   * new workflow: uploads param 
   * old workflow: Six::updateContext passes hostside param into optix context variables 

   Six MATCHING NEEDS DUPLICATION FROM Params INTO CONTEXT VARIABLES AND BUFFERS

HMM: looking like getting qudarap/qsim.h to work with OptiX < 7 is more effort than it is worth 

* would have to shadow it into context variables 
* CUDA textures would not work without optix textureSamplers 

**/

#include <iostream>
#include <cstdlib>
#include <chrono>

#include <optix.h>
#if OPTIX_VERSION < 70000
#else
#include <optix_stubs.h>
#endif

#include <cuda_runtime.h>
#include <glm/glm.hpp>

// sysrap
#include "SProc.hh"
#include "SGLM.h"
#include "NP.hh"
#include "SRG.h"
#include "SCAM.h"
#include "SEventConfig.hh"
#include "SGeoConfig.hh"
#include "SSim.hh"
#include "SStr.hh"
#include "SSys.hh"
#include "SEvt.hh"
#include "SMeta.hh"
#include "SPath.hh"
#include "SVec.hh"
#include "SLOG.hh"
#include "scuda.h"
#include "squad.h"
#include "sframe.h"
#include "salloc.h"


#ifdef WITH_SGLM
#else
#include "Opticks.hh"
#include "Composition.hh"
#include "FlightPath.hh"
#endif


/**
HMM: Composition is a bit of a monster - bringing in a boatload of classes 
LONGTERM: see if can pull out the essentials into a smaller class

* SGLM.h is already on the way to doing this kinda thing in a single header 
* Composition::getEyeUVW is the crux method needed 
* Composition or its replacement only relevant for rendering, not for simulation 
**/

// csg 
#include "CSGPrim.h"
#include "CSGFoundry.h"
#include "CSGView.h"

// qudarap
#include "QU.hh"
#include "QSim.hh"
#include "qsim.h"
#include "QEvent.hh"

// CSGOptiX
#include "Frame.h"
#include "Params.h"

#if OPTIX_VERSION < 70000
#include "Six.h"
#else
#include "Ctx.h"
#include "CUDA_CHECK.h"   
#include "OPTIX_CHECK.h"   
#include "PIP.h"
#include "SBT.h"
#endif

#include "CSGOptiX.h"

const plog::Severity CSGOptiX::LEVEL = SLOG::EnvLevel("CSGOptiX", "DEBUG" ); 
CSGOptiX* CSGOptiX::INSTANCE = nullptr ; 
CSGOptiX* CSGOptiX::Get()
{
    return INSTANCE ; 
}

#if OPTIX_VERSION < 70000 
const char* CSGOptiX::PTXNAME = "CSGOptiX6" ; 
const char* CSGOptiX::GEO_PTXNAME = "CSGOptiX6geo" ; 
#else
const char* CSGOptiX::PTXNAME = "CSGOptiX7" ; 
const char* CSGOptiX::GEO_PTXNAME = nullptr ; 
#endif

int CSGOptiX::Version()
{
    int vers = 0 ; 
#if OPTIX_VERSION < 70000
    vers = 6 ; 
#else
    vers = 7 ; 
#endif
    return vers ; 
}


/**
CSGOptiX::RenderMain CSGOptiX::SimtraceMain CSGOptiX::SimulateMain
---------------------------------------------------------------------

These three mains are use by the minimal main tests:

+-------------+---------------------------+---------------------+
|  script     | mains                     | notes               | 
+=============+===========================+=====================+
| cxr_min.sh  | tests/CSGOptiXRMTest.cc   | minimal render      | 
+-------------+---------------------------+---------------------+
| cxt_min.sh  | tests/CSGOptiXTMTest.cc   | minimal simtrace    |
+-------------+---------------------------+---------------------+
| cxs_min.sh  | tests/CSGOptiXSMTest.cc   | minimal simulate    |
+-------------+---------------------------+---------------------+

Note that the former SEvt setup and frame hookup 
done in the main is now moved into CSGFoundry::AfterLoadOrCreate

Also try moving SEvt::BeginOfEvent SEvt::EndOfEvent into QSim

Note that currently rendering persisting does not use SEvt in the
same way as simtrace and simulate, but it could do in future. 

**/

void CSGOptiX::RenderMain() // static
{
    SEventConfig::SetRGMode("render"); 
    CSGFoundry* fd = CSGFoundry::Load(); 
    CSGOptiX* cx = CSGOptiX::Create(fd) ;
    cx->render(); 
}
void CSGOptiX::SimtraceMain()
{
    SEventConfig::SetRGMode("simtrace"); 
    CSGFoundry* fd = CSGFoundry::Load(); 
    CSGOptiX* cx = CSGOptiX::Create(fd) ;
    cx->simtrace(0); 
}
void CSGOptiX::SimulateMain() // static
{
    SEventConfig::SetRGMode("simulate"); 
    CSGFoundry* fd = CSGFoundry::Load(); 
    CSGOptiX* cx = CSGOptiX::Create(fd) ;
    cx->simulate(0); 
}

/**
CSGOptiX::Main
----------------

This "proceed" approach means that a single executable 
does very different things depending on the RGMode envvar. 
That is not convenient for bookkeeping based on executable names 
so instead use three separate executables that each use the 
corresponding Main static method. 

**/
void CSGOptiX::Main() // static
{
    CSGFoundry* fd = CSGFoundry::Load(); 
    CSGOptiX* cx = CSGOptiX::Create(fd) ;
    cx->proceed(); 
}

const char* CSGOptiX::Desc()
{
    std::stringstream ss ; 
    ss << "CSGOptiX::Desc" 
       << " Version " << Version() 
       << " PTXNAME " << PTXNAME 
       << " GEO_PTXNAME " << ( GEO_PTXNAME ? GEO_PTXNAME : "-" ) 
       ; 
    std::string s = ss.str(); 
    return strdup(s.c_str()); 
}


const char* CSGOptiX::desc() const 
{
    std::stringstream ss ; 
    ss << Desc() ; 
#if OPTIX_VERSION < 70000
#else
    ss << pip->desc() ; 
#endif
    std::string s = ss.str(); 
    return strdup(s.c_str()); 
}

/**
CSGOptiX::InitGeo
-------------------

CSGFoundry not const as upload sets device pointers
CSGOptiX::InitGeo currently takes 20s for full JUNO geometry, 
where the total gxs.sh running time for one event is 24s. 

HMM:that was optimized down to under 1s, by removal of some unused stree.h stuff ?

**/

void CSGOptiX::InitGeo(  CSGFoundry* fd )
{
    LOG(LEVEL) << "[" ; ; 
    fd->upload(); 
    LOG(LEVEL) << "]" ; ; 
}

/**
CSGOptiX::InitSim
-------------------

Instanciation of QSim/QEvent requires an SEvt instance 

**/

void CSGOptiX::InitSim( const SSim* ssim  )
{
    LOG(LEVEL) << "[" ; ; 
    if(SEventConfig::IsRGModeRender()) return ; 

    LOG_IF(fatal, ssim == nullptr ) << "simulate/simtrace modes require SSim/QSim setup" ;
    assert(ssim);  

    QSim::UploadComponents(ssim);  

    QSim* qs = QSim::Create() ; 
    LOG(LEVEL) << "]" << qs->desc() ; 
}


/**
CSGOptiX::Create
--------------------

**/

CSGOptiX* CSGOptiX::Create(CSGFoundry* fd )   
{
    LOG(LEVEL) << "[ fd.descBase " << ( fd ? fd->descBase() : "-" ) ; 

    QU::alloc = new salloc ;   // HMM: maybe this belongs better in QSim ? 

    InitSim(fd->sim); // uploads SSim arrays instanciating QSim
    InitGeo(fd);      // uploads geometry 

    CSGOptiX* cx = new CSGOptiX(fd) ; 

    if(!SEventConfig::IsRGModeRender())
    {
        QSim* qs = QSim::Get() ; 
        qs->setLauncher(cx); 
    } 

    LOG(LEVEL) << "]" ; 
    return cx ; 
}


#ifdef WITH_SGLM
Params* CSGOptiX::InitParams( int raygenmode, const SGLM* sglm  ) // static
{
    LOG(LEVEL) << "[" ; 
    return new Params(raygenmode, sglm->Width(), sglm->Height(), 1 ) ; 
    LOG(LEVEL) << "]" ; 
}
#else
Params* CSGOptiX::InitParams( int raygenmode, const Opticks* ok  ) // static
{
    LOG(LEVEL) << "[" ; 
    return new Params(raygenmode, ok->getWidth(), ok->getHeight(), ok->getDepth() ) ; 
    LOG(LEVEL) << "]" ; 
}
#endif



CSGOptiX::CSGOptiX(const CSGFoundry* foundry_) 
    :
#ifdef WITH_SGLM
#else
    ok(Opticks::Instance()),
    composition(ok->getComposition()),
#endif
    sglm(new SGLM),   // instanciate always to allow view matrix comparisons
    flight(SGeoConfig::FlightConfig()),
    foundry(foundry_),
    prefix(SSys::getenvvar("OPTICKS_PREFIX","/usr/local/opticks")),  // needed for finding ptx
    outdir(SEventConfig::OutFold()),    
    cmaketarget("CSGOptiX"),  
    ptxpath(SStr::PTXPath( prefix, cmaketarget, PTXNAME )),
#if OPTIX_VERSION < 70000 
    geoptxpath(SStr::PTXPath(prefix, cmaketarget, GEO_PTXNAME )),
#else
    geoptxpath(nullptr),
#endif
    tmin_model(SSys::getenvfloat("TMIN",0.1)),    // CAUTION: tmin very different in rendering and simulation 
    raygenmode(SEventConfig::RGMode()),
#ifdef WITH_SGLM
    params(InitParams(raygenmode,sglm)),
#else
    params(InitParams(raygenmode,ok)),
#endif

#if OPTIX_VERSION < 70000
    six(new Six(ptxpath, geoptxpath, params)),
    frame(new Frame(params->width, params->height, params->depth, six->d_pixel, six->d_isect, six->d_photon)), 
#else
    ctx(nullptr),
    pip(nullptr), 
    sbt(nullptr),
    frame(nullptr), 
#endif
    meta(new SMeta),
    dt(0.),
    sim(QSim::Get()),  
    event(sim == nullptr  ? nullptr : sim->event)
{
    init(); 
}

void CSGOptiX::init()
{
#ifdef WITH_SGLM
    sglm->addlog("CSGOptiX::init", "start"); 
#endif

    LOG(LEVEL) 
        << "[" 
        << " raygenmode " << raygenmode
        << " SRG::Name(raygenmode) " << SRG::Name(raygenmode)
        << " sim " << sim 
        << " event " << event 
        ;  

    assert( prefix && "expecting PREFIX envvar pointing to writable directory" );
    assert( outdir && "expecting OUTDIR envvar " );

    LOG(LEVEL) << " ptxpath " << ptxpath  ; 
    LOG(LEVEL) << " geoptxpath " << ( geoptxpath ? geoptxpath : "-" ) ; 

    initCtx();
    initPIP();
    initSBT();
    initFrameBuffer(); 
    initCheckSim(); 
    initStack(); 
    initParams(); 
    initGeometry();
    initRender(); 
    initSimulate(); 

    LOG(LEVEL) << "]" ; 
}


void CSGOptiX::initCtx()
{
    LOG(LEVEL) << "[" ; 
#if OPTIX_VERSION < 70000
#else
    ctx = new Ctx ; 
    LOG(LEVEL) << std::endl << ctx->desc() ; 
#endif
    LOG(LEVEL) << "]" ; 
}

void CSGOptiX::initPIP()
{
    LOG(LEVEL) << "[" ; 
#if OPTIX_VERSION < 70000
#else
    pip = new PIP(ptxpath, ctx->props ) ;  
#endif
    LOG(LEVEL) << "]" ; 
}

void CSGOptiX::initSBT()
{
    LOG(LEVEL) << "[" ; 
#if OPTIX_VERSION < 70000
#else
    sbt = new SBT(pip) ; 
#endif
    LOG(LEVEL) << "]" ; 
}

void CSGOptiX::initFrameBuffer()
{
    LOG(LEVEL) << "[" ; 
    frame = new Frame(params->width, params->height, params->depth) ; 
    LOG(LEVEL) << "]" ; 
}


void CSGOptiX::initCheckSim()
{
    LOG(LEVEL) << " sim " << sim << " event " << event ; 
    if(SEventConfig::IsRGModeRender() == false)
    {
        LOG_IF(fatal, sim == nullptr) << "simtrace/simulate modes require instanciation of QSim before CSGOptiX " ; 
        assert(sim); 
        
    }
}


void CSGOptiX::initStack()
{
    LOG(LEVEL); 
#if OPTIX_VERSION < 70000
#else
    pip->configureStack(); 
#endif

}


void CSGOptiX::initParams()
{
    params->device_alloc(); 
}

/**
CSGOptiX::initGeometry
------------------------

Notice that the geometry is uploaded to GPU before calling this by CSGOptiX::InitGeo
The SBT::setFoundry kicks off the creation of the NVIDIA OptiX geometry
from the uploaded CSGFoundry with SBT::createGeom. 

**/

void CSGOptiX::initGeometry()
{
    LOG(LEVEL) << "[" ; 
    params->node = foundry->d_node ; 
    params->plan = foundry->d_plan ; 
    params->tran = nullptr ; 
    params->itra = foundry->d_itra ; 

    bool is_uploaded =  params->node != nullptr ;
    LOG_IF(fatal, !is_uploaded) << "foundry must be uploaded prior to CSGOptiX::initGeometry " ;  
    assert( is_uploaded ); 

#if OPTIX_VERSION < 70000
    six->setFoundry(foundry);
#else
    LOG(LEVEL) << "[ sbt.setFoundry " ; 
    sbt->setFoundry(foundry); 
    LOG(LEVEL) << "] sbt.setFoundry " ; 
#endif
    const char* top = Top(); 
    setTop(top); 
    LOG(LEVEL) << "]" ; 
}

/**
CSGOptiX::initRender
---------------------
**/

void CSGOptiX::initRender()
{
    if(SEventConfig::IsRGModeRender()) 
    {
        setFrame(); // MOI 
    }

    params->pixels = frame->d_pixel ;
    params->isect  = frame->d_isect ; 
#ifdef WITH_FRAME_PHOTON
    params->fphoton = frame->d_photon ; 
#else
    params->fphoton = nullptr ; 
#endif
}

/**
CSGOptiX::initSimulate
------------------------

* Once only (not per-event) simulate setup tasks ..  perhaps rename initPhys

Sets device pointers for params.sim params.evt so must be after upload 

Q: where are sim and evt uploaded ?
A: QSim::QSim and QSim::init_sim are where sim and evt are populated and uploaded 


HMM: get d_sim (qsim.h) now holds d_evt (qevent.h) but this is getting evt again rom QEvent ?
TODO: eliminate params->evt to make more use of the qsim.h encapsulation 

**/

void CSGOptiX::initSimulate() 
{
    LOG(LEVEL) ; 
    params->sim = sim ? sim->getDevicePtr() : nullptr ;  // qsim<float>*
    params->evt = event ? event->getDevicePtr() : nullptr ;  // qevent*
    params->tmin = SEventConfig::PropagateEpsilon() ;  // eg 0.1 0.05 to avoid self-intersection off boundaries
    params->tmax = 1000000.f ; 
}


/**
CSGOptiX::simulate
--------------------

NB the distinction between this and simulate_launch, this 
uses QSim::simulate to do genstep setup prior to calling 
CSGOptiX::simulate_launch via the SCSGOptiX.h protocol

**/
double CSGOptiX::simulate(int eventID)
{
    assert(sim); 
    double dt = sim->simulate(eventID) ;
    return dt ; 
}

/**
CSGOptiX::simtrace
--------------------

NB the distinction between this and simtrace_launch, this 
uses QSim::simtrace to do genstep setup prior to calling 
CSGOptiX::simtrace_launch via the SCSGOptiX.h protocol

**/

double CSGOptiX::simtrace(int eventID)
{
    assert(sim); 
    double dt = sim->simtrace(eventID) ;
    return dt ; 
}

double CSGOptiX::proceed()
{
    double dt = -1. ; 
    switch(SEventConfig::RGMode())
    {
        case SRG_SIMULATE: dt = simulate(0) ; break ; 
        case SRG_RENDER:   dt = render()   ; break ; 
        case SRG_SIMTRACE: dt = simtrace(0) ; break ; 
    }
    return dt ; 
}







const char* CSGOptiX::TOP = "i0" ; 
const char* CSGOptiX::Top()
{
    const char* top = SSys::getenvvar("TOP", TOP ); 
    bool top_valid = top != nullptr && strlen(top) > 1 && top[0] == 'i' ;  
    if(!top_valid)
    { 
        LOG(error) << "TOP envvar not invalid  [" << top << "] override with default [" << TOP << "]"  ; 
        top = TOP ; 
    }
    return top ; 
}

void CSGOptiX::setTop(const char* tspec)
{
    LOG(LEVEL) << "[" << " tspec " << tspec ; 

#if OPTIX_VERSION < 70000
    six->setTop(tspec); 
#else
    sbt->setTop(tspec);
    AS* top = sbt->getTop(); 
    params->handle = top->handle ; 
#endif

    LOG(LEVEL) << "]" << " tspec " << tspec ; 
}





/**
CSGOptiX::setFrame
--------------------------

The no argument method uses MOI envvar or default of "-1"

For global geometry which typically uses default iidx of 0 there is special 
handling of iidx -1/-2/-3 implemented in CSGTarget::getCenterExtent


iidx -2
    ordinary xyzw frame calulated by SCenterExtentFrame

iidx -3
    rtp tangential frame calulated by SCenterExtentFrame


Setting CE center-extent establishes the coordinate system
via calls to Composition::setCenterExtent which results in the 
definition of a model2world 4x4 matrix which becomes the frame of 
reference used by the EYE LOOK UP navigation controls.  


Q: CSGOptiX::setFrame is clearly needed for render but is it needed for simtrace, simulate ?
A: Currently think that it is just a bookkeeping convenience for simtrace and not needed for simulate. 

   * not anymore, at SEvt level the frame is used for input photon targetting 

**/

void CSGOptiX::setFrame()
{
    setFrame(SSys::getenvvar("MOI", "-1"));  // TODO: generalize to FRS
}

void CSGOptiX::setFrame(const char* frs)
{
    LOG(LEVEL) << " frs " << frs ; 
    sframe fr ; 
    foundry->getFrame(fr, frs) ; 
    setFrame(fr); 
}
void CSGOptiX::setFrame(const float4& ce )
{
    sframe fr_ ;   // m2w w2m default to identity 
    fr_.ce = ce ;      
    setFrame(fr_); 
}

/**
CSGOptiX::setFrame into the SGLM.h instance
----------------------------------------------

Note that SEvt already holds an sframe used for input photon transformation, 
the sframe here is used for raytrace rendering.  Could perhaps rehome sglm 
into SEvt and use a single sframe for both input photon transformation 
and rendering ?

**/

void CSGOptiX::setFrame(const sframe& fr_ )
{
    sglm->set_frame(fr_); 

    LOG(LEVEL) << "sglm.desc:" << std::endl << sglm->desc() ; 

    const float4& ce = sglm->fr.ce ; 
    const qat4& m2w = sglm->fr.m2w ; 
    const qat4& w2m = sglm->fr.w2m ; 


#ifdef WITH_SGLM
#else
    // without SGLM is the old way of doing things to be eliminated
    bool autocam = true ; 
    composition->setCenterExtent(ce, autocam, &m2w, &w2m );  // model2world view setup 
    composition->setNear(sglm->tmin_abs()); 
    LOG(info) << std::endl << composition->getCameraDesc() ;  
#endif

    LOG(LEVEL) 
        << " ce [ " << ce.x << " " << ce.y << " " << ce.z << " " << ce.w << "]" 
        << " sglm.TMIN " << sglm->TMIN
        << " sglm.tmin_abs " << sglm->tmin_abs() 
        << " sglm.m2w.is_zero " << m2w.is_zero()
        << " sglm.w2m.is_zero " << w2m.is_zero()
        ; 

    LOG(LEVEL) << "m2w " << m2w ; 
    LOG(LEVEL) << "w2m " << w2m ; 

    LOG(LEVEL) << "]" ; 
}







void CSGOptiX::prepareRenderParam()
{
    glm::vec3 eye ;
    glm::vec3 U ; 
    glm::vec3 V ; 
    glm::vec3 W ; 

    float tmin ; 
    float tmax ; 
    unsigned cameratype ; 
    float extent ; 
    float length ; 

#ifdef WITH_SGLM
    extent = sglm->fr.ce.w ; 
    eye = sglm->e ; 
    U = sglm->u ; 
    V = sglm->v ; 
    W = sglm->w ; 
    tmin = sglm->get_near_abs() ; 
    tmax = sglm->get_far_abs() ; 
    cameratype = sglm->cam ; 
    length = 0.f ; 
#else
    glm::vec4 ZProj ;
    extent = composition->getExtent(); 
    composition->getEyeUVW(eye, U, V, W, ZProj); // must setModelToWorld in composition first
    tmin = composition->getNear(); 
    tmax = composition->getFar(); 
    cameratype = composition->getCameraType(); 
    length = composition->getLength(); 
#endif


    if(!flight) 
    {
        LOG(LEVEL)
            << std::endl 
            << std::setw(20) << " extent "     << extent << std::endl 
            << std::setw(20) << " sglm.fr.ce.w "  << sglm->fr.ce.w << std::endl 
            << std::setw(20) << " sglm.getGazeLength "  << sglm->getGazeLength()  << std::endl 
            << std::setw(20) << " comp.length"   << length 
            << std::endl 
            << std::setw(20) << " tmin "       << tmin  << std::endl 
            << std::setw(20) << " tmax "       << tmax  << std::endl 
            << std::endl 
            << std::setw(20) << " sglm.near "  << sglm->near  << std::endl 
            << std::setw(20) << " sglm.get_near_abs "  << sglm->get_near_abs()  << std::endl 
            << std::endl 
            << std::setw(20) << " sglm.far "   << sglm->far  << std::endl 
            << std::setw(20) << " sglm.get_far_abs "   << sglm->get_far_abs()  << std::endl 
            << std::endl 
            << std::setw(25) << " sglm.get_nearfar_basis "  << sglm->get_nearfar_basis()  
            << std::setw(25) << " sglm.get_nearfar_mode "  << sglm->get_nearfar_mode()  
            << std::endl 
            << std::setw(25) << " sglm.get_focal_basis "  << sglm->get_focal_basis()  
            << std::setw(25) << " sglm.get_focal_mode "  << sglm->get_focal_mode()  
            << std::endl 
            << std::setw(20) << " eye ("       << eye.x << " " << eye.y << " " << eye.z << " ) " << std::endl 
            << std::setw(20) << " U ("         << U.x << " " << U.y << " " << U.z << " ) " << std::endl
            << std::setw(20) << " V ("         << V.x << " " << V.y << " " << V.z << " ) " << std::endl
            << std::setw(20) << " W ("         << W.x << " " << W.y << " " << W.z << " ) " << std::endl
            << std::endl 
            << std::setw(20) << " cameratype " << cameratype << " "           << SCAM::Name(cameratype) << std::endl 
            << std::setw(20) << " sglm.cam " << sglm->cam << " " << SCAM::Name(sglm->cam) << std::endl 
            ;

        LOG(LEVEL) << std::endl << "SGLM::DescEyeBasis (sglm->e,w,v,w) " << std::endl << SGLM::DescEyeBasis( sglm->e, sglm->u, sglm->v, sglm->w ) << std::endl ;
        LOG(LEVEL) << std::endl <<  "sglm.descEyeBasis " << std::endl << sglm->descEyeBasis() << std::endl ; 
        LOG(LEVEL) << std::endl << "Composition basis " << std::endl << SGLM::DescEyeBasis( eye, U, V, W ) << std::endl ;
        LOG(LEVEL) << std::endl  << "sglm.descELU " << std::endl << sglm->descELU() << std::endl ; 
        LOG(LEVEL) << std::endl << "sglm.descLog " << std::endl << sglm->descLog() << std::endl ; 

    }


    params->setView(eye, U, V, W);
    params->setCamera(tmin, tmax, cameratype ); 

    LOG(LEVEL) << std::endl << params->desc() ; 

    if(flight) return ; 

#ifdef WITH_SGLM
    LOG(LEVEL)
        << "sglm.desc " << std::endl 
        << sglm->desc() 
        ; 
#else
    LOG(LEVEL)
        << "composition.desc " << std::endl 
        << composition->desc() 
        ; 
#endif

}


/**
CSGOptiX::prepareSimulateParam
-------------------------------

Per-event simulate setup invoked just prior to optix launch 

**/

void CSGOptiX::prepareSimulateParam()  
{
    LOG(LEVEL); 
}

/**
CSGOptiX::prepareParam and upload
-------------------------------------

This is invoked by CSGOptiX::launch just before the OptiX launch, 
depending on raygenmode the simulate/render param are prepared and uploaded. 

Q: can Six use the same uploaded params ? 
A: not from device code it seems : only by using the hostside params to 
   populate the pre-7 optix::context

**/

void CSGOptiX::prepareParam()
{
#ifdef WITH_SGLM
    const float4& ce = sglm->fr.ce ;   
#else
    const glm::vec4& ce = composition->getCenterExtent();   
#endif

    params->setCenterExtent(ce.x, ce.y, ce.z, ce.w); 
    switch(raygenmode)
    {
        case SRG_RENDER   : prepareRenderParam()   ; break ; 
        case SRG_SIMTRACE : prepareSimulateParam() ; break ; 
        case SRG_SIMULATE : prepareSimulateParam() ; break ; 
    }

#if OPTIX_VERSION < 70000
    six->updateContext();  // Populates optix::context with values from hostside params
#else
    params->upload();  
    LOG_IF(LEVEL, !flight) << params->detail(); 
#endif
}





/**
CSGOptiX::launch
-------------------

For what happens next, see CSGOptiX7.cu::__raygen__rg OR CSGOptiX6.cu::raygen
Depending on params.raygenmode the "render" or "simulate" method is called. 
 
**/

double CSGOptiX::launch()
{
    prepareParam(); 
    if(raygenmode != SRG_RENDER) assert(event) ; 

    unsigned width = 0 ; 
    unsigned height = 0 ; 
    unsigned depth  = 0 ; 
    switch(raygenmode)
    {
        case SRG_RENDER:    { width = params->width           ; height = params->height ; depth = params->depth ; } ; break ;  
        case SRG_SIMTRACE:  { width = event->getNumSimtrace() ; height = 1              ; depth = 1             ; } ; break ;   
        case SRG_SIMULATE:  { width = event->getNumPhoton()   ; height = 1              ; depth = 1             ; } ; break ; 
    }
    assert( width > 0 ); 

    typedef std::chrono::time_point<std::chrono::high_resolution_clock> TP ;
    typedef std::chrono::duration<double> DT ;
    TP t0 = std::chrono::high_resolution_clock::now();

    LOG(LEVEL) 
         << " raygenmode " << raygenmode
         << " SRG::Name(raygenmode) " << SRG::Name(raygenmode)
         << " width " << width 
         << " height " << height 
         << " depth " << depth
         ;

#if OPTIX_VERSION < 70000
    assert( width <= 1000000 ); 
    six->launch(width, height, depth ); 
#else
    CUdeviceptr d_param = (CUdeviceptr)Params::d_param ; ;
    assert( d_param && "must alloc and upload params before launch"); 
    CUstream stream;
    CUDA_CHECK( cudaStreamCreate( &stream ) );
    OPTIX_CHECK( optixLaunch( pip->pipeline, stream, d_param, sizeof( Params ), &(sbt->sbt), width, height, depth ) );
    CUDA_SYNC_CHECK();    
    // see CSG/CUDA_CHECK.h the CUDA_SYNC_CHECK does cudaDeviceSyncronize
    // THIS LIKELY HAS LARGE PERFORMANCE IMPLICATIONS : BUT NOT EASY TO AVOID (MULTI-BUFFERING ETC..)  
#endif

    TP t1 = std::chrono::high_resolution_clock::now();
    DT _dt = t1 - t0;

    dt = _dt.count() ; 
    launch_times.push_back(dt);  

    LOG(LEVEL) 
          << " (params.width, params.height, params.depth) ( " 
          << params->width << "," << params->height << "," << params->depth << ")"  
          << std::fixed << std::setw(7) << std::setprecision(4) << dt  
          ; 
    return dt ; 
}


/**
CSGOptiX::render_launch CSGOptiX::simtrace_launch CSGOptiX::simulate_launch
--------------------------------------------------------------------------------

CAUTION : *simulate_launch* and *simtrace_launch*  
MUST BE invoked from QSim::simulate and QSim::simtrace using the SCSGOptiX.h protocol. 
This is because genstep preparations are needed prior to launch. 

These three methods currently all call *CSGOptiX::launch* 
with params.raygenmode switch function inside OptiX7Test.cu:__raygen__rg 
As it is likely better to instead have multiple raygen entry points 
are retaining the distinct methods up here. 

*render* is also still needed to fulfil SRenderer protocol base 

**/
double CSGOptiX::render_launch()
{  
    assert(raygenmode == SRG_RENDER) ; 
    return launch() ; 
}   
double CSGOptiX::simtrace_launch()
{ 
    assert(raygenmode == SRG_SIMTRACE) ; 
    return launch() ; 
}  
double CSGOptiX::simulate_launch()
{ 
    assert(raygenmode == SRG_SIMULATE) ; 
    return launch()  ;
}   

const CSGFoundry* CSGOptiX::getFoundry() const 
{
    return foundry ; 
}

std::string CSGOptiX::AnnotationTime( double dt, const char* extra )  // static 
{
    std::stringstream ss ; 
    ss << std::fixed << std::setw(10) << std::setprecision(4) << dt ;
    if(extra) ss << " " << extra << " " ; 
    std::string str = ss.str(); 
    return str ; 
}
std::string CSGOptiX::Annotation( double dt, const char* bot_line, const char* extra )  // static 
{
    std::stringstream ss ; 
    ss << AnnotationTime(dt, extra) ; 
    if(bot_line) ss << std::setw(30) << " " << bot_line ; 
    std::string str = ss.str(); 
    return str ; 
}

const char* CSGOptiX::getDefaultSnapPath() const 
{
    assert( foundry );  
    const char* cfbase = foundry->getOriginCFBase(); 
    assert( cfbase ); 
    const char* path = SPath::Resolve(cfbase, "CSGOptiX/snap.jpg" , FILEPATH ); 
    return path ; 
}




/**
CSGOptiX::getRenderStemDefault
--------------------------------

Old opticks has "--nameprefix" argument, this aims to 
do similar with NAMEPREFIX envvar. 

**/
const char* CSGOptiX::getRenderStemDefault() const 
{
    std::stringstream ss ; 
    ss << SSys::getenvvar("NAMEPREFIX","nonamepfx") ; 
    ss << "_" ; 
#ifdef WITH_SGLM
    ss << sglm->get_frame_name() ; 
#else
    ss << "nosglm" ; 
#endif
    
    std::string str = ss.str(); 
    return strdup(str.c_str()); 
}


/**
CSGOptiX::render  (formerly render_snap)
-------------------------------------------
**/

double CSGOptiX::render( const char* stem_ )
{
    const char* stem = stem_ ? stem_ : getRenderStemDefault() ;  // without ext 
#ifdef WITH_SGLM
    sglm->addlog("CSGOptiX::render_snap", stem ); 
#endif

    double dt = render_launch();  

    const char* topline = SSys::getenvvar("TOPLINE", SProc::ExecutableName() ); 
    const char* botline_ = SSys::getenvvar("BOTLINE", nullptr ); 
    const char* outdir = SEventConfig::OutDir();
    const char* outpath = SEventConfig::OutPath(stem, -1, ".jpg" );
    const char* extra = SSim::GetContextBrief();  // scontext::brief giving GPU name 
    std::string bottom_line = CSGOptiX::Annotation(dt, botline_, extra ); 
    const char* botline = bottom_line.c_str() ; 

    LOG(LEVEL)
          << SEventConfig::DescOutPath(stem, -1, ".jpg" );
          ;  
 
    LOG(LEVEL)  
          << " stem " << stem 
          << " outpath " << outpath 
          << " outdir " << ( outdir ? outdir : "-" )
          << " dt " << dt 
          << " topline [" <<  topline << "]"
          << " botline [" <<  botline << "]"
          ; 

    LOG(info) << outpath  << " : " << AnnotationTime(dt, extra)  ; 

    snap(outpath, botline, topline  );   

#ifdef WITH_SGLM
    sglm->fr.save( outdir, stem ); 
    sglm->writeDesc( outdir, stem, ".log" ); 
#endif

    return dt ; 
}




/**
CSGOptiX::snap : Download frame pixels and write to file as jpg.
------------------------------------------------------------------
**/

void CSGOptiX::snap(const char* path_, const char* bottom_line, const char* top_line, unsigned line_height)
{
    const char* path = path_ ? SPath::Resolve(path_, FILEPATH ) : getDefaultSnapPath() ; 
    LOG(LEVEL) << " path " << path ; 

#if OPTIX_VERSION < 70000
    const char* top_extra = nullptr ;
#else
    const char* top_extra = pip->desc(); 
#endif
    const char* topline = SStr::Concat(top_line, top_extra); 

    LOG(LEVEL) << " path_ [" << path_ << "]" ; 
    LOG(LEVEL) << " topline " << topline  ; 

    LOG(LEVEL) << "[ frame.download " ; 
    frame->download(); 
    LOG(LEVEL) << "] frame.download " ; 

    LOG(LEVEL) << "[ frame.annotate " ; 
    frame->annotate( bottom_line, topline, line_height ); 
    LOG(LEVEL) << "] frame.annotate " ; 

    LOG(LEVEL) << "[ frame.snap " ; 
    frame->snap( path  );  
    LOG(LEVEL) << "] frame.snap " ; 

    if(!flight || SStr::Contains(path,"00000"))
    {
        saveMeta(path); 
    }
}

#ifdef WITH_FRAME_PHOTON
void CSGOptiX::writeFramePhoton(const char* dir, const char* name)
{
#if OPTIX_VERSION < 70000
    assert(0 && "not implemented pre-7"); 
#else
    frame->writePhoton(dir, name); 
#endif
}
#endif


int CSGOptiX::render_flightpath() // for making mp4 movies
{
#ifdef WITH_SGLM
    LOG(fatal) << "flightpath rendering not yet implemented in WITH_SGLM branch " ; 
    int rc = 1 ; 
#else
    FlightPath* fp = ok->getFlightPath();   // FlightPath lazily instanciated here (held by Opticks)
    int rc = fp->render( (SRenderer*)this  );
#endif
    return rc ; 
}

void CSGOptiX::saveMeta(const char* jpg_path) const
{
    const char* json_path = SStr::ReplaceEnd(jpg_path, ".jpg", ".json"); 
    LOG(LEVEL) << "[ json_path " << json_path  ; 

    nlohmann::json& js = meta->js ;
    js["jpg"] = jpg_path ; 

#ifdef WITH_SGLM
    js["emm"] = SGeoConfig::EnabledMergedMesh() ;
#else
    js["emm"] = ok->getEnabledMergedMesh() ;
    js["argline"] = ok->getArgLine();
    js["nameprefix"] = ok->getNamePrefix() ;
#endif

    if(foundry->hasMeta())
    {
        js["cfmeta"] = foundry->meta ; 
    }


    const char* extra = SSim::GetContextBrief(); 
    js["scontext"] = extra ? extra : "-" ; 

    const std::vector<double>& t = launch_times ;
    if( t.size() > 0 )
    {
        double mn, mx, av ;
        SVec<double>::MinMaxAvg(t,mn,mx,av);

        js["mn"] = mn ;
        js["mx"] = mx ;
        js["av"] = av ;
    }

    meta->save(json_path);
    LOG(LEVEL) << "] json_path " << json_path  ; 
}


const NP* CSGOptiX::getIAS_Instances(unsigned ias_idx) const
{
    const NP* instances = nullptr ; 
#if OPTIX_VERSION < 70000
#else
    instances = sbt ? sbt->getIAS_Instances(ias_idx) : nullptr ; 
#endif
    return instances ; 
}

/**
CSGOptiX::save
----------------

For debug only 

**/

void CSGOptiX::save(const char* dir) const
{
    LOG(LEVEL) << "[ dir " << ( dir ? dir : "-" ) ; 
    const NP* instances = getIAS_Instances(0) ; 
    if(instances) instances->save(dir, RELDIR, "instances.npy") ;  
    LOG(LEVEL) << "] dir " << ( dir ? dir : "-" ) ; 
}




/**
CSGOptiX::_OPTIX_VERSION
-------------------------

This depends on the the optix.h header only which provides the OPTIX_VERSION macro
so it could be done at the lowest level, no need for it to be 
up at this "elevation"

TODO: relocate to OKConf or SysRap, BUT this must wait until switch to full proj 7 

**/

#define xstr(s) str(s)
#define str(s) #s

int CSGOptiX::_OPTIX_VERSION()   // static 
{
    char vers[16] ; 
    snprintf(vers, 16, "%s",xstr(OPTIX_VERSION)); 
    return std::atoi(vers) ;  
}
