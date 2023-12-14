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
#include "sproc.h"
#include "ssys.h"
#include "smeta.h"
#include "scontext.h"   // GPU metadata
#include "SProf.hh"

#include "SGLM.h"
#include "NP.hh"
#include "SRG.h"
#include "SCAM.h"
#include "SEventConfig.hh"
#include "SGeoConfig.hh"
#include "SSim.hh"
#include "SStr.hh"
#include "SEvt.hh"
#include "SMeta.hh"
#include "SPath.hh"
#include "SVec.hh"
#include "SLOG.hh"
#include "scuda.h"
#include "squad.h"
#include "sframe.h"
#include "salloc.h"



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

Note that SEvt setup and frame hookup formerly 
done in the main is now moved into CSGFoundry::AfterLoadOrCreate
and invokation of SEvt::beginOfEvent SEvt::endOfEvent is done from 
QSim

Note that currently rendering persisting does not use SEvt in the
same way as simtrace and simulate, but it could do in future. 

**/

int CSGOptiX::RenderMain() // static
{
    SEventConfig::SetRGModeRender(); 
    CSGFoundry* fd = CSGFoundry::Load(); 
    CSGOptiX* cx = CSGOptiX::Create(fd) ;
    cx->render(); 
    return 0 ; 
}
int CSGOptiX::SimtraceMain()
{
    SEventConfig::SetRGModeSimtrace(); 
    CSGFoundry* fd = CSGFoundry::Load(); 
    CSGOptiX* cx = CSGOptiX::Create(fd) ;
    cx->simtrace(0); 
    return 0 ; 
}
int CSGOptiX::SimulateMain() // static
{
    SProf::Add("CSGOptiX__SimulateMain_HEAD"); 
    SEventConfig::SetRGModeSimulate(); 
    CSGFoundry* fd = CSGFoundry::Load(); 
    CSGOptiX* cx = CSGOptiX::Create(fd) ;
    for(int i=0 ; i < SEventConfig::NumEvent() ; i++) cx->simulate(i); 
    SProf::UnsetTag(); 
    SProf::Add("CSGOptiX__SimulateMain_TAIL"); 
    SProf::Write("run_meta.txt", true ); // append:true 
    cx->write_Ctx_log(); 
    return 0 ; 
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
int CSGOptiX::Main() // static
{
    CSGFoundry* fd = CSGFoundry::Load(); 
    CSGOptiX* cx = CSGOptiX::Create(fd) ;
    cx->proceed(); 
    return 0 ; 
}

const char* CSGOptiX::Desc()
{
    std::stringstream ss ; 
    ss << "CSGOptiX::Desc" 
       << " Version " << Version() 
       << " PTXNAME " << PTXNAME 
       << " GEO_PTXNAME " << ( GEO_PTXNAME ? GEO_PTXNAME : "-" ) 
#ifdef WITH_CUSTOM4
       << " WITH_CUSTOM4 " 
#else
       << " NOT:WITH_CUSTOM4 "
#endif
       ; 
    std::string str = ss.str(); 
    return strdup(str.c_str()); 
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
CSGOptiX::InitEvt
-------------------

**/

void CSGOptiX::InitEvt( CSGFoundry* fd  )
{
    // sim->serialize() ;  // SSim::serialize stree::serialize into NPFold 

    SEvt* sev = SEvt::CreateOrReuse(SEvt::EGPU) ; 

    sev->setGeo((SGeo*)fd);    // Q: IS THIS USED BY ANYTHING ?  A: YES, Essential set_matline of Cerenkov Genstep 

    std::string* rms = SEvt::RunMetaString() ; 
    assert(rms); 
        
    bool stamp = false ; 
    smeta::Collect(*rms, "CSGOptiX__InitEvt", stamp );  
}

/**
CSGOptiX::InitSim
-------------------

Instanciation of QSim/QEvent requires an SEvt instance 

**/

void CSGOptiX::InitSim( SSim* ssim  )
{
    LOG(LEVEL) << "[" ; ; 

    if(SEventConfig::IsRGModeRender()) return ; 

    LOG_IF(fatal, ssim == nullptr ) << "simulate/simtrace modes require SSim/QSim setup" ;
    assert(ssim);  


    if( ssim->hasTop() == false )
    {
        ssim->serialize() ;  // SSim::serialize stree::serialize into NPFold  (moved from InitEvt)
    } 
    else
    {
        LOG(LEVEL) << " NOT calling SSim::serialize : as already done, loaded ? " ;  
    }

    QSim::UploadComponents(ssim);  

    QSim* qs = QSim::Create() ; 

    LOG(LEVEL) << "]" << qs->desc() ; 
}



/**
CSGOptiX::InitMeta
-------------------

**/

void CSGOptiX::InitMeta(const SSim* ssim  )
{
    std::string gm = GetGPUMeta() ;            // (QSim) scontext sdevice::brief
    SEvt::SetRunMetaString("GPUMeta", gm.c_str() );  // set CUDA_VISIBLE_DEVICES to control 

    std::string switches = QSim::Switches() ;
    SEvt::SetRunMetaString("QSim__Switches", switches.c_str() );  

#ifdef WITH_CUSTOM4
    std::string c4 = "TBD" ; //C4Version::Version(); // octal version number bug in Custom4 v0.1.8 : so skip the version metadata 
    SEvt::SetRunMetaString("C4Version", c4.c_str()); 
#else
    SEvt::SetRunMetaString("C4Version", "NOT-WITH_CUSTOM4" );  
#endif

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
CSGOptiX::Create
--------------------

**/

CSGOptiX* CSGOptiX::Create(CSGFoundry* fd )   
{
    SProf::Add("CSGOptiX__Create_HEAD"); 
    LOG(LEVEL) << "[ fd.descBase " << ( fd ? fd->descBase() : "-" ) ; 

    SetSCTX(); 
    QU::alloc = new salloc ;   // HMM: maybe this belongs better in QSim ? 

    InitEvt(fd); 
    InitSim( const_cast<SSim*>(fd->sim) ); // QSim instanciation after uploading SSim arrays
    InitMeta(fd->sim);                     // recording GPU, switches etc.. into run metadata
    InitGeo(fd);                           // uploads geometry 

    CSGOptiX* cx = new CSGOptiX(fd) ; 

    if(!SEventConfig::IsRGModeRender())
    {
        QSim* qs = QSim::Get() ; 
        qs->setLauncher(cx); 
    } 


    LOG(LEVEL) << "]" ; 
    SProf::Add("CSGOptiX__Create_TAIL"); 
    return cx ; 
}










Params* CSGOptiX::InitParams( int raygenmode, const SGLM* sglm  ) // static
{
    LOG(LEVEL) << "[" ; 
    return new Params(raygenmode, sglm->Width(), sglm->Height(), 1 ) ; 
    LOG(LEVEL) << "]" ; 
}


scontext* CSGOptiX::SCTX = nullptr ; 
void CSGOptiX::SetSCTX()
{ 
    LOG(LEVEL) << "[ new scontext" ;  
    SCTX = new scontext ; // sometimes hangs for few seconds checking for GPU 
    LOG(LEVEL) << "] new scontext" ; 
    LOG(LEVEL) << SCTX->desc() ;    
}

std::string CSGOptiX::GetGPUMeta(){ return SCTX ? SCTX->brief() : "ERR-NO-CSGOptiX-SCTX" ; }


CSGOptiX::CSGOptiX(const CSGFoundry* foundry_) 
    :
    sglm(new SGLM),   // instanciate always to allow view matrix comparisons
    flight(SGeoConfig::FlightConfig()),
    foundry(foundry_),
    prefix(ssys::getenvvar("OPTICKS_PREFIX","/usr/local/opticks")),  // needed for finding ptx
    outdir(SEventConfig::OutFold()),    
    cmaketarget("CSGOptiX"),  
    ptxpath(SStr::PTXPath( prefix, cmaketarget, PTXNAME )),
#if OPTIX_VERSION < 70000 
    geoptxpath(SStr::PTXPath(prefix, cmaketarget, GEO_PTXNAME )),
#else
    geoptxpath(nullptr),
#endif
    tmin_model(ssys::getenvfloat("TMIN",0.1)),    // CAUTION: tmin very different in rendering and simulation 
    raygenmode(SEventConfig::RGMode()),
    params(InitParams(raygenmode,sglm)),

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
    sctx(nullptr),
    sim(QSim::Get()),  
    event(sim == nullptr  ? nullptr : sim->event)
{
    init(); 
}

void CSGOptiX::init()
{
    sglm->addlog("CSGOptiX::init", "start"); 

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
    initFrame(); 

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
CSGOptiX::initFrame (formerly G4CXOpticks::setupFrame)
---------------------------------------------------------

The frame used depends on envvars INST, MOI, OPTICKS_INPUT_PHOTON_FRAME 
it comprises : fr.ce fr.m2w fr.w2m set by CSGTarget::getFrame 

Q: why is the frame needed ?
A: cx rendering viewpoint, input photon frame and the simtrace genstep grid 
   are all based on the frame center, extent and transforms 

Q: Given the sframe and SEvt are from sysrap it feels too high level to do this here, 
   should be at CSG or sysrap level perhaps ? 
   And then CSGOptix could grab the SEvt frame at its initialization. 

TODO: see CSGFoundry::AfterLoadOrCreate for maybe auto frame hookup

**/

void CSGOptiX::initFrame()
{
    sframe _fr = foundry->getFrameE() ; 
    LOG(LEVEL) << _fr ; 
    SEvt::SetFrame(_fr) ; 
    setFrame(_fr);  
}




/**
CSGOptiX::simulate
--------------------

NB the distinction between this and simulate_launch, this 
uses QSim::simulate to do genstep setup prior to calling 
CSGOptiX::simulate_launch via the SCSGOptiX.h protocol

The QSim::simulate argument end:true is used in order 
to invoke SEvt::endOfEvent after the save, this is because
at CSGOptiX level there us no need to allow the user to 
copy hits or other content from SEvt elsewhere. 


**/
double CSGOptiX::simulate(int eventID)
{
    SProf::SetTag(eventID, "A%0.3d_" ) ; 
    assert(sim); 
    bool end = true ;  
    double dt = sim->simulate(eventID, end) ; // (QSim)
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
    double dt = sim->simtrace(eventID) ;  // (QSim)
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
    const char* top = ssys::getenvvar("TOP", TOP ); 
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
    setFrame(ssys::getenvvar("MOI", "-1"));  // TODO: generalize to FRS
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

    extent = sglm->fr.ce.w ; 
    eye = sglm->e ; 
    U = sglm->u ; 
    V = sglm->v ; 
    W = sglm->w ; 
    tmin = sglm->get_near_abs() ; 
    tmax = sglm->get_far_abs() ; 
    cameratype = sglm->cam ; 
    length = 0.f ; 

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

    LOG(LEVEL)
        << "sglm.desc " << std::endl 
        << sglm->desc() 
        ; 

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
    const float4& ce = sglm->fr.ce ;   

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
    ss << ssys::getenvvar("NAMEPREFIX","nonamepfx") ; 
    ss << "_" ; 
    ss << sglm->get_frame_name() ; 
    
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
    sglm->addlog("CSGOptiX::render_snap", stem ); 

    double dt = render_launch();  

    const char* topline = ssys::getenvvar("TOPLINE", sproc::ExecutableName() ); 
    const char* botline_ = ssys::getenvvar("BOTLINE", nullptr ); 
    const char* outdir = SEventConfig::OutDir();
    const char* outpath = SEventConfig::OutPath(stem, -1, ".jpg" );
    std::string _extra = GetGPUMeta();  // scontext::brief giving GPU name 
    const char* extra = strdup(_extra.c_str()) ;  

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

    sglm->fr.save( outdir, stem ); 
    sglm->writeDesc( outdir, stem, ".log" ); 

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
    LOG(fatal) << "flightpath rendering not yet implemented in now default SGLM branch " ; 
    return 1 ; 
}

void CSGOptiX::saveMeta(const char* jpg_path) const
{
    const char* json_path = SStr::ReplaceEnd(jpg_path, ".jpg", ".json"); 
    LOG(LEVEL) << "[ json_path " << json_path  ; 

    nlohmann::json& js = meta->js ;
    js["jpg"] = jpg_path ; 
    js["emm"] = SGeoConfig::EnabledMergedMesh() ;

    if(foundry->hasMeta())
    {
        js["cfmeta"] = foundry->meta ; 
    }

    std::string extra = GetGPUMeta(); 
    js["scontext"] = extra.empty() ? "-" : strdup(extra.c_str()) ; 

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

For debug only  : saves serialized OptixInstances 
seems these were never used for anything. 
See IAS_Builder::Serialize

**/

void CSGOptiX::save(const char* dir) const
{
    assert(0); 

    LOG(LEVEL) << "[ dir " << ( dir ? dir : "-" ) ; 
    const NP* instances = getIAS_Instances(0) ; 
    if(instances) instances->save(dir, RELDIR, "instances.npy") ;  
    LOG(LEVEL) << "] dir " << ( dir ? dir : "-" ) ; 
}

void CSGOptiX::write_Ctx_log(const char* dir) const
{
#if OPTIX_VERSION < 70000
#else
    std::string ctxlog = Ctx::GetLOG() ; 
    spath::Write(ctxlog.c_str() , dir, CTX_LOGNAME  );     
#endif
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
