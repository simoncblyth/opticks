/**
CSGOptiX.cc
============

* NOTE : < 7 BRANCH NO LONGER VIABLE BUT ITS EXPEDIENT TO KEEP
  IT FOR LAPTOP COMPILATION

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
#include "spath.h"
#include "smeta.h"
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

// csg
#include "CSGPrim.h"
#include "CSGFoundry.h"
#include "CSGView.h"

// qudarap
#include "qrng.h"
#include "QU.hh"
#include "QSim.hh"
#include "qsim.h"
#include "QEvt.hh"

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
    delete cx ;
    return 0 ;
}
int CSGOptiX::SimtraceMain()
{
    SEventConfig::SetRGModeSimtrace();
    CSGFoundry* fd = CSGFoundry::Load();
    CSGOptiX* cx = CSGOptiX::Create(fd) ;
    cx->simtrace(0);
    delete cx ;
    return 0 ;
}


/**
CSGOptiX::SimulateMain
-----------------------

**/

int CSGOptiX::SimulateMain() // static
{
    SProf::Add("CSGOptiX__SimulateMain_HEAD");
    SEventConfig::SetRGModeSimulate();
    CSGFoundry* fd = CSGFoundry::Load();
    CSGOptiX* cx = CSGOptiX::Create(fd) ;
    bool reset = true ;
    for(int i=0 ; i < SEventConfig::NumEvent() ; i++) cx->simulate(i, reset);
    SProf::UnsetTag();
    SProf::Add("CSGOptiX__SimulateMain_TAIL");
    SProf::Write();
    cx->write_Ctx_log();
    delete cx ;
    return 0 ;
}





const char* CSGOptiX::Desc()
{
    std::stringstream ss ;
    ss << "CSGOptiX::Desc"
       << " Version " << Version()
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
CSGOptiX::InitEvt  TODO : THIS DOES NOT USE GPU : SO SHOULD BE ELSEWHERE
--------------------------------------------------------------------------

Invoked from CSGOptiX::Create


Q: Why the SEvt geometry connection ?
A: Needed for global to local transform conversion

Q: What uses SEVt::setGeo (SGeo) ?
A: Essential set_matline of Cerenkov Genstep

**/

void CSGOptiX::InitEvt( CSGFoundry* fd  )
{
    SEvt* sev = SEvt::CreateOrReuse(SEvt::EGPU) ;

    sev->setGeo((SGeo*)fd);

    std::string* rms = SEvt::RunMetaString() ;
    assert(rms);

    bool stamp = false ;
    smeta::Collect(*rms, "CSGOptiX__InitEvt", stamp );
}

/**
CSGOptiX::InitSim
-------------------

Invoked from CSGOptiX::Create
Instanciation of QSim/QEvt requires an SEvt instance

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

Invoked from CSGOptiX::Create prior to instanciation

**/

void CSGOptiX::InitMeta()
{
    std::string gm = SEventConfig::GetGPUMeta() ;     // (QSim) scontext sdevice::brief
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

Invoked from CSGOptiX::Create

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

Canonical invokation from G4CXOpticks::setGeometry_ when a GPU is detected


WIP: static methods cannot be enforced in a protocol, so perhaps just do the below
within the ctor ?

**/

CSGOptiX* CSGOptiX::Create(CSGFoundry* fd )
{
    SProf::Add("CSGOptiX__Create_HEAD");
    LOG(LEVEL) << "[ fd.descBase " << ( fd ? fd->descBase() : "-" ) ;


    InitEvt(fd);
    InitSim( const_cast<SSim*>(fd->sim) ); // QSim instanciation after uploading SSim arrays
    InitMeta();                            // recording GPU, switches etc.. into run metadata
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





CSGOptiX::~CSGOptiX()
{
    destroy();
}

/**
CSGOptiX::CSGOptiX
--------------------

Instanciated at near to main level in both running modes:

* pure-Opticks running (no Geant4) eg via cxs_min.sh uses CSGOptiX::SimulateMain
  that instanciates via CSGOptiX::Create with event processing CSGOptiX::simulate
  called directly from CSGOptiX::SimulateMain

* Geant4 integrated running eg from OJ python "main", instanciation again uses
  CSGOptiX::Create that is invoked from G4CXOpticks::setGeometry_
  with event processing CSGOptiX::simulate called from
  junoSD_PMT_v2_Opticks::EndOfEvent_Simulate

**/


CSGOptiX::CSGOptiX(const CSGFoundry* foundry_)
    :
    sglm(SGLM::Get()),
    flight(SGeoConfig::FlightConfig()),
    foundry(foundry_),
    outdir(SEventConfig::OutFold()),
#ifdef CONFIG_Debug
    _optixpath("${CSGOptiX__optixpath:-$OPTICKS_PREFIX/optix/objects-Debug/CSGOptiX_OPTIX/CSGOptiX7.ptx}"),
#elif CONFIG_Release
    _optixpath("${CSGOptiX__optixpath:-$OPTICKS_PREFIX/optix/objects-Release/CSGOptiX_OPTIX/CSGOptiX7.ptx}"),
#else
    _optixpath(nullptr),
#endif
    optixpath(_optixpath ? spath::Resolve(_optixpath) : nullptr),
    tmin_model(ssys::getenvfloat("TMIN",0.1)),    // CAUTION: tmin very different in rendering and simulation
    kernel_count(0),
    raygenmode(SEventConfig::RGMode()),
    params(InitParams(raygenmode,sglm)),
    ctx(nullptr),
    pip(nullptr),
    sbt(nullptr),
    framebuf(nullptr),
    meta(new SMeta),
    kernel_dt(0.),
    sctx(nullptr),
    sim(QSim::Get()),
    qev(sim == nullptr  ? nullptr : sim->qev)   // QEvt
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
        << " qev " << qev
        ;

    assert( outdir && "expecting OUTDIR envvar " );

    LOG(LEVEL) << " _optixpath " << _optixpath  ;
    LOG(LEVEL) << " optixpath " << optixpath  ;

    initMeta();
    initCtx();
    initPIP();
    initSBT();
    initCheckSim();
    initStack();
    initParams();
    initGeometry();
    initSimulate();
    initFrame();
    initRender();
    initPIDXYZ();

    LOG(LEVEL) << "]" ;
}

/**
CSGOptiX::initMeta
-------------------

Record metadata regarding the *optixpath* kernel source into RunMeta

**/


void CSGOptiX::initMeta()
{
    int64_t mtime = spath::last_write_time(optixpath);
    std::string str = sstamp::Format(mtime);
    int64_t age_secs = sstamp::age_seconds(mtime);
    int64_t age_days = sstamp::age_days(mtime);

    SEvt::SetRunMetaString("optixpath", optixpath );
    SEvt::SetRunMetaString("optixpath_mtime_str", str.c_str() );
    SEvt::SetRunMeta<int64_t>("optixpath_mtime", mtime );
    SEvt::SetRunMeta<int64_t>("optixpath_age_secs", age_secs );
    SEvt::SetRunMeta<int64_t>("optixpath_age_days", age_days );
}


/**
CSGOptiX::initCtx
-------------------

Instanciate the OptixDeviceContext

**/

void CSGOptiX::initCtx()
{
    LOG(LEVEL) << "[" ;
    ctx = new Ctx ;
    LOG(LEVEL) << std::endl << ctx->desc() ;
    LOG(LEVEL) << "]" ;
}


/**
CSGOptiX::initPIP
-------------------

Instanciate PIP pipeline

**/

void CSGOptiX::initPIP()
{
    LOG(LEVEL) << "["  ;
    LOG(LEVEL)
        << " optixpath " << ( optixpath ? optixpath : "-" )
        ;

    pip = new PIP(optixpath, ctx->props ) ;
    LOG(LEVEL) << "]" ;
}

/**
CSGOptiX::initSBT
--------------------

Instanciate SBT shader binding table

**/


void CSGOptiX::initSBT()
{
    LOG(LEVEL) << "[" ;
    sbt = new SBT(pip) ;
    LOG(LEVEL) << "]" ;
}



/**
CSGOptiX::initCheckSim
-----------------------

Check (QSim)sim instance for non-render modes

**/


void CSGOptiX::initCheckSim()
{
    if(SEventConfig::IsRGModeRender()) return ;
    LOG(LEVEL) << " sim " << sim << " qev " << qev ;
    LOG_IF(fatal, sim == nullptr) << "simtrace/simulate modes require instanciation of QSim before CSGOptiX " ;
    assert(sim);
}


void CSGOptiX::initStack()
{
    LOG(LEVEL);
    pip->configureStack();
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

    LOG(LEVEL) << "[ sbt.setFoundry " ;
    sbt->setFoundry(foundry);
    params->handle = sbt->getTOPHandle() ;
    LOG(LEVEL) << "] sbt.setFoundry " ;
    LOG(LEVEL) << "]" ;
}


/**
CSGOptiX::initSimulate
------------------------

* Once only (not per-event) simulate setup tasks ..  perhaps rename initPhys

Sets device pointers for params.sim params.evt so must be after upload

Q: where are sim and evt uploaded ?
A: QSim::QSim and QSim::init_sim are where sim and evt are populated and uploaded


HMM: get d_sim (qsim.h) now holds d_evt (sevent.h) but this is getting evt again rom QEvt ?
TODO: eliminate params->evt to make more use of the qsim.h encapsulation

**/

void CSGOptiX::initSimulate()
{
    LOG(LEVEL) ;
    params->sim = sim ? sim->getDevicePtr() : nullptr ;  // qsim<float>*
    params->evt = qev ? qev->getDevicePtr() : nullptr ;  // sevent*

    params->tmin0 = SEventConfig::PropagateEpsilon0() ;  // epsilon used after step points with flags in below mask
    params->PropagateEpsilon0Mask = SEventConfig::PropagateEpsilon0Mask();  // eg from CK|SI|TO|SC|RE

    params->PropagateRefine = SEventConfig::PropagateRefine();
    params->PropagateRefineDistance = SEventConfig::PropagateRefineDistance();  // approx distance beyond which to refine intersect with 2nd trace

    params->tmin = SEventConfig::PropagateEpsilon() ;  // eg 0.1 0.05 to avoid self-intersection off boundaries
    params->tmax = 1000000.f ;
    params->max_time = SEventConfig::MaxTime() ;


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
    sframe _fr = foundry->getFrameE() ;   // TODO: migrate to lighweight sfr from stree level
    LOG(LEVEL) << _fr ;
    SEvt::SetFrame(_fr) ;

    sfr _lfr = _fr.spawn_lite();
    setFrame(_lfr);
}




/**
CSGOptiX::initRender
--------------------------

To use externally managed device pixels call this
prior to render/render_launch with the device pixel pointer,
otherwise this is called with nullptr d_pixel that arranges
internally allocated device pixels.

**/

void CSGOptiX::initRender()
{
    LOG(LEVEL) << "[" ;
    framebuf = new Frame(params->width, params->height, params->depth, nullptr ) ;
    LOG(LEVEL) << "]" ;

    if(SEventConfig::IsRGModeRender())
    {
        setFrame(); // MOI
        // HMM: done twice ?
    }

    params->pixels = framebuf->d_pixel ;
    params->isect  = framebuf->d_isect ;
#ifdef WITH_FRAME_PHOTON
    params->fphoton = framebuf->d_photon ;
#else
    params->fphoton = nullptr ;
#endif
}

void CSGOptiX::initPIDXYZ()
{
    qvals(params->pidxyz, "PIDXYZ", "-1:-1:-1", -1 ) ;
    const char* PIDXYZ = ssys::getenvvar("PIDXYZ") ;
    if(PIDXYZ && strcmp(PIDXYZ,"MIDDLE") == 0 )
    {
        LOG(info) << " special casing PIDXYZ MIDDLE " ;
        params->pidxyz.x = params->width/2 ;
        params->pidxyz.y = params->height/2 ;
        params->pidxyz.z = params->depth/2 ;
    }

    LOG(LEVEL) << " params->pidxyz " << params->pidxyz ;
}


void CSGOptiX::setExternalDevicePixels(uchar4* _d_pixel )
{
    framebuf->setExternalDevicePixels(_d_pixel) ;
    params->pixels = framebuf->d_pixel ;
}


void CSGOptiX::destroy()
{
    LOG(LEVEL);
    delete sbt ;
    delete pip ;
}




/**
CSGOptiX::simulate
--------------------

NB the distinction between this and simulate_launch, this
uses QSim::simulate to do genstep setup prior to calling
CSGOptiX::simulate_launch via the SCSGOptiX.h protocol

The QSim::simulate argument reset:true is used in order
to invoke SEvt::endOfEvent after the save, this is because
at CSGOptiX level there is no need to allow the user to
copy hits or other content from SEvt elsewhere.





**/
double CSGOptiX::simulate(int eventID, bool reset)
{
    assert(sim);
    double dt = sim->simulate(eventID, reset) ; // (QSim)
    return dt ;
}


void CSGOptiX::reset(int eventID)
{
    assert(sim);
    sim->reset(eventID); // (QSim)
}




/**
CSGOptiX::simulate
-------------------

High level interface used by CSGOptiXService.h

**/

NP* CSGOptiX::simulate(const NP* gs, int eventID)
{
    return sim->simulate(gs, eventID);
}



/**
CSGOptiX::simtrace
--------------------

NB the distinction between this and simtrace_launch, this
uses QSim::simtrace to do genstep setup prior to calling
CSGOptiX::simtrace_launch via the SCSGOptiX.h protocol

Simtrace effectively always has reset:true because it
always uses SEvt saving, unlike "simulate" which needs
to support grabbing of hits into Geant4 collections.

**/

double CSGOptiX::simtrace(int eventID)
{
    LOG(LEVEL) << "[" ;
    assert(sim);
    double dt = sim->simtrace(eventID) ;  // (QSim)
    LOG(LEVEL) << "] " << dt  ;
    return dt ;
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
    sframe fr = foundry->getFrame(frs) ;
    sfr lfr = fr.spawn_lite();
    setFrame(lfr);

}
void CSGOptiX::setFrame(const float4& ce )
{
    sfr lfr ;   // m2w w2m default to identity

    lfr.ce.x = ce.x ;
    lfr.ce.y = ce.y ;
    lfr.ce.z = ce.z ;
    lfr.ce.w = ce.w ;

    setFrame(lfr);
}

/**
CSGOptiX::setFrame into the SGLM.h instance
----------------------------------------------

Note that SEvt already holds an sframe used for input photon transformation,
the sframe here is used for raytrace rendering.  Could perhaps rehome sglm
into SEvt and use a single sframe for both input photon transformation
and rendering ?

**/

void CSGOptiX::setFrame(const sfr& lfr )
{
    sglm->set_frame(lfr);   // TODO: aim to remove sframe from sglm ? instead operate at ce (or sometimes m2w w2m level)

    LOG(LEVEL) << "sglm.desc:" << std::endl << sglm->desc() ;

    LOG(LEVEL) << lfr.desc() ;

    LOG(LEVEL)
        << " sglm.TMIN " << sglm->TMIN
        << " sglm.tmin_abs " << sglm->tmin_abs()
        ;

    LOG(LEVEL) << "]" ;
}







void CSGOptiX::prepareParamRender()
{
    int prepareParamRender_DEBUG = ssys::getenvint(_prepareParamRender_DEBUG, 0) ;

    float extent = sglm->fr.ce.w ;
    const glm::vec3& eye = sglm->e ;
    const glm::vec3& U = sglm->u ;
    const glm::vec3& V = sglm->v ;
    const glm::vec3& W = sglm->w ;
    const glm::vec3& WNORM = sglm->wnorm ;
    const glm::vec4& ZPROJ = sglm->zproj ;

    float tmin = sglm->get_near_abs() ;
    float tmax = sglm->get_far_abs() ;
    unsigned vizmask = sglm->vizmask ;
    unsigned cameratype = sglm->cam ;
    int traceyflip = sglm->traceyflip ;
    int rendertype = sglm->rendertype ;
    float length = 0.f ;

    LOG_IF(info, prepareParamRender_DEBUG > 0 && kernel_count == 0)
        << _prepareParamRender_DEBUG << ":" << prepareParamRender_DEBUG
        << std::endl
        << std::setw(20) << " kernel_count " << kernel_count << std::endl
        << std::setw(20) << " extent "     << extent << std::endl
        << std::setw(20) << " sglm.fr.ce.w "  << sglm->fr.ce.w << std::endl
        << std::setw(20) << " sglm.getGazeLength "  << sglm->getGazeLength()  << std::endl
        << std::setw(20) << " comp.length"   << length
        << std::endl
        << std::setw(20) << " tmin "       << tmin  << std::endl
        << std::setw(20) << " tmax "       << tmax  << std::endl
        << std::setw(20) << " vizmask "    << vizmask  << std::endl
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
        << std::setw(20) << " WNORM ("     << WNORM.x << " " << WNORM.y << " " << WNORM.z << " ) " << std::endl
        << std::endl
        << std::setw(20) << " cameratype " << cameratype << " "           << SCAM::Name(cameratype) << std::endl
        << std::setw(20) << " traceyflip " << traceyflip << std::endl
        << std::setw(20) << " sglm.cam " << sglm->cam << " " << SCAM::Name(sglm->cam) << std::endl
        << std::setw(20) << " ZPROJ ("     << ZPROJ.x << " " << ZPROJ.y << " " << ZPROJ.z << " " << ZPROJ.w << " ) " << std::endl
        << std::endl
        << "SGLM::DescEyeBasis (sglm->e,w,v,w)\n"
        << SGLM::DescEyeBasis( sglm->e, sglm->u, sglm->v, sglm->w )
        << std::endl
        << std::endl
        <<  "sglm.descEyeBasis\n"
        << sglm->descEyeBasis()
        << std::endl
        << "Composition basis  SGLM::DescEyeBasis( eye, U, V, W ) \n"
        << SGLM::DescEyeBasis( eye, U, V, W )
        << std::endl
        << "sglm.descELU \n"
        << sglm->descELU()
        << std::endl
        << std::endl
        << "sglm.descLog "
        << std::endl
        << sglm->descLog()
        << std::endl
        ;



    params->setView(eye, U, V, W, WNORM );
    params->setCamera(tmin, tmax, cameratype, traceyflip, rendertype, ZPROJ );
    params->setVizmask(vizmask);

    LOG(level) << std::endl << params->desc() ;

    if(flight) return ;

    LOG(level)
        << "sglm.desc " << std::endl
        << sglm->desc()
        ;

}


/**
CSGOptiX::prepareParamSimulate
-------------------------------

Per-event simulate setup invoked just prior to optix launch

QSim::get_photon_slot_offset/QEvt::get_photon_slot_offset returns

**/

void CSGOptiX::prepareParamSimulate()
{
    LOG(LEVEL);
    params->set_photon_slot_offset(sim->get_photon_slot_offset());
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
    const glm::tvec4<double>& ce = sglm->fr.ce ;

    params->setCenterExtent(ce.x, ce.y, ce.z, ce.w);
    switch(raygenmode)
    {
        case SRG_RENDER   : prepareParamRender()   ; break ;
        case SRG_SIMTRACE : prepareParamSimulate() ; break ;
        case SRG_SIMULATE : prepareParamSimulate() ; break ;
    }

    params->upload();
    LOG_IF(level, !flight) << params->detail();
}


/**
CSGOptiX::launch
-------------------

For what happens next, see CSGOptiX7.cu::__raygen__rg OR CSGOptiX6.cu::raygen
Depending on params.raygenmode the "render" or "simulate" method is called.

Formerly followed an OptiX 7 SDK example, creating a stream for the launch::

    CUstream stream ;
    CUDA_CHECK( cudaStreamCreate( &stream ) );
    OPTIX_CHECK( optixLaunch( pip->pipeline, stream, d_param, sizeof( Params ), &(sbt->sbt), width, height, depth ) );

But that leaks 14kb for every launch, see:

* ~/opticks/notes/issues/okjob_GPU_memory_leak.rst
* ~/opticks/CSGOptiX/cxs_min_igs.sh

Instead using default "stream=0" avoids the leak.
Presumably that means every launch uses the same single default stream.

**/

double CSGOptiX::launch()
{
    bool DEBUG_SKIP_LAUNCH = ssys::getenvbool("CSGOptiX__launch_DEBUG_SKIP_LAUNCH") ;

    prepareParam();
    if(raygenmode != SRG_RENDER) assert(qev) ;

    unsigned width = 0 ;
    unsigned height = 0 ;
    unsigned depth  = 0 ;
    switch(raygenmode)
    {
        case SRG_RENDER:    { width = params->width         ; height = params->height ; depth = params->depth ; } ; break ;
        case SRG_SIMTRACE:  { width = qev->getNumSimtrace() ; height = 1              ; depth = 1             ; } ; break ;
        case SRG_SIMULATE:  { width = qev->getNumPhoton()   ; height = 1              ; depth = 1             ; } ; break ;
    }

    bool expect = width > 0 ;
    LOG_IF(fatal, !expect)
        << " qev.getNumSimtrace " << ( qev ? qev->getNumSimtrace() : -1 )
        << " qev.getNumPhoton   " << ( qev ? qev->getNumPhoton() : -1 )
        << " width " << width
        << " height " << height
        << " depth " << depth
        << " expect " << ( expect ? "YES" : "NO " )
        ;

    assert(expect );

    typedef std::chrono::time_point<std::chrono::high_resolution_clock> TP ;
    typedef std::chrono::duration<double> DT ;
    TP _t0 = std::chrono::high_resolution_clock::now();
    int64_t t0 = sstamp::Now();

    LOG(LEVEL)
         << " raygenmode " << raygenmode
         << " SRG::Name(raygenmode) " << SRG::Name(raygenmode)
         << " width " << width
         << " height " << height
         << " depth " << depth
         << " DEBUG_SKIP_LAUNCH " << ( DEBUG_SKIP_LAUNCH ? "YES" : "NO " )
         ;

    if(DEBUG_SKIP_LAUNCH == false)
    {
        CUdeviceptr d_param = (CUdeviceptr)Params::d_param ; ;
        assert( d_param && "must alloc and upload params before launch");

        //cudaStream_t stream = SMgr::Stream();
        cudaStream_t stream = 0 ; // default stream
        OPTIX_CHECK( optixLaunch( pip->pipeline, (CUstream)stream, d_param, sizeof( Params ), &(sbt->sbt), width, height, depth ) );

        CUDA_SYNC_CHECK();
        // see CSG/CUDA_CHECK.h the CUDA_SYNC_CHECK does cudaDeviceSyncronize
        // THIS LIKELY HAS LARGE PERFORMANCE IMPLICATIONS : BUT NOT EASY TO AVOID (MULTI-BUFFERING ETC..)
        kernel_count += 1 ;
    }


    TP _t1 = std::chrono::high_resolution_clock::now();
    DT _dt = _t1 - _t0;

    int64_t t1 = sstamp::Now();
    int64_t dt = t1 - t0 ;

    kernel_dt = _dt.count() ;   // formerly mis-named launch_dt
    kernel_times.push_back(kernel_dt);

    kernel_times_.push_back(dt);

    LOG(LEVEL)
          << " (params.width, params.height, params.depth) ( "
          << params->width << "," << params->height << "," << params->depth << ")"
          << std::fixed << std::setw(7) << std::setprecision(4) << kernel_dt
          ;
    return kernel_dt ;
}



/**
CSGOptiX::render_launch CSGOptiX::simtrace_launch CSGOptiX::simulate_launch
--------------------------------------------------------------------------------

All the launch set (double)dt

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

Example NAMEPREFIX from cxr_min.sh::

   cxr_min__eye_0,0.8,0__zoom_1.0__tmin_0.1_

Example resulting stem::

   cxr_min__eye_0,0.8,0__zoom_1.0__tmin_0.1__ALL

WIP: interactive view control makes this approach obsolete
To provide dynamic naming have added::

    ssys::setenvctx
    ssys::setenvmap

The idea being to generate a map<string,string> of
view params for feeding into the environment that
is used in the resolution of a name pattern that is
specied from bash, ie::

    cxr_min_eye_${EYE}__zoom_${ZOOM}__tmin_${TMIN}

HMM: SGLM seems more appropriate place to do this than here

**/
const char* CSGOptiX::getRenderStemDefault() const
{
    const std::string& fr_name = sglm->fr.get_name() ;

    std::stringstream ss ;
    ss << ssys::getenvvar("NAMEPREFIX","nonamepfx") ;
    ss << "_" ;
    ss << ( fr_name.empty() ? "no_frame_name" : fr_name ) ;

    std::string str = ss.str();
    return strdup(str.c_str());
}




bool CSGOptiX::handle_snap(int wanted_snap)
{
    bool can_handle = wanted_snap == 1 || wanted_snap == 2 ;
    if(!can_handle) return false ;
    switch(wanted_snap)
    {
        case 1: render_save()          ; break ;
        case 2: render_save_inverted() ; break ;
    }
    return true ;
}


/**
CSGOptiX::render (formerly render_snap)
-------------------------------------------
**/

double CSGOptiX::render( const char* stem_ )
{
    render_launch();
    render_save(stem_);
    return kernel_dt ;
}



/**
CSGOptiX::render_save
----------------------

TODO: update file naming impl, currently using old inflexible approach

**/
void CSGOptiX::render_save(const char* stem_)
{
    render_save_(stem_, false);
}
void CSGOptiX::render_save_inverted(const char* stem_)
{
    render_save_(stem_, true);
}


void CSGOptiX::render_save_(const char* stem_, bool inverted)
{
    const char* outdir = SEventConfig::OutDir();
    const char* stem = stem_ ? stem_ : getRenderStemDefault() ;  // without ext

    bool unique = true ;
    const char* outpath = SEventConfig::OutPath(stem, -1, ".jpg", unique );

    LOG(LEVEL)
          << SEventConfig::DescOutPath(stem, -1, ".jpg", unique );
          ;

    std::string u_outdir ;
    std::string u_stem ;
    std::string u_ext ;

    [[maybe_unused]] int rc = spath::SplitExt( u_outdir, u_stem, u_ext, outpath )  ;
    assert(rc == 0 );

    sglm->addlog("CSGOptiX::render_snap", u_stem.c_str() );


    const char* topline = ssys::getenvvar("TOPLINE", sproc::ExecutableName() );
    std::string _extra = SEventConfig::GetGPUMeta();  // scontext::brief giving GPU name
    const char* extra = strdup(_extra.c_str()) ;

    const char* botline_ = ssys::getenvvar("BOTLINE", nullptr );
    std::string bottom_line = CSGOptiX::Annotation(kernel_dt, botline_, extra );
    const char* botline = bottom_line.c_str() ;


    LOG(LEVEL)
          << " stem " << stem
          << " outpath " << outpath
          << " outdir " << ( outdir ? outdir : "-" )
          << " kernel_dt " << kernel_dt
          << " topline [" <<  topline << "]"
          << " botline [" <<  botline << "]"
          ;

    LOG(info) << outpath  << " : " << AnnotationTime(kernel_dt, extra)  ;

    unsigned line_height = 24 ;
    snap(outpath, botline, topline, line_height, inverted  );


    sglm->save( u_outdir.c_str(), u_stem.c_str() );
}


/**
CSGOptiX::snap : Download frame pixels and write to file as jpg.
------------------------------------------------------------------

WIP: contrast this with SGLFW::snap_local and consider if more consolidation is possible


SGLFW::snap_local
    Uses OpenGL glReadPixels to download pixels and write them to file

CSGOptiX::snap
    OptiX/CUDA level download ray traced pixels and write to file

**/

void CSGOptiX::snap(const char* path_, const char* bottom_line, const char* top_line, unsigned line_height, bool inverted )
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
    if( inverted == false )
    {
        framebuf->download();
    }
    else
    {
        framebuf->download_inverted();
    }
    LOG(LEVEL) << "] frame.download " ;

    LOG(LEVEL) << "[ frame.annotate " ;
    framebuf->annotate( bottom_line, topline, line_height );
    LOG(LEVEL) << "] frame.annotate " ;

    LOG(LEVEL) << "[ frame.snap " ;
    framebuf->snap( path  );
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
    framebuf->writePhoton(dir, name);
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

    std::string extra = SEventConfig::GetGPUMeta();
    js["scontext"] = extra.empty() ? "-" : strdup(extra.c_str()) ;

    const std::vector<double>& t = kernel_times ;
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
SEventConfig::so it could be done at the lowest level, no need for it to be
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
