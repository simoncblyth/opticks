/**
CSGOptiX.cc
============

This code contains two branches for old (OptiX < 7) and new (OptiX 7+) API

Branched aspects:

1. OptiX7Test.cu vs OptiX6Test.cu  
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
#include "SStr.hh"
#include "SSys.hh"
#include "SEvt.hh"
#include "SMeta.hh"
#include "SPath.hh"
#include "SVec.hh"
#include "PLOG.hh"
#include "scuda.h"
#include "squad.h"
#include "sframe.h"

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

const plog::Severity CSGOptiX::LEVEL = PLOG::EnvLevel("CSGOptiX", "DEBUG" ); 

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

const char* CSGOptiX::desc() const 
{
    std::stringstream ss ; 
    ss << "CSGOptiX " ; 
#if OPTIX_VERSION < 70000
    ss << " Six " ; 
#else
    ss << pip->desc() ; 
#endif
    std::string s = ss.str(); 
    return strdup(s.c_str()); 
}


void CSGOptiX::InitGeo(  CSGFoundry* fd )
{
    fd->upload(); 
}

void CSGOptiX::InitSim( const SSim* ssim  )
{
    if(SEventConfig::IsRGModeRender()) return ; 
    if(ssim == nullptr) LOG(fatal) << "simulate/simtrace modes require SSim/QSim setup" ;
    assert(ssim);  

    QSim::UploadComponents(ssim);  

    QSim* sim = new QSim ; 
    LOG(info) << sim->desc() ; 
}


/**
CSGOptiX::Create
--------------------

**/

CSGOptiX* CSGOptiX::Create(CSGFoundry* fd )   
{
    LOG(info) << "fd.descBase " << ( fd ? fd->descBase() : "-" ) ; 
    std::cout << "fd.descBase " << ( fd ? fd->descBase() : "-" ) << std::endl ; 

    InitSim(fd->sim); 
    InitGeo(fd); 

    CSGOptiX* cx = new CSGOptiX(fd) ; 

    QSim* qs = QSim::Get() ; 

    qs->setLauncher(cx); 
    QEvent* event = qs->event ; 
    event->setMeta( fd->meta.c_str() );

    // TODO: setup QEvent as SProvider of NP arrays to SEvt so SEvt can control QEvent download

    return cx ; 
}

CSGOptiX::CSGOptiX(const CSGFoundry* foundry_) 
    :
#ifdef WITH_SGLM
#else
    ok(Opticks::Instance()),
    composition(ok->getComposition()),
#endif
    sglm(new SGLM),   // instanciate always to allow view matrix comparisons
    moi("-1"),
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
    params(new Params(raygenmode, sglm->Width(), sglm->Height(), 1 )),
#else
    params(new Params(raygenmode, ok->getWidth(), ok->getHeight(), ok->getDepth() )),
#endif

#if OPTIX_VERSION < 70000
    six(new Six(ptxpath, geoptxpath, params)),
    frame(new Frame(params->width, params->height, params->depth, six->d_pixel, six->d_isect, six->d_photon)), 
#else
    ctx(new Ctx),
    pip(new PIP(ptxpath)), 
    sbt(new SBT(pip)),
    frame(new Frame(params->width, params->height, params->depth)), 
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
    LOG(LEVEL) 
        << "[" 
        << " raygenmode " << raygenmode
        << " SRG::Name(raygenmode) " << SRG::Name(raygenmode)
        ;  

    assert( prefix && "expecting PREFIX envvar pointing to writable directory" );
    assert( outdir && "expecting OUTDIR envvar " );

    LOG(LEVEL) << " ptxpath " << ptxpath  ; 
    LOG(LEVEL) << " geoptxpath " << ( geoptxpath ? geoptxpath : "-" ) ; 

    initStack(); 
    initParams(); 
    initGeometry();
    initRender(); 
    initSimulate(); 

    LOG(LEVEL) << "]" ; 
}

void CSGOptiX::initStack()
{
    LOG(info); 
#if OPTIX_VERSION < 70000
#else
    pip->configureStack(); 
#endif

}


void CSGOptiX::initParams()
{
    params->device_alloc(); 
}
void CSGOptiX::initGeometry()
{
    params->node = foundry->d_node ; 
    params->plan = foundry->d_plan ; 
    params->tran = nullptr ; 
    params->itra = foundry->d_itra ; 

    bool is_uploaded =  params->node != nullptr ;
    if(!is_uploaded) LOG(fatal) << "foundry must be uploaded prior to CSGOptiX::initGeometry " ;  
    assert( is_uploaded ); 

#if OPTIX_VERSION < 70000
    six->setFoundry(foundry);
#else
    sbt->setFoundry(foundry); 
#endif

    const char* top = Top(); 
    setTop(top); 
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
    params->fphoton = frame->d_photon ; 
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
    if(SEventConfig::IsRGModeRender() == false)
    {
        if(sim == nullptr) LOG(fatal) << "simtrace/simulate modes require instanciation of QSim before CSGOptiX " ; 
        assert(sim); 
    }

    params->sim = sim ? sim->getDevicePtr() : nullptr ;  // qsim<float>*
    params->evt = event ? event->getDevicePtr() : nullptr ;  // qevent*
    params->tmin = SEventConfig::PropagateEpsilon() ;  // eg 0.1 0.05 to avoid self-intersection off boundaries
    params->tmax = 1000000.f ; 
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

**/

void CSGOptiX::setFrame()
{
    setFrame(SSys::getenvvar("MOI", "-1"));  // TODO: generalize to FRS
}
void CSGOptiX::setFrame(const char* frs)
{
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
void CSGOptiX::setFrame(const sframe& fr_ )
{
    fr = fr_ ; 
    const float4& ce = fr.ce ; 
    const qat4* m2w = &fr.m2w ; 
    const qat4* w2m = &fr.w2m ; 

    LOG(info) << "[" ; 

    float extent = ce.w ; 
    float tmin = extent*tmin_model ;   // tmin_model from TMIN envvar with default of 0.1 (units of extent) 

    // HMM:SGLM::set_frame ?
    sglm->set_ce(ce.x, ce.y, ce.z, ce.w ); 
    sglm->set_m2w(m2w, w2m);
    sglm->update();  
    sglm->set_near_abs(tmin) ; 
    sglm->update();  

    LOG(info) << "sglm.desc:" << std::endl << sglm->desc() ; 


#ifdef WITH_SGLM
#else
    bool autocam = true ; 
    composition->setCenterExtent(ce, autocam, m2w, w2m );  // model2world view setup 
    composition->setNear(tmin); 
    LOG(info) << std::endl << composition->getCameraDesc() ;  
#endif

    LOG(info) 
        << " ce [ " << ce.x << " " << ce.y << " " << ce.z << " " << ce.w << "]" 
        << " tmin_model " << tmin_model
        << " tmin " << tmin 
        << " m2w " << m2w
        << " w2m " << w2m
        ; 

    if(m2w) LOG(info) << "m2w " << *m2w ; 
    if(w2m) LOG(info) << "w2m " << *w2m ; 

    LOG(info) << "]" ; 
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
    extent = sglm->ce.w ; 
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
        LOG(info)
            << std::endl 
            << std::setw(20) << " extent "     << extent << std::endl 
            << std::setw(20) << " sglm.ce.w "  << sglm->ce.w << std::endl 
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

        std::cout << "SGLM::DescEyeBasis (sglm->e,w,v,w) " << std::endl << SGLM::DescEyeBasis( sglm->e, sglm->u, sglm->v, sglm->w ) << std::endl ;
        std::cout <<  "sglm.descEyeBasis " << std::endl << sglm->descEyeBasis() << std::endl ; 
        std::cout << "Composition basis " << std::endl << SGLM::DescEyeBasis( eye, U, V, W ) << std::endl ;
        LOG(info) << std::endl  << "sglm.descELU " << std::endl << sglm->descELU() << std::endl ; 
        LOG(info) << std::endl << "sglm.descLog " << std::endl << sglm->descLog() << std::endl ; 

    }


    params->setView(eye, U, V, W);
    params->setCamera(tmin, tmax, cameratype ); 

    LOG(info) << std::endl << params->desc() << std::endl ; 


    if(flight) return ; 

#ifdef WITH_SGLM
    LOG(info)
        << "sglm.desc " << std::endl 
        << sglm->desc() 
        ; 
#else
    LOG(info)
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
    const glm::vec4& ce = sglm->ce ;   
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
    if(!flight) params->dump(" CSGOptiX::prepareParam"); 
#endif
}





/**
CSGOptiX::launch
-------------------

For what happens next, see OptiX7Test.cu::__raygen__rg OR OptiX6Test.cu::raygen
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

#if OPTIX_VERSION < 70000
    LOG(info) 
         << " width " << width 
         << " height " << height 
         << " depth " << depth
         ;

    assert( width <= 1000000 ); 
    six->launch(width, height, depth ); 
#else
    CUdeviceptr d_param = (CUdeviceptr)Params::d_param ; ;
    assert( d_param && "must alloc and upload params before launch"); 
    CUstream stream;
    CUDA_CHECK( cudaStreamCreate( &stream ) );
    OPTIX_CHECK( optixLaunch( pip->pipeline, stream, d_param, sizeof( Params ), &(sbt->sbt), width, height, depth ) );
    CUDA_SYNC_CHECK();
#endif

    TP t1 = std::chrono::high_resolution_clock::now();
    DT _dt = t1 - t0;

    dt = _dt.count() ; 
    launch_times.push_back(dt);  

    LOG(LEVEL) 
          << " (width, height, depth) ( " << params->width << "," << params->height << "," << params->depth << ")"  
          << std::fixed << std::setw(7) << std::setprecision(4) << dt  
          ; 
    return dt ; 
}


/**
CSGOptiX::render CSGOptiX::simtrace CSGOptiX::simulate
---------------------------------------------------------

These three methods currently all call *CSGOptiX::launch* 
with params.raygenmode switch function inside OptiX7Test.cu:__raygen__rg 
As it is likely better to instead have multiple raygen entry points 
are retaining the distinct methods up here. 

*render* is also still needed to fulfil SRenderer protocol base 

The *simulate* and *simtrace*  are typically invoked from QSim::simulate 
and QSim::simtrace

**/
double CSGOptiX::render()
{  
    assert(raygenmode == SRG_RENDER) ; 
    return launch() ; 
}   
double CSGOptiX::simtrace()
{ 
    assert(raygenmode == SRG_SIMTRACE) ; 
    return launch() ; 
}  
double CSGOptiX::simulate()
{ 
    assert(raygenmode == SRG_SIMULATE) ; 
    return launch()  ;
}   

const CSGFoundry* CSGOptiX::getFoundry() const 
{
    return foundry ; 
}

std::string CSGOptiX::Annotation( double dt, const char* bot_line, const char* extra )  // static 
{
    std::stringstream ss ; 
    ss << std::fixed << std::setw(10) << std::setprecision(4) << dt ;
    if(bot_line) ss << std::setw(30) << " " << bot_line ; 
    if(extra) ss << " " << extra ; 

    std::string anno = ss.str(); 
    return anno ; 
}

const char* CSGOptiX::getDefaultSnapPath() const 
{
    assert( foundry );  
    const char* cfbase = foundry->getOriginCFBase(); 
    assert( cfbase ); 
    const char* path = SPath::Resolve(cfbase, "CSGOptiX/snap.jpg" , FILEPATH ); 
    return path ; 
}




void CSGOptiX::render_snap( const char* name_ )
{
    const char* name = name_ ? name_ : SStr::Format("cx%s", moi ) ; 

    double dt = render();  

    const char* topline = SSys::getenvvar("TOPLINE", SProc::ExecutableName() ); 
    const char* botline_ = SSys::getenvvar("BOTLINE", nullptr ); 
    const char* outpath = SEventConfig::OutPath(name, -1, ".jpg" );
    std::string bottom_line = CSGOptiX::Annotation(dt, botline_ ); 
    const char* botline = bottom_line.c_str() ; 

    LOG(error)  
          << " name " << name 
          << " outpath " << outpath 
          << " dt " << dt 
          << " topline [" <<  topline << "]"
          << " botline [" <<  botline << "]"
          ; 

    snap(outpath, botline, topline  );   
}




/**
CSGOptiX::snap : Download frame pixels and write to file as jpg.
------------------------------------------------------------------
**/

void CSGOptiX::snap(const char* path_, const char* bottom_line, const char* top_line, unsigned line_height)
{
    const char* path = path_ ? SPath::Resolve(path_, FILEPATH ) : getDefaultSnapPath() ; 
    LOG(info) << " path " << path ; 

#if OPTIX_VERSION < 70000
    const char* top_extra = nullptr ;
#else
    const char* top_extra = pip->desc(); 
#endif
    const char* topline = SStr::Concat(top_line, top_extra); 

    LOG(LEVEL) << " path_ [" << path_ << "]" ; 
    LOG(LEVEL) << " topline " << topline  ; 

    frame->download(); 
    frame->annotate( bottom_line, topline, line_height ); 
    frame->snap( path  );  

    if(!flight || SStr::Contains(path,"00000"))
    {
        saveMeta(path); 
    }
}

void CSGOptiX::writeFramePhoton(const char* dir, const char* name)
{
#if OPTIX_VERSION < 70000
    assert(0 && "not implemented pre-7"); 
#else
    frame->writePhoton(dir, name); 
#endif
}


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
    LOG(info) << json_path ; 
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
