#include <iostream>
#include <cstdlib>

#include <optix.h>
#if OPTIX_VERSION < 70000
#else
#include <optix_stubs.h>
#endif

#include <cuda_runtime.h>
#include <glm/glm.hpp>

#include "SStr.hh"
#include "SSys.hh"
#include "SMeta.hh"
#include "SPath.hh"
#include "SVec.hh"
#include "QBuf.hh"

#include "BTimeStamp.hh"
#include "PLOG.hh"
#include "Opticks.hh"
#include "Composition.hh"
#include "FlightPath.hh"

#include "scuda.h"
#include "squad.h"

#include "CSGPrim.h"
#include "CSGFoundry.h"

#include "CSGView.h"
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

// simulation
#include "QSim.hh"
#include "qsim.h"
#include "QSeed.hh"
#include "QEvent.hh"



const plog::Severity CSGOptiX::LEVEL = PLOG::EnvLevel("CSGOptiX", "INFO" ); 

#if OPTIX_VERSION < 70000 
const char* CSGOptiX::PTXNAME = "OptiX6Test" ; 
const char* CSGOptiX::GEO_PTXNAME = "geo_OptiX6Test" ; 

#else
const char* CSGOptiX::PTXNAME = "OptiX7Test" ; 
const char* CSGOptiX::GEO_PTXNAME = nullptr ; 
#endif

const char* CSGOptiX::ENV(const char* key, const char* fallback)
{
    const char* value = getenv(key) ; 
    return value ? value : fallback ; 
}

CSGOptiX::CSGOptiX(Opticks* ok_, const CSGFoundry* foundry_) 
    :
    ok(ok_),
    raygenmode(ok->getRaygenMode()),
    flight(ok->hasArg("--flightconfig")),
    composition(ok->getComposition()),
    foundry(foundry_),
    prefix(ENV("OPTICKS_PREFIX","/usr/local/opticks")),  // needed for finding ptx
    outdir(ok->getOutDir()),    // evar:OUTDIR default overridden by --outdir option   
    cmaketarget("CSGOptiX"),  
    ptxpath(SStr::PTXPath( prefix, cmaketarget, PTXNAME )),
#if OPTIX_VERSION < 70000 
    geoptxpath(SStr::PTXPath(prefix, cmaketarget, GEO_PTXNAME )),
#else
    geoptxpath(nullptr),
#endif
    tmin_model(SSys::getenvfloat("TMIN",0.1)), 
    jpg_quality(SStr::GetEValue<int>("QUALITY", 50)),
    params(new Params(raygenmode, ok->getWidth(), ok->getHeight(), ok->getDepth() )),
#if OPTIX_VERSION < 70000
    six(new Six(ok, ptxpath, geoptxpath, params)),
#else
    ctx(new Ctx),
    pip(new PIP(ptxpath)), 
    sbt(new SBT(ok, pip)),
    frame(new Frame(params->width, params->height, params->depth)),  // CUDA holds the pixels 
#endif
    meta(new SMeta),
    peta(new quad4), 
    metatran(nullptr),
    sim(raygenmode == 0 ? nullptr : new QSim<float>),
    evt(raygenmode == 0 ? nullptr : new QEvent)
{
    init(); 
}

void CSGOptiX::init()
{
    LOG(LEVEL) << "[" ; 
    assert( prefix && "expecting PREFIX envvar pointing to writable directory" );
    assert( outdir && "expecting OUTDIR envvar " );

    LOG(LEVEL) << " ptxpath " << ptxpath  ; 
    LOG(LEVEL) << " geoptxpath " << ( geoptxpath ? geoptxpath : "-" ) ; 

    initPeta(); 
    initParams(); 
    initGeometry();
    initRender(); 
    initSimulate(); 

    LOG(LEVEL) << "]" ; 
}


void CSGOptiX::initPeta()
{ 
    peta->zero(); 
    //unsigned* ptr = &(peta->q0.u.x) ;  
    //for(unsigned i=0 ; i < 16 ; i++ ) *(ptr + i) = 0u ; 
}

void CSGOptiX::setMetaTran(const Tran<double>* metatran_ )
{
    metatran = metatran_ ; 
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


void CSGOptiX::initRender()
{
#if OPTIX_VERSION < 70000
    six->initFrame();     // sets params->pixels, isect from optix device pointers
#else
    params->pixels = frame->getDevicePixel(); 
    params->isect  = frame->getDeviceIsect(); 
    params->fphoton = frame->getDevicePhoton(); 
#endif
}

void CSGOptiX::initSimulate() // once only (not per-event) simulate setup tasks .. 
{
    params->sim = sim ? sim->getDevicePtr() : nullptr ;  // qsim<float>*
    params->evt = evt ? evt->getDevicePtr() : nullptr ;  // qevent*
    params->tmin = 0.f ;      // perhaps needs to be epsilon to avoid self-intersection off boundaries ?
    params->tmax = 1000000.f ; 
}


void CSGOptiX::setGensteps(const NP* gs)
{
    assert( evt ); 
    evt->setGensteps(gs); 
}

void CSGOptiX::prepareSimulateParam()   // per-event simulate setup prior to optix launch 
{
    LOG(info) << "[" ; 
    params->num_photons = evt->getNumPhotons() ; 

    LOG(info) << "]" ; 
}


/**
CSGOptiX::setCE
------------------

Setting center_extent establishes the coordinate system. 

**/

void CSGOptiX::setCE(const float4& v )
{
    glm::vec4 ce(v.x, v.y, v.z, v.w); 
    setCE(ce); 
}

void CSGOptiX::setCEGS(const std::vector<int>& cegs)
{
    //params->setCEGS(cegs_); 
    assert( cegs.size() == 7 );   // use QEvent::StandardizeCEGS to convert 4 to 7  

    peta->q0.i.x = cegs[0] ; 
    peta->q0.i.y = cegs[1] ; 
    peta->q0.i.z = cegs[2] ; 
    peta->q0.i.w = cegs[3] ; 

    peta->q1.i.x = cegs[4] ; 
    peta->q1.i.y = cegs[5] ; 
    peta->q1.i.z = cegs[6] ; 
    peta->q1.i.w = 0 ; 
}

void CSGOptiX::setCE(const glm::vec4& ce )
{
    peta->q2.f.x = ce.x ;   // moved from q1
    peta->q2.f.y = ce.y ; 
    peta->q2.f.z = ce.z ; 
    peta->q2.f.w = ce.w ; 

    bool aim = true ; 
    composition->setCenterExtent(ce, aim);  // model2world view setup 

    float extent = ce.w ; 
    float tmin = extent*tmin_model ; 
    LOG(info) 
        << " ce [ " << ce.x << " " << ce.y << " " << ce.z << " " << ce.w << "]" 
        << " tmin_model " << tmin_model
        << " tmin " << tmin 
        ; 

    composition->setNear(tmin); 
}

void CSGOptiX::setNear(float near)
{
    composition->setNear(near); 
}



void CSGOptiX::prepareRenderParam()
{
    float extent = composition->getExtent(); 

    glm::vec3 eye ;
    glm::vec3 U ; 
    glm::vec3 V ; 
    glm::vec3 W ; 
    glm::vec4 ZProj ;

    composition->getEyeUVW(eye, U, V, W, ZProj); // must setModelToWorld in composition first

    float tmin = composition->getNear(); 
    float tmax = composition->getFar(); 
    unsigned cameratype = composition->getCameraType(); 

    if(!flight) LOG(info)
        << " extent " << extent
        << " tmin " << tmin 
        << " tmax " << tmax 
        << " eye (" << eye.x << " " << eye.y << " " << eye.z << " ) "
        << " U (" << U.x << " " << U.y << " " << U.z << " ) "
        << " V (" << V.x << " " << V.y << " " << V.z << " ) "
        << " W (" << W.x << " " << W.y << " " << W.z << " ) "
        << " cameratype " << cameratype
        ;

    params->setView(eye, U, V, W);
    params->setCamera(tmin, tmax, cameratype ); 

    if(!flight) LOG(info)
        << "composition.desc " << std::endl 
        << composition->desc() 
        ; 
}

void CSGOptiX::prepareParam()
{
    glm::vec4& ce = composition->getCenterExtent();   
    params->setCenterExtent(ce.x, ce.y, ce.z, ce.w); 

    switch(raygenmode)
    {
       case 0:prepareRenderParam() ; break ; 
       case 1:prepareSimulateParam() ; break ; 
    }

#if OPTIX_VERSION < 70000
    six->updateContext(); 
#else
    params->upload();  
    params->dump(" CSGOptiX::prepareParam"); 
#endif
}

double CSGOptiX::launch(unsigned width, unsigned height, unsigned depth)
{
    LOG(LEVEL) << "[" ; 
    double t0, t1 ; 
    t0 = BTimeStamp::RealTime();
#if OPTIX_VERSION < 70000
    // hmm width, heigth, deth not used pre-7 ?
    six->launch(); 
#else
    CUdeviceptr d_param = (CUdeviceptr)Params::d_param ; ;
    assert( d_param && "must alloc and upload params before launch"); 
    CUstream stream;
    CUDA_CHECK( cudaStreamCreate( &stream ) );
    OPTIX_CHECK( optixLaunch( pip->pipeline, stream, d_param, sizeof( Params ), &(sbt->sbt), width, height, depth ) );
    CUDA_SYNC_CHECK();
#endif
    t1 = BTimeStamp::RealTime();
    double dt = t1 - t0 ; 
    launch_times.push_back(dt);  
    LOG(LEVEL) << "] " << std::fixed << std::setw(7) << std::setprecision(4) << dt  ; 
    return dt ; 
}

double CSGOptiX::render()
{
    prepareParam(); 
    assert( raygenmode == 0 ); 
    double dt = launch(params->width, params->height, params->depth );
    return dt ; 
}

double CSGOptiX::simulate()
{
    prepareParam(); 
    assert( raygenmode > 0 ); 
    unsigned num_photons = params->num_photons ; 
    assert( num_photons > 0 ); 
    double dt = launch(num_photons, 1u, 1u );
    return dt ; 
}

std::string CSGOptiX::Annotation( double dt, const char* bot_line )  // static 
{
    std::stringstream ss ; 
    ss << std::fixed << std::setw(10) << std::setprecision(4) << dt ;
    if(bot_line) ss << std::setw(30) << " " << bot_line ; 
    std::string anno = ss.str(); 
    return anno ; 
}


/**
CSGOptiX::snap : Download frame pixels and write to file as jpg.
------------------------------------------------------------------
**/

void CSGOptiX::snap(const char* path_, const char* bottom_line, const char* top_line, unsigned line_height)
{
    int create_dirs = 1 ; // 1:filepath 
    const char* path = SPath::Resolve(path_, create_dirs ); 

#if OPTIX_VERSION < 70000
    six->snap(path, bottom_line, top_line, line_height); 
#else
    frame->download(); 
    frame->annotate( bottom_line, top_line, line_height ); 
    frame->writeJPG(path, jpg_quality);  
#endif
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
    FlightPath* fp = ok->getFlightPath();   // FlightPath lazily instanciated here (held by Opticks)
    int rc = fp->render( (SRenderer*)this  );
    return rc ; 
}

void CSGOptiX::saveMeta(const char* jpg_path) const
{
    const char* json_path = SStr::ReplaceEnd(jpg_path, ".jpg", ".json"); 
    //const char* npy_path = SStr::ReplaceEnd(jpg_path, ".jpg", ".npy"); 

    nlohmann::json& js = meta->js ;
    js["argline"] = ok->getArgLine();
    js["nameprefix"] = ok->getNamePrefix() ;
    js["jpg"] = jpg_path ; 
    js["emm"] = ok->getEnabledMergedMesh() ;

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

void CSGOptiX::savePeta(const char* fold, const char* name) const
{
    int create_dirs = 1 ; // 1:filepath
    const char* path = SPath::Resolve(fold, name, create_dirs) ; 
    LOG(info) << path ; 
    NP::Write(path, (float*)(&peta->q0.f.x), 1, 4, 4 );
}

void CSGOptiX::saveMetaTran(const char* fold, const char* name) const
{
    if(metatran == nullptr) return ; 

    int create_dirs = 1 ; // 1:filepath
    const char* path = SPath::Resolve(fold, name, create_dirs) ; 
    LOG(info) << path ; 

    NP* mta = NP::Make<double>(3, 4, 4 );  
    metatran->write( mta->values<double>() ) ; 
    mta->save(fold, name);  
}





