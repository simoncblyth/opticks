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



const plog::Severity CSGOptiX::LEVEL = PLOG::EnvLevel("CSGOptiX", "DEBUG" ); 

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
    simulate_dt(0.),
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
void CSGOptiX::initRender()
{
#if OPTIX_VERSION < 70000
    six->initFrame();     
    // sets params->pixels, isect from optix device pointers
    // HUH:instanciating Six does this already  
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

void CSGOptiX::setMetaTran(const Tran<double>* metatran_ )
{
    metatran = metatran_ ; 
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


void CSGOptiX::setCEGS(const std::vector<int>& cegs)
{
    assert( cegs.size() == 7 );   // use QEvent::StandardizeCEGS to convert 4 to 7  

    peta->q0.i.x = cegs[0] ;  // ix0
    peta->q0.i.y = cegs[1] ;  // ix1
    peta->q0.i.z = cegs[2] ;  // iy0 
    peta->q0.i.w = cegs[3] ;  // iy1

    peta->q1.i.x = cegs[4] ;  // iz0
    peta->q1.i.y = cegs[5] ;  // iz1 
    peta->q1.i.z = cegs[6] ;  // num_photons
    peta->q1.i.w = 0 ;     // TODO: gridscale according to ana/gridspec.py 
}

/**
CSGOptiX::setCE
------------------

Setting CE center-extent establishes the coordinate system
via calls to Composition::setCenterExtent which results in the 
definition of a model2world 4x4 matrix which becomes the frame of 
reference used by the EYE LOOK UP navigation controls.  

**/

void CSGOptiX::setComposition(const float4& v, const qat4* m2w, const qat4* w2m )
{
    glm::vec4 ce(v.x, v.y, v.z, v.w); 
    setComposition(ce, m2w, w2m ); 
}

/**
CSGOptiX::setComposition
-------------------------

**/

void CSGOptiX::setComposition(const glm::vec4& ce, const qat4* m2w, const qat4* w2m )
{
    peta->q2.f.x = ce.x ;   // moved from q1
    peta->q2.f.y = ce.y ; 
    peta->q2.f.z = ce.z ; 
    peta->q2.f.w = ce.w ; 

    bool autocam = true ; 

    composition->setCenterExtent(ce, autocam, m2w, w2m );  // model2world view setup 

    float extent = ce.w ; 
    float tmin = extent*tmin_model ;   // tmin_model from TMIN envvar with default of 0.1 (units of extent) 

    LOG(info) 
        << " ce [ " << ce.x << " " << ce.y << " " << ce.z << " " << ce.w << "]" 
        << " tmin_model " << tmin_model
        << " tmin " << tmin 
        << " m2w " << m2w
        << " w2m " << w2m
        ; 

    if(m2w) LOG(info) << "m2w " << *m2w ; 
    if(w2m) LOG(info) << "w2m " << *w2m ; 

    composition->setNear(tmin); 
}


/**
CSGOptiX::setNear
-------------------

TODO: not getting what is set eg 0.1., investigate 
**/

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
    if(!flight) params->dump(" CSGOptiX::prepareParam"); 
#endif
}

/**
CSGOptiX::launch
-------------------

For what happens next, see 

OptiX7Test.cu::__raygen__rg

OptiX6Test.cu::raygen


Depending on params.raygenmode the "render" or "simulate" method is called. 
 
**/

double CSGOptiX::launch(unsigned width, unsigned height, unsigned depth)
{
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

    LOG(LEVEL) 
          << " (width, height, depth) ( " << width << "," << height << "," << depth << ")"  
          << std::fixed << std::setw(7) << std::setprecision(4) << dt  
          ; 
    return dt ; 
}

/**
Huh GEOM=Hama_16 ./cxs.sh  with only 14.8M photons causing an exception::

    2021-12-20 19:49:22.143 INFO  [158578] [CSGOptiX::launch@330] [ (width, height, depth) ( 14825700,1,1)
    terminate called after throwing an instance of 'sutil::CUDA_Exception'
      what():  CUDA error on synchronize with error 'an illegal memory access was encountered' (/data/blyth/junotop/opticks/CSGOptiX/CSGOptiX.cc:342)

    ./cxs.sh: line 230: 158578 Aborted                 (core dumped) $GDB CSGOptiXSimulateTest


094 elif [ "$GEOM" == "Hama_16" ]; then
 95     
 96     ##  CUDA error on synchronize with error 'an illegal memory access was encountered' (/data/blyth/junotop/opticks/CSGOptiX/CSGOptiX.cc:342)
 97     moi=Hama
 98     cegs=256:0:144:100
 99     gridscale=0.20
100     gsplot=0
101 


In [1]: (256*2+1)*(144*2+1)*100
Out[1]: 14825700

In [2]: (256*2+1)*(144*2+1)*100


**/


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

    if( raygenmode == 0 )
    {
        LOG(fatal) << " WRONG EXECUTABLE FOR CSGOptiX::render cx.raygenmode " << raygenmode ; 
        assert(0); 
    }

    unsigned num_photons = params->num_photons ; 
    assert( num_photons > 0 ); 
    simulate_dt = launch(num_photons, 1u, 1u );
    LOG(info) << " simulate_dt " << simulate_dt ;
    return simulate_dt ; 
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

    LOG(LEVEL) << " path_ [" << path_ << "]"  ; 

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

/**
CSGOptiX::snapSimulateTest
---------------------------

Saving data for 2D cross sections, used by tests/CSGOptiXSimulateTest.cc 


**/

void CSGOptiX::snapSimulateTest(const char* outdir, const char* botline, const char* topline) 
{
    evt->setMeta( foundry->meta ); 
    evt->savePhoton( outdir, "photons.npy");   // this one can get very big 
    evt->saveGenstep(outdir, "genstep.npy");  
    evt->saveMeta(   outdir, "fdmeta.txt" ); 

    const char* namestem = "CSGOptiXSimulateTest" ; 
    const char* ext = ".jpg" ; 
    int index = -1 ;  
    const char* outpath = ok->getOutPath(namestem, ext, index ); 
    LOG(error) << " outpath " << outpath ; 

    std::string bottom_line = CSGOptiX::Annotation( simulate_dt, botline ); 
    snap(outpath, bottom_line.c_str(), topline  );   

    writeFramePhoton(outdir, "fphoton.npy" );   // as only 1 possible frame photon per-pixel the size never gets very big 
    savePeta(        outdir, "peta.npy");   
    saveMetaTran(    outdir, "metatran.npy"); 
}



/**
CSGOptiX::_OPTIX_VERSION
-------------------------

This depends on the the optix.h header only which provides the OPTIX_VERSION macro
so it could be done at the lowest level, no need for it to be 
up at this "elevation"

**/

#define xstr(s) str(s)
#define str(s) #s

int CSGOptiX::_OPTIX_VERSION()   // static 
{
    char vers[16] ; 
    snprintf(vers, 16, "%s",xstr(OPTIX_VERSION)); 
    return std::atoi(vers) ;  
}
