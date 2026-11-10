#pragma once
/**
CSGOptiX.h
===========



**/

#include <vector>
#include <string>
#include <glm/fwd.hpp>

#include <optix.h>

#include "plog/Severity.h"
#include "sfr.h"

#include "CSGOPTIX_API_EXPORT.hh"

struct SMeta ;
struct NP ;
struct quad4 ;
struct quad6 ;
struct qat4 ;
struct float4 ;
struct uchar4 ;

struct scontext ;
struct SGLM ;
struct SSim ;

struct CSGFoundry ;
struct SFoundry ;
struct CSGView ;

template <typename T> struct Tran ;
struct QSim ;
struct QEvt ;

struct Params ;
struct Params_Helper ;

class Opticks ;
class Composition ;

struct Ctx ;
struct PIP ;
struct SBT ;
struct Frame ;


#include "SSimulator.h"

struct CSGOPTIX_API CSGOptiX : public SSimulator
{
    friend struct QSim ;

    static constexpr const char* RELDIR = "CSGOptiX" ;
    static const plog::Severity LEVEL ;
    static CSGOptiX*   INSTANCE ;
    static CSGOptiX*   Get();
    static int         Version();

    static int         RenderMain();    // used by tests/CSGOptiXRdrTest.cc
    static int         SimtraceMain();  // used by tests/CSGOptiXTMTest.cc
    static int         SimulateMain();  // used by tests/CSGOptiXSMTest.cc


    static const char* Desc();

    SGLM*             sglm ;

    const char*       flight ;
    const SFoundry*   foundry ;
    const char*       outdir ;
    const char*       optixsuffix ;    // .ptx OR .optixir
    const char*       _optixpath ;
    const char*       optixpath ;
    float             tmin_model ;
    plog::Severity    level = plog::debug ;   // quell prolific logging using level instead of LEVEL

    std::vector<unsigned>  solid_selection ;
    std::vector<double>    kernel_times ;
    std::vector<int64_t>   kernel_times_ ;
    size_t                 kernel_count ;

    int               raygenmode ;
    Params_Helper*    params_helper  ;


    Ctx* ctx ;
    PIP* pip ;
    SBT* sbt ;

    Frame* framebuf ;
    SMeta* meta ;
    double kernel_dt ;   // execution time of prior kernel
    scontext*    sctx ;
    QSim*        sim ;
    QEvt*        qev ;

    const char* desc() const ;

private:
    static Params_Helper* InitParamsHelper( int raygenmode, const SGLM* sglm  ) ;
    static void InitMeta( SSim* ssim );
    static void InitSim(  SSim* ssim );
    static void InitGeo(  SFoundry* fd );

public:
    static CSGOptiX* Create(SFoundry* foundry );


    virtual ~CSGOptiX();
    CSGOptiX(const SFoundry* foundry );

private:
    void init();
    void initMeta();
    void initCtx();
    void initPIP();
    void initSBT();
    void initCheckSim();
    void initStack();
    void initParams();
    void initGLM();
    void initGeometry();
    void initSimulate();
    void initRender();
    void initPIDXYZ();
 public:
    void setExternalDevicePixels(uchar4* _d_pixel );
    void destroy();
 public:


#ifdef OBSOLETE_CSGOPTIX_SET_FRAME
    void setFrame();
    void setFrame(const char* moi);
    void setFrame(const float4& ce);
    void setFrame(const sfr& fr_);
#endif


    static constexpr const char* _prepareParamRender_DEBUG = "CSGOptiX__prepareParamRender_DEBUG" ;
    static constexpr const char* _prepareParamRender_DEBUG_zdepth = "CSGOptiX__prepareParamRender_DEBUG_zdepth" ;

    void prepareParamRender(size_t idx);
    void prepareParamSimulate(size_t idx);
    void prepareParamSimulate_SER(size_t idx);
    void prepareParam(size_t idx);

 private:
    double launch();
 private:
    const char* getRenderStemDefault() const ;
 public:

    bool handle_snap(int wanted_snap);

    double render(const char* stem_=nullptr);
    void   render_save(const char* stem_=nullptr);
    void   render_save_inverted(const char* stem_=nullptr);
    void   render_save_(const char* stem_=nullptr, bool inverted=false);

    double simtrace(int eventID);
    double simulate(int eventID, bool reset);
    void   reset(int eventID);

    NP* simulate(const NP* gs, int eventID);



    // these launch  methods fulfil SCSGOptix protocal base
    // the latter two get invoked from QSim::simtrace QSim::simulate following genstep uploading
 public:
    double render_launch();
 private:
    double simtrace_launch();
    double simulate_launch();

 public:
    const SFoundry* getFoundry() const ;
    static std::string AnnotationTime( double dt, const char* extra=nullptr );
    static std::string Annotation( double dt, const char* bot_line, const char* extra=nullptr );
    const char* getDefaultSnapPath() const ;
    //void snap(const char* path=nullptr, const char* bottom_line=nullptr, const char* top_line=nullptr, unsigned line_height=24, bool inverted=false );  // part of SRenderer protocol base
    void snap(const char* path, const char* bottom_line, const char* top_line, unsigned line_height, bool inverted );  // part of SRenderer protocol base


#ifdef WITH_FRAME_PHOTON
    void writeFramePhoton(const char* dir, const char* name);
#endif
    int  render_flightpath();

    void saveMeta(const char* jpg_path) const ;

    static constexpr const char* CTX_LOGNAME = "CSGOptiX__Ctx.log"  ;
    void write_Ctx_log(const char* dir=nullptr) const ;

    static int   _OPTIX_VERSION() ;
};

