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
struct CSGView ; 

template <typename T> struct Tran ; 
struct QSim ; 
struct QEvent ; 

struct Params ; 
class Opticks ; 
class Composition ; 

#if OPTIX_VERSION < 70000
struct Six ; 
struct Dummy ; 
#else
struct Ctx ; 
struct PIP ; 
struct SBT ; 
#endif
struct Frame ; 

#include "SCSGOptiX.h"

struct CSGOPTIX_API CSGOptiX : public SCSGOptiX
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
    static int         Main();          // NOT USED


    static const char* Desc(); 

    SGLM*             sglm ; 

    const char*       flight ; 
    const CSGFoundry* foundry ; 
    const char*       outdir ; 
    const char*       ptxpath ; 
    const char*       geoptxpath ; 
    float             tmin_model ; 
    plog::Severity    level = plog::debug ;   // quell prolific logging using level instead of LEVEL

    std::vector<unsigned>  solid_selection ;
    std::vector<double>  launch_times ;
    int                  launch_count ; 

    int               raygenmode ; 
    Params*           params  ; 


#if OPTIX_VERSION < 70000
    Six* six ;  
    Dummy* dummy0 ; 
    Dummy* dummy1 ; 
#else
    Ctx* ctx ; 
    PIP* pip ; 
    SBT* sbt ; 
#endif

    Frame* framebuf ; 
    SMeta* meta ; 
    double launch_dt ;   // of prior launch  

    scontext*    sctx ; 
    QSim*        sim ; 
    QEvent*      event ;  

    const char* desc() const ; 

private:
    static Params* InitParams( int raygenmode, const SGLM* sglm  ) ; 
    static void InitEvt(  CSGFoundry* fd  ); 
    static void InitMeta( const SSim* ssim ); 
    static void InitSim(  SSim* ssim ); 
    static void InitGeo(  CSGFoundry* fd ); 

public:
    static CSGOptiX* Create(CSGFoundry* foundry ); 


    virtual ~CSGOptiX(); 
    CSGOptiX(const CSGFoundry* foundry ); 

private:
    void init(); 
    void initCtx(); 
    void initPIP(); 
    void initSBT(); 
    void initCheckSim(); 
    void initStack(); 
    void initParams();
    void initGeometry();
    void initSimulate();
    void initFrame(); 
    void initRender();
    void initPIDXYZ();
 public: 
    void setExternalDevicePixels(uchar4* _d_pixel ); 
    void destroy(); 
 public: 

    void setFrame(); 
    void setFrame(const char* moi);
    void setFrame(const float4& ce); 
    void setFrame(const sfr& fr_); 


    static constexpr const char* _prepareParamRender_DEBUG = "CSGOptiX__prepareParamRender_DEBUG" ; 
    void prepareParamRender(); 
    void prepareParamSimulate(); 
    void prepareParam(); 

 private: 
    double launch(); 
 private: 
    const char* getRenderStemDefault() const ; 
 public: 
    double render(const char* stem_=nullptr);   
    void   render_save(const char* stem_=nullptr); 
    void   render_save_inverted(const char* stem_=nullptr); 
    void   render_save_(const char* stem_=nullptr, bool inverted=false); 

    double simtrace(int eventID); 
    double simulate(int eventID); 
    double proceed(); 


    // these launch  methods fulfil SCSGOptix protocal base
    // the latter two get invoked from QSim::simtrace QSim::simulate following genstep uploading   
 public:
    double render_launch();   
 private: 
    double simtrace_launch(); 
    double simulate_launch();    

 public: 
    const CSGFoundry* getFoundry() const ; 
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

