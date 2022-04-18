#pragma once

#include <vector>
#include <optix.h>
#include <string>
#include <glm/fwd.hpp>
#include "plog/Severity.h"

#include "CSGOPTIX_API_EXPORT.hh"

struct SMeta ; 
struct NP ; 
struct quad4 ; 
struct qat4 ; 
struct float4 ; 

struct CSGFoundry ; 
struct CSGView ; 

template <typename T> struct QSim ; 
template <typename T> struct Tran ; 
struct QEvent ; 

struct Params ; 
class Opticks ; 
class Composition ; 

#if OPTIX_VERSION < 70000
struct Six ; 
#else
struct Ctx ; 
struct PIP ; 
struct SBT ; 
#endif
struct Frame ; 

#include "SRenderer.hh"

struct CSGOPTIX_API CSGOptiX : public SRenderer 
{
    static const plog::Severity LEVEL ; 
    static const char* TOP ; 
    static const char* PTXNAME ; 
    static const char* GEO_PTXNAME ; 
    static const char* ENV(const char* key, const char* fallback);

    Opticks*          ok ;  
    int               raygenmode ; 
    bool              flight ; 
    Composition*      composition ; 
 
    const CSGFoundry* foundry ; 
    const char*       prefix ; 
    const char*       outdir ; 
    const char*       cmaketarget ; 
    const char*       ptxpath ; 
    const char*       geoptxpath ; 
    float             tmin_model ; 
    int               jpg_quality ; 

    std::vector<double>  launch_times ;

    Params*           params  ; 
#if OPTIX_VERSION < 70000
    Six* six ;  
#else
    Ctx* ctx ; 
    PIP* pip ; 
    SBT* sbt ; 
#endif
    Frame* frame ; 
    SMeta* meta ; 
    quad4* peta ; 
    const Tran<double>* metatran ; 
    double dt ; 


    QSim<float>* sim ; 
    QEvent*      event ;  
 
    const char* desc() const ; 
    CSGOptiX(Opticks* ok, const CSGFoundry* foundry ); 

    void init(); 
    void initStack(); 
    void initPeta();
    void initParams();
    void initGeometry();
    void initRender();
    void initSimulate();

    static const char* Top() ; 
 private: 
    void setTop(const char* tspec); 
 public: 
    void setGensteps(const NP* gs);

    void setCEGS(const std::vector<int>& cegs); 
    void setComposition(const float4& ce,    const qat4* m2w, const qat4* w2m ); 
    void setComposition(const glm::vec4& ce, const qat4* m2w, const qat4* w2m ); 
    void setNear(float near); 

    void prepareRenderParam(); 
    void prepareSimulateParam(); 
    void prepareParam(); 

 private: 
    double launch(); 
 public: 
    double render();     // part of SRenderer protocol base
    double simtrace(); 
    double simulate();    
 public: 
    static std::string Annotation( double dt, const char* bot_line, const char* extra=nullptr ); 
    const char* getDefaultSnapPath() const ; 
    void snap(const char* path=nullptr, const char* bottom_line=nullptr, const char* top_line=nullptr, unsigned line_height=24);  // part of SRenderer protocol base

    void writeFramePhoton(const char* dir, const char* name);
    int  render_flightpath(); 

    void saveMeta(const char* jpg_path) const ;
    void savePeta(const char* fold, const char* name) const ; 
    void setMetaTran(const Tran<double>* metatran ); 
    void saveMetaTran(const char* fold, const char* name) const ; 

    void snapSimtraceTest(const char* outdir, const char* botline, const char* topline) ; // uses snap, so not const 

    static int   _OPTIX_VERSION() ; 
};

