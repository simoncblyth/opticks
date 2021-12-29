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
    static const char* PTXNAME ; 
    static const char* GEO_PTXNAME ; 
    static const char* ENV(const char* key, const char* fallback);
    static int   _OPTIX_VERSION() ; 

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
    Frame* frame ; 
#endif
    SMeta* meta ; 
    quad4* peta ; 
    const Tran<double>* metatran ; 
    double simulate_dt ; 


    QSim<float>* sim ; 
    QEvent*      evt ;  


    CSGOptiX(Opticks* ok, const CSGFoundry* foundry ); 

    void init(); 
    void initPeta();
    void initParams();
    void initGeometry();
    void initRender();
    void initSimulate();
 
    void setTop(const char* tspec); 

    // render related 
    void setCEGS(const std::vector<int>& cegs); 

    void setCE(const float4& ce); 
    void setCE(const glm::vec4& ce); 
    void setMetaTran(const Tran<double>* metatran ); 

    void setNear(float near); 

    void prepareRenderParam(); 
    void prepareSimulateParam(); 
    void prepareParam(); 

    int  render_flightpath(); 
    void saveMeta(const char* jpg_path) const ;
    void savePeta(const char* fold, const char* name) const ; 
    void saveMetaTran(const char* fold, const char* name) const ; 

    static std::string Annotation( double dt, const char* bot_line ); 

    // [ fulfil SRenderer protocol base
    double render();    
    void snap(const char* path, const char* bottom_line, const char* top_line=nullptr, unsigned line_height=24); 
    // ]

    void writeFramePhoton(const char* dir, const char* name);

    void setGensteps(const NP* gs);
    double simulate();    
    double launch(unsigned width, unsigned height, unsigned depth) ; 

    void snapSimulateTest(const char* outdir, const char* botline, const char* topline) ; // uses snap, so not const 



};

