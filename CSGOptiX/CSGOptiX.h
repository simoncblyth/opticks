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
struct quad6 ; 
struct qat4 ; 
struct float4 ; 
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

#ifdef WITH_SGLM
#else
    Opticks*          ok ;  
    Composition*      composition ; 
#endif
    SGLM*             sglm ; 

    const char*       moi ; 
    const char*       flight ; 
    const CSGFoundry* foundry ; 
    const char*       prefix ; 
    const char*       outdir ; 
    const char*       cmaketarget ; 
    const char*       ptxpath ; 
    const char*       geoptxpath ; 
    float             tmin_model ; 
    int               jpg_quality ; 

    std::vector<unsigned>  solid_selection ;
    std::vector<double>  launch_times ;

    int               raygenmode ; 
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


    QSim*        sim ; 
    QEvent*      event ;  
 
    const char* desc() const ; 

private:
    static void InitGeo(  CSGFoundry* fd ); 
    static void InitSim( const SSim* ssim ); 
public:
    static CSGOptiX* Create(CSGFoundry* foundry ); 

    CSGOptiX(const CSGFoundry* foundry ); 

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
    void uploadGenstep();

    void setCEGS(const std::vector<int>& cegs); 

    void setComposition();
    void setComposition(const char* moi);
    void setComposition(const float4& ce,    const qat4* m2w=nullptr, const qat4* w2m=nullptr ); 
    void setComposition(const glm::vec4& ce, const qat4* m2w=nullptr, const qat4* w2m=nullptr ); 

    void prepareRenderParam(); 
    void prepareSimulateParam(); 
    void prepareParam(); 

 private: 
    double launch(); 
 public: 
    void render_snap(const char* name=nullptr); 
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

    void snapSimtraceTest() const ;

    static int   _OPTIX_VERSION() ; 
};

