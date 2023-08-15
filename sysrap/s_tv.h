#pragma once
/**
s_tv.h : simple wrapper to give uniform behaviour to spa/sxf/sbb
===================================================================

**/

#include <string>
#include <sstream>
#include <iomanip>

#include <glm/glm.hpp>
#include "glm/gtx/string_cast.hpp"
#include <glm/gtx/transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "s_pool.h"

struct _s_tv
{
    static constexpr const int NV = 32 ; 
    glm::tmat4x4<double> t ; 
    glm::tmat4x4<double> v ;
};

#include "SYSRAP_API_EXPORT.hh"

struct SYSRAP_API s_tv
{
    typedef s_pool<s_tv,_s_tv> POOL ;
    static POOL* pool ;
    static constexpr const bool LEAK = false ; 
    static void SetPOOL( POOL* pool_ ); 
    static int level() ; 

    static constexpr const char* NAME = "s_tv.npy" ; 
    static void Serialize( _s_tv& p, const s_tv* o ); 
    static s_tv* Import(  const _s_tv* p, const std::vector<_s_tv>& buf ); 
    
    s_tv(); 
    ~s_tv(); 

    int pid ; 
    glm::tmat4x4<double> t ; 
    glm::tmat4x4<double> v ;
 
    bool is_root_importable() const ; 
    std::string desc() const ;  
}; 


inline int s_tv::level() {  return pool ? pool->level : ssys::getenvint("sn__level",-1) ; } // static 

inline void s_tv::Serialize( _s_tv& p, const s_tv* o )
{
    p.t = o->t ; 
    p.v = o->v ; 
} 
inline s_tv* s_tv::Import( const _s_tv* p, const std::vector<_s_tv>& )
{
    s_tv* o = new s_tv ; 
    o->t = p->t ; 
    o->v = p->v ; 
    return o ; 
}


inline s_tv::s_tv()
    :
    pid(pool ? pool->add(this) : -1),
    t(1.),
    v(1.)
{
    if(level() > 1) std::cerr << "s_tv::s_tv pid " << pid << std::endl ; 
}
inline s_tv::~s_tv()
{
    if(level() > 1) std::cerr << "s_tv::~s_tv pid " << pid << std::endl ; 
    if(pool) pool->remove(this); 
}

inline bool s_tv::is_root_importable() const 
{
    return true ; 
}

inline std::string s_tv::desc() const 
{
    std::stringstream ss ;
    ss 
        << "t " << glm::to_string(t) 
        << std::endl 
        << "v " << glm::to_string(v) 
        << std::endl 
        ;
        
    std::string str = ss.str(); 
    return str ; 
}


