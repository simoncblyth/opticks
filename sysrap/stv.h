#pragma once
/**
stv.h : simple wrapper to give uniform behaviour to spa/sxf/sbb
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

struct _stv
{
    static constexpr const int NV = 32 ; 
    glm::tmat4x4<double> t ; 
    glm::tmat4x4<double> v ;
};

struct stv
{
    typedef s_pool<stv,_stv> POOL ;
    static POOL pool ;
    static constexpr const char* NAME = "stv.npy" ; 
    static void Serialize( _stv& p, const stv* o ); 
    static stv* Import(  const _stv* p, const std::vector<_stv>& buf ); 
    
    stv(); 
    ~stv(); 

    int pid ; 
    glm::tmat4x4<double> t ; 
    glm::tmat4x4<double> v ;
 
    bool is_root_importable() const ; 
    std::string desc() const ;  
}; 

inline void stv::Serialize( _stv& p, const stv* o )
{
    p.t = o->t ; 
    p.v = o->v ; 
} 
inline stv* stv::Import( const _stv* p, const std::vector<_stv>& )
{
    stv* o = new stv ; 
    o->t = p->t ; 
    o->v = p->v ; 
    return o ; 
}


inline stv::stv()
    :
    pid(pool.add(this)),
    t(1.),
    v(1.)
{
}
inline stv::~stv()
{
    pool.remove(this); 
}

inline bool stv::is_root_importable() const 
{
    return true ; 
}


inline std::string stv::desc() const 
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


