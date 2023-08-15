#pragma once
/**
s_bb.h
========

*/

#include <string>
#include <sstream>
#include <iomanip>

#include "s_pool.h"

struct _s_bb
{
    static constexpr const int NV = 6 ; 

    double x0 ; 
    double y0 ; 
    double z0 ; 
    double x1 ; 
    double y1 ; 
    double z1 ; 

};

#include "SYSRAP_API_EXPORT.hh"

struct SYSRAP_API s_bb
{
    typedef s_pool<s_bb,_s_bb> POOL ;
    static POOL* pool ;
    static constexpr const bool LEAK = false ; 
    static void SetPOOL( POOL* pool_ ); 
    static int level() ; 

    static constexpr const char* NAME = "s_bb.npy" ; 
    static void Serialize( _s_bb& p, const s_bb* o ); 
    static s_bb* Import(  const _s_bb* p, const std::vector<_s_bb>& buf ); 
    
    s_bb(); 
    ~s_bb(); 

    int pid ; 

    double x0 ; 
    double y0 ; 
    double z0 ; 
    double x1 ; 
    double y1 ; 
    double z1 ; 
 
    bool is_root_importable() const ; 
    std::string desc() const ;  
}; 


inline int s_bb::level() {  return pool ? pool->level : ssys::getenvint("sn__level",-1) ; } // static 

inline void s_bb::Serialize( _s_bb& p, const s_bb* o )
{
    p.x0 = o->x0 ; 
    p.y0 = o->y0 ; 
    p.z0 = o->z0 ; 
    p.x1 = o->x1 ; 
    p.y1 = o->y1 ; 
    p.z1 = o->z1 ; 
} 
inline s_bb* s_bb::Import( const _s_bb* p, const std::vector<_s_bb>& )
{
    s_bb* o = new s_bb ; 
    o->x0 = p->x0 ; 
    o->y0 = p->y0 ; 
    o->z0 = p->z0 ; 
    o->x1 = p->x1 ; 
    o->y1 = p->y1 ; 
    o->z1 = p->z1 ; 
    return o ; 
}


inline s_bb::s_bb()
    :
    pid(pool ? pool->add(this) : -1),
    x0(0.),
    y0(0.),
    z0(0.),
    x1(0.),
    y1(0.),
    z1(0.)
{
    if(level() > 1) std::cerr << "s_bb::s_bb pid " << pid << std::endl ; 
}
inline s_bb::~s_bb()
{
    if(level() > 1) std::cerr << "s_bb::~s_bb pid " << pid << std::endl ; 
    if(pool) pool->remove(this); 
}

inline bool s_bb::is_root_importable() const 
{
    return true ; 
}

inline std::string s_bb::desc() const 
{
    std::stringstream ss ;
    ss 
       << " x0 " << std::setw(10) << std::fixed << std::setprecision(3) << x0
       << " y0 " << std::setw(10) << std::fixed << std::setprecision(3) << y0
       << " z0 " << std::setw(10) << std::fixed << std::setprecision(3) << z0 
       << " x1 " << std::setw(10) << std::fixed << std::setprecision(3) << x1
       << " y1 " << std::setw(10) << std::fixed << std::setprecision(3) << y1 
       << " z1 " << std::setw(10) << std::fixed << std::setprecision(3) << z1 
       ;
        
    std::string str = ss.str(); 
    return str ; 
}


