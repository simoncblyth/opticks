#pragma once
/**
s_pa.h
========

*/

#include <string>
#include <sstream>
#include <iomanip>

#include "s_pool.h"

struct _s_pa
{
    static constexpr const int NV = 6 ; 

    double x0, y0, z0, w0, x1, y1 ;



};

#include "SYSRAP_API_EXPORT.hh"

struct SYSRAP_API s_pa
{
    typedef s_pool<s_pa,_s_pa> POOL ;
    static POOL* pool ;
    static constexpr const bool LEAK = false ; 
    static void SetPOOL( POOL* pool_ ); 
    static int level() ; 

    static constexpr const char* NAME = "s_pa.npy" ; 
    static void Serialize( _s_pa& p, const s_pa* o ); 
    static s_pa* Import(  const _s_pa* p, const std::vector<_s_pa>& buf ); 
    
    s_pa(); 
    ~s_pa(); 

    int pid ; 

    double x0, y0, z0, w0, x1, y1 ;


    double zmin() const { return x1 ; }
    double zmax() const { return y1 ; }
    void decrease_zmin(double dz){ assert( dz >= 0.) ; x1 -= dz ; }
    void increase_zmax(double dz){ assert( dz >= 0.) ; y1 += dz ; }

 
    bool is_root_importable() const ; 
    std::string desc() const ;  
}; 


inline int s_pa::level() {  return pool ? pool->level : ssys::getenvint("sn__level",-1) ; } // static 

inline void s_pa::Serialize( _s_pa& p, const s_pa* o )
{
    p.x0 = o->x0 ; 
    p.y0 = o->y0 ; 
    p.z0 = o->z0 ; 
    p.w0 = o->w0 ; 
    p.x1 = o->x1 ; 
    p.y1 = o->y1 ; 
} 
inline s_pa* s_pa::Import( const _s_pa* p, const std::vector<_s_pa>& )
{
    s_pa* o = new s_pa ; 
    o->x0 = p->x0 ; 
    o->y0 = p->y0 ; 
    o->z0 = p->z0 ; 
    o->w0 = p->w0 ; 
    o->x1 = p->x1 ; 
    o->y1 = p->y1 ; 
    return o ; 
}


inline s_pa::s_pa()
    :
    pid(pool ? pool->add(this) : -1),
    x0(0.),
    y0(0.),
    z0(0.),
    w0(0.),
    x1(0.),
    y1(0.)
{
    if(level() > 1) std::cerr << "s_pa::s_pa pid " << pid << std::endl ; 
}
inline s_pa::~s_pa()
{
    if(level() > 1) std::cerr << "s_pa::~s_pa pid " << pid << std::endl ; 
    if(pool) pool->remove(this); 
}

inline bool s_pa::is_root_importable() const 
{
    return true ; 
}

inline std::string s_pa::desc() const 
{
    std::stringstream ss ;
    ss 
       << " x0 " << std::setw(10) << std::fixed << std::setprecision(3) << x0 
       << " y0 " << std::setw(10) << std::fixed << std::setprecision(3) << y0
       << " z0 " << std::setw(10) << std::fixed << std::setprecision(3) << z0 
       << " w0 " << std::setw(10) << std::fixed << std::setprecision(3) << w0 
       << " x1 " << std::setw(10) << std::fixed << std::setprecision(3) << x1 
       << " y1 " << std::setw(10) << std::fixed << std::setprecision(3) << y1 
       ;
        
    std::string str = ss.str(); 
    return str ; 
}


