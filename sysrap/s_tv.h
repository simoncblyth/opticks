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
    static constexpr const char* ITEM = "2,4,4" ;  

    glm::tmat4x4<double> t ; 
    glm::tmat4x4<double> v ;
};

#include "SYSRAP_API_EXPORT.hh"

struct SYSRAP_API s_tv
{
    static constexpr const bool LEAK = false ; 
    static constexpr const char* NAME = "s_tv.npy" ; 

    typedef s_pool<s_tv,_s_tv> POOL ;
    static POOL* pool ;
    static void SetPOOL( POOL* pool_ ); 
    static int level() ; 
    static void Serialize( _s_tv& p, const s_tv* o ); 
    static s_tv* Import(  const _s_tv* p, const std::vector<_s_tv>& buf ); 
    
    s_tv(); 
    ~s_tv(); 

    int pid ; 
    glm::tmat4x4<double> t ; 
    glm::tmat4x4<double> v ;
 
    bool is_root() const { return true ; } 
    std::string desc_full() const ;    // full but poorly formatted 
    std::string desc() const ;  

    static std::string DescTranslate(const glm::tmat4x4<double>& tr);
    static std::string DescOffset4(const glm::tmat4x4<double>& tr, unsigned offset  );
    static std::string Desc(const glm::tmat4x4<double>& tr);
    static std::string Desc(const glm::tvec4<double>& t);
    static std::string Desc(const double* tt, int num);



}; 

inline void s_tv::SetPOOL( POOL* pool_ ){ pool = pool_ ; } // static 
inline int  s_tv::level() {  return pool ? pool->level : ssys::getenvint("sn__level",-1) ; } // static 

inline void s_tv::Serialize( _s_tv& p, const s_tv* o ) // static
{
    p.t = o->t ; 
    p.v = o->v ; 
} 
inline s_tv* s_tv::Import( const _s_tv* p, const std::vector<_s_tv>& ) // static
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


inline std::string s_tv::desc_full() const 
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

inline std::string s_tv::desc() const 
{
    std::stringstream ss ;
    ss 
        << " t " << DescTranslate(t)
        << " v " << DescTranslate(v)
        ;
    std::string str = ss.str(); 
    return str ; 
}


inline std::string s_tv::DescTranslate(const glm::tmat4x4<double>& tr )
{
    return DescOffset4(tr, 12); 
}
inline std::string s_tv::DescOffset4(const glm::tmat4x4<double>& tr, unsigned offset  )
{
    const double* tt = glm::value_ptr(tr) ; 
    return Desc(tt + offset, 4 ); 
}
inline std::string s_tv::Desc(const glm::tmat4x4<double>& tr)
{
    const double* tt = glm::value_ptr(tr); 
    return Desc(tt, 16 ); 
}
inline std::string s_tv::Desc(const glm::tvec4<double>& t)
{
    const double* tt = glm::value_ptr(t); 
    return Desc(tt, 4 ); 
}
inline std::string s_tv::Desc(const double* tt, int num)
{
    std::stringstream ss ; 
    for(int i=0 ; i < num ; i++) 
        ss 
            << ( i % 4 == 0 && num > 4 ? ".\n" : "" ) 
            << " " << std::fixed << std::setw(10) << std::setprecision(4) << tt[i] 
            << ( i == num-1 && num > 4 ? ".\n" : "" ) 
            ;

    std::string str = ss.str(); 
    return str ; 
}






