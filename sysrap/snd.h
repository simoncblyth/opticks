#pragma once
/**
snd.h : constituent CSG node in preparation
============================================= 

snd.h intended as minimal first step, transiently holding 
parameters of G4VSolid CSG trees for subsequent use by CSGNode::Make
and providing dependency fire break between G4 and CSG 

* initially thought snd.h would be transient with no persisting
  and no role on GPU. But that is inconsistent with the rest of stree.h and also 
  want to experiment with non-binary intersection in future, 
  so can use snd.h as testing ground for non-binary solid persisting.  

* snd.h instances are one-to-one related with CSG/CSGNode.h, 

Usage requires initializing the static "pools"::

    #include "snd.h"
    std::vector<snd> snd::node  = {} ; 
    std::vector<spa> snd::param = {} ; 
    std::vector<sxf> snd::xform = {} ; 
    std::vector<sbb> snd::aabb  = {} ; 

**/

#include <string>
#include <sstream>
#include <iomanip>
#include <vector>

#include "OpticksCSG.h"
#include "scuda.h"
#include "stran.h"
#include "NPFold.h"
#include "NPX.h"

#include "SYSRAP_API_EXPORT.hh"


struct SYSRAP_API spa
{
    static constexpr const char* NAME = "spa" ; 
    static constexpr const int N = 6 ; 
    double x0, y0, z0, w0, x1, y1 ;  
    std::string desc() const ; 
}; 

inline std::string spa::desc() const
{
    const double* d = &x0 ; 
    int wid = 8 ; 
    int pre = 3 ; 
    std::stringstream ss ;
    ss << "(" ;  
    for(int i=0 ; i < N ; i++) ss << std::setw(wid) << std::fixed << std::setprecision(pre) << d[i] << " " ; 
    ss << ")" ;  
    std::string str = ss.str(); 
    return str ; 
}

struct SYSRAP_API sbb
{
    static constexpr const char* NAME = "sbb" ; 
    static constexpr const int N = 6 ; 
    double x0, y0, z0, x1, y1, z1 ;  
    std::string desc() const ; 
}; 

inline std::string sbb::desc() const
{
    const double* d = &x0 ; 
    int wid = 8 ; 
    int pre = 3 ; 
    std::stringstream ss ;
    ss << "(" ;  
    for(int i=0 ; i < N ; i++) ss << std::setw(wid) << std::fixed << std::setprecision(pre) << d[i] << " " ; 
    ss << ")" ;  
    std::string str = ss.str(); 
    return str ; 
}

struct SYSRAP_API sxf
{
    static constexpr const char* NAME = "sxf" ; 
    glm::tmat4x4<double> mat ; 
    std::string desc() const ;  
}; 

inline std::string sxf::desc() const 
{
    std::stringstream ss ;
    ss << glm::to_string(mat) ; 
    std::string str = ss.str(); 
    return str ; 
}



struct SYSRAP_API snd
{
    static constexpr const char* NAME = "snd" ; 
    static constexpr const double zero = 0. ; 

    // pools that are index referenced from the snd instances 
    // HMM: maybe use separate struct for these pools ?
    static std::vector<snd>   node ; 
    static std::vector<spa>   param ; 
    static std::vector<sbb>   aabb ; 
    static std::vector<sxf>   xform ; 

    static int Add(const snd& nd ); 
    static std::string Brief(); 
    static std::string Desc(); 
    static NPFold* Serialize(); 
    static void    Import(const NPFold* fold); 

    template<typename T>
    static std::string Desc(int idx, const std::vector<T>& vec); 

    static constexpr const int N = 6 ;
    int tc ;  // typecode
    int fc ;  // first_child
    int nc ;  // num_child  

    int pa ;  // ref param
    int bb ;  // ref bbox
    int xf ;  // ref transform 


    void init(); 
    void setTypecode( unsigned tc ); 
    void setParam( double x,  double y,  double z,  double w,  double z1, double z2 ); 
    void setAABB(  double x0, double y0, double z0, double x1, double y1, double z1 );
    void setXForm( const glm::tmat4x4<double>& t ); 

    std::string brief() const ; 
    std::string desc() const ; 


    // follow CSGNode where these param will end up 
    static snd Sphere(double radius); 
    static snd ZSphere(double radius, double z1, double z2); 
    static snd Box3(double fullside); 
    static snd Box3(double fx, double fy, double fz ); 
    static snd Boolean( OpticksCSG_t op,  int l, int r ); 
};

inline int snd::Add(const snd& nd ) // static
{
    int p = node.size(); 
    node.push_back(nd); 
    return p ; 
}
inline std::string snd::Brief() // static
{
    std::stringstream ss ; 
    ss << "snd" 
       << " node " << node.size() 
       << " param " << param.size() 
       << " aabb " << aabb.size() 
       << " xform " << xform.size() 
       ;
    std::string str = ss.str(); 
    return str ; 
}
inline std::string snd::Desc() // static
{
    std::stringstream ss ; 
    ss << Brief() << std::endl ; 
    for(unsigned i=0 ; i < node.size() ; i++) ss << node[i].desc() << std::endl ; 
    std::string str = ss.str(); 
    return str ; 
}


inline NPFold* snd::Serialize()  // static 
{
    NPFold* fold = new NPFold ; 
    fold->add("node",  NPX::ArrayFromVec<int,    snd>(node)); 
    fold->add("param", NPX::ArrayFromVec<double, spa>(param)); 
    fold->add("aabb",  NPX::ArrayFromVec<double, sbb>(aabb)); 
    fold->add("xform", NPX::ArrayFromVec<double, sxf>(xform)); 
    return fold ; 
}
inline void snd::Import(const NPFold* fold) // static
{ 
    NPX::VecFromArray<snd>(node,  fold->get("node"));  // NB the vec are cleared first 
    NPX::VecFromArray<spa>(param, fold->get("param")); 
    NPX::VecFromArray<sbb>(aabb,  fold->get("aabb")); 
    NPX::VecFromArray<sxf>(xform, fold->get("xform")); 
}




template<typename T>
inline std::string snd::Desc(int idx, const std::vector<T>& vec)   // static 
{
    int w = 3 ; 
    std::stringstream ss ; 
    ss << T::NAME << ":" << std::setw(w) << idx << " " ;  
    if(idx < 0) 
    {
        ss << "(none)" ; 
    }
    else if( idx >= vec.size() )
    {
        ss << "(invalid)" ; 
    }
    else
    {
        const T& obj = vec[idx] ;  
        ss << obj.desc() ; 
    }
    std::string str = ss.str(); 
    return str ; 
}

inline void snd::init()
{
    tc = -1 ;
    fc = -1 ; 
    nc = -1 ;

    pa = -1 ; 
    bb = -1 ;
    xf = -1 ; 
}

inline void snd::setTypecode( unsigned _tc )
{
    init(); 
    tc = _tc ; 
}
inline void snd::setXForm(const glm::tmat4x4<double>& t )
{
    sxf o ; 
    o.mat = t ; 
    xf = xform.size() ; 
    xform.push_back(o) ; 
}
inline void snd::setParam( double x, double y, double z, double w, double z1, double z2 )
{
    pa = param.size() ; 
    param.push_back({ x, y, z, w, z1, z2 }); 
}
inline void snd::setAABB( double x0, double y0, double z0, double x1, double y1, double z1 )
{
    bb = aabb.size(); 
    aabb.push_back( {x0, y0, z0, x1, y1, z1} ); 
}

inline std::string snd::brief() const 
{
    int w = 3 ; 
    std::stringstream ss ; 
    ss 
       << " tc:" << std::setw(w) << tc 
       << " fc:" << std::setw(w) << fc 
       << " nc:" << std::setw(w) << nc 
       << " pa:" << std::setw(w) << pa 
       << " bb:" << std::setw(w) << bb 
       << " xf:" << std::setw(w) << xf
       << " "    << CSG::Tag(tc) 
       ; 
    std::string str = ss.str(); 
    return str ; 
}

inline std::string snd::desc() const 
{
    int w = 3 ; 
    std::stringstream ss ; 
    ss 
       << brief() << std::endl  
       << Desc<spa>(pa, param) << std::endl  
       << Desc<sbb>(bb, aabb ) << std::endl 
       << Desc<sxf>(xf, xform) << std::endl 
       ; 

    for(int i=0 ; i < nc ; i++) ss << Desc<snd>(fc+i, node) << std::endl ; 

    std::string str = ss.str(); 
    return str ; 
}

inline std::ostream& operator<<(std::ostream& os, const snd& v)  
{
    os << v.desc() ;  
    return os; 
}

inline snd snd::Sphere(double radius)  // static
{
    assert( radius > zero ); 
    snd nd = {} ;
    nd.setTypecode(CSG_SPHERE) ; 
    nd.setParam( zero, zero, zero, radius, zero, zero );  
    nd.setAABB(  -radius, -radius, -radius,  radius, radius, radius  );  
    return nd ;
}

inline snd snd::ZSphere(double radius, double z1, double z2)  // static
{
    assert( radius > zero ); 
    assert( z2 > z1 );  
    snd nd = {} ;
    nd.setTypecode(CSG_ZSPHERE) ; 
    nd.setParam( zero, zero, zero, radius, z1, z2 );  
    nd.setAABB(  -radius, -radius, z1,  radius, radius, z2  );  
    return nd ;
}

inline snd snd::Box3(double fullside)  // static 
{
    return Box3(fullside, fullside, fullside); 
}
inline snd snd::Box3(double fx, double fy, double fz )  // static 
{
    assert( fx > 0. );  
    assert( fy > 0. );  
    assert( fz > 0. );  

    snd nd = {} ;
    nd.setTypecode(CSG_BOX3) ; 
    nd.setParam( fx, fy, fz, 0.f, 0.f, 0.f );  
    nd.setAABB( -fx*0.5 , -fy*0.5, -fz*0.5, fx*0.5 , fy*0.5, fz*0.5 );   
    return nd ; 
}

inline snd snd::Boolean( OpticksCSG_t op, int l, int r ) // static 
{
    assert( l > -1 && r > -1 );
    assert( l+1 == r );  

    snd nd = {} ;
    nd.setTypecode( op ); 
    nd.nc = 2 ; 
    nd.fc = l ;
 
    return nd ; 
}


