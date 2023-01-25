#pragma once
/**
snd.h : constituent CSG node in preparation
============================================= 

snd.h intended as minimal first step, transiently holding 
parameters of G4VSolid CSG trees for subsequent use by CSGNode::Make
and providing dependency fire break between G4 and CSG 

* role is to TRANSIENTLY represent the CSG tree and hold the parameters
  needed to create the CSGPrim/CSGNode which are focussed on  
  serialization+intersection

* snd.h instances explicitly WILL NOT BE serializable and 
  have no role on GPU, so can freely use CPU only techniques
  such as using pointers to express the binary (and other) tree types 

* snd.h instances are one-to-one related with CSG/CSGNode.h, 


TODO: holding transforms 

**/

#include <string>
#include <sstream>
#include <iomanip>

#include "OpticksCSG.h"
#include "scuda.h"
#include "stran.h"

struct snd
{
    static constexpr const int N = 6 ; 
    static constexpr const double zero = 0. ; 

    double* param ; 
    double* aabb  ; 
    double* planes ; 
    unsigned tc ;  

    snd* left ; 
    snd* right ; 
    snd* parent ; 

    Tran<double>* tr ; 


    void setParam(double x,  double y,  double z,  double w,  double z1, double z2 ); 
    void setAABB( double x0, double y0, double z0, double x1, double y1, double z1 );
    void setTypecode( unsigned tc ); 

    static std::string Desc(const double* d, int wid=8, int pre=2 ); 
    std::string desc() const ; 

    // follow CSGNode where these param will end up 
    static snd Sphere(double radius); 
    static snd ZSphere(double radius, double z1, double z2); 
    static snd Box3(double fullside); 
    static snd Box3(double fx, double fy, double fz ); 
    static snd Boolean( OpticksCSG_t op,  snd* l, snd* r ); 

};

inline void snd::setParam( double x, double y, double z, double w, double z1, double z2 )
{
    if(param == nullptr) param = new double[N] ; 
    param[0] = x ; 
    param[1] = y ; 
    param[2] = z ; 
    param[3] = w ;
    param[4] = z1 ;
    param[5] = z2 ;
}

inline void snd::setAABB( double x0, double y0, double z0, double x1, double y1, double z1 )
{
    if(aabb == nullptr) aabb = new double[N] ; 
    aabb[0] = x0 ; 
    aabb[1] = y0 ; 
    aabb[2] = z0 ; 
    aabb[3] = x1 ; 
    aabb[4] = y1 ; 
    aabb[5] = z1 ; 
}

inline void snd::setTypecode( unsigned _tc )
{
    tc = _tc ; 
}
inline std::string snd::Desc(const double* d, int wid, int pre  )
{
    std::stringstream ss ;
    if( d == nullptr ) 
    {
        ss << "(null)" ; 
    }
    else
    { 
        ss << "(" ;  
        for(int i=0 ; i < N ; i++) ss << std::setw(wid) << std::fixed << std::setprecision(pre) << d[i] << " " ; 
        ss << ")" ;  
    }
    std::string str = ss.str(); 
    return str ; 
} 

inline std::string snd::desc() const 
{
    std::stringstream ss ; 
    ss << "snd::desc " 
       << CSG::Tag(tc) 
       << " param " << Desc(param) 
       << " aabb  " << Desc(aabb )
       << " tr " << ( tr ? "Y" : "N" )
       << std::endl 
       ; 

    if(left)  ss << "l:" << left->desc() ; 
    if(right) ss << "r:" << right->desc() ; 
    if(left && !right) ss << "ERROR: boolean left BUT no right " << std::endl  ; 
    if(tr)    ss << tr->desc() ;   

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
    nd.setParam( zero, zero, zero, radius, zero, zero );  
    nd.setAABB(  -radius, -radius, -radius,  radius, radius, radius  );  
    nd.setTypecode(CSG_SPHERE) ; 
    return nd ;
}

inline snd snd::ZSphere(double radius, double z1, double z2)  // static
{
    assert( radius > zero ); 
    assert( z2 > z1 );  
    snd nd = {} ;
    nd.setParam( zero, zero, zero, radius, z1, z2 );  
    nd.setAABB(  -radius, -radius, z1,  radius, radius, z2  );  
    nd.setTypecode(CSG_ZSPHERE) ; 
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
    nd.setParam( fx, fy, fz, 0.f, 0.f, 0.f );  
    nd.setAABB( -fx*0.5 , -fy*0.5, -fz*0.5, fx*0.5 , fy*0.5, fz*0.5 );   
    nd.setTypecode(CSG_BOX3) ; 
    return nd ; 
}

inline snd snd::Boolean( OpticksCSG_t op,  snd* l, snd* r ) // static 
{
    snd nd = {} ;
    nd.left = l ; 
    nd.right = r ; 
    nd.setTypecode( op ); 
    return nd ; 
}


