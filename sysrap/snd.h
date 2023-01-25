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

template<typename T>
struct snd
{
    static constexpr const int N = 6 ; 
    static constexpr const T zero = 0. ; 

    T* param ; 
    T* aabb  ; 
    T* planes ; 
    unsigned tc ;  

    snd* left ; 
    snd* right ; 
    snd* parent ; 

    void setParam(T x,  T y,  T z,  T w,  T z1, T z2 ); 
    void setAABB( T x0, T y0, T z0, T x1, T y1, T z1 );
    void setTypecode( unsigned tc ); 

    static std::string Desc(const T* d, int wid=8, int pre=2 ); 
    std::string desc() const ; 

    static snd<T> Sphere(T radius); 
    static snd<T> ZSphere(T radius, T z1, T z2); 

};

template<typename T>
inline void snd<T>::setParam( T x, T y, T z, T w, T z1, T z2 )
{
    if(param == nullptr) param = new T[N] ; 
    param[0] = x ; 
    param[1] = y ; 
    param[2] = z ; 
    param[3] = w ;
    param[4] = z1 ;
    param[5] = z2 ;
}

template<typename T>
inline void snd<T>::setAABB( T x0, T y0, T z0, T x1, T y1, T z1 )
{
    if(aabb == nullptr) aabb = new T[N] ; 
    aabb[0] = x0 ; 
    aabb[1] = y0 ; 
    aabb[2] = z0 ; 
    aabb[3] = x1 ; 
    aabb[4] = y1 ; 
    aabb[5] = z1 ; 
}

template<typename T>
inline void snd<T>::setTypecode( unsigned _tc )
{
    tc = _tc ; 
}
template<typename T>
inline std::string snd<T>::Desc(const T* d, int wid, int pre  )
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

template<typename T>
inline std::string snd<T>::desc() const 
{
    std::stringstream ss ; 
    ss << "snd<" << ( sizeof(T) == 8 ? "double" : "float" ) << ">::desc " 
       << CSG::Tag(tc) 
       << " param " << Desc(param) 
       << " aabb  " << Desc(aabb )
       ; 

    std::string str = ss.str(); 
    return str ; 
}

inline std::ostream& operator<<(std::ostream& os, const snd<double>& v)  
{
    os << v.desc() << std::endl ;  
    return os; 
}
inline std::ostream& operator<<(std::ostream& os, const snd<float>& v)  
{
    os << v.desc() << std::endl ;  
    return os; 
}

template<typename T>
inline snd<T> snd<T>::Sphere(T radius)  // static
{
    assert( radius > zero ); 
    snd<double> nd = {} ;
    nd.setParam( zero, zero, zero, radius, zero, zero );  
    nd.setAABB(  -radius, -radius, -radius,  radius, radius, radius  );  
    nd.setTypecode(CSG_SPHERE) ; 
    return nd ;
}

template<typename T>
inline snd<T> snd<T>::ZSphere(T radius, T z1, T z2)  // static
{
    assert( radius > zero ); 
    assert( z2 > z1 );  
    snd<double> nd = {} ;
    nd.setParam( zero, zero, zero, radius, z1, z2 );  
    nd.setAABB(  -radius, -radius, z1,  radius, radius, z2  );  
    nd.setTypecode(CSG_ZSPHERE) ; 
    return nd ;
}


