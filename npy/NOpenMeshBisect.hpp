#pragma once

#include <functional>
#include <boost/cstdint.hpp>

#include "NOpenMeshType.hpp"
#include "NOpenMeshEnum.hpp"

template <typename T>
struct NPY_API  NOpenMeshBisectFunc
{
    typedef typename std::function<float(float,float,float)> SDF ; 
    typedef typename T::Point              P ; 

    NOpenMeshBisectFunc( const SDF& sdf, const P& a, const P& b ) ;

    void position(P& tp, const float t) const ;
    float operator()(const float t) const ;

    const SDF& sdf ; 
    const P&     a ; 
    const P&     b ; 
};

struct NPY_API NOpenMeshBisectTol
{
    NOpenMeshBisectTol( float tolerance ) : tolerance(tolerance) {} ;
    bool operator()(const float& min, const float& max ) 
    {
        return max - min < tolerance ; 
    }
    float tolerance ;
};

template <typename T>
struct NPY_API  NOpenMeshBisect
{
    typedef typename std::function<float(float,float,float)> SDF ; 
    typedef typename T::Point              P ; 

     NOpenMeshBisect( const SDF& sdf, const P& a, const P& b , float tolerance ) ;

     void bisect(P& frontier, float& t );

     const NOpenMeshBisectFunc<T> func ; 
     const NOpenMeshBisectTol     tol ; 
     const bool                 invalid ;
     const bool                 degenerate ;
     boost::uintmax_t           iterations ; 

};



