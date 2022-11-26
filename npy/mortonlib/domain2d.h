#pragma once

#include <cstdint>
#include <vector>
#include "morton2d.h"
#include "NP.hh"

struct domain2d
{
    double dx[2] ; 
    double dy[2] ; 
    uint64_t scale ; 

    domain2d( double dx0, double dx1 , double dy0, double dy1, uint64_t scale_=0xffffffff )  
    {
        dx[0] = dx0 ; 
        dx[1] = dx1 ;
        dy[0] = dy0 ; 
        dy[1] = dy1 ;
        scale = scale_ ; 
    }

    // fx_ fy_ returns in domain values in range 0->1
    double   fx_( double x ) const { return (x - dx[0])/(dx[1]-dx[0]) ; } 
    double   fy_( double y ) const { return (y - dy[0])/(dy[1]-dy[0]) ; } 

    // ix_, iy_ return integer scaled values in range 0->scale 
    uint64_t ix_( double x ) const { return scale*fx_(x) ; }
    uint64_t iy_( double y ) const { return scale*fy_(y) ; }

    // convert coordinates in domain into integers in 0->scale 
    // then bit interleave them into uint64_t morton key 
    uint64_t key_( double x, double y )
    {
        uint64_t ix = ix_(x) ; 
        uint64_t iy = iy_(y) ; 
        morton2d<uint64_t> m2(ix,iy); 
        return m2.key ;     
    }

    // decode morton key integer back into domain doubles 
    void decode_( double& x, double& y, uint64_t ik )
    {
        uint64_t ix, iy ;
        morton2d<uint64_t> m2(ik); 
        m2.decode( ix, iy ); 

        double fx = double(ix)/double(scale) ; 
        double fy = double(iy)/double(scale) ; 
        x = dx[0] + fx*(dx[1]-dx[0]) ; 
        y = dy[0] + fy*(dy[1]-dy[0]) ; 
    }


    static void coarsen( std::vector<uint64_t>& uu , const std::vector<uint64_t>& kk , uint64_t mask ); 
    void get_circle( std::vector<uint64_t>& kk, double radius ); 
    NP* make_array( const std::vector<uint64_t>& kk ); 
    NP* make_array( const std::vector<uint64_t>& uu, uint64_t mask ); 

}; 



/**
domain2d::coarsen
-------------------

Coarsen coordinates by masking the least significant bits of morton code
and uniqify them into uu. 

This is a form a histogramming of 2d coordinates using 1d morton code
that represents the coordinates. 

HMM: have to mask almost all 64 bits to see reduction in counts 
Presumably this is  because the input points are in a grid, they are 
not on top of each other.  

**/

inline void domain2d::coarsen( std::vector<uint64_t>& uu , const std::vector<uint64_t>& kk , uint64_t mask )
{
    uu.resize(0); 
    for(unsigned i=0 ; i < kk.size() ; i++)
    {
        const uint64_t& ik0 = kk[i]  ;  
        uint64_t ik = ik0 & mask ;  
        if( std::find( uu.begin(), uu.end(), ik ) == uu.end() ) uu.push_back(ik) ; 
    }
}

/**
domain2d::get_circle
-----------------------

Fills vector of morton codes of xy coordinates that are close to a circle. 

**/

inline void domain2d::get_circle( std::vector<uint64_t>& kk, double radius )
{
    for(double x=dx[0] ; x < dx[1] ; x += (dx[1]-dx[0])/200. )
    for(double y=dy[0] ; y < dy[1] ; y += (dy[1]-dy[0])/200. )
    {
        double dist = sqrtf(x*x+y*y) - radius ;    
        bool circle = std::abs(dist) < 1.1f ; 
        if(!circle) continue ; 

        uint64_t k = key_(x,y) ;  
        kk.push_back( k ); 
    }
}

/**
domain2d::make_array
----------------------

Decodes the vector of morton code integers into an array of xy coordinates. 

**/

inline NP* domain2d::make_array( const std::vector<uint64_t>& kk  )
{
    NP* a = NP::Make<float>( kk.size(), 2 ) ; 
    float* aa = a->values<float>();  
    for(unsigned i=0 ; i < kk.size() ; i++) 
    { 
        const uint64_t& ik = kk[i] ;
        double x, y ; 
        decode_(x, y, ik); 

        aa[2*i+0] = x ; 
        aa[2*i+1] = y ; 
    }
    return a ; 
}

inline NP* domain2d::make_array( const std::vector<uint64_t>& kk, uint64_t mask )
{
    std::vector<uint64_t> uu ;
    coarsen(uu, kk, mask );  
    NP* a = make_array( uu ); 
    return a ; 
}

