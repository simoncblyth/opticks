#pragma once
/**
srec.h : highly compressed photon step records used for debugging only
=========================================================================

**/

#include "scuda.h"

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
#include <string>
#include <iostream>
#include <sstream>
#endif


struct srec 
{
    short4 post ; // position and time into 4*int16 = 64 bits 
    uchar4 polw ; // tightly packed, polarization and wavelength into 4*int8 = 32 bits 
    uchar4 flag ;   

    void zero(); 

    void set_position(const float3& pos, const float4& ce );  
    void get_position(      float3& pos, const float4& ce ) const ; 

    void set_time( const float  t, const float2& td );  
    void get_time(       float& t, const float2& td ) const ; 

    void set_polarization( const float3& pol );  
    void get_polarization(       float3& pol ) const ; 

    void set_wavelength( const float  w, const float2& wd );  
    void get_wavelength(       float& w, const float2& wd ) const ; 


#if defined(__CUDACC__) || defined(__CUDABE__)
#else
    std::string desc() const ; 
#endif

};


#if defined(__CUDACC__) || defined(__CUDABE__)
#define FLOAT2UINT_RN __float2uint_rn 
#else
#define FLOAT2UINT_RN lrint  
#endif 


void srec::zero()
{
    post.x = 0 ; 
    post.y = 0 ; 
    post.z = 0 ; 
    post.w = 0 ; 

    polw.x = 0 ; 
    polw.y = 0 ; 
    polw.z = 0 ; 
    polw.w = 0 ; 

    flag.x = 0 ; 
    flag.y = 0 ; 
    flag.z = 0 ; 
    flag.w = 0 ; 
}


#if defined(__CUDACC__) || defined(__CUDABE__)
#else

std::string srec::desc() const 
{
    std::stringstream ss ; 
    ss 
       << " post " << post  
       << " polw " << polw 
       << " flag " << flag  
       ;

    std::string s = ss.str(); 
    return s ; 
}
#endif


/**
srec::set_position
-------------------

NB positions outside the domain defined by the center-extent *ce* will just cycle. 
That is an inherent limitation of the domain compression.  But as compressed records
are just used for debug, that is fine. Just have to make sure the center-extent covers
the region of interest. 

**/

inline void srec::set_position( const float3& pos, const float4& ce )
{
    post.x = FLOAT2UINT_RN( 32767.0f * (pos.x - ce.x)/ce.w ) ;
    post.y = FLOAT2UINT_RN( 32767.0f * (pos.y - ce.y)/ce.w ) ;
    post.z = FLOAT2UINT_RN( 32767.0f * (pos.z - ce.z)/ce.w ) ;
}

/**
srec::set_time
---------------

Time domain is treated as a center at zero with an extent 
even though that is wasting half the bits  (no -ve times) 
because it simplifies analysis to treat times the same as positions. 

**/

inline void srec::set_time( const float  t, const float2& td )
{
    post.w = FLOAT2UINT_RN(  32767.0f*(t - td.x)/td.y );
}

inline void srec::get_position( float3& pos, const float4& ce ) const
{
    pos.x = float(post.x)*ce.w/32767.0f + ce.x ;
    pos.y = float(post.y)*ce.w/32767.0f + ce.y ;
    pos.z = float(post.z)*ce.w/32767.0f + ce.z ;
}

inline void srec::get_time(       float& t, const float2& td ) const
{
    t = float(post.w)*td.y/32767.0f + td.x ;
}


/**
srec::set_polarization
-----------------------

Components of polarization have implicit domain of -1.f to 1.f 

* pol   :  -1.f -> 1.f
* pol+1 :   0. -> 2.f   

**/
inline void srec::set_polarization( const float3& pol )
{
    polw.x = FLOAT2UINT_RN((pol.x+1.f)*127.f );
    polw.y = FLOAT2UINT_RN((pol.y+1.f)*127.f );
    polw.z = FLOAT2UINT_RN((pol.z+1.f)*127.f );
}

inline void srec::get_polarization( float3& pol ) const
{
    pol.x = (float(polw.x)/127.f) - 1.f ;
    pol.y = (float(polw.y)/127.f) - 1.f ;
    pol.z = (float(polw.z)/127.f) - 1.f ;
}

/**
srec::set_wavelength
----------------------

Wavelength domain is specified by a range from wd.x to wd.y, 
unlike other domains that are specified by a center point and an extent. 

**/

inline void srec::set_wavelength( const float  w, const float2& wd )
{
    polw.w = FLOAT2UINT_RN( 255.f * ( w - wd.x) / wd.y ) ;
}
inline void srec::get_wavelength(       float& w, const float2& wd ) const
{
    w = float(polw.w)*wd.y/255.f + wd.x ;
}


