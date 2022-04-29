#pragma once
/**
srec.h : highly domain compressed photon step records used for debugging only
==============================================================================

Domain compression means that must carry around domain metadata in order to 
encode or decode the arrays. 

Principal user of srec.h is qevent::add_rec

NB seqhis seqmat histories are defined at photon level, so it does not 
make sense to include them here at step-record level 

For persisting srec arrays use::

   NP* rec = NP<short>::Make(num_rec, max_rec, 4, 2)



**/

#include "scuda.h"

#if defined(__CUDACC__) || defined(__CUDABE__)
#    define SREC_METHOD __device__ __forceinline__
#else
#    define SREC_METHOD inline 
#endif


#if defined(__CUDACC__) || defined(__CUDABE__)
#else
#include <string>
#include <iostream>
#include <sstream>
#endif


struct srec 
{
    short4 post ; // position and time into 4*int16 = 64 bits 
    char4  polw ; // tightly packed, polarization and wavelength into 4*int8 = 32 bits 
    uchar4 flag ; // 4*int8 = 32 bits   

    SREC_METHOD void zero(); 

    SREC_METHOD void set_position(const float3& pos, const float4& ce );  
    SREC_METHOD void get_position(      float3& pos, const float4& ce ) const ; 

    SREC_METHOD void set_time( const float  t, const float2& td );  
    SREC_METHOD void get_time(       float& t, const float2& td ) const ; 

    SREC_METHOD void set_polarization( const float3& pol );  
    SREC_METHOD void get_polarization(       float3& pol ) const ; 

    SREC_METHOD void set_wavelength( const float  w, const float2& wd );  
    SREC_METHOD void get_wavelength(       float& w, const float2& wd ) const ; 


#if defined(__CUDACC__) || defined(__CUDABE__)
#else
    SREC_METHOD std::string desc() const ; 
#endif

};


SREC_METHOD void srec::zero()
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

SREC_METHOD std::string srec::desc() const 
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



#if defined(__CUDACC__) || defined(__CUDABE__)
#define FLOAT2INT_RN(x) (__float2int_rn(x)) 
#else
#define FLOAT2INT_RN(x) (lrint(x))  
#endif 


/**
srec::set_position
-------------------

NB positions outside the domain defined by the center-extent *ce* will just cycle. 
That is an inherent limitation of the domain compression.  But as compressed records
are just used for debug, that is fine. Just have to make sure the center-extent covers
the region of interest. 

**/



SREC_METHOD void srec::set_position( const float3& pos, const float4& ce )
{
    post.x = FLOAT2INT_RN( 32767.0f * (pos.x - ce.x)/ce.w ) ;
    post.y = FLOAT2INT_RN( 32767.0f * (pos.y - ce.y)/ce.w ) ;
    post.z = FLOAT2INT_RN( 32767.0f * (pos.z - ce.z)/ce.w ) ;
}

/**
srec::set_time
---------------

Time domain is treated as a center at zero with an extent 
even though that is wasting half the bits  (no -ve times) 
because it simplifies analysis to treat times the same as positions. 

**/

SREC_METHOD void srec::set_time( const float  t, const float2& td )
{
    post.w = FLOAT2INT_RN(  32767.0f*(t - td.x)/td.y );
}

SREC_METHOD void srec::get_position( float3& pos, const float4& ce ) const
{
    pos.x = float(post.x)*ce.w/32767.0f + ce.x ;
    pos.y = float(post.y)*ce.w/32767.0f + ce.y ;
    pos.z = float(post.z)*ce.w/32767.0f + ce.z ;
}

SREC_METHOD void srec::get_time(       float& t, const float2& td ) const
{
    t = float(post.w)*td.y/32767.0f + td.x ;
}


/**
srec::set_polarization
-----------------------

Components of polarization have implicit domain of -1.f to 1.f 

**/
SREC_METHOD void srec::set_polarization( const float3& pol )
{
    polw.x = FLOAT2INT_RN(pol.x*127.f );
    polw.y = FLOAT2INT_RN(pol.y*127.f );
    polw.z = FLOAT2INT_RN(pol.z*127.f );
}

SREC_METHOD void srec::get_polarization( float3& pol ) const
{
    pol.x = (float(polw.x)/127.f) ;
    pol.y = (float(polw.y)/127.f) ;
    pol.z = (float(polw.z)/127.f) ;
}

/**
srec::set_wavelength
----------------------

Wavelength domain was previous specificed by a range, 
are now using a symmetrical center-extent form for consistency with other
domains, see qevent.h 

**/

SREC_METHOD void srec::set_wavelength( const float w, const float2& wd )
{
    polw.w = FLOAT2INT_RN(( w - wd.x)*127.f/ wd.y ) ;
}
SREC_METHOD void srec::get_wavelength(       float& w, const float2& wd ) const
{
    w = float(polw.w)*wd.y/127.f + wd.x ;
}


