#pragma once

/**
sevent : host/device communication instance
=============================================

Instantiation of sevent is done by QEvent::QEvent
and the instance is subsequently uploaded to the device after 
device buffer allocations hence the sevent instance
provides event config and device buffer pointers 
both on device and host. 

Note that *num_seed* and *num_photon* will be equal in 
normal operation which uses QEvent::setGensteps. 
However for clarity separate fields are used to 
distinguish photon test running that directly uses
QEvent::setNumPhoton 

In CPU only Geant4 running an sevent.h instance is 
instanciated by U4Recorder::U4Recorder to assist with 
Geant4 mocking of Opticks events.  

**/

#if defined(__CUDACC__) || defined(__CUDABE__)
#    define SEVENT_METHOD __device__ __forceinline__
#else
#    define SEVENT_METHOD inline 
#endif

struct float4 ; 
struct float2 ; 
struct quad4 ; 
struct quad6 ; 
struct srec ; 
struct sseq ; 
struct sphoton ; 

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
#include <string>
#include <sstream>
#include <iomanip>
#include "SEventConfig.hh"
#endif

struct sevent
{
    static constexpr unsigned genstep_itemsize = 6*4 ; 
    static constexpr unsigned genstep_numphoton_offset = 3 ; 
    static constexpr float w_lo = 80.f ; 
    static constexpr float w_hi = 800.f ; 
    static constexpr float w_center = (w_lo+w_hi)/2.f ; // convert wavelength range into center-extent form 
    static constexpr float w_extent = (w_hi-w_lo)/2.f ; 

    float4 center_extent ; 
    float2 time_domain ; 
    float2 wavelength_domain ; 

    // values here come from SEventConfig 
    int      max_genstep ; // eg:      100,000
    int      max_photon  ; // eg:  100,000,000
    int      max_simtrace ; // eg: 100,000,000
    int      max_bounce  ; // eg:            9 
    int      max_record  ; // eg:           10  full step record 
    int      max_rec     ; // eg:           10  compressed step record
    int      max_seq     ; // eg:           16  seqhis/seqbnd

    int      num_genstep ; 
    quad6*   genstep ; 

    int      num_seed ; 
    int*     seed ;     

    int      num_photon ; 
    sphoton* photon ; 

    int      num_record ; 
    sphoton* record ; 

    int      num_rec ; 
    srec*    rec ; 

    int      num_seq ; 
    sseq*    seq ; 

    int      num_hit ; 
    sphoton* hit ; 

    int      num_simtrace ; 
    quad4*   simtrace ; 


    SEVENT_METHOD void add_rec( srec& r, unsigned idx, unsigned bounce, const sphoton& p); 
    SEVENT_METHOD void add_simtrace( unsigned idx, const quad4& p, const quad2* prd, float tmin ); 


    // not including prd here as that is clearly for debugging only 

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
    SEVENT_METHOD void init(); 
    SEVENT_METHOD void init_domain(float extent, float time_max); 

    SEVENT_METHOD std::string descMax() const ; 
    SEVENT_METHOD std::string descNum() const ; 
    SEVENT_METHOD std::string descBuf() const ; 
    SEVENT_METHOD std::string desc() const ; 

    SEVENT_METHOD void get_domain(quad4& dom) const ; 
    SEVENT_METHOD void get_config(quad4& cfg) const ; 
    SEVENT_METHOD void zero(); 
#endif 

}; 


#if defined(__CUDACC__) || defined(__CUDABE__)
#else

SEVENT_METHOD void sevent::init()
{
    max_genstep  = SEventConfig::MaxGenstep() ; 
    max_photon   = SEventConfig::MaxPhoton()  ; 
    max_simtrace = SEventConfig::MaxSimtrace()  ; 
    max_bounce   = SEventConfig::MaxBounce()  ; 
    max_record   = SEventConfig::MaxRecord()  ;  // full step record
    max_rec      = SEventConfig::MaxRec()  ;     // compressed step record 
    max_seq      = SEventConfig::MaxSeq()  ;     // seqhis 

    zero(); 

    float extent = SEventConfig::MaxExtent() ; 
    float time_max = SEventConfig::MaxTime() ; 

    init_domain( extent, time_max );  
}
SEVENT_METHOD void sevent::init_domain(float extent, float time_max)
{
    center_extent.x = 0.f ; 
    center_extent.y = 0.f ; 
    center_extent.z = 0.f ; 
    center_extent.w = extent ; 

    time_domain.x = 0.f ; 
    time_domain.y = time_max ; 
  
    wavelength_domain.x = w_center ;  // TODO: try to make this constexpr 
    wavelength_domain.y = w_extent ; 
}

SEVENT_METHOD std::string sevent::descMax() const
{
    int w = 5 ; 
    std::stringstream ss ; 
    ss 
        << "sevent::descMax " 
        << " evt.max_genstep "   << std::setw(w) << max_genstep  
        << " evt.max_photon  "   << std::setw(w) << max_photon  
        << " evt.max_simtrace  " << std::setw(w) << max_simtrace  
        << " evt.max_bounce  "   << std::setw(w) << max_bounce 
        << " evt.max_record  "   << std::setw(w) << max_record 
        << " evt.max_rec  "      << std::setw(w) << max_rec
        ;

    std::string s = ss.str();  
    return s ; 
}

SEVENT_METHOD std::string sevent::descNum() const
{
    int w = 5 ; 
    std::stringstream ss ; 
    ss 
        << " sevent::descNum  " 
        << " evt.num_genstep " << std::setw(w) << num_genstep 
        << " evt.num_seed "    << std::setw(w) << num_seed   
        << " evt.num_photon "  << std::setw(w) << num_photon
        << " evt.num_simtrace "  << std::setw(w) << num_simtrace
        << " evt.num_record "  << std::setw(w) << num_record
        ;
    std::string s = ss.str();  
    return s ; 
}

SEVENT_METHOD std::string sevent::descBuf() const
{
    int w = 5 ; 
    std::stringstream ss ; 
    ss 
        << std::setw(20) << " sevent::descBuf " 
        << std::endl 
        << std::setw(20) << " evt.genstep     " << std::setw(w) << ( genstep ? "Y" : "N" ) 
        << std::setw(20) << " num_genstep "     << std::setw(7) << num_genstep 
        << std::setw(20) << " max_genstep "     << std::setw(7) << max_genstep 
        << std::endl 
        << std::setw(20) << " evt.seed   "      << std::setw(w) << ( seed    ? "Y" : "N" ) 
        << std::setw(20) << " num_seed "        << std::setw(7) << num_seed 
        << std::setw(20) << " max_photon "      << std::setw(7) << max_photon
        << std::endl 
        << std::setw(20) << " evt.photon "      << std::setw(w) << ( photon  ? "Y" : "N" ) 
        << std::setw(20) << " num_photon "      << std::setw(7) << num_photon 
        << std::setw(20) << " max_photon "      << std::setw(7) << max_photon 
        << std::endl 
        << std::setw(20) << " evt.record "      << std::setw(w) << ( record  ? "Y" : "N" ) 
        << std::setw(20) << " num_record "      << std::setw(7) << num_record 
        << std::setw(20) << " max_record "      << std::setw(7) << max_record 
        << std::endl 
        << std::setw(20) << " evt.rec "         << std::setw(w) << ( record  ? "Y" : "N" ) 
        << std::setw(20) << " num_rec "         << std::setw(7) << num_rec 
        << std::setw(20) << " max_rec "         << std::setw(7) << max_rec 
        << std::endl 
        << std::setw(20) << " evt.seq "         << std::setw(w) << ( seq     ? "Y" : "N" ) 
        << std::setw(20) << " num_seq "         << std::setw(7) << num_seq 
        << std::setw(20) << " max_seq "         << std::setw(7) << max_seq 
        << std::endl 
        << std::setw(20) << " evt.hit "         << std::setw(w) << ( hit     ? "Y" : "N" ) 
        << std::setw(20) << " num_hit "         << std::setw(7) << num_hit 
        << std::setw(20) << " max_photon "      << std::setw(7) << max_photon
        << std::endl 
        << std::setw(20) << " evt.simtrace "    << std::setw(w) << ( simtrace  ? "Y" : "N" ) 
        << std::setw(20) << " num_simtrace "    << std::setw(7) << num_simtrace 
        << std::setw(20) << " max_simtrace "    << std::setw(7) << max_simtrace 
        << std::endl
        ;
    std::string s = ss.str();  
    return s ; 
}


SEVENT_METHOD std::string sevent::desc() const
{
    std::stringstream ss ; 
    ss << descMax() << std::endl ;
    ss << descBuf() << std::endl ;
    ss << descNum() << std::endl ;
    std::string s = ss.str();  
    return s ; 
}





/**
sevent::get_domain
-------------------

HMM: could also use metadata (key, value) pairs on the domain NP array 

ana/evt.py::

    2052         post_center, post_extent = self.post_center_extent()  # center and extent are quads, created by combining position and time domain ce 
    2053         p = self.rx[:,recs,0].astype(np.float32)*post_extent/32767.0 + post_center

**/


SEVENT_METHOD void sevent::get_domain( quad4& dom ) const 
{
   float4 post_center = make_float4( center_extent.x, center_extent.y, center_extent.z, time_domain.x ); 
   float4 post_extent = make_float4( center_extent.w, center_extent.w, center_extent.w, time_domain.y ); 
   float4 polw_center = make_float4( 0.f, 0.f, 0.f, wavelength_domain.x  );  
   float4 polw_extent = make_float4( 1.f, 1.f, 1.f, wavelength_domain.y );  

   dom.q0.f = post_center ; 
   dom.q1.f = post_extent ; 
   dom.q2.f = polw_center ; 
   dom.q3.f = polw_extent ; 

   // xyz duplication allows position and time to be decompressed together 
   // also polarization and wavelength can be decompressed together using same trick 
}


SEVENT_METHOD void sevent::get_config( quad4& cfg ) const 
{
   cfg.q0.u.x = max_genstep ; 
   cfg.q0.u.y = max_photon ; 
   cfg.q0.u.z = max_bounce ; 
   cfg.q0.u.w = 0 ; 

   cfg.q1.u.x = max_record ; 
   cfg.q1.u.y = max_rec ; 
   cfg.q1.u.z = max_seq ; 
   cfg.q1.u.w = 0 ;

   cfg.q2.u.x = num_genstep ; 
   cfg.q2.u.y = num_seed ; 
   cfg.q2.u.z = num_photon ; 
   cfg.q2.u.w = num_hit ; 

   cfg.q3.u.x = num_record ; 
   cfg.q3.u.y = num_rec ; 
   cfg.q3.u.z = num_seq ; 
   cfg.q3.u.w = 0 ;
}


SEVENT_METHOD void sevent::zero()
{
    num_genstep = 0 ; 
    num_seed  = 0 ; 
    num_photon = 0 ; 
    num_record = 0 ; 
    num_rec = 0 ; 
    num_seq = 0 ; 
    num_hit = 0 ; 
    num_simtrace = 0 ; 

    genstep = nullptr ; 
    seed = nullptr ; 
    photon = nullptr ; 
    record = nullptr ; 
    rec = nullptr ; 
    seq = nullptr ; 
    hit = nullptr ; 
    simtrace = nullptr ; 
    
}
#endif 


/**
sevent::add_rec
----------------

Populates compressed "srec& r" from "const sphoton& p" using the domains
and copies into evt->rec array using the (idx,bounce) slot. 

**/


SEVENT_METHOD void  sevent::add_rec( srec& r, unsigned idx, unsigned bounce, const sphoton& p )
{
    r.set_position(     p.pos,  center_extent ); 
    r.set_time(         p.time, time_domain ); 
    r.set_polarization( p.pol ); 
    r.set_wavelength(   p.wavelength, wavelength_domain ); 
    // flags ?

    rec[max_rec*idx+bounce] = r ;      
}

/**
sevent::add_simtrace
----------------------

NB simtrace "photon" *a* is very different from real ones

TODO: rename the simtrace output array from photon to simtrace for clarity


a.q0.f
    prd.q0.f normal, distance, aka "isect" 

a.q1
    intersect position from pos+t*dir, 0.

a.q2
    initial pos, tmin

a.q3 
    initial dir, prd.identity


**/

SEVENT_METHOD void sevent::add_simtrace( unsigned idx, const quad4& p, const quad2* prd, float tmin )
{
    float t = prd->distance() ; 
    quad4 a ;  

    a.q0.f  = prd->q0.f ; 

    a.q1.f.x = p.q0.f.x + t*p.q1.f.x ; 
    a.q1.f.y = p.q0.f.y + t*p.q1.f.y ; 
    a.q1.f.z = p.q0.f.z + t*p.q1.f.z ; 
    a.q1.i.w = 0.f ;  

    a.q2.f.x = p.q0.f.x ; 
    a.q2.f.y = p.q0.f.y ; 
    a.q2.f.z = p.q0.f.z ; 
    a.q2.u.w = prd->boundary() ; // was tmin, but expecting bnd from CSGOptiXSimtraceTest.py:Photons

    a.q3.f.x = p.q1.f.x ;
    a.q3.f.y = p.q1.f.y ;
    a.q3.f.z = p.q1.f.z ;
    a.q3.u.w = prd->identity() ;  // identity from __closesthit__ch (( prim_idx & 0xffff ) << 16 ) | ( instance_id & 0xffff ) 

    simtrace[idx] = a ;
}




