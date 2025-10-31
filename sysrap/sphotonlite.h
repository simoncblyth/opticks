#pragma once

/**
sphotonlite.hh
===============

+----+-----------------------+----------------+-----------------------+----------------+------------------------------+
| q  |      x                |      y         |     z                 |      w         |  notes                       |
+====+=======================+================+=======================+================+==============================+
|    | u:hitcount_identity   |  f:time        | u:lposcost_lposfphi   | u:flagmask     |                              |
| q0 |                       |                |                       |                |                              |
|    | off:0, 2              | off:4          | off:8,10              | off:12         |  off:byte offsets            |
+----+-----------------------+----------------+-----------------------+----------------+------------------------------+


**/


#if defined(__CUDACC__) || defined(__CUDABE__)
#    define SPHOTONLITE_METHOD __host__ __device__ __forceinline__
#else
#    define SPHOTONLITE_METHOD inline
#endif


#include <cstdint>
#if defined(__CUDACC__) || defined(__CUDABE__)
#else
   #include <iostream>
   #include <iomanip>
   #include <sstream>
   #include <cstring>
   #include <cassert>
   #include "OpticksPhoton.h"
   #include "OpticksPhoton.hh"
   #include "NP.hh"
#endif

struct sphotonlite
{
    uint32_t  hitcount_identity ;
    float     time ;
    uint32_t  lposcost_lposfphi ;
    uint32_t  flagmask ;

    // HMM: actually 16 bits would be enough, or even 1 bit with fixed hitmask

    SPHOTONLITE_METHOD void init(unsigned _identity, float _time, unsigned _flagmask);
    SPHOTONLITE_METHOD void set_lpos(float lposcost, float lposfphi);
    SPHOTONLITE_METHOD void set_hitcount_identity( unsigned hitcount, unsigned identity );

    SPHOTONLITE_METHOD unsigned hitcount() const { return hitcount_identity >> 16 ; }
    SPHOTONLITE_METHOD unsigned identity() const { return hitcount_identity & 0xffffu ; }

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
    SPHOTONLITE_METHOD void get_lpos(float& lposcost, float& lposfphi) const ;
    SPHOTONLITE_METHOD std::string lpos_() const ;
    SPHOTONLITE_METHOD std::string flagmask_() const ;
    SPHOTONLITE_METHOD std::string desc() const ;
    SPHOTONLITE_METHOD static NP* make_demoarray(int num_photon);
#endif

};

/**
sphotonlite::init
-------------------

Example::

   sphotonlite l ;
   l.init( p.identity, p.time, p.flagmask );

**/


inline void sphotonlite::init(unsigned _identity, float _time, unsigned _flagmask)
{
    hitcount_identity = ( 0x1u << 16 ) | ( _identity & 0xFFFFu ) ; // hitcount starts as one
    time = _time ;
    lposcost_lposfphi = 0u ;
    flagmask = _flagmask ;
}


inline void sphotonlite::set_hitcount_identity( unsigned hitcount, unsigned identity )
{
    hitcount_identity = (( hitcount & 0xFFFFu ) << 16 ) | ( identity & 0xFFFFu ) ; // hitcount starts as one
}

/**
sphotonlite::set_lpos
----------------------

lposcost
     cosine of the intersect local position polar angle,
     within range 0.->1. [assumes only front hemi-sphere is sensitive]

lposfphi
     fraction of azimuthal angle in radians of local intersect position,
     scaled by 2pi to be within range 0.->1.

**/


inline void sphotonlite::set_lpos(float lposcost, float lposfphi)
{
#if defined(__CUDACC__) || defined(__CUDABE__)
#else
    assert( lposcost >= 0.f && lposcost <= 1.f );
    assert( lposfphi >= 0.f && lposfphi <= 1.f );
#endif
    lposcost_lposfphi =
                        ((uint16_t)(lposcost * 0xffffu + 0.5f) << 16) |
                        ((uint16_t)(lposfphi * 0xffffu + 0.5f) <<  0)
                      ;
}


#if defined(__CUDACC__) || defined(__CUDABE__)
#else
inline void sphotonlite::get_lpos(float& lposcost, float& lposfphi) const
{
    lposcost = float(lposcost_lposfphi >> 16    ) / 0xffffu ;
    lposfphi = float(lposcost_lposfphi & 0xffffu) / 0xffffu ;

    assert( lposcost >= 0.f && lposcost <= 1.f );
    assert( lposfphi >= 0.f && lposfphi <= 1.f );
}

inline std::string sphotonlite::lpos_() const
{
    float lposcost, lposfphi ;
    get_lpos(lposcost, lposfphi);

    std::stringstream ss ;
    ss
        << " cost " << std::setw(7) << std::fixed << std::setprecision(5) << lposcost
        << " fphi " << std::setw(7) << std::fixed << std::setprecision(5) << lposfphi
        ;
    std::string str = ss.str() ;
    return str ;
}


inline std::string sphotonlite::desc() const
{
    std::stringstream ss ;
    ss << "[sphotonlite:"
       << " t " <<  std::setw(8) << time
       << " id " <<  std::setw(8) << identity()
       << " hc " <<  std::setw(8) << hitcount()
       << " lp " <<  lpos_()
       << " fm " <<  flagmask_()
       << "]"
       ;

    std::string str = ss.str() ;
    return str ;
}



/**

In [2]: t.demoarray.view(np.uint32)[:,0,0] >> 16   ## hitcount
Out[2]: array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint32)

In [3]: t.demoarray.view(np.uint32)[:,0,0] & 0xffff   ## identity
Out[3]: array([   0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000], dtype=uint32)


In [7]: (t.demoarray.view(np.uint32)[:,0,2] >> 16)/0xffff
Out[7]: array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

n [6]: (t.demoarray.view(np.uint32)[:,0,2] & 0xffff)/0xffff
Out[6]: array([0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6])


**/



inline NP* sphotonlite::make_demoarray(int num_photon) // static
{
    NP* l = NP::Make<float>(num_photon, 1, 4);
    sphotonlite* ll  = (sphotonlite*)l->bytes();
    for(int i=0 ; i < num_photon ; i++)
    {
        ll[i].set_hitcount_identity( i, i*1000 );
        ll[i].flagmask = EFFICIENCY_COLLECT | SURFACE_DETECT ;
        ll[i].time = i*0.1f ;
        ll[i].set_lpos( 0.5f, 0.6f );
    }
    return l ;
}

inline std::string sphotonlite::flagmask_() const
{
    return OpticksPhoton::FlagMaskLabel(flagmask) ;
}


#endif



