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
   struct sphotonlite_selector ;
#endif


struct sphotonlite
{
#if defined(__CUDACC__) || defined(__CUDABE__)
#else
    static constexpr const char* NAME = "sphotonlite" ;
#endif

    uint32_t  hitcount_identity ; // hi16:hitcount, lo16:identity
    float     time ;
    uint32_t  lposcost_lposfphi ; // hi16:lposcost, lo16:lposfphi
    uint32_t  flagmask ;

    SPHOTONLITE_METHOD void init(unsigned _identity, float _time, unsigned _flagmask);
    SPHOTONLITE_METHOD void set_lpos(float lposcost, float lposfphi);

    SPHOTONLITE_METHOD void set_hitcount_identity( unsigned hitcount, unsigned identity );
    SPHOTONLITE_METHOD void set_hitcount( unsigned hitcount );
    SPHOTONLITE_METHOD void set_identity( unsigned identity );

    SPHOTONLITE_METHOD unsigned hitcount() const { return hitcount_identity >> 16 ; }
    SPHOTONLITE_METHOD unsigned identity() const { return hitcount_identity & 0xffffu ; }

    struct key_functor
    {
        float    tw;
        SPHOTONLITE_METHOD uint64_t operator()(const sphotonlite& p) const
        {
            unsigned id = p.identity() ;
            unsigned bucket = static_cast<unsigned>(p.time / tw);
            return (uint64_t(id) << 48) | uint64_t(bucket);
        }
    };

    struct reduce_op
    {
        SPHOTONLITE_METHOD sphotonlite operator()(const sphotonlite& a, const sphotonlite& b) const
        {
            sphotonlite r = a;
            r.time = fminf(a.time, b.time);
            r.flagmask |= b.flagmask;
            unsigned hc = a.hitcount() + b.hitcount();
            r.set_hitcount_identity(hc, a.identity());
            return r;
        }
    };

    // this requires any bit match, not all bit like the below sphotonlite_selector
    struct select_pred
    {
        unsigned mask;
        SPHOTONLITE_METHOD bool operator()(const sphotonlite& p) const { return (p.flagmask & mask) != 0; }
    };





#if defined(__CUDACC__) || defined(__CUDABE__)
#else

    bool operator==(const sphotonlite& other) const noexcept
    {
        return hitcount_identity  == other.hitcount_identity &&
               time               == other.time              &&
               lposcost_lposfphi  == other.lposcost_lposfphi &&
               flagmask           == other.flagmask ;
    }
    bool operator!=(const sphotonlite& other) const noexcept
    {
        return !(*this == other);
    }


    SPHOTONLITE_METHOD void get_lpos(float& lposcost, float& lposfphi) const ;
    SPHOTONLITE_METHOD void get_lpos_theta_phi(float& lpos_theta, float& lpos_phi) const ;

    SPHOTONLITE_METHOD std::string lpos_() const ;


    SPHOTONLITE_METHOD std::string flagmask_() const ;
    SPHOTONLITE_METHOD std::string desc() const ;

    SPHOTONLITE_METHOD static NP* zeros(    size_t num_photon);
    SPHOTONLITE_METHOD static bool expected( const NP* l );

    SPHOTONLITE_METHOD static NP* demoarray(size_t num_photon);
    SPHOTONLITE_METHOD static NP* select( const NP* photonlite, const sphotonlite_selector* photonlite_selector );
    SPHOTONLITE_METHOD static void loadbin( std::vector<sphotonlite>& photonlite, const char* path );

    SPHOTONLITE_METHOD static std::string Desc(const NP* a, size_t edge=20);
    SPHOTONLITE_METHOD static NP*         MockupForMergeTest(size_t ni);


    SPHOTONLITE_METHOD static std::string desc_diff( const sphotonlite* a, const sphotonlite* b, size_t ni );

#endif

};

/**
sphotonlite_selector
--------------------

All bits that are set within the *hitmask* are required to be set within the *flagmask* for a match
This functor binds to the below predicate signature, where S=sphotonlite::

     std::function<bool(const S*)>

Instances of this functor are used on CPU via the below sphotonlite::select
and on GPU from QEvt::gatherHitLite_

**/

struct sphotonlite_selector
{
    uint32_t hitmask ;
    sphotonlite_selector(uint32_t hitmask_) : hitmask(hitmask_) {};
    SPHOTONLITE_METHOD bool operator() (const sphotonlite& p) const { return ( p.flagmask  & hitmask ) == hitmask  ; }
    SPHOTONLITE_METHOD bool operator() (const sphotonlite* p) const { return ( p->flagmask & hitmask ) == hitmask  ; }
};






/**
sphotonlite::init
-------------------

Example, CSGOptiX/CSGOptiX7.cu::


    457     if( evt->photonlite )
    458     {
    459         sphotonlite l ;
    460         l.init( ctx.p.identity, ctx.p.time, ctx.p.flagmask );
    461         l.set_lpos(prd->lposcost(), prd->lposfphi() );
    462         evt->photonlite[idx] = l ;  // *idx* (not *photon_idx*) as needs to go from zero for photons from a slice of genstep array
    463     }



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
    hitcount_identity = (( hitcount & 0xFFFFu ) << 16 ) | ( identity & 0xFFFFu ) ;
}
inline void sphotonlite::set_hitcount( unsigned hc )
{
    hitcount_identity = ( hitcount_identity & 0x0000ffffu ) | (( 0x0000ffffu & hc ) << 16);
}
inline void sphotonlite::set_identity( unsigned id )
{
    hitcount_identity = ( hitcount_identity & 0xffff0000u ) | (( 0x0000ffffu & id ) <<  0);
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


inline void sphotonlite::get_lpos_theta_phi(float& lpos_theta, float& lpos_phi) const
{
    float lposcost = float(lposcost_lposfphi >> 16    ) / 0xffffu ;
    float lposfphi = float(lposcost_lposfphi & 0xffffu) / 0xffffu ;

    lpos_theta = std::acos( lposcost );  // HMM: could store fthe instead of cost ?
    lpos_phi = phi_from_fphi( lposfphi );

}

inline std::string sphotonlite::lpos_() const
{
    float lposcost, lposfphi ;
    get_lpos(lposcost, lposfphi);

    float lpos_theta, lpos_phi ;
    get_lpos_theta_phi(lpos_theta, lpos_phi);


    std::stringstream ss ;
    ss
        << " cost " << std::setw(7) << std::fixed << std::setprecision(5) << lposcost
        << " theta " << std::setw(7) << std::fixed << std::setprecision(5) << lpos_theta
        << " fphi " << std::setw(7) << std::fixed << std::setprecision(5) << lposfphi
        << " phi " << std::setw(7) << std::fixed << std::setprecision(5) << lpos_phi


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



    In [2]: t.demoarray.view(np.uint32)[:,0] >> 16   ## hitcount
    Out[2]: array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint32)

    In [3]: t.demoarray.view(np.uint32)[:,0] & 0xffff   ## identity
    Out[3]: array([   0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000], dtype=uint32)


    In [7]: (t.demoarray.view(np.uint32)[:,2] >> 16)/0xffff
    Out[7]: array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

    n [6]: (t.demoarray.view(np.uint32)[:,2] & 0xffff)/0xffff
    Out[6]: array([0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6])


**/


inline NP* sphotonlite::zeros(size_t num_photon) // static
{
    NP* l = NP::Make<uint32_t>(num_photon, 4);
    return l ;
}

inline bool sphotonlite::expected( const NP* l ) // static
{
    bool expected_type = l && l->uifc == 'u' && l->ebyte == 4 ;
    bool expected_shape =  l && l->has_shape(-1, 4) ;
    bool expected_arr = expected_type && expected_shape ;

    if(!expected_arr) std::cerr
        << "sphotonlite::expected"
        << " expected_arr " << ( expected_arr ? "YES" : "NO " )
        << " expected_type " << ( expected_type ? "YES" : "NO " )
        << " expected_shape " << ( expected_shape ? "YES" : "NO " )
        << " l.sstr " << ( l ? l->sstr() : "-" )
        ;
    return expected_arr ;
}


inline NP* sphotonlite::demoarray(size_t num_photon) // static
{
    NP* l = zeros(num_photon);
    sphotonlite* ll  = (sphotonlite*)l->bytes();
    for(size_t i=0 ; i < num_photon ; i++)
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

inline NP* sphotonlite::select( const NP* photonlite, const sphotonlite_selector* photonlite_selector )
{
    NP* hitlite = photonlite ? photonlite->copy_if<uint32_t, sphotonlite>(*photonlite_selector) : nullptr ;
    return hitlite ;
}


inline void sphotonlite::loadbin( std::vector<sphotonlite>& photonlite, const char* path )
{
    // 1. open path with pointer at-end "ate", use tellg to determine number of hits

    std::ifstream f(path, std::ios::binary | std::ios::ate);
    assert(f && "sphotonlite::loadbin failed to open path");
    size_t bytes = f.tellg();
    size_t n = bytes / sizeof(sphotonlite);
    f.seekg(0);

    // 2. read into photonlite vector
    photonlite.resize(n);
    f.read((char*)photonlite.data(), bytes);
}


inline std::string sphotonlite::Desc(const NP* a, size_t edge) // static
{
    std::stringstream ss ;
    ss << "[sphotonlite::Desc a.sstr " << ( a ? a->sstr() : "-" ) << "\n" ;
    size_t ni = a->num_items();
    sphotonlite* ll = (sphotonlite*)a->bytes();
    for(size_t i=0 ; i < ni ; i++)
    {
        if(i < edge || i > (ni - edge)) ss << std::setw(6) << i << " : " << ll[i].desc() << "\n" ;
        else if( i == edge )  ss << std::setw(6) << "" << " : " << "..." << "\n" ;
    }

    std::string str = ss.str() ;
    return str ;
}

SPHOTONLITE_METHOD NP* sphotonlite::MockupForMergeTest(size_t ni) // static
{
    NP* a = NP::Make<unsigned>(ni, 4);
    sphotonlite* aa = (sphotonlite*)a->bytes();
    for(size_t i=0 ; i < ni ; i++)
    {
        sphotonlite& p = aa[i];
        p.time = float( i % 100 )*0.1f ;
        p.set_hitcount_identity( 1, i % 100 );
        p.flagmask = i % 5 == 0 ? EFFICIENCY_COLLECT : EFFICIENCY_CULL ;
    }
    return a ;
}


inline std::string sphotonlite::desc_diff( const sphotonlite* a, const sphotonlite* b, size_t ni )
{
    size_t tot_diff = 0 ;
    size_t tot_same = 0 ;

    std::stringstream ss ;
    ss << "[sphotonlite::desc_diff\n"
       << " ni " << ni
       << " a " << ( a ? "YES" : "NO " )
       << " b " << ( b ? "YES" : "NO " )
       << "\n"
       ;

    if( a != nullptr && b != nullptr )
    {
        for(size_t i = 0; i < ni; ++i)
        {
            if (a[i] != b[i])
            {
                tot_diff +=1 ;
                ss << "Index " << i << " differs:\n"
                          << "  a: " << a[i].desc() << "\n"
                          << "  b: " << b[i].desc() << "\n";
            }
            else
            {
                tot_same +=1 ;
            }
        }
    }

    ss << "]sphotonlite::desc_diff\n"
       << " tot_diff " << tot_diff
       << " tot_same " << tot_same
       << "\n"
       ;

    std::string str = ss.str() ;
    return str ;
}








#endif



