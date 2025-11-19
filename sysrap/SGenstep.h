#pragma once
/**
SGenstep.h : genstep static utilities
========================================

TODO: incorporate all/most of the genstep methods from SEvent.{hh,cc}
      here to avoid near duplicated impls


Used by:

sysrap/SGenerate.h
   GetGencode

sysrap/SFrameGenstep.cc
   GenstepID, ConfigureGenstep, MakeArray, GridAxes, GridAxesName

sysrap/SCenterExtentGenstep.cc
   [ON WAY OUT] GridAxes, GridAxesName

sysrap/SEvent.cc
   ConfigureGenstep, MakeArray used by SEvent::MakeCountGenstep

qudarap/QEvt.cc
   Check, Desc, GetGencode used by QEvt::setGenstepUpload


+---------+----------+-------------+
| qty     | NumPy    |  Note       |
+=========+==========+=============+
| q0.i.x  | [:,0,0]  |  gencode    |
+---------+----------+-------------+
| q0.i.w  | [:,0,3]  | num_photon  |
+---------+----------+-------------+


**/

#include <string>
#include <vector>
#include <limits>
#include <csignal>

struct quad6 ;
struct NP ;
struct sslice ;

#include "sxyz.h"
#include "SYSRAP_API_EXPORT.hh"

struct SYSRAP_API SGenstep
{
    static constexpr const int64_t MAX_SLOT_PER_SLICE = std::numeric_limits<int32_t>::max();
    static constexpr const char* XYZ_ = "XYZ" ;
    static constexpr const char* YZ_  = "YZ" ;
    static constexpr const char* XZ_  = "XZ" ;
    static constexpr const char* XY_  = "XY" ;

    static const char* GridAxesName( int gridaxes );
    static int GridAxes(int nx, int ny, int nz);
    static unsigned GenstepID( int ix, int iy, int iz, int iw=0 ) ;

    static void ConfigureGenstep( quad6& gs,  int gencode, int gridaxes, int gsid, int photons_per_genstep );
    static int GetGencode( const quad6& gs ) ;
    static int64_t GetNumPhoton( const quad6& gs ) ;
    static void SetNumPhoton( quad6& gs, int64_t num );

    static const quad6& GetGenstep(const NP* gs, unsigned gs_idx );
    static int64_t GetGenstepSlices(std::vector<sslice>& slice, const NP* gs, int64_t max_slot );
    static void CheckGenstepSlices(const std::vector<sslice>& slice, const NP* gs, int64_t max_slot );

    static int GetGencode( const quad6* qq, unsigned gs_idx  );
    static int GetGencode(    const NP* gs, unsigned gs_idx  );

    static int64_t GetNumPhoton( const quad6* qq, unsigned gs_idx  );
    static int64_t GetNumPhoton( const NP* gs, unsigned gs_idx  );

    static int64_t GetPhotonTotal( const NP* gs );
    static int64_t GetPhotonTotal( const NP* gs, int gs_start, int gs_stop  );

    static void Check(const NP* gs);
    static NP* MakeArray(const std::vector<quad6>& gs );

    template<typename T>
    static NP* MakeTestArray(const std::vector<T>& num_ph);

    template<typename T>
    static std::string DescNum(const std::vector<T>& num_ph);

    static std::string Desc(const NP* gs, int edgeitems);

};


#include <cassert>

#include "scuda.h"
#include "squad.h"
#include "sphoton.h"
#include "sc4u.h"
#include "sslice.h"

#include "OpticksGenstep.h"
#include "NP.hh"



inline const char* SGenstep::GridAxesName( int gridaxes)  // static
{
    const char* s = nullptr ;
    switch( gridaxes )
    {
        case XYZ: s = XYZ_ ; break ;
        case YZ:  s = YZ_  ; break ;
        case XZ:  s = XZ_  ; break ;
        case XY:  s = XY_  ; break ;
    }
    return s ;
}

/**
SGenstep::GridAxes
--------------------------

The nx:ny:nz dimensions of the grid are used to classify it into::

    YZ
    XZ
    XY
    XYZ

For a planar grid one of the nx:ny:nz grid dimensions is zero.
XYZ is a catch all for non-planar grids.

**/

inline int SGenstep::GridAxes(int nx, int ny, int nz)  // static
{
    int gridaxes = XYZ ;
    if( nx == 0 && ny > 0 && nz > 0 )
    {
        gridaxes = YZ ;
    }
    else if( nx > 0 && ny == 0 && nz > 0 )
    {
        gridaxes = XZ ;
    }
    else if( nx > 0 && ny > 0 && nz == 0 )
    {
        gridaxes = XY ;
    }
    return gridaxes ;
}

/**
SGenstep::GenstepID
-------------------

Pack four signed integers (assumed to be in char range -128 to 127)
into a 32 bit unsigtned char using C4U uniform.

**/

inline unsigned SGenstep::GenstepID( int ix, int iy, int iz, int iw )
{
    C4U gsid ;   // sc4u.h

    gsid.c4.x = ix ;
    gsid.c4.y = iy ;
    gsid.c4.z = iz ;
    gsid.c4.w = iw ;

    return gsid.u ;
}




/**
SGenstep::ConfigureGenstep
---------------------------

TODO: pack enums to make room for a photon_offset

* gsid was MOVED from (1,3) to (0,2) when changing genstep to carry transform

**/

inline void SGenstep::ConfigureGenstep( quad6& gs,  int gencode, int gridaxes, int gsid, int photons_per_genstep )
{
    assert( gencode == OpticksGenstep_TORCH || gencode == OpticksGenstep_FRAME );
    assert( gridaxes == XYZ ||  gridaxes == YZ || gridaxes == XZ || gridaxes == XY );

    gs.q0.i.x = gencode ;
    gs.q0.i.y = gridaxes ;
    gs.q0.u.z = gsid ;
    gs.q0.i.w = photons_per_genstep ;
}

inline int SGenstep::GetGencode( const quad6& gs )
{
    return gs.q0.i.x ;
}
inline int64_t SGenstep::GetNumPhoton( const quad6& gs )
{
    return gs.q0.i.w ;
}





inline void SGenstep::SetNumPhoton( quad6& gs, int64_t num )
{
    bool num_allowed = num <= MAX_SLOT_PER_SLICE ;
    assert( num_allowed );
    if(!num_allowed) std::raise(SIGINT);

    gs.q0.i.w = num ;
}




inline const quad6& SGenstep::GetGenstep(const NP* gs, unsigned gs_idx )
{
    Check(gs);
    assert( int(gs_idx) < gs->shape[0] );
    quad6* qq = (quad6*)gs->bytes() ;
    const quad6& q = qq[gs_idx] ;
    return q ;
}


/**
SGenstep::GetGenstepSlices
--------------------------

Populates the genstep slice vector with gs[start:stop] slices
that do not exceed max_photon for the total number of photons.

The third argument *max_slot* was formerly incorrectly named *max_photon*,
which is wrong because its the VRAM constrained *max_slot* which is relevant
to how the work needed to be sliced, not the arbitrary and very large
*max_photon* which is not VRAM constrained.

**/

inline int64_t SGenstep::GetGenstepSlices(std::vector<sslice>& slice, const NP* gs, int64_t max_slot )
{
    Check(gs);
    int num_gs = gs ? gs->shape[0] : 0 ;
    const quad6* qq = gs ? (quad6*)gs->cvalues<float>() : nullptr ;

    sslice sl = {};

    sl.gs_start = 0 ;   // gs index starting the slice
    sl.gs_stop = 0 ;    // gs index stopping the slice, ie one beyond the last index, python style gs[start:stop]
    sl.ph_offset = 0 ;  // total photons before the start of this slice
    sl.ph_count = 0 ;   // total photons within this slice


    int64_t tot_ph = 0 ;

    bool extend_slice = false ;

    for(int i=0 ; i < num_gs ; i++)
    {
        bool last_gs = i == num_gs - 1;

        const quad6& q = qq[i] ;
        int64_t num_ph = GetNumPhoton(q);

        bool slice_too_big = num_ph > MAX_SLOT_PER_SLICE ;
        if(slice_too_big ) std::cerr
            << "SGenstep::GetGenstepSlices FATAL "
            << " slice_too_big " << ( slice_too_big ? "YES" : "NO " )
            << " i " << i
            << " num_gs " << num_gs
            << " num_ph " << num_ph
            << " MAX_SLOT_PER_SLICE " << MAX_SLOT_PER_SLICE
            << "\n"
            ;
        assert( !slice_too_big );


        tot_ph += num_ph ;

        int64_t cand = sl.ph_count + num_ph ;
        extend_slice = cand <= max_slot ;

        if(0) std::cout
            << "SGenstep::GetGenstepSlices"
            << " i " << std::setw(4) << i
            << " sl.ph_count " << std::setw(7) << sl.ph_count
            << " num_ph " << std::setw(7) << num_ph
            << " max_slot " << std::setw(7) << max_slot
            << " cand  " << std::setw(7) << num_ph
            << " extend_slice " << ( extend_slice ? "YES" : "NO " )
            << " last_gs " << ( last_gs ? "YES" : "NO " )
            << "\n"
            ;

        if(extend_slice)  // genstep fits within limit
        {
            sl.gs_stop = i+1 ;
            sl.ph_count += num_ph ;
        }
        else  // genstep does not fit, so close slice and start another
        {
            sl.gs_stop = i ;
            slice.push_back(sl) ;

            sl.ph_count = 0 ;          // back to zero
            sl.ph_count += num_ph ;
            sl.gs_start = i ;
            sl.gs_stop = i+1 ;
        }
        if(last_gs) slice.push_back(sl) ;
    }
    sslice::SetOffset(slice);

    CheckGenstepSlices(slice, gs, max_slot);

    return tot_ph ;
}

inline void SGenstep::CheckGenstepSlices(const std::vector<sslice>& slice, const NP* gs, int64_t max_slot )
{
    int64_t gs_tot = GetPhotonTotal(gs);
    int64_t sl_tot = sslice::TotalPhoton(slice);

    bool tot_consistent = gs_tot == sl_tot ;

    if(!tot_consistent) std::cerr
       << "SGenstep::CheckGenstepSlices"
       << " gs_tot " << gs_tot
       << " sl_tot " << sl_tot
       << " tot_consistent " << ( tot_consistent ? "YES" : "NO " )
       << "\n"
       ;
    assert(tot_consistent);


    const quad6* qq = gs ? (quad6*)gs->cvalues<float>() : nullptr ;
    int num_sl = slice.size();
    for(int i=0 ; i < num_sl ; i++)
    {
        const sslice& sl = slice[i];

        int sl_ph_count = 0 ;
        for(int j=sl.gs_start ; j < sl.gs_stop ; j++ )
        {
            const quad6& q = qq[j] ;
            int64_t num_ph = GetNumPhoton(q);
            sl_ph_count += num_ph ;
        }
        bool ph_count_expected = sl_ph_count == sl.ph_count ;

        if(!ph_count_expected) std::cerr
            << "SGenstep::CheckGenstepSlices"
            << " i " << i
            << " sl.ph_count " << sl.ph_count
            << " sl_ph_count " << sl_ph_count
            << "\n"
            ;

        assert( ph_count_expected );
        assert( sl_ph_count <= max_slot ); // total for slice should never exceed max_slot
    }
}



inline int SGenstep::GetGencode( const quad6* qq, unsigned gs_idx  ) // static
{
    if( qq == nullptr ) return OpticksGenstep_INVALID ;
    const quad6& q = qq[gs_idx] ;
    return GetGencode(q) ;
}
inline int SGenstep::GetGencode( const NP* gs, unsigned gs_idx  ) // static
{
    const quad6& q = GetGenstep(gs, gs_idx);
    return GetGencode(q) ;
}


inline int64_t SGenstep::GetNumPhoton( const quad6* qq, unsigned gs_idx  ) // static
{
    const quad6& q = qq[gs_idx] ;
    return GetNumPhoton(q) ;
}

inline int64_t SGenstep::GetNumPhoton( const NP* gs, unsigned gs_idx  ) // static
{
    if( gs == nullptr) return 0 ;
    const quad6& q = GetGenstep(gs, gs_idx);
    return GetNumPhoton(q) ;
}


inline int64_t SGenstep::GetPhotonTotal( const NP* gs ) // static
{
    int num_gs = gs ? gs->shape[0] : 0 ;
    return GetPhotonTotal( gs, 0, num_gs );
}

/**
Genstep::GetPhotonTotal
------------------------

Returns total photon in genstep slice gs[start:stop]

**/

inline int64_t SGenstep::GetPhotonTotal( const NP* gs, int gs_start, int gs_stop ) // static
{
    int num_gs = gs ? gs->shape[0] : 0 ;
    assert( gs_start <= num_gs );
    assert( gs_stop  <= num_gs );
    int64_t tot = 0 ;
    for(int i=gs_start ; i < gs_stop ; i++ ) tot += GetNumPhoton(gs, i );
    return tot ;
}





inline NP* SGenstep::MakeArray(const std::vector<quad6>& gs ) // static
{
    assert( gs.size() > 0);
    NP* a = NP::Make<float>( gs.size(), 6, 4 );
    a->read2<float>( (float*)gs.data() );
    Check(a);
    return a ;
}

/**
SGenstep::MakeTestArray
------------------------

Make test array filled with gensteps containing only num_photon

**/

template<typename T>
inline NP* SGenstep::MakeTestArray(const std::vector<T>& num_ph) // static
{
    int num_gs = num_ph.size();
    NP* gs = NP::Make<float>(num_gs, 6, 4);
    quad6* qq = (quad6*)gs->values<float>() ;
    for(int i=0 ; i < num_gs ; i++) SGenstep::SetNumPhoton(qq[i], num_ph[i]);
    Check(gs);
    return gs ;
}


template<typename T>
inline std::string SGenstep::DescNum(const std::vector<T>& num_ph)
{
    T tot = 0 ;
    std::stringstream ss ;
    ss << "SGenstep::DescNum\n" ;
    for(int i=0 ; i < int(num_ph.size()) ; i++ )
    {
        T num = num_ph[i] ;
        ss
            << std::setw(3) << i << " : "
            << " num : " << std::setw(7) << num
            << " tot : " << std::setw(7) << tot
            << "\n"
            ;

        tot += num ;
    }

    ss << " Grand total : " << tot << "\n\n" ;

    std::string str = ss.str() ;
    return str ;
}



inline void SGenstep::Check(const NP* gs)  // static
{
    if( gs == nullptr ) return ;
    assert( gs->uifc == 'f' && gs->ebyte == 4 );
    assert( gs->has_shape(-1, 6, 4) );
}

inline std::string SGenstep::Desc(const NP* gs, int edgeitems) // static
{
    int num_genstep = gs ? gs->shape[0] : 0 ;

    quad6* gs_v = gs ? (quad6*)gs->cvalues<float>() : nullptr ;
    std::stringstream ss ;
    ss << "SGenstep::DescGensteps num_genstep " << num_genstep << " (" ;

    int total = 0 ;
    for(int i=0 ; i < num_genstep ; i++)
    {
        const quad6& _gs = gs_v[i];
        unsigned gs_pho = _gs.q0.u.w  ;

        if( i < edgeitems || i > num_genstep - edgeitems ) ss << gs_pho << " " ;
        else if( i == edgeitems )  ss << "... " ;

        total += gs_pho ;
    }
    ss << ") total " << total  ;
    std::string s = ss.str();
    return s ;
}


