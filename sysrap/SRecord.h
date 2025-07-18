#pragma once
/**
SRecord.h
==============

Used from SGLFW_Evt.h

**/

#include "NP.hh"


#include "ssys.h"
#include "spath.h"
#include "sphoton.h"
#include "sseq_record.h"
#include "sstr.h"
#include "scuda.h"



struct SRecord
{
    static constexpr const char* NAME = "record.npy" ;
    static constexpr const char* RPOS_SPEC = "4,GL_FLOAT,GL_FALSE,64,0,false";
    static constexpr const char* _level = "SRecord__level" ;
    static int level ;

    NP* record;
    int record_first ;
    int record_count ;  // all step points across all photon

    float4 mn = {} ;
    float4 mx = {} ;
    float4 ce = {} ;

    static NP*      LoadArray( const char* _fold, const char* _slice );
    static SRecord* Load(      const char* _fold, const char* _slice=nullptr );

    SRecord(NP* record);
    void init() ;

    const float* get_mn() const ;
    const float* get_mx() const ;
    const float* get_ce() const ;

    const float get_t0() const ;
    const float get_t1() const ;
    std::string desc() const ;

    void getPhotonAtTime( std::vector<sphoton>* pp, std::vector<quad4>* qq, const char* iprt ) const ;
    static bool FindInterpolatedPhotonFromRecordAtTime( sphoton& p, const std::vector<sphoton>& point,  float t);

    NP*  getPhotonAtTime(   const char* iprt ) const;
    NP*  getSimtraceAtTime( const char* iprt ) const;


};


int SRecord::level = ssys::getenvint(_level,0) ;


/**
SRecord::LoadArray
-------------------

Several forms of slice selection are handled.

The slice argument supports envvar tokens like "$AFOLD_RECORD_SLICE"
that yield a selection string or direct such strings.

0. seqhis history string, eg::

    TO BT BT BT BT BR BT BT BT BT BT BT SC BT BT BT BT SD

1. "where" selection, eg pick photon records with y coordinate
   of the first step point less than a value::

     "[:,0,0,1] < -1.0"

2. "indexSlice" selection, eg every 1000th or the first 10.
   This uses partial loading of items from files which enables working
   with very large record files, eg 66GB. Example slices::

     "[::1000]"
     "[:10]"

3. path to index array eg "/tmp/w54.npy" : create that file from ipython with::

    In [39]: w54
    Out[39]: array([71464, 71485, 71663, 71678, 71690, 71780, 71782, 71798, 71799, 71803, 71810, 71914, 71918, 71919, 71975, 71978, 72019, 72026, 72032, 73149, 73212, 73344, 73372])
    In [40]: np.save("/tmp/w54.npy", w54)


**/

inline NP* SRecord::LoadArray(const char* _fold, const char* _slice )
{
    const char* path = spath::Resolve(_fold, NAME);
    bool looks_unresolved = spath::LooksUnresolved(path, _fold);
    if(looks_unresolved)
    {
        std::cout
            << "SRecord::LoadArray"
            << " FAILED : DUE TO MISSING ENVVAR\n"
            << " _fold [" << ( _fold ? _fold : "-" ) << "]\n"
            << " path ["  << (  path ?  path : "-" ) << "]\n"
            << " looks_unresolved " << ( looks_unresolved ? "YES" : "NO " )
            << "\n"
            ;
        return nullptr ;
    }

    NP* a = nullptr ;

    if( _slice == nullptr )
    {
        a = NP::Load(path);
        a->set_meta<std::string>("SRecord__LoadArray", "NP::Load");
    }
    else if(sseq_record::LooksLikeRecordSeqSelection(_slice)) // can be indirect via envvar, once resolved look for starting with "CK/SI/TO"
    {
        a = sseq_record::LoadRecordSeqSelection(_fold, _slice);
        a->set_meta<std::string>("SRecord__LoadArray_METHOD", "sseq_record::LoadRecordSeqSelection");
        a->set_meta<std::string>("SRecord__LoadArray_SLICE", _slice );
    }
    else if(NP::LooksLikeWhereSelection(_slice)) // can be indirect via envvar, once resolved look for "<" or ">"
    {
        a = NP::LoadThenSlice<float>(path, _slice);
        a->set_meta<std::string>("SRecord__LoadArray_METHOD", "NP::LoadThenSlice");
        a->set_meta<std::string>("SRecord__LoadArray_SLICE", _slice );
    }
    else    // eg "[0:100]" OR "/tmp/w54.npy" OR "/tmp/w54.npy[0:10]"
    {
        a = NP::LoadSlice(path, _slice);
        a->set_meta<std::string>("SRecord__LoadArray_METHOD", "NP::LoadSlice");
        a->set_meta<std::string>("SRecord__LoadArray_SLICE", _slice );
    }
    return a ;
}


inline SRecord* SRecord::Load(const char* _fold, const char* _slice )
{
    NP* _record = LoadArray(_fold, _slice);
    return _record ? new SRecord(_record) : nullptr ;
}




inline SRecord::SRecord(NP* _record)
    :
    record(_record),
    record_first(0),
    record_count(0)
{
    init();
}

/**
SRecord::init
-------------------

Expected shape of record array like (10000, 10, 4, 4)

**/


inline void SRecord::init()
{
    assert(record->shape.size() == 4);
    bool is_compressed = record->ebyte == 2 ;
    assert( is_compressed == false );

    record_first = 0 ;
    record_count = record->shape[0]*record->shape[1] ;   // all step points across all photon

    bool skip_flagmask_zero = true ;
    sphoton::MinMaxPost(&mn.x, &mx.x, record, skip_flagmask_zero );
    ce = scuda::center_extent( mn, mx );

    record->set_meta<float>("x0", mn.x );
    record->set_meta<float>("x1", mx.x );

    record->set_meta<float>("y0", mn.y );
    record->set_meta<float>("y1", mx.y );

    record->set_meta<float>("z0", mn.z );
    record->set_meta<float>("z1", mx.z );

    record->set_meta<float>("t0", mn.w );
    record->set_meta<float>("t1", mx.w );

    record->set_meta<float>("cx", ce.x );
    record->set_meta<float>("cy", ce.y );
    record->set_meta<float>("cz", ce.z );
    record->set_meta<float>("ce", ce.w );
}



inline const float* SRecord::get_mn() const
{
    return &mn.x ;
}
inline const float* SRecord::get_mx() const
{
    return &mx.x ;
}
inline const float* SRecord::get_ce() const
{
    return &ce.x ;
}

inline const float SRecord::get_t0() const
{
    return mn.w ;
}
inline const float SRecord::get_t1() const
{
    return mx.w ;
}


inline std::string SRecord::desc() const
{
    const char* lpath = record ? record->lpath.c_str() : nullptr ;

    std::stringstream ss ;
    ss
        << "[SRecord.desc\n"
        << " lpath [" << ( lpath ? lpath : "-" ) << "]\n"
        << std::setw(20) << " mn " << mn
        << std::endl
        << std::setw(20) << " mx " << mx
        << std::endl
        << std::setw(20) << " ce " << ce
        << std::endl
        << std::setw(20) << " record.sstr " << record->sstr()
        << std::endl
        << std::setw(20) << " record_first " << record_first
        << std::endl
        << std::setw(20) << " record_count " << record_count
        << std::endl
        << ( record ? record->descMeta() : "-" )
        << std::endl
        << "]SRecord.desc\n"
        ;
    std::string str = ss.str() ;
    return str ;
}

/**
SRecord::getPhotonAtTime
--------------------------

Provide photons corresponding to input simulation time(s)
by interpolation between positions in the record array. Momentum and
polarization are taken from the first of the straddling step points.
When not straddling the photon is skipped, corresponding
to it not being alive at the simulation time provided.

Config example from ~/o/CSGOptiX/cxt_precision.sh with multiple times::

     export OPTICKS_INPUT_PHOTON=$AFOLD/record.npy
     export OPTICKS_INPUT_PHOTON_RECORD_SLICE="TO BT BT BT SA"
     export OPTICKS_INPUT_PHOTON_RECORD_TIME=[10:80:10]  # ns

Python analysis is simplified by arranging RECORD_SLICE
and RECORD_TIME such that all photons result in the same number of
staddles by ensuring that the lowest and highest times are
during the lifetime of the photon. Going further, arranging times
to all be inbetween two step points eg "TO->BT" makes the analysis
of intersects from different distances as simple as possible.


Q : Simtrace input uses q0.f and q1.f for pos and mom, like sphoton.h
    Simtrace output uses q2.f and q3.f for origin pos and mom.
    Why different layout from input and output, as
    demonstrated here and in sevent::add_simtrace ?

A : Perhaps to match sphoton for input. When get chance rationalize to use one layout.
    This is not urgent, and will require widespread changes across cpp and py.

**/

inline void SRecord::getPhotonAtTime( std::vector<sphoton>* pp, std::vector<quad4>* qq, const char* _iprt ) const
{
    NP* iprt = NP::ARange_FromString<float>(_iprt);
    const float* tt = iprt ? iprt->cvalues<float>() : nullptr ;

    int ni = record->shape[0] ;
    int nj = iprt ? iprt->num_items() : 0 ;
    int nl = record->shape[1] ;
    int item_bytes = record->item_bytes() ; // encompasses multiple step point sphoton

    assert( nj > 0 && tt );
    int count0 = -1 ;

    for(int i=0 ; i < ni ; i++)
    {
        // populate step point vector
        std::vector<sphoton> point(nl) ;
        memcpy( point.data(), record->bytes() + i*item_bytes,  item_bytes );

        int count = 0 ;
        for(int j=0 ; j < nj ; j++) // time loop
        {
            float t = tt[j] ;
            sphoton p = {} ;
            bool found = FindInterpolatedPhotonFromRecordAtTime( p, point, t);
            if(!found) continue ;

            count += 1 ;   // count times at which an interpolated photon was found

            quad4 q = {} ;
            p.populate_simtrace_input(q);

            if(pp) pp->push_back(p);
            if(qq) qq->push_back(q);
        }

        bool consistent_count = count0 == -1 || count0 == count ;
        if( count0 == -1 ) count0 = count ;

        if(!consistent_count || level > 0) std::cout
            << "SRecord::getPhotonAtTime"
            << " i " << std::setw(6) << i
            << " count0 " << count0
            << " count " << count
            << " consistent_count " << ( consistent_count ? "YES" : "NO " )
            << ( consistent_count ? "" : "(adjust time range to achieve consistency)" )
            << " level " << level
            << "\n"
            ;
    }   // i: photon loop


    if(level > 0) std::cout
         << "SRecord::getPhotonAtTime"
         << " [" << _level << "] " << level
         << " _iprt " << ( _iprt ? _iprt : "-" )
         << " iprt " << ( iprt ? iprt->sstr() : "-" )
         << " ni " << ni
         << " nj " << nj
         << " nl " << nl
         << " tt " << ( tt ? "YES" : "NO " )
         << " record " << ( record ? record->sstr() : "-" )
         << " count0 " << count0
         << " pp.size " << ( pp ? int(pp->size()) : -1 )
         << " qq.size " << ( qq ? int(qq->size()) : -1 )
         << "\n"
         ;

}



/**
SRecord::FindInterpolatedPhotonFromRecordAtTime
------------------------------------------------

Look for record step points with times that straddle the provided time *t*.
If found interpolate the straddling step points to populate *p*.
When a straddle is found returns true.

**/


inline bool SRecord::FindInterpolatedPhotonFromRecordAtTime( sphoton& p, const std::vector<sphoton>& point,  float t)
{
    int nl = point.size();
    for(int l=1 ; l < nl ; l++)
    {
        const sphoton& p0 = point[l-1] ;
        const sphoton& p1 = point[l] ;
        bool straddle = p0.time <= t  && t < p1.time ;

        if(level > 3) std::cout
            << "SRecord::FindInterpolatedPhotonFromRecordAtTime"
            << " nl " << nl
            << " l " << std::setw(3) << l
            << " t " << std::fixed << std::setw(7) << std::setprecision(3) << t
            << " p0.time " << std::fixed << std::setw(7) << std::setprecision(3) << p0.time
            << " p1.time " << std::fixed << std::setw(7) << std::setprecision(3) << p1.time
            << " straddle " << ( straddle ? "YES" : "NO " )
            << "\n"
            ;

        if(straddle)
        {
            p = p0 ;  // (pol,mom) from p0
            float frac = (t - p0.time)/(p1.time - p0.time) ;
            p.pos = lerp( p0.pos, p1.pos, frac ) ; // interpolated position
            p.time = t ;
            return true ;
        }
    }
    return false ;
}


inline NP* SRecord::getPhotonAtTime( const char* iprt ) const
{
    std::vector<sphoton> vpp ;
    getPhotonAtTime(&vpp, nullptr, iprt);
    int num_pp = vpp.size();
    NP* pp = num_pp > 0 ?  NPX::ArrayFromVec<float,sphoton>(vpp, 4, 4 ) : nullptr ;
    return pp ;
}

inline NP* SRecord::getSimtraceAtTime( const char* iprt ) const
{
    std::vector<quad4> vqq ;
    getPhotonAtTime(nullptr, &vqq, iprt);
    int num_qq = vqq.size();
    NP* qq = num_qq > 0 ?  NPX::ArrayFromVec<float,quad4>(vqq, 4, 4 ) : nullptr ;
    return qq ;
}


