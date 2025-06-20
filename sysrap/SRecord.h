#pragma once
/**
SRecord.h
==============

Used from SGLFW_Event.h

**/


#include "spath.h"
#include "scuda.h"
#include "NP.hh"


struct SRecord
{
    static constexpr const char* RPOS_SPEC = "4,GL_FLOAT,GL_FALSE,64,0,false";

    NP* record;
    int record_first ;
    int record_count ;  // all step points across all photon

    float4 mn = {} ;
    float4 mx = {} ;
    float4 ce = {} ;

    static SRecord* Load(const char* path, const char* _slice=nullptr );
    SRecord(NP* record);
    void init() ;

    const float* get_mn() const ;
    const float* get_mx() const ;
    const float* get_ce() const ;

    const float get_t0() const ;
    const float get_t1() const ;

    std::string desc() const ;
};

/**
SRecord::Load
------------------

Two forms of slice selection are handled.

1. "where" selection, eg pick photon records with y coordinate
   of the first step point less than a value::

     "[:,0,0,1] < -1.0"

2. "indexSlice" selection, eg every 1000th or the first 10::

     "[::1000]"
     "[:10]"


The indexSlice form uses partial loading of items from files
to enable working with very large record files, eg 66GB.

The _slice can be specified via envvar with eg "$AFOLD_RECORD_SLICE"

**/

inline SRecord* SRecord::Load(const char* _path, const char* _slice )
{
    const char* path = spath::Resolve(_path);
    bool looks_unresolved = spath::LooksUnresolved(path, _path);

    if(looks_unresolved)
    {
        std::cout
            << "SRecord::Load"
            << " FAILED : DUE TO MISSING ENVVAR\n"
            << " _path [" << ( _path ? _path : "-" ) << "]\n"
            << " path ["  << (  path ?  path : "-" ) << "]\n"
            << " looks_unresolved " << ( looks_unresolved ? "YES" : "NO " )
            << "\n"
            ;
        return nullptr ;
    }

    NP* a = nullptr ;
    if(NP::LooksLikeWhereSelection(_slice))
    {
        a = NP::LoadThenSlice<float>(path, _slice);
    }
    else
    {
        a = NP::LoadSlice(path, _slice);
    }
    return new SRecord(a);
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

    int item_stride = 4 ;
    int item_offset = 0 ;

    record->minmax2D_reshaped<4,float>(&mn.x, &mx.x, item_stride, item_offset );
    // temporarily 2D array with item: 4-element space-time points
    // HMM: with sparse "post" cloud this might miss the action
    // by trying to see everything ?

    ce = scuda::center_extent( mn, mx );
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
        << "]SRecord.desc\n"
        ;
    std::string str = ss.str() ;
    return str ;
}

