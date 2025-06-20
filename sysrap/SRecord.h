#pragma once
/**
SRecord.h
==============

Used from SGLFW_Event.h

**/

#include "NP.hh"

#include "spath.h"
#include "sseq_record.h"
#include "sstr.h"
#include "scuda.h"



struct SRecord
{
    static constexpr const char* NAME = "record.npy" ;
    static constexpr const char* RPOS_SPEC = "4,GL_FLOAT,GL_FALSE,64,0,false";

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
};

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

2. "indexSlice" selection, eg every 1000th or the first 10::

     "[::1000]"
     "[:10]"


The indexSlice form uses partial loading of items from files
to enable working with very large record files, eg 66GB.


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
    }
    else if(sseq_record::LooksLikeRecordSeqSelection(_slice))
    {
        a = sseq_record::LoadRecordSeqSelection(_fold, _slice);
    }
    else if(NP::LooksLikeWhereSelection(_slice))
    {
        a = NP::LoadThenSlice<float>(path, _slice);
    }
    else
    {
        a = NP::LoadSlice(path, _slice);
    }
    return a ;
}


inline SRecord* SRecord::Load(const char* _fold, const char* _slice )
{
    NP* _record = LoadArray(_fold, _slice);
    return new SRecord(_record);
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

