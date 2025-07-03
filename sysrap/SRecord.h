#pragma once
/**
SRecord.h
==============

Used from SGLFW_Evt.h

TODO: add API to provide photons corresponding to an input simulation time
by interpolation between positions in the record array. Momentum and
polarization can be taken from the first of the straddling step points.
When not straddling need to skip that photon, corresponding
to it not being alive at the simulation time provided.

**/

#include "NP.hh"

#include "spath.h"
#include "sphoton.h"
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
        a->set_meta<std::string>("SRecord__LoadArray", "NP::Load");
    }
    else if(sseq_record::LooksLikeRecordSeqSelection(_slice))
    {
        a = sseq_record::LoadRecordSeqSelection(_fold, _slice);
        a->set_meta<std::string>("SRecord__LoadArray_METHOD", "sseq_record::LoadRecordSeqSelection");
        a->set_meta<std::string>("SRecord__LoadArray_SLICE", _slice );
    }
    else if(NP::LooksLikeWhereSelection(_slice))
    {
        a = NP::LoadThenSlice<float>(path, _slice);
        a->set_meta<std::string>("SRecord__LoadArray_METHOD", "NP::LoadThenSlice");
        a->set_meta<std::string>("SRecord__LoadArray_SLICE", _slice );
    }
    else
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

    sphoton::MinMaxPost(&mn.x, &mx.x, record );
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

