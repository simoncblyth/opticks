#pragma once
/**
SRecordInfo.h
==============

Used from SGLFW_Event.h


**/


#include "scuda.h"
#include "NP.hh"


struct SRecordInfo
{
    static constexpr const char* RPOS_SPEC = "4,GL_FLOAT,GL_FALSE,64,0,false";

    NP* record;
    int record_first ;
    int record_count ;  // all step points across all photon

    float4 mn = {} ;
    float4 mx = {} ;
    float4 ce = {} ;

    static SRecordInfo* Load(const char* path, const char* _slice=nullptr );
    SRecordInfo(NP* record);
    void init() ;

    const float* get_mn() const ;
    const float* get_mx() const ;
    const float* get_ce() const ;

    const float get_t0() const ;
    const float get_t1() const ;

    std::string desc() const ;
};

/**
SRecordInfo::Load
------------------

**/

inline SRecordInfo* SRecordInfo::Load(const char* _path, const char* _slice )
{
    NP* a = NP::LoadSlice<float>(_path, _slice); 
    return new SRecordInfo(a); 
}


inline SRecordInfo::SRecordInfo(NP* _record)
    :
    record(_record),
    record_first(0),
    record_count(0)
{
    init();
}

/**
SRecordInfo::init
-------------------

Expected shape of record array like (10000, 10, 4, 4)

**/


inline void SRecordInfo::init()
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






inline const float* SRecordInfo::get_mn() const
{
    return &mn.x ;
}
inline const float* SRecordInfo::get_mx() const
{
    return &mx.x ;
}
inline const float* SRecordInfo::get_ce() const
{
    return &ce.x ;
}

inline const float SRecordInfo::get_t0() const
{
    return mn.w ;
}
inline const float SRecordInfo::get_t1() const
{
    return mx.w ;
}


inline std::string SRecordInfo::desc() const
{
    const char* lpath = record ? record->lpath.c_str() : nullptr ; 

    std::stringstream ss ;
    ss
        << "[SRecordInfo.desc\n"
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
        << "]SRecordInfo.desc\n"
        ;
    std::string str = ss.str() ;
    return str ;
}

