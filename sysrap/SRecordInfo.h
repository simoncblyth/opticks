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

    bool dump ;
    NP* record;
    int record_first ;
    int record_count ;  // all step points across all photon

    float4 mn = {} ;
    float4 mx = {} ;
    float4 ce = {} ;

    static SRecordInfo* Load(const char* path);
    SRecordInfo(NP* record);
    void init() ;

    const float* get_mn() const ;
    const float* get_mx() const ;
    const float* get_ce() const ;
    const float get_t0() const ;
    const float get_t1() const ;
    const int   get_num_time() const ;

    std::string desc() const ;
};


inline SRecordInfo* SRecordInfo::Load(const char* _path)
{
    const char* path = spath::Resolve(_path);
    if(!spath::Exists(path)) return nullptr ;

    NP* _a = NP::Load(path) ;
    NP* a = _a ? NP::MakeNarrowIfWide(_a) : nullptr  ;

    SRecordInfo* s = new SRecordInfo(a) ;
    return s ;
}


inline SRecordInfo::SRecordInfo(NP* _record)
    :
    dump(ssys::getenvbool("SRecordInfo_dump")),
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
    float t0 = ssys::getenvfloat("T0", mn.w );
    return t0;
}
inline const float SRecordInfo::get_t1() const
{
    float t1 = ssys::getenvfloat("T1", mx.w );
    return t1;
}

/**
SRecordInfo::get_num_time
--------------------------

Number of render loop time bumps to go from t0 to t1

**/


inline const int SRecordInfo::get_num_time() const
{
    int nt = ssys::getenvint("NT", 5000 );
    return nt ;
}


inline std::string SRecordInfo::desc() const
{
    std::stringstream ss ;
    ss
        << "[SRecordInfo.desc\n"
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

