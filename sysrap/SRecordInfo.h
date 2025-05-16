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

    static SRecordInfo* Load(const char* record_path);
    SRecordInfo();

    void init_minmax2D() ;
    void load(const char* record_path) ;

    const float* get_mn() const ;
    const float* get_mx() const ;
    const float* get_ce() const ;
    const float get_t0() const ;
    const float get_t1() const ;
    const float get_ts() const ;

    std::string desc() const ;
};


inline SRecordInfo* SRecordInfo::Load(const char* record_path)
{
    SRecordInfo* s = new SRecordInfo ;
    s->load(record_path);
    return s ;
}

inline SRecordInfo::SRecordInfo()
    :
    dump(ssys::getenvbool("SRecordInfo_dump")),
    record(nullptr),
    record_first(0),
    record_count(0)
{
}

/**
SRecordInfo::load
-------------------

Expected shape like (10000, 10, 4, 4)

**/


inline void SRecordInfo::load(const char* path)
{
    NP* _a = NP::Load(path) ;
    record = NP::MakeNarrowIfWide(_a) ;

    if(record==nullptr) std::cout 
         << "SRecordInfo::load FAILED for  [" << ( path ? path : "-" )  << "]\n" 
         << " CREATE EXAMPLE WITH [TEST=make_record_array ~/o/sysrap/tests/sphoton_test.sh] \n" 
         ;
    assert(record);

    assert(record->shape.size() == 4);
    bool is_compressed = record->ebyte == 2 ;
    assert( is_compressed == false );

    record_first = 0 ;
    record_count = record->shape[0]*record->shape[1] ;   // all step points across all photon
}


inline void SRecordInfo::init_minmax2D(){

    static const int N = 4 ;

    int item_stride = 4 ;
    int item_offset = 0 ;

    record->minmax2D_reshaped<N,float>(&mn.x, &mx.x, item_stride, item_offset );
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

inline const float SRecordInfo::get_ts() const
{
    float ts = ssys::getenvfloat("TS", 5000. );
    return ts;
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

