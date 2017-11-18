#pragma once

#include <string>

struct CG4Ctx ; 
struct CRecState ; 

#include "CRecorder.h"
#include "CFG4_API_EXPORT.hh"


struct CFG4_API CPhoton
{
    const CG4Ctx& _ctx ; 
    CRecState&    _state ; 

    unsigned _badflag ; 
    unsigned _slot_constrained ; 
    unsigned _material ; 
    uifchar4   _c4 ; 

    unsigned long long _seqhis ; 
    unsigned long long _seqmat ; 
    unsigned long long _mskhis ; 

    unsigned long long _his ; 
    unsigned long long _mat ; 
    unsigned long long _flag ; 

    unsigned long long _his_prior ; 
    unsigned long long _mat_prior ; 
    unsigned long long _flag_prior ; 


    //CPhoton(const CPhoton& other);
    CPhoton(const CG4Ctx& ctx, CRecState& state);

    void clear();

    void add(unsigned flag, unsigned  material);
    void increment_slot() ; 

    bool is_rewrite_slot() const  ;
    bool is_flag_done() const ;
    bool is_done() const ;
    bool is_hard_truncate() ;
    void scrub_mskhis( unsigned flag );

    std::string desc() const ; 
    std::string brief() const ; 


};
 
