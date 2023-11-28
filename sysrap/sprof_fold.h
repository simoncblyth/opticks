#pragma once
/**
sprof_fold.h
===============

Presents sprof.h profile metadata from a collection of SEvt NPFold

**/

#include "NPFold.h"

struct sprof_fold
{
    static constexpr const char* STAMP_KEY = "subprofile" ; 
    static constexpr const char* DELTA_KEY = "delta_subprofile" ; 
    static constexpr const char* LABELS_KEY = "labels" ; 

    const NPFold* af ; 
    const char* symbol ; 

    const NP* st ;  
    const NP* dt ;
    const NP* la ;
    const std::vector<std::string>* rows ; 
    const std::vector<std::string>* cols ; 
 
    sprof_fold( const NPFold* af, const char* symbol ); 
    std::string desc() const ; 
};


inline sprof_fold::sprof_fold( const NPFold* af_, const char* symbol_ ) 
    :
    af(af_),
    symbol(symbol_ ? strdup(symbol_) : nullptr),
    st(af ? af->get(STAMP_KEY) : nullptr),
    dt(af ? af->get(DELTA_KEY) : nullptr),
    la(af ? af->get(LABELS_KEY) : nullptr),
    rows(st ? &(st->names) : nullptr), 
    cols(la ? &(la->names) : nullptr)
{
}

inline std::string sprof_fold::desc() const 
{
    std::stringstream ss ; 
    ss 
       << "[sprof_fold::desc " << ( symbol ? symbol : "-" )  << std::endl 
       << ( dt ? dt->descTable_<int64_t>(8,cols,rows) : "-" ) << std::endl 
       << "]sprof_fold::desc " << ( symbol ? symbol : "-" )  << std::endl 
       ;
    std::string str =  ss.str() ; 
    return str ; 
} 
    


