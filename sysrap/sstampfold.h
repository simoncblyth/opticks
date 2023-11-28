#pragma once
/**
sstampfold.h
===============

Presents sstamp.h timestamp metadata from a collection of SEvt NPFold

**/

#include "NPFold.h"

struct sstampfold
{
    static constexpr const char* COUNT_KEY = "subcount" ; 
    static constexpr const char* STAMP_KEY = "substamp" ; 
    static constexpr const char* DELTA_KEY = "delta_substamp" ; 
    static constexpr const char* LABELS_KEY = "labels" ; 

    const NPFold* af ; 
    const char* symbol ; 

    const NP* sc ;  
    const NP* st ;  
    const NP* dt ;
    const NP* la ;

    const std::vector<std::string>* rows ; 
    const std::vector<std::string>* cols ; 
 
    sstampfold( const NPFold* af, const char* symbol ); 
    std::string desc() const ; 
};


inline sstampfold::sstampfold( const NPFold* af_, const char* symbol_ ) 
    :
    af(af_),
    symbol(symbol_ ? strdup(symbol_) : nullptr),
    sc(af ? af->get(COUNT_KEY) : nullptr),
    st(af ? af->get(STAMP_KEY) : nullptr),
    dt(af ? af->get(DELTA_KEY) : nullptr),
    la(af ? af->get(LABELS_KEY) : nullptr),
    rows(st ? &(st->names) : nullptr), 
    cols(la ? &(la->names) : nullptr)
{
}

inline std::string sstampfold::desc() const 
{
    std::stringstream ss ; 
    ss 
       << "[sstampfold::desc " << ( symbol ? symbol : "-" )  << std::endl 
       << ( sc ? sc->descTable<int>(8) : "-" ) << std::endl 
       << std::endl 
       << ( dt ? dt->descTable_<int64_t>(8,cols,rows) : "-" ) << std::endl 
       << "]sstampfold::desc " << ( symbol ? symbol : "-" )  << std::endl 
       ;
    std::string str =  ss.str() ; 
    return str ; 
} 
    


