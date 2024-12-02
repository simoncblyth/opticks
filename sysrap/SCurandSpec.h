#pragma once
/**
SCurandSpec.h
==============


**/

#include <iostream>
#include "SCurandChunk.h"
#include "SYSRAP_API_EXPORT.hh"

struct SYSRAP_API SCurandSpec
{
    typedef unsigned long long ULL ;

    static constexpr const char* SEED_OFFSET = "0:0" ;  
    static constexpr const char SEED_OFFSET_DELIM  = ':' ; 

    static constexpr const char* CHUNKSIZES = "10x1M,9x10M,5x20M" ;  
    static constexpr const char SIZEGROUP_DELIM  = ',' ; 
    static constexpr const char MUL_NUM_DELIM = 'x' ; 

    static void ParseSeedOffset(ULL& seed, ULL& offset, const char* _spec=nullptr ); 
    static void ParseChunkSizes(     std::vector<ULL>& size, const char* _spec=nullptr ); 
    static std::string Desc(const std::vector<ULL>& size); 
};


inline void SCurandSpec::ParseSeedOffset( ULL& seed, ULL& offset, const char* _spec )
{
    const char* spec = _spec ? _spec : SEED_OFFSET ; 

    std::vector<std::string> elem ; 
    sstr::Split(_spec, SEED_OFFSET_DELIM, elem ); 

    seed   = elem.size() > 0 ? std::atoll( elem[0].c_str() ) : 0ull ;     
    offset = elem.size() > 0 ? std::atoll( elem[1].c_str() ) : 0ull ;     
}

inline void SCurandSpec::ParseChunkSizes(std::vector<ULL>& size, const char* _spec )
{
    const char* spec = _spec ? _spec : CHUNKSIZES ; 

    std::vector<std::string> group ; 
    sstr::Split(spec, SIZEGROUP_DELIM, group ); 

    int num_group = group.size(); 
    for(int i=0 ; i < num_group ; i++)
    {
        const char* g = group[i].c_str(); 
    
        std::vector<std::string> mul_num ; 
        sstr::Split(g, MUL_NUM_DELIM, mul_num); 
        assert(  mul_num.size() == 2 ); 
        const char* MUL = mul_num[0].c_str();  
        const char* NUM = mul_num[1].c_str();  

        ULL mul = std::atoll(MUL); 
        ULL num = SCurandChunk::ParseNum(NUM); 

        for(ULL j=0 ; j < mul ; j++) size.push_back(num); 
    }
}

inline std::string SCurandSpec::Desc(const std::vector<ULL>& _size)
{
    std::stringstream ss; 
    ss << "SCurandSpec::Desc" << " size.size " << _size.size() << "[" ; 
    ULL tot = 0 ; 
    for(int i=0 ; i < int(_size.size()) ; i++ ) 
    {
        ULL num = _size[i]; 
        tot += num ; 
        ss << SCurandChunk::FormatNum(num) << " " ; 
    }
    ss << "] tot:" << SCurandChunk::FormatNum(tot)  ;  
    std::string str = ss.str(); 
    return str ;   
}



