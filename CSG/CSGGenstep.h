#pragma once

/**
CSGGenstep.h
===============

**/

#include <vector>
#include "plog/Severity.h"

struct float4 ; 
struct qat4 ; 
template <typename T> struct Tran ;  

#include "CSG_API_EXPORT.hh"


struct NP ; 

struct CSG_API CSGGenstep
{
    CSGGenstep( const CSGFoundry* foundry );  
    void create(const char* moi, bool ce_offset );
    void generate_photons_cpu();
    void save(const char* basedir) const ; 

    // below are "private"

    static const plog::Severity LEVEL ; 
    void init(); 
    void locate(const char* moi); 
    void override_locate() ; 
    void configure_grid() ; 

    const CSGFoundry* foundry ; 
    float gridscale ;  
    const char* moi ; 
    int midx ; 
    int mord ; 
    int iidx ; 
    float4 ce ;
    qat4*  qt ;  
    Tran<double>* geotran ;
    std::vector<int> cegs ; 

    NP* gs ; 
    NP* pp ; 


}; 
