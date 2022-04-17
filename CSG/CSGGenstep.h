#pragma once

/**
CSGGenstep.h : Creator of CenterExtent "CE" Gensteps used by CSGOptiXSimulateTest
==================================================================================

Sensitive to envvars:

CEGS
   center-extent-genstep
   expect 4 or 7 ints delimited by colon nx:ny:nz:num_pho OR nx:px:ny:py:nz:py:num_pho 


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
    void create(const char* moi, bool ce_offset, bool ce_scale );
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
    qat4*  m2w ;  
    qat4*  w2m ;  
    Tran<double>* geotran ;
    std::vector<int> cegs ; 

    NP* gs ; 
    NP* pp ; 


}; 
