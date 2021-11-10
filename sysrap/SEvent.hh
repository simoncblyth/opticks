#pragma once

struct NP ; 
struct quad6 ; 
template <typename T> struct Tran ;

#include <vector>
#include "plog/Severity.h"
#include "SYSRAP_API_EXPORT.hh"

struct SYSRAP_API SEvent
{
    static const plog::Severity LEVEL ; 

    static NP* MakeGensteps(const std::vector<quad6>& gs ); 
    static void StandardizeCEGS(        const float4& ce,       std::vector<int>& cegs, float gridscale );
    static NP* MakeCenterExtentGensteps(const float4& ce, const std::vector<int>& cegs, float gridscale, const Tran<double>* geotran ) ;
    static NP* MakeCountGensteps();
    static NP* MakeCountGensteps(const std::vector<int>& photon_counts_per_genstep);
    static void GenerateCenterExtentGenstepsPhotons( std::vector<quad4>& pp, const NP* gsa ); 

};



