#pragma once
/**
SFrameGenstep.hh
==================

TODO: contrast with SCenterExtentGenstep and replace all used of that with this 

**/

#include <vector>
#include "plog/Severity.h"

struct float3 ; 
struct float4 ; 
struct NP ; 
template <typename T> struct Tran ;

#include "SYSRAP_API_EXPORT.hh"

struct SYSRAP_API SFrameGenstep
{
    static const plog::Severity LEVEL ; 
    static void CE_OFFSET(std::vector<float3>& ce_offset, const float4& ce ) ; 
    static std::string Desc(const std::vector<float3>& ce_offset ); 
    static std::string Desc(const std::vector<int>& cegs ); 

    static void GetGridConfig(std::vector<int>& cegs,  const char* ekey, char delim, const char* fallback ); 
    static NP* MakeCenterExtentGensteps(sframe& fr); 
    static NP* MakeCenterExtentGensteps(const float4& ce, const std::vector<int>& cegs, float gridscale, const Tran<double>* geotran, const std::vector<float3>& ce_offset, bool ce_scale ) ;

    static void StandardizeCEGS( std::vector<int>& cegs );
    static void GetBoundingBox( float3& mn, float3& mx, const float4& ce, const std::vector<int>& standardized_cegs, float gridscale, const float3& ce_offset ) ; 

    static void GenerateCenterExtentGenstepsPhotons( std::vector<quad4>& pp, const NP* gsa, float gridscale ); 
    static NP* GenerateCenterExtentGenstepsPhotons_( const NP* gsa, float gridscale ) ; 

    static void GenerateSimtracePhotons( std::vector<quad4>& simtrace, const std::vector<quad6>& genstep ); 


    static void SetGridPlaneDirection( float4& dir, int gridaxes, double cosPhi, double sinPhi, double cosTheta, double sinTheta ); 
};




