#pragma once
/**
SFrameGenstep.hh
==================

TODO: contrast with SCenterExtentGenstep and replace all used of that with this 

**/

#include <vector>
#include "plog/Severity.h"

struct float4 ; 
struct NP ; 
template <typename T> struct Tran ;

#include "SYSRAP_API_EXPORT.hh"

struct SYSRAP_API SFrameGenstep
{
    static const plog::Severity LEVEL ; 

    static NP* MakeCenterExtentGensteps(sframe& fr); 
    static NP* MakeCenterExtentGensteps(const float4& ce, const std::vector<int>& cegs, float gridscale, const Tran<double>* geotran, bool ce_offset, bool ce_scale ) ;

    static void StandardizeCEGS( const float4& ce,       std::vector<int>& cegs, float gridscale );
    static void GetBoundingBox( float3& mn, float3& mx, const float4& ce, const std::vector<int>& standardized_cegs, float gridscale, bool ce_offset ) ; 

    static void GenerateCenterExtentGenstepsPhotons( std::vector<quad4>& pp, const NP* gsa, float gridscale ); 
    static NP* GenerateCenterExtentGenstepsPhotons_( const NP* gsa, float gridscale ) ; 
    static void SetGridPlaneDirection( float4& dir, int gridaxes, double cosPhi, double sinPhi, double cosTheta, double sinTheta ); 
};




