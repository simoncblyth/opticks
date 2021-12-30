#pragma once

struct NP ; 
struct quad6 ; 
template <typename T> struct Tran ;

#include <vector>
#include "plog/Severity.h"
#include "SYSRAP_API_EXPORT.hh"

#include "sevent.h"

struct SYSRAP_API SEvent
{
    static const plog::Severity LEVEL ; 


    static const char* XYZ_ ; 
    static const char* YZ_  ; 
    static const char* XZ_  ; 
    static const char* XY_  ; 
    static const char* GridAxesName( int gridaxes ); 
    static int GridAxes(int nx, int ny, int nz); 

    static const char* DIMENSION_0_ ; 
    static const char* DIMENSION_1_ ; 
    static const char* DIMENSION_2_ ; 
    static const char* DIMENSION_3_ ;
 
    static const char* DirModeName( int dirmode ); 

    static NP* MakeGensteps(const std::vector<quad6>& gs ); 
    static void StandardizeCEGS(        const float4& ce,       std::vector<int>& cegs, float gridscale );
    static void GetBoundingBox( float3& mn, float3& mx, const float4& ce, const std::vector<int>& standardized_cegs, float gridscale, bool ce_offset ) ; 

    static NP* MakeCenterExtentGensteps(const float4& ce, const std::vector<int>& cegs, float gridscale, const Tran<double>* geotran, bool ce_offset ) ;
    static NP* MakeCountGensteps();
    static NP* MakeCountGensteps(const std::vector<int>& photon_counts_per_genstep);
    static void GenerateCenterExtentGenstepsPhotons( std::vector<quad4>& pp, const NP* gsa ); 
    static NP* GenerateCenterExtentGenstepsPhotons_( const NP* gsa ) ; 

    static void SetGridPlaneDirection( float4& dir, int gridaxes, float cosPhi, float sinPhi ); 

};



