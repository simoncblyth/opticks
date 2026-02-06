#pragma once
/**
SFrameGenstep.hh
==================

TODO: contrast this with SCenterExtentGenstep and replace all use of that with this

* principal advantage of SFrameGenstep over SCenterExtentGenstep is the sframe.h/sframe.py
  providing a central object on which to hang metadata that is
  available both from C++ and python

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
    static constexpr const char* CEGS              = "CEGS" ;
    static constexpr const char* CEGS_RADIAL_RANGE = "CEGS_RADIAL_RANGE" ;
    static constexpr char CEGS_delim = ':' ;


    static const plog::Severity LEVEL ;
    static void CE_OFFSET(std::vector<float3>& ce_offset, const float4& ce ) ;
    static std::string Desc(const std::vector<float3>& ce_offset );
    static std::string Desc(const std::vector<int>& cegs );

    static void GetGridConfig_(std::vector<int>& cegs, const char* ekey, char delim, const char* fallback );
    static std::string GetGridConfig(std::vector<int>& cegs);


    static const char* CEGS_XY ;
    static const char* CEGS_XZ ;  // default

    static bool HasConfigEnv();

#ifdef WITH_OLD_FRAME
    static NP* MakeCenterExtentGenstep_FromFrame(sframe& fr);
#else
    static NP* MakeCenterExtentGenstep_FromFrame(sfr& fr);
#endif
    static NP* MakeCenterExtentGenstep_From_CE_geotran(const float4& ce, const std::vector<int>& cegs, float gridscale, const Tran<double>* geotran);


    static NP* Make_CEGS_NPY_Genstep( const NP* CEGS_NPY, const Tran<double>* geotran );

    static NP* MakeCenterExtentGenstep(
        const float4& ce,
        const std::vector<int>& cegs,
        float gridscale,
        const Tran<double>* geotran,
        const std::vector<float3>& ce_offset,
        bool ce_scale,
        std::vector<float>* cegs_radial_range ) ;



    static void StandardizeCEGS( std::vector<int>& cegs );
    static void GetBoundingBox(
        float3& mn,
        float3& mx,
        const float4& ce,
        const std::vector<int>& standardized_cegs,
        float gridscale,
        const float3& ce_offset ) ;

    static void GenerateCenterExtentGenstepPhotons(
        std::vector<quad4>& pp,
        const NP* gsa,
        float gridscale );

    static NP* GenerateCenterExtentGenstepPhotons_(
        const NP* gsa,
        float gridscale ) ;

    static void GenerateSimtracePhotons(
        std::vector<quad4>& simtrace,
        const std::vector<quad6>& genstep );

    static void SetGridPlaneDirection(
        float4& dir,
        int gridaxes,
        double cosPhi,
        double sinPhi,
        double cosTheta,
        double sinTheta );
};


