#pragma once
/**
SFrameGenstep.hh
==================

TODO: contrast with SCenterExtentGenstep and replace all used of that with this 

* principal advantage of SFrameGenstep over SCenterExtentGenstep is the sframe.h/sframe.py 
  providing a central object available both from C++ and python 



::

    epsilon:opticks blyth$ opticks-fl SFrameGenstep
    ./ana/framegensteps.py
    ./CSGOptiX/cxs_Hama.sh
    ./CSGOptiX/tests/CSGOptiXSimtraceTest.cc
    ./CSG/tests/CSGFoundry_MakeCenterExtentGensteps_Test.cc
    ./extg4/X4Simtrace.cc
    ./sysrap/SFrameGenstep.hh
    ./sysrap/CMakeLists.txt
    ./sysrap/SCenterExtentFrame.h
    ./sysrap/tests/SFrameGenstep_MakeCenterExtentGensteps_Test.sh
    ./sysrap/tests/CMakeLists.txt
    ./sysrap/tests/SFrameGenstep_MakeCenterExtentGensteps_Test.cc
    ./sysrap/tests/SEventTest.cc
    ./sysrap/SFrameGenstep.cc
    ./sysrap/SCenterExtentGenstep.cc
    ./sysrap/sframe.h
    ./sysrap/SEvt.cc
    ./sysrap/SCenterExtentGenstep.hh
    ./g4cx/G4CXOpticks.cc
    ./g4cx/gxt.sh

    epsilon:opticks blyth$ opticks-fl SCenterExtentGenstep
    ./CSG/CSGGeometry.cc
    ./CSG/CSGQuery.cc
    ./CSG/tests/CSGIntersectSolidTest.py
    ./CSG/CSGGenstep.h
    ./extg4/X4Intersect.hh
    ./extg4/X4Intersect.cc
    ./sysrap/SFrameGenstep.hh
    ./sysrap/CMakeLists.txt
    ./sysrap/SCenterExtentGenstep.py
    ./sysrap/tests/CMakeLists.txt
    ./sysrap/tests/SEventTest.cc
    ./sysrap/tests/SCenterExtentGenstepTest.cc
    ./sysrap/SCenterExtentGenstep.cc
    ./sysrap/SCenterExtentGenstep.sh
    ./sysrap/SCenterExtentGenstep.hh

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

    static const char* CEGS_XY ; 
    static const char* CEGS_XZ ;  // default 

    static NP* MakeCenterExtentGensteps(sframe& fr); 
    static NP* MakeCenterExtentGensteps(const float4& ce, const std::vector<int>& cegs, float gridscale, const Tran<double>* geotran, const std::vector<float3>& ce_offset, bool ce_scale ) ;

    static void StandardizeCEGS( std::vector<int>& cegs );
    static void GetBoundingBox( float3& mn, float3& mx, const float4& ce, const std::vector<int>& standardized_cegs, float gridscale, const float3& ce_offset ) ; 

    static void GenerateCenterExtentGenstepsPhotons( std::vector<quad4>& pp, const NP* gsa, float gridscale ); 
    static NP* GenerateCenterExtentGenstepsPhotons_( const NP* gsa, float gridscale ) ; 

    static void GenerateSimtracePhotons( std::vector<quad4>& simtrace, const std::vector<quad6>& genstep ); 


    static void SetGridPlaneDirection( float4& dir, int gridaxes, double cosPhi, double sinPhi, double cosTheta, double sinTheta ); 
};




