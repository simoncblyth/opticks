#pragma once

#include <vector>

#include "plog/Severity.h"
#include "GAS.h"
#include "CSGPrim.h"
#include "BI.h"

/**
GAS_Builder
=============

Only used from SBT.cc 

* 1st try approach failed GAS:BI:AABB  1:N:N   : one BI for every layer of the compound GAS
* 2nd try approach  GAS:BI:AABB  1:1:N  : only one BI for all layers of compound GAS

**/

struct GAS_Builder
{
    static const plog::Severity LEVEL ; 
    static void Build(     GAS& gas, const SCSGPrimSpec& psd );

    template<typename T>
    static CUdeviceptr DevicePointerCast( const T* d_ptr ); 

    static void Build_11N( GAS& gas, const SCSGPrimSpec& psd );
    static BI MakeCustomPrimitivesBI_11N(const SCSGPrimSpec& psd);

    static void DumpAABB(                const float* aabb, unsigned num_aabb, unsigned stride_in_bytes ) ; 
    static void BoilerPlate(GAS& gas);  
};


