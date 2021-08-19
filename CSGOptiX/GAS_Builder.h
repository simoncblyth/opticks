#pragma once

#include <vector>
#include "GAS.h"
#include "CSGPrim.h"
#include "BI.h"

/**
GAS_Builder
=============

* original approach GAS:BI:AABB  1:N:N   : one BI for every layer of the compound GAS
* 2nd try approach  GAS:BI:AABB  1:1:N  : only one BI for all layers of compound GAS

**/

struct GAS_Builder
{
    static void Build(     GAS& gas, const CSGPrimSpec& psd );
    static void Build_11N( GAS& gas, const CSGPrimSpec& psd );
    static BI MakeCustomPrimitivesBI_11N(const CSGPrimSpec& psd);
    static void DumpAABB(                const float* aabb, unsigned num_aabb, unsigned stride_in_bytes ) ; 
    static void BoilerPlate(GAS& gas);  
};


