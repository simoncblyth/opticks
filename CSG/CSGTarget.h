#pragma once

#include "plog/Severity.h"

struct CSGFoundry ; 

struct CSGTarget
{
    static const plog::Severity LEVEL ; 
    const CSGFoundry* foundry ; 

    CSGTarget( const CSGFoundry* foundry );  

    int getCenterExtent(float4& ce, int midx, int mord, int iidx=-1, qat4* q=nullptr) const ;
    int getLocalCenterExtent( float4& lce, int midx, int mord) const ;
    int getGlobalCenterExtent(float4& gce, int midx, int mord, int iidx, qat4* q) const ; 

    int getTransform(qat4& q, int midx, int mord, int iidx) const  ; 
    const qat4* getInstanceTransform(int midx, int mord, int iidx) const ; 


};


