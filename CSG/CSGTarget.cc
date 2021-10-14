#include "PLOG.hh"
#include "scuda.h"
#include "CSGFoundry.h"
#include "CSGTarget.h"

const plog::Severity CSGTarget::LEVEL = PLOG::EnvLevel("CSGTarget", "DEBUG" ); 

CSGTarget::CSGTarget( const CSGFoundry* foundry_ )
    :
    foundry(foundry_)
{
}

/**
CSGTarget::getCenterExtent
----------------------------

Used by CSGFoundry::getCenterExtent

*ce*
    center-extent float4 set when return code is zero
*midx*
    solid (aka mesh) index : identify the shape
*mord*
    solid (aka mesh) ordinal : pick between shapes when there are more than one, used with global(non-instanced) geometry 
*iidx*
    instance index, >-1 for global instance, -1 for local non-instanced 

**/

int CSGTarget::getCenterExtent(float4& ce, int midx, int mord, int iidx) const 
{
    if( iidx == -1 )
    {
        int lrc = getLocalCenterExtent(ce, midx, mord); 
        if(lrc != 0) return 1 ; 
    }
    else
    {
        int grc = getGlobalCenterExtent(ce, midx, mord, iidx);
        if(grc != 0) return 2 ;
    }
    return 0 ; 
}

/**
CSGTarget::getLocalCenterExtent
---------------------------------

Collects prim matching the *midx* and selects the *mord* ordinal one
from which to read the localCE 

**/

int CSGTarget::getLocalCenterExtent(float4& lce, int midx, int mord) const 
{
    std::vector<CSGPrim> prim ; 
    foundry->getMeshPrim(prim, midx );  
    bool mord_in_range = mord < prim.size() ; 

    LOG(info)  
        << " midx " << midx
        << " mord " << mord 
        << " prim.size " << prim.size()
        << " mord_in_range " << mord_in_range
        ;   

    if(!mord_in_range) return 1 ; 

    const CSGPrim& lpr = prim[mord] ;   

    float4 localCE = lpr.ce(); 

    lce.x = localCE.x ; 
    lce.y = localCE.y ; 
    lce.z = localCE.z ; 
    lce.w = localCE.w ; 

    LOG(info) << " lce " << lce  ;   
    return 0 ; 
}



/**
CSGTarget::getGlobalCenterExtent
---------------------------------

1. first find the MORD-inal prim *lpr* which has MIDX for its midx/lvIdx
2. use the prim to lookup indices for the solid(gas_idx) and prim 
3. collect instance transforms matching the *gas_idx*
4. select the *iidx* instance transform to construct a global-prim *gpr* 
5. fill in *gce* with the global center-extren from  


*midx* 
    solid (aka mesh, aka lv) index
*mord*
    solid ordinal : this is particularly useful with the global geometry where there are 
    no instances to select between. But there are repeated uses of the mesh that 
    this ordinal picks between. For instanced geometry this will mostly be zero(?)
*iidx*
    instance index, for example this could select a particular PMT 

**/


int CSGTarget::getGlobalCenterExtent(float4& gce, int midx, int mord, int iidx) const 
{
    std::vector<CSGPrim> prim ; 
    foundry->getMeshPrim(prim, midx ); // collect prim matching the MIDX 

    bool mord_in_range = mord < prim.size() ; 
    if(!mord_in_range) 
    {
        LOG(error)  << " midx " << midx << " mord " << mord << " prim.size " << prim.size() << " mord_in_range " << mord_in_range ;   
        return 1 ; 
    }

    // first find the MORD-inal prim which has MIDX for its lvIdx
    const CSGPrim& lpr = prim[mord] ; 
    const float4 local_ce = lpr.ce() ; 

    // use the prim to lookup indices for  the solid and prim 
    unsigned repeatIdx = lpr.repeatIdx(); 
    unsigned primIdx = lpr.primIdx(); 
    unsigned gas_idx = repeatIdx ; 

    // collect the instances 
    std::vector<qat4> inst ; 
    foundry->getInstanceTransformsGAS(inst, gas_idx ); 

    bool iidx_in_range = iidx < inst.size(); 
    LOG(info) 
        << " repeatIdx " << repeatIdx
        << " primIdx " << primIdx
        << " inst.size " << inst.size()
        << " iidx " << iidx
        << " iidx_in_range " << iidx_in_range 
        << " local_ce " << local_ce 
        ; 

    if(!iidx_in_range) return 2 ; 

    qat4 q(inst[iidx].cdata());   // copy the instance
    unsigned ins_idx, gas_idx2, ias_idx ; 
    q.getIdentity(ins_idx, gas_idx2, ias_idx )  ;
    q.clearIdentity();           // clear before doing any transforming 
    assert( gas_idx == gas_idx2 ); 

    CSGPrim gpr = {} ; 
    CSGPrim::Copy(gpr, lpr); 
    q.transform_aabb_inplace( gpr.AABB_() ); 

    LOG(info) 
        << " q " << q 
        << " ins_idx " << ins_idx
        << " ias_idx " << ias_idx
        ; 

    float4 globalCE = gpr.ce(); 
    gce.x = globalCE.x ; 
    gce.y = globalCE.y ; 
    gce.z = globalCE.z ; 
    gce.w = globalCE.w ; 

    LOG(info) 
        << " gpr " << gpr.desc()
        << " gce " << gce 
        ; 

    return 0 ; 
}


