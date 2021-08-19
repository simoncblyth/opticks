#include "PLOG.hh"

#include "sutil_vec_math.h"
#include "CSGFoundry.h"
#include "CSGTarget.h"

const plog::Severity CSGTarget::LEVEL = PLOG::EnvLevel("CSGTarget", "DEBUG" ); 

CSGTarget::CSGTarget( const CSGFoundry* foundry_ )
    :
    foundry(foundry_)
{
}

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


int CSGTarget::getLocalCenterExtent(float4& lce, int midx, int mord) const 
{
    // collect prim matching the MIDX and select the ORDINAL one
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


