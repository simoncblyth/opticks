#include "PLOG.hh"
#include "scuda.h"
#include "sqat4.h"
#include "SCenterExtentFrame.hh"

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



2022-01-03 15:06:16.746 INFO  [5325503] [CSGTarget::getLocalCenterExtent@91]  midx 88 mord 10 prim.size 64 mord_in_range 1
2022-01-03 15:06:16.746 INFO  [5325503] [CSGTarget::getLocalCenterExtent@109]  lce (-17336.020,-4160.728,-809.117,66.045) 


-2:world2model_rtpw : scale down and rotate    ( world2model = irotate * iscale * itranslate ) 

 MOI solidXJfixture:10:-2 midx 88 mord 10 iidx -2 gce (-17336.020,-4160.728,-809.117,66.045) 
 q (-0.015, 0.001, 0.004, 0.000) (-0.004, 0.000,-0.015, 0.000) (-0.001,-0.015, 0.000, 0.000) (-270.221,-0.000, 0.000, 1.000) 

-3:model2world_rtpw : scale up and rotate      ( model2world = translate * scale * rotate )

 MOI solidXJfixture:10:-3 midx 88 mord 10 iidx -3 gce (-17336.020,-4160.728,-809.117,66.045) 
 q (-64.155,-15.397,-2.994, 0.000) ( 2.912, 0.699,-65.977, 0.000) (15.413,-64.221, 0.000, 0.000) (-17336.020,-4160.728,-809.117, 1.000) 

-4:world2model_xyzw : uniform scaling down only, no rotation

 MOI solidXJfixture:10:-4 midx 88 mord 10 iidx -4 gce (-17336.020,-4160.728,-809.117,66.045) 
 q ( 0.015, 0.000, 0.000, 0.000) ( 0.000, 0.015, 0.000, 0.000) ( 0.000, 0.000, 0.015, 0.000) (262.489,62.999,12.251, 1.000) 

-5:model2world_xyzw  : uniform scaling up only, no rotation

 MOI solidXJfixture:10:-5 midx 88 mord 10 iidx -5 gce (-17336.020,-4160.728,-809.117,66.045) 
 q (66.045, 0.000, 0.000, 0.000) ( 0.000,66.045, 0.000, 0.000) ( 0.000, 0.000,66.045, 0.000) (-17336.020,-4160.728,-809.117, 1.000) 


**/

int CSGTarget::getCenterExtent(float4& ce, int midx, int mord, int iidx, qat4* m2w, qat4* w2m ) const 
{
    LOG(LEVEL) << " (midx mord iidx) " << "(" << midx << " " << mord << " " << iidx << ") " ;  
    if( iidx == -1 )
    {
        // HMM: CSGFoundry::getCenterExtent BRANCHES FOR iidx == -1 SO THIS WILL NOT BE CALLED 

        LOG(info) << "(iidx == -1) qptr transform will not be set, typically defaulting to identity " ; 
        int lrc = getLocalCenterExtent(ce, midx, mord); 
        if(lrc != 0) return 1 ; 
    }
    else if( iidx == -2 || iidx == -3 )
    {
        LOG(info) << "(iidx == -2/-3  EXPERIMENTAL qptr transform will be set to SCenterExtentFrame transforms " ; 
        int lrc = getLocalCenterExtent(ce, midx, mord); 
        if(lrc != 0) return 1 ; 

        SCenterExtentFrame<double> cef_rtpw( ce.x, ce.y, ce.z, ce.w, true );  
        qat4 world2model_rtpw(cef_rtpw.world2model_data);  // converting to qat4 narrows to float 
        qat4 model2world_rtpw(cef_rtpw.model2world_data); 

        SCenterExtentFrame<double> cef_xyzw( ce.x, ce.y, ce.z, ce.w, false );  
        qat4 world2model_xyzw(cef_xyzw.world2model_data);  // converting to qat4 narrows to float
        qat4 model2world_xyzw(cef_xyzw.model2world_data); 

        if( iidx == -2 )
        {
            qat4::copy(*m2w, model2world_xyzw) ; 
            qat4::copy(*w2m, world2model_xyzw) ; 
        }
        else if( iidx == -3 )
        {
            qat4::copy(*m2w, model2world_rtpw) ; 
            qat4::copy(*w2m, world2model_rtpw) ; 
        }
    }
    else
    {
        int grc = getGlobalCenterExtent(ce, midx, mord, iidx, m2w ); // TODO: paired transforms also ?
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
    foundry->getMeshPrimCopies(prim, midx );  
    bool mord_in_range = mord < int(prim.size()) ; 

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

1. first find the MORD-inal prim *lpr* which has MIDX for its midx

   * midx corresponds to GGeo lvIdx or soIdx

2. use the prim to lookup indices for the solid(gas_idx) and prim 
3. collect instance transforms matching the *gas_idx*
4. select the *iidx* instance transform to construct a global-prim *gpr* 
5. fill in *gce* with the global center-extren from  

*gce*
    output global center extent float4
*midx* 
    input mesh index (aka lv index) 
*mord*
    input mesh ordinal : this is particularly useful with the global geometry where there are 
    no instances to select between. But there are repeated uses of the mesh that 
    this ordinal picks between. For instanced geometry this will mostly be zero(?)
*iidx*
    input instance index, for example this could select a particular PMT 
*qptr*
    output instance transform pointer


TODO: check this with global non-instanced geometry 

**/

int CSGTarget::getGlobalCenterExtent(float4& gce, int midx, int mord, int iidx, qat4* qptr ) const 
{
    const qat4* qi = getInstanceTransform(midx, mord, iidx); 
    if(qi == nullptr)
    {
        LOG(fatal) 
            << " failed to get InstanceTransform (midx mord iidx) " 
            << "(" << midx << " " << mord << " " << iidx << ")" 
            ;
        return 1 ;  
    }

    if(qptr) qat4::copy(*qptr, *qi);  

    qat4 q(qi->cdata());   // copy the instance (transform and identity info)

    unsigned ins_idx, gas_idx, ias_idx ; 
    q.getIdentity(ins_idx, gas_idx, ias_idx )  ;
    q.clearIdentity();           // clear before doing any transforming 


    const CSGPrim* lpr = foundry->getMeshPrim(midx, mord);  

    CSGPrim gpr = {} ; 
    CSGPrim::Copy(gpr, *lpr);   // start global prim from local 
    q.transform_aabb_inplace( gpr.AABB_() ); 

    LOG(LEVEL) 
        << " q " << q 
        << " ins_idx " << ins_idx
        << " ias_idx " << ias_idx
        ; 
    float4 globalCE = gpr.ce(); 
    gce.x = globalCE.x ; 
    gce.y = globalCE.y ; 
    gce.z = globalCE.z ; 
    gce.w = globalCE.w ; 

    LOG(LEVEL) 
        << " gpr " << gpr.desc()
        << " gce " << gce 
        ; 

    return 0 ; 
}


/**
CSGTarget::getTransform TODO eliminate this switching instead to getInstanceTransform
----------------------------------------------------------------------------------------

**/

int CSGTarget::getTransform(qat4& q, int midx, int mord, int iidx) const 
{
    const qat4* qi = getInstanceTransform(midx, mord, iidx); 
    if( qi == nullptr )
    {
        return 1 ; 
    }
    qat4::copy(q, *qi); 
    return 0 ; 
}

/**
CSGTarget::getInstanceTransform
---------------------------------

This method was added to eliminate duplication between CSGTarget::getTransform and  CSGTarget::getGlobalCenterExtent 

Formally used temporary vector of qat4 transforms::

   std::vector<qat4> inst ; 
   foundry->getInstanceTransformsGAS(inst, gas_idx ); 

But that makes usage difficult due to the limited lifetime of the 
temporary inst vector. So instead have switched to collecting pointers 
to the actual original instances from CSGFoundry not copies of them. 

**/

const qat4* CSGTarget::getInstanceTransform(int midx, int mord, int iidx) const 
{
    const CSGPrim* lpr = foundry->getMeshPrim(midx, mord);  
    if(!lpr)
    {
        LOG(fatal) << "Foundry::getMeshPrim failed for (midx mord) " << "(" << midx << " " <<  mord << ")"  ; 
        return nullptr ; 
    }

    const float4 local_ce = lpr->ce() ; 
    unsigned repeatIdx = lpr->repeatIdx(); // use the prim to lookup indices for  the solid and prim 
    unsigned primIdx = lpr->primIdx(); 
    unsigned gas_idx = repeatIdx ; 

    LOG(LEVEL) 
        << " (midx mord iidx) " << "(" << midx << " " << mord << " " << iidx << ") "
        << " lpr " << lpr
        << " repeatIdx " << repeatIdx
        << " primIdx " << primIdx
        << " local_ce " << local_ce 
        ; 

    const qat4* qi = foundry->getInstanceGAS(gas_idx, iidx ); 
    return qi ; 
}

