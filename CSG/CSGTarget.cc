#include <csignal>

#include "SLOG.hh"
#include "scuda.h"
#include "sqat4.h"
#include "sframe.h"
#include "SCenterExtentFrame.h"

#include "CSGFoundry.h"
#include "CSGTarget.h"

const plog::Severity CSGTarget::LEVEL = SLOG::EnvLevel("CSGTarget", "DEBUG" );

CSGTarget::CSGTarget( const CSGFoundry* foundry_ )
    :
    foundry(foundry_)
{
}



/**
CSGTarget::getFrame "getFrameFromInstanceLookup"
-------------------------------------------------

* TODO: avoid this inverting the "m2w" instance transform to give w2m

Note that there are typically multiple CSGPrim within the compound CSGSolid
and that the inst_idx corresponds to the entire compound CSGSolid (aka GMergedMesh).
Hence the ce included with the frame is the one from the full compound CSGSolid.

* DONE: new minimal U4Tree.h/stree.h geo translation collects paired m2w and w2m transforms
  and uses those to give both inst and iinst in double precision

* TODO: move the stree.h inst/iinst members and methods into CSGFoundry
  and use them from a future "CSGFoundry::CreateFromSTree" to add
  paired double precision transforms, avoiding Invert

TODO: would be better to have sframe access from stree.h

**/

int CSGTarget::getFrame(sframe& fr, int inst_idx ) const
{
    const qat4* _t = foundry->getInst(inst_idx);

    LOG(LEVEL)
        << " inst_idx " << inst_idx
        << " _t " << ( _t ? "YES" : "NO " )
        ;

    LOG_IF(error, _t == nullptr)
        << " inst_idx " << inst_idx
        << " failed to foundry->getInst(inst_idx), "
        << " foundry->getNumInst() : " << foundry->getNumInst()
        ;
    if(_t == nullptr) return 1 ;


    int ins_idx,  gas_idx, sensor_identifier, sensor_index ;
    _t->getIdentity(ins_idx,  gas_idx, sensor_identifier, sensor_index );

    LOG(LEVEL)
        << " inst_idx " << inst_idx
        << " _t " << ( _t ? "YES" : "NO " )
        << " ins_idx " << ins_idx
        << " gas_idx " << gas_idx
        << " sensor_identifier " << sensor_identifier
        << " sensor_index " << sensor_index
        ;

    LOG_IF( fatal , gas_idx < 0 )
        << " gas_idx " << gas_idx
        << " foundry->inst.size " << foundry->inst.size()
        << " BAD sqat4.h inst ? "
        ;

    assert( gas_idx > -1 );

    assert( ins_idx == inst_idx );
    fr.set_inst(inst_idx);

    // HMM: these values are already there inside the matrices ?
    fr.set_identity(ins_idx, gas_idx, sensor_identifier, sensor_index ) ;

    qat4 t(_t->cdata());   // copy the instance (transform and identity info)
    const qat4* v = Tran<double>::Invert(&t);

    qat4::copy(fr.m2w,  t);
    qat4::copy(fr.w2m, *v);

    delete v ;

    // identity info IS NOT cleared by Tran::Invert
    // as there is special handling to retain it (see stran.h)
    // the explicit clearing below fixes a bug revealed during
    // Raindrop revival
    // see notes/issues/Raindrop_revival_fix_CSGTarget_getFrame_nan_from_not_clearing_identity_info.rst


    fr.m2w.clearIdentity();
    fr.w2m.clearIdentity();

    // TODO: adopt sframe::setTransform

    const CSGSolid* solid = foundry->getSolid(gas_idx);
    fr.ce = solid->center_extent ;


    LOG(LEVEL)
        << " inst_idx " << inst_idx
        << " _t " << ( _t ? "YES" : "NO " )
        << " ins_idx " << ins_idx
        << " gas_idx " << gas_idx
        << " sensor_identifier " << sensor_identifier
        << " sensor_index " << sensor_index
        << " fr.m2w.q3.f.w " <<  fr.m2w.q3.f.w
        << " fr.m2w.q3.i.w " <<  fr.m2w.q3.i.w
        << " fr.w2m.q3.f.w " <<  fr.w2m.q3.f.w
        << " fr.w2m.q3.i.w " <<  fr.w2m.q3.i.w
        ;

    return 0 ;
}



/**
CSGTarget::getFrame
----------------------


midx
    mesh index (aka lv)
mord
    mesh ordinal (picking between multipler occurrences of midx
gord
    GAS ordinal [NB this is not the GAS index]


NB the GAS index is determined from (midx, mord)
and then gord picks between potentially multiple occurrences


Q: is indexing by MOI and inst_idx equivalent ? OR: Can a MOI be converted into inst_idx and vice versa ?
A: see notes with CSGFoundry::getFrame

**/

int CSGTarget::getFrame(sframe& fr,  int midx, int mord, int gord ) const
{
    fr.set_midx_mord_gord( midx, mord, gord );
    int rc = getFrameComponents( fr.ce, midx, mord, gord, &fr.m2w , &fr.w2m );
    LOG(LEVEL) << " midx " << midx << " mord " << mord << " gord " << gord << " rc " << rc ;
    return rc ;
}


/**
CSGTarget::getFrameComponents
-------------------------------

Used by CSGFoundry::getCenterExtent

*ce*
    center-extent float4 set when return code is zero
*midx*
    solid (aka mesh) index : identify the shape
*mord*
    solid (aka mesh) ordinal : pick between shapes when there are more than one, used with global(non-instanced) geometry
*gord*
    GAS ordinal 0,1,2,...  or when -ve special cases



2022-01-03 15:06:16.746 INFO  [5325503] [CSGTarget::getLocalCenterExtent@91]  midx 88 mord 10 prim.size 64 mord_in_range 1
2022-01-03 15:06:16.746 INFO  [5325503] [CSGTarget::getLocalCenterExtent@109]  lce (-17336.020,-4160.728,-809.117,66.045)


-2:world2model_rtpw : scale down and rotate    ( world2model = irotate * iscale * itranslate )

 MOI solidXJfixture:10:-2
    midx 88 mord 10 gord -2
    gce (-17336.020,-4160.728,-809.117,66.045)
    q (-0.015, 0.001, 0.004, 0.000)
      (-0.004, 0.000,-0.015, 0.000)
      (-0.001,-0.015, 0.000, 0.000)
      (-270.221,-0.000, 0.000, 1.000)

-3:model2world_rtpw : scale up and rotate      ( model2world = translate * scale * rotate )

 MOI solidXJfixture:10:-3
    midx 88 mord 10 gord -3
    gce (-17336.020,-4160.728,-809.117,66.045)
    q (-64.155,-15.397,-2.994, 0.000)
      ( 2.912, 0.699,-65.977, 0.000)
      (15.413,-64.221, 0.000, 0.000)
      (-17336.020,-4160.728,-809.117, 1.000)

-4:world2model_xyzw : uniform scaling down only, no rotation

 MOI solidXJfixture:10:-4
    midx 88 mord 10 gord -4
    gce (-17336.020,-4160.728,-809.117,66.045)
    q ( 0.015, 0.000, 0.000, 0.000)
      ( 0.000, 0.015, 0.000, 0.000)
      ( 0.000, 0.000, 0.015, 0.000)
      (262.489,62.999,12.251, 1.000)

-5:model2world_xyzw  : uniform scaling up only, no rotation

 MOI solidXJfixture:10:-5
    midx 88 mord 10 gord -5
    gce (-17336.020,-4160.728,-809.117,66.045)
    q (66.045, 0.000, 0.000, 0.000)
      ( 0.000,66.045, 0.000, 0.000)
      ( 0.000, 0.000,66.045, 0.000)
      (-17336.020,-4160.728,-809.117, 1.000)


HMM : INCONSISTENCY IN THE MEANING BETWEEN THIS CSGTarget::getFrameComponents
LOOKS LIKE -1 in stree::get_frame is equivalent to -2 from here

WHAT USE IS THE -1 FROM HERE ?


**/

int CSGTarget::getFrameComponents(float4& ce, int midx, int mord, int gord, qat4* m2w, qat4* w2m ) const
{
    LOG(LEVEL) << " (midx mord gord) " << "(" << midx << " " << mord << " " << gord << ") " ;
    if( gord == -1 || gord == -2 || gord == -3 )
    {
        int lrc = getLocalCenterExtent(ce, midx, mord);
        if(lrc != 0) return 1 ;

        if( gord == -1 )
        {
            LOG(info) << "(gord == -1) qptr transform will not be set, typically defaulting to identity " ;
            assert(0) ; // WHAT IS USING -1 WITH CE SET AND IDENTITY TRANSFORMS ?  COMMONLY 0:0:-1 default MOI
        }
        else if( gord == -2 )
        {
            bool rtp_tangential = false ;
            bool extent_scale = false ;  // NB recent change switching off extent scaling
            SCenterExtentFrame<double> cef_xyzw( ce.x, ce.y, ce.z, ce.w, rtp_tangential, extent_scale );
            m2w->read_narrow(cef_xyzw.model2world_data);
            w2m->read_narrow(cef_xyzw.world2model_data);
        }
        else if( gord == -3 )
        {
            bool rtp_tangential = true ;
            bool extent_scale = false ;   // NB recent change witching off extent scaling
            SCenterExtentFrame<double> cef_rtpw( ce.x, ce.y, ce.z, ce.w, rtp_tangential, extent_scale );
            m2w->read_narrow(cef_rtpw.model2world_data);
            w2m->read_narrow(cef_rtpw.world2model_data);
        }
    }
    else
    {
        int grc = getGlobalCenterExtent(ce, midx, mord, gord, m2w, w2m );
        //  HMM: the m2w here populated is from the (midx, mord, gord) instance transform, with identity info
        if(grc != 0) return 2 ;
    }
    return 0 ;
}




/**
CSGTarget::getLocalCenterExtent : lookup (midx,mord) CSGPrim and returns its CE
-------------------------------------------------------------------------------------

Collects prim matching the *midx* and selects the *mord* ordinal one
from which to read the localCE

**/

int CSGTarget::getLocalCenterExtent(float4& lce, int midx, int mord) const
{
    std::vector<CSGPrim> prim ;
    foundry->getMeshPrimCopies(prim, midx );
    bool mord_in_range = mord < int(prim.size()) ;

    LOG(LEVEL)
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

    LOG(LEVEL) << " lce " << lce  ;
    return 0 ;
}



/**
CSGTarget::getGlobalCenterExtent
---------------------------------

1. getInstanceTransform from (midx, mord, gord) giving m2w
2. invert m2w giving w2m

4. select the *gord* instance transform to construct a global-prim *gpr*
5. fill in *gce* with the global center-extren from

*gce*
    output global center extent float4
*midx*
    input mesh index (aka lv index)
*mord*
    input mesh ordinal : this is particularly useful with the global geometry where there are
    no instances to select between. But there are repeated uses of the mesh that
    this ordinal picks between. For instanced geometry this will mostly be zero(?)
*gord*
    input GAS ordinal, allow to select between multiple instances of the GAS,
    that has also been used when negative for other selections

*qptr*
    output instance transform pointer. When non-null the instance
    transform will be copied into this qat4 which will contain
    identity integers in its fourth column





HMM with global non-instanced geometry the transforms should be identity

**/

int CSGTarget::getGlobalCenterExtent(float4& gce, int midx, int mord, int gord, qat4* m2w, qat4* w2m ) const
{
    const qat4* t = getInstanceTransform(midx, mord, gord);
    const qat4* v = t ? Tran<double>::Invert(t ) : nullptr ;

    LOG_IF(fatal, t == nullptr)
        << " failed to get InstanceTransform (midx mord gord) " << "(" << midx << " " << mord << " " << gord << ")" ;

    LOG_IF(fatal, v == nullptr)
        << " failed Tran<double>::Invert " ;

    if(t == nullptr) return 1 ;
    if(v == nullptr) return 2 ;

    LOG(LEVEL)
        << std::endl
        << t->desc('t')
        << std::endl
        << v->desc('v')
        ;

    if(m2w)
    {
        qat4::copy(*m2w, *t);
        m2w->clearIdentity();  // recent addition
    }
    if(w2m)
    {
        qat4::copy(*w2m, *v);
        w2m->clearIdentity();  // recent addition
    }

    qat4 q(t->cdata());   // copy the instance (transform and identity info)

    int ins_idx,  gas_idx, sensor_identifier, sensor_index ;
    q.getIdentity(ins_idx,  gas_idx, sensor_identifier, sensor_index );
    q.clearIdentity();    // clearIdentity sets the (3,3) 1. : needed before doing any transforming

    // TODO: could incorporate this identity into the sframe ?




    const CSGPrim* lpr = foundry->getMeshPrim(midx, mord);

    CSGPrim gpr = {} ;
    CSGPrim::Copy(gpr, *lpr);   // start global prim from local

    q.transform_aabb_inplace( gpr.AABB_() );

    LOG(LEVEL)
        << std::endl
        << " q " << q
        << std::endl
        << " ins_idx " << ins_idx
        << " gas_idx " << gas_idx
        << " sensor_identifier " << sensor_identifier
        << " sensor_index " << sensor_index
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

int CSGTarget::getTransform(qat4& q, int midx, int mord, int gord) const
{
    const qat4* qi = getInstanceTransform(midx, mord, gord);
    if( qi == nullptr )
    {
        return 1 ;
    }
    qat4::copy(q, *qi);
    return 0 ;
}

/**
CSGTarget::getInstanceTransform (midx,mord) CSGPrim -> repeatIdx -> which with gord -> instance transform
------------------------------------------------------------------------------------------------------------

1. *CSGFoundry::getMeshPrim* finds the (midx, mord) (CSGPrim)lpr
2. (CSGPrim)lpr gives the repeatIdx (aka:gas_idx or compound solid index)
3. *CSGFoundry::getInstance_with_GAS_ordinal* finds the (gas_idx, gord) instance transform

NB gord was previously named iidx (but that clashes with other uses of that)

This method avoids duplication between CSGTarget::getTransform and  CSGTarget::getGlobalCenterExtent

Note that the (midx,mord) CSGPrim is accessed purely to
find out which gas_idx it has and then use that together with
the gord gas-ordinal to get the corresponding instance transform.

Using (midx,mord) as input rather than gas_idx directly is because
they are more fundamental whereas the gas_idx may change when adjust
instancing criteria for example.

The mord is often zero, but its needed for handling possible repeats of a midx/solid.

**/

const qat4* CSGTarget::getInstanceTransform(int midx, int mord, int gord) const
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
        << " (midx mord gord) " << "(" << midx << " " << mord << " " << gord << ") "
        << " lpr " << lpr
        << " repeatIdx " << repeatIdx
        << " primIdx " << primIdx
        << " local_ce " << local_ce
        ;

    const qat4* qi = foundry->getInstance_with_GAS_ordinal(gas_idx, gord );
    return qi ;
}




