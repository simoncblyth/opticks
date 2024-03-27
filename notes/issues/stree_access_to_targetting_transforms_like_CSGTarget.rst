stree_access_to_targetting_transforms_like_CSGTarget
======================================================


::

    420 int CSGTarget::getTransform(qat4& q, int midx, int mord, int gord) const
    421 {
    422     const qat4* qi = getInstanceTransform(midx, mord, gord);
    423     if( qi == nullptr )
    424     {
    425         return 1 ;
    426     }
    427     qat4::copy(q, *qi);
    428     return 0 ;
    429 }



::

    431 /**
    432 CSGTarget::getInstanceTransform (midx,mord) CSGPrim -> repeatIdx -> which with gord -> instance transform  
    433 ------------------------------------------------------------------------------------------------------------
    434 
    435 1. *CSGFoundry::getMeshPrim* finds the (midx, mord) (CSGPrim)lpr
    436 2. (CSGPrim)lpr gives the repeatIdx (aka:gas_idx or compound solid index) 
    437 3. *CSGFoundry::getInstance_with_GAS_ordinal* finds the (gas_idx, gord) instance transform 
    438 
    439 NB gord was previously named iidx (but that clashes with other uses of that)   
    440 
    441 This method avoids duplication between CSGTarget::getTransform and  CSGTarget::getGlobalCenterExtent 
    442 
    443 Note that the reason to access the (midx,mord) CSGPrim is purely to  
    444 find out which gas_idx it is in and then use that together with 
    445 the gord gas-ordinal to get the transform.
    446 
    447 **/
    448 
    449 const qat4* CSGTarget::getInstanceTransform(int midx, int mord, int gord) const
    450 {
    451     const CSGPrim* lpr = foundry->getMeshPrim(midx, mord);
    452     if(!lpr)
    453     {
    454         LOG(fatal) << "Foundry::getMeshPrim failed for (midx mord) " << "(" << midx << " " <<  mord << ")"  ;
    455         return nullptr ; 
    456     }   
    457     
    458     const float4 local_ce = lpr->ce() ;
    459     unsigned repeatIdx = lpr->repeatIdx(); // use the prim to lookup indices for  the solid and prim 
    460     unsigned primIdx = lpr->primIdx(); 
    461     unsigned gas_idx = repeatIdx ; 
    462     
    463     LOG(LEVEL)
    464         << " (midx mord gord) " << "(" << midx << " " << mord << " " << gord << ") "
    465         << " lpr " << lpr
    466         << " repeatIdx " << repeatIdx
    467         << " primIdx " << primIdx
    468         << " local_ce " << local_ce
    469         ;  
    470         
    471     const qat4* qi = foundry->getInstance_with_GAS_ordinal(gas_idx, gord );
    472     return qi ; 
    473 }   



Not equiv to the below::

    2991 inline void stree::get_repeat_lvid(std::vector<int>& lvids, int q_repeat_index, int q_repeat_ordinal ) const
    2992 {
    2993     get_repeat_field(lvids, 'L', q_repeat_index, q_repeat_ordinal );
    2994 }

