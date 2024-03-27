stree_access_to_targetting_transforms_like_CSGTarget
======================================================


CE from stree ?
----------------

* center of instanced from the transform : but not for global ? and not extent 
* HMM: need to add some bbox to stree perhaps ? 


Where do CSGPrim bbox come from ?
----------------------------------

* CSGImport::importPrim  combines bb of snd.h  
* CSGImport::importNode 


::

   std::array<double,6> bb ;
    double* aabb = leaf ? bb.data() : nullptr ;
    // NB : TRANSFORM VERY DEPENDENT ON node.repeat_index == 0 OR not 
    const Tran<double>* tv = leaf ? st->get_combined_tran_and_aabb( aabb, node, nd, nullptr ) : nullptr ;
    unsigned tranIdx = tv ?  1 + fd->addTran(tv) : 0 ;   // 1-based index referencing foundry transforms



CSG way of getting the instance tr
---------------------------------------

* added stree::get_frame stree::pick_lvid_ordinal_repeat_ordinal_inst to do equiv from stree

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

