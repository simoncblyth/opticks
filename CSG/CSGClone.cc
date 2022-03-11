
#include "CSGFoundry.h"
#include "CSGSolid.h"
#include "CSGPrim.h"
#include "CSGNode.h"

#include "CSGClone.h"

CSGFoundry* CSGClone::Clone(const CSGFoundry* src )
{
    CSGFoundry* dst = new CSGFoundry ; 
    Copy(dst, src); 
    return dst ; 
}

/**
Point of cloning is to provide an easily verifiable starting point 
to implementing Prim selection so must not descend to very low level cloning.

Follow CSGFoundry::addDeepCopySolid and try to improve on it 

**/

void CSGClone::Copy(CSGFoundry* dst, const CSGFoundry* src )
{ 
    // HMM: some solids may disappear as a result of Prim selection 
    // so will need to change inst too ?

    unsigned numSolid = src->getNumSolid() ;

    for(unsigned i=0 ; i < numSolid ; i++)
    {
        const CSGSolid* sso = src->getSolid(i);
        unsigned numPrim = sso->numPrim ;  
        // When selecting numPrim will sometimes be less in dst, sometimes even zero, 
        // so need to count selected Prim before adding the solid.
        CSGSolid* dso = dst->addSolid(numPrim, sso->label );   
        int dPrimOffset = dso->primOffset ;    

        for(int primIdx=sso->primOffset ; primIdx < sso->primOffset+sso->numPrim ; primIdx++)
        {
             const CSGPrim* spr = src->getPrim(primIdx); 

             unsigned numNode = spr->numNode()  ;  // not envisaging node selection, so this will be same in src and dst 
             int nodeOffset_ = -1 ; 
             CSGPrim* dpr = dst->addPrim(numNode, nodeOffset_ ); 

             // blind copying makes no sense as offsets will be different with selection  
             // only some metadata referring back to the GGeo geometry is appropriate for copying

             dpr->setMeshIdx(spr->meshIdx());    
             dpr->setRepeatIdx(spr->repeatIdx()); 

             unsigned dPrimIdx_global = dst->getNumPrim() ; 
             unsigned dPrimIdx_local = dPrimIdx_global - dPrimOffset ; 
             dpr->setPrimIdx(dPrimIdx_local); 
            
             for(int nodeIdx=spr->nodeOffset() ; nodeIdx < spr->nodeOffset()+spr->numNode() ; nodeIdx++)
             {
                 const CSGNode* snd = src->getNode(nodeIdx); 

                 unsigned sTranIdx = snd->gtransformIdx(); 
                 const qat4* tra = sTranIdx == 0 ? nullptr : src->getTran(sTranIdx-1u) ; 
                 const qat4* itr = sTranIdx == 0 ? nullptr : src->getItra(sTranIdx-1u) ; 
                 unsigned dTranIdx = tra && itr ? 1u + dst->addTran( tra, itr ) : 0 ; 

                 CSGNode dnd = {} ;
                 CSGNode::Copy(dnd, *snd );   // dumb straight copy : so need to fix transform and plan references  
                 dnd.setTransform( dTranIdx); 
             }
        }
    }
}

