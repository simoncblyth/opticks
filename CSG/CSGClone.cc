
#include "scuda.h"
#include "sqat4.h"

#include "CSGFoundry.h"
#include "CSGSolid.h"
#include "CSGPrim.h"
#include "CSGNode.h"

#include "CSGClone.h"
#include "PLOG.hh"


const plog::Severity CSGClone::LEVEL = PLOG::EnvLevel("CSGClone", "DEBUG" ); 


CSGFoundry* CSGClone::Clone(const CSGFoundry* src )
{
    CSGFoundry* dst = new CSGFoundry ; 
    Copy(dst, src); 
    return dst ; 
}

/**
The ooint of cloning is to provide an easily verifiable starting point 
to implementing Prim selection so must not descend to very low level cloning.

Follow CSGFoundry::addDeepCopySolid and try to improve on it 

**/

void CSGClone::Copy(CSGFoundry* dst, const CSGFoundry* src )
{ 
    unsigned sNumSolid = src->getNumSolid() ;
    int* solidMap = new int[sNumSolid] ; 

    for(unsigned i=0 ; i < sNumSolid ; i++)
    {
        unsigned sSolidIdx = i ; 
        const CSGSolid* sso = src->getSolid(i);
        solidMap[sSolidIdx] = -1 ; 
        unsigned numPrim = sso->numPrim ;  

        // When selecting numPrim will sometimes be less in dst, sometimes even zero, 
        // so need to count selected Prim before adding the solid.

        unsigned dSolidIdx = dst->getNumSolid() ; 
        assert( dSolidIdx == sSolidIdx ); 
        CSGSolid* dso = dst->addSolid(numPrim, sso->label );   
        int dPrimOffset = dso->primOffset ;     // HMM need to account for each prim added ?
        solidMap[sSolidIdx] = dSolidIdx ; 

        //LOG(LEVEL) << " dPrimOffset " << dPrimOffset ; 

        for(int primIdx=sso->primOffset ; primIdx < sso->primOffset+sso->numPrim ; primIdx++)
        {
             const CSGPrim* spr = src->getPrim(primIdx); 

             unsigned numNode = spr->numNode()  ;  // not envisaging node selection, so this will be same in src and dst 
             CSGPrim* dpr = dst->addPrim(numNode, -1 ); 

             int nodeOffset = dpr->nodeOffset(); 
             //LOG(LEVEL) << " nodeOffset " << nodeOffset ;  

             // blind copying makes no sense as offsets will be different with selection  
             // only some metadata referring back to the GGeo geometry is appropriate for copying

             dpr->setMeshIdx(spr->meshIdx());    
             dpr->setRepeatIdx(spr->repeatIdx()); 

             unsigned dPrimIdx_global = dst->getNumPrim() ; 
             unsigned dPrimIdx_local = dPrimIdx_global - dPrimOffset ; 
             dpr->setPrimIdx(dPrimIdx_local); 
             dpr->setAABB( spr->AABB() ); 

             LOG(LEVEL) << "spr " << spr->desc() ; 
             LOG(LEVEL) << "dpr " << dpr->desc() ; 

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

                 dst->addNode(dnd);

                 //TODO: treat plan just like tran 
             }   // over nodes of the prim
        }        // over prim of the solid

        dso->center_extent = sso->center_extent ;  // HMM: this is cheating 

    }   // over solid



    // HMM: some solids may disappear as a result of Prim selection 
    // so will need to change inst too ?

    unsigned sNumInst = src->getNumInst(); 
    for(unsigned i=0 ; i < sNumInst ; i++)
    {
        unsigned sInstIdx = i ; 
        const qat4* ins = src->getInst(sInstIdx) ; 

        unsigned ins_idx ; 
        unsigned gas_idx ; 
        unsigned ias_idx ;         

        ins->getIdentity(ins_idx, gas_idx, ias_idx );

        assert( ins_idx == sInstIdx ); 
        assert( ias_idx == 0u ); 
        assert( gas_idx < sNumSolid ); 

        int sSolidIdx = gas_idx ; 
        int dSolidIdx = solidMap[sSolidIdx] ; 

        if( dSolidIdx > -1 )
        {
            const float* tr16 = ins->cdata(); 
            dst->addInstance(tr16,  dSolidIdx, ias_idx ); 
        }
    }
}

