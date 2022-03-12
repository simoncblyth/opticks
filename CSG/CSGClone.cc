
#include "scuda.h"
#include "sqat4.h"
#include "OpticksCSG.h"

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
        LOG(LEVEL) << " sso " << sso->desc() ; 


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

             unsigned dPrimIdx_global = dst->getNumPrim() ;  // destination numPrim prior to prim addition
             unsigned dPrimIdx_local = dPrimIdx_global - dPrimOffset ; // make the PrimIdx local to the solid 

             CSGPrim* dpr = dst->addPrim(numNode, -1 ); 

             assert( dpr->nodeOffset() == spr->nodeOffset() ); 

             // blind copying makes no sense as offsets will be different with selection  
             // only some metadata referring back to the GGeo geometry is appropriate for copying

             dpr->setMeshIdx(spr->meshIdx());    
             dpr->setRepeatIdx(spr->repeatIdx()); 
             dpr->setPrimIdx(dPrimIdx_local); 
             dpr->setAABB( spr->AABB() ); 

             LOG(LEVEL) << "spr " << spr->desc() ; 
             LOG(LEVEL) << "dpr " << dpr->desc() ; 

             for(int nodeIdx=spr->nodeOffset() ; nodeIdx < spr->nodeOffset()+spr->numNode() ; nodeIdx++)
             {
                 const CSGNode* snd = src->getNode(nodeIdx); 
                 unsigned stypecode = snd->typecode(); 
                 unsigned sTranIdx = snd->gtransformIdx(); 

                 bool has_planes = CSG::HasPlanes(stypecode) ; 
                 bool has_transform = sTranIdx > 0u ; 
 

                 LOG(LEVEL) 
                     << " nodeIdx " << nodeIdx 
                     << " stypecode " << stypecode 
                     << " sTranIdx " << sTranIdx 
                     << " csgname " << CSG::Name(stypecode) 
                     << " has_planes " << has_planes
                     << " has_transform " << has_transform
                     ; 

                 std::vector<float4> splanes ; 
                 if(has_planes)
                 {
                     LOG(LEVEL) << " planeIdx " << snd->planeIdx() << " planeNum " << snd->planeNum() ;  
                     for(unsigned planIdx=snd->planeIdx() ; planIdx < snd->planeIdx() + snd->planeNum() ; planIdx++)
                     {  
                         const float4* splan = src->getPlan(planIdx);  
                         splanes.push_back(*splan); 
                     }
                 }

                 unsigned dTranIdx = 0u ; 
                 if(has_transform)
                 {
                     const qat4* tra = src->getTran(sTranIdx-1u) ; 
                     const qat4* itr = src->getItra(sTranIdx-1u) ; 
                     dTranIdx = 1u + dst->addTran( tra, itr ) ;
                     LOG(LEVEL) << " tra " << tra << " itr " << itr << " dTranIdx " << dTranIdx ; 
                 }
                   
                 CSGNode dnd = {} ;
                 CSGNode::Copy(dnd, *snd );   // dumb straight copy : so need to fix transform and plan references  
                 dnd.setTransform( dTranIdx ); 

                 CSGNode* dptr = dst->addNode(dnd, &splanes);   

                 assert( dptr->planeNum() == snd->planeNum() ); 
                 assert( dptr->planeIdx() == snd->planeIdx() ); 

             }   // over nodes of the prim
        }        // over prim of the solid

        dso->center_extent = sso->center_extent ;  // HMM: this is cheating, need to accumulate when using selection 

    }   // over solids



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

