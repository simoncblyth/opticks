
#include "scuda.h"
#include "sqat4.h"
#include "SBitSet.hh"
#include "OpticksCSG.h"

#include "CSGFoundry.h"
#include "CSGSolid.h"
#include "CSGPrim.h"
#include "CSGNode.h"

#include "CSGCopy.h"
#include "PLOG.hh"


const plog::Severity CSGCopy::LEVEL = PLOG::EnvLevel("CSGCopy", "DEBUG" ); 


CSGFoundry* CSGCopy::Clone(const CSGFoundry* src )
{
    CSGFoundry* dst = new CSGFoundry ; 
    Copy(dst, src, nullptr); 
    return dst ; 
}

CSGFoundry* CSGCopy::Select(const CSGFoundry* src, const SBitSet* elv )
{
    CSGFoundry* dst = new CSGFoundry ; 
    Copy(dst, src, elv); 
    return dst ; 
}

/**
CSGCopy::Copy
--------------

The point of cloning is to provide an easily verifiable starting point 
to implementing Prim selection so must not descend to very low level cloning.

* this has some similarities with CSGFoundry::addDeepCopySolid and CSG_GGeo/CSG_GGeo_Convert.cc

* Blind copying makes no sense as offsets will be different with selection  
  only some metadata referring back to the GGeo geometry is appropriate for copying

* When selecting numPrim will sometimes be less in dst, sometimes even zero, 
  so need to count selected Prim before adding the solid.

**/

void CSGCopy::Copy(CSGFoundry* dst, const CSGFoundry* src, const SBitSet* elv )
{ 
    unsigned sNumSolid = src->getNumSolid() ;
    int* solidMap = new int[sNumSolid] ; 

    for(unsigned i=0 ; i < sNumSolid ; i++)
    {
        unsigned sSolidIdx = i ; 
        solidMap[sSolidIdx] = -1 ; 

        const CSGSolid* sso = src->getSolid(i);
        unsigned numSelectedPrim = src->getNumSelectedPrimInSolid(sso, elv );  
        LOG(LEVEL) << " sso " << sso->desc() << " numSelectedPrim " << numSelectedPrim ; 
        if( numSelectedPrim == 0 ) continue ;  

        unsigned dSolidIdx = dst->getNumSolid() ; // index before adding (0-based)
        if( elv == nullptr ) assert( dSolidIdx == sSolidIdx ); 

        CSGSolid* dso = dst->addSolid(numSelectedPrim, sso->label );   
        int dPrimOffset = dso->primOffset ;       
        assert( dPrimOffset == int(dst->prim.size()) );  

        solidMap[sSolidIdx] = dSolidIdx ; 

        CopySolidPrim(dPrimOffset, dst, sso, src, elv, true );  

        unsigned numSelectedPrimCheck = dst->prim.size() - dPrimOffset ; 
        assert( numSelectedPrim == numSelectedPrimCheck );  

        dso->center_extent = sso->center_extent ;  // HMM: this is cheating, need to accumulate when using selection 
    }   // over solids

    CopySolidInstances( solidMap, sNumSolid, dst, src ); 
}

/**
CSGCopy::CopySolidPrim
------------------------

**/

void CSGCopy::CopySolidPrim(int dPrimOffset, CSGFoundry* dst, const CSGSolid* sso, const CSGFoundry* src, const SBitSet* elv, bool dump )
{
    for(int primIdx=sso->primOffset ; primIdx < sso->primOffset+sso->numPrim ; primIdx++)
    {
         const CSGPrim* spr = src->getPrim(primIdx); 
         unsigned meshIdx = spr->meshIdx() ; 
         unsigned repeatIdx = spr->repeatIdx() ; 
         bool selected = elv == nullptr ? true : elv->is_set(meshIdx) ; 
         if( selected == false ) continue ; 

         unsigned numNode = spr->numNode()  ;  // not envisaging node selection, so this will be same in src and dst 
         unsigned dPrimIdx_global = dst->getNumPrim() ;            // destination numPrim prior to prim addition
         unsigned dPrimIdx_local = dPrimIdx_global - dPrimOffset ; // make the PrimIdx local to the solid 

         CSGPrim* dpr = dst->addPrim(numNode, -1 ); 
         if( elv == nullptr ) assert( dpr->nodeOffset() == spr->nodeOffset() ); 

         dpr->setMeshIdx(meshIdx);    
         dpr->setRepeatIdx(repeatIdx); 
         dpr->setPrimIdx(dPrimIdx_local); 

         AABB dbb = {} ;
         CopyPrimNodes(dbb, dst, spr, src, false ); 
         //dpr->setAABB( dbb.data() );  
         dpr->setAABB( spr->AABB() );  // will not be so with selection 

         unsigned mismatch = 0 ; 
         std::string cf = AABB::Compare(mismatch, spr->AABB(), dbb.data(), 1, 1e-6 ) ; 
         if (mismatch > 0 )
         {
             std::cout << std::endl ;  
             std::cout << "spr " << spr->desc() << std::endl ; 
             std::cout << "dpr " << dpr->desc() << std::endl ; 
             std::cout << "dbb " << std::setw(20) << " " << dbb.desc() << std::endl ; 
             std::cout << " AABB::Compare " << cf << std::endl ; 
         }
    }   // over prim of the solid
}


/**
CSGCopy::CopyPrimNodes
-------------------------

**/

void CSGCopy::CopyPrimNodes(AABB& bb, CSGFoundry* dst, const CSGPrim* spr, const CSGFoundry* src, bool dump )
{
    for(int nodeIdx=spr->nodeOffset() ; nodeIdx < spr->nodeOffset()+spr->numNode() ; nodeIdx++)
    {
        const CSGNode* snd = src->getNode(nodeIdx); 
        unsigned stypecode = snd->typecode(); 
        unsigned sTranIdx = snd->gtransformIdx(); 

        bool has_planes = CSG::HasPlanes(stypecode) ; 
        bool has_transform = sTranIdx > 0u ; 

        if(dump) LOG(LEVEL) 
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
            if(dump) LOG(LEVEL) << " planeIdx " << snd->planeIdx() << " planeNum " << snd->planeNum() ;  
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
            if(dump) LOG(LEVEL) << " tra " << tra << " itr " << itr << " dTranIdx " << dTranIdx ; 
        }
           
        CSGNode dnd = {} ;
        CSGNode::Copy(dnd, *snd );   // dumb straight copy : so need to fix transform and plan references  
        dnd.setTransform( dTranIdx ); 

        CSGNode* dptr = dst->addNode(dnd, &splanes);   

        assert( dptr->planeNum() == snd->planeNum() ); 
        assert( dptr->planeIdx() == snd->planeIdx() ); 

        bool negated = dptr->is_complemented_primitive();
        float* naabb = dptr->AABB();   // hmm is transform applied ?
        if(!negated) bb.include_aabb( naabb );

        // see tail of CSG_GGeo_Convert::convertNode which uses qat4::transform_aabb_inplace to change CSGNode aabb
    }   
}


/**
CSGCopy::CopySolidInstances
-------------------------------

As some solids may disappear as a result of Prim selection 
it is necessary to change potentially all the inst solid references. 

**/

void CSGCopy::CopySolidInstances( const int* solidMap, unsigned sNumSolid, CSGFoundry* dst, const CSGFoundry* src )
{
    assert( sNumSolid == src->getNumSolid()) ;

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

