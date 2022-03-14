
#include "scuda.h"
#include "sqat4.h"
#include "SBitSet.hh"
#include "SSys.hh"
#include "OpticksCSG.h"

#include "CSGFoundry.h"
#include "CSGSolid.h"
#include "CSGPrim.h"
#include "CSGNode.h"

#include "CSGCopy.h"
#include "PLOG.hh"


const plog::Severity CSGCopy::LEVEL = PLOG::EnvLevel("CSGCopy", "DEBUG" ); 
const int CSGCopy::DUMP_RIDX = SSys::getenvint("DUMP_RIDX", -1) ; 
const int CSGCopy::DUMP_NPS = SSys::getenvint("DUMP_NPS", 0) ; 



CSGFoundry* CSGCopy::Clone(const CSGFoundry* src )
{
    CSGCopy cpy(src, nullptr); 
    cpy.copy(); 
    LOG(info) << cpy.desc(); 
    return cpy.dst ; 
}

CSGFoundry* CSGCopy::Select(const CSGFoundry* src, const SBitSet* elv )
{
    CSGCopy cpy(src, elv); 
    cpy.copy(); 
    LOG(info) << cpy.desc(); 
    return cpy.dst ; 
}

CSGCopy::CSGCopy(const CSGFoundry* src_, const SBitSet* elv_)
    :
    src(src_),
    sNumSolid(src->getNumSolid()),
    solidMap(new int[sNumSolid]), 
    sSolidIdx(~0u), 
    elv(elv_),
    dst(new CSGFoundry)
{
}

std::string CSGCopy::desc() const 
{
    std::stringstream ss ; 
    ss 
        << std::endl 
        << "src:" 
        << src->desc() 
        << std::endl 
        << "dst:" 
        << dst->desc()
        << std::endl 
        ; 
    std::string s = ss.str(); 
    return s ; 
}

CSGCopy::~CSGCopy()
{
    delete [] solidMap ; 
}

unsigned CSGCopy::Dump( unsigned sSolidIdx )
{
    return ( DUMP_RIDX >= 0 && unsigned(DUMP_RIDX) == sSolidIdx ) ? DUMP_NPS : 0u ; 
}

/**
CSGCopy::copy
--------------

The point of cloning is to provide an easily verifiable starting point 
to implementing Prim selection so must not descend to very low level cloning.

* this has some similarities with CSGFoundry::addDeepCopySolid and CSG_GGeo/CSG_GGeo_Convert.cc

* Blind copying makes no sense as offsets will be different with selection  
  only some metadata referring back to the GGeo geometry is appropriate for copying

* When selecting numPrim will sometimes be less in dst, sometimes even zero, 
  so need to count selected Prim before adding the solid.

**/

void CSGCopy::copy()
{ 
    copyMeshName();
 
    for(unsigned i=0 ; i < sNumSolid ; i++)
    {
        sSolidIdx = i ; 
        solidMap[sSolidIdx] = -1 ; 

        unsigned dump_ = Dump(sSolidIdx); 
        bool dump_solid = dump_ & 0x1 ; 
        if(dump_solid)
        {
            LOG(info) << "sSolidIdx " << sSolidIdx << " DUMP_RIDX " << DUMP_RIDX  << " DUMP_NPS " << DUMP_NPS << " dump_solid " << dump_solid  ;   
        }

        const CSGSolid* sso = src->getSolid(sSolidIdx);
        unsigned numSelectedPrim = src->getNumSelectedPrimInSolid(sso, elv );  
        const std::string& solidLabel = src->getSolidLabel(sSolidIdx); 
        if(dump_solid) LOG(LEVEL) << " sso " << sso->desc() << " numSelectedPrim " << numSelectedPrim << " solidLabel " << solidLabel ; 

        if( numSelectedPrim == 0 ) continue ;  

        dst->addSolidLabel( solidLabel.c_str() );  

        unsigned dSolidIdx = dst->getNumSolid() ; // index before adding (0-based)
        if( elv == nullptr ) assert( dSolidIdx == sSolidIdx ); 

        CSGSolid* dso = dst->addSolid(numSelectedPrim, sso->label );   
        int dPrimOffset = dso->primOffset ;       
        assert( dPrimOffset == int(dst->prim.size()) );  

        solidMap[sSolidIdx] = dSolidIdx ; 

        AABB solid_bb = {} ;

        copySolidPrim(solid_bb, dPrimOffset, sso);  

        unsigned numSelectedPrimCheck = dst->prim.size() - dPrimOffset ; 
        assert( numSelectedPrim == numSelectedPrimCheck );  

        //dso->center_extent = sso->center_extent ;  // HMM: this is cheating, need to accumulate when using selection 
        dso->center_extent = solid_bb.center_extent(); 

        if(dump_solid) LOG(LEVEL) << " dso " << dso->desc() ; 

    }   // over solids of the entire geometry 

    copySolidInstances(); 
}


void CSGCopy::copyMeshName()
{
    assert( dst->meshname.size() == 0); 
    src->getMeshName(dst->meshname); 
    assert( src->meshname.size() == dst->meshname.size() ); 
}



/**
CSGCopy::copySolidPrim
------------------------

See the AABB mechanics at the tail of CSGFoundry::addDeepCopySolid

**/

void CSGCopy::copySolidPrim(AABB& solid_bb, int dPrimOffset, const CSGSolid* sso )
{
    unsigned dump_ = Dump(sSolidIdx); 
    bool dump_prim = ( dump_ & 0x2 ) != 0u ; 

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

         AABB prim_bb = {} ;
         copyPrimNodes(prim_bb, spr ); 
         dpr->setAABB( prim_bb.data() ); 
         //dpr->setAABB( spr->AABB() );  // will not be so with selection 

         unsigned mismatch = 0 ; 
         std::string cf = AABB::Compare(mismatch, spr->AABB(), prim_bb.data(), 1, 1e-6 ) ; 
         if ( dump_prim && mismatch > 0 )
         {
             std::cout << std::endl ;  
             std::cout << "spr " << spr->desc() << std::endl ; 
             std::cout << "dpr " << dpr->desc() << std::endl ; 
             std::cout << "prim_bb " << std::setw(20) << " " << prim_bb.desc() << std::endl ; 
             std::cout << " AABB::Compare " << cf << std::endl ; 
         }

         solid_bb.include_aabb(prim_bb.data()); 
    }   // over prim of the solid
}


/**
CSGCopy::copyPrimNodes
-------------------------

**/

void CSGCopy::copyPrimNodes(AABB& prim_bb, const CSGPrim* spr )
{
    for(int nodeIdx=spr->nodeOffset() ; nodeIdx < spr->nodeOffset()+spr->numNode() ; nodeIdx++)
    {
        copyNode( prim_bb, nodeIdx ); 
    }   
}

/**
CSGCopy::copyNode
--------------------

see tail of CSG_GGeo_Convert::convertNode which uses qat4::transform_aabb_inplace to change CSGNode aabb
also see tail of CSGFoundry::addDeepCopySolid

**/

void CSGCopy::copyNode(AABB& prim_bb, unsigned nodeIdx )
{
    unsigned dump_ = Dump(sSolidIdx); 
    bool dump_node = ( dump_ & 0x4 ) != 0u ; 

    const CSGNode* snd = src->getNode(nodeIdx); 
    unsigned stypecode = snd->typecode(); 
    unsigned sTranIdx = snd->gtransformIdx(); 
    bool has_planes = CSG::HasPlanes(stypecode) ; 
    bool has_transform = sTranIdx > 0u ; 

    std::vector<float4> splanes ; 
    src->getNodePlanes(splanes, snd); 

    unsigned dTranIdx = 0u ; 
    const qat4* tra = nullptr ; 
    const qat4* itr = nullptr ; 

    if(has_transform)
    {
        tra = src->getTran(sTranIdx-1u) ; 
        itr = src->getItra(sTranIdx-1u) ; 
        dTranIdx = 1u + dst->addTran( tra, itr ) ;
    }
       
    CSGNode nd = {} ;
    CSGNode::Copy(nd, *snd );   // dumb straight copy : so need to fix transform and plan references  
    nd.setTransform( dTranIdx ); 

    CSGNode* dnd = dst->addNode(nd, &splanes);   

    if( elv == nullptr )
    {
        assert( dnd->planeNum() == snd->planeNum() );  
        assert( dnd->planeIdx() == snd->planeIdx() ); 
    }

    bool negated = dnd->is_complemented_primitive();
    bool zero = dnd->typecode() == CSG_ZERO ; 
    bool include_bb = negated == false && zero == false ; 

    float* naabb = dnd->AABB();   

    if(include_bb) 
    {
        dnd->setAABBLocal() ;         // reset to local with no transform applied
        if(tra)
        {
            tra->transform_aabb_inplace( naabb );
        }
        prim_bb.include_aabb( naabb );
    } 

    //if(dump_node) LOG(LEVEL) 
    if(dump_node) std::cout 
        << " nd " << std::setw(6) << nodeIdx 
        << " tc " << std::setw(4) << stypecode 
        << " st " << std::setw(4) << sTranIdx 
        << " dt " << std::setw(4) << dTranIdx 
        << " cn " << std::setw(12) << CSG::Name(stypecode) 
        << " hp " << has_planes
        << " ht " << has_transform
        << " ng " << negated
        << " ib " << include_bb
        << " bb " << AABB::Desc(naabb) 
        << std::endl 
        ; 
}


/**
CSGCopy::copySolidInstances
-------------------------------

As some solids may disappear as a result of Prim selection 
it is necessary to change potentially all the inst solid references. 

**/

void CSGCopy::copySolidInstances()
{
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

