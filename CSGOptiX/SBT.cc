
#include <iostream>
#include <iomanip>
#include <cstring>
#include <sstream>
#include <fstream>

#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>

#include "Opticks.hh"

#include "sutil_vec_math.h"  

#include "OPTIX_CHECK.h"
#include "CUDA_CHECK.h"

#include "CSGFoundry.h"
#include "CSGSolid.h"
#include "CSGNode.h"

#include "Binding.h"
#include "Params.h"

#include "GAS.h"
#include "GAS_Builder.h"

#include "IAS.h"
#include "IAS_Builder.h"

#include "PIP.h"
#include "SBT.h"

#include "CU.h"

#include "PLOG.hh"


/**
SBT
====

SBT needs PIP as the packing of SBT record headers requires 
access to their corresponding program groups (PGs).  
This is one aspect of establishing the connection between the 
PGs and their data.

**/

SBT::SBT(const Opticks* ok_, const PIP* pip_)
    :
    ok(ok_),
    solid_selection(ok->getSolidSelection()),
    emm(ok->getEMM()),
    pip(pip_),
    raygen(nullptr),
    miss(nullptr),
    hitgroup(nullptr),
    check(nullptr),
    foundry(nullptr),
    top(nullptr)
{
    init(); 
}

void SBT::init()
{
    LOG(info); 
    createRaygen();
    updateRaygen();
    createMiss();
    updateMiss(); 
}


/**
SBT::setFoundry
------------------

1. creates GAS using aabb obtained via geo
2. creates IAS
3. creates Hitgroup SBT records

**/

void SBT::setFoundry(const CSGFoundry* foundry_)
{
    foundry = foundry_ ; 
    createGeom(); 
}

void SBT::createGeom()
{
    LOG(info) << "[" ; 
    createGAS();  
    createIAS(); 
    createHitgroup(); 
    checkHitgroup(); 
    LOG(info) << "]" ; 
}


/**
SBT::createMiss
--------------------

NB the records have opaque header and user data
**/

void SBT::createMiss()
{
    LOG(info); 
    miss = new Miss ; 
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_miss ), sizeof(Miss) ) );
    sbt.missRecordBase = d_miss;
    OPTIX_CHECK( optixSbtRecordPackHeader( pip->miss_pg, miss ) );

    sbt.missRecordStrideInBytes     = sizeof( Miss );
    sbt.missRecordCount             = 1;
}

void SBT::updateMiss()
{
    //float3 purple = make_float3(0.3f, 0.1f, 0.5f); 
    //float3 white = make_float3( 1.0f, 1.0f, 1.0f); 
    //float3 lightgrey = make_float3( 0.9f, 0.9f, 0.9f); 
    float3 midgrey = make_float3( 0.6f, 0.6f, 0.6f); 
    const float3& bkg = midgrey  ; 
   
    miss->data.r = bkg.x ;
    miss->data.g = bkg.y ;
    miss->data.b = bkg.z ;

    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( d_miss ),
                miss,
                sizeof(Miss),
                cudaMemcpyHostToDevice
                ) );
}

/**
Raygen is typedef to SbtRecord<RaygenData> 
so this is setting up access to raygen data : but that 
is just a placeholder with most everything coming from params 
**/

void SBT::createRaygen()
{
    raygen = new Raygen ; 
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_raygen ),   sizeof(Raygen) ) );
    sbt.raygenRecord = d_raygen;
    OPTIX_CHECK( optixSbtRecordPackHeader( pip->raygen_pg,   raygen ) );
}

void SBT::updateRaygen()
{
    LOG(info); 
    raygen->data = {};
    raygen->data.placeholder = 42.0f ;

    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( d_raygen ),
                raygen,
                sizeof( Raygen ),
                cudaMemcpyHostToDevice
                ) );
}


/**
SBT::createGAS
----------------

For each compound shape the aabb of each prim (aka layer) is 
uploaded to GPU in order to create GAS for each compound shape.

Note that the prim could be a CSG tree of constituent nodes each 
with their own aabb, but only one aabb corresponding to the overall 
prim extent is used.

**/

void SBT::createGAS()  
{
    if( solid_selection.size() == 0  )
    {
        createGAS_Standard();  
    }
    else
    {
        createGAS_Selection();  
    }
}

void SBT::createGAS_Standard()
{ 
    unsigned num_solid = foundry->getNumSolid();   // STANDARD_SOLID
    for(unsigned i=0 ; i < num_solid ; i++)
    {
        unsigned gas_idx = i ; 
        bool enabled = ok->isEnabledMergedMesh(gas_idx) ;
        bool enabled2 = emm & ( 0x1 << gas_idx ) ;  
        assert( enabled == enabled2 );  

        if( enabled )
        {
            LOG(info) << " emm proceed " << gas_idx ; 
            createGAS(gas_idx); 
        }
        else
        {
            LOG(error) << " emm skip " << gas_idx ; 
        }
    }
    LOG(info) << descGAS() ; 
}

void SBT::createGAS_Selection()
{ 
    for(unsigned i=0 ; i < solid_selection.size() ; i++)
    {
        unsigned gas_idx = solid_selection[i] ; 
        createGAS(gas_idx); 
    }
}
 

void SBT::createGAS(unsigned gas_idx)
{
    CSGPrimSpec ps = foundry->getPrimSpec(gas_idx); 
    GAS gas = {} ;  
    GAS_Builder::Build(gas, ps);
    vgas[gas_idx] = gas ;  
}

const GAS& SBT::getGAS(unsigned gas_idx) const 
{
    unsigned count = vgas.count(gas_idx); 
    assert( count < 2 ); 
    if( count == 0 ) LOG(fatal) << " no such gas_idx " << gas_idx ; 
    return vgas.at(gas_idx); 
}



void SBT::createIAS()
{
    if( solid_selection.size() == 0  )
    {
        createIAS_Standard(); 
    }
    else
    {
        createIAS_Selection();
    }
}

void SBT::createIAS_Standard()
{
    unsigned num_ias = foundry->getNumUniqueIAS() ; 
    for(unsigned i=0 ; i < num_ias ; i++)
    {
        unsigned ias_idx = foundry->ias[i]; 
        createIAS(ias_idx); 
    }
}

void SBT::createIAS(unsigned ias_idx)
{
    unsigned num_inst = foundry->getNumInst(); 
    unsigned num_ias_inst = foundry->getNumInstancesIAS(ias_idx, emm); 
    LOG(info) << " ias_idx " << ias_idx << " num_inst " << num_inst ;  

    std::vector<qat4> inst ; 
    foundry->getInstanceTransformsIAS(inst, ias_idx, emm );
    assert( num_ias_inst == inst.size() ); 

    createIAS(inst); 
}

void SBT::createIAS_Selection()
{
    unsigned ias_idx = 0 ; 
    createSolidSelectionIAS( ias_idx, solid_selection ); 
}

void SBT::createSolidSelectionIAS(unsigned ias_idx, const std::vector<unsigned>& solid_selection)
{
    unsigned num_select = solid_selection.size() ; 
    assert( num_select > 0 ); 
    float mxe = foundry->getMaxExtent(solid_selection); 

    std::vector<qat4> inst ; 
    unsigned ins_idx = 0 ; 
    unsigned middle = num_select/2 ; 

    for(unsigned i=0 ; i < num_select ; i++)
    {
        unsigned gas_idx = solid_selection[i] ; 
        int ii = int(i) - int(middle) ; 

        qat4 q ; 
        q.setIdentity(ins_idx, gas_idx, ias_idx );
        q.q3.f.x = 2.0*mxe*float(ii) ;   

        inst.push_back(q); 
        ins_idx += 1 ; 
    }
    createIAS(inst); 
}


void SBT::createIAS(const std::vector<qat4>& inst )
{
    IAS ias = {} ;  
    IAS_Builder::Build(ias, inst, this );
    vias.push_back(ias);  
}


const IAS& SBT::getIAS(unsigned ias_idx) const 
{
    bool in_range =  ias_idx < vias.size() ; 
    if(!in_range) LOG(fatal) << "OUT OF RANGE ias_idx " << ias_idx << " vias.size " << vias.size() ; 
    assert(in_range); 
    return vias[ias_idx]; 
}



AS* SBT::getTop() const 
{
    return top ; 
}

void SBT::setTop(const char* spec)
{
    AS* a = getAS(spec); 
    setTop(a); 
}
void SBT::setTop(AS* top_)
{   
    top = top_ ;
}

AS* SBT::getAS(const char* spec) const 
{
   assert( strlen(spec) > 1 );  
   char c = spec[0]; 
   assert( c == 'i' || c == 'g' );  
   int idx = atoi( spec + 1 );  

   LOG(info) << " spec " << spec << " c " << c << " idx " << idx  ; 

   AS* a = nullptr ; 
   if( c == 'i' )
   {   
       const IAS& ias = vias[idx]; 
       a = (AS*)&ias ; 
   }   
   else if( c == 'g' )
   {   
       const GAS& gas = getGAS(idx) ; 
       a = (AS*)&gas ; 
   }   
   return a ; 
}

/**
SBT::getOffset
----------------

The layer_idx_ within the shape_idx_ composite shape.
NB layer_idx is local to the solid. 

**/

unsigned SBT::getOffset(unsigned solid_idx_ , unsigned layer_idx_ ) const 
{
    unsigned offset_sbt = _getOffset(solid_idx_, layer_idx_ ); 
 
    bool dump = false ; 
    if(dump) std::cout 
        << "SBT::getOffset"
        << " solid_idx_ " << solid_idx_
        << " layer_idx_ " << layer_idx_
        << " offset_sbt " << offset_sbt 
        << std::endl
        ;

    return offset_sbt ; 
}

/**
SBT::_getOffset
----------------

Implemented as an inner method avoiding "goto" 
to break out of multiple for loops.

**/
unsigned SBT::_getOffset(unsigned solid_idx_ , unsigned layer_idx_ ) const 
{
    unsigned offset_sbt = 0 ; 

    typedef std::map<unsigned, GAS>::const_iterator IT ; 
    for(IT it=vgas.begin() ; it !=vgas.end() ; it++)
    {
        unsigned gas_idx = it->first ; 
        const GAS& gas = it->second ; 

        //assert( ok->isEnabledMergedMesh(gas_idx) ); 
        unsigned num_bi = gas.bis.size(); 
        assert(num_bi == 1); 

        for(unsigned j=0 ; j < num_bi ; j++)
        { 
            const BI& bi = gas.bis[j] ; 
            const OptixBuildInputCustomPrimitiveArray& buildInputCPA = bi.buildInput.aabbArray ;
            unsigned num_sbt = buildInputCPA.numSbtRecords ; 

            for( unsigned k=0 ; k < num_sbt ; k++)
            { 
                //unsigned layer_idx = is_1NN ? j : k ;  
                unsigned layer_idx = k ;  
                if( solid_idx_ == gas_idx && layer_idx_ == layer_idx ) return offset_sbt ;
                offset_sbt += 1 ; 
            }
        }         
    }
    LOG(error) << "did not find targetted shape " ; 
    assert(0); 
    return offset_sbt ;  
}


unsigned SBT::getTotalRec() const 
{
    unsigned tot_bi = 0 ; 
    unsigned tot_rec = 0 ; 

    typedef std::map<unsigned, GAS>::const_iterator IT ; 
    for(IT it=vgas.begin() ; it !=vgas.end() ; it++)
    {
        unsigned gas_idx = it->first ; 
        const GAS& gas = it->second ; 
        //assert( ok->isEnabledMergedMesh(gas_idx) ); 
 
        unsigned num_bi = gas.bis.size(); 
        tot_bi += num_bi ; 
        for(unsigned j=0 ; j < num_bi ; j++)
        { 
            const BI& bi = gas.bis[j] ; 
            const OptixBuildInputCustomPrimitiveArray& buildInputCPA = bi.buildInput.aabbArray ;
            unsigned num_rec = buildInputCPA.numSbtRecords ; 
            tot_rec += num_rec ; 
        }         
    }
    assert( tot_bi > 0 && tot_rec > 0 );  
    return tot_rec ;  
}


/**
SBT::descGAS
--------------

Description of the sbt record counts per GAS, which corresponds 
to the number of prim per solid for all solids.
This is meaningful after createGAS.

**/

std::string SBT::descGAS() const 
{
    unsigned tot_rec = 0 ; 
    unsigned tot_bi = 0 ; 
    std::stringstream ss ; 
    ss 
        << "SBT::descGAS"
        << " num_gas " << vgas.size() 
        << " bi.numRec ( " 
        ;

    typedef std::map<unsigned, GAS>::const_iterator IT ; 
    for(IT it=vgas.begin() ; it !=vgas.end() ; it++)
    {
        unsigned gas_idx = it->first ; 
        const GAS& gas = it->second ; 
        //assert( ok->isEnabledMergedMesh(gas_idx) ); 

        unsigned num_bi = gas.bis.size(); 
        tot_bi += num_bi ; 
        for(unsigned j=0 ; j < num_bi ; j++)
        { 
            const BI& bi = gas.bis[j] ; 
            const OptixBuildInputCustomPrimitiveArray& buildInputCPA = bi.buildInput.aabbArray ;
            unsigned num_rec = buildInputCPA.numSbtRecords ; 
            ss << num_rec << " " ;  
            tot_rec += num_rec ; 
        }         
    }

    ss << " ) "
       << " tot_rec " << tot_rec 
       << " tot_bi " << tot_bi 
       ;

    std::string s = ss.str(); 
    return s ; 
}



/**
SBT::createHitgroup
---------------------

Note:

1. all HitGroup SBT records have the same hitgroup_pg, different shapes 
   are distinguished by program data not program code 


**/

void SBT::createHitgroup()
{
    unsigned num_solid = foundry->getNumSolid(); 
    unsigned num_gas = vgas.size(); 
    //assert( num_gas == num_solid );   // not when emm active : then there are less gas than solid
    unsigned tot_rec = getTotalRec(); 

    LOG(info)
        << " num_solid " << num_solid 
        << " num_gas " << num_gas 
        << " tot_rec " << tot_rec 
        ; 

    hitgroup = new HitGroup[tot_rec] ; 
    HitGroup* hg = hitgroup ; 


    for(unsigned i=0 ; i < tot_rec ; i++)   // pack headers CPU side
         OPTIX_CHECK( optixSbtRecordPackHeader( pip->hitgroup_pg, hitgroup + i ) ); 
    
    unsigned sbt_offset = 0 ; 


    typedef std::map<unsigned, GAS>::const_iterator IT ; 
    for(IT it=vgas.begin() ; it !=vgas.end() ; it++)
    {
        unsigned gas_idx = it->first ; 
        const GAS& gas = it->second ; 
        //assert( ok->isEnabledMergedMesh(gas_idx) ); 
        unsigned num_bi = gas.bis.size(); 

        const CSGSolid* so = foundry->getSolid(gas_idx) ;
        int numPrim = so->numPrim ; 
        int primOffset = so->primOffset ; 

        LOG(info) << "gas_idx " << gas_idx << " so.numPrim " << numPrim << " so.primOffset " << primOffset  ; 

        for(unsigned j=0 ; j < num_bi ; j++)
        { 
            const BI& bi = gas.bis[j] ; 
            const OptixBuildInputCustomPrimitiveArray& buildInputCPA = bi.buildInput.aabbArray ;
            unsigned num_rec = buildInputCPA.numSbtRecords ; 
            assert( num_rec == numPrim ) ; 

            for( unsigned k=0 ; k < num_rec ; k++)
            { 
                unsigned localPrimIdx = k ;   
                unsigned globalPrimIdx = primOffset + localPrimIdx ;   
                const CSGPrim* prim = foundry->getPrim( globalPrimIdx ); 
                setPrimData( hg->data, prim ); 

                unsigned check_sbt_offset = getOffset(gas_idx, localPrimIdx ); 

                /*
                std::cout 
                    << "SBT::createHitgroup "
                    << " gas(i) " << i 
                    << " bi(j) " << j
                    << " sbt(k) " << k 
                    << " gas_idx " << gas_idx 
                    << " localPrimIdx " << localPrimIdx 
                    << " globalPrimIdx " << globalPrimIdx 
                    << " check_sbt_offset " << check_sbt_offset
                    << " sbt_offset " << sbt_offset
                    << std::endl 
                    ; 

                */
                assert( check_sbt_offset == sbt_offset  ); 

                hg++ ; 
                sbt_offset++ ; 
            }
        }
    }

    CUDA_CHECK( cudaMalloc(reinterpret_cast<void**>( &d_hitgroup ), sizeof(HitGroup)*tot_rec ));
    CUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>( d_hitgroup ), hitgroup, sizeof(HitGroup)*tot_rec, cudaMemcpyHostToDevice ));

    sbt.hitgroupRecordBase  = d_hitgroup;
    sbt.hitgroupRecordStrideInBytes = sizeof(HitGroup);
    sbt.hitgroupRecordCount = tot_rec ;
}

void SBT::setPrimData( HitGroupData& data, const CSGPrim* prim)
{
    data.numNode = prim->numNode(); 
    data.nodeOffset = prim->nodeOffset();  
}

void SBT::checkPrimData( HitGroupData& data, const CSGPrim* prim)
{
    assert( data.numNode == prim->numNode() ); 
    assert( data.nodeOffset == prim->nodeOffset() );  
}
void SBT::dumpPrimData( const HitGroupData& data ) const 
{
    std::cout 
        << "SBT::dumpPrimData"
        << " data.numNode " << data.numNode
        << " data.nodeOffset " << data.nodeOffset
        << std::endl 
        ; 
}

void SBT::checkHitgroup()
{
    unsigned num_solid = foundry->getNumSolid(); 
    unsigned num_prim = foundry->getNumPrim(); 
    unsigned num_sbt = sbt.hitgroupRecordCount ;

    LOG(info)
        << " num_sbt (sbt.hitgroupRecordCount) " << num_sbt
        << " num_solid " << num_solid
        << " num_prim " << num_prim
        ; 

    //assert( num_prim == num_sbt );   // not with emm enabled

   // hmm this is not so easy with skips

/*
    check = new HitGroup[num_prim] ; 
    CUDA_CHECK( cudaMemcpy(check, reinterpret_cast<void*>( sbt.hitgroupRecordBase ), sizeof( HitGroup )*num_sbt, cudaMemcpyDeviceToHost ));
    HitGroup* hg = check ; 
    for(unsigned i=0 ; i < num_sbt ; i++)
    {
        unsigned globalPrimIdx = i ; 
        const CSGPrim* prim = foundry->getPrim(globalPrimIdx);         
        checkPrimData( hg->data, prim ); 
        hg++ ; 
    }
    delete [] check ; 

*/

}

