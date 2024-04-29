
#include <iostream>
#include <iomanip>
#include <cstring>
#include <csignal>
#include <sstream>
#include <fstream>

#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>

#include "SGeoConfig.hh"
#include "SSys.hh"
#include "SVec.hh"
#include "scuda.h"  
#include "SScene.h"

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
#include "Properties.h"

#include "CU.h"
#include "SLOG.hh"

/**
SBT
====

SBT needs PIP as the packing of SBT record headers requires 
access to their corresponding program groups (PGs).  
This is one aspect of establishing the connection between the 
PGs and their data.

**/

const plog::Severity SBT::LEVEL = SLOG::EnvLevel("SBT", "DEBUG"); 


SBT::SBT(const PIP* pip_)
    :
    emm(SGeoConfig::EnabledMergedMesh()), 
    pip(pip_),
    properties(pip->properties),
    raygen(nullptr),
    miss(nullptr),
    hitgroup(nullptr),
    check(nullptr),
    foundry(nullptr),
    scene(nullptr)
{
    init(); 
}

SBT::~SBT()
{
    destroy(); 
}


void SBT::init()
{
    LOG(LEVEL) << "[" ; 
    createRaygen();
    updateRaygen();
    createMiss();
    updateMiss(); 
    LOG(LEVEL) << "]" ; 
}


void SBT::destroy()
{
    destroyRaygen(); 
    destroyMiss(); 
    destroyHitgroup(); 
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
    foundry = foundry_ ;          // analytic
    scene = foundry->getScene();  // triangulated

    createGeom(); 
}

/**
SBT::createGeom
-----------------

createGAS 
    SCSGPrimSpec for each compound solid are converted to GAS and collected into map 
createIAS
    instance transforms with compound solid references are converted into the IAS
createHitgroup    
    bringing it all together

**/
void SBT::createGeom()
{
    LOG(LEVEL) << "[" ; 
    createGAS();    
    LOG(LEVEL) << "] createGAS " ; 
    createIAS(); 
    LOG(LEVEL) << "] createIAS " ; 
    createHitgroup(); 
    LOG(LEVEL) << "] createHitGroup " ; 
    checkHitgroup(); 
    LOG(LEVEL) << "] checkHitGroup " ; 
    LOG(LEVEL) << "]" ; 
}


/**
SBT::createMiss
--------------------

NB the records have opaque header and user data
**/

void SBT::createMiss()
{
    LOG(LEVEL); 
    miss = new Miss ; 
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_miss ), sizeof(Miss) ) );
    sbt.missRecordBase = d_miss;
    OPTIX_CHECK( optixSbtRecordPackHeader( pip->miss_pg, miss ) );

    sbt.missRecordStrideInBytes     = sizeof( Miss );
    sbt.missRecordCount             = 1;
}

void SBT::destroyMiss()
{
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_miss ) ) );
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

void SBT::destroyRaygen()
{
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_raygen ) ) );
}





void SBT::updateRaygen()
{
    LOG(LEVEL); 
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
    unsigned num_solid = foundry->getNumSolid();   // STANDARD_SOLID
    for(unsigned i=0 ; i < num_solid ; i++)
    {
        unsigned gas_idx = i ; 

        bool enabled = SGeoConfig::IsEnabledMergedMesh(gas_idx) ;
        bool enabled2 = emm & ( 0x1 << gas_idx ) ;  
        bool enabled_expect = enabled == enabled2 ;  
        assert( enabled_expect );  
        if(!enabled_expect) std::raise(SIGINT); 

        if( enabled )
        {
            LOG(LEVEL) << " emm proceed " << gas_idx ; 
            createGAS(gas_idx); 
        }
        else
        {
            LOG(LEVEL) << " emm skip " << gas_idx ; 
        }
    }
    LOG(LEVEL) << descGAS() ; 
}


/**
SBT::createGAS 
---------------

1. gets SCSGPrimSpec for a the *gas_idx* compound solid from foundry 
2. converts the SCSGPrimSpec into a GAS, passing in bbox device array pointers
3. inserts gas into vgas map using *gas_idx* key 


Wheeling in triangulated in one go is too great a leap.
Instead stay purely analytic and try adopting SOPTIX_Accel
instead of GAS and IAS. 

Triangluated will need smth like::

    SMeshGroup* mg = scene->meshgroup[gas_idx] ;
    SOPTIX_MeshGroup* xmg = SOPTIX_MeshGroup::Create( mg ) ;
    SOPTIX_Accel* _gas = new SOPTIX_Accel( ctx->context, xmg->buildInputs );  


**/
void SBT::createGAS(unsigned gas_idx)
{
#ifdef WITH_SOPTIX_ACCEL

    SCSGPrimSpec ps = foundry->getPrimSpec(gas_idx); 

    SOPTIX_BuildInput* cpa = new SOPTIX_BuildInput_CPA(ps) ; 
    SOPTIX_Accel* gas = SOPTIX_Accel::Create(Context::ctx, &(cpa->buildInput), 1 );  
    gas->bi = cpa ;  

    vgas[gas_idx] = gas ;  

#else
    GAS gas = {} ;  
    SCSGPrimSpec ps = foundry->getPrimSpec(gas_idx); 
    GAS_Builder::Build(gas, ps);
    vgas[gas_idx] = gas ;  
#endif

}



OptixTraversableHandle SBT::getGASHandle(unsigned gas_idx) const
{
    unsigned count = vgas.count(gas_idx); 
    LOG_IF(fatal, count == 0) << " no such gas_idx " << gas_idx ; 
    assert( count == 1 ); 

#ifdef WITH_SOPTIX_ACCEL
    SOPTIX_Accel* _gas = vgas.at(gas_idx) ;
    OptixTraversableHandle handle = _gas->handle ; 
#else
    const GAS& gas = vgas.at(gas_idx); 
    OptixTraversableHandle handle = gas.handle ; 
#endif
 
    return handle ; 
}



void SBT::createIAS()
{
    unsigned num_ias = foundry->getNumUniqueIAS() ; 
    bool num_ias_expect = num_ias == 1 ; 
    assert( num_ias_expect );  
    if(!num_ias_expect) std::raise(SIGINT); 

    unsigned ias_idx = 0 ; 
    createIAS(ias_idx); 
}

/**
SBT::createIAS
----------------

Hmm: usually only one IAS. 

Seems like should be caching the inst used to construct the GPU geometry in use.  
For ease of lookup using the flat instance_id obtained from intersect identity.  

**/

void SBT::createIAS(unsigned ias_idx)
{
    unsigned num_inst = foundry->getNumInst(); 
    unsigned num_ias_inst = foundry->getNumInstancesIAS(ias_idx, emm); 
    LOG(LEVEL) 
        << " ias_idx " << ias_idx 
        << " num_inst " << num_inst 
        << " num_ias_inst(getNumInstancesIAS) " << num_ias_inst
        ;  

    std::vector<qat4> inst ; 
    foundry->getInstanceTransformsIAS(inst, ias_idx, emm );
    assert( num_ias_inst == inst.size() ); 

    createIAS(inst); 
    LOG(LEVEL) << descIAS(inst); 
}

/** 
SBT::dumpIAS
---------------

ins_idx flatly proceeds across the entire instanced geometry (actually the IAS but there is only one of those)

* the flat ins_idx can be used to lookup the tranform and its instrumented geometry info (gas_idx, ias_idx) 
  from the instances vector
* so bit packing the gas_idx into GPU side instanceId would be just a convenience to avoid having to do that lookup, 
  better to keep things as simple as possible GPU side and just provide CSGFoundry API to do that lookup 
  from the unadorned flat:: 

      unsigned instance_id = optixGetInstanceId() ;        // see IAS_Builder::Build and InstanceId.h 

  * will probably need to lookup the transform anyhow  

* BUT the ins_idx does not help to identify within the globals, all of them being in the first line with ins_idx 0 gas_idx 0 

:: 

    2021-08-22 22:51:53.931 INFO  [52005] [SBT::dumpIAS@289]  inst.size 46117 SBT_DUMP_IAS 1
     i          0 ins_idx          0 gas_idx          0 ias_idx          0
     i          1 ins_idx          1 gas_idx          1 ias_idx          0
     i          2 ins_idx          2 gas_idx          1 ias_idx          0
     i          3 ins_idx          3 gas_idx          1 ias_idx          0
     i          4 ins_idx          4 gas_idx          1 ias_idx          0
     i          5 ins_idx          5 gas_idx          1 ias_idx          0
     i          6 ins_idx          6 gas_idx          1 ias_idx          0
     ...
     i      25591 ins_idx      25591 gas_idx          1 ias_idx          0
     i      25592 ins_idx      25592 gas_idx          1 ias_idx          0
     i      25593 ins_idx      25593 gas_idx          1 ias_idx          0
     i      25594 ins_idx      25594 gas_idx          1 ias_idx          0
     i      25595 ins_idx      25595 gas_idx          1 ias_idx          0
     i      25596 ins_idx      25596 gas_idx          1 ias_idx          0
     i      25597 ins_idx      25597 gas_idx          1 ias_idx          0
     i      25598 ins_idx      25598 gas_idx          1 ias_idx          0
     i      25599 ins_idx      25599 gas_idx          1 ias_idx          0
     i      25600 ins_idx      25600 gas_idx          1 ias_idx          0
     i      25601 ins_idx      25601 gas_idx          2 ias_idx          0
     i      25602 ins_idx      25602 gas_idx          2 ias_idx          0
     i      25603 ins_idx      25603 gas_idx          2 ias_idx          0
     i      25604 ins_idx      25604 gas_idx          2 ias_idx          0
     ...
     i      38208 ins_idx      38208 gas_idx          2 ias_idx          0
     i      38209 ins_idx      38209 gas_idx          2 ias_idx          0
     i      38210 ins_idx      38210 gas_idx          2 ias_idx          0
     i      38211 ins_idx      38211 gas_idx          2 ias_idx          0
     i      38212 ins_idx      38212 gas_idx          2 ias_idx          0
     i      38213 ins_idx      38213 gas_idx          3 ias_idx          0
     i      38214 ins_idx      38214 gas_idx          3 ias_idx          0
     i      38215 ins_idx      38215 gas_idx          3 ias_idx          0
     i      38216 ins_idx      38216 gas_idx          3 ias_idx          0
     i      38217 ins_idx      38217 gas_idx          3 ias_idx          0
     i      38218 ins_idx      38218 gas_idx          3 ias_idx          0
     ...
     i      43206 ins_idx      43206 gas_idx          3 ias_idx          0
     i      43207 ins_idx      43207 gas_idx          3 ias_idx          0
     i      43208 ins_idx      43208 gas_idx          3 ias_idx          0
     i      43209 ins_idx      43209 gas_idx          3 ias_idx          0
     i      43210 ins_idx      43210 gas_idx          3 ias_idx          0
     i      43211 ins_idx      43211 gas_idx          3 ias_idx          0
     i      43212 ins_idx      43212 gas_idx          3 ias_idx          0
     i      43213 ins_idx      43213 gas_idx          4 ias_idx          0
     i      43214 ins_idx      43214 gas_idx          4 ias_idx          0
     i      43215 ins_idx      43215 gas_idx          4 ias_idx          0
     i      43216 ins_idx      43216 gas_idx          4 ias_idx          0
    ...
     i      45605 ins_idx      45605 gas_idx          4 ias_idx          0
     i      45606 ins_idx      45606 gas_idx          4 ias_idx          0
     i      45607 ins_idx      45607 gas_idx          4 ias_idx          0
     i      45608 ins_idx      45608 gas_idx          4 ias_idx          0
     i      45609 ins_idx      45609 gas_idx          4 ias_idx          0
     i      45610 ins_idx      45610 gas_idx          4 ias_idx          0
     i      45611 ins_idx      45611 gas_idx          4 ias_idx          0
     i      45612 ins_idx      45612 gas_idx          4 ias_idx          0
     i      45613 ins_idx      45613 gas_idx          5 ias_idx          0
     i      45614 ins_idx      45614 gas_idx          5 ias_idx          0
     i      45615 ins_idx      45615 gas_idx          5 ias_idx          0
     i      45616 ins_idx      45616 gas_idx          5 ias_idx          0
     ..
     i      46112 ins_idx      46112 gas_idx          5 ias_idx          0
     i      46113 ins_idx      46113 gas_idx          5 ias_idx          0
     i      46114 ins_idx      46114 gas_idx          5 ias_idx          0
     i      46115 ins_idx      46115 gas_idx          5 ias_idx          0
     i      46116 ins_idx      46116 gas_idx          5 ias_idx          0

      2021-08-22 22:49:12.346 INFO  [47848] [SBT::dumpIAS@289]  inst.size 46117 SBT_DUMP_IAS 0
     gas_idx          0 num_ins_idx          1 ins_idx_mn          0 ins_idx_mx          0 ins_idx_mx - ins_idx_mx + 1 (num_ins_idx2)          1
     gas_idx          1 num_ins_idx      25600 ins_idx_mn          1 ins_idx_mx      25600 ins_idx_mx - ins_idx_mx + 1 (num_ins_idx2)      25600
     gas_idx          2 num_ins_idx      12612 ins_idx_mn      25601 ins_idx_mx      38212 ins_idx_mx - ins_idx_mx + 1 (num_ins_idx2)      12612
     gas_idx          3 num_ins_idx       5000 ins_idx_mn      38213 ins_idx_mx      43212 ins_idx_mx - ins_idx_mx + 1 (num_ins_idx2)       5000
     gas_idx          4 num_ins_idx       2400 ins_idx_mn      43213 ins_idx_mx      45612 ins_idx_mx - ins_idx_mx + 1 (num_ins_idx2)       2400
     gas_idx          5 num_ins_idx        504 ins_idx_mn      45613 ins_idx_mx      46116 ins_idx_mx - ins_idx_mx + 1 (num_ins_idx2)        504
    2021-08-22 22:49:12.352 INFO  [47848] [SBT::createHitgroup@645]  num_solid 6 num_gas 6 tot_rec 2473

**/

std::string SBT::descIAS(const std::vector<qat4>& inst ) const 
{
    std::stringstream ss ; 
    bool sbt_dump_ias = SSys::getenvbool("SBT_DUMP_IAS") ; 
    ss
        << "SBT::descIAS"
        << " inst.size " << inst.size()
        << " SBT_DUMP_IAS " << sbt_dump_ias 
        << std::endl 
        ; 

    typedef std::map<int, std::vector<int>> MUV ; 
    MUV ins_idx_per_gas ; 

    for(unsigned i=0 ; i < inst.size() ; i++)
    {
        const qat4& q = inst[i] ;   
        int ins_idx,  gas_idx, sensor_identifier, sensor_index ;
        q.getIdentity(ins_idx,  gas_idx, sensor_identifier, sensor_index );

        // collect ins_idx for each gas_idx 
        ins_idx_per_gas[gas_idx].push_back(ins_idx); 

        if(sbt_dump_ias) ss 
           << " i "       << std::setw(10) << i 
           << " ins_idx " << std::setw(10) << ins_idx 
           << " gas_idx " << std::setw(10) << gas_idx 
           << " sensor_identifier " << std::setw(10) << sensor_identifier
           << " sensor_index " << std::setw(10) << sensor_index
           << std::endl
           ; 
    }

    MUV::const_iterator b = ins_idx_per_gas.begin(); 
    MUV::const_iterator e = ins_idx_per_gas.end(); 
    MUV::const_iterator i ; 

    for( i=b ; i != e ; i++)
    {
        int gas_idx = i->first ; 
        const std::vector<int>& v = i->second ; 
        int num_ins_idx = int(v.size()) ; 

        int ins_idx_mn, ins_idx_mx ; 
        SVec<int>::MinMax(v, ins_idx_mn, ins_idx_mx)  ; 
        int num_ins_idx2 = ins_idx_mx - ins_idx_mn + 1 ; 

        ss
            << " gas_idx " << std::setw(10) <<  gas_idx 
            << " num_ins_idx " << std::setw(10) << num_ins_idx
            << " ins_idx_mn " << std::setw(10) << ins_idx_mn
            << " ins_idx_mx " << std::setw(10) << ins_idx_mx
            << " ins_idx_mx - ins_idx_mx + 1 (num_ins_idx2) " << std::setw(10) << num_ins_idx2 
            << std::endl 
            ; 

        assert( num_ins_idx == num_ins_idx2 ); 
    }
    std::string s = ss.str(); 
    return s ; 
}



/**
SBT::createIAS
----------------

This is taking 0.43s for 48477 inst from JUNO full geometry. 

**/

void SBT::createIAS(const std::vector<qat4>& inst )
{
    LOG(LEVEL) << "[ inst.size " << inst.size() ; 

#ifdef WITH_SOPRIX_ACCEL

#else
    IAS ias = {} ;  
    IAS_Builder::Build(ias, inst, this );
    vias.push_back(ias);  
#endif

    LOG(LEVEL) << "] inst.size " << inst.size() ; 
}




OptixTraversableHandle SBT::getIASHandle(unsigned ias_idx) const
{
    assert( ias_idx < vias.size() ); 

#ifdef WITH_SOPTIX_ACCEL
    SOPTIX_Accel* _ias = vias[ias_idx] ;
    OptixTraversableHandle handle = _ias->handle ; 
#else
    const IAS& ias = vias[ias_idx]; 
    OptixTraversableHandle handle = ias.handle ; 
#endif
    return handle ; 
} 


OptixTraversableHandle SBT::getTOPHandle() const 
{
    return getIASHandle(0);  
}



/**
SBT::getOffset
----------------

Canonically invoked from IAS_Builder::CollectInstances

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

Iterates over vgas GAS map in *gas_idx* key order 0,1,2,.. and within 
each GAS iterates over the "layers" (aka CSGPrim of the CSGSolid)
counting the number of *sbt* records encountered until reach (solid_idx_, layer_idx_)
at which point returns *offset_sbt*. 

This assumes(implies) that only enabled mergedmesh have 
vgas entries.  



HMM: currently this gets invoked for every instance, costing 0.42s for 48477 inst
when only really need to traverse them all once and keep a record of 
for subsequent lookup.  

   solid_idx_ layer_idx_ offset_sbt 

So that 0.42s can be made to go to zero by doing this once. 

**/
unsigned SBT::_getOffset(unsigned solid_idx_ , unsigned layer_idx_ ) const 
{
    unsigned offset_sbt = 0 ; 

    typedef std::map<unsigned, GAS>::const_iterator IT ; 
    for(IT it=vgas.begin() ; it !=vgas.end() ; it++)
    {
        unsigned gas_idx = it->first ; 
        const GAS& gas = it->second ; 

        unsigned num_bi = gas.bis.size(); 
        assert(num_bi == 1); 

        for(unsigned j=0 ; j < num_bi ; j++)
        { 
            const BI& bi = gas.bis[j] ; 
            const OptixBuildInputCustomPrimitiveArray& buildInputCPA = bi.getBuildInputCPA() ;
            unsigned num_sbt = buildInputCPA.numSbtRecords ;  // <-- corresponding to bbox of the GAS

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

/**
SBT::getTotalRec
------------------

Returns the total number of SBT records for all layers (aka CSGPrim) 
of all GAS in the map. 

Corresponds to the total number of enabled Prim in all enabled solids.

**/

unsigned SBT::getTotalRec() const 
{
    unsigned tot_bi = 0 ; 
    unsigned tot_rec = 0 ; 

    typedef std::map<unsigned, GAS>::const_iterator IT ; 
    for(IT it=vgas.begin() ; it !=vgas.end() ; it++)
    {
        unsigned gas_idx = it->first ; 

        bool enabled = SGeoConfig::IsEnabledMergedMesh(gas_idx)  ; 
        LOG_IF(error, !enabled) << "gas_idx " << gas_idx << " enabled " << enabled ; 


        const GAS& gas = it->second ; 
 
        unsigned num_bi = gas.bis.size(); 
        tot_bi += num_bi ; 
        for(unsigned j=0 ; j < num_bi ; j++)
        { 
            const BI& bi = gas.bis[j] ; 
            const OptixBuildInputCustomPrimitiveArray& buildInputCPA = bi.getBuildInputCPA() ;
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

        bool enabled = SGeoConfig::IsEnabledMergedMesh(gas_idx)  ; 
        LOG_IF(error, !enabled) << "gas_idx " << gas_idx << " enabled " << enabled ; 

        unsigned num_bi = gas.bis.size(); 
        tot_bi += num_bi ; 
        for(unsigned j=0 ; j < num_bi ; j++)
        { 
            const BI& bi = gas.bis[j] ; 
            const OptixBuildInputCustomPrimitiveArray& buildInputCPA = bi.getBuildInputCPA() ;
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

The hitgroup array has records for all active Prims of all active Solid.
The records hold (numNode, nodeOffset) of all those active Prim.  


Note:

1. all HitGroup SBT records have the same hitgroup_pg, different shapes 
   are distinguished by program data not program code 


Prim Selection
~~~~~~~~~~~~~~~~

Thoughts on how to implement Prim selection with CSGPrim::MakeSpec

**/

void SBT::createHitgroup()
{
    unsigned num_solid = foundry->getNumSolid(); 
    unsigned num_gas = vgas.size(); 
    //assert( num_gas == num_solid );   // not when emm active : then there are less gas than solid
    unsigned tot_rec = getTotalRec();   // corresponds to the total number of enabled Prim in all enabled solids

    LOG(LEVEL)
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

        unsigned num_bi = gas.bis.size(); 
        assert( num_bi == 1 ); 
         
        const CSGSolid* so = foundry->getSolid(gas_idx) ;
        int numPrim = so->numPrim ; 
        int primOffset = so->primOffset ; 

        LOG(LEVEL) << "gas_idx " << gas_idx << " so.numPrim " << numPrim << " so.primOffset " << primOffset  ; 

        for(unsigned j=0 ; j < num_bi ; j++)
        { 
            const BI& bi = gas.bis[j] ; 
            // Q: is there a bi for each node ?
            // A: NO, roughly speaking the bi hold the bbox references for all CSGPrim of the solid(=GAS) 

            const OptixBuildInputCustomPrimitiveArray& buildInputCPA = bi.getBuildInputCPA() ;
            unsigned num_rec = buildInputCPA.numSbtRecords ; 
            assert( num_rec == unsigned(numPrim) ) ; 

            for( unsigned k=0 ; k < num_rec ; k++)
            { 
                unsigned localPrimIdx = k ;   
                unsigned globalPrimIdx = primOffset + localPrimIdx ;   
                const CSGPrim* prim = foundry->getPrim( globalPrimIdx ); 

                setPrimData( hg->data, prim );  // copy numNode, nodeOffset from CSGPrim into hg->data

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

                bool sbt_offset_expect = check_sbt_offset == sbt_offset ; 
                assert( sbt_offset_expect  ); 
                if(!sbt_offset_expect) std::raise(SIGINT); 

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

void SBT::destroyHitgroup()
{
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_hitgroup ) ) );
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

    LOG(LEVEL)
        << " num_sbt (sbt.hitgroupRecordCount) " << num_sbt
        << " num_solid " << num_solid
        << " num_prim " << num_prim
        ; 

}

