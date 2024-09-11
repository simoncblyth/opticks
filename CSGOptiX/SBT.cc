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
#include "NPX.h"

#include "OPTIX_CHECK.h"
#include "CUDA_CHECK.h"

#include "CSGFoundry.h"
#include "CSGSolid.h"
#include "CSGNode.h"

#include "Binding.h"
#include "Params.h"
#include "Ctx.h"

#include "PIP.h"
#include "SBT.h"
#include "Properties.h"

#include "CU.h"
#include "SLOG.hh"

#ifdef WITH_SOPTIX_ACCEL
#include "SOPTIX_Accel.h"
#include "SOPTIX_BuildInput_CPA.h"
#include "SOPTIX_BuildInput_IA.h"
#include "SOPTIX_BuildInput_Mesh.h"
#include "SOPTIX_MeshGroup.h"
#else
#include "GAS.h"
#include "GAS_Builder.h"
#include "IAS.h"
#include "IAS_Builder.h"
#endif



/**
SBT
====

SBT needs PIP as the packing of SBT record headers requires 
access to their corresponding program groups (PGs).  
This is one aspect of establishing the connection between the 
PGs and their data.

**/

const plog::Severity SBT::LEVEL = SLOG::EnvLevel("SBT", "DEBUG"); 


std::string SBT::Desc()  // static
{
    std::stringstream ss ; 
    ss << "SBT::Desc" 
#ifdef WITH_SOPTIX_ACCEL
       << " WITH_SOPTIX_ACCEL" 
#else
       << " NOT:WITH_SOPTIX_ACCEL"
#endif
       ;
    std::string str = ss.str(); 
    return str ; 
}



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
SBT::createRaygen
------------------

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
SBT::createMiss
--------------------

NB the records have opaque header and user data
**/

void SBT::createMiss()
{
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
SBT::setFoundry
------------------

Canonical invokation from CSGOptiX::CSGOptiX//CSGOptiX::initGeometry 

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
    LOG(LEVEL) << SGeoConfig::DescEMM() ; 

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

**/

#ifdef WITH_SOPTIX_ACCEL
void SBT::createGAS(unsigned gas_idx)
{
    SOPTIX_BuildInput* bi = nullptr ; 
    SOPTIX_Accel* gas = nullptr ; 

    bool trimesh = foundry->isSolidTrimesh(gas_idx); // now based on forced triangulation config 

    const std::string& mmlabel = foundry->getSolidMMLabel(gas_idx); 

    LOG(LEVEL) 
        << " WITH_SOPTIX_ACCEL "
        << " gas_idx " << gas_idx
        << " trimesh " << ( trimesh ? "YES" : "NO " )
        << " mmlabel " << mmlabel 
        ;

    if(trimesh)
    {
        // note similarity to SOPTIX_Scene::init_GAS
        const SMeshGroup* mg = scene->getMeshGroup(gas_idx) ;
        LOG_IF(fatal, mg == nullptr) 
            << " FAILED to SScene::getMeshGroup"
            << " gas_idx " << gas_idx 
            << "\n" 
            << scene->desc()
            ;
        assert(mg);   

        SOPTIX_MeshGroup* xmg = SOPTIX_MeshGroup::Create( mg ) ;
        gas = SOPTIX_Accel::Create(Ctx::context, xmg->bis );  
        xgas[gas_idx] = xmg ; 
    } 
    else
    {
        // analytic geometry 
        SCSGPrimSpec ps = foundry->getPrimSpec(gas_idx); 
        bi = new SOPTIX_BuildInput_CPA(ps) ; 
        gas = SOPTIX_Accel::Create(Ctx::context, bi );  
    }
    vgas[gas_idx] = gas ;  
}

#else
void SBT::createGAS(unsigned gas_idx)
{
    bool trimesh = foundry->isSolidTrimesh(gas_idx); 
    LOG(fatal, trimesh == true ) << " NOT:WITH_SOPTIX_ACCEL ONLY SUPPORTS ANALYTIC GEOMETRY : INVALID SGeoConfig::Trimesh setting " ; 
    assert( trimesh == false ); 

    SCSGPrimSpec ps = foundry->getPrimSpec(gas_idx); 
    GAS gas = {} ;  
    GAS_Builder::Build(gas, ps);
    vgas[gas_idx] = gas ;  
}
#endif



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

2024-04-30 11:08:33.056 INFO  [65240] [SBT::collectInstances@468] ] instances.size 47887
2024-04-30 11:08:33.056 INFO  [65240] [SBT::createIAS@372] SBT::descIAS inst.size 47887 SBT_DUMP_IAS 0
 gas_idx          0 num_ins_idx          1 ins_idx_mn          0 ins_idx_mx          0 ins_idx_mx - ins_idx_mx + 1 (num_ins_idx2)          1
 gas_idx          1 num_ins_idx      25600 ins_idx_mn          1 ins_idx_mx      25600 ins_idx_mx - ins_idx_mx + 1 (num_ins_idx2)      25600
 gas_idx          2 num_ins_idx      12615 ins_idx_mn      25601 ins_idx_mx      38215 ins_idx_mx - ins_idx_mx + 1 (num_ins_idx2)      12615
 gas_idx          3 num_ins_idx       4997 ins_idx_mn      38216 ins_idx_mx      43212 ins_idx_mx - ins_idx_mx + 1 (num_ins_idx2)       4997
 gas_idx          4 num_ins_idx       2400 ins_idx_mn      43213 ins_idx_mx      45612 ins_idx_mx - ins_idx_mx + 1 (num_ins_idx2)       2400
 gas_idx          5 num_ins_idx        590 ins_idx_mn      45613 ins_idx_mx      46202 ins_idx_mx - ins_idx_mx + 1 (num_ins_idx2)        590
 gas_idx          6 num_ins_idx        590 ins_idx_mn      46203 ins_idx_mx      46792 ins_idx_mx - ins_idx_mx + 1 (num_ins_idx2)        590
 gas_idx          7 num_ins_idx        590 ins_idx_mn      46793 ins_idx_mx      47382 ins_idx_mx - ins_idx_mx + 1 (num_ins_idx2)        590
 gas_idx          8 num_ins_idx        504 ins_idx_mn      47383 ins_idx_mx      47886 ins_idx_mx - ins_idx_mx + 1 (num_ins_idx2)        504


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


    collectInstances(inst); 
    
    LOG(LEVEL) << descIAS(inst); 

#ifdef WITH_SOPTIX_ACCEL
    SOPTIX_BuildInput* ia = new SOPTIX_BuildInput_IA(instances) ; 
    SOPTIX_Accel* ias = SOPTIX_Accel::Create(Ctx::context, ia );  
    vias.push_back(ias);  
#else
    IAS ias = {} ;  
    IAS_Builder::Build(ias, instances );
    vias.push_back(ias);  
#endif

}





/**
SBT::collectInstances
----------------------

Converts *inst* a vector of geometry identity instrumented transforms into
a vector of OptixInstance. The instance.sbtOffset are set using SBT::getOffset
for the gas_idx and with prim_idx:0 indicating the outer prim(aka layer) 
of the GAS.

Canonically invoked during CSGOptiX instanciation, from stack::

    CSGOptiX::CSGOptiX/CSGOptiX::init/CSGOptiX::initGeometry/SBT::setFoundry/SBT::createGeom/SBT::createIAS


Collecting OptixInstance was taking 0.42s for 48477 inst, 
as SBT::getOffset was being called for every instance. Instead 
of doing this caching the result in the gasIdx_sbtOffset brings
the time down to zero. 

HMM: Could make better use of instanceId, eg with bitpack gas_idx, ias_idx ?
See note in InstanceId.h its not so easy due to bit limits.  
But it doesnt matter much as can just do lookups CPU side based 
on simple indices from GPU side. 

**/


void SBT::collectInstances( const std::vector<qat4>& ias_inst ) 
{
    LOG(LEVEL) << "[ ias_inst.size " << ias_inst.size() ;  // eg 48477  

    unsigned num_ias_inst = ias_inst.size() ; 
    unsigned flags = OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT ;  
    unsigned prim_idx = 0u ;  // need sbt offset for the outer prim(aka layer) of the GAS 

    std::map<unsigned, unsigned> gasIdx_sbtOffset ;  

    for(unsigned i=0 ; i < num_ias_inst ; i++)
    {
        const qat4& q = ias_inst[i] ;   
        int ins_idx,  gasIdx, sensor_identifier, sensor_index ; 
        q.getIdentity(ins_idx, gasIdx, sensor_identifier, sensor_index );
        unsigned instanceId = q.get_IAS_OptixInstance_instanceId() ; 

        bool instanceId_is_allowed = instanceId < properties->limitMaxInstanceId ; 
        LOG_IF(fatal, !instanceId_is_allowed)
            << " instanceId " << instanceId 
            << " sbt->properties->limitMaxInstanceId " << properties->limitMaxInstanceId
            << " instanceId_is_allowed " << ( instanceId_is_allowed ? "YES" : "NO " )
            ; 
        assert( instanceId_is_allowed  ) ; 

        OptixTraversableHandle handle = getGASHandle(gasIdx); 

        bool found = gasIdx_sbtOffset.count(gasIdx) == 1 ; 
        unsigned sbtOffset = found ? gasIdx_sbtOffset.at(gasIdx) : getOffset(gasIdx, prim_idx ) ;
        if(!found) 
        {
            gasIdx_sbtOffset[gasIdx] = sbtOffset ; 
            LOG(LEVEL)
                << " i " << std::setw(7) << i 
                << " gasIdx " << std::setw(3) << gasIdx 
                << " sbtOffset " << std::setw(6) << sbtOffset 
                << " gasIdx_sbtOffset.size " << std::setw(3) << gasIdx_sbtOffset.size()
                << " instanceId " << instanceId
                ;
        }

        //unsigned visibilityMask = 255;  // cf SOPTIX_Scene::init_Instances 
        unsigned visibilityMask = properties->visibilityMask(gasIdx); 

        OptixInstance instance = {} ; 
        q.copy_columns_3x4( instance.transform ); 
        instance.instanceId = instanceId ;  
        instance.sbtOffset = sbtOffset ;            
        instance.visibilityMask = visibilityMask ; 

        instance.flags = flags ;
        instance.traversableHandle = handle ; 
    
        instances.push_back(instance); 
    }
    LOG(LEVEL) << "] instances.size " << instances.size() ; 
}

NP* SBT::serializeInstances() const 
{
    return NPX::ArrayFromVec<unsigned, OptixInstance>(instances) ; 
}


/**
SBT::descIAS (actually descINST would be more appropriate)
------------------------------------------------------------

1. traverse over *inst* collecting *ins_idx* for each gas into a map keyed on gas_idx *ins_idx_per_gas*
2. emit description of that map 

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

Canonically invoked from both::

   SBT::collectInstances
   SBT::createHitgroup

The q_layer_idx is 0-based within the q_gas_idx composite shape, 
ie it is local to the solid. 

**/

int SBT::getOffset(unsigned q_gas_idx, unsigned q_layer_idx ) const 
{
    int offset_sbt = _getOffset(q_gas_idx, q_layer_idx ); 
 
    LOG_IF(LEVEL, q_layer_idx < 10) 
        << " q_gas_idx " << q_gas_idx
        << " q_layer_idx " << q_layer_idx
        << " offset_sbt " << offset_sbt 
        ;

    assert( offset_sbt > -1 ); 
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

This assumes(implies) that only enabled mergedmesh have vgas entries.  


**/
int SBT::_getOffset(unsigned q_gas_idx , unsigned q_layer_idx ) const 
{
    int offset_sbt = 0 ; 

    for(IT it=vgas.begin() ; it !=vgas.end() ; it++)
    {
        unsigned gas_idx = it->first ; 
        bool trimesh = foundry->isSolidTrimesh(gas_idx); 
        const std::string& mmlabel = foundry->getSolidMMLabel(gas_idx); 
        const CSGSolid* so = foundry->getSolid(gas_idx) ;
        int numPrim = so->numPrim ; 

#ifdef WITH_SOPTIX_ACCEL
        SOPTIX_Accel* gas = it->second ; 
#else
        const GAS* gas = &(it->second) ;   
#endif

        int num_bi = gas->bis.size(); 
        LOG(debug) 
            << " gas_idx " << gas_idx 
            << " num_bi " << num_bi 
            << " trimesh " << ( trimesh ? "YES" : "NO " )
            << " mmlabel " << mmlabel 
            ;

        if(!trimesh) assert(num_bi == 1); 
        if(trimesh)
        {
            bool are_equal = num_bi == numPrim ; 
            LOG_IF(fatal, !are_equal )
                << " UNEXPECTED trimesh with  "  
                << " UNEQUAL: "
                << " num_bi " << num_bi 
                << " numPrim " << numPrim
                << " gas_idx " << gas_idx
                << " mmlabel " << mmlabel 
                ; 
            assert(are_equal ); 
        }

        for(int j=0 ; j < num_bi ; j++)
        {

#ifdef WITH_SOPTIX_ACCEL
            const SOPTIX_BuildInput* bi = gas->bis[j] ; 
            int num_sbt = bi->numSbtRecords() ; 
            if(!trimesh) assert( bi->is_BuildInputCustomPrimitiveArray() && num_sbt == numPrim ); 
            if(trimesh)  assert( bi->is_BuildInputTriangleArray() && num_sbt == 1 ); 
#else
            assert( trimesh == false ); 
            const BI& bi = gas->bis[j] ; 
            const OptixBuildInputCustomPrimitiveArray& buildInputCPA = bi.getBuildInputCPA() ;
            unsigned num_sbt = buildInputCPA.numSbtRecords ;  // <-- corresponding to bbox of the GAS
#endif
            LOG(debug) 
                 << " gas_idx " << gas_idx 
                 << " num_bi " << num_bi
                 << " j " << j 
                 << " num_sbt " << num_sbt    
                 ; 

            for( int k=0 ; k < num_sbt ; k++)
            { 
                unsigned localPrimIdx = trimesh ? j : k ;   
                if( q_gas_idx == gas_idx && q_layer_idx == localPrimIdx ) return offset_sbt ;
                offset_sbt += 1 ; 
            }
        }         
    }
    LOG(error) 
        << "did not find targetted shape " 
        << " vgas.size " << vgas.size() 
        << " q_gas_idx_ " << q_gas_idx 
        << " q_layer_idx " << q_layer_idx 
        << " offset_sbt " << offset_sbt
        ; 
    return -1 ;  
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
    unsigned tot_sbt = 0 ; 

    for(IT it=vgas.begin() ; it !=vgas.end() ; it++)
    {
        unsigned gas_idx = it->first ; 
        bool trimesh = foundry->isSolidTrimesh(gas_idx); 
        const std::string& mmlabel = foundry->getSolidMMLabel(gas_idx); 

        bool enabled = SGeoConfig::IsEnabledMergedMesh(gas_idx)  ; 
        LOG_IF(error, !enabled) << "gas_idx " << gas_idx << " enabled " << enabled ; 


#ifdef WITH_SOPTIX_ACCEL
        SOPTIX_Accel* gas = it->second ; 
#else
        const GAS* gas = &(it->second) ;   
#endif
        unsigned num_bi = gas->bis.size(); 
        tot_bi += num_bi ; 

        LOG(LEVEL) 
            << " gas_idx " << gas_idx
            << " num_bi " << num_bi
            << " trimesh " << ( trimesh ? "YES" : "NO " )
            << " mmlabel " << mmlabel  
            ;

        for(unsigned j=0 ; j < num_bi ; j++)
        { 
#ifdef WITH_SOPTIX_ACCEL
            const SOPTIX_BuildInput* bi = gas->bis[j] ; 
            unsigned num_sbt = bi->numSbtRecords() ; 
#else
            assert( trimesh == false ); 
            const BI& bi = gas->bis[j] ; 
            const OptixBuildInputCustomPrimitiveArray& buildInputCPA = bi.getBuildInputCPA() ;
            unsigned num_sbt = buildInputCPA.numSbtRecords ; 
#endif
            tot_sbt += num_sbt ; 

            LOG(LEVEL) 
                << " gas_idx " << gas_idx
                << " num_bi " << num_bi
                << " j " << j
                << " num_sbt " << num_sbt
                ;

        }         
    }
    assert( tot_bi > 0 && tot_sbt > 0 );  
    return tot_sbt ;  
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
    unsigned tot_sbt = 0 ; 
    unsigned tot_bi = 0 ; 
    std::stringstream ss ; 
    ss 
        << "SBT::descGAS"
        << " num_gas " << vgas.size() 
        << " bi.numRec ( " 
        ;

    for(IT it=vgas.begin() ; it !=vgas.end() ; it++)
    {
        unsigned gas_idx = it->first ; 
        bool trimesh = foundry->isSolidTrimesh(gas_idx); 
        const std::string& mmlabel = foundry->getSolidMMLabel(gas_idx); 

#ifdef WITH_SOPTIX_ACCEL
        SOPTIX_Accel* gas = it->second ; 
#else
        assert( trimesh == false ); 
        const GAS* gas = &(it->second) ;   
#endif

        bool enabled = SGeoConfig::IsEnabledMergedMesh(gas_idx)  ; 
        LOG_IF(error, !enabled) 
             << " gas_idx " << gas_idx 
             << " enabled " << enabled
             << " trimesh " << ( trimesh ? "YES" : "NO " )
             << " mmlabel " << mmlabel  
             ; 

        unsigned num_bi = gas->bis.size(); 
        tot_bi += num_bi ; 
        for(unsigned j=0 ; j < num_bi ; j++)
        { 

#ifdef WITH_SOPTIX_ACCEL
            const SOPTIX_BuildInput* bi = gas->bis[j] ; 
            unsigned num_sbt = bi->numSbtRecords() ; 
#else
            const BI& bi = gas->bis[j] ; 
            const OptixBuildInputCustomPrimitiveArray& buildInputCPA = bi.getBuildInputCPA() ;
            unsigned num_sbt = buildInputCPA.numSbtRecords ; 
#endif
            ss << num_sbt << " " ;  
            tot_sbt += num_sbt ; 
        }         
    }

    ss << " ) "
       << " tot_sbt " << tot_sbt 
       << " tot_bi " << tot_bi 
       ;

    std::string str = ss.str(); 
    return str ; 
}



/**
SBT::createHitgroup
---------------------

Analytic case
~~~~~~~~~~~~~~

The hitgroup array has records for all active Prims of all active Solid.
The records hold (numNode, nodeOffset) of all those active Prim.  

For analytic geom all HitGroup SBT records have the same hitgroup_pg, 
different shapes are distinguished by program data not program code 

Prim Selection
~~~~~~~~~~~~~~~~

Thoughts on how to implement Prim selection with CSGPrim::MakeSpec

Q: is there a bi for each node ?
A: NO, roughly speaking the bi hold the bbox references for all CSGPrim of the solid(=GAS) 


How to do this when each solid can be tri/ana ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Q: Still one hitgroup_pg (PIP.cc) ?



Ideas for simplification
~~~~~~~~~~~~~~~~~~~~~~~~~

1. Could collect a vector like SOPTIX_SBT.h avoiding the need for getTotalRec


Note tri/ana structural difference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+----------------+-----------------------------------------+
|  Geom type     | Structure (n-CSGPrim for each CSGSolid) |
+================+=========================================+
| Analytic       |   1 GAS : 1 BI : n-SBT records          |
+----------------+-----------------------------------------+
| Triangulated   |   1 GAS : n-BI : 1-SBT record           |
+----------------+-----------------------------------------+

**/


#ifdef WITH_SOPTIX_ACCEL
void SBT::createHitgroup()
{
    unsigned num_solid = foundry->getNumSolid(); 
    unsigned num_gas = vgas.size(); 
    unsigned tot_rec = getTotalRec();   // corresponds to the total number of enabled Prim in all enabled solids

    LOG(LEVEL)
        << " WITH_SOPTIX_ACCEL " 
        << " num_solid " << num_solid 
        << " num_gas " << num_gas 
        << " tot_rec " << tot_rec 
        ; 

    hitgroup = new HitGroup[tot_rec] ; 
    HitGroup* hg = hitgroup ; 

    for(unsigned i=0 ; i < tot_rec ; i++)   // pack headers CPU side
         OPTIX_CHECK( optixSbtRecordPackHeader( pip->hitgroup_pg, hitgroup + i ) ); 
 
    unsigned sbt_offset = 0 ; 

    for(IT it=vgas.begin() ; it !=vgas.end() ; it++)
    {
        unsigned gas_idx = it->first ; 
        SOPTIX_Accel* gas = it->second ; 


        int num_bi = gas->bis.size(); 

        bool trimesh = foundry->isSolidTrimesh(gas_idx); 
        const std::string& mmlabel = foundry->getSolidMMLabel(gas_idx); 

        const SOPTIX_MeshGroup* xmg = trimesh ? xgas.at(gas_idx) : nullptr ;
        const SCUDA_MeshGroup* cmg = xmg ? xmg->cmg : nullptr ;


        const CSGSolid* so = foundry->getSolid(gas_idx) ;
        int numPrim = so->numPrim ; 
        int primOffset = so->primOffset ; 

        LOG(LEVEL) 
            << " WITH_SOPTIX_ACCEL "  
            << " gas_idx " << gas_idx 
            << " trimesh " << ( trimesh ? "YES" : "NO " )
            << " num_bi " << num_bi  
            << " mmlabel " << mmlabel 
            << " so.numPrim " << numPrim 
            << " so.primOffset " << primOffset  
            ; 

        if(!trimesh) assert( num_bi == 1 );  
        if(trimesh) assert( num_bi == numPrim );   

        
        // ana: outer loop is mute, inner loop over CSGPrim "layers" 
        // tri: inner loop is mute, outer loop is over CSGPrim "layers"

        for(int j=0 ; j < num_bi ; j++)
        { 
            const SOPTIX_BuildInput* bi = gas->bis[j] ; 
            int num_sbt = bi->numSbtRecords() ; 

            if(!trimesh) assert( bi->is_BuildInputCustomPrimitiveArray() && num_sbt == numPrim ); 
            if(trimesh)  assert( bi->is_BuildInputTriangleArray() && num_sbt == 1 ); 

            LOG(LEVEL) 
                << " gas_idx " << gas_idx
                << " num_sbt " << num_sbt << "(ana: num_sbt is num_CSGPrim in CSGSolid, tri: num_sbt is 1)" 
                ;

            for( int k=0 ; k < num_sbt ; k++)
            { 
                unsigned localPrimIdx = trimesh ? j : k ;   
                unsigned globalPrimIdx = primOffset + localPrimIdx ;   
                const CSGPrim* prim = foundry->getPrim( globalPrimIdx );
                int boundary = foundry->getPrimBoundary_(prim); 
                assert( boundary > -1 );

                if( trimesh == false )  // analytic 
                {  
                    setPrimData( hg->data.prim, prim );  // copy numNode, nodeOffset from CSGPrim into hg->data
                }
                else 
                {
                    setMeshData( hg->data.mesh, cmg, localPrimIdx, boundary );   
                }

                unsigned check_sbt_offset = getOffset(gas_idx, localPrimIdx ); 

                bool sbt_offset_expect = check_sbt_offset == sbt_offset ; 
                assert( sbt_offset_expect  ); 
                if(!sbt_offset_expect) std::raise(SIGINT); 

                hg++ ; 
                sbt_offset++ ; 
            }
        }
    }
    UploadHitGroup(sbt, d_hitgroup, hitgroup, tot_rec ); 
}
#else

/**
SBT::createHitgroup NOT:WITH_SOPTIX_ACCEL only CustomPrimitiveArray implemented 
----------------------------------------------------------------------------------

**/
void SBT::createHitgroup()
{
    unsigned num_solid = foundry->getNumSolid(); 
    unsigned num_gas = vgas.size(); 
    unsigned tot_rec = getTotalRec();   // corresponds to the total number of enabled Prim in all enabled solids

    LOG(info) << " NOT:WITH_SOPTIX_ACCEL " ; 
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


    for(IT it=vgas.begin() ; it !=vgas.end() ; it++)
    {
        unsigned gas_idx = it->first ;  
        const GAS* gas = &(it->second) ;   
        unsigned num_bi = gas->bis.size(); 
        assert( num_bi == 1 );  // always 1 with analytic, can be more with triangulated SMeshGroup 
       
        bool trimesh = foundry->isSolidTrimesh(gas_idx); 
        LOG_IF( fatal, trimesh ) << " NOT:WITH_SOPTIX_ACCEL trimesh NOT ALLOWED gas_idx " << gas_idx ; 
        assert( !trimesh );   
 
        const CSGSolid* so = foundry->getSolid(gas_idx) ;
        int numPrim = so->numPrim ; 
        int primOffset = so->primOffset ; 

        LOG(LEVEL) << "gas_idx " << gas_idx << " so.numPrim " << numPrim << " so.primOffset " << primOffset  ; 

        for(unsigned j=0 ; j < num_bi ; j++)
        { 
            const BI& bi = gas->bis[j] ; 
            const OptixBuildInputCustomPrimitiveArray& buildInputCPA = bi.getBuildInputCPA() ;
            unsigned num_sbt = buildInputCPA.numSbtRecords ; 
            assert( num_sbt == unsigned(numPrim) ) ;

            for( unsigned k=0 ; k < num_sbt ; k++)
            { 
                unsigned localPrimIdx = k ;   

                unsigned globalPrimIdx = primOffset + localPrimIdx ;   
                const CSGPrim* prim = foundry->getPrim( globalPrimIdx ); 
                setPrimData( hg->data.prim, prim );  // copy numNode, nodeOffset from CSGPrim into hg->data
                unsigned check_sbt_offset = getOffset(gas_idx, localPrimIdx ); 

                bool sbt_offset_expect = check_sbt_offset == sbt_offset ; 
                assert( sbt_offset_expect  ); 
                if(!sbt_offset_expect) std::raise(SIGINT); 

                hg++ ; 
                sbt_offset++ ; 
            }
        }
    }
    UploadHitGroup(sbt, d_hitgroup, hitgroup, tot_rec ); 
}
#endif



void SBT::UploadHitGroup(OptixShaderBindingTable& sbt, CUdeviceptr& d_hitgroup, HitGroup* hitgroup, size_t tot_rec )
{
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




/**
SBT::setPrimData
-----------------

Called from SBT::createHitgroup to populate HitGroupData for analytic geometry. 

**/

void SBT::setPrimData( CustomPrim& cp, const CSGPrim* prim)
{
    cp.numNode = prim->numNode(); 
    cp.nodeOffset = prim->nodeOffset();  
}

void SBT::checkPrimData( CustomPrim& cp, const CSGPrim* prim)
{
    assert( cp.numNode == prim->numNode() ); 
    assert( cp.nodeOffset == prim->nodeOffset() );  

}
void SBT::dumpPrimData( const CustomPrim& cp ) const 
{
    std::cout 
        << "SBT::dumpPrimData"
        << " cp.numNode " << cp.numNode
        << " cp.nodeOffset " << cp.nodeOffset
        << std::endl 
        ; 
}


#ifdef WITH_SOPTIX_ACCEL
/**
SBT::setMeshData
-------------------

Note similarity to SOPTIX_SBT::initHitgroup 

**/

void SBT::setMeshData( TriMesh& tm, const SCUDA_MeshGroup* cmg, int j, int boundary )
{
    tm.boundary = boundary ; 
    tm.vertex = reinterpret_cast<float3*>( cmg->vtx.pointer(j) );  
    tm.normal = reinterpret_cast<float3*>( cmg->nrm.pointer(j) );  
    tm.indice = reinterpret_cast<uint3*>(  cmg->idx.pointer(j) ); 
}
#endif






