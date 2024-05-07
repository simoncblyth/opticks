#pragma once
/**
SOPTIX_SBT.h : create sbt from pipeline and scene by uploding the prog and hitgroup records
============================================================================================

OptixShaderBindingTable binds together geometry, "shader" programs, records with program data 

Used for example from::

   sysrap/tests/SOPTIX_Scene_test.cc
   sysrap/tests/SGLFW_SOPTIX_Scene_test.cc

Good general explanatiom of SBT Shader Binding Table

* https://www.willusher.io/graphics/2019/11/20/the-sbt-three-ways

**/

#include "SOPTIX_Binding.h"

struct SOPTIX_SBT
{ 
    SOPTIX_Pipeline& pip ;     
    SOPTIX_Scene& scn ;     

    std::vector<SOPTIX_HitgroupRecord> hitgroup_records;
    OptixShaderBindingTable sbt = {};  

    SOPTIX_SBT( SOPTIX_Pipeline& pip, SOPTIX_Scene& scn ); 

    void init(); 
    void initRaygen(); 
    void initMiss(); 
    void initHitgroup(); 

    std::string desc() const ; 
    std::string descPartBI() const ; 

    static std::string Desc(const OptixShaderBindingTable& sbt) ; 

};


inline SOPTIX_SBT::SOPTIX_SBT( SOPTIX_Pipeline& _pip, SOPTIX_Scene& _scn )
    :
    pip(_pip),
    scn(_scn)
{
    init(); 
}
 
inline void SOPTIX_SBT::init()
{
    initRaygen(); 
    initMiss(); 
    initHitgroup(); 
}

inline void SOPTIX_SBT::initRaygen()
{
    CUdeviceptr  raygen_record ; 
   
    const size_t raygen_record_size = sizeof( SOPTIX_RaygenRecord );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &raygen_record ), raygen_record_size ) );
    
    SOPTIX_RaygenRecord rg_sbt;
    OPTIX_CHECK( optixSbtRecordPackHeader( pip.raygen_pg, &rg_sbt ) );
    //SOPTIX_RaygenData& data = rg_sbt.data

    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>(raygen_record),
                &rg_sbt,
                raygen_record_size,
                cudaMemcpyHostToDevice
                ) );

    sbt.raygenRecord = raygen_record ; 
}

inline void SOPTIX_SBT::initMiss()
{
    CUdeviceptr miss_record ;

    const size_t miss_record_size = sizeof( SOPTIX_MissRecord );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &miss_record ), miss_record_size ) );
    
    SOPTIX_MissRecord ms_sbt;
    OPTIX_CHECK( optixSbtRecordPackHeader( pip.miss_pg, &ms_sbt ) );
    SOPTIX_MissData& data = ms_sbt.data ; 
    //data.bg_color = make_float3( 0.3f, 0.1f, 0.f ) ; 
    data.bg_color = make_float3( 1.0f, 0.0f, 0.f ) ; 

    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( miss_record ),
                &ms_sbt,
                miss_record_size,
                cudaMemcpyHostToDevice
                ) );

    sbt.missRecordBase = miss_record;
    sbt.missRecordStrideInBytes = static_cast<uint32_t>( miss_record_size );
    sbt.missRecordCount = 1 ; 
}


/**
SOPTIX_SBT::initHitgroup
-------------------------

HMM: with analytic geometry have "boundary" that 
comes from the CSGNode. To do that with triangles 
need to plant the boundary indices into HitgroupData.  
That means need hitgroup records for each sub-SMesh 
(thats equivalent to each CSGPrim)

TODO: check this labelling 

Need nested loop like CSGOptiX/SBT.cc SBT::createHitgroup::
 
     GAS 
        BuildInput       (actually 1:1 with GAS for analytic) 
           sub-SMesh 

So need access to scene data to form the SBT 

NB this uses CPU/GPU types defined in SOPTIX_Binding.h

**/

inline void SOPTIX_SBT::initHitgroup()
{
    std::cout << "SOPTIX_SBT::initHitgroup " << descPartBI() << std::endl ; 

    size_t num_mg = scn.meshgroup.size();  // SOPTIX_Scene  
    for(size_t i=0 ; i < num_mg ; i++)
    { 
        SOPTIX_MeshGroup* xmg = scn.meshgroup[i] ; 
        size_t num_bi = xmg->num_buildInputs()  ; 
        const SCUDA_MeshGroup* cmg = xmg->cmg ; 
        size_t num_part = cmg->num_part()  ; 
        assert( num_part == num_bi ); 
        size_t edge = 20 ; 

        for(size_t j=0 ; j < num_bi ; j++)
        {   
            const SOPTIX_BuildInput* bi = xmg->bis[j] ; 
            assert( bi->is_BuildInputTriangleArray() ); 
            unsigned numSbtRecords = bi->numSbtRecords();  
            if( j < edge || j > (num_bi - edge) ) std::cout 
                << "SOPTIX_SBT::initHitgroup"
                << " i " << i 
                << " num_mg " << num_mg 
                << " j " << j
                << " num_bi " << num_bi 
                << " numSbtRecords " << numSbtRecords
                << "\n" 
                ; 
            assert( numSbtRecords == 1 ); 

            SOPTIX_HitgroupRecord hg_sbt;
            OPTIX_CHECK( optixSbtRecordPackHeader( pip.hitgroup_pg, &hg_sbt ) );
            SOPTIX_HitgroupData& data = hg_sbt.data ; 
            SOPTIX_TriMesh& trimesh = data.mesh ; 

            trimesh.vertex = reinterpret_cast<float3*>( cmg->vtx.pointer(j) ); 
            trimesh.normal = reinterpret_cast<float3*>( cmg->nrm.pointer(j) ); 
            trimesh.indice = reinterpret_cast<uint3*>(  cmg->idx.pointer(j) ); 

            hitgroup_records.push_back(hg_sbt);
        } 
    }

    CUdeviceptr hitgroup_record_base ;
    const size_t hitgroup_record_size = sizeof( SOPTIX_HitgroupRecord );
    const size_t hitgroup_record_bytes = hitgroup_record_size*hitgroup_records.size() ; 

    CUDA_CHECK( cudaMalloc( 
                reinterpret_cast<void**>( &hitgroup_record_base ), 
                hitgroup_record_bytes ) );

    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( hitgroup_record_base ),
                hitgroup_records.data(),
                hitgroup_record_bytes,
                cudaMemcpyHostToDevice
                ) );

    sbt.hitgroupRecordBase = hitgroup_record_base ; 
    sbt.hitgroupRecordStrideInBytes = static_cast<uint32_t>( hitgroup_record_size );
    sbt.hitgroupRecordCount         = static_cast<uint32_t>( hitgroup_records.size() );
}


inline std::string SOPTIX_SBT::desc() const 
{
    std::stringstream ss ;
    ss << "[SOPTIX_SBT::desc\n" ; 
    ss << " hitgroup_records.size " << hitgroup_records.size() << "\n" ; 
    ss << Desc(sbt) ; 
    ss << "]SOPTIX_SBT::desc\n" ; 
    std::string str = ss.str(); 
    return str ; 
} 

inline std::string SOPTIX_SBT::descPartBI() const 
{
    std::stringstream ss ;
    ss << "[SOPTIX_SBT::descPartBI\n" ; 
    size_t num_mg = scn.meshgroup.size();  
    for(size_t i=0 ; i < num_mg ; i++)
    { 
        SOPTIX_MeshGroup* xmg = scn.meshgroup[i] ; 
        size_t num_bi = xmg->num_buildInputs()  ; 
        const SCUDA_MeshGroup* cmg = xmg->cmg ; 
        size_t num_part = cmg->num_part()  ; 
        assert( num_part == num_bi ); 

        ss 
            << std::setw(4) << i 
            << " SCUDA_MeshGroup::num_part " << std::setw(6) << num_part 
            << " SOPTIX_MeshGroup::num_bi  " << std::setw(6) << num_bi
            << "\n"
            ; 
    }
    ss << "]SOPTIX_SBT::descPartBI\n" ; 
    std::string str = ss.str(); 
    return str ; 
}


inline std::string SOPTIX_SBT::Desc(const OptixShaderBindingTable& sbt)  // static 
{
    std::stringstream ss ;
    ss 
        << " sbt.raygenRecord                " << sbt.raygenRecord << "\n" 
        << " sbt.missRecordBase              " << sbt.missRecordBase  << "\n"
        << " sbt.missRecordStrideInBytes     " << sbt.missRecordStrideInBytes << "\n"
        << " sbt.missRecordCount             " << sbt.missRecordCount << "\n"
        << " sbt.hitgroupRecordBase          " << sbt.hitgroupRecordBase << "\n" 
        << " sbt.hitgroupRecordStrideInBytes " << sbt.hitgroupRecordStrideInBytes << "\n"
        << " sbt.hitgroupRecordCount         " << sbt.hitgroupRecordCount << "\n"  
       ; 
    std::string str = ss.str(); 
    return str ; 
}


