#pragma once
/**
SOPTIX_Scene.h : top level, holds vectors of SCUDA_MeshGroup SOPTIX_MeshGroup and OptixInstance 
=================================================================================================

HMM: maybe SOPTIX_Geom.h so can rename SOPTIX.h to SOPTIX_Scene.h for parallel with SGLFW_Scene.h ?

**/

#include <bitset>

struct SOPTIX_Scene
{ 
    bool            dump ; 
    SOPTIX_Context* ctx ; 
    const SScene*   scene ; 

    std::vector<SOPTIX_MeshGroup*> meshgroup ;
    std::vector<SOPTIX_Accel*> meshgas ; 
    std::vector<OptixInstance> instances ; 

    CUdeviceptr instances_buffer ; 
    std::vector<OptixBuildInput> buildInputs ; // for IAS
    SOPTIX_Accel* ias ; 

    std::string desc() const; 
    std::string descGAS() const; 
    std::string descIAS() const; 

    SOPTIX_Scene( SOPTIX_Context* ctx, const SScene* scene );  

    void init(); 

    void init_GAS(); 
    void init_Instances(); 
    void init_IAS();

    OptixTraversableHandle getHandle(int idx) const ;  
};


inline std::string SOPTIX_Scene::desc() const 
{
    int num_mm = scene->meshmerge.size() ; 
    int num_mg = scene->meshgroup.size() ; 
    std::stringstream ss ;
    ss << "[ SOPTIX_Scene::desc"
        << " num_mm " << num_mm 
        << " num_mg " << num_mg 
        << std::endl 
        ; 
    ss << descGAS() ; 
    ss << descIAS() ; 
    ss << "] SOPTIX_Scene::desc " << std::endl ; 
    std::string str = ss.str(); 
    return str ;
}
inline std::string SOPTIX_Scene::descGAS() const 
{
    int num_gas = int(meshgroup.size()); 
    std::stringstream ss ;
    ss << "[ SOPTIX_Scene::descGAS num_gas " << num_gas << std::endl ;
    for(int i=0 ; i < num_gas ; i++ ) ss << meshgroup[i]->desc() ; 
    ss << "] SOPTIX_Scene::descGAS num_gas " << num_gas << std::endl ;
    std::string str = ss.str(); 
    return str ;
}

inline std::string SOPTIX_Scene::descIAS() const 
{
    std::stringstream ss ;
    ss << "[ SOPTIX_Scene::descIAS\n" ;
    ss << ias->desc() ;  
    ss << "] SOPTIX_Scene::descIAS\n" ;
    std::string str = ss.str(); 
    return str ;
}

inline SOPTIX_Scene::SOPTIX_Scene( SOPTIX_Context* _ctx, const SScene* _scene )
    :
    dump(false),
    ctx(_ctx),
    scene(_scene),
    instances_buffer(0),
    ias(nullptr)
{
    init(); 
}

inline void SOPTIX_Scene::init()
{
    init_GAS();
    init_Instances();
    init_IAS();
}

inline void SOPTIX_Scene::init_GAS()
{
    int num_mg = scene->meshgroup.size() ; 
    if(dump) std::cout << "SOPTIX_Scene::init_GAS num_mg " << num_mg << std::endl ; 

    for(int i=0 ; i < num_mg ; i++)
    {
        const SMeshGroup*  mg = scene->meshgroup[i]; 
        SOPTIX_MeshGroup* xmg = SOPTIX_MeshGroup::Create(mg) ;  
        meshgroup.push_back(xmg);

        SOPTIX_Accel* gas = new SOPTIX_Accel( ctx->context, xmg->buildInputs );     
        meshgas.push_back(gas);   
    }
}



/**
SOPTIX_Scene::initInstances
-----------------------------

**/


inline void SOPTIX_Scene::init_Instances()
{
    unsigned visibilityMask_FULL = ctx->props->visibilityMask(); 
    unsigned visibilityMask_BITS = std::bitset<32>(visibilityMask_FULL).count(); 
    assert( visibilityMask_FULL == 0xffu ); 
    assert( visibilityMask_BITS == 8 ); 

    const std::vector<glm::tmat4x4<float>>& inst_tran = scene->inst_tran ;
    size_t num_gas  = scene->inst_info.size(); 
    size_t num_inst = scene->inst_tran.size(); 

    if(dump) std::cout << "SOPTIX_Scene::init_Instances num_gas " << num_gas << std::endl ; 

    unsigned tot = 0 ; 
    unsigned flags = OPTIX_INSTANCE_FLAG_NONE ; 
    flags |= OPTIX_INSTANCE_FLAG_DISABLE_TRIANGLE_FACE_CULLING ;  
    flags |= OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT ;  

    size_t sbtOffset = 0 ; 
    for(unsigned i=0 ; i < num_gas ; i++)
    {
        const int4& _inst_info = scene->inst_info[i] ;
        unsigned ridx = _inst_info.x ; 
        unsigned count = _inst_info.y ; 
        unsigned offset = _inst_info.z ; 

        SOPTIX_MeshGroup* xmg = meshgroup[i] ; 
        size_t num_bi = xmg->num_buildInputs(); 
        SOPTIX_Accel* gas = meshgas[i] ;
        OptixTraversableHandle handle = gas->handle ; 

        unsigned marker_bit = std::min( i, visibilityMask_BITS - 1 );  
        unsigned visibilityMask = 0x1 << marker_bit ;  

        assert( ridx == i ); 
        for(unsigned j=0 ; j < count ; j++)
        {
            unsigned idx = offset + j ; 
            assert( idx < num_inst ); 
            assert( idx == tot ); 
            tot += 1 ;

            const glm::tmat4x4<float>& tran = inst_tran[idx] ; 
            OptixInstance instance = {} ;
  
            stra<float>::Copy_Columns_3x4( instance.transform, tran );  
            instance.instanceId = idx ; // app supplied
            instance.sbtOffset = sbtOffset ; 
            instance.visibilityMask = visibilityMask ;
            instance.flags = flags ;
            instance.traversableHandle = handle ; 

            instances.push_back(instance);
        }
        sbtOffset += num_bi ; 
    }
}

inline void SOPTIX_Scene::init_IAS()
{
    unsigned num_bytes = sizeof( OptixInstance )*instances.size() ; 

    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &instances_buffer ), num_bytes ) );
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( instances_buffer ),
                instances.data(),
                num_bytes,
                cudaMemcpyHostToDevice
                ) );

 
    OptixBuildInput buildInput = {}; 
    buildInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    OptixBuildInputInstanceArray& instanceArray = buildInput.instanceArray ; 
    instanceArray.instances = instances_buffer ;  
    instanceArray.numInstances = instances.size() ; 

    buildInputs.push_back(buildInput); 

    ias = new SOPTIX_Accel( ctx->context, buildInputs ); 
}


inline OptixTraversableHandle SOPTIX_Scene::getHandle(int idx) const 
{  
    assert( idx < int(meshgas.size())) ; 
    return idx == -1 ? ias->handle : meshgas[idx]->handle ;
}


