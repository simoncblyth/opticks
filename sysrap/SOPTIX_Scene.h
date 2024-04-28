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

#ifdef OLD_SPLIT_APPROACH
    std::vector<SCUDA_MeshGroup*> cuda_meshgroup ;
#endif
    std::vector<SOPTIX_MeshGroup*> meshgroup ;
    std::vector<OptixInstance> instances ; 

    CUdeviceptr instances_buffer ; 
    SOPTIX_Accel* ias ; 

    std::string desc() const; 
    std::string descGAS() const; 
    std::string descIAS() const; 

    SOPTIX_Scene( SOPTIX_Context* ctx, const SScene* scene );  

    void init(); 

#ifdef OLD_SPLIT_APPROACH
    void init_MeshUpload(); 
    void init_GAS(); 
    void init_MeshUpload_free(); 
#else
    void init_MeshUpload_GAS(); 
#endif
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
#ifdef OLD_SPLIT_APPROACH
    init_MeshUpload(); 
    init_GAS();
#else
    init_MeshUpload_GAS();  // TODO: if works OK rename to init_GAS
#endif
    init_Instances();
    init_IAS();
}



#ifdef OLD_SPLIT_APPROACH
inline void SOPTIX_Scene::init_MeshUpload()
{
    int num_mg = scene->meshgroup.size() ; 
    if(dump) std::cout << "SOPTIX_Scene::init_MeshUpload num_mg " << num_mg << std::endl ; 

    for(int i=0 ; i < num_mg ; i++)
    {
        const SMeshGroup* mg = scene->meshgroup[i]; 
        SCUDA_MeshGroup* _mg = SCUDA_MeshGroup::Upload(mg) ;
        cuda_meshgroup.push_back(_mg); 
    }
}
inline void SOPTIX_Scene::init_GAS()
{
    int num_cmg = cuda_meshgroup.size() ; 
    if(dump) std::cout << "SOPTIX_Scene::init_GAS num_cmg " << num_cmg << std::endl ; 
    for(int i=0 ; i < num_cmg ; i++)
    {
        SCUDA_MeshGroup* cmg = cuda_meshgroup[i] ; 
        SOPTIX_MeshGroup* mg = new SOPTIX_MeshGroup(ctx->context, cmg) ;  
        meshgroup.push_back(mg);
    }
}
inline void SOPTIX_Scene::init_MeshUpload_free()
{
    int num_cmg = cuda_meshgroup.size() ; 
    for(int i=0 ; i < num_cmg ; i++)
    {
        SCUDA_MeshGroup* cmg = cuda_meshgroup[i] ; 
        cmg->free();   
    }
}
#endif


inline void SOPTIX_Scene::init_MeshUpload_GAS()
{
    int num_mg = scene->meshgroup.size() ; 
    if(dump) std::cout << "SOPTIX_Scene::init_MeshUpload_GAS num_mg " << num_mg << std::endl ; 

    for(int i=0 ; i < num_mg ; i++)
    {
        const SMeshGroup*  mg = scene->meshgroup[i]; 
        SOPTIX_MeshGroup* xmg = SOPTIX_MeshGroup::Create(ctx->context, mg) ;  
        meshgroup.push_back(xmg);
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
        OptixTraversableHandle handle = xmg->gas->handle ; 

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

    std::vector<OptixBuildInput> buildInputs ;
    buildInputs.push_back(buildInput); 

    ias = new SOPTIX_Accel( ctx->context, buildInputs ); 
}


inline OptixTraversableHandle SOPTIX_Scene::getHandle(int idx) const 
{  
    assert( idx < int(meshgroup.size())) ; 
    return idx == -1 ? ias->handle : meshgroup[idx]->gas->handle ;
}


