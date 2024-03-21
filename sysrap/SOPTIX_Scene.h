#pragma once


struct SOPTIX_Scene
{ 
    SOPTIX* ox ; 
    SScene* scene ; 

    std::vector<SCUDA_Mesh*> cuda_mesh ;
    std::vector<SOPTIX_Mesh*> optix_mesh ;
    std::vector<OptixInstance> instances ; 

    CUdeviceptr instances_buffer ; 
    SOPTIX_Accel* ias ; 

    std::string desc() const; 
    std::string descGAS() const; 
    std::string descIAS() const; 

    SOPTIX_Scene( SOPTIX* ox, SScene* scene );  

    void init(); 
    void initMeshUpload(); 
    void initGAS(); 
    void initInstances(); 
    void initIAS(); 
};


inline std::string SOPTIX_Scene::desc() const 
{
    int num_mesh = scene->mesh_grup.size() ; 
    std::stringstream ss ;
    ss << "[ SOPTIX_Scene::desc num_mesh " << num_mesh << std::endl ; 
    ss << descGAS() ; 
    ss << descIAS() ; 
    ss << "] SOPTIX_Scene::desc num_mesh " << num_mesh << std::endl ; 
    std::string str = ss.str(); 
    return str ;
}
inline std::string SOPTIX_Scene::descGAS() const 
{
    int num_gas = int(optix_mesh.size()); 
    std::stringstream ss ;
    ss << "[ SOPTIX_Scene::descGAS num_gas " << num_gas << std::endl ;
    for(int i=0 ; i < num_gas ; i++ ) ss << optix_mesh[i]->desc() ; 
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

inline SOPTIX_Scene::SOPTIX_Scene( SOPTIX* _ox, SScene* _scene )
    :
    ox(_ox),
    scene(_scene),
    instances_buffer(0),
    ias(nullptr)
{
    init(); 
}

inline void SOPTIX_Scene::init()
{
    initMeshUpload(); 
    initGAS();
 
    initInstances();
    initIAS(); 
}

inline void SOPTIX_Scene::initMeshUpload()
{
    int num_mesh = scene->mesh_grup.size() ; 
    for(int i=0 ; i < num_mesh ; i++)
    {
        const SMesh* m = scene->mesh_grup[i]; 
        SCUDA_Mesh* _mesh = new SCUDA_Mesh(m) ; 
        cuda_mesh.push_back(_mesh); 
    }
}

inline void SOPTIX_Scene::initGAS()
{
    for(int i=0 ; i < int(cuda_mesh.size()) ; i++)
    {
        SCUDA_Mesh* _mesh = cuda_mesh[i] ;  
        SOPTIX_Mesh* mesh = new SOPTIX_Mesh(ox, _mesh) ;  
        optix_mesh.push_back(mesh);
    }
}

/**
SOPTIX_Scene::initInstances
-----------------------------

sbtOffset
~~~~~~~~~~~~

https://forums.developer.nvidia.com/t/sbt-problem-when-using-multiple-gas-objects/108824/2

**/


inline void SOPTIX_Scene::initInstances()
{
    const std::vector<glm::tmat4x4<float>>& inst_tran = scene->inst_tran ;
    int num_gas  = scene->inst_info.size(); 
    int num_inst = scene->inst_tran.size(); 

    int tot = 0 ; 
    unsigned sbtOffset = 0 ; 

    unsigned visibilityMask = ox->props->visibilityMask(); 
    assert( visibilityMask == 0xffu ); 
    
    unsigned flags = OPTIX_INSTANCE_FLAG_NONE ; 
    flags |= OPTIX_INSTANCE_FLAG_DISABLE_TRIANGLE_FACE_CULLING ;  
    flags |= OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT ;  


    for(int i=0 ; i < num_gas ; i++)
    {
        const int4& _inst_info = scene->inst_info[i] ;
        int ridx = _inst_info.x ; 
        int count = _inst_info.y ; 
        int offset = _inst_info.z ; 

        SOPTIX_Mesh* mesh = optix_mesh[i] ; 

        assert( ridx == i ); 
        for(int j=0 ; j < count ; j++)
        {
            int idx = offset + j ; 
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
            instance.traversableHandle = mesh->gas->handle ; 

            instances.push_back(instance);
            sbtOffset += 1 ;  // ??? one sbt record per GAS build input per RAY_TYPE
        }
    }
}

inline void SOPTIX_Scene::initIAS()
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
    OptixBuildInputInstanceArray instanceArray = buildInput.instanceArray ; 
    instanceArray.instances = instances_buffer ;  
    instanceArray.numInstances = instances.size() ; 

    std::vector<OptixBuildInput> buildInputs ;
    buildInputs.push_back(buildInput); 

    ias = new SOPTIX_Accel( ox->context, buildInputs ); 
}



