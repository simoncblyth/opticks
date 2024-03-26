#pragma once
/**
SOPTIX_Scene.h
===============



**/
struct SOPTIX_Scene
{ 
    SOPTIX* ox ; 
    SScene* scene ; 

    std::vector<SCUDA_MeshGroup*> cuda_meshgroup ;
    std::vector<SOPTIX_MeshGroup*> meshgroup ;
    std::vector<OptixInstance> instances ; 

    CUdeviceptr instances_buffer ; 
    SOPTIX_Accel* ias ; 

    std::string desc() const; 
    std::string descGAS() const; 
    std::string descIAS() const; 

    SOPTIX_Scene( SOPTIX* ox, SScene* scene );  

    void init(); 
    void init_MeshUpload(); 
    void init_GAS(); 
    void init_Instances(); 
    void init_IAS();
    void init_MeshUpload_free(); 

    //void init_PTXModule();
    //void init_ProgramGroups();
    //void init_Pipeline();
    //void init_SBT();
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
    init_MeshUpload(); 
    init_GAS();
    init_Instances();
    init_IAS();

    //init_MeshUpload_free(); 
    // earlier ? vtx,idx split from ins 
    // needs to be after SBT is setup ? 

    //init_PTXModule();
    //init_ProgramGroups();
    //init_Pipeline();
    //init_SBT();
}



/**
SOPTIX_Scene::init_MeshUpload
-------------------------------

**/

inline void SOPTIX_Scene::init_MeshUpload()
{
    int num_mg = scene->meshgroup.size() ; 
    std::cout << "SOPTIX_Scene::init_MeshUpload num_mg " << num_mg << std::endl ; 

    for(int i=0 ; i < num_mg ; i++)
    {
        const SMeshGroup* mg = scene->meshgroup[i]; 
        SCUDA_MeshGroup* _mg = new SCUDA_MeshGroup(mg) ; 
        cuda_meshgroup.push_back(_mg); 
    }
}

inline void SOPTIX_Scene::init_MeshUpload_free()
{
    int num_mg = cuda_meshgroup.size() ; 
    for(int i=0 ; i < num_mg ; i++)
    {
        SCUDA_MeshGroup* _mg = cuda_meshgroup[i] ; 
        _mg->free();   
    }
}


inline void SOPTIX_Scene::init_GAS()
{
    int num_cmg = cuda_meshgroup.size() ; 
    std::cout << "SOPTIX_Scene::init_GAS num_cmg " << num_cmg << std::endl ; 
    for(int i=0 ; i < num_cmg ; i++)
    {
        SCUDA_MeshGroup* _mg = cuda_meshgroup[i] ; 
        SOPTIX_MeshGroup* mg = new SOPTIX_MeshGroup(ox->context, _mg) ;  
        meshgroup.push_back(mg);
    }
}

/**
SOPTIX_Scene::initInstances
-----------------------------

**/


inline void SOPTIX_Scene::init_Instances()
{
    const std::vector<glm::tmat4x4<float>>& inst_tran = scene->inst_tran ;
    int num_gas  = scene->inst_info.size(); 
    int num_inst = scene->inst_tran.size(); 

    std::cout << "SOPTIX_Scene::init_Instances num_gas " << num_gas << std::endl ; 

    int tot = 0 ; 

    unsigned visibilityMask = ox->props->visibilityMask(); 
    assert( visibilityMask == 0xffu ); 
    
    unsigned flags = OPTIX_INSTANCE_FLAG_NONE ; 
    flags |= OPTIX_INSTANCE_FLAG_DISABLE_TRIANGLE_FACE_CULLING ;  
    flags |= OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT ;  

    size_t sbtOffset = 0 ; 
    for(int i=0 ; i < num_gas ; i++)
    {
        const int4& _inst_info = scene->inst_info[i] ;
        int ridx = _inst_info.x ; 
        int count = _inst_info.y ; 
        int offset = _inst_info.z ; 

        SOPTIX_MeshGroup* mg = meshgroup[i] ; 
        OptixTraversableHandle handle = mg->gas->handle ; 

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
            instance.traversableHandle = handle ; 

            instances.push_back(instance);
        }

        size_t num_bi = mg->num_buildInputs(); 
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

    ias = new SOPTIX_Accel( ox->context, buildInputs ); 
}

//inline void SOPTIX_Scene::init_PTXModule(){};
//inline void SOPTIX_Scene::init_ProgramGroups(){};
//inline void SOPTIX_Scene::init_Pipeline();
//inline void SOPTIX_Scene::init_SBT(){};

