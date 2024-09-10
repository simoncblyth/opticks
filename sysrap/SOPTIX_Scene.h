#pragma once
/**
SOPTIX_Scene.h : top level, holds vectors of SCUDA_MeshGroup SOPTIX_MeshGroup and OptixInstance 
=================================================================================================

HMM: maybe SOPTIX_Geom.h so can rename SOPTIX.h to SOPTIX_Scene.h for parallel with SGLFW_Scene.h ?


HMM: in tri/ana integrated running this is probably not needed as most the geometry will
be analytic 

**/

#include "ssys.h"
#include "SOPTIX_BuildInput_IA.h"

struct SOPTIX_Scene
{ 
    static constexpr const char* _DUMP = "SOPTIX_Scene__DUMP" ; 
    bool            DUMP ; 
    SOPTIX_Context* ctx ; 
    const SScene*   scene ; 

    std::vector<SOPTIX_MeshGroup*> meshgroup ;
    std::vector<SOPTIX_Accel*> meshgas ; 
    std::vector<OptixInstance> instances ; 

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
    DUMP(ssys::getenvbool(_DUMP)), 
    ctx(_ctx),
    scene(_scene),
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
    if(DUMP) std::cout << "SOPTIX_Scene::init_GAS num_mg " << num_mg << std::endl ; 

    for(int i=0 ; i < num_mg ; i++)
    {
        const SMeshGroup*  mg = scene->meshgroup[i]; 
        SOPTIX_MeshGroup* xmg = SOPTIX_MeshGroup::Create(mg) ;  
        meshgroup.push_back(xmg);

        SOPTIX_Accel* gas = SOPTIX_Accel::Create( ctx->context, xmg->bis );     
        meshgas.push_back(gas);   
    }
}



/**
SOPTIX_Scene::initInstances
-----------------------------


SScene::inst_tran
    instance level, typically many thousands

SScene::inst_col3
    instance level, same size as inst_tran

SScene::inst_info 
    compound solid level {ridx, inst_count, inst_offset, 0}, typicallly order ~10

    * size of this vector dictates the number of GAS
    * the inst_count and 



**/


inline void SOPTIX_Scene::init_Instances()
{
    size_t num_gas  = scene->inst_info.size(); 
    size_t num_inst = scene->inst_tran.size(); 
    [[maybe_unused]] size_t num_col3 = scene->inst_col3.size(); 
    assert( num_inst == num_col3 ); 

    const std::vector<glm::tmat4x4<float>>& inst_tran = scene->inst_tran ;

    if(DUMP) std::cout 
        << "SOPTIX_Scene::init_Instances"
        << " num_gas " << num_gas 
        << " num_inst " << num_inst
        << " num_col3 " << num_col3
        << std::endl 
        ; 

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

        unsigned visibilityMask = ctx->props->visibilityMask(i) ; 

        if(DUMP) std::cout 
            << "SOPTIX_Scene::init_Instances"
            << " i " << i 
            << " ridx (_inst_info.x) " << ridx
            << " count (_inst_info.y " << count 
            << " offset (_inst_info.z)  " << offset 
            << " num_bi " << num_bi 
            << " visibilityMask " << visibilityMask
            << " sbtOffset " << sbtOffset
            << "\n"
            ;

        assert( ridx == i ); 
        for(unsigned j=0 ; j < count ; j++)
        {
            unsigned idx = offset + j ; 
            bool in_range = idx < num_inst ;  

            if(DUMP && in_range == false) std::cout 
                << "SOPTIX_Scene::init_Instances"
                << " j " << j 
                << " (offset + j)[idx] " << idx
                << " num_inst " << num_inst 
                << " in_range " << ( in_range ? "YES" : "NO " )   
                << " tot " << tot 
                << std::endl 
                ; 

            assert( in_range ); 
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
    SOPTIX_BuildInput* bi = new SOPTIX_BuildInput_IA(instances);  
    ias = SOPTIX_Accel::Create( ctx->context, bi ); 
}


inline OptixTraversableHandle SOPTIX_Scene::getHandle(int idx) const 
{  
    assert( idx < int(meshgas.size())) ; 
    return idx == -1 ? ias->handle : meshgas[idx]->handle ;
}


