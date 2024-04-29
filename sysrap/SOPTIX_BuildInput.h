#pragma once

struct SOPTIX_BuildInput
{
    const char* name = nullptr ; 
    OptixBuildInput buildInput = {} ;

    SOPTIX_BuildInput(const char* name); 
    std::string desc() const ; 

    bool is_BuildInputTriangleArray() const ; 
    bool is_BuildInputCustomPrimitiveArray() const ; 
    bool is_BuildInputInstanceArray() const ; 

    unsigned numSbtRecords() const ; 

};


inline SOPTIX_BuildInput::SOPTIX_BuildInput(const char* _name)
    :
    name(_name)
{
}

inline std::string SOPTIX_BuildInput::desc() const 
{
    std::stringstream ss ; 
    ss << "[SOPTIX_BuildInput::desc" << std::endl ; 
    ss << " name " << ( name ? name : "-" ) << std::endl ; 
    if(is_BuildInputTriangleArray())
    {
        ss <<  SOPTIX_Desc::BuildInputTriangleArray(buildInput) ; 
    }
    else if(is_BuildInputCustomPrimitiveArray())
    {
        ss <<  SOPTIX_Desc::BuildInputCustomPrimitiveArray(buildInput) ; 
    }
    else if(is_BuildInputInstanceArray())
    {
        ss << SOPTIX_Desc::BuildInputInstanceArray(buildInput) ; 
    }
    ss << "]SOPTIX_BuildInput::desc" << std::endl ; 
    std::string str = ss.str() ; 
    return str ; 
}

inline bool SOPTIX_BuildInput::is_BuildInputTriangleArray() const 
{
    return buildInput.type == OPTIX_BUILD_INPUT_TYPE_TRIANGLES ;
} 
inline bool SOPTIX_BuildInput::is_BuildInputCustomPrimitiveArray() const 
{
    return buildInput.type == OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES ;
} 
inline bool SOPTIX_BuildInput::is_BuildInputInstanceArray() const 
{
    return buildInput.type == OPTIX_BUILD_INPUT_TYPE_INSTANCES ;
}

     
inline unsigned SOPTIX_BuildInput::numSbtRecords() const 
{
    unsigned num = 0 ;
    switch(buildInput.type)
    {
        case OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES: num = buildInput.customPrimitiveArray.numSbtRecords ;
        case OPTIX_BUILD_INPUT_TYPE_TRIANGLES:         num = buildInput.triangleArray.numSbtRecords ;
    }
    return num ; 
}


