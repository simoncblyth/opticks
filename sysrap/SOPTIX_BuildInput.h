#pragma once

struct SOPTIX_BuildInput
{
    const char* name = nullptr ; 
    OptixBuildInput buildInput = {} ;

    SOPTIX_BuildInput(const char* name); 
    std::string desc() const ; 

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
    if(strcmp(name,"BuildInputTriangleArray")==0)
    {
        ss <<  SOPTIX_Desc::BuildInputTriangleArray(buildInput) ; 
    }
    else if(strcmp(name,"BuildInputCustomPrimitiveArray")==0)
    {
        ss <<  SOPTIX_Desc::BuildInputCustomPrimitiveArray(buildInput) ; 
    }
    ss << "]SOPTIX_BuildInput::desc" << std::endl ; 
    std::string str = ss.str() ; 
    return str ; 
}




