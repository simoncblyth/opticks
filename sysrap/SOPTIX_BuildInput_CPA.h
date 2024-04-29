#pragma once


struct SOPTIX_BuildInput_CPA
{
    OptixBuildInput buildInput = {} ;

    SOPTIX_BuildInput_CPA( const SCUDA_CPA* _cpa ); 
    std::string desc() const ; 
};


inline std::string SOPTIX_BuildInput_CPA::desc() const 
{
    std::stringstream ss ; 
    ss << "[SOPTIX_BuildInput_CPA::desc" << std::endl ; 
    ss <<  SOPTIX_Desc::BuildInputCustomPrimitiveArray(buildInput) ; 
    ss << "]SOPTIX_BuildInput_CPA::desc" << std::endl ; 
    std::string str = ss.str() ; 
    return str ; 
}

 
