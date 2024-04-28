#pragma once
#include <optix.h>

struct BI
{
    unsigned        mode ;   // 1: 11N (now the only mode)  0:1NN (original mode that failed to work)
    CUdeviceptr     d_aabb ;
    CUdeviceptr     d_sbt_index ;
    unsigned*       flags ;  
    unsigned*       sbt_index ;  

    OptixBuildInput buildInput ; 

          OptixBuildInputCustomPrimitiveArray& getBuildInputCPA()  ; 
    const OptixBuildInputCustomPrimitiveArray& getBuildInputCPA() const ; 

          OptixBuildInputTriangleArray&        getBuildInputTA() ; 
    const OptixBuildInputTriangleArray&        getBuildInputTA() const ; 
    
};

inline OptixBuildInputCustomPrimitiveArray& BI::getBuildInputCPA()
{
#if OPTIX_VERSION == 70000
    return buildInput.aabbArray ;  
#elif OPTIX_VERSION > 70000
    return buildInput.customPrimitiveArray ;  
#endif
}
inline const OptixBuildInputCustomPrimitiveArray& BI::getBuildInputCPA() const
{
#if OPTIX_VERSION == 70000
    return buildInput.aabbArray ;  
#elif OPTIX_VERSION > 70000
    return buildInput.customPrimitiveArray ;  
#endif
}


inline OptixBuildInputTriangleArray& BI::getBuildInputTA()
{
    return buildInput.triangleArray ; 
}
inline const OptixBuildInputTriangleArray& BI::getBuildInputTA() const
{
    return buildInput.triangleArray ; 
}



