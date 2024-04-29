#pragma once

struct SOPTIX_Desc
{
    static std::string AccelBufferSizes( const OptixAccelBufferSizes& accelBufferSizes ); 
    static std::string BuildInputTriangleArray( const OptixBuildInput& buildInput ); 
    static std::string BuildInputCustomPrimitiveArray( const OptixBuildInput& buildInput ); 
    static std::string BuildInputInstanceArray( const OptixBuildInput& buildInput ); 
};


inline std::string SOPTIX_Desc::AccelBufferSizes( const OptixAccelBufferSizes& accelBufferSizes ) // static
{
    std::stringstream ss ;  
    ss
        << "[SOPTIX_Desc::AccelBufferSizes"
        << std::endl
        << "accelBufferSizes.outputSizeInBytes     : " << accelBufferSizes.outputSizeInBytes
        << std::endl 
        << "accelBufferSizes.tempSizeInBytes       : " << accelBufferSizes.tempSizeInBytes
        << std::endl 
        << "accelBufferSizes.tempUpdateSizeInBytes : " << accelBufferSizes.tempUpdateSizeInBytes
        << std::endl 
        << "]SOPTIX_Desc::AccelBufferSizes"
        << std::endl 
        ; 
    std::string str = ss.str(); 
    return str ;
}

inline std::string SOPTIX_Desc::BuildInputTriangleArray( const OptixBuildInput& buildInput ) 
{
    std::stringstream ss ; 
    ss << "[SOPTIX_Desc::BuildInputTriangleArray" << std::endl ; 
    ss << " buildInput.triangleArray.numVertices      : " << buildInput.triangleArray.numVertices << std::endl ; 
    ss << " buildInput.triangleArray.numIndexTriplets : " << buildInput.triangleArray.numIndexTriplets << std::endl ; 
    ss << " buildInput.triangleArray.flags[0]         : " << buildInput.triangleArray.flags[0] << std::endl ; 
    ss << "]SOPTIX_Desc::BuildInputTriangleArray" << std::endl ; 
    std::string str = ss.str() ; 
    return str ; 
}

inline std::string SOPTIX_Desc::BuildInputCustomPrimitiveArray( const OptixBuildInput& buildInput ) 
{
    std::stringstream ss ; 
    ss << "[SOPTIX_Desc::BuildInputCustomPrimitiveArray" << std::endl ; 
    ss << " buildInput.customPrimitiveArray.primitiveIndexOffset         : " << buildInput.customPrimitiveArray.primitiveIndexOffset << std::endl ; 
    ss << " buildInput.customPrimitiveArray.aabbBuffers[0]   0x          : " << std::hex << buildInput.customPrimitiveArray.aabbBuffers[0] << std::dec << std::endl ; 
    ss << " buildInput.customPrimitiveArray.sbtIndexOffsetBuffer         : " << buildInput.customPrimitiveArray.sbtIndexOffsetBuffer << std::endl ; 
    ss << " buildInput.customPrimitiveArray.strideInBytes                : " << buildInput.customPrimitiveArray.strideInBytes << std::endl ;     
    ss << " buildInput.customPrimitiveArray.sbtIndexOffsetStrideInBytes  : " << buildInput.customPrimitiveArray.sbtIndexOffsetStrideInBytes << std::endl ; 
    ss << "]SOPTIX_Desc::BuildInputCustomPrimitiveArray" << std::endl ; 
    std::string str = ss.str() ; 
    return str ; 
}

inline std::string SOPTIX_Desc::BuildInputInstanceArray( const OptixBuildInput& buildInput ) 
{
    std::stringstream ss ; 
    ss << "[SOPTIX_Desc::BuildInputInstanceArray" << std::endl ; 
    ss << "]SOPTIX_Desc::BuildInputInstanceArray" << std::endl ; 
    std::string str = ss.str() ; 
    return str ; 
}


