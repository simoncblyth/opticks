#pragma once



struct SOPTIX
{  
    static std::string DescAccelBufferSizes( const OptixAccelBufferSizes& accelBufferSizes ); 
    static std::string DescBuildInputTriangleArray( const OptixBuildInput& buildInput ); 
};


inline std::string SOPTIX::DescAccelBufferSizes( const OptixAccelBufferSizes& accelBufferSizes ) // static
{
    std::stringstream ss ;  
    ss
        << "[SOPTIX::DescAccelBufferSizes"
        << std::endl
        << "accelBufferSizes.outputSizeInBytes     : " << accelBufferSizes.outputSizeInBytes
        << std::endl 
        << "accelBufferSizes.tempSizeInBytes       : " << accelBufferSizes.tempSizeInBytes
        << std::endl 
        << "accelBufferSizes.tempUpdateSizeInBytes : " << accelBufferSizes.tempUpdateSizeInBytes
        << std::endl 
        << "]SOPTIX::DescAccelBufferSizes"
        << std::endl 
        ; 
    std::string str = ss.str(); 
    return str ;
}

inline std::string SOPTIX::DescBuildInputTriangleArray( const OptixBuildInput& buildInput ) 
{
    std::stringstream ss ; 
    ss << "[SOPTIX::DescBuildInputTriangleArray" << std::endl ; 
    ss << " buildInput.triangleArray.numVertices      : " << buildInput.triangleArray.numVertices << std::endl ; 
    ss << " buildInput.triangleArray.numIndexTriplets : " << buildInput.triangleArray.numIndexTriplets << std::endl ; 
    ss << " buildInput.triangleArray.flags[0]         : " << buildInput.triangleArray.flags[0] << std::endl ; 
    ss << "]SOPTIX::DescBuildInputTriangleArray" << std::endl ; 
    std::string str = ss.str() ; 
    return str ; 
}



