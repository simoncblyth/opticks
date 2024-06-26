#pragma once
/**
SOPTIX_Properties.h : optixDeviceContextGetProperty results
=============================================================
**/

struct SOPTIX_Properties
{
    unsigned rtcoreVersion ; 
    unsigned limitMaxTraceDepth ; 
    unsigned limitMaxTraversableGraphDepth ;
    unsigned limitMaxPrimitivesPerGas ;
    unsigned limitMaxInstancesPerIas ;
    unsigned limitMaxInstanceId ;
    unsigned limitNumBitsInstanceVisibilityMask ;
    unsigned limitMaxSbtRecordsPerGas ;
    unsigned limitMaxSbtOffset ;   

    unsigned visibilityMask() const ;     
    SOPTIX_Properties(OptixDeviceContext context); 
    std::string desc() const ; 
}; 



/**
SOPTIX_Properties::visibilityMask
-------------------------------------

::

    +---------------------+--------+----------+
    | ( 0x1 << 8 ) - 1    |   255  |  0xff    |
    +---------------------+--------+----------+

**/

inline unsigned SOPTIX_Properties::visibilityMask() const
{
    return ( 0x1 << limitNumBitsInstanceVisibilityMask ) - 1 ;  
}

inline SOPTIX_Properties::SOPTIX_Properties(OptixDeviceContext context)
{

    OPTIX_CHECK( optixDeviceContextGetProperty(context, OPTIX_DEVICE_PROPERTY_LIMIT_MAX_TRACE_DEPTH                   , &limitMaxTraceDepth                 , sizeof(unsigned int)) );
    OPTIX_CHECK( optixDeviceContextGetProperty(context, OPTIX_DEVICE_PROPERTY_LIMIT_MAX_TRAVERSABLE_GRAPH_DEPTH       , &limitMaxTraversableGraphDepth      , sizeof(unsigned int)) );
    OPTIX_CHECK( optixDeviceContextGetProperty(context, OPTIX_DEVICE_PROPERTY_LIMIT_MAX_PRIMITIVES_PER_GAS            , &limitMaxPrimitivesPerGas           , sizeof(unsigned int)) );
    OPTIX_CHECK( optixDeviceContextGetProperty(context, OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCES_PER_IAS             , &limitMaxInstancesPerIas            , sizeof(unsigned int)) );
    OPTIX_CHECK( optixDeviceContextGetProperty(context, OPTIX_DEVICE_PROPERTY_RTCORE_VERSION                          , &rtcoreVersion                      , sizeof(unsigned int)) );
    OPTIX_CHECK( optixDeviceContextGetProperty(context, OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCE_ID                   , &limitMaxInstanceId                 , sizeof(unsigned int)) );
    OPTIX_CHECK( optixDeviceContextGetProperty(context, OPTIX_DEVICE_PROPERTY_LIMIT_NUM_BITS_INSTANCE_VISIBILITY_MASK , &limitNumBitsInstanceVisibilityMask , sizeof(unsigned int)) );
    OPTIX_CHECK( optixDeviceContextGetProperty(context, OPTIX_DEVICE_PROPERTY_LIMIT_MAX_SBT_RECORDS_PER_GAS           , &limitMaxSbtRecordsPerGas           , sizeof(unsigned int)) );
    OPTIX_CHECK( optixDeviceContextGetProperty(context, OPTIX_DEVICE_PROPERTY_LIMIT_MAX_SBT_OFFSET                    , &limitMaxSbtOffset                  , sizeof(unsigned int)) );
}

inline std::string SOPTIX_Properties::desc() const 
{
    std::stringstream ss ; 
    ss
        << "SOPTIX_Properties::desc" << std::endl 
        << std::setw(40) << "limitMaxTraceDepth" 
        << " : "
        << std::setw(10) <<  limitMaxTraceDepth 
        << std::endl 
        << std::setw(40) << "limitMaxTraversableGraphDepth" 
        << " : "
        << std::setw(10) <<  limitMaxTraversableGraphDepth 
        << std::endl 
        << std::setw(40) << "limitMaxPrimitivesPerGas" 
        << " : "
        << std::setw(10) << limitMaxPrimitivesPerGas 
        << std::setw(10) <<  std::hex << limitMaxPrimitivesPerGas << std::dec
        << std::endl 
        << std::setw(40) << "limitMaxInstancesPerIas" 
        << " : "
        << std::setw(10) << limitMaxInstancesPerIas 
        << std::setw(10) <<  std::hex << limitMaxInstancesPerIas << std::dec 
        << std::endl 
        << std::setw(40) << "rtcoreVersion" 
        << " : "
        << std::setw(10) <<  rtcoreVersion 
        << std::endl 
        << std::setw(40) << "limitMaxInstanceId" 
        << " : "
        << std::setw(10) << limitMaxInstanceId   
        << std::setw(10) <<  std::hex << limitMaxInstanceId << std::dec
        << std::endl 
        << std::setw(40) << "limitNumBitsInstanceVisibilityMask" 
        << " : "
        << std::setw(10) << limitNumBitsInstanceVisibilityMask 
        << std::endl 
        << std::setw(40) << "visibilityMask()"
        << " : "
        << std::setw(10) << visibilityMask() 
        << std::setw(10) << std::hex << visibilityMask() << std::dec  
        << std::endl 
        << std::setw(40) << "limitMaxSbtRecordsPerGas" 
        << " : "
        << std::setw(10) << limitMaxSbtRecordsPerGas 
        << std::setw(10) <<  std::hex << limitMaxSbtRecordsPerGas << std::dec
        << std::endl 
        << std::setw(40) << "limitMaxSbtOffset" 
        << " : "
        << std::setw(10) <<  limitMaxSbtOffset 
        << std::setw(10) <<  std::hex << limitMaxSbtOffset << std::dec
        << std::endl 
        ;  

    std::string str = ss.str(); 
    return str ; 
}


