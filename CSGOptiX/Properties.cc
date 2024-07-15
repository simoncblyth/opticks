#include "Properties.h"
#include "Ctx.h"
#include <optix.h>
#include <optix_stubs.h>
#include "OPTIX_CHECK.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <bitset>
#include <cassert>

Properties::Properties()
{
    OPTIX_CHECK( optixDeviceContextGetProperty(Ctx::context, OPTIX_DEVICE_PROPERTY_LIMIT_MAX_TRACE_DEPTH                   , &limitMaxTraceDepth                 , sizeof(unsigned int)) );
    OPTIX_CHECK( optixDeviceContextGetProperty(Ctx::context, OPTIX_DEVICE_PROPERTY_LIMIT_MAX_TRAVERSABLE_GRAPH_DEPTH       , &limitMaxTraversableGraphDepth      , sizeof(unsigned int)) );
    OPTIX_CHECK( optixDeviceContextGetProperty(Ctx::context, OPTIX_DEVICE_PROPERTY_LIMIT_MAX_PRIMITIVES_PER_GAS            , &limitMaxPrimitivesPerGas           , sizeof(unsigned int)) );
    OPTIX_CHECK( optixDeviceContextGetProperty(Ctx::context, OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCES_PER_IAS             , &limitMaxInstancesPerIas            , sizeof(unsigned int)) );
    OPTIX_CHECK( optixDeviceContextGetProperty(Ctx::context, OPTIX_DEVICE_PROPERTY_RTCORE_VERSION                          , &rtcoreVersion                      , sizeof(unsigned int)) );
    OPTIX_CHECK( optixDeviceContextGetProperty(Ctx::context, OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCE_ID                   , &limitMaxInstanceId                 , sizeof(unsigned int)) );
    OPTIX_CHECK( optixDeviceContextGetProperty(Ctx::context, OPTIX_DEVICE_PROPERTY_LIMIT_NUM_BITS_INSTANCE_VISIBILITY_MASK , &limitNumBitsInstanceVisibilityMask , sizeof(unsigned int)) );
    OPTIX_CHECK( optixDeviceContextGetProperty(Ctx::context, OPTIX_DEVICE_PROPERTY_LIMIT_MAX_SBT_RECORDS_PER_GAS           , &limitMaxSbtRecordsPerGas           , sizeof(unsigned int)) );
    OPTIX_CHECK( optixDeviceContextGetProperty(Ctx::context, OPTIX_DEVICE_PROPERTY_LIMIT_MAX_SBT_OFFSET                    , &limitMaxSbtOffset                  , sizeof(unsigned int)) );
}


/**
Properties::visibilityMask_FULL
-------------------------------------

::

    +---------------------+--------+----------+
    | ( 0x1 << 8 ) - 1    |   255  |  0xff    |
    +---------------------+--------+----------+

**/

unsigned Properties::visibilityMask_FULL() const
{
    return ( 0x1 << limitNumBitsInstanceVisibilityMask ) - 1 ;   
}



unsigned Properties::visibilityMask(unsigned idx) const
{
    unsigned FULL = visibilityMask_FULL(); 
    assert( FULL == 0xffu );  
    unsigned BITS = std::bitset<32>(FULL).count(); 
    assert( BITS == 8 );  
    unsigned marker_bit = std::min( idx, BITS - 1 );  
    unsigned visibilityMask = 0x1 << marker_bit ;   
    assert( ( visibilityMask & 0xffffff00 ) == 0 ) ;   
    return visibilityMask ;
}



std::string Properties::desc() const 
{
    std::stringstream ss ; 
    ss
        << "Properties::desc" << std::endl 
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
        << std::setw(10) <<  std::dec << limitMaxPrimitivesPerGas 
        << std::setw(10) <<  std::hex << limitMaxPrimitivesPerGas 
        << std::endl 
        << std::setw(40) << "limitMaxInstancesPerIas" 
        << " : "
        << std::setw(10) <<  std::dec << limitMaxInstancesPerIas 
        << std::setw(10) <<  std::hex << limitMaxInstancesPerIas 
        << std::endl 
        << std::setw(40) << "rtcoreVersion" 
        << " : "
        << std::setw(10) <<  rtcoreVersion 
        << std::endl 
        << std::setw(40) << "limitMaxInstanceId" 
        << " : "
        << std::setw(10) <<  std::dec << limitMaxInstanceId   
        << std::setw(10) <<  std::hex << limitMaxInstanceId 
        << std::endl 
        << std::setw(40) << "limitNumBitsInstanceVisibilityMask" 
        << " : "
        << std::setw(10) <<  std::dec << limitNumBitsInstanceVisibilityMask 
        << std::endl 
        << std::setw(40) << "visibilityMask_FULL" 
        << " : "
        << std::setw(10) <<  std::hex << visibilityMask_FULL() << std::dec 
        << std::endl 
        << std::setw(40) << "limitMaxSbtRecordsPerGas" 
        << " : "
        << std::setw(10) <<  std::dec << limitMaxSbtRecordsPerGas 
        << std::setw(10) <<  std::hex << limitMaxSbtRecordsPerGas 
        << std::endl 
        << std::setw(40) << "limitMaxSbtOffset" 
        << " : "
        << std::setw(10) <<  std::dec << limitMaxSbtOffset 
        << std::setw(10) <<  std::hex << limitMaxSbtOffset 
        << std::dec
        << std::endl 
        ;  

    std::string str = ss.str(); 
    return str ; 
}




