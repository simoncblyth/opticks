#pragma once
/**
SOPTIX_Properties.h : optixDeviceContextGetProperty results
=============================================================
**/
#include <bitset>

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

    unsigned visibilityMask_FULL() const ;     
    unsigned visibilityMask(unsigned idx) const ;
     
    SOPTIX_Properties(OptixDeviceContext context); 
    std::string desc() const ; 
}; 



/**
SOPTIX_Properties::visibilityMask_FULL
----------------------------------------

::

    +---------------------+--------+----------+
    | ( 0x1 << 8 ) - 1    |   255  |  0xff    |
    +---------------------+--------+----------+

**/

inline unsigned SOPTIX_Properties::visibilityMask_FULL() const
{
    return ( 0x1 << limitNumBitsInstanceVisibilityMask ) - 1 ;  
}


/**
SOPTIX_Properties::visibilityMask
------------------------------------

    +----------+--------------------------------------+
    |  idx     |   visibilityMask(idx)                |
    +==========+======================================+
    |   0      |   0x1 << std::min(0, 7) = 0x1 << 0   | 
    |   1      |   0x1 << std::min(1, 7) = 0x1 << 1   | 
    |   2      |   0x1 << std::min(2, 7) = 0x1 << 2   | 
    |   3      |   0x1 << std::min(3, 7) = 0x1 << 3   | 
    |   4      |   0x1 << std::min(4, 7) = 0x1 << 4   | 
    |   5      |   0x1 << std::min(5, 7) = 0x1 << 5   | 
    |   6      |   0x1 << std::min(6, 7) = 0x1 << 6   | 
    +----------+--------------------------------------+
    |   7      |   0x1 << std::min(7, 7) = 0x1 << 7   | 
    |   8      |   0x1 << std::min(8, 7) = 0x1 << 7   | 
    |   9      |   0x1 << std::min(9, 7) = 0x1 << 7   | 
    |  10      |   0x1 << std::min(10,7) = 0x1 << 7   | 
    +----------+--------------------------------------+


For idx of 7 or more visibilityMask does not change, reflecting
the limited number of bits of the mask. 

This means that visibility of geometry with the first seven idx 
(0,1,2,3,4,5,6) can be individually controlled but all the rest 
of the geometry with higher idx can only be controlled 
all together. 

For example with mmlabel.txt::
 
  +-----+----------------------------------------+
  | idx | mmlable                                |
  +=====+========================================+ 
  |  0  | 3218:sWorld                            |
  |  1  | 5:PMT_3inch_pmt_solid                  |
  |  2  | 9:NNVTMCPPMTsMask_virtual              |
  |  3  | 12:HamamatsuR12860sMask_virtual        |
  |  4  | 4:mask_PMT_20inch_vetosMask_virtual    |
  |  5  | 1:sStrutBallhead                       |
  |  6  | 1:uni1                                 |
  +-----+----------------------------------------+
  |  7  | 1:base_steel                           |
  |  8  | 1:uni_acrylic1                         |
  |  9  | 130:sPanel                             |
  +-----+----------------------------------------+

::

    VIZMASK=5 ~/o/sysrap/tests/ssst1.sh run   # just 1:sStrutBallhead 
    VIZMASK=6 ~/o/sysrap/tests/ssst1.sh run   # just 1:uni1
    VIZMASK=7 ~/o/sysrap/tests/ssst1.sh run   # just 1:base_steel
    VIZMASK=8 ~/o/sysrap/tests/ssst1.sh run   # just 1:uni_acrylic1 OpenGL, blank with OptiX
    VIZMASK=9 ~/o/sysrap/tests/ssst1.sh run   # just 130:sPanel OpenGL, blank with OptiX


NB currently the OptiX render and the OpenGL render do not match for
VIZMASK=7,8,9 because OptiX has the 8 bit limitation but OpenGL does not. 

**/

inline unsigned SOPTIX_Properties::visibilityMask(unsigned idx) const
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
        << std::setw(40) << "visibilityMask_FULL()"
        << " : "
        << std::setw(10) << visibilityMask_FULL() 
        << std::setw(10) << std::hex << visibilityMask_FULL() << std::dec  
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


