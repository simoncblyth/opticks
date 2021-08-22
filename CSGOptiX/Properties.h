#pragma once

/**
Properties
============

OptiX 7.0::

    Properties::dump
                          limitMaxTraceDepth :         31
               limitMaxTraversableGraphDepth :         16
                    limitMaxPrimitivesPerGas :  536870912  20000000
                     limitMaxInstancesPerIas :   16777216   1000000
                               rtcoreVersion :          0
                          limitMaxInstanceId :   16777215    ffffff
          limitNumBitsInstanceVisibilityMask :          8
                    limitMaxSbtRecordsPerGas :   16777216   1000000
                           limitMaxSbtOffset :   16777215    ffffff
**/

struct Properties 
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

    Properties(); 
    void dump() const ; 
};
