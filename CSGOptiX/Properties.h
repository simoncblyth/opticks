#pragma once

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
