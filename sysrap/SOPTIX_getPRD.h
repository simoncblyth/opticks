#pragma once
/**
SOPTIX_getPRD.h : unpackPointer from optixPayload
==================================================

An arbitrary payload is associated with each ray that is initialized with the optixTrace call. 
The payload is passed to all the IS, AH, CH and MS programs that are executed during this invocation of trace. 
The payload can be read and written by each program 

Using unpackPointer from "scuda_pointer.h" via SOPTIX.cu

**/


template<typename T> static __forceinline__ __device__ T * SOPTIX_getPRD()
{
    const uint32_t u0 = optixGetPayload_0();
    const uint32_t u1 = optixGetPayload_1();
    return reinterpret_cast<T*>( unpackPointer( u0, u1 ) );
}

