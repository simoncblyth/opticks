#pragma once
/**
Pointer.h
============

https://github.com/ingowald/optix7course/blob/master/example08_addingTextures/devicePrograms.cu

See env-;optix7c-

**/

static __forceinline__ __device__ void* unpackPointer( uint32_t i0, uint32_t i1 )
{
    const uint64_t uptr = static_cast<uint64_t>( i0 ) << 32 | i1;
    void*           ptr = reinterpret_cast<void*>( uptr ); 
    return ptr;
}

static __forceinline__ __device__ void  packPointer( void* ptr, uint32_t& i0, uint32_t& i1 )
{
    const uint64_t uptr = reinterpret_cast<uint64_t>( ptr );
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}

/**
getPRD
--------

An arbitrary payload is associated with each ray that is initialized with the optixTrace call. 
The payload is passed to all the IS, AH, CH and MS programs that are executed during this invocation of trace. 
The payload can be read and written by each program 

**/

template<typename T> static __forceinline__ __device__ T *getPRD()
{ 
    const uint32_t u0 = optixGetPayload_0();
    const uint32_t u1 = optixGetPayload_1();
    return reinterpret_cast<T*>( unpackPointer( u0, u1 ) );
}


