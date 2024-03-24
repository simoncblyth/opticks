#pragma once
/**
scuda_pointer.h
=================

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



