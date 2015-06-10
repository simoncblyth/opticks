#pragma once


// union of "standard" sized vector types, all 4*32 = 128 bit 
union quad
{
   float4 f ;
   int4   i ;
   uint4  u ;
};


// union of "half" sized vector types, all 4*16 = 64 bit 
// see /Developer/NVIDIA/CUDA-5.5/include/vector_types.h
union squad
{
   short4   s ;
   ushort4  u ;
};






