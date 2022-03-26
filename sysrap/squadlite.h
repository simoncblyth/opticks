#pragma once


struct float3 { float x,y,z ;  };
struct float4 { float x,y,z,w ;  };
struct uint4 {  unsigned x,y,z,w ;  };
float3 make_float3(float x, float y, float z)          { float3 v ; v.x = x ; v.y = y ; v.z = z ;           return v ; } 
float4 make_float4(float x, float y, float z, float w ){ float4 v ; v.x = x ; v.y = y ; v.z = z ; v.w = w ; return v ; } 
uint4  make_uint4(unsigned x, unsigned y, unsigned z, unsigned w ){ uint4 v ; v.x = x ; v.y = y ; v.z = z ; v.w = w ; return v ; } 
union quad { float4 f ; uint4  u ;  }; 
struct quad4 { quad q0, q1, q2, q3 ; }; 

