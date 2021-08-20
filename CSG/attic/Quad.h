#pragma once

// below also in sysrap scuda.h : TODO: eliminate this

union quad
{
   float4 f ; 
   int4   i ; 
   uint4  u ; 
};

struct quad4 { quad q0, q1, q2, q3 ; };

