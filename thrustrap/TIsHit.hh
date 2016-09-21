#pragma once

#define TIS_HIT_K 'W'

#include "float4x4.h"


struct TIsHit : public thrust::unary_function<float4x4,bool>
{
    TIsHit() {}

    __host__ __device__
    bool operator()(float4x4 v)
    {   
        tquad q3 ; 
        q3.f = v.q3 ; 
#if TIS_HIT_K == 'X'
        return q3.u.x > 0 ;
#elif TIS_HIT_K == 'Y'
        return q3.u.y > 0 ;
#elif TIS_HIT_K == 'Z'
        return q3.u.z > 0 ;
#elif TIS_HIT_K == 'W'
        return q3.u.w > 0 ;
#endif
    }   
};

