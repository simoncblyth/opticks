#pragma once


__device__ float
uniform(curandState *s, const float &low, const float &high)
{
    return low + curand_uniform(s)*(high-low);
}

__device__ float3
uniform_sphere(curandState *s) 
{
    float theta = uniform(s, 0.0f, 2.f*M_PIf);
    float u = uniform(s, -1.0f, 1.0f);
    float c = sqrtf(1.0f-u*u);

    return make_float3(c*cosf(theta), c*sinf(theta), u); 
}


