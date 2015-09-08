#include "TProc.hh"
#include "assert.h"

#include "math_constants.h"

#include <thrust/system/cuda/execution_policy.h> 
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>


struct indexer
{
    indexer() {}

    __host__ __device__ uint4 operator()(unsigned int i)
    {   
        //unsigned int sel = i % 4 ; 
        return make_uint4( 1u, 2u, 3u, 4u);
    }   
};


void TProc::check()
{
    m_vtx.Summary("TProc::check m_vtx");
}


struct test 
{
    test() {}

    __host__ __device__ uint4 operator()(const uint4& v )
    {   
        return make_uint4( v.x+1, v.y+1, v.z+1, v.w+1);
    }   
};




struct scale
{
    float factor ; 
    scale(float factor) : factor(factor) {}
    __host__ __device__ float4 operator()(float4& v)
    {   
        return make_float4( v.x * factor, v.y * factor, v.z * factor, v.w  ) ;  // not scaling w 
    }   
};


void TProc::tscale(float factor)
{
    //printf("TProc::tscale factor %10.4f \n", factor);
    thrust::device_ptr<float4> pvtx = thrust::device_pointer_cast((float4*)m_vtx.dev_ptr);

    scale sc(factor) ; 
    thrust::transform(pvtx, pvtx+m_vtx.size, pvtx,  sc );
}



struct circle
{
    unsigned int nvert ; 
    float        radius ; 

    circle(unsigned int nvert, float radius) : nvert(nvert), radius(radius) {}

    __host__ __device__ float4 operator()(unsigned int i) const 
    {   
        float frac = float(i)/float(nvert) ; 
        float sinPhi, cosPhi;
        sincosf(2.f*CUDART_PI_F*frac,&sinPhi,&cosPhi);
        return make_float4( radius*sinPhi,  radius*cosPhi,  0.0f, 1.0f) ;
    }   
};

void TProc::tgenerate(float radius)
{
    thrust::device_ptr<float4> pvtx = thrust::device_pointer_cast((float4*)m_vtx.dev_ptr);

    printf("TProc::tgenerate radius %10.4f \n", radius);

    circle c(m_vtx.size, radius);

    thrust::counting_iterator<int> first(0);

    thrust::counting_iterator<int> last(m_vtx.size);

    thrust::transform(first, last, pvtx,  c );
}



