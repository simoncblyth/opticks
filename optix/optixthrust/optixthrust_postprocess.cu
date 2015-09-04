
//#include <cuda.h>
#include <optix_world.h>

#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "optixthrust.hh"
#include "helpers.hh"


struct scale
{
    float m_factor ; 

    scale(float factor) : m_factor(factor) {}

    __host__ __device__ float4 operator()(float4& v)
    {   
        return make_float4( v.x * m_factor, v.y * m_factor, v.z * m_factor, v.w * m_factor ) ; 
    }   
};


void OptiXThrust::postprocess()
{
    optix::Buffer buffer = m_context["output_buffer"]->getBuffer();

    thrust::device_ptr<float4> dptr = getThrustDevicePtr<float4>(buffer, m_device);    

    scale f(10.f);

    thrust::transform( dptr, dptr + m_size, dptr, f );

    float4 init = make_float4(0.f,0.f,0.f,0.f); 

    thrust::plus<float4> op ; 
 
    float4 sum = thrust::reduce(dptr, dptr + m_size , init, op );
 
    printf("OptiXThrust::postprocess sum %10.4f %10.4f %10.4f %10.4f \n", sum.x, sum.y, sum.z, sum.w );

}





