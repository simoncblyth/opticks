
//#include <cuda.h>
#include <optix_world.h>

#include <iostream>
#include <iterator>

#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "gloptixthrust.hh"
#include "helpers.hh"


struct scale
{
    float m_factor ; 

    scale(float factor) : m_factor(factor) {}

    __host__ __device__ float4 operator()(float4& v)
    {   
        return make_float4( v.x * m_factor, v.y * m_factor, v.z * m_factor, v.w  ) ;  // not scaling w 
    }   
};


void GLOptiXThrust::postprocess(float factor)
{
    optix::Buffer buffer = m_context["output_buffer"]->getBuffer();

    thrust::device_ptr<float4> dptr = getThrustDevicePtr<float4>(buffer, m_device);    
    thrust::device_vector<float4> dvec(dptr, dptr+m_size);

    printf("GLOptiXThrust::postprocess scale by factor %10.4f \n", factor);
    scale f(factor);

    thrust::transform( dvec.begin(), dvec.end(), dvec.begin(), f );

    //thrust::maximum<float4> op ;
    //float4 res = thrust::reduce(dvec.begin(), dvec.end() , init, op );
    //printf("GLOptiXThrust::postprocess max %10.4f %10.4f %10.4f %10.4f \n", res.x, res.y, res.z, res.w );

    float4 a = dvec[0] ;
    printf("dvec[0] %10.4f %10.4f %10.4f %10.4f \n", a.x, a.y, a.z, a.w );

    //std::cout << "dvec:" << std::endl;
    //thrust::copy(dvec.begin(), dvec.end(), std::ostream_iterator(std::cout, " "));
    //std::cout << std::endl << std::endl;


    buffer->markDirty();
}

void GLOptiXThrust::sync()
{
    printf("GLOptiXThrust::sync\n");
    cudaDeviceSynchronize();
}




