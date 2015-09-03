
//#include <cuda.h>
#include <optix_world.h>

#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "optixthrust.hh"
#include "helpers.hh"


void OptiXThrust::postprocess()
{
    int deviceNumber = 0 ; 

    thrust::device_ptr<float4> dptr = getThrustDevicePtr<float4>(m_buffer, deviceNumber);    

    float4 init = make_float4(0.f,0.f,0.f,0.f); 
 
    int size = m_width*m_height ; 

    thrust::plus<float4> binary_op;

    float4 sum = thrust::reduce(dptr, dptr + size , init, binary_op);
 
    printf("OptiXThrust::postprocess sum %10.4f %10.4f %10.4f %10.4f \n", sum.x, sum.y, sum.z, sum.w );
}



