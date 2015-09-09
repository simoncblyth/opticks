
//#include <cuda.h>
#include <optix_world.h>

#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/count.h>
#include <thrust/copy.h>

#include "optixthrust.hh"
#include "helpers.hh"

#include "quad.h"
#include "photon.h"

struct scale
{
    float m_factor ; 

    scale(float factor) : m_factor(factor) {}

    __host__ __device__ float4 operator()(float4& v)
    {   
        return make_float4( v.x * m_factor, v.y * m_factor, v.z * m_factor, v.w * m_factor ) ; 
    }   
};

void OptiXThrust::photon_test()
{
    photon_t p = make_photon_tagged( 1u );
    quad q ; 
    q.f = p.d ; 
    printf("photon_test p.d.w (f) %10.4f p.d.w (u) %u \n", p.d.w, q.u.w  );
}

void OptiXThrust::postprocess()
{
    optix::Buffer buffer = m_context["output_buffer"]->getBuffer();

    thrust::device_ptr<float4> dptr = getThrustDevicePtr<float4>(buffer, m_device);    

    scale f(0.5f);

    thrust::transform( dptr, dptr + m_size, dptr, f );

    float4 init = make_float4(0.f,0.f,0.f,0.f); 

    thrust::plus<float4> op ; 
 
    float4 sum = thrust::reduce(dptr, dptr + m_size , init, op );
 
    printf("OptiXThrust::postprocess sum %10.4f %10.4f %10.4f %10.4f \n", sum.x, sum.y, sum.z, sum.w );

}


struct is_tagged : public thrust::unary_function<float4,bool>
{
    unsigned int tag ;  
    is_tagged(unsigned int tag) : tag(tag) {}

    __host__ __device__
    bool operator()(float4 v)
    {
        quad q ; 
        q.f = v ; 
        return q.u.w == tag ;
    }
};


union fui {
   float        f ; 
   unsigned int u ; 
   int          i ; 
};



void OptiXThrust::compaction()
{
/*

* host memory allocation is minimal, just the required size using count_if  

* union tricks used to plant tagging uints inside the float4

* use temporary device vector to hold the selection, assuming
  that dev to dev compaction will be preferable to dev to host  

* cudaMemcpy used for device to host copying rather than thrust
  to minimize extra copying : eg allocate an NPY of required size and then 
  cudaMemcpy directly into it.  

  Actually inside CUDA this is not direct as copying to normal paged 
  memory requires staging via pinned. If this copy turns out to 
  be bottleneck could avoid the staging using cudaMalloc of host pinned memory ?

  * http://devblogs.nvidia.com/parallelforall/how-optimize-data-transfers-cuda-cc/

*/
    optix::Buffer buffer = m_context["output_buffer"]->getBuffer();

    thrust::device_ptr<float4> dptr = getThrustDevicePtr<float4>(buffer, m_device);    

    unsigned int tag = 5 ; 

    is_tagged tagger(tag) ; 

    size_t num_tagged = thrust::count_if(dptr, dptr+m_size, tagger );

    printf("OptiXThrust::compaction num_tagged with tag %u : %lu \n", tag, num_tagged);

    thrust::device_vector<float4> tmp(num_tagged) ; 

    thrust::copy_if(dptr, dptr+m_size, tmp.begin(), tagger );

    float4* dev = thrust::raw_pointer_cast(tmp.data());

    float4* host = new float4[num_tagged] ; 

    cudaMemcpy( host, dev, num_tagged*sizeof(float4),  cudaMemcpyDeviceToHost );

    for(unsigned int i=0 ; i < num_tagged ; i++)
    {
        const float4& v = host[i] ;
        fui w ;
        w.f = v.w ; 
        printf(" %u : %10.4f %10.4f %10.4f %10.4f : %u \n", i, v.x, v.y, v.z, v.w, w.u );
    }

}




