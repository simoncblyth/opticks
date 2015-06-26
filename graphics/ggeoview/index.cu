
// thrust-optix interop cmake build testing
//  https://github.com/thrust/thrust/issues/204

#include <thrust/device_vector.h> 
#include <thrust/remove.h> 

namespace optix { 
   class __align__(16) Aabb { 
      float3 m_min; 
      float3 m_max; 
   }; 
}// end namespace optix 


template<typename T > 
struct isZero { 
    __host__ __device__ T operator()(const T &x) const {return x==0;} 
}; 

void aabbValidCompaction(optix::Aabb *boxes, unsigned int *stencil, size_t num) 
{ 
    thrust::device_ptr<optix::Aabb > dev_begin_ptr(boxes); 
    thrust::device_ptr<optix::Aabb > dev_end_ptr(boxes + num); 
    thrust::device_ptr<unsigned int > dev_stencil_ptr(stencil); 
    thrust::remove_if(dev_begin_ptr, dev_end_ptr, dev_stencil_ptr, isZero<unsigned int >()); 
} 
