#include "TAry.hh"
#include "assert.h"

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


void TAry::check()
{
    m_src.Summary("TAry::transform m_src");
    m_dst.Summary("TAry::transform m_dst");

    assert(m_src.dev_ptr != m_dst.dev_ptr) ;
    assert(m_src.size == m_dst.size) ;
    assert(m_src.num_bytes == m_dst.num_bytes) ;
}


struct test 
{
    test() {}

    __host__ __device__ uint4 operator()(const uint4& v )
    {   
        return make_uint4( v.x+1, v.y+1, v.z+1, v.w+1);
    }   
};


void TAry::transform()
{
    thrust::device_ptr<uint4> psrc = thrust::device_pointer_cast((uint4*)m_src.dev_ptr);
    thrust::device_ptr<uint4> pdst = thrust::device_pointer_cast((uint4*)m_dst.dev_ptr);

    test tst ; 
    thrust::transform(psrc, psrc+m_src.size, pdst,  tst );
}

void TAry::tfill()
{
    printf("TAry::tfill \n");
    thrust::device_ptr<uint4> pdst = thrust::device_pointer_cast((uint4*)m_dst.dev_ptr);
    uint4 value = make_uint4( 1, 2, 3, 4);
    thrust::fill( pdst, pdst + m_dst.size, value );
}

void TAry::tcopy()
{
    printf("TAry::tcopy \n");
    thrust::device_ptr<uint4> psrc = thrust::device_pointer_cast((uint4*)m_src.dev_ptr);
    thrust::device_ptr<uint4> pdst = thrust::device_pointer_cast((uint4*)m_dst.dev_ptr);
    thrust::copy( psrc, psrc+m_src.size , pdst );
}


__global__ void kcopy( uint4* src, uint4* dst ) 
{
    int offset = blockIdx.x*blockDim.x + threadIdx.x;
    dst[offset].x = src[offset].x;
    dst[offset].y = src[offset].y;
    dst[offset].z = src[offset].z;
    dst[offset].w = src[offset].w;
}

void TAry::kcall()
{
    unsigned int N = m_src.size ; 
    printf("TAry::kcall %u \n", N);
    kcopy<<<N,1>>>( (uint4*)m_src.dev_ptr, (uint4*)m_dst.dev_ptr );
}   
void TAry::memcpy()   // works
{
    printf("TAry::memcpy \n");
    cudaMemcpy( m_dst.dev_ptr, m_src.dev_ptr, m_dst.num_bytes, cudaMemcpyDeviceToDevice );
}
void TAry::memset()   // works : but operates at byte level so difficult to set smth other than zero
{
    printf("TAry::memset \n");
    cudaMemset( m_dst.dev_ptr, 0, m_dst.num_bytes );
}





