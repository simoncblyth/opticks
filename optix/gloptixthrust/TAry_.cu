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


struct test 
{
    test() {}

    __host__ __device__ uint4 operator()(const uint4& v )
    {   
        return make_uint4( v.x+1, v.y+1, v.z+1, v.w+1);
    }   
};



/*
void TAry::copyToHost( void* host )
{
    printf("TAry::copyToHost num bytes %d \n", m_num_bytes);
    cudaMemcpy(host, m_dptr, m_num_bytes, cudaMemcpyDeviceToHost);
}
*/



void TAry::check()
{
    m_src.Summary("TAry::transform m_src");
    m_dst.Summary("TAry::transform m_dst");

    assert(m_src.dev_ptr != m_dst.dev_ptr) ;
    assert(m_src.size == m_dst.size) ;
    assert(m_src.num_bytes == m_dst.num_bytes) ;
}


void TAry::transform()
{
    thrust::device_ptr<uint4> psrc = thrust::device_pointer_cast((uint4*)m_src.dev_ptr);
    thrust::device_ptr<uint4> pdst = thrust::device_pointer_cast((uint4*)m_dst.dev_ptr);

    thrust::device_vector<uint4> vsrc(psrc, psrc+m_src.size);
    thrust::device_vector<uint4> vdst(pdst, pdst+m_dst.size);

    thrust::copy( vsrc.begin(), vsrc.end(), vdst.begin());

    thrust::host_vector<uint4>   hsrc(m_src.size) ;
    thrust::copy( vsrc.begin(), vsrc.end(), hsrc.begin());

    thrust::host_vector<uint4>   hdst(m_dst.size) ;
    thrust::copy( vdst.begin(), vdst.end(), hdst.begin());

    for(unsigned int i=0 ; i < m_src.size ; i++)
        printf("TAry(hsrc)  %2u : %2u %2u %2u %2u \n", i, hsrc[i].x, hsrc[i].y, hsrc[i].z, hsrc[i].w ); 
    for(unsigned int i=0 ; i < m_dst.size ; i++)
        printf("TAry(hdst)  %2u : %2u %2u %2u %2u \n", i, hdst[i].x, hdst[i].y, hdst[i].z, hdst[i].w ); 
}

/*
   cuda functions and kernel calls succeed to update the OpenGL buffer, so
   it seems the problem is with Thrust, and not with the interop-ing

   https://github.com/thrust/thrust/issues/683

*/


void TAry::tcopy()
{
    thrust::device_ptr<uint4> psrc = thrust::device_pointer_cast((uint4*)m_src.dev_ptr);
    thrust::device_ptr<uint4> pdst = thrust::device_pointer_cast((uint4*)m_dst.dev_ptr);

    thrust::device_vector<uint4> vsrc(psrc, psrc+m_src.size);
    thrust::device_vector<uint4> vdst(pdst, pdst+m_dst.size);

    thrust::copy( thrust::cuda::par.on(0), vsrc.begin(), vsrc.end(), vdst.begin());
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



void TAry::transform_old()
{
    thrust::device_ptr<uint4> p = thrust::device_pointer_cast((uint4*)m_src.dev_ptr);

    thrust::host_vector<uint4> h(m_src.size) ;

    thrust::device_vector<uint4> d(p, p+m_src.size);

    thrust::copy( d.begin(), d.end(), h.begin());

    for(unsigned int i=0 ; i < m_src.size ; i++)
    {
        uint4 u = h[i] ; 
        printf("TAry(bef)  %2u : %2u %2u %2u %2u \n", i, u.x, u.y, u.z, u.w ); 
    }


    /*
    thrust::counting_iterator<unsigned int> first(0);
    thrust::counting_iterator<unsigned int> last(m_src.size);
    indexer idx ;
    thrust::transform(first, last, v.begin(),  idx );
    */

    test tst ; 
    thrust::transform(d.begin(),d.end(), d.begin(),  tst );

    //cudaDeviceSynchronize();
    cudaStreamSynchronize(0);

    thrust::copy( d.begin(), d.end(), h.begin());
    for(unsigned int i=0 ; i < m_src.size ; i++)
    {
        uint4 u = h[i] ; 
        printf("TAry(aft)  %2u : %2u %2u %2u %2u \n", i, u.x, u.y, u.z, u.w ); 
    }

}





/*
struct scale
{
    float m_factor ; 

    scale(float factor) : m_factor(factor) {}

    __host__ __device__ float4 operator()(float4& v)
    {   
        return make_float4( v.x * m_factor, v.y * m_factor, v.z * m_factor, v.w  ) ;  // not scaling w 
        //return make_float4( 0.5f, 0.5f, 0.5f, 1.0f  ) ;  // not scaling w 
    }   
};

void OBuffer::thrust_transform(float factor)
{
    thrust::device_ptr<float4> dptr = thrust::device_pointer_cast((float4*)m_dptr);

    thrust::device_vector<float4> dvec(dptr, dptr+m_size);

    printf("OBuffer::thrust_transform scale by factor %10.4f \n", factor);
    scale f(factor);

    thrust::transform( dvec.begin(), dvec.end(), dvec.begin(), f );

    float4 a = dvec[0] ;
    printf("dvec[0] %10.4f %10.4f %10.4f %10.4f \n", a.x, a.y, a.z, a.w );

}


struct generate
{
    unsigned int m_nvert ; 
    float       m_radius ; 

    generate(unsigned int nvert, float radius) : m_nvert(nvert), m_radius(radius) {}

    __host__ __device__ float4 operator()(unsigned int i) const 
    {   
        float frac = float(i)/float(m_nvert) ; 
        float sinPhi, cosPhi;
        sincosf(2.f*M_PIf*frac,&sinPhi,&cosPhi);
        return make_float4( m_radius*sinPhi,  m_radius*cosPhi,  0.0f, 1.0f) ;

    }   
};



void TProc::thrust_generate(float radius)
{
    mapGLToCUDA();

    thrust::device_ptr<float4> dptr = thrust::device_pointer_cast((float4*)m_dptr);

    thrust::device_vector<float4> dvec(dptr, dptr+m_size);

    printf("OBuffer::thrust_generate radius %10.4f \n", radius);

    generate g(m_size, radius);

    thrust::counting_iterator<int> first(0);

    thrust::counting_iterator<int> last(m_size);

    thrust::transform(first, last, dvec.begin(),  g );

    float4 a = dvec[0] ;
    printf("dvec[0] %10.4f %10.4f %10.4f %10.4f \n", a.x, a.y, a.z, a.w );

    unmapGLToCUDA();
}

*/


