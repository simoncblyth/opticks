#include "TAry.hh"
#include "assert.h"

//#include <thrust/reduce.h>
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



void TAry::copyToHost( void* host )
{
    printf("TAry::copyToHost num bytes %d \n", m_num_bytes);
    cudaMemcpy(host, m_dptr, m_num_bytes, cudaMemcpyDeviceToHost);
}


void TAry::transform()
{
    printf("TAry::transform dptr %p size %d \n", m_dptr, m_size );

    thrust::device_ptr<uint4> p = thrust::device_pointer_cast((uint4*)m_dptr);

    thrust::host_vector<uint4> h(m_size) ;

    thrust::device_vector<uint4> d(p, p+m_size);

    thrust::copy( d.begin(), d.end(), h.begin());

    for(unsigned int i=0 ; i < m_size ; i++)
    {
        uint4 u = h[i] ; 
        printf("TAry(bef)  %2u : %2u %2u %2u %2u \n", i, u.x, u.y, u.z, u.w ); 
    }

    /*
    thrust::counting_iterator<unsigned int> first(0);
    thrust::counting_iterator<unsigned int> last(m_size);
    indexer idx ;
    thrust::transform(first, last, v.begin(),  idx );
    */

    test tst ; 
    thrust::transform(d.begin(),d.end(), d.begin(),  tst );

    //cudaDeviceSynchronize();
    cudaStreamSynchronize(0);


    thrust::copy( d.begin(), d.end(), h.begin());
    for(unsigned int i=0 ; i < m_size ; i++)
    {
        uint4 u = h[i] ; 
        printf("TAry(aft)  %2u : %2u %2u %2u %2u \n", i, u.x, u.y, u.z, u.w ); 
    }

    if(m_hostcopy)
    {
        //copyToHost(m_hostcopy);
        memcpy(m_hostcopy, h.data(), m_num_bytes );
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


