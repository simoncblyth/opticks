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
        unsigned int sel = i % 4 ; 
        return make_uint4( sel, 0u, 0u, 0u);
    }   
};


void TAry::transform()
{
    printf("TAry::transform dptr %p size %d \n", m_dptr, m_size );

    thrust::device_ptr<uint4> p = thrust::device_pointer_cast((uint4*)m_dptr);

    thrust::device_vector<uint4> v(p, p+m_size);

    thrust::counting_iterator<unsigned int> first(0);

    thrust::counting_iterator<unsigned int> last(m_size);

    indexer idx ;

    thrust::transform(first, last, v.begin(),  idx );

    cudaDeviceSynchronize();

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


