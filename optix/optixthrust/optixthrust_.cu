
#include <cuda_runtime_api.h>
#include <optix_world.h>

#include <thrust/reduce.h>
#include <thrust/for_each.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/count.h>
#include <thrust/copy.h>
#include <thrust/functional.h>

#include "strided_range.h"
#include "repeated_range.h"

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



struct photon_mask4 
{
    unsigned int lo, hi ;

    photon_mask4(unsigned int lo, unsigned int hi) : lo(lo), hi(hi) {} 

    template <typename Tuple> __device__ int operator()(Tuple t)
    {
        quad d ; 
        d.f = thrust::get<3>(t);
        unsigned int code = d.u.w ; 
        return code >= lo && code < hi ? 1 : 0 ; 
    }
};


union fui {
   float        f ; 
   unsigned int u ; 
   int          i ; 
};

struct dumper 
{
    __host__ __device__
    void operator()(float4 v)
    {
        quad q ; 
        q.f = v ; 
        printf("dumper xyzw:f %10.4f %10.4f %10.4f %10.4f  zw:u %5u %5u \n", v.x, v.y, v.z, v.w, q.u.z, q.u.w  );
    }
};

struct dumper4 
{
    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple t)
    {
        quad a, b, c, d ; 
        a.f = thrust::get<0>(t);
        b.f = thrust::get<1>(t);
        c.f = thrust::get<2>(t);
        d.f = thrust::get<3>(t);

        printf(
      "dumper4 [a] xyzw:f %10.4f %10.4f %10.4f %10.4f  zw:u %5u %5u \n"
      "        [b] xyzw:f %10.4f %10.4f %10.4f %10.4f  zw:u %5u %5u \n"
      "        [c] xyzw:f %10.4f %10.4f %10.4f %10.4f  zw:u %5u %5u \n"
      "        [d] xyzw:f %10.4f %10.4f %10.4f %10.4f  zw:u %5u %5u \n",
             a.f.x, a.f.y, a.f.z, a.f.w, a.u.z, a.u.w,   
             b.f.x, b.f.y, b.f.z, b.f.w, b.u.z, b.u.w,   
             c.f.x, c.f.y, c.f.z, c.f.w, c.u.z, c.u.w,   
             d.f.x, d.f.y, d.f.z, d.f.w, d.u.z, d.u.w
        );
    }
};








void OptiXThrust::sync()
{
    printf("OptiXThrust::sync\n");

    cudaDeviceSynchronize();  

    // Without the sync the process terminates before 
    // any printf from functor output streams get pumped out to the terminal, 
    // in some cases.
    // Curiously that doesnt seem to happen with device_vector, but does with device_ptr 
    // Maybe their dtors are delayed by the dumping
}


void OptiXThrust::for_each_dump()
{
    printf("OptiXThrust::for_each_dump size %u : dumping device_ptr  \n", m_size );

    thrust::device_ptr<float4> d = getThrustDevicePtr<float4>(m_buffer, m_device);    

    thrust::for_each( d, d+m_size, dumper() ); 
}

void OptiXThrust::strided()
{
    printf("OptiXThrust::strided size %u : dumping device_ptr  \n", m_size );

    thrust::device_ptr<float4> d = getThrustDevicePtr<float4>(m_buffer, m_device);    

    typedef thrust::device_vector<float4>::iterator Iterator;

    strided_range<Iterator> tenth(d + 1, d + m_size, 10);

    thrust::for_each( tenth.begin(), tenth.end(), dumper() ); 
}


void OptiXThrust::strided4()
{
    printf("OptiXThrust::strided4 size %u : dumping device_ptr  \n", m_size );

    assert(m_size % 4 == 0);

    thrust::device_ptr<float4> d = getThrustDevicePtr<float4>(m_buffer, m_device);    

    typedef thrust::device_vector<float4>::iterator Iterator;

    // stride by 4 with offsets 0:3 covers all entries 
    strided_range<Iterator> four0(d + 0, d + m_size, 4);
    strided_range<Iterator> four1(d + 1, d + m_size, 4);
    strided_range<Iterator> four2(d + 2, d + m_size, 4);
    strided_range<Iterator> four3(d + 3, d + m_size, 4);

    // 4-by-float4 feeding the functor
    thrust::for_each( 
          thrust::make_zip_iterator(thrust::make_tuple( four0.begin(), four1.begin(), four2.begin(), four3.begin() )),
          thrust::make_zip_iterator(thrust::make_tuple( four0.end(),   four1.end()  , four2.end()  , four3.end()   )),
          dumper4()
         );
}


void OptiXThrust::compaction4() 
{
    // Initial approach of count_if/copy_if targeting 
    // a zip iterator output failed at runtime.
    //
    // But actually the approach adopted of using a mask 
    // is preferable as the mask has an obvious meaning 
    // and could be created by other means, such as with OptiX.

    printf("OptiXThrust::compaction4 size %u \n", m_size );

    assert(m_size % 4 == 0);

    // masker functor is passed the four float4 of each photon
    // demo code : just masking based on uint index encoded in 4th float4
    photon_mask4 mskr(5, 15);   

    thrust::device_ptr<float4> d = getThrustDevicePtr<float4>(m_buffer, m_device);    

    thrust::device_vector<int> mask(m_size/4) ; 

    typedef thrust::device_vector<float4>::iterator Iterator;
    strided_range<Iterator> four0(d + 0, d + m_size, 4); 
    strided_range<Iterator> four1(d + 1, d + m_size, 4);
    strided_range<Iterator> four2(d + 2, d + m_size, 4);
    strided_range<Iterator> four3(d + 3, d + m_size, 4);

    thrust::transform(   // step thru float4 buffer in groups of 4*float4 
          thrust::make_zip_iterator(thrust::make_tuple( four0.begin(), four1.begin(), four2.begin(), four3.begin() )),
          thrust::make_zip_iterator(thrust::make_tuple( four0.end(),   four1.end()  , four2.end()  , four3.end()   )),
          mask.begin(),
          mskr );

    //printf("mask\n"); thrust::copy(mask.begin(), mask.end(), std::ostream_iterator<int>(std::cout, "\n")); 
    unsigned int num = thrust::count(mask.begin(), mask.end(), 1);  // number of 1s in the mask
    if( num > 0 )
    {
        int* d_mask = thrust::raw_pointer_cast(mask.data());
        float4* photons = make_masked_buffer( d_mask, mask.size(), num ); 
        dump_photons( photons, num );
    }
}


float4* OptiXThrust::make_masked_buffer( int* d_mask, unsigned int mask_size, unsigned int num )
{
    thrust::device_ptr<int> m = thrust::device_pointer_cast(d_mask) ; 

    assert(m_size % 4 == 0);

    assert(mask_size == m_size/4 );

    printf("OptiXThrust::make_masked_buffer num selected : %u \n", num );

    typedef thrust::device_vector<int>::iterator MaskIterator;

    repeated_range<MaskIterator> mask4( m , m + mask_size, 4);  // repeat by 4, so can use against raw float4 buffer 

    unsigned int t_size = num*4 ; 

    thrust::device_vector<float4> tmp(t_size) ; 

    thrust::device_ptr<float4> t = tmp.data() ;

    thrust::device_ptr<float4> d = getThrustDevicePtr<float4>(m_buffer, m_device);    

    thrust::copy_if( d,  d+m_size, mask4.begin(), t , thrust::identity<int>() );  
    // mask is input to predicate which controls the copy : identity predicate returns mask

    float4* dev = thrust::raw_pointer_cast(t);

    float4* host = new float4[t_size] ; 

    cudaMemcpy( host, dev, t_size*sizeof(float4),  cudaMemcpyDeviceToHost );

    return host ; 
}



void OptiXThrust::dump_photons( float4* host , unsigned int num )
{
    printf("OptiXThrust::dump_photons %u \n", num );
    for(unsigned int i=0 ; i < num ; i++)
    {
        quad a,b,c,d ;
        a.f = host[i*4+0] ;
        b.f = host[i*4+1] ;
        c.f = host[i*4+2] ;
        d.f = host[i*4+3] ;

        printf(
      "  (%u) \n"
      "        [a] xyzw:f %10.4f %10.4f %10.4f %10.4f  zw:u %5u %5u \n"
      "        [b] xyzw:f %10.4f %10.4f %10.4f %10.4f  zw:u %5u %5u \n"
      "        [c] xyzw:f %10.4f %10.4f %10.4f %10.4f  zw:u %5u %5u \n"
      "        [d] xyzw:f %10.4f %10.4f %10.4f %10.4f  zw:u %5u %5u \n",
             i,
             a.f.x, a.f.y, a.f.z, a.f.w, a.u.z, a.u.w,   
             b.f.x, b.f.y, b.f.z, b.f.w, b.u.z, b.u.w,   
             c.f.x, c.f.y, c.f.z, c.f.w, c.u.z, c.u.w,   
             d.f.x, d.f.y, d.f.z, d.f.w, d.u.z, d.u.w
        );
    }
}


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




