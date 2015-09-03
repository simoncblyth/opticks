#include "callgrow.hh"
#include "CudaGLBuffer.hh"

// nvcc specifics hiding from clang in here
// alternate approach without having to subclass
//
// splitting struct definition into header and implementation
// with #include <thrust/detail/config/host_device.h>
// succeeds to compile the header with non-nvcc compilers but seems 
// useless as cannot compile the thrust::transform usage of the functor 
// in non-nvcc so just stick with higher level call wrapper approach 
//

struct grow_value
{
    unsigned int m_count ; 

    grow_value(unsigned int count) : m_count(count) {}

    __host__ __device__ float3 operator()(float3& v)
    {
        //float s = 0.01f * (m_count % 100)  ; 
        float s = 0.5 ; 
        return make_float3( v.x + s , v.y + s , 0.f ) ; 
        //return make_float3( 0.f , 0.1f , 0.2f );
    }
};

struct grow_index
{
    unsigned int m_count ; 

    grow_index(unsigned int count) : m_count(count) {}

    __host__ __device__ float3 operator()(unsigned int i)
    {
        float s = 0.1f + 0.01f * (m_count % 100)  ; 
        unsigned int j = i % 4 ; 
        float3 vec ; 
        switch(j)
        {
            case 0: vec = make_float3( 0.0f,     s,  0.0f) ; break ;
            case 1: vec = make_float3(    s,    -s,  0.0f) ; break ;
            case 2: vec = make_float3(   -s,    -s,  0.0f) ; break ;
            case 3: vec = make_float3( 0.0f,  0.0f,  0.0f) ; break ;
        }  
        return vec ; 
    }
};




void callgrow_value(CudaGLBuffer<float3>* cgb, unsigned int n, bool mapunmap)
{
    printf("callgrow_value %d mapunmap %d \n", n, mapunmap);
    grow_value f(n);
    cgb->thrust_transform_value( f, mapunmap );
}


void callgrow_index(CudaGLBuffer<float3>* cgb, unsigned int n, bool mapunmap)
{
    printf("callgrow_index %d mapunmap %d \n", n, mapunmap);
    grow_index f(n);
    cgb->thrust_transform_index( f, mapunmap );
}



