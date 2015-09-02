#include <iostream>
#include <string>
#include <cuda_gl_interop.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <optixu/optixpp_namespace.h>

/*
*GTOBuffer* : OpenGL Thrust/CUDA OptiX Interop Buffer
=======================================================

*GTO* corresponds to the interop layering order

   *G*  OpenGL
   *T*  CUDA/Thrust  
   *O*  OptiX

*/

std::string ptxpath(const char* name, const char* ptxdir){
    char path[128] ; 
    snprintf(path, 128, "%s/%s", getenv(ptxdir), name );
    return path ;   
}

template <class T>
class GTOBuffer {
   public:
   public:
       GTOBuffer(
           optix::Context context, 
           const char* name, 
           unsigned int type, 
           unsigned int buffer_id, 
           unsigned int flags, 
           cudaStream_t stream=0);
   public:
       void Summary(const char* msg="InteropBuffer::Summary");
       template <typename F> void thrust_transform(const F& f);
   public:
       void optixMap();
       void optixLaunch(unsigned int entry );
       void optixUnmap();
   private:
       static RTformat FORMAT ;
   private:
       void mapResources();
       thrust::device_ptr<T> getDevicePtr();   
       T* getRawPtr();   
       void unmapResources();
   private:  
       optix::Context m_context ; 
       optix::Buffer  m_buffer ; 
       const char*    m_name ; 
       unsigned int   m_type ; // eg RT_BUFFER_INPUT_OUTPUT
       unsigned int   m_device_number ; 
   private:  
       unsigned int   m_buffer_id ;   
   private:  
       unsigned int   m_flags ;   
       cudaStream_t   m_stream ; 
       bool           m_mapped ; 
       struct cudaGraphicsResource* m_graphics_resource ;
   private:
       size_t         m_bufsize ; // dimensions obtained by getRawPtr
       unsigned int   m_first ; 
       unsigned int   m_last ; 
       unsigned int   m_width ; 
};


template <typename T>
inline GTOBuffer<T>::GTOBuffer(
           optix::Context context, 
           const char* name, 
           unsigned int type, 
           unsigned int buffer_id, 
           unsigned int flags, 
           cudaStream_t stream)
     :
     m_context(context),
     m_name(strdup(name)),
     m_type(type),
     m_device_number(0), 
     m_buffer_id(buffer_id),
     m_flags(flags),
     m_stream(stream),
     m_mapped(false),
     m_bufsize(0),
     m_first(0),
     m_last(0),
     m_width(0)
{
    cudaGraphicsGLRegisterBuffer(&m_graphics_resource, m_buffer_id, m_flags);
}

template <typename T>
inline void GTOBuffer<T>::mapResources()
{
    assert(m_mapped == false);
    unsigned int count = 1 ; 
    cudaError_t error = cudaGraphicsMapResources(count, &m_graphics_resource, m_stream);
    m_mapped = true ; 
}
template <typename T>
inline void GTOBuffer<T>::unmapResources()
{
    assert(m_mapped == true);
    unsigned int count = 1 ; 
    cudaError_t error = cudaGraphicsUnmapResources(count, &m_graphics_resource, m_stream);
    m_mapped = false ; 
}
template <typename T>
inline void GTOBuffer<T>::Summary(const char* msg)
{
    printf("%s buffer_id %d bufsize %lu first %d last %d \n", msg, m_buffer_id, m_bufsize, m_first, m_last );
}

template <typename T>
inline T* GTOBuffer<T>::getRawPtr() 
{
    assert(m_mapped == true);

    T* d_ptr;
    cudaGraphicsResourceGetMappedPointer((void **)&d_ptr, &m_bufsize, m_graphics_resource);

    m_first = 0 ; 
    m_last = m_bufsize/sizeof(T) ;

    return d_ptr ; 
}

template <typename T>
inline thrust::device_ptr<T> GTOBuffer<T>::getDevicePtr() 
{
    assert(m_mapped == true);
    T* raw_ptr = getRawPtr();
    thrust::device_ptr<T> dev_ptr = thrust::device_pointer_cast(raw_ptr);
    return dev_ptr ; 
}


template<> RTformat GTOBuffer<float3>::FORMAT = RT_FORMAT_FLOAT3  ;
template<> RTformat GTOBuffer<float4>::FORMAT = RT_FORMAT_FLOAT4  ;


template <typename T>
inline void GTOBuffer<T>::optixMap()
{
    mapResources();

    assert(m_mapped == true);

    T* d_ptr = getRawPtr();

    CUdeviceptr device_pointer = reinterpret_cast<CUdeviceptr>(d_ptr) ; 
    // https://devtalk.nvidia.com/default/topic/551556/failed-on-optix-buffer-setdevicepointer-/

    RTformat format = FORMAT ; 

    m_width = m_bufsize/sizeof(T) ;

    std::cout << "optixMap "  
              << " name " << m_name
              << " bufsize " << m_bufsize
              << " width " << m_width 
              << std::endl 
              ;

    m_buffer = m_context->createBufferForCUDA(m_type, format, m_width);

    m_buffer->setDevicePointer(m_device_number, device_pointer );

    m_context[m_name]->set( m_buffer );
}


template <typename T>
inline void GTOBuffer<T>::optixUnmap()
{
    unmapResources();
}

template <typename T>
inline void GTOBuffer<T>::optixLaunch(unsigned int entry)
{
    std::cout << "optixLaunch "  
              << " name " << m_name
              << " bufsize " << m_bufsize
              << " width " << m_width 
              << std::endl 
              ;

    m_context->launch(entry, 0);
    m_context->launch(entry, m_width);
}





template <typename T>
template <typename F>
inline void GTOBuffer<T>::thrust_transform(const F& f)
{
    mapResources();

    thrust::device_ptr<T> dev_ptr = getDevicePtr();
    thrust::counting_iterator<int> first(m_first);
    thrust::counting_iterator<int> last(m_last);
    thrust::transform(first, last, dev_ptr,  f );

    unmapResources();
}




