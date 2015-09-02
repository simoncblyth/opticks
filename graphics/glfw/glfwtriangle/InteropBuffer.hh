#include <cuda_gl_interop.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>

class InteropBuffer {
   public:
       InteropBuffer(unsigned int buffer_id, unsigned int flags, cudaStream_t stream=0);
       void Summary(const char* msg="InteropBuffer::Summary");
       template <typename T, typename F> void transform(const F& f);
   private:
       void mapResources();
       template <typename T> thrust::device_ptr<T> getDevicePtr();   
       template <typename T> T* getRawPtr();   
       void unmapResources();
   private:
       unsigned int m_buffer_id ;   
       unsigned int m_flags ;   
       cudaStream_t m_stream ; 
       size_t       m_bufsize ; 
       unsigned int m_first ; 
       unsigned int m_last ; 
       bool         m_mapped ; 
       struct cudaGraphicsResource* m_graphics_resource ;
};

inline InteropBuffer::InteropBuffer(unsigned int buffer_id, unsigned int flags, cudaStream_t stream) :
     m_buffer_id(buffer_id),
     m_flags(flags),
     m_stream(stream),
     m_bufsize(0),
     m_first(0),
     m_last(0),
     m_mapped(false)
{
    cudaGraphicsGLRegisterBuffer(&m_graphics_resource, m_buffer_id, m_flags);
}

inline void InteropBuffer::mapResources()
{
    assert(m_mapped == false);
    unsigned int count = 1 ; 
    cudaError_t error = cudaGraphicsMapResources(count, &m_graphics_resource, m_stream);
    m_mapped = true ; 
}
inline void InteropBuffer::unmapResources()
{
    assert(m_mapped == true);
    unsigned int count = 1 ; 
    cudaError_t error = cudaGraphicsUnmapResources(count, &m_graphics_resource, m_stream);
    m_mapped = false ; 
}
inline void InteropBuffer::Summary(const char* msg)
{
    printf("%s buffer_id %d bufsize %lu first %d last %d \n", msg, m_buffer_id, m_bufsize, m_first, m_last );
}

template <typename T>
inline T* InteropBuffer::getRawPtr() 
{
    assert(m_mapped == true);
    T* raw_ptr;
    cudaGraphicsResourceGetMappedPointer((void **)&raw_ptr, &m_bufsize, m_graphics_resource);

    m_first = 0 ; 
    m_last = m_bufsize/sizeof(T) ;
    return raw_ptr ; 
}

template <typename T>
inline thrust::device_ptr<T> InteropBuffer::getDevicePtr() 
{
    assert(m_mapped == true);
    T* raw_ptr = getRawPtr<T>();
    thrust::device_ptr<T> dev_ptr = thrust::device_pointer_cast(raw_ptr);
    return dev_ptr ; 
}

template <typename T, typename F>
inline void InteropBuffer::transform(const F& f)
{
    mapResources();

    thrust::device_ptr<T> dev_ptr = getDevicePtr<T>();
    thrust::counting_iterator<int> first(m_first);
    thrust::counting_iterator<int> last(m_last);
    thrust::transform(first, last, dev_ptr,  f );

    unmapResources();
}


