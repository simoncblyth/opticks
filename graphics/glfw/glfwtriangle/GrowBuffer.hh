//
// *GrowBuffer* allows nvcc/CUDA specifics (ie *grow* functor __device__ __host__ modifiers) 
// to be hidden in the implementation *GrowBuffer.cu* in order to keep this header 
// free of such specifics to allow linkage with clang/gcc obj
//
// The name *GrowBuffer* aint so great, the values of the vertex coordinates
// are growing and shrinking, not the the buffer itself. 
//

#include "InteropBuffer.hh"

class GrowBuffer : public InteropBuffer {
   public:
       GrowBuffer( unsigned int buffer_id, unsigned int flags, cudaStream_t stream=0);
       void trans(unsigned int n);
};

inline GrowBuffer::GrowBuffer( unsigned int buffer_id, unsigned int flags, cudaStream_t stream) 
    : InteropBuffer(buffer_id, flags, stream)
{
}


