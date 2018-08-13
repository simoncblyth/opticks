THRAP : ThrustRap : 
=====================

TBuf
    CUDA single buffer operations : upload/download/reductions/slicing/dumping

TBufPair
    seedDestination using iexpand.h and strided_range.h     

TIsHit
    thrust::unary_function<float4x4,bool> looking for SURFACE_DETECT 
    in the GPU side photon buffer

TRngBuf
    GPU side generation of reproducible buffers for random numbers using cuRAND 

TSparse
    used for GPU sorting and indexing of photon history 64bit uints, finding 
    the chart topping photon histories  

TUtil
    CBufSpec make_bufspec from thrust::device_vector 



