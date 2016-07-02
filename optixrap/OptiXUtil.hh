#pragma once


#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#endif

#include <thrust/device_vector.h>

#ifdef __clang__
#pragma clang diagnostic pop
#endif


#include "OXPPNS.hh"

#include "OXRAP_API_EXPORT.hh"

class OXRAP_API OptiXUtil {
    public:

    static unsigned int getBufferSize1D(optix::Buffer& buffer)
    {
        RTsize size ; 
        buffer->getSize(size);
        return size ; 
    }        

    template<typename T>
    static T* getDevicePtr(optix::Buffer & buffer, int deviceNumber)
    {
        CUdeviceptr d;
        buffer->getDevicePointer(deviceNumber, &d);
        return (T*)d;
    }

    template<typename T>
    static thrust::device_ptr<T> getThrustDevicePtr(optix::Buffer & buffer, int deviceNumber)
    {
        return thrust::device_pointer_cast(getDevicePtr<T>(buffer, deviceNumber));
    }


};



