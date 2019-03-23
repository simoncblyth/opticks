#pragma once

/**
OptiXUtil
============

Device pointer utilities used for Thrust/OpenGL/OptiX/CUDA interop

DevNotes
----------

Usually thrust/device_vector only seen in .cu and compiled by nvcc ?

* not used ?

**/
#include "THRAP_PUSH.hh"
#include <cuda.h>
#include <thrust/device_vector.h>
#include "THRAP_POP.hh"


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
        return (T*)buffer->getDevicePointer(deviceNumber);;
    }

    template<typename T>
    static thrust::device_ptr<T> getThrustDevicePtr(optix::Buffer & buffer, int deviceNumber)
    {
        return thrust::device_pointer_cast(getDevicePtr<T>(buffer, deviceNumber));
    }


};



