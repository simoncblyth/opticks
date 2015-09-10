#pragma once

static void* getRawPtr(optix::Buffer & buffer, int deviceNumber)
{
    CUdeviceptr d;
    buffer->getDevicePointer(deviceNumber, &d);
    return (void*)d;
}

template<typename T>
static T* getDevicePtr(optix::Buffer & buffer, int deviceNumber)
{
    return (T*)getRawPtr(buffer, deviceNumber) ;
}


template<typename T>
static thrust::device_ptr<T> getThrustDevicePtr(optix::Buffer & buffer, int deviceNumber)
{
    return thrust::device_pointer_cast(getDevicePtr<T>(buffer, deviceNumber));
}


