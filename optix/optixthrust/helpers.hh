#pragma once

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


