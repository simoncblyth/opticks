/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

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



