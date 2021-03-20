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

#include "THRAP_HEAD.hh"
#include "TUtil.hh"
#include <thrust/device_vector.h>
#include "THRAP_TAIL.hh"

#include "CBufSpec.hh"
#include "float4x4.h"


template <typename T>
CBufSpec make_bufspec(const thrust::device_vector<T>& d_vec )
{     
    const T* raw_ptr = thrust::raw_pointer_cast(d_vec.data());

    unsigned int size = d_vec.size() ;
    unsigned int nbytes =  size*sizeof(T) ;
      
    return CBufSpec( (void*)raw_ptr, size, nbytes );
} 



template THRAP_API CBufSpec make_bufspec<unsigned long long>(const thrust::device_vector<unsigned long long>& );
template THRAP_API CBufSpec make_bufspec<unsigned int>(const thrust::device_vector<unsigned int>& );
template THRAP_API CBufSpec make_bufspec<unsigned char>(const thrust::device_vector<unsigned char>& );
template THRAP_API CBufSpec make_bufspec<int>(const thrust::device_vector<int>& );
template THRAP_API CBufSpec make_bufspec<float>(const thrust::device_vector<float>& );
template THRAP_API CBufSpec make_bufspec<double>(const thrust::device_vector<double>& );

template THRAP_API CBufSpec make_bufspec<float4>(const thrust::device_vector<float4>& );

template THRAP_API CBufSpec make_bufspec<float6x4>(const thrust::device_vector<float6x4>& );
template THRAP_API CBufSpec make_bufspec<float4x4>(const thrust::device_vector<float4x4>& );
template THRAP_API CBufSpec make_bufspec<float2x4>(const thrust::device_vector<float2x4>& );



