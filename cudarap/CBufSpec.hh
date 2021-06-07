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
#include <cstdio>
#include "CBufSlice.hh"

#include "CUDARAP_API_EXPORT.hh"

/**
CBufSpec 
=========

Simple struct wrapper for a device pointer with size(numItems) and num_bytes.
The *slice* method returns CBufSlice struct that represents a strided slice 
of this buffer.

**/

struct CUDARAP_API CBufSpec 
{
    void*              dev_ptr ; 
    unsigned long long size ; 
    unsigned long long num_bytes ; 
    bool               hexdump ; 

    CBufSpec(void* dev_ptr_, unsigned long long size_, unsigned long long num_bytes_, bool hexdump_=false) 
        :
        dev_ptr(dev_ptr_),
        size(size_),
        num_bytes(num_bytes_),
        hexdump(hexdump_)
    {
    }
    void Summary(const char* msg) const
    {
        printf("%s : dev_ptr %p size %llu num_bytes %llu hexdump %u \n", msg, dev_ptr, size, num_bytes, hexdump ); 
    }

    CBufSlice slice( unsigned long long stride, unsigned long long begin, unsigned long long end=0ull ) const 
    {
        if(end == 0ull) end = size ;   
        return CBufSlice(dev_ptr, size, num_bytes, stride, begin, end);
    }

}; 

// size expected to be num_bytes/sizeof(T)


