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

struct CUDARAP_API CBufSpec 
{
   void*        dev_ptr ; 
   unsigned int size ; 
   unsigned int num_bytes ; 
   bool         hexdump ; 

   CBufSpec(void* dev_ptr_, unsigned int size_, unsigned int num_bytes_, bool hexdump_=false) 
     :
       dev_ptr(dev_ptr_),
       size(size_),
       num_bytes(num_bytes_),
       hexdump(hexdump_)
   {
   }
   void Summary(const char* msg) const
   {
       printf("%s : dev_ptr %p size %u num_bytes %u hexdump %u \n", msg, dev_ptr, size, num_bytes, hexdump ); 
   }

   CBufSlice slice( unsigned int stride, unsigned int begin, unsigned int end=0u ) const 
   {
        if(end == 0u) end = size ;   
        return CBufSlice(dev_ptr, size, num_bytes, stride, begin, end);
   }



}; 

// size expected to be num_bytes/sizeof(T)


