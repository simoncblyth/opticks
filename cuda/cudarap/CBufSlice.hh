#pragma once
#include <cstdio>

#include "CUDARAP_API_EXPORT.hh"

struct CUDARAP_API CBufSlice 
{
   void*        dev_ptr ;
   unsigned int size ; 
   unsigned int num_bytes ;  
   unsigned int stride ; 
   unsigned int begin ; 
   unsigned int end ; 

   CBufSlice(void* dev_ptr, unsigned int size, unsigned int num_bytes, unsigned int stride, unsigned int begin, unsigned int end ) 
     :
       dev_ptr(dev_ptr),
       size(size),
       num_bytes(num_bytes),
       stride(stride),
       begin(begin),
       end(end)
   {
   }
   void Summary(const char* msg)
   {
       printf("%s : dev_ptr %p size %u num_bytes %u stride %u begin %u end %u \n", msg, dev_ptr, size, num_bytes, stride, begin, end ); 
   }


}; 


