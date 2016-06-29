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

   CBufSlice(void* dev_ptr_, unsigned int size_, unsigned int num_bytes_, unsigned int stride_, unsigned int begin_, unsigned int end_ ) 
     :
       dev_ptr(dev_ptr_),
       size(size_),
       num_bytes(num_bytes_),
       stride(stride_),
       begin(begin_),
       end(end_)
   {
   }
   void Summary(const char* msg)
   {
       printf("%s : dev_ptr %p size %u num_bytes %u stride %u begin %u end %u \n", msg, dev_ptr, size, num_bytes, stride, begin, end ); 
   }


}; 


