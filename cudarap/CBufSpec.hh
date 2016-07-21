#pragma once
#include <cstdio>

#include "CUDARAP_API_EXPORT.hh"

struct CUDARAP_API CBufSpec 
{
   void*        dev_ptr ; 
   unsigned int size ; 
   unsigned int num_bytes ; 

   CBufSpec(void* dev_ptr_, unsigned int size_, unsigned int num_bytes_) 
     :
       dev_ptr(dev_ptr_),
       size(size_),
       num_bytes(num_bytes_)
   {
   }
   void Summary(const char* msg) const
   {
       printf("%s : dev_ptr %p size %u num_bytes %u \n", msg, dev_ptr, size, num_bytes ); 
   }


}; 

// size expected to be num_bytes/sizeof(T)


