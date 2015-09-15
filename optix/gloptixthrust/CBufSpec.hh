#pragma once
#include "stdio.h"

struct CBufSpec 
{
   void*        dev_ptr ; 
   unsigned int size ; 
   unsigned int num_bytes ; 

   CBufSpec(void* dev_ptr, unsigned int size, unsigned int num_bytes) 
     :
       dev_ptr(dev_ptr),
       size(size),
       num_bytes(num_bytes)
   {
   }
   void Summary(const char* msg)
   {
       printf("%s : dev_ptr %p size %u num_bytes %u \n", msg, dev_ptr, size, num_bytes ); 
   }


}; 


