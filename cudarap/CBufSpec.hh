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


