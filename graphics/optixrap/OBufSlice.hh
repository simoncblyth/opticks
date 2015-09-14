#pragma once

class OBuf ; 

struct OBufSlice 
{
   OBuf*        buf ;
   unsigned int stride ;
   unsigned int begin ;  
   unsigned int end ;  

   OBufSlice( OBuf* buf, unsigned int stride, unsigned int begin, unsigned int end ) 
     :
       buf(buf), 
       stride(stride),
       begin(begin), 
       end(end)
    {}
  
};


