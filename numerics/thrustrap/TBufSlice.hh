#pragma once

class TBuf ;

struct TBufSlice
{
   TBuf*        buf ;
   unsigned int stride ;
   unsigned int begin ;  
   unsigned int end ;  

   TBufSlice( TBuf* buf, unsigned int stride, unsigned int begin, unsigned int end )
     :
       buf(buf),
       stride(stride), 
       begin(begin),
       end(end)
    {}     
           
};         

