#pragma once

#include "CBufSpec.hh"

// *TAry* is an exploration of convenient ways to work with Thrust and CUDA 
// the CBufSpec is central to this, holding the raw dev_ptr together 
// with size and num bytes

class TAry {
   public:
       TAry( CBufSpec src, CBufSpec dst ); 
   public:
       void check(); 
       void tcopy(); 
       void tfill(); 
       void transform(); 
       void memcpy();
       void memset();
       void kcall();
       void transform_old(); 
   private:
       void init();
   private:
       CBufSpec      m_src ; 
       CBufSpec      m_dst ; 

};


inline TAry::TAry(CBufSpec src, CBufSpec dst) :
    m_src(src),
    m_dst(dst)
{
}





