#pragma once

#include "BufSpec.hh"

class TAry {
   public:
       TAry( BufSpec src, BufSpec dst ); 
   public:
       void check(); 
       void tcopy(); 
       void transform(); 
       void memcpy();
       void memset();
       void kcall();
       void transform_old(); 
     //  void copyToHost( void* host );
   private:
       void init();
   private:
       BufSpec      m_src ; 
       BufSpec      m_dst ; 

};


inline TAry::TAry(BufSpec src, BufSpec dst) :
    m_src(src),
    m_dst(dst)
{
}





