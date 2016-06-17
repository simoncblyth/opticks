#pragma once

#include "BRAP_API_EXPORT.hh"
#include "BRAP_FLAGS.hh"

class BRAP_API BDemo {
   public:
        BDemo(int i);
        void check();
   private:
        int m_i ; 

};


inline BDemo::BDemo(int i)
    :
    m_i(i)
{
}
