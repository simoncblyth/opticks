#pragma once

template <typename T> class NPY ; 
#include "NPY_API_EXPORT.hh"

/**
DummyPhotonsNPY
================





**/

class NPY_API DummyPhotonsNPY 
{
    public:
       static NPY<float>* Make(unsigned num_photons, unsigned hitmask, unsigned modulo=10 );  // formerly hitmask was default of:  0x1 << 5  (32)
    private:
       DummyPhotonsNPY(unsigned num_photons, unsigned hitmask, unsigned modulo);
       NPY<float>* getNPY();
       void        init();
    private:
       NPY<float>* m_data    ; 
       unsigned    m_hitmask ; 
       unsigned    m_modulo  ; 
};



