#pragma once

template <typename T> class NPY ; 
#include "NPY_API_EXPORT.hh"
class NPY_API DummyPhotonsNPY 
{
   public:
      static NPY<float>* make(unsigned num_photons, unsigned hitmask=0x1 << 5);
      DummyPhotonsNPY(unsigned num_photons, unsigned hitmask);
      NPY<float>* getNPY();
   private:
      void makeStriped();
   private:
      NPY<float>* m_data ; 
      unsigned    m_hitmask ; 
};



