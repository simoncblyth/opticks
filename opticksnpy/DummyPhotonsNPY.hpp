#pragma once

template <typename T> class NPY ; 
#include "NPY_API_EXPORT.hh"
class NPY_API DummyPhotonsNPY 
{
   public:
      static NPY<float>* make(unsigned num_photons);
      DummyPhotonsNPY(unsigned num_photons);
      NPY<float>* getNPY();
   private:
      void makeStriped();
   private:
      NPY<float>* m_data ; 
};



