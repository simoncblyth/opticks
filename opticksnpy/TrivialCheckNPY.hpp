#pragma once

template <typename T> class NPY ; 
class G4StepNPY ; 

#include "NPY_API_EXPORT.hh"

class NPY_API TrivialCheckNPY {

       enum {
           IS_UINDEX,
           IS_UINDEX_SCALED,
           IS_UCONSTANT,
           IS_UCONSTANT_SCALED
       };

   public:  
       TrivialCheckNPY(NPY<float>* photons, NPY<float>* gensteps);
       int checkItemValue(unsigned istep, NPY<float>* npy, unsigned i0, unsigned i1, unsigned jj, unsigned kk, const char* label, int expect, int constant=0, int scale=0 );
   public:  
       void dump(const char* msg="TrivialCheckNPY::dump");
       int check(const char* msg);
  private:
        void checkGensteps(NPY<float>* gs);
        int checkPhotons(unsigned istep, NPY<float>* photons, unsigned i0, unsigned i1, unsigned gencode, unsigned numPhotons);
  private:
        NPY<float>*  m_photons ; 
        NPY<float>*  m_gensteps ; 
        G4StepNPY*   m_g4step ; 
};



