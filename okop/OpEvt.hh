#pragma once
#include "OKOP_API_EXPORT.hh"

/*
OpEvt
======

Light weight "API" event 

*/

template <typename T> class NPY  ; 

class OKOP_API OpEvt {
    public:
         OpEvt();
         void addGenstep( float* data, unsigned num_float );
         unsigned getNumGensteps() const ; 
         NPY<float>* getEmbeddedGensteps();

         void saveEmbeddedGensteps(const char* path) const ;
         void loadEmbeddedGensteps(const char* path);
    private:          
         NPY<float>* m_genstep ; 

};
 
