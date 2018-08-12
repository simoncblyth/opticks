#pragma once
#include "OKOP_API_EXPORT.hh"

/*
OpEvt
======

Light weight "API" event 

Canonical m_opevt instance is resident of OpMgr and 
is instanciated when OpMgr::addGenstep is called.

HMM : CFG4.CCollector does all that this does and more (but not too much more)

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

         void resetGensteps();
    private:          
         NPY<float>* m_genstep ; 

};
 
