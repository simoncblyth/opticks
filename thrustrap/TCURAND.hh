#pragma once

/**
TCURAND
==========

High level interface for dynamic GPU generation of curand 
random numbers, exposing functionality from TRngBuf  

**/

#include "THRAP_API_EXPORT.hh" 
#include "plog/Severity.h"

template <typename T> class NPY ; 
template <typename T> class TCURANDImp ; 

template<typename T>
class THRAP_API TCURAND
{
        static const plog::Severity LEVEL ; 
    public:
        TCURAND( unsigned ni, unsigned nj, unsigned nk);  
        void     setIBase(unsigned ibase); 
        unsigned getIBase() const ; 
        NPY<T>*  getArray() const ; 
    private:
        void     generate();       // called by setIBase, updates contents of array 
    private:
        TCURANDImp<T>*  m_imp ; 

}; 



