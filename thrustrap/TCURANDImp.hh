#pragma once

/**
TCURANDImp
==========

cuRAND GPU generation of random numbers using thrust and NPY 

**/

#include <thrust/device_vector.h>
#include "CBufSpec.hh"
template <typename T> class NPY ; 
template <typename T> class TRngBuf ; 

#include "THRAP_API_EXPORT.hh" 

template<typename T>
class THRAP_API TCURANDImp
{
        template<class U>  friend class TCURAND ; 
    public:
        TCURANDImp( unsigned ni, unsigned nj, unsigned nk ) ;
        NPY<T>*  getArray() const ; 
        void     setIBase(unsigned ibase ); 
        unsigned getIBase() const ; 
    private:
        int     preinit();  
        void    init();  
        void    generate();   // called by setIBase, updates contents of array
    private:
        int      m_preinit ;   
        unsigned m_ni ; 
        unsigned m_nj ; 
        unsigned m_nk ; 
        unsigned m_elem ; 

        NPY<T>*                   m_ox ;   
        thrust::device_vector<T>  m_dox ; 
        CBufSpec                  m_spec ;   
        TRngBuf<T>*               m_trb ;   


};



 
