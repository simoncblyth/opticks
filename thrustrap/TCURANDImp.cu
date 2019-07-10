
#include "NPY.hpp"
#include "TUtil.hh"
#include "TRngBuf.hh"
#include "TCURANDImp.hh"
#include "Opticks.hh"

template <typename T>
int TCURANDImp<T>::preinit() 
{
    OKI_PROFILE("_TCURANDImp::TCURANDImp"); 
    return 0 ; 
}
        
template <typename T>
TCURANDImp<T>::TCURANDImp( unsigned ni, unsigned nj, unsigned nk ) 
    :
    m_preinit(preinit()),
    m_ni(ni),
    m_nj(nj),
    m_nk(nk),
    m_elem( ni*nj*nk ),
    m_ox(NPY<T>::make( ni, nj, nk )),
    m_dox(m_elem),
    m_spec(make_bufspec<T>(m_dox)), 
    m_trb(new TRngBuf<T>( ni, nj*nk, m_spec ))
{
    init(); 
}


template <typename T>
void TCURANDImp<T>::init() 
{
    m_ox->zero(); 
    OKI_PROFILE("TCURANDImp::TCURANDImp"); 
}


template <typename T>
void TCURANDImp<T>::setIBase(unsigned ibase)
{
    m_trb->setIBase( ibase ); 
    generate(); 
}

template <typename T>
unsigned TCURANDImp<T>::getIBase() const 
{
    return m_trb->getIBase(); 
}





/**
TCURANDImp<T>::generate
-------------------------

GPU generation and download to host, updating m_ox array 

**/

template <typename T>
void TCURANDImp<T>::generate()
{
    m_trb->generate(); 
    bool verbose = true ; 
    m_trb->download( m_ox, verbose ) ; 
}

template <typename T>
NPY<T>* TCURANDImp<T>::getArray() const 
{
    return m_ox ; 
}


template class TCURANDImp<float>;
template class TCURANDImp<double>;


