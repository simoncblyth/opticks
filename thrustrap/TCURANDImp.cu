
#include "NPY.hpp"
#include "TUtil.hh"
#include "TRngBuf.hh"
#include "TCURANDImp.hh"
        
template <typename T>
TCURANDImp<T>::TCURANDImp( unsigned ni, unsigned nj, unsigned nk ) 
    :
    m_ni(ni),
    m_nj(nj),
    m_nk(nk),
    m_elem( ni*nj*nk ),
    m_ox(NPY<T>::make( ni, nj, nk )),
    m_dox(m_elem),
    m_spec(make_bufspec<T>(m_dox)), 
    m_trb(new TRngBuf<T>( ni, nj*nk, m_spec ))
{
    m_ox->zero(); 
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


