
#include "NPY.hpp"
#include "TUtil.hh"
#include "TRngBuf.hh"
#include "TCURANDImp.hh"
#include "Opticks.hh"
#include "PLOG.hh"

template <typename T>
const plog::Severity TCURANDImp<T>::LEVEL = PLOG::EnvLevel("TCURANDImp", "DEBUG"); 
 

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
    m_predox(predox()),
    m_dox(m_elem),
    m_postdox(postdox()),
    m_spec(make_bufspec<T>(m_dox)), 
    m_trb(new TRngBuf<T>( ni, nj*nk, m_spec ))
{
    init(); 
}


template <typename T>
void TCURANDImp<T>::init() 
{
    LOG(LEVEL) << desc() ;   
    m_ox->zero(); 
    OKI_PROFILE("TCURANDImp::TCURANDImp"); 
}


template <typename T>
int TCURANDImp<T>::predox() 
{
    OKI_PROFILE("_dvec_dox"); 
    return 0 ; 
}

template <typename T>
int TCURANDImp<T>::postdox() 
{
    OKI_PROFILE("dvec_dox"); 
    return 0 ; 
}



template <typename T>
std::string TCURANDImp<T>::desc()  const 
{
    std::stringstream ss ; 
    ss << "TCURANDImp"
       << " ox " << m_ox->getShapeString() 
       << " elem " << m_elem
       ; 
    return ss.str(); 
}


template <typename T>
void TCURANDImp<T>::setIBase(unsigned ibase)
{
    LOG(LEVEL) << " ibase " << ibase ;   
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


