#include "TCURAND.hh"
#include "TCURANDImp.hh"


template <typename T>
TCURAND<T>::TCURAND(unsigned ni, unsigned nj, unsigned nk)
    :
    m_imp(new TCURANDImp<T>(ni, nj, nk))
{
}

template <typename T>
void TCURAND<T>::setIBase(unsigned ibase)
{
    m_imp->setIBase(ibase); 
}

template <typename T>
unsigned TCURAND<T>::getIBase() const 
{
    return m_imp->getIBase(); 
}


template <typename T>
void TCURAND<T>::generate()
{
    m_imp->generate(); 
}

template <typename T>
NPY<T>* TCURAND<T>::getArray() const 
{
    return m_imp->getArray() ; 
}


template class TCURAND<float>;
template class TCURAND<double>;


