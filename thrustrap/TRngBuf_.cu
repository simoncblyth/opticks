#include <iostream>
#include <iomanip>

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h> 
#include <curand_kernel.h> 

#include "TRngBuf.hh"

#include "NPY.hpp"
#include "Opticks.hh"
#include "PLOG.hh"


/**
TRngBuf<T>::TRngBuf
----------------------

m_ibase
    absolute index of first photon 
m_ni
    number of photon slots
m_nj
    number of randoms to prepare for each photon

**/


template <typename T>
const plog::Severity TRngBuf<T>::LEVEL = PLOG::EnvLevel("TRngBuf", "DEBUG");     // static 


template <typename T>
int TRngBuf<T>::preinit() const 
{
    OKI_PROFILE("_TRngBuf::TRngBuf"); 
    return 0 ; 
}

template <typename T>
TRngBuf<T>::TRngBuf(unsigned ni, unsigned nj, CBufSpec spec, unsigned long long seed, unsigned long long offset )
    :
    m_preinit(preinit()),
    TBuf("trngbuf", spec, "\n" ),
    m_ibase(0),
    m_ni(ni),
    m_nj(nj),
    m_num_elem(ni*nj),
    m_id_offset(0),
    m_id_max(1000),
    m_seed(seed),
    m_offset(offset),
    m_dev((T*)getDevicePtr())
{
    init(); 
}

template <typename T>
void TRngBuf<T>::init() const 
{
    OKI_PROFILE("TRngBuf::TRngBuf"); 
}




template <typename T>
void TRngBuf<T>::setIBase(unsigned ibase)
{
    m_ibase = ibase ; 
}

template <typename T>
unsigned TRngBuf<T>::getIBase() const 
{
    return m_ibase ; 
}



/**
TRngBuf<T>::generate
---------------------

Loop over tranches of the photon slots 
calling generate for each tranche of up to m_id_max
photons.

**/

template <typename T>
void TRngBuf<T>::generate()
{

    LOG(LEVEL)
        << " ibase " << m_ibase
        << " ni " << m_ni
        << " id_max " << m_id_max
        ;

    unsigned seq = 0 ; 
    unsigned id_offset = 0 ; 
    while( id_offset < m_ni )
    {
        unsigned remaining = m_ni - id_offset ;    
        unsigned id_per_gen = m_id_max ; 
        if( id_per_gen > remaining ) id_per_gen = remaining ; 

        LOG(LEVEL)
              << " seq " << seq
              << " id_offset " << std::setw(10) << id_offset
              << " id_per_gen " << std::setw(10) << id_per_gen
              << " remaining " << std::setw(10) << remaining
              ;

        generate( id_offset, 0, id_per_gen );

        id_offset += id_per_gen ; 
        seq++ ; 
    }
}

/**
TRngBuf<T>::generate
---------------------

thrust::for_each "launch" covering uid from id_offset+id_0 to id_offset+id_1
notice how the functor manages to capture the m_id_offset constant 
together with other param like m_seed for use on the device.

Suspect the repeated curand_init for every id maybe a very 
inefficient way of doing this. Perhaps, but the Opticks approach 
of having a separate curand seqence number for every photon 
is worth the expense for the clarity and flexibility for doing 
things like masked running.

**/

template <typename T>
void TRngBuf<T>::generate(unsigned id_offset, unsigned id_0, unsigned id_1)
{
    m_id_offset = id_offset ;  
    thrust::for_each( 
          thrust::counting_iterator<unsigned>(id_0), 
          thrust::counting_iterator<unsigned>(id_1), 
           *this);
}


/**
TRngBuf<T>::operator()
---------------------------

m_nj 
    number of randoms to generate for each photon slot 

uoffset
    current set (ni,nj) index into an array of randoms, not 
    incorporating m_ibase 


**/


template <typename T>
__device__ 
void TRngBuf<T>::operator()(unsigned id) 
{ 
    unsigned uid = id + m_id_offset ; 

    curandState s; 
    curand_init(m_seed, m_ibase + uid , m_offset, &s); 

    for(unsigned j = 0; j < m_nj; ++j) 
    {
        unsigned uoffset = uid*m_nj+j ;
        if(uoffset < m_num_elem)
        {
            m_dev[uoffset] = curand_uniform(&s)  ; 
        }
    }
} 
 
template class TRngBuf<float>;
template class TRngBuf<double>;




