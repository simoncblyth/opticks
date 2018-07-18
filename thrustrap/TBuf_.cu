#include <cstdio>
#include <iterator>
#include <iomanip>
#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/count.h>

#include "TBuf.hh"
#include "TUtil.hh"
#include "TIsHit.hh"

#include "float4x4.h"
#include "strided_range.h"
#include "strided_repeated_range.h"

#include "NPY.hpp"



TBuf::TBuf(const char* name, CBufSpec spec, const char* delim) :
        m_name(strdup(name)),
        m_spec(spec),
        m_delim(strdup(delim))
{
    m_spec.Summary("TBuf::TBuf.m_spec"); 
}

CBufSlice TBuf::slice( unsigned int stride, unsigned int begin, unsigned int end ) const 
{
    if(end == 0u) end = m_spec.size ;  
    return CBufSlice(m_spec.dev_ptr, m_spec.size, m_spec.num_bytes, stride, begin, end);
}

void TBuf::Summary(const char* msg) const 
{
    printf("%s %s \n", msg, m_name );
}

void* TBuf::getDevicePtr() const 
{
    return m_spec.dev_ptr ; 
}
unsigned int TBuf::getNumBytes() const 
{
    return m_spec.num_bytes ; 
}
unsigned int TBuf::getSize() const 
{
    return m_spec.size ; 
}

unsigned int TBuf::getItemSize() const 
{
    return m_spec.size > 0 ? m_spec.num_bytes / m_spec.size : 0  ; 
}






template <typename T>
void TBuf::download(NPY<T>* npy, bool verbose) const 
{
    unsigned numItems_npy = npy->getNumItems();
    unsigned numItems_tbuf = getSize(); 

    if(numItems_tbuf == 0)
    {
        std::cout << "TBuf::download SKIP "
                  << " numItems_tbuf " << numItems_tbuf
                  << std::endl ; 

        m_spec.Summary("CBufSpec.Summary (empty tbuf?)"); 
        return ; 
    }

    if(numItems_npy == 0)
    {    
        unsigned itemSize_tbuf = getItemSize();
        unsigned itemSize_npy = sizeof(T) ;
        assert(itemSize_tbuf % itemSize_npy == 0);

        unsigned itemFactor = itemSize_tbuf / itemSize_npy ; 
        assert(itemFactor % 4 == 0) ;

        unsigned numQuad = itemFactor/4 ; 

        if(verbose)
        std::cout << "TBuf::download resizing empty npy"
                  << " itemSize_tbuf (eg float4x4 => 4*4*4 = 64) " << itemSize_tbuf
                  << " itemSize_npy  (eg float => 4 ) " << itemSize_npy
                  << " itemFactor (eg float4x4/float => 16 ) " << itemFactor  
                  << " numQuad " << numQuad 
                  << " numItems_tbuf " << numItems_tbuf
                  << std::endl 
                  ;

        assert(npy->hasItemShape(numQuad,4));
        npy->setNumItems(numItems_tbuf);
    } 

    unsigned int numBytes_npy = npy->getNumBytes(0) ;
    unsigned int numBytes_tbuf = getNumBytes();
    bool numBytes_match = numBytes_npy == numBytes_tbuf ;

    if(!numBytes_match)
        std::cout << "TBuf::download FATAL numBytes MISMATCH "
                  << " numBytes_npy " << numBytes_npy 
                  << " numBytes_tbuf " << numBytes_tbuf
                  << std::endl 
                  ;  
    assert(numBytes_match);

    void* src = getDevicePtr();
    void* dst = npy->zero();
    cudaMemcpy( dst, src, numBytes_tbuf, cudaMemcpyDeviceToHost );
}





// TODO: generalize with selector template type such as TIsHit 
//
//
//  Initially tried to 
//      TBuf* TBuf::make_selection
//
//  but that means the d_selected thrust::device_vector 
//  goes out of scope prior to being able to download
//  its content to host.
//
//  Avoid complication of hanging onto thrust::device_vector
//  by combining the making of the selection TBuf and 
//  the download 
//



unsigned TBuf::downloadSelection4x4(const char* name, NPY<float>* npy, bool verbose) const 
{
    return downloadSelection<float4x4>(name, npy, verbose);
}

template <typename T>
unsigned TBuf::downloadSelection(const char* name, NPY<float>* selection, bool verbose) const 
{
    thrust::device_ptr<T> ptr = thrust::device_pointer_cast((T*)getDevicePtr()) ;

    unsigned numItems = getSize();

    TIsHit selector ;

    unsigned numSel = thrust::count_if(ptr, ptr+numItems, selector );

    if(verbose)
    std::cout << "TBuf::downloadSelection"
              << " name : " << name 
              << " numItems :" << numItems 
              << " numSel :" << numSel 
              << " sizeof(T) : " << sizeof(T)
              << std::endl 
              ; 


    // device buffer is deallocated when d_selected goes out of scope 
    // so do the download to host within this scope

    thrust::device_vector<T> d_selected(numSel) ;    

    thrust::copy_if(ptr, ptr+numItems, d_selected.begin(), selector );

    CBufSpec cselected = make_bufspec<T>(d_selected); 

    TBuf tsel(name, cselected );
 
    if(verbose)
    tsel.dump<T>("TBuf::downloadSelection tsel dump<T> 0:numSel", 1, 0, numSel );

    if(numSel > 0)
    {
        bool itemsize_match = sizeof(T) == tsel.getItemSize() ;
        if(!itemsize_match)
            std::cerr << "TBuf::downloadSelection   FATAL"
                      << " sizeof(T) " << sizeof(T)
                      << " tsel ItemSize " << tsel.getItemSize()
                      << std::endl 
                 ; 
        assert(itemsize_match);
    }

    assert(tsel.getSize() == numSel );

    tsel.download(selection, verbose);

    return numSel ; 
}





template <typename T>
void TBuf::upload(NPY<T>* npy) const 
{
    unsigned int numBytes = npy->getNumBytes(0) ;
    unsigned int numBytes2 = getNumBytes();

    if(numBytes != numBytes2)
        std::cout << "TBuf::upload FATAL numBytes mismatch "
                  << " numBytes " << numBytes 
                  << " numBytes2 " << numBytes2
                  << std::endl ;  

    assert(numBytes == numBytes2);
    void* src = npy->getBytes() ;
    void* dst = getDevicePtr();
    cudaMemcpy( dst, src, numBytes, cudaMemcpyHostToDevice );
}





void TBuf::zero()
{
    cudaMemset( getDevicePtr(), 0, getNumBytes());
}


template <typename T>
void TBuf::fill(T value) const 
{
    thrust::device_ptr<T> p = thrust::device_pointer_cast((T*)getDevicePtr()) ;

    unsigned numval = getSize();

    thrust::fill(p, p+numval , value);
}



void TBuf::dump4x4(const char* msg, unsigned int stride, unsigned int begin, unsigned int end ) const 
{
     dump<float4x4>(msg, stride, begin, end);
}



template <typename T>
void TBuf::dump(const char* msg, unsigned int stride, unsigned int begin, unsigned int end ) const 
{
    Summary(msg);

    thrust::device_ptr<T> p = thrust::device_pointer_cast((T*)getDevicePtr()) ;

    std::ostream& out = std::cout ; 
    if(m_spec.hexdump) out << std::hex ; 

    if( stride == 0 )
    {
        thrust::copy( p + begin, p + end, std::ostream_iterator<T>(out, m_delim) );
    }
    else
    {
        typedef typename thrust::device_vector<T>::iterator Iterator;
        strided_range<Iterator> sri( p + begin, p + end, stride );
        thrust::copy( sri.begin(), sri.end(), std::ostream_iterator<T>(out, m_delim) );
    }
    out << std::endl ; 

}


template <typename T>
void TBuf::dumpint(const char* msg, unsigned int stride, unsigned int begin, unsigned int end) const 
{

    // dumpint necessitated in addition to dump as streaming unsigned char gives characters not integers

    Summary(msg);

    thrust::device_ptr<T> p = thrust::device_pointer_cast((T*)getDevicePtr()) ;

    thrust::host_vector<T> h ;

    if( stride == 0 )
    {
        h.resize(thrust::distance(p+begin, p+end));
        thrust::copy( p + begin, p + end, h.begin());
    }
    else
    {
        typedef typename thrust::device_vector<T>::iterator Iterator;
        strided_range<Iterator> sri( p + begin, p + end, stride );
        h.resize(thrust::distance(sri.begin(), sri.end()));
        thrust::copy( sri.begin(), sri.end(), h.begin() );
    }

    for(unsigned int i=0 ; i < h.size() ; i++)
    {
        std::cout
                 << std::setw(7) << i
                 << std::setw(7) << int(h[i])
                 << std::endl ;
    }
}


template <typename T>
T TBuf::reduce(unsigned int stride, unsigned int begin, unsigned int end ) const 
{
    thrust::device_ptr<T> p = thrust::device_pointer_cast((T*)getDevicePtr()) ;

    T result ;
    if( stride == 0 )
    {
        result = thrust::reduce( p + begin, p + end );
    }
    else
    {
        typedef typename thrust::device_vector<T>::iterator Iterator;
        strided_range<Iterator> sri( p + begin, p + end, stride );
        result = thrust::reduce( sri.begin(), sri.end() );
    }
    return result ;
}



template <typename T>
void TBuf::repeat_to( TBuf* other, unsigned int stride, unsigned int begin, unsigned int end, unsigned int repeats ) const 
{
    thrust::device_ptr<T> src = thrust::device_pointer_cast((T*)getDevicePtr()) ;
    thrust::device_ptr<T> tgt = thrust::device_pointer_cast((T*)other->getDevicePtr()) ;

    typedef typename thrust::device_vector<T>::iterator Iterator;

    strided_repeated_range<Iterator> si( src + begin, src + end, stride, repeats);

    thrust::copy( si.begin(), si.end(),  tgt );    
}






template void TBuf::dump<float4x4>(const char*, unsigned int, unsigned int, unsigned int) const ;
template void TBuf::dump<float4>(const char*, unsigned int, unsigned int, unsigned int) const ;
template void TBuf::dump<double>(const char*, unsigned int, unsigned int, unsigned int) const ;
template void TBuf::dump<float>(const char*, unsigned int, unsigned int, unsigned int) const ;
template void TBuf::dump<int>(const char*, unsigned int, unsigned int, unsigned int) const ;
template void TBuf::dump<unsigned int>(const char*, unsigned int, unsigned int, unsigned int) const ;
template void TBuf::dump<unsigned long long>(const char*, unsigned int, unsigned int, unsigned int) const ;

template void TBuf::dumpint<unsigned char>(const char*, unsigned int, unsigned int, unsigned int) const ;

template void TBuf::repeat_to<unsigned char>(TBuf*, unsigned int, unsigned int, unsigned int, unsigned int) const ;
template unsigned int TBuf::reduce<unsigned int>(unsigned int, unsigned int, unsigned int) const ;

template void TBuf::download<double>(NPY<double>*, bool) const ;
template void TBuf::download<float>(NPY<float>*, bool) const ;
template void TBuf::download<unsigned char>(NPY<unsigned char>*, bool) const ;

template void TBuf::upload<double>(NPY<double>*) const ;
template void TBuf::upload<float>(NPY<float>*) const ;
template void TBuf::upload<unsigned>(NPY<unsigned>*) const ;
template void TBuf::upload<unsigned char>(NPY<unsigned char>*) const ;

template void TBuf::fill<unsigned>(unsigned value) const ;
template void TBuf::fill<unsigned char>(unsigned char value) const ;


template unsigned TBuf::downloadSelection<float4x4>(const char*, NPY<float>*, bool ) const ;


