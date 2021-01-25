/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

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
#include "OpticksPhoton.h"
// keep minimal : nvcc is tempramental 

#include "float4x4.h"
#include "strided_range.h"
#include "strided_repeated_range.h"

#include "NPY.hpp"



TBuf::TBuf(const char* name, CBufSpec spec, const char* delim) 
    :
    m_name(strdup(name)),
    m_spec(spec),
    m_delim(strdup(delim))
{
    //m_spec.Summary("TBuf::TBuf.m_spec"); 
}

CBufSlice TBuf::slice( unsigned long long stride, unsigned long long begin, unsigned long long end ) const 
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
unsigned long long TBuf::getNumBytes() const 
{
    return m_spec.num_bytes ; 
}
unsigned long long TBuf::getSize() const 
{
    return m_spec.size ; 
}

unsigned long long TBuf::getItemSize() const 
{
    return m_spec.size > 0 ? m_spec.num_bytes / m_spec.size : 0  ; 
}



template <typename T>
void TBuf::download(NPY<T>* npy, bool verbose) const 
{
    unsigned numItems_npy = npy->getNumItems();
    unsigned long long numItems_tbuf = getSize(); 

    bool create_empty_npy = true ; 


    if(numItems_tbuf == 0)
    {
        std::cout 
            << "TBuf::download SKIP "
            << " numItems_tbuf " << numItems_tbuf
            << std::endl ; 

        m_spec.Summary("CBufSpec.Summary (empty tbuf?)"); 
        if(create_empty_npy)
        {
            std::cout << "create_empty_npy" << std::endl ; 
            npy->zero();
        }
        return ; 
    }

    if(numItems_npy == 0)
    {    
        unsigned long long itemSize_tbuf = getItemSize();
        unsigned long long itemSize_npy = sizeof(T) ;
        assert(itemSize_tbuf % itemSize_npy == 0);

        unsigned long long itemFactor = itemSize_tbuf / itemSize_npy ; 
        assert(itemFactor % 4 == 0) ;

        unsigned long long numQuad = itemFactor/4 ; 

        if(verbose)
        std::cout 
            << "TBuf::download resizing empty npy"
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
        std::cout 
            << "TBuf::download FATAL numBytes MISMATCH "
            << " numBytes_npy " << numBytes_npy 
            << " numBytes_tbuf " << numBytes_tbuf
            << std::endl 
            ;  
    assert(numBytes_match);

    void* src = getDevicePtr();
    npy->zero();

    void* dst = npy->getBytes(); 

    cudaMemcpy( dst, src, numBytes_tbuf, cudaMemcpyDeviceToHost );
}






/**
TBuf::downloadSelection4x4
-----------------------------

This hides the float4x4 type down in here in the _.cu 
where nvcc makes it available by default, such that the  
user doesnt need access to the type.

Canonically invoked by::

    OEvent::downloadHitsCompute
    OEvent::downloadHitsInterop

Using hitmask that is hardcoded into OEvent ctor::

     65 OEvent::OEvent(Opticks* ok, OContext* ocontext)
     66    :
     67    m_log(new SLog("OEvent::OEvent", "", LEVEL)),
     68    m_ok(ok),
     69    //m_hitmask(SURFACE_DETECT),
     70    m_hitmask(TORCH | BULK_SCATTER | BOUNDARY_TRANSMIT | SURFACE_ABSORB),
     71    m_compute(ok->isCompute()),
     72    m_dbghit(m_ok->isDbgHit()),            // --dbghi


**/

unsigned TBuf::downloadSelection4x4(const char* name, NPY<float>* npy, unsigned hitmask, bool verbose) const 
{
    return downloadSelection<float4x4>(name, npy, hitmask, verbose);
}

unsigned TBuf::downloadSelection2x4(const char* name, NPY<float>* npy, unsigned hitmask, bool verbose) const 
{
    return downloadSelection<float2x4>(name, npy, hitmask, verbose);
}



/**
TBuf::downloadSelection
------------------------

Initially tried to 
    TBuf* TBuf::make_selection

but that means the d_selected thrust::device_vector 
goes out of scope prior to being able to download
its content to host.

Avoid complication of hanging onto thrust::device_vector
by combining the making of the selection TBuf and 
the download 

**/

template <typename T>
unsigned TBuf::downloadSelection(const char* name, NPY<float>* selection, unsigned hitmask, bool verbose) const 
{
    thrust::device_ptr<T> ptr = thrust::device_pointer_cast((T*)getDevicePtr()) ;

    unsigned long long numItems = getSize();

    TIsHit<T> is_hit(hitmask); 

    unsigned numSel = thrust::count_if(ptr, ptr+numItems, is_hit );

    if(verbose)
    std::cout 
        << "TBuf::downloadSelection"
        << " name : " << name 
        << " numItems :" << numItems 
        << " numSel :" << numSel 
        << " sizeof(T) : " << sizeof(T)
        << " hitmask 0x" << std::hex << hitmask << std::dec
        << std::endl 
        ; 


    // device buffer is deallocated when d_selected goes out of scope 
    // so do the download to host within this scope

    thrust::device_vector<T> d_selected(numSel) ;    

    thrust::copy_if(ptr, ptr+numItems, d_selected.begin(), is_hit );

    CBufSpec cselected = make_bufspec<T>(d_selected); 

    TBuf tsel(name, cselected );
 
    if(verbose)
    tsel.dump<T>("TBuf::downloadSelection tsel dump<T> 0:numSel", 1, 0, numSel );

    if(numSel > 0)
    {
        bool itemsize_match = sizeof(T) == tsel.getItemSize() ;
        if(!itemsize_match)
            std::cerr 
                << "TBuf::downloadSelection   FATAL"
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

    unsigned long long numval = getSize();

    thrust::fill(p, p+numval , value);
}



void TBuf::dump6x4(const char* msg, unsigned long long stride, unsigned long long begin, unsigned long long end ) const 
{
     dump<float6x4>(msg, stride, begin, end);
}
void TBuf::dump4x4(const char* msg, unsigned long long stride, unsigned long long begin, unsigned long long end ) const 
{
     dump<float4x4>(msg, stride, begin, end);
}
void TBuf::dump2x4(const char* msg, unsigned long long stride, unsigned long long begin, unsigned long long end ) const 
{
     dump<float2x4>(msg, stride, begin, end);
}


/**
TBuf::dump
-----------

Streams are in float4x4.h 

**/

template <typename T>
void TBuf::dump(const char* msg, unsigned long long stride, unsigned long long begin, unsigned long long end ) const 
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
void TBuf::dumpint(const char* msg, unsigned long long stride, unsigned long long begin, unsigned long long end) const 
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
T TBuf::reduce(unsigned long long stride, unsigned long long begin, unsigned long long end ) const 
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
void TBuf::repeat_to( TBuf* other, unsigned long long stride, unsigned long long begin, unsigned long long end, unsigned long long repeats ) const 
{
    thrust::device_ptr<T> src = thrust::device_pointer_cast((T*)getDevicePtr()) ;
    thrust::device_ptr<T> tgt = thrust::device_pointer_cast((T*)other->getDevicePtr()) ;

    typedef typename thrust::device_vector<T>::iterator Iterator;

    strided_repeated_range<Iterator> si( src + begin, src + end, stride, repeats);

    thrust::copy( si.begin(), si.end(),  tgt );    
}






template void TBuf::dump<float6x4>(const char*, unsigned long long, unsigned long long, unsigned long long) const ;
template void TBuf::dump<float4x4>(const char*, unsigned long long, unsigned long long, unsigned long long) const ;
template void TBuf::dump<float2x4>(const char*, unsigned long long, unsigned long long, unsigned long long) const ;

template void TBuf::dump<float4>(const char*, unsigned long long, unsigned long long, unsigned long long) const ;
template void TBuf::dump<double>(const char*, unsigned long long, unsigned long long, unsigned long long) const ;
template void TBuf::dump<float>(const char*, unsigned long long, unsigned long long, unsigned long long) const ;
template void TBuf::dump<int>(const char*, unsigned long long, unsigned long long, unsigned long long) const ;
template void TBuf::dump<unsigned>(const char*, unsigned long long, unsigned long long, unsigned long long) const ;
template void TBuf::dump<unsigned long long>(const char*, unsigned long long, unsigned long long, unsigned long long) const ;

template void TBuf::dumpint<unsigned char>(const char*, unsigned long long, unsigned long long, unsigned long long) const ;

template void TBuf::repeat_to<unsigned char>(TBuf*, unsigned long long, unsigned long long, unsigned long long, unsigned long long) const ;
template unsigned int TBuf::reduce<unsigned int>(unsigned long long, unsigned long long, unsigned long long) const ;

template void TBuf::download<double>(NPY<double>*, bool) const ;
template void TBuf::download<float>(NPY<float>*, bool) const ;
template void TBuf::download<unsigned>(NPY<unsigned>*, bool) const ;
template void TBuf::download<unsigned char>(NPY<unsigned char>*, bool) const ;
template void TBuf::download<unsigned long long>(NPY<unsigned long long>*, bool) const ;

template void TBuf::upload<double>(NPY<double>*) const ;
template void TBuf::upload<float>(NPY<float>*) const ;
template void TBuf::upload<unsigned>(NPY<unsigned>*) const ;
template void TBuf::upload<unsigned char>(NPY<unsigned char>*) const ;
template void TBuf::upload<unsigned long long>(NPY<unsigned long long>*) const ;

template void TBuf::fill<unsigned>(unsigned value) const ;
template void TBuf::fill<unsigned char>(unsigned char value) const ;

template unsigned TBuf::downloadSelection<float6x4>(const char*, NPY<float>*, unsigned, bool ) const ;
template unsigned TBuf::downloadSelection<float4x4>(const char*, NPY<float>*, unsigned, bool ) const ;
template unsigned TBuf::downloadSelection<float2x4>(const char*, NPY<float>*, unsigned, bool ) const ;

