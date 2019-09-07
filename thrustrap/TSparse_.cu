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
#include <sstream>
#include <iostream>
#include <iomanip>

// rejig? BHex for boost avoidance ?
#include "BHex.hh"
#include "Index.hpp"

#include "TSparse.hh"
#include "TBuf.hh"


#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/constant_iterator.h>

#include "strided_range.h"

// avoid boost as this is compiled by nvcc

//
// *dev_tsparse_lookup* 
//        is a global symbol pointing at device constant memory 
//        **CAUTION : UNAVOIDABLE NAME COLLISION DANGER**
//    
//        http://stackoverflow.com/questions/7961792/device-constant-const/7963395#7963395
//

__constant__ unsigned long long dev_tsparse_lookup[TSPARSE_LOOKUP_N]; 


template <typename T>
TSparse<T>::TSparse(const char* label, CBufSlice source, bool hexkey ) :
        m_label(strdup(label)),
        m_reldir(NULL),
        m_source(source),
        m_num_unique(0u),
        m_index_h(NULL),
        m_hexkey(hexkey)
{
    init();
}

template <typename T>
Index* TSparse<T>::getIndex()
{
    return m_index_h ;
}


template <typename T>
void TSparse<T>::init()
{
    m_index_h = new Index(m_label, m_reldir);
}


template <typename T>
void TSparse<T>::make_lookup()
{
    count_unique();
    update_lookup();
    populate_index(m_index_h);
}


template <typename T>
void TSparse<T>::count_unique()
{
    typedef typename thrust::device_vector<T>::iterator Iterator;

    thrust::device_ptr<T> psrc = thrust::device_pointer_cast((T*)m_source.dev_ptr) ; 

    strided_range<Iterator> src( psrc + m_source.begin, psrc + m_source.end, m_source.stride ); 

    thrust::device_vector<T> data(src.begin(), src.end());  // copy to avoid sorting original

    thrust::sort(data.begin(), data.end());

    // inner_product of sorted data with shifted by one self finds "edges" between values 
    m_num_unique = thrust::inner_product(
                                  data.begin(),data.end() - 1,   // first1, last1  
                                             data.begin() + 1,   // first2
                                                       int(1),   // output type init 
                                          thrust::plus<int>(),   // reduction operator
                                    thrust::not_equal_to<T>()    // pair-by-pair operator, returning 1 at edges 
                                      ); 


#ifdef DEBUG
    printf("TSparse<T>::count_unique m_num_unique %d \n", m_num_unique) ; 
#endif

    m_values.resize(m_num_unique);
    m_counts.resize(m_num_unique);
  
    // find all unique key values with their counts
    thrust::reduce_by_key(
                                data.begin(),    // keys_first
                                  data.end(),    // keys_last 
           thrust::constant_iterator<int>(1),    // values_first 
                            m_values.begin(),    // keys_output 
                            m_counts.begin()     // values_output
                         ); 
  
   // *reduce_by_key* is a generalization of reduce to key-value pairs. For each group
   // of consecutive keys in the range [keys_first, keys_last) that are equal,
   // reduce_by_key copies the first element of the group to the keys_output. 
   // The corresponding values in the range are reduced using the plus and the result
   // copied to values_output.
   //
   // As *data* is sorted this means get each unique key once in m_values together
   // the occurrent count in m_counts    

    thrust::sort_by_key( 
                          m_counts.begin(),     // keys_first
                            m_counts.end(),     // keys_last 
                          m_values.begin(),     // values_first
                     thrust::greater<int>()
                       );

    // sorts keys and values into descending key order, as the counts are in 
    // the key slots this sorts in descending count order

}


template <typename T>
void TSparse<T>::update_lookup()
{
    // partial pullback, only need small number (~32) of most popular ones on host 

    unsigned int n = TSPARSE_LOOKUP_N ; 

    m_values_h.resize(n, 0);  // 0 as empty 
    m_counts_h.resize(n, 0);

    unsigned int ncopy = std::min( m_num_unique, n ); 

    thrust::copy_n( m_values.begin(), ncopy , m_values_h.begin() );
    thrust::copy_n( m_counts.begin(), ncopy , m_counts_h.begin() );

    T* data = m_values_h.data();

#ifdef DEBUG
    printf("TSparse<T>::update_lookup<T>\n");
    for(unsigned int i=0 ; i < TSPARSE_LOOKUP_N ; i++) 
          std::cout << std::dec << std::setw(4) << i 
                    << " " << std::hex << std::setw(16) << *(data + i) << std::dec 
                    << std::endl ;
#endif
 
    cudaMemcpyToSymbol(dev_tsparse_lookup,data,TSPARSE_LOOKUP_N*sizeof(T));
}

/*
Nov 2015: 
   Fixed Issue with *update_lookup* occuring when the number of uniques 
   is less than the defined number of lookup slots, the indexing got messed up. 
   Picking a slot in GUI yielded no selection or the wrong selection.  
   The cause was zeros repesenting empty in the lookup array taking valid 
   indices. Fix was to switch to descending count sort order on device
   to avoid empties messing with the indices. 
*/

template <typename T>
std::string TSparse<T>::dump_(const char* msg) const 
{
    std::stringstream ss ; 
    ss << msg << " : num_unique " << m_num_unique << std::endl ; 
    //printf("%s : num_unique %u \n", msg, m_num_unique );
    for(unsigned int  i=0 ; i < m_values_h.size() ; i++)
    {
        ss << "[" << std::setw(2) << i << "] "  ;
        if(m_hexkey) ss << std::hex ; 
        ss << std::setw(16) << m_values_h[i] ;
        if(m_hexkey) ss << std::dec ; 
        ss << std::setw(10) << m_counts_h[i] ;
        ss << std::endl ;  
    } 
    return ss.str();
}



template <typename T>
void TSparse<T>::dump(const char* msg) const 
{
    std::cout << dump_(msg) ; 
}



template <typename T, typename S>
struct apply_lookup_functor : public thrust::unary_function<T,S>
{
    S m_offset ;
    S m_missing ;

    apply_lookup_functor(S offset, S missing)
        :
        m_offset(offset),    
        m_missing(missing)
        {
        }    


    // host function cannot access __constant__ memory hence device only
    __device__   
    S operator()(T seq)
    {
        S idx(m_missing) ; 
        for(unsigned int i=0 ; i < TSPARSE_LOOKUP_N ; i++)
        {
            if(seq == dev_tsparse_lookup[i]) idx = i + m_offset ;
            // NB not breaking as hope this will keep memory access lined up between threads 
        }
        return idx ; 
    } 
};


template <typename T>
template <typename S>
void TSparse<T>::apply_lookup(CBufSlice target)
{
    typedef typename thrust::device_vector<T>::iterator T_Iterator;
    typedef typename thrust::device_vector<S>::iterator S_Iterator;

    thrust::device_ptr<T> psrc = thrust::device_pointer_cast((T*)m_source.dev_ptr) ; 
    thrust::device_ptr<S> ptgt = thrust::device_pointer_cast((S*)target.dev_ptr) ; 

    strided_range<T_Iterator> src( psrc + m_source.begin, psrc + m_source.end, m_source.stride ); 
    strided_range<S_Iterator> tgt( ptgt +   target.begin, ptgt +   target.end,   target.stride ); 

    S missing = std::numeric_limits<S>::max() ;
    S offset  =  1 ; 

    thrust::transform( src.begin(), src.end(), tgt.begin(), apply_lookup_functor<T,S>(offset, missing) ); 
}



template <typename T>
void TSparse<T>::populate_index(Index* index)
{
    for(unsigned int i=0 ; i < m_values_h.size() ; i++)
    { 
        T val = m_values_h[i] ; 
        int cnt = m_counts_h[i] ;
        std::string key = m_hexkey ? BHex<T>::as_hex(val) : BHex<T>::as_dec(val) ;

#ifdef DEBUG
        std::cout << "TSparse<T>::populate_index " 
                  << " i " << std::setw(4) << i
                  << " val " << std::setw(10) << val
                  << " cnt " << std::setw(10) << cnt
                  << " key " << key 
                  << std::endl 
                  ;
#endif

        if(cnt > 0) index->add(key.c_str(), cnt ); 
    }
}


template class THRAP_API TSparse<unsigned long long> ;
template class THRAP_API TSparse<int> ;

template THRAP_API void TSparse<unsigned long long>::apply_lookup<unsigned char>(CBufSlice target);
template THRAP_API void TSparse<int>::apply_lookup<char>(CBufSlice target);


