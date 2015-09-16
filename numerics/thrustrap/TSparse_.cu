#include "TSparse.hh"
#include "TBuf.hh"

#include "stringutil.hpp"
#include "Index.hpp"

#include <iostream>
#include <iomanip>

#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/constant_iterator.h>

#include "strided_range.h"

//
// *dev_tsparse_lookup* 
//        is a global symbol pointing at device constant memory 
//        **CAUTION : UNAVOIDABLE NAME COLLISION DANGER**
//    
//        http://stackoverflow.com/questions/7961792/device-constant-const/7963395#7963395
//

__constant__ unsigned long long dev_tsparse_lookup[TSPARSE_LOOKUP_N]; 

template <typename T>
void TSparse<T>::make_lookup()
{
    count_unique();
    update_lookup();
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
                          m_values.begin()      // values_first
                       );
    // sorts keys and values into ascending key order, as the counts are in 
    // the key slots this sorts in ascending count order

}


template <typename T>
void TSparse<T>::update_lookup()
{
    // partial pullback, only need small number (~32) of most popular ones on host 

    unsigned int n = TSPARSE_LOOKUP_N ; 

    m_values_h.resize(n, 0);   // TODO verify that using 0 as "empty" will not be mis-construed as a valid index 
    m_counts_h.resize(n, 0);

    if(m_num_unique > n)
    {
        thrust::copy( m_values.end() - n, m_values.end(), m_values_h.begin() );
        thrust::copy( m_counts.end() - n, m_counts.end(), m_counts_h.begin() );
    }
    else
    {
        thrust::copy( m_values.begin(), m_values.end(), m_values_h.begin() );
        thrust::copy( m_counts.begin(), m_counts.end(), m_counts_h.begin() );
    }  

    thrust::reverse(m_values_h.begin(), m_values_h.end());
    thrust::reverse(m_counts_h.begin(), m_counts_h.end());

    T* data = m_values_h.data();

    m_index_h = make_index();

#ifdef DEBUG
    printf("TSparse<T>::update_lookup<T>\n");
    for(unsigned int i=0 ; i < TSPARSE_LOOKUP_N ; i++) 
          std::cout << std::dec << std::setw(4) << i 
                    << " " << std::hex << std::setw(16) << *(data + i) << std::dec 
                    << std::endl ;
#endif
 
    cudaMemcpyToSymbol(dev_tsparse_lookup,data,TSPARSE_LOOKUP_N*sizeof(T));
}




template <typename T>
void TSparse<T>::dump(const char* msg)
{
    printf("%s : num_unique %u \n", msg, m_num_unique );
    for(unsigned int  i=0 ; i < m_values_h.size() ; i++)
    {
        std::cout << std::hex << std::setw(16) << m_values_h[i] 
                  << std::dec << std::setw(10) << m_counts_h[i]
                  << std::endl ;  
    } 
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
Index* TSparse<T>::make_index()
{
    Index* index = new Index(m_label);
    for(unsigned int i=0 ; i < m_values_h.size() ; i++)
    { 
        std::string name = as_hex(m_values_h[i]);
        index->add(name.c_str(), m_counts_h[i] ); 
    }
    return index ; 
}




template class TSparse<unsigned long long> ;
template void TSparse<unsigned long long>::apply_lookup<unsigned char>(CBufSlice target);


