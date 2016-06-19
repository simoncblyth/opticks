#include <numeric>
#include <algorithm>
#include <cstring>

#include "BStr.hh"
#include "BHex.hh"
#include "BLog.hh"

#include "NPY.hpp"
#include "Index.hpp"

#include "Sparse.hh"


template <typename T>
Sparse<T>::Sparse(const char* label, NPY<T>* source, bool hexkey) 
   :
   m_label(strdup(label)),
   m_source(source),
   m_hexkey(hexkey),
   m_num_unique(0),
   m_num_lookup(0),
   m_index(NULL)
{
   init();
}

template <typename T>
Index* Sparse<T>::getIndex()
{
    return m_index ; 
}





template <typename T>
void Sparse<T>::init()
{
    m_index = new Index(m_label);
}

template <typename T>
void Sparse<T>::make_lookup()
{
    count_unique();
    update_lookup();
    populate_index(m_index);
}

template <typename T>
void Sparse<T>::count_unique()
{
    typedef std::vector<T> V ;  
    V data(m_source->data());

    std::sort(data.begin(), data.end());

    m_num_unique = std::inner_product(
                                  data.begin(),data.end() - 1,   // first1, last1  
                                             data.begin() + 1,   // first2
                                                       int(1),   // output type init 
                                             std::plus<int>(),   // reduction operator
                                       std::not_equal_to<T>()    // pair-by-pair operator, returning 1 at edges 
                                      );  

    LOG(info) << "Sparse<T>::count_unique"
              << " label " << m_label
              << " num_unique " << m_num_unique 
              ;

    reduce_by_key(data);
    sort_by_key();

}


template <typename T>
void Sparse<T>::update_lookup()
{
    unsigned int max_uniques = SPARSE_LOOKUP_N  ;

    T zero(0);
    m_lookup.resize( max_uniques, zero); 

    m_num_lookup = std::min( m_num_unique, max_uniques );

    for(unsigned int  i=0 ; i < m_num_lookup ; i++)
    {
        P valuecount = m_valuecount[i] ;
        T value = valuecount.first ;
        m_lookup[i] = value ; 
    }

    memcpy(m_sparse_lookup, m_lookup.data(), SPARSE_LOOKUP_N*sizeof(T));
}


template <typename T>
unsigned int Sparse<T>::count_value(const T value) const 
{
    typedef std::vector<T> V ;
    V& data = m_source->data();
    return std::count(data.begin(), data.end(), value );
}


template <typename T>
void Sparse<T>::reduce_by_key(std::vector<T>& data)
{
    m_valuecount.resize(m_num_unique);
 
    unsigned int n = data.size();
    T* vals = data.data() ; 
    T prev = *(vals + 0);
    int count(1) ; 
    unsigned int unique(0) ;

    T value(0) ; 

    // from the second
    for(unsigned int i=1 ; i < n ; i++)
    {
        value = *(vals+i) ;

#if DEBUG
        if(count < 10)
        LOG(info)
                  << std::setw(3) << "c"
                  << std::setw(8) << i 
                  << std::hex << std::setw(16) << value 
                  << std::dec << std::setw(16) << value 
                  << std::setw(8) << count 
                  ; 
#endif

        if(value == prev)
        {
            count += 1 ; 
        }
        else 
        {
#if DEBUG
            if(unique < 100)
            LOG(info) 
                      << std::setw(3) << "u"
                      << std::setw(8) << i 
                      << std::hex << std::setw(16) << value 
                      << std::dec << std::setw(16) << value 
                      << std::setw(8) << count 
                      << std::setw(8) << unique
                      ; 
#endif

            m_valuecount[unique++] = P(prev, count) ;
            prev = value ; 
            count = 1 ; 
        } 
    }
 
    // no transition possible, so must special case the last
    if(value == prev) 
    {
        m_valuecount[unique++] = P(value, count) ;
    }

    LOG(info) << "Sparse<T>::reduce_by_key"
              << " unique " << unique 
              << " num_unique " << m_num_unique 
              ;

    assert(unique == m_num_unique);
}


template <class T>
struct second_descending : public std::binary_function<T,T,bool> {
    bool operator()(const T& a, const T& b) const {
        return a.second > b.second ;
    }
};

template <typename T>
void Sparse<T>::sort_by_key()
{
    std::sort( m_valuecount.begin(), m_valuecount.end(), second_descending<P>());
}



template <typename T>
std::string Sparse<T>::dump_(const char* msg, bool slowcheck) const
{
    std::stringstream ss ;
    ss << msg << " : num_unique " << m_num_unique << std::endl ;

    for(unsigned int  i=0 ; i < m_valuecount.size() ; i++)
    {
        P valuecount = m_valuecount[i] ;
        T value = valuecount.first ;
        int count = valuecount.second ;

        ss << "[" << std::setw(2) << i << "] "  ;
        if(m_hexkey) ss << std::hex ;
        ss << std::setw(16) << value ;
        if(m_hexkey) ss << std::dec ;
        ss << std::setw(10) << count ;

        if(slowcheck)
        {
            ss << std::setw(10) << count_value(value) ;
        }
        ss << std::endl ;
    }

    return ss.str();
}


template <typename T> 
void Sparse<T>::dump(const char* msg) const 
{
    bool slowcheck = false ; 
    LOG(info) << dump_(msg, slowcheck) ; 
}



template <typename T>
void Sparse<T>::populate_index(Index* index)
{
    for(unsigned int i=0 ; i < m_num_lookup ; i++)
    {
        P valuecount = m_valuecount[i] ;
        T value = valuecount.first ;
        int count = valuecount.second ;

        std::string key = m_hexkey ? BHex<T>::as_hex(value) : BHex<T>::as_dec(value) ;
        if(count > 0) index->add(key.c_str(), count );

#ifdef DEBUG
        std::cout << "Sparse<T>::populate_index "
                  << " i " << std::setw(4) << i
                  << " value " << std::setw(10) << value
                  << " count " << std::setw(10) << count
                  << " key " << key
                  << std::endl
                  ;
#endif

    }
}



template <typename T, typename S>
struct apply_lookup_functor : public std::unary_function<T, S>
{
    S m_offset ;
    S m_missing ;
    S m_size ; 
    T* m_lookup ; 

    apply_lookup_functor(S offset, S missing, S size, T* lookup)
        :
        m_offset(offset),
        m_missing(missing),
        m_size(size),
        m_lookup(lookup)
    {
    }


   S operator()(T seq)
   {
        S idx(m_missing);
        for(unsigned int i=0 ; i < m_size ; i++)
        {
            if(seq == m_lookup[i]) idx = i + m_offset ;
        }
        return idx ;
   }
};



template <typename T>
template <typename S>
void Sparse<T>::apply_lookup(S* target, unsigned int stride, unsigned int offset)
{
    S s_missing = std::numeric_limits<S>::max() ;
    S s_offset  =  1 ;
    apply_lookup_functor<T,S> fn(s_offset, s_missing, SPARSE_LOOKUP_N, m_sparse_lookup );

    T* src = m_source->getValues();
    unsigned int size = m_source->getShape(0);

    for(unsigned int i=0 ; i < size ; i++)
    {
         T value = *(src + i) ;
         *(target + i*stride + offset) = fn(value) ; 
    }
}




template class Sparse<unsigned long long> ;
template void Sparse<unsigned long long>::apply_lookup<unsigned char>(unsigned char* target, unsigned int stride, unsigned int offset);



