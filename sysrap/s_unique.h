#pragma once
/**
s_unique.h : similar to np.unique
===================================

**/

#include <vector>
#include <unordered_map>
#include <algorithm>
#include <numeric>

#include <string>
#include <sstream>
#include <iomanip>
#include <cassert>


/**
s_unique
----------

Loosely following the NumPy np.unique API prepare unique value table.

uvals
    unique values
first, last
    iterators for input values
count
    number of times each of the unique values appears
order
    indices into uvals giving descending count order
index
    indices into original values of first occurence, same length as uvals
inverse
    indices into the uvals that reconstruct the original values, same length as original
original
     original values provided from the iterator


Started from:

* https://stackoverflow.com/questions/70868307/c-equivalent-of-numpy-unique-on-stdvector-with-return-index-and-return-inver

**/


template<class T, class Iterator>
inline void s_unique(
     std::vector<T>& uvals,
     Iterator first,
     Iterator last,
     std::vector<std::size_t>* count=nullptr,
     std::vector<std::size_t>* order=nullptr,
     std::vector<std::size_t>* index=nullptr,
     std::vector<std::size_t>* inverse=nullptr,
     std::vector<T>* original=nullptr
)
{
    using index_map = std::unordered_map<T, std::size_t>;  // T value and count
    using map_iter = typename index_map::iterator;
    using map_value = typename index_map::value_type;

    // clear optional output vectors
    for(std::vector<std::size_t>* arg: {count, order, index, inverse}) if(arg) arg->clear();

    index_map map;

    std::size_t cur_idx = 0;
    for(Iterator i = first; i != last; ++cur_idx, ++i)
    {
        const T value = *i ;
        const std::pair<map_iter, bool> inserted = map.try_emplace(value, uvals.size());

        const bool& is_first_such_value = inserted.second ;
        map_value& ival = *inserted.first;

        if(is_first_such_value)
        {
            uvals.push_back(ival.first);
            if(index) index->push_back(cur_idx);
            if(count) count->push_back(1);
        }
        else
        {
            if(count) (*count)[ival.second] += 1;
        }
        if(inverse)  inverse->push_back(ival.second);
        if(original) original->push_back(value);
    }

    if(count && order)
    {
        // prep indices and sort them into descending count order
        order->resize(count->size()) ;
        std::iota(order->begin(), order->end(), 0);
        const std::vector<std::size_t>& cn = *count ;
        auto descending = [&cn](const std::size_t& a, const std::size_t &b) { return cn[a] > cn[b];}  ;
        std::sort(order->begin(), order->end(), descending );
    }
}



/**
s_unique_desc
---------------

Presents unique value table with labels provided by unams.
The table is ordered in descending count order.

**/

template<class T>
inline std::string s_unique_desc(
     const std::vector<T>& uvals,
     const std::vector<std::string>* unams=nullptr,
     std::vector<std::size_t>* count=nullptr,
     std::vector<std::size_t>* order=nullptr,
     std::vector<std::size_t>* index=nullptr,
     std::vector<std::size_t>* inverse=nullptr,
     std::vector<T>* original=nullptr,
     int w = 6,
     bool check=false
)
{
    int num_uvals = uvals.size() ;
    int num_unams = unams ? unams->size() : -1 ;
    int num_count = count ? count->size() : -1 ;
    int num_order = order ? order->size() : -1 ;
    int num_index = index ? index->size() : -1 ;
    int num_inverse = inverse ? inverse->size() : -1  ;
    int num_original = original ? original->size() : -1  ;

    if(num_count > -1) assert( num_uvals == num_count ) ;
    if(num_order > -1) assert( num_uvals == num_order ) ;
    if(num_index > -1) assert( num_uvals == num_index ) ;

    std::stringstream ss ;
    ss << "[s_unique_desc\n" ;
    ss << std::setw(15) << " num_uvals "    << std::setw(8) << num_uvals << "\n" ;
    ss << std::setw(15) << " num_unams "    << std::setw(8) << num_unams << "\n" ;
    ss << std::setw(15) << " num_count "    << std::setw(8) << num_count << "\n" ;
    ss << std::setw(15) << " num_order "    << std::setw(8) << num_order << "\n" ;
    ss << std::setw(15) << " num_index "    << std::setw(8) << num_index << "\n" ;
    ss << std::setw(15) << " num_inverse "  << std::setw(8) << num_inverse << "\n" ;
    ss << std::setw(15) << " num_original " << std::setw(8) << num_original << "\n" ;

    std::size_t cn_total = 0 ;
    for(std::size_t u=0 ; u < num_uvals ; u++ )
    {
        std::size_t i = order == nullptr ? u : (*order)[u] ;

        T uval = uvals[i] ;
        std::string un = unams ? (*unams)[uval] : "-" ;

        std::size_t cn = count ? (*count)[i] : 0 ;
        std::size_t ix = index ? (*index)[i] : 0 ;
        if(cn > 0)  cn_total += cn ;

        ss             << " uv " << std::setw(w) << uval << " : " ;
        if(count)   ss << " cn " << std::setw(w) << cn << " " ;
        if(index)   ss << " ix " << std::setw(w) << ix << " " ;
        ss << " un " << un << "\n" ;
    }
    ss << "    " << std::setw(w) << " " << "   " ;
    if(count)     ss << " TOT" << std::setw(w) << cn_total << "\n" ;

    ss << "\n" ;
    if( check && num_inverse > -1 && num_original > -1)
    {
        assert( num_inverse == num_original ) ;
        for(std::size_t i=0 ; i < num_original ; i++ )
        {
            std::size_t uidx = (*inverse)[i] ;  // index into uvals to reproduce original

            const T value = (*original)[i]  ;
            const T lookup = uvals[uidx] ;
            bool match = value == lookup ;

            ss << " value " << std::setw(8) << value << " : " ;
            ss << " uidx  " << std::setw(8) << uidx  << " " ;
            ss << " lookup " << std::setw(8) << lookup  << " " ;
            ss << " match " << ( match ? "YES" : "NO " ) ;
            ss << "\n" ;
        }
    }

    ss << "]s_unique_desc\n" ;
    std::string str = ss.str() ;
    return str ;
}

