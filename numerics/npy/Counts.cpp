#include "Counts.hpp"
#include "jsonutil.hpp"

#include "Index.hpp"

#include <algorithm>
#include <iostream>
#include <iomanip>
#include "string.h"

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal





template<typename T>
class FindKey {
    public:
       FindKey(const char* wanted) : m_wanted(wanted) {} 
       bool operator()(const std::pair<std::string, T>& p) const { return strcmp(m_wanted.c_str(), p.first.c_str()) == 0 ; }
    private:
       std::string m_wanted  ;
};



template<typename T>
typename std::vector<std::pair<std::string, T> >::iterator Counts<T>::find(const char* key)
{
    VSTI it = std::find_if( m_counts.begin(), m_counts.end(), FindKey<T>(key) );
    return it ;
} 



template<typename T>
T  Counts<T>::getCount(const char* key)
{
    T ret(0) ;
    VSTI it = find(key) ;
    if(it != m_counts.end()) ret = it->second ;  
    return ret ; 
}


template<typename T>
void Counts<T>::checkfind(const char* key)
{
    VSTI it = find(key) ;
    if(it != m_counts.end())
    {
        std::cout << "::checkfind " 
                  << std::setw(25) << key
                  << std::setw(25) << it->first  
                  << std::setw(25) << it->second
                  << std::endl ; 
    } 
}


template<typename T>
void Counts<T>::addPair(const ST& p)
{
    m_counts.push_back(p);
} 

template<typename T>
void Counts<T>::add(const char* key, T count)
{
    VSTI it = find(key) ;
    if(it == m_counts.end()) addPair(ST(key, count)); 
    else
        it->second += count ;
}

template<typename T>
void Counts<T>::addMap(const MST& m)
{
    for(typename MST::const_iterator it=m.begin() ; it != m.end() ; it++ ) addPair(*it) ;
} 

template<typename T>
bool Counts<T>::ascending_count_order(const ST& a, const ST& b)
{
    return b.second > a.second ;
}

template<typename T>
bool Counts<T>::descending_count_order(const ST& a, const ST& b)
{
    return a.second > b.second ;
}

template<typename T>
void Counts<T>::sort(bool ascending)
{
    std::sort(m_counts.begin(), m_counts.end(), ascending ? ascending_count_order : descending_count_order  );
}

template<typename T>
void Counts<T>::dump(const char* msg, unsigned long nline)
{
    LOG(info) << msg << " " << m_name ; 

    unsigned long size = m_counts.size() ;
    for(unsigned int i=0 ; i < std::min(nline,size); i++)
    {
        ST& p = m_counts[i];
        std::cout << std::setw(5) << std::dec << i 
                  << std::setw(35) << p.first 
                  << std::setw(20) << std::dec << p.second
                  << std::endl ; 
    }
}




template<typename T>
void Counts<T>::save(const char* path)
{
    saveList<std::string, T>(m_counts, path);
}

template<typename T>
void Counts<T>::save(const char* dir, const char* name)
{
    saveList<std::string, T>(m_counts, dir, name);
}

template<typename T>
void Counts<T>::load_(const char* path)
{
    loadList<std::string, T>( m_counts, path);
}

template<typename T>
void Counts<T>::load_(const char* dir, const char* name)
{
    loadList<std::string, T>( m_counts, dir, name);
}

template<typename T>
Counts<T>* Counts<T>::load(const char* dir, const char* name)
{
    Counts<T>* cn = new Counts<T>();
    cn->load_(dir, name);
    return cn ; 
}

template<typename T>
Counts<T>* Counts<T>::load(const char* path)
{
    Counts<T>* cn = new Counts<T>();
    cn->load_(path);
    return cn ; 
}

template<typename T>
Index* Counts<T>::make_index(const char* itemtype)
{
    Index* index = new Index(itemtype);
    for(VSTI it=m_counts.begin() ; it != m_counts.end() ; it++) index->add(it->first.c_str(), it->second);
    return index ; 
}


template class Counts<unsigned int>;

