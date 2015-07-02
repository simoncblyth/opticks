#pragma once
#include <string>
#include <vector>
#include <algorithm>
#include <map>
#include <iostream>
#include <iomanip>
#include "string.h"

template<typename T>
class CountsNPY {
    public:
          typedef typename std::map<std::string, T>  MST ; 
          typedef typename std::pair<std::string, T> ST ; 
          static bool ascending_count_order(const ST& a, const ST& b);
          static bool descending_count_order(const ST& a, const ST& b);
    public:
          CountsNPY(const char* name=NULL);
          void add(const ST& p);
          void add(const MST& m);
    public:
          void sort(bool ascending=true);
          void dump(const char* msg="CountsNPY::dump", unsigned long nline=32);
    public:
          std::vector<ST>& counts();          
    private:
          const char*      m_name ;  
          std::vector<ST>  m_counts  ;

};

template<typename T>
inline CountsNPY<T>::CountsNPY(const char* name)
       :
       m_name(name ? strdup(name) : NULL)
{
} 

template<typename T>
inline std::vector<std::pair<std::string,T>>& CountsNPY<T>::counts()
{
     return m_counts ; 
} 


template<typename T>
inline void CountsNPY<T>::add(const ST& p)
{
    m_counts.push_back(p);
} 

template<typename T>
inline void CountsNPY<T>::add(const MST& m)
{
    for(typename MST::const_iterator it=m.begin() ; it != m.end() ; it++ ) add(*it) ;
} 

template<typename T>
bool CountsNPY<T>::ascending_count_order(const ST& a, const ST& b)
{
    return b.second > a.second ;
}

template<typename T>
bool CountsNPY<T>::descending_count_order(const ST& a, const ST& b)
{
    return a.second > b.second ;
}

template<typename T>
void CountsNPY<T>::sort(bool ascending)
{
    std::sort(m_counts.begin(), m_counts.end(), ascending ? ascending_count_order : descending_count_order  );
}

template<typename T>
void CountsNPY<T>::dump(const char* msg, unsigned long nline)
{
    std::cout << msg << std::endl ; 
    for(unsigned int i=0 ; i < std::min(nline,m_counts.size()) ; i++)
    {
        ST& p = m_counts[i];
        std::cout << std::setw(5) << i 
                  << std::setw(35) << p.first 
                  << std::setw(20) << p.second
                  << std::endl ; 
    }
}


