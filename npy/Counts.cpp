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

#include <algorithm>
#include <iostream>
#include <iomanip>
#include <cstring>

// brap-
#include "BList.hh"

// npy-
#include "Counts.hpp"
#include "Index.hpp"

#include "PLOG.hh"

template<typename T>
Counts<T>::Counts(const char* name)
       :
       m_name(name ? strdup(name) : NULL)
{
} 

template<typename T>
std::vector<std::pair<std::string,T> >& Counts<T>::counts()
{
     return m_counts ; 
} 

template<typename T>
unsigned int  Counts<T>::size()
{
     return m_counts.size() ; 
} 

template<typename T>
typename std::pair<std::string, T>&  Counts<T>::get(unsigned int index)
{
    return m_counts[index] ;
} 



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

    unsigned long size_ = m_counts.size() ;
    for(unsigned int i=0 ; i < std::min(nline,size_); i++)
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
    BList<std::string,T>::save(&m_counts, path);
}

template<typename T>
void Counts<T>::save(const char* dir, const char* name)
{
    BList<std::string,T>::save(&m_counts, dir, name);
}

template<typename T>
void Counts<T>::load_(const char* path)
{
    BList<std::string,T>::load(&m_counts, path);
}

template<typename T>
void Counts<T>::load_(const char* dir, const char* name)
{
    BList<std::string,T>::load(&m_counts, dir, name);
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
Index* Counts<T>::make_index(const char* itemtype, const char* reldir)
{
    Index* index = new Index(itemtype, reldir);
    for(VSTI it=m_counts.begin() ; it != m_counts.end() ; it++) index->add(it->first.c_str(), it->second);
    return index ; 
}


template class Counts<unsigned int>;

