#pragma once
/**
s_pool.h (alt names: sregistry.h, sbacking.h, sstore.h spersist.h)
=======================================================================

NB after calls to *s_pool::remove* the integer keys returned by *s_pool::index* 
will not match the *pid* returned by *s_pool::add* and *s_pool::remove*
as the *pid* is based on a global count of additions to the pool with no accounting 
for any deletions, whereas the the *index* adjusts to the size of the current pool providing 
a contiguous key. 

**/

#include <cassert>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <map>
#include <vector>

#include "ssys.h"


template<typename T>
struct s_find
{
    const T* q ; 
    s_find(const T* q_) : q(q_) {} ;  
    bool operator()(const std::pair<int, T*>& p){ return q == p.second ; }  
};

template<typename T>    
struct s_pool
{
    typedef typename std::map<int, T*> POOL ; 
    POOL pool ; 
    int count ; 
    int level ;  

    s_pool(); 

    int size() const ; 
    int get_num_root() const ; 
    T*  get_root(int lvid) const ; 

    std::string brief(const char* msg=nullptr) const ; 
    std::string desc(const char* msg=nullptr) const ; 

    int index(const T* q) const ; 
    int add( T* o ); 
    int remove( T* o ); 

    template<typename P>
    void serialize(   std::vector<P>& buf ) const ; 

    template<typename P>
    void import(const std::vector<P>& buf ) ; 

    template<typename P>
    static std::string Desc(const std::vector<P>& buf ); 
};

template<typename T>
s_pool<T>::s_pool()
    :
    count(0),
    level(ssys::getenvint("s_pool_level",0))
{
}

template<typename T>
int s_pool<T>::size() const 
{
    return pool.size(); 
}
template<typename T>
int s_pool<T>::get_num_root() const 
{
    int count_root = 0 ; 
    typedef typename POOL::const_iterator IT ; 
    for(IT it=pool.begin() ; it != pool.end() ; it++) 
    {
        T* n = it->second ;  
        if(n->is_root()) count_root += 1 ; 
    }
    return count_root ; 
}
template<typename T>
T* s_pool<T>::get_root(int lvid) const 
{
    T* root = nullptr ; 
    int count_root = 0 ; 
    typedef typename POOL::const_iterator IT ; 
    for(IT it=pool.begin() ; it != pool.end() ; it++) 
    {
        T* n = it->second ;  
        if(n->is_root()) 
        {
            if( lvid == count_root ) root = n ; 
            count_root += 1 ; 
        }
    }
    return root ; 
}

template<typename T>
std::string s_pool<T>::brief(const char* msg) const 
{
    std::stringstream ss ; 
    ss
       << "s_pool::brief "
       << ( msg ? msg : "-" )
       << " count " << count 
       << " pool.size " << pool.size() 
       << " num_root " << get_num_root()
       ;
    std::string str = ss.str(); 
    return str ; 
}

template<typename T>
std::string s_pool<T>::desc(const char* msg) const 
{
    std::stringstream ss ; 
    ss << "s_pool::desc "
       << ( msg ? msg : "-" )
       << " count " << count 
       << " pool.size " << pool.size() 
       << " num_root " << get_num_root()
       << std::endl
        ; 

    typedef typename POOL::const_iterator IT ; 
    for(IT it=pool.begin() ; it != pool.end() ; it++) 
    {
        int key = it->first ; 
        T* n = it->second ;  
        ss << std::setw(3) << key << " : " << n->desc() << std::endl ; 
    }
    std::string str = ss.str(); 
    return str ; 
}


/**
s_pool::index
--------------

Contiguous index of *q* within all active objects in creation order.
NB this is different from the *pid* because node deletions will 
cause gaps in the pid values whereas the indices will be contiguous. 

**/

template<typename T>
int s_pool<T>::index(const T* q) const 
{
    if( q == nullptr ) return -1 ;     
    s_find<T> find(q); 
    size_t idx = std::distance( pool.begin(), std::find_if( pool.begin(), pool.end(), find )); 
    return idx < pool.size() ? idx : -1 ;  
}
template<typename T>
int s_pool<T>::add(T* o)
{
    int pid = count ; 
    pool[pid] = o ; 
    if(level > 0) std::cerr << "s_pool::add pid " << pid << std::endl ; 
    count += 1 ; 
    return pid ; 
}
template<typename T>
int s_pool<T>::remove(T* o)
{
    s_find<T> find(o); 
    typename POOL::iterator it = std::find_if( pool.begin(), pool.end(), find ) ; 

    int pid = -1 ; 
    if( it == pool.end() )
    {
        if(level > 0) std::cerr << "s_pool::remove failed to find the object : already removed, double dtors ?  " << std::endl ; 
    }
    else
    {
        pid = it->first ; 
        if(level > 0) std::cerr << "s_pool::remove pid " << pid << std::endl ; 
        pool.erase(it); 
    }
    return pid ; 
} 

template<typename T>
template<typename P>
inline void s_pool<T>::serialize( std::vector<P>& buf ) const 
{
    buf.resize(pool.size());  
    for(typename POOL::const_iterator it=pool.begin() ; it != pool.end() ; it++)
    {
        size_t idx = std::distance(pool.begin(), it ); 
        T::Serialize( buf[idx], it->second ); 
    }
}

template<typename T>
template<typename P>
inline void s_pool<T>::import(const std::vector<P>& buf ) 
{
    if(level > 0) std::cerr << "s_pool::import buf.size " << buf.size() << std::endl ; 
    for(size_t idx=0 ; idx < buf.size() ; idx++) T::Import( &buf[idx], buf ) ; 
}

template<typename T>
template<typename P>
inline std::string s_pool<T>::Desc(const std::vector<P>& buf )  // static
{
    std::stringstream ss ; 
    ss << "s_pool::Desc buf.size " <<  buf.size() << std::endl ; 
    for(size_t idx=0 ; idx < buf.size() ; idx++) ss << " idx " << std::setw(3) << idx  << " : " <<  buf[idx].desc() << std::endl ; 
    std::string str = ss.str(); 
    return str ; 
}





