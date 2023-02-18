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
#include <iostream>
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

    int index(const T* q) const ; 
    int add( T* o ); 
    int remove( T* o ); 


    template<typename P>
    void serialize(   std::vector<P>& buf ) const ; 

    template<typename P>
    void import(const std::vector<P>& buf ) ; 
};

template<typename T>
s_pool<T>::s_pool()
    :
    count(0),
    level(ssys::getenvint("s_pool_level",0))
{
}

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
    assert( it != pool.end() ); 
    int pid = it->first ; 
    if(level > 0) std::cerr << "s_pool::remove pid " << pid << std::endl ; 
    pool.erase(it); 
    return pid ; 
} 

template<typename T>
template<typename P>
inline void s_pool<T>::serialize( std::vector<P>& buf ) const 
{
    int total = pool.size(); 
    buf.resize(total);  
    if(level > 0) std::cerr << "[ s_pool::serialize total " << total << std::endl ; 

    typename POOL::const_iterator it = pool.begin() ; 

    int idx = 0 ; 
    while( it != pool.end() )
    {
        //int key = it->first ; 
        T* t    = it->second ;  

        int idx1 = index(t) ; 
        assert( idx1 == idx ); 
        assert( idx < total ); 

        P& p = buf[idx]; 

        P::Serialize( p, t ); 

        it++ ; idx++ ;  
    }
    if(level > 0) std::cerr << "] s_pool::serialize" << std::endl ; 
}

template<typename T>
template<typename P>
inline void s_pool<T>::import(const std::vector<P>& buf ) 
{
    int total = buf.size() ;
    if(level > 0) std::cerr << "[ s_pool::import total " << total << std::endl ; 
    for(int idx=0 ; idx < total ; idx++)
    { 
        const P& p = buf[idx]; 
        T* t = P::Import( p ); 
        //assert(t); 
    }  
    if(level > 0) std::cerr << "] s_pool::import" << std::endl ; 
}

