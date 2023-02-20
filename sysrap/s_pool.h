#pragma once
/**
s_pool.h (alt names: sregistry.h, sbacking.h, sstore.h spersist.h)
=======================================================================

Types:

* T : source type eg *sn* which can have non-value members, eg pointers and vector of pointers
* P : persisting type eg *_sn* paired with the source type where pointers are replaced with integers 
* S: serialization type used to form the NP.hh array, typically int but may also be float or double  


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
#include "NP.hh"


template<typename T>
struct s_find
{
    const T* q ; 
    s_find(const T* q_) : q(q_) {} ;  
    bool operator()(const std::pair<int, T*>& p){ return q == p.second ; }  
};

template<typename T, typename P>    
struct s_pool
{
    typedef typename std::map<int, T*> POOL ; 
    POOL pool ; 
    int count ; 
    int level ;  

    s_pool(); 

    int size() const ; 
    int get_num_root() const ; 
    T*  get_root(int idx) const ; 

    std::string brief(const char* msg=nullptr) const ; 
    std::string desc(const char* msg=nullptr) const ; 

    int index(const T* q) const ; 
    int add( T* o ); 
    int remove( T* o ); 

    void serialize_(   std::vector<P>& buf ) const ; 
    void import_(const std::vector<P>& buf ) ; 

    template<typename S> NP*  serialize() const ; 
    template<typename S> void import(const NP* a) ; 

    static std::string Desc(const std::vector<P>& buf ); 
};

template<typename T, typename P>
s_pool<T,P>::s_pool()
    :
    count(0),
    level(ssys::getenvint("s_pool_level",0))
{
}

template<typename T, typename P>
int s_pool<T,P>::size() const 
{
    return pool.size(); 
}
template<typename T, typename P>
int s_pool<T,P>::get_num_root() const 
{
    int count_root = 0 ; 
    typedef typename POOL::const_iterator IT ; 
    for(IT it=pool.begin() ; it != pool.end() ; it++) 
    {
        T* n = it->second ;  
        if(n->is_root_importable()) count_root += 1 ; 
    }
    return count_root ; 
}
template<typename T, typename P>
T* s_pool<T,P>::get_root(int idx) const 
{
    T* root = nullptr ; 
    int count_root = 0 ; 
    typedef typename POOL::const_iterator IT ; 
    for(IT it=pool.begin() ; it != pool.end() ; it++) 
    {
        T* n = it->second ;  
        if(n->is_root_importable()) 
        {
            if( idx == count_root ) root = n ; 
            count_root += 1 ; 
        }
    }
    return root ; 
}

template<typename T, typename P>
std::string s_pool<T,P>::brief(const char* msg) const 
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

template<typename T, typename P>
std::string s_pool<T,P>::desc(const char* msg) const 
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

template<typename T, typename P>
int s_pool<T, P>::index(const T* q) const 
{
    if( q == nullptr ) return -1 ;     
    s_find<T> find(q); 
    size_t idx = std::distance( pool.begin(), std::find_if( pool.begin(), pool.end(), find )); 
    return idx < pool.size() ? idx : -1 ;  
}

template<typename T, typename P>
int s_pool<T, P>::add(T* o)
{
    int pid = count ; 
    pool[pid] = o ; 
    if(level > 0) std::cerr << "s_pool::add pid " << pid << std::endl ; 
    count += 1 ; 
    return pid ; 
}

template<typename T, typename P>
int s_pool<T,P>::remove(T* o)
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

template<typename T, typename P>
inline void s_pool<T,P>::serialize_( std::vector<P>& buf ) const 
{
    buf.resize(pool.size());  
    for(typename POOL::const_iterator it=pool.begin() ; it != pool.end() ; it++)
    {
        size_t idx = std::distance(pool.begin(), it ); 
        if(level > 1) std::cerr << "s_pool::serialize_ " << idx << std::endl ; 
        T::Serialize( buf[idx], it->second ); 
    }
}

template<typename T, typename P>
inline void s_pool<T,P>::import_(const std::vector<P>& buf ) 
{
    if(level > 0) std::cerr << "s_pool::import_ buf.size " << buf.size() << std::endl ; 
    for(size_t idx=0 ; idx < buf.size() ; idx++) T::Import( &buf[idx], buf ) ; 
}

template<typename T, typename P>
template<typename S>
inline NP* s_pool<T,P>::serialize() const 
{
    std::vector<P> buf ; 
    serialize_(buf); 

    NP* a = NP::Make<S>( buf.size(), P::NV ) ; 
    a->read2<S>((S*)buf.data()); 

    return a ; 
}

template<typename T, typename P>
template<typename S>
inline void s_pool<T,P>::import( const NP* a ) 
{
    assert( a->shape[1] == P::NV );  

    std::vector<P> buf(a->shape[0]) ; 
    a->write<S>((S*)buf.data()); 

    import_(buf); 
}

template<typename T, typename P>
inline std::string s_pool<T,P>::Desc(const std::vector<P>& buf )  // static
{
    std::stringstream ss ; 
    ss << "s_pool::Desc buf.size " <<  buf.size() << std::endl ; 
    for(size_t idx=0 ; idx < buf.size() ; idx++) ss << " idx " << std::setw(3) << idx  << " : " <<  buf[idx].desc() << std::endl ; 
    std::string str = ss.str(); 
    return str ; 
}


