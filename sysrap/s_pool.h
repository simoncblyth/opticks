#pragma once
/**
s_pool.h (alt names: sregistry.h, sbacking.h, sstore.h spersist.h)
=======================================================================

Types:

* T : source type eg *sn* which can have non-value members, eg pointers and vector of pointers
* P : persisting type eg *_sn* paired with the source type where pointers are replaced with integers 
* S : serialization type used to form the NP.hh array, typically int but may also be float or double  

NB after calls to *s_pool::remove* the integer keys returned by *s_pool::index* 
will not match the *pid* returned by *s_pool::add* and *s_pool::remove*
as the *pid* is based on a global count of additions to the pool with no accounting 
for any deletions, whereas the the *index* adjusts to the size of the current pool providing 
a contiguous key. 


::

    epsilon:opticks blyth$ opticks-fl "s_pool\.h"
    ./sysrap/tests/Obj.h
    ./sysrap/s_pool.h
    ./sysrap/sndtree.h
    ./sysrap/sn.h
    ./sysrap/stv.h


**/

#include <cassert>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <map>
#include <vector>
#include <functional>

#include "ssys.h"
#include "NPX.h"


/**
s_find
---------

s_find is used by s_pool::index and s_pool::remove in std::find_if 
traversals of the pools which are maps of T* pointers with int keys  

**/

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
    const char* label ; 
    int count ; 
    int level ;  

    s_pool(const char* label); 

    int size() const ; 
    int num_root() const ; 
    bool all_root() const ; 

    T*  get_root(int idx) const ; 
    T*  get(int idx) const ; 

    void find(std::vector<T*>& vec, std::function<bool(const T*)> predicate ) const ; 

    std::string brief() const ; 
    std::string desc() const ; 

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
inline s_pool<T,P>::s_pool(const char* label)
    :
    label(label ? strdup(label) : nullptr),
    count(0),
    level(ssys::getenvint("s_pool_level",0))
{
}

template<typename T, typename P>
inline int s_pool<T,P>::size() const 
{
    return pool.size(); 
}
template<typename T, typename P>
inline int s_pool<T,P>::num_root() const 
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

template<typename T, typename P>
inline bool s_pool<T,P>::all_root() const 
{
    return size() == num_root() ; 
}



/**
s_pool<T,P>::get_root
----------------------

HMM: this is also used for s_pool<stv,_stv> transforms 
which are a single-node-tree, so the name is a little confusing.
Of course that just means that every transform is regarded
as "root".  

**/

template<typename T, typename P>
inline T* s_pool<T,P>::get_root(int idx) const 
{
    T* root = nullptr ; 
    int count_root = 0 ; 
    typedef typename POOL::const_iterator IT ; 
    for(IT it=pool.begin() ; it != pool.end() ; it++) 
    {
        T* n = it->second ;  
        if(n->is_root()) 
        {
            if( idx == count_root ) root = n ; 
            count_root += 1 ; 
        }
    }
    return root ; 
}

template<typename T, typename P>
inline T* s_pool<T,P>::get(int idx) const 
{
    return pool.count(idx) == 1 ? pool.at(idx) : nullptr ; 
}


template<typename T, typename P>
inline void s_pool<T,P>::find(std::vector<T*>& vec, std::function<bool(const T*)> predicate ) const 
{
    typedef typename POOL::const_iterator IT ; 
    for(IT it=pool.begin() ; it != pool.end() ; it++) 
    {
        T* n = it->second ;  
        if(predicate(n)) vec.push_back(n) ; 
    }
}


template<typename T, typename P>
inline std::string s_pool<T,P>::brief() const 
{
    std::stringstream ss ; 
    ss
       << "s_pool::brief "
       << " label " << ( label ? label : "-" )
       << " count " << count 
       << " pool.size " << pool.size() 
       << " num_root " << num_root()
       ;
    std::string str = ss.str(); 
    return str ; 
}

template<typename T, typename P>
inline std::string s_pool<T,P>::desc() const 
{
    std::stringstream ss ; 
    ss << "s_pool::desc "
       << " label " << ( label ? label : "-" )
       << " count " << count 
       << " pool.size " << pool.size() 
       << " num_root " << num_root()
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
inline int s_pool<T, P>::index(const T* q) const 
{
    if( q == nullptr && level > 0) std::cerr 
         << "s_pool::index got nullptr arg "
         << " pool.size " << pool.size()
         << std::endl 
         ;
     
    if( q == nullptr ) return -1 ;     

    s_find<T> find(q); 
    size_t idx_ = std::distance( pool.begin(), std::find_if( pool.begin(), pool.end(), find )); 
    int idx = idx_ < pool.size() ? idx_ : -1 ;  

    if( idx == -1 && level > 0) std::cerr
         << "s_pool::index failed to find non-nullptr  "
         << " pool.size " << pool.size()
         << std::endl 
         ;
 
    return idx ; 
}

template<typename T, typename P>
inline int s_pool<T, P>::add(T* o)
{
    int pid = count ; 
    pool[pid] = o ; 
    if(level > 0) std::cerr 
        << "s_pool::add " 
        << ( label ? label : "-" ) 
        << " pid " << pid 
        << std::endl 
        ; 
    count += 1 ; 
    return pid ; 
}

template<typename T, typename P>
inline int s_pool<T,P>::remove(T* o)
{
    s_find<T> find(o); 
    typename POOL::iterator it = std::find_if( pool.begin(), pool.end(), find ) ; 

    int pid = -1 ; 
    if( it == pool.end() )
    {
        if(level > 0) std::cerr 
           << "s_pool::remove " 
           << ( label ? label : "-"  ) 
           << " failed to find the object : already removed, double dtors ?  " 
           << std::endl 
           ; 
    }
    else
    {
        pid = it->first ; 
        if(level > 0) std::cerr 
            << "s_pool::remove " 
            << ( label ? label : "-"  ) 
            << " pid " << pid 
            << std::endl
            ; 
        pool.erase(it); 
    }
    return pid ; 
} 

/**
s_pool<T,P>::serialize_
------------------------

For example with s_pool<sn,_sn> this serializes the entire pool 
into the std::vector<_sn>& buf of the argument by 
calling sn::Serialize between refs to _sn and sn pointer.    

**/

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

/**
s_pool<T,P>::import_
----------------------

Import from eg std::vector<_sn>& buf into the s_pool<sn,_sn> 
by calling sn::Import with arguments  _sn* and the full vector.  

**/

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
    serialize_(buf);                                 // from pool into the buf of P objects 
    return NPX::ArrayFromVec_<S, P>(buf, P::ITEM) ;  // from buf into array  
}

template<typename T, typename P>
template<typename S>
inline void s_pool<T,P>::import( const NP* a ) 
{
    if(level > 0) std::cerr << "s_pool::import a.sstr " << ( a ? a->sstr() : "-" )  << std::endl ; 
    std::vector<P> buf(a->shape[0]) ; 

    NPX::VecFromArray<P>(buf, a );  // from array into buf

    import_(buf);                   // from buf into pool 
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


