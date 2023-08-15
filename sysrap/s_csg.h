#pragma once

#include "stv.h"
#include "sn.h"

#include "NPFold.h"
#include "SYSRAP_API_EXPORT.hh"

struct SYSRAP_API s_csg
{
    static s_csg* INSTANCE ; 
    static s_csg* Get(); 
    static NPFold* Serialize(); 
    static void Import(const NPFold* fold); 


    stv::POOL* tv_pool ; 
    sn::POOL*  n_pool ; 

    s_csg(); 
    void init(); 

    int total_size() const ; 
    std::string brief() const ; 
    std::string desc() const ; 

    NPFold* serialize() const ; 
    void import(const NPFold* fold); 
};

inline s_csg::s_csg()
    :
    tv_pool(new stv::POOL),
    n_pool(new   sn::POOL)
{
    init(); 
}

inline void s_csg::init()
{
    INSTANCE = this ; 
    stv::SetPOOL(tv_pool) ; 
    sn::SetPOOL(n_pool) ; 
}

inline int s_csg::total_size() const
{
    return tv_pool->size() + n_pool->size() ; 
}

inline std::string s_csg::brief() const
{
    std::stringstream ss ; 
    ss << "s_csg::brief total_size " << total_size() 
       << std::endl  
       << " tv_pool : " << ( tv_pool ? tv_pool->brief() : "-" ) 
       << std::endl
       << " n_pool : " << ( n_pool ? n_pool->brief() : "-" ) 
       << std::endl 
       ;
    std::string str = ss.str(); 
    return str ; 
}

inline std::string s_csg::desc() const
{
    std::stringstream ss ; 
    ss << "s_csg::desc total_size " << total_size()  ; 
    std::string str = ss.str(); 
    return str ; 
}



inline NPFold* s_csg::serialize() const   
{
    NPFold* fold = new NPFold ; 
    fold->add(sn::NAME,  n_pool->serialize<int>() );  
    fold->add(stv::NAME, tv_pool->serialize<double>() ); 
    return fold ; 
}



/**
s_csg::Import
-------------

NB transforms are imported before nodes
so the transform hookup during node import works. 

That ordering is required because the sn have pointers 
referencing the transforms, hence transforms must be imported first. 

Similarly the aabb and param will need to be imported before the sn. 
Notice the ordering of the sn import doesnt matter because the full _sn
buffer gets imported in one go prior to doing any sn hookup. 

**/

inline void s_csg::import(const NPFold* fold) 
{
    tv_pool->import<double>(fold->get(stv::NAME)) ; 
    n_pool->import<int>(fold->get(sn::NAME)) ; 
}



