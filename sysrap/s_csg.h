#pragma once

#include "s_pa.h"
#include "s_bb.h"
#include "s_tv.h"
#include "sn.h"

#include "NPFold.h"
#include "SYSRAP_API_EXPORT.hh"

struct SYSRAP_API s_csg
{
    static s_csg* INSTANCE ; 
    static s_csg* Get(); 
    static NPFold* Serialize(); 
    static void Import(const NPFold* fold); 

    s_pa::POOL* pa ; 
    s_bb::POOL* bb ; 
    s_tv::POOL* tv ; 
    sn::POOL*   n ; 

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
    pa(new s_pa::POOL),
    bb(new s_bb::POOL),
    tv(new s_tv::POOL),
    n(new    sn::POOL)
{
    init(); 
}

inline void s_csg::init()
{
    INSTANCE = this ; 
    s_pa::SetPOOL(pa) ; 
    s_bb::SetPOOL(bb) ; 
    s_tv::SetPOOL(tv) ; 
    sn::SetPOOL(n) ; 
}

inline int s_csg::total_size() const
{
    return pa->size() + bb->size() + tv->size() + n->size() ; 
}

inline std::string s_csg::brief() const
{
    std::stringstream ss ; 
    ss << "s_csg::brief total_size " << total_size() 
       << std::endl  
       << " pa : " << ( pa ? pa->brief() : "-" ) << std::endl
       << " bb : " << ( bb ? bb->brief() : "-" ) << std::endl
       << " tv : " << ( tv ? tv->brief() : "-" ) << std::endl
       << " n : " << ( n ? n->brief() : "-" ) 
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
    fold->add(sn::NAME,  n->serialize<int>() );  
    fold->add(s_bb::NAME, bb->serialize<double>() ); 
    fold->add(s_pa::NAME, pa->serialize<double>() ); 
    fold->add(s_tv::NAME, tv->serialize<double>() ); 
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
    pa->import<double>(fold->get(s_pa::NAME)) ; 
    bb->import<double>(fold->get(s_bb::NAME)) ; 
    tv->import<double>(fold->get(s_tv::NAME)) ; 
    n->import<int>(fold->get(sn::NAME)) ; 
}


