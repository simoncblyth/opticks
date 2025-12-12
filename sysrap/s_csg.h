#pragma once
/**
s_csg.h : Manage persisting of CSG constituent nodes
======================================================

Holds node tree info, param, bbox, transforms of CSG constituent nodes.

NB: Restrict this to handling persisting only.
Doing other things here such as node finding
leads to circular dependencies.

Note that this just does persisting managed from _csg member of stree,
other than very low level accessors access to sn nodes is done
via sn.h methods such as::

   sn::GetLVNodes



**/

#include "ssys.h"

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
    static void Load(const char* base) ;
    static std::string Desc();

    int level ;
    s_pa::POOL* pa ;
    s_bb::POOL* bb ;
    s_tv::POOL* tv ;
    sn::POOL*   nd ;

    s_csg();
    void init();

    int total_size() const ;
    std::string brief() const ;
    std::string desc() const ;

    NPFold* serialize() const ;
    void import(const NPFold* fold);

    int num_pa() const ;
    int num_bb() const ;
    int num_tv() const ;
    int num_nd() const ;

    // low level accessors by index
    s_pa* get_pa(int idx) const ;
    s_bb* get_bb(int idx) const ;
    s_tv* get_tv(int idx) const ;
    sn*   get_nd(int idx) const ;

};


inline s_csg* s_csg::Get() { return INSTANCE ; }

inline NPFold* s_csg::Serialize()
{
    assert(INSTANCE);
    return INSTANCE ? INSTANCE->serialize() : nullptr ;
}

inline void s_csg::Import(const NPFold* fold)
{
    if(INSTANCE == nullptr) new s_csg ;
    assert( INSTANCE ) ;

    int tot = INSTANCE->total_size() ;
    if(tot != 0) std::cerr << INSTANCE->brief() ;

    assert( tot == 0 && "s_csg::Import into an already populated pool is not supported" );
    INSTANCE->import(fold);
}

inline void s_csg::Load(const char* base)
{
    NPFold* fold = NPFold::Load(base);
    Import(fold);
}

inline std::string s_csg::Desc()
{
    std::stringstream ss ;
    ss << "s_csg::Desc INSTANCE " << ( INSTANCE ? "YES" : "NO " ) << std::endl ;
    if(INSTANCE) ss << INSTANCE->brief() << std::endl ;
    if(INSTANCE) ss << INSTANCE->desc() << std::endl ;
    std::string str = ss.str();
    return str ;
}




inline s_csg::s_csg()
    :
    level(ssys::getenvint("s_csg_level",0)),
    pa(new s_pa::POOL("pa")),
    bb(new s_bb::POOL("bb")),
    tv(new s_tv::POOL("tv")),
    nd(new   sn::POOL("nd"))
{
    if(level > 0) std::cerr << "s_csg::s_csg s_csg_level : " << level << std::endl ;
    init();
}

inline void s_csg::init()
{
    INSTANCE = this ;
    s_pa::SetPOOL(pa) ;
    s_bb::SetPOOL(bb) ;
    s_tv::SetPOOL(tv) ;
    sn::SetPOOL(nd) ;
}

inline int s_csg::total_size() const
{
    return num_pa() + num_bb() + num_tv() + num_nd() ;
}

inline int s_csg::num_pa() const { return pa->size() ; }
inline int s_csg::num_bb() const { return bb->size() ; }
inline int s_csg::num_tv() const { return tv->size() ; }
inline int s_csg::num_nd() const { return nd->size() ; }

inline std::string s_csg::brief() const
{
    std::stringstream ss ;
    ss << "s_csg::brief"
       << " total_size " << total_size()
       << " num_pa " << num_pa()
       << " num_bb " << num_bb()
       << " num_tv " << num_tv()
       << " num_nd " << num_nd()
       << std::endl
       << " pa : " << ( pa ? pa->brief() : "-" ) << std::endl
       << " bb : " << ( bb ? bb->brief() : "-" ) << std::endl
       << " tv : " << ( tv ? tv->brief() : "-" ) << std::endl
       << " nd : " << ( nd ? nd->brief() : "-" )
       << std::endl
       ;
    std::string str = ss.str();
    return str ;
}

inline std::string s_csg::desc() const
{
    std::stringstream ss ;
    ss << "s_csg::desc total_size " << total_size()  ;
    ss << ( pa ? pa->desc() : "-" ) << std::endl ;
    ss << ( bb ? bb->desc() : "-" ) << std::endl ;
    ss << ( tv ? tv->desc() : "-" ) << std::endl ;
    ss << ( nd ? nd->desc() : "-" ) << std::endl ;
    std::string str = ss.str();
    return str ;
}

/**
s_csg::serialize
-----------------

In preparation for serialization sn::PrepareToSerialize is invoked
which sets identity transforms on any sn nodes without transforms.
See sn_test::Serialize_Import for details.

**/


inline NPFold* s_csg::serialize() const
{
    sn::PrepareToSerialize();

    if(level > 0) std::cerr << "s_csg::serialize s_csg_level : " << level << std::endl ;
    NPFold* fold = new NPFold ;
    fold->add(  sn::NAME, nd->serialize<int>() );
    fold->add(s_bb::NAME, bb->serialize<double>() );
    fold->add(s_pa::NAME, pa->serialize<double>() );
    fold->add(s_tv::NAME, tv->serialize<double>() );
    return fold ;
}



/**
s_csg::import
--------------

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
    if(level > 0) std::cerr
        << "s_csg::import "
        << " s_csg_level " << level
        << " fold " << ( fold ? fold->desc() : "-" )
        << std::endl
        ;

    pa->import<double>(fold->get(s_pa::NAME)) ;
    bb->import<double>(fold->get(s_bb::NAME)) ;
    tv->import<double>(fold->get(s_tv::NAME)) ;
    nd->import<int>(   fold->get(  sn::NAME)) ;
}


inline s_pa* s_csg::get_pa(int idx) const { return pa->getbyidx(idx) ; }
inline s_bb* s_csg::get_bb(int idx) const { return bb->getbyidx(idx) ; }
inline s_tv* s_csg::get_tv(int idx) const { return tv->getbyidx(idx) ; }
inline sn*   s_csg::get_nd(int idx) const { return nd->getbyidx(idx) ; }


