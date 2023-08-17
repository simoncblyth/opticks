#pragma once

#include "ssys.h"

#include "s_pa.h"
#include "s_bb.h"
#include "s_tv.h"
#include "sn.h"

#include "NPFold.h"
#include "SYSRAP_API_EXPORT.hh"

struct s_csg_find_lvid
{
    int lvid ; 
    s_csg_find_lvid(int q_lvid) : lvid(q_lvid) {}   
    bool operator()(const sn* n){ return lvid == n->lvid ; }  
};

struct s_csg_find_lvid_root
{
    int lvid ; 
    s_csg_find_lvid_root(int q_lvid) : lvid(q_lvid) {}   
    bool operator()(const sn* n){ return lvid == n->lvid && n->is_root() ; }  
};

struct SYSRAP_API s_csg
{
    static s_csg* INSTANCE ; 
    static s_csg* Get(); 
    static NPFold* Serialize(); 
    static void Import(const NPFold* fold); 
    static void Load(const char* base) ; 

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


    void find_lvid( std::vector<sn*>& nds, int lvid ) const ; 
    sn*  find_lvid_root(int lvid) const ; 

    static void FindLVID(std::vector<sn*>& nds, int lvid); 
    static sn*  FindLVIDRoot(int lvid); 

    static std::string Desc(const std::vector<sn*>& nds); 
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




inline s_csg::s_csg()
    :
    level(ssys::getenvint("s_csg_level",0)),
    pa(new s_pa::POOL("pa")),
    bb(new s_bb::POOL("bb")),
    tv(new s_tv::POOL("tv")),
    nd(new   sn::POOL("nd"))
{
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
    return pa->size() + bb->size() + tv->size() + nd->size() ; 
}

inline std::string s_csg::brief() const
{
    std::stringstream ss ; 
    ss << "s_csg::brief total_size " << total_size() 
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



inline NPFold* s_csg::serialize() const   
{
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

inline void s_csg::find_lvid(std::vector<sn*>& nds, int lvid) const 
{
    s_csg_find_lvid flv(lvid); 
    nd->find(nds, flv );   
} 
inline sn* s_csg::find_lvid_root(int lvid ) const 
{
    std::vector<sn*> nds ; 
    s_csg_find_lvid_root flvr(lvid); 
    nd->find(nds, flvr );   
    int count = nds.size() ; 
    assert( count <= 1 ); 
    return count == 1 ? nds[0] : nullptr ; 
}

inline void s_csg::FindLVID(std::vector<sn*>& nds, int lvid) // static
{
    assert(INSTANCE); 
    INSTANCE->find_lvid(nds, lvid); 
}
inline sn* s_csg::FindLVIDRoot(int lvid) // static
{
    assert(INSTANCE); 
    return INSTANCE->find_lvid_root(lvid); 
}


inline std::string s_csg::Desc(const std::vector<sn*>& nds) // static
{
    std::stringstream ss ; 
    ss << "s_csg::Desc nds.size " << nds.size() << std::endl ; 
    for(int i=0 ; i < int(nds.size()) ; i++) 
    {
        const sn* n = nds[i] ; 
        ss << n->desc() << std::endl ;  
    }
    std::string str = ss.str();
    return str ; 
}



/**
s_csg::getLVRoot
----------------

Returns first *snd* node for *lvid* with snd::is_root true


const sn* s_csg::getLVRoot(int lvid ) const 
{
    const sn* root = nullptr ; 

    int count = 0 ; 
    int num_node = node.size(); 
    for(int i=0 ; i < num_node ; i++)
    {
        const snd& nd = node[i] ; 
        if(nd.lvid == lvid && nd.is_root()) 
        {
            root = &nd ; 
            count += 1 ; 
        }
    }
    assert( count <= 1 ); 
    return root ; 
}

**/



