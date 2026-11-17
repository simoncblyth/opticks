#pragma once
/**
s_RPN.h
========


**/

#include "srpn.h"
#include "sparm.h"
#include "sqat4.h"

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
#include <cassert>
#include <vector>
#include "NPFold.h"
#endif

struct s_RPN
{
#if defined(__CUDACC__) || defined(__CUDABE__)
    srpn*   node ;
    sparm*  parm ;
    qat4*   itra ;
    unsigned num_node ;
    unsigned num_parm ;
    unsigned num_itra ;
#else
    srpn*  d_node ;
    sparm* d_parm ;
    qat4*  d_itra ;

    std::vector<srpn>   node ;
    std::vector<sparm>  parm ;
    std::vector<qat4>   itra ;
    std::vector<int>    lvid ;   // debug only

    bool is_empty() const ;
    void add_opernode( unsigned typecode, unsigned leftspan, unsigned rightspan, int lvid );
    void add_listnode( unsigned typecode, unsigned num_listprim, int lvid );
    void add_primnode( unsigned typecode, bool complement, bool identity, sparm pa, qat4 tr, int lvid );

    NPFold* serialize() const ;
    void import_(const NPFold* fold);

    void save(const char* base, const char* rel) const;
    void load(const char* base, const char* rel);

    void save(const char* dir) const ;
    void load(const char* dir);

    std::string desc() const ;
    void upload();
#endif

};


#if defined(__CUDACC__) || defined(__CUDABE__)
#else

inline bool s_RPN::is_empty() const
{
    return node.size() == 0 && parm.size() == 0 && itra.size() == 0 ;
}

inline void s_RPN::add_opernode( unsigned typecode, unsigned leftspan, unsigned rightspan, int _lvid )
{
    srpn nd = {} ;
    nd.set_opernode(typecode, leftspan, rightspan);
    node.push_back(nd);
    lvid.push_back(_lvid);
}

inline void s_RPN::add_listnode( unsigned typecode, unsigned num_listprim, int _lvid )
{
    srpn nd = {} ;
    nd.set_listnode(typecode, num_listprim );
    node.push_back(nd);
    lvid.push_back(_lvid);
}

inline void s_RPN::add_primnode( unsigned typecode, bool complement, bool identity, sparm pa, qat4 it, int _lvid )
{
    assert( parm.size() == itra.size() );
    unsigned index = parm.size();
    parm.push_back(pa);
    itra.push_back(it);

    srpn nd = {} ;
    nd.set_primnode( typecode, complement, identity, index );
    node.push_back(nd);
    lvid.push_back(_lvid);
}
inline NPFold* s_RPN::serialize() const
{
    NPFold* fold = new NPFold ;
    fold->add("node", NPX::ArrayFromVec<unsigned, srpn>( node, -1 )) ;  // -1:suppress extra dimension
    fold->add("itra", NPX::ArrayFromVec<float,    qat4>( itra, 4, 4 )) ;
    fold->add("parm", NPX::ArrayFromVec<float,   sparm>( parm, 2, 4 )) ;
    fold->add("lvid", NPX::ArrayFromVec<int, int>( lvid, -1 )) ;  // -1:suppress extra dimension
    return fold ;
}
inline void s_RPN::import_(const NPFold* fold)
{
    NPX::VecFromArray<srpn>(  node , fold->get("node")  );
    NPX::VecFromArray<qat4>(  itra , fold->get("itra")  );
    NPX::VecFromArray<sparm>( parm , fold->get("parm")  );
    NPX::VecFromArray<int>(   lvid , fold->get("lvid")  );
}


inline void s_RPN::save(const char* base, const char* rel) const
{
    NPFold* fold = serialize();
    fold->save(base, rel);
}
inline void s_RPN::load(const char* base, const char* rel)
{
    const char* dir = spath::Resolve(base, rel);
    NPFold* fold = NPFold::Load(dir);
    import_(fold);
}

inline void s_RPN::save(const char* dir) const
{
    NPFold* fold = serialize();
    fold->save(dir);
}
inline void s_RPN::load(const char* dir_)
{
    const char* dir = spath::Resolve(dir_);
    NPFold* fold = NPFold::Load(dir);
    import_(fold);
}


inline std::string s_RPN::desc() const
{
    std::stringstream ss;
    ss << "[s_RPN " ;
    ss << " node:" << node.size() << "" ;
    ss << " lvid:" << lvid.size() << "" ;
    ss << " itra:" << itra.size() << "" ;
    ss << " parm:" << parm.size() << "\n" ;

    ss << "-node,lvid[\n" ;
    for(size_t i=0 ; i < node.size() ; i++) ss << node[i].desc() << " lv " << lvid[i] << "\n" ;
    ss << "-node, lvid]\n" ;

    ss << "-itra[\n" ;
    for(size_t i=0 ; i < itra.size() ; i++) ss << itra[i].desc() << "\n" ;
    ss << "-itra]\n" ;

    ss << "-parm[\n" ;
    for(size_t i=0 ; i < parm.size() ; i++) ss << parm[i].desc() << "\n" ;
    ss << "-parm]\n" ;


    ss << "]s_RPN\n" ;
    std::string str = ss.str();
    return str ;
}






inline void s_RPN::upload()
{
}


#endif
