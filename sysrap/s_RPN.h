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

    void add_opernode( unsigned typecode, unsigned leftspan, unsigned rightspan );
    void add_listnode( unsigned typecode, unsigned num_listprim );
    void add_primnode( unsigned typecode, bool complement, bool identity, sparm pa, qat4 tr );

    void upload();
#endif

};


#if defined(__CUDACC__) || defined(__CUDABE__)
#else

inline void s_RPN::add_opernode( unsigned typecode, unsigned leftspan, unsigned rightspan )
{
    srpn nd = {} ;
    nd.set_opernode(typecode, leftspan, rightspan);
    node.push_back(nd);
}

inline void s_RPN::add_listnode( unsigned typecode, unsigned num_listprim )
{
    srpn nd = {} ;
    nd.set_listnode(typecode, num_listprim );
    node.push_back(nd);
}

inline void s_RPN::add_primnode( unsigned typecode, bool complement, bool identity, sparm pa, qat4 it )
{
    assert( parm.size() == itra.size() );
    unsigned index = parm.size();
    parm.push_back(pa);
    itra.push_back(it);

    srpn nd = {} ;
    nd.set_primnode( typecode, complement, identity, index );
    node.push_back(nd);
}

inline void s_RPN::upload()
{
}


#endif
