#pragma once

struct nmat4triple ;
template <typename T> class NPY ; 

#include "NPY_API_EXPORT.hh"
/**
nxform
======

Thought this might enable avoiding duplication 
of transform mechanics.  But the users of this
are nodes within a node tree, so its kinda complex
to make this available.

Wanted to use nxform<YOG::Nd> but cannot 
use the normal explicit template instanciation approach 
as do not want NPY to depend on YOG.  Thus 
have to place all the implementation into the 
header, so everything is available at point of use 
within X4PhysicalVolume.

**/


template <typename N>
struct NPY_API nxform
{
    static const nmat4triple* make_global_transform(const N* n)  ;  // for node structs 

    nxform(unsigned num_nodes_, bool debug_);

    const nmat4triple* make_triple( const float* data);
    
    unsigned     num_nodes ; 
    bool         debug ; 
    unsigned     num_triple  ; 
    unsigned     num_triple_mismatch ;
    NPY<float>*  triple ;
  
};


#include "NPY.hpp"
#include "NGLM.hpp"
#include "NGLMExt.hpp"
#include "NGLMCF.hpp"
#include "PLOG.hh"


template <typename N>
nxform<N>::nxform(unsigned num_nodes_, bool debug_)
    :
    num_nodes(num_nodes_),
    debug(debug_),
    num_triple(0),
    num_triple_mismatch(0),
    triple(debug_ ? NPY<float>::make( num_nodes, 3, 4, 4 ) : NULL )
{
} 


template <typename N>
const nmat4triple* nxform<N>::make_triple( const float* data)
{
    // spell out nglmext::invert_trs for debugging discrepancies

    glm::mat4 T = glm::make_mat4(data) ;
    ndeco d ;
    nglmext::polar_decomposition( T, d  ) ; 

    glm::mat4 isirit = d.isirit ;
    glm::mat4 i_trs = glm::inverse( T ) ;

    NGLMCF cf(isirit, i_trs );

    if(!cf.match)
    {
        num_triple_mismatch++ ;
        //LOG(warning) << cf.desc("nd::make_triple polar_decomposition inverse and straight inverse are mismatched " );
    }

    glm::mat4 V = isirit ;
    glm::mat4 Q = glm::transpose(V) ;

    nmat4triple* tvq = new nmat4triple(T, V, Q);

    if(triple)  // collecting triples for mismatch debugging 
    {
        triple->setMat4Triple( tvq , num_triple++ );
    }
    return tvq ;
}


/**
nxform<N>::make_global_transform
-----------------------------------

node structs that can work with this require
transform and parent members   

1. collects nmat4triple pointers whilst following 
   parent links up the tree, ie in leaf-to-root order 

2. returns the reversed product of those 


**/

template <typename N>
const nmat4triple* nxform<N>::make_global_transform(const N* n) // static
{
    std::vector<const nmat4triple*> tvq ; 
    while(n)
    {
        if(n->transform) tvq.push_back(n->transform);
        n = n->parent ; 
    }
    bool reverse = true ; // as tvq in leaf-to-root order
    return tvq.size() == 0 ? NULL : nmat4triple::product(tvq, reverse) ; 
}




