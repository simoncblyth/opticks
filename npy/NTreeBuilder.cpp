/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

#include <iostream>
#include <csignal>
#include <sstream>

#include "NNodeCollector.hpp"
#include "NTreeBuilder.hpp"
#include "NTreeAnalyse.hpp"
#include "PLOG.hh"


template <typename T> const plog::Severity NTreeBuilder<T>::LEVEL = PLOG::EnvLevel("NTreeBuilder", "DEBUG")  ; 

template <typename T> const char* NTreeBuilder<T>::PRIM_ = "PRIM" ; 
template <typename T> const char* NTreeBuilder<T>::BILEAF_ = "BILEAF" ; 
template <typename T> const char* NTreeBuilder<T>::MIXED_ = "MIXED" ; 
template <typename T> const char* NTreeBuilder<T>::BuilderMode( NTreeBuilderMode_t mode )
{
    const char* s = NULL ;
    switch( mode )
    {
        case PRIM:   s = PRIM_ ; break ; 
        case BILEAF: s = BILEAF_ ; break ; 
        case MIXED:  s = MIXED_ ; break ; 
    }
    return s ; 
}


template <typename T>
T* NTreeBuilder<T>::UnionTree(const std::vector<T*>& prims, bool dump )  // static
{
    return CommonTree(prims, CSG_UNION, dump ) ; 
}
template <typename T>
T* NTreeBuilder<T>::CommonTree(const std::vector<T*>& prims, OpticksCSG_t operator_, bool dump ) // static
{
    std::vector<T*> otherprim ; 
    NTreeBuilder tb(prims, otherprim, operator_, PRIM, dump);  
    return tb.root() ; 
}
template <typename T>
T* NTreeBuilder<T>::BileafTree(const std::vector<T*>& bileafs, OpticksCSG_t operator_, bool dump ) // static
{
    std::vector<T*> otherprim ; 
    NTreeBuilder tb(bileafs, otherprim, operator_, BILEAF, dump );  
    return tb.root() ; 
}

template <typename T>
T* NTreeBuilder<T>::MixedTree(const std::vector<T*>& bileafs, const std::vector<T*>& otherprim,  OpticksCSG_t operator_, bool dump ) // static
{
    NTreeBuilder tb(bileafs, otherprim, operator_, MIXED, dump );  
    return tb.root() ; 
}


/**
NTreeBuilder<T>::FindBinaryTreeHeight
---------------------------------------

Return complete binary tree height sufficient for num_leaves
        
   height: 0, 1, 2, 3,  4,  5,  6,   7,   8,   9,   10, 
   tprim : 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 

**/
template <typename T>
int NTreeBuilder<T>::FindBinaryTreeHeight(unsigned num_leaves) // static
{
    int  height = -1 ;
    for(int h=0 ; h < 10 ; h++ )
    {
        int tprim = 1 << h ;   
        if( tprim >= int(num_leaves) )
        {
           height = h ;
           break ;
        }
    }
    assert(height > -1 ); 
    return height ; 
}

template <typename T>
NTreeBuilder<T>::NTreeBuilder( const std::vector<T*>& subs, const std::vector<T*>& otherprim, OpticksCSG_t operator_, NTreeBuilderMode_t mode, bool dump )
    :
    m_subs(subs),
    m_otherprim(otherprim),
    m_operator(operator_),
    m_optag(CSG::Tag(operator_)),
    m_mode(mode),
    m_num_prim(0),
    m_height(0),
    m_root(NULL),
    m_verbosity(3),
    m_dump(dump)
{
    init(); 
} 




template <typename T>
std::string NTreeBuilder<T>::desc() const 
{
    std::stringstream ss ; 
    ss 
       << " num_subs " << m_subs.size() 
       << " num_otherprim " << m_otherprim.size() 
       << " num_prim " << m_num_prim
       << " height " << m_height 
       << " mode " << BuilderMode(m_mode)
       << " operator " << CSG::Name(m_operator) 
       << " dump " << m_dump
       ; 
    return ss.str(); 
}


template <typename T>
T* NTreeBuilder<T>::root() const 
{
   return m_root ; 
}

template <typename T>
void NTreeBuilder<T>::setRoot(T* root)
{
    m_root = root ; 
}


template <typename T>
unsigned NTreeBuilder<T>::getNumPrim() const 
{
    unsigned num_prim = 0 ; 
    switch(m_mode) 
    {
        case PRIM  : num_prim = m_subs.size()     ; break ;  
        case BILEAF: num_prim = m_subs.size() * 2 ; break ;  
        case MIXED : num_prim = m_subs.size() * 2 + m_otherprim.size() ; break ;  
    }
    return num_prim ; 
}


template <typename T>
void NTreeBuilder<T>::init()
{
    if(m_mode == PRIM || m_mode == BILEAF )
    {
        assert( m_otherprim.size() == 0 ) ; 
    }

    for(unsigned i=0 ; i < m_subs.size() ; i++)
    {
        T* sub = m_subs[i] ;

        if( m_mode == PRIM  )
        {
            assert( sub->is_primitive() ); 
        }
        else if( m_mode == BILEAF )
        { 
            assert( sub->is_bileaf() );
        }
    }

    m_num_prim = getNumPrim() ; 
    m_height = FindBinaryTreeHeight(m_num_prim) ; 

    //std::raise(SIGINT);   // from  X4CSG::GenerateTest X4Solid::Balance 
    LOG(LEVEL) << desc() ; 

    m_subs_copy = m_subs ; 
    std::reverse( m_subs_copy.begin(), m_subs_copy.end() ); 

    m_otherprim_copy = m_otherprim ; 
    std::reverse( m_otherprim_copy.begin(), m_otherprim_copy.end() ); 


    if(m_height == 0)
    {
         assert( m_num_prim == 1 && m_mode == PRIM );
         setRoot( m_subs[0] ); 
    } 
    else if( m_mode == PRIM || m_mode == BILEAF )
    {
         T* root = build_r( m_mode == BILEAF ? m_height - 1 : m_height ) ; 
         // bileaf are 2 levels, so height - 1  
         // HMM: would prune get rid of extra levels anyhow ? 
         setRoot(root);

         populate(m_subs_copy); 
         prune();
    }
    else if( m_mode == MIXED )
    {
         T* root = build_r( m_height ) ; 
         setRoot(root);
    
         if(m_dump) LOG(LEVEL) << "MIXED before populate \n" << NTreeAnalyse<T>::Desc(m_root) ; 
         populate(m_otherprim_copy); 
         populate(m_subs_copy); 

         if(m_dump) LOG(LEVEL) << "MIXED after populate \n" << NTreeAnalyse<T>::Desc(m_root) ; 

         // see notes/issues/OKX4Test_sFasteners_generalize_tree_balancing.rst 
         prune();

         if(m_dump) LOG(LEVEL) << "MIXED after prune \n" << NTreeAnalyse<T>::Desc(m_root) ; 

         rootprune(); 
         if(m_dump) LOG(LEVEL) << "MIXED after rootprune \n" << NTreeAnalyse<T>::Desc(m_root) ; 
    }


}


/**
NTreeBuilder<T>::build_r
--------------------------
    
Build complete binary tree with all operators the same
and CSG_ZERO placeholders for elevation 0

**/

template <typename T>
T* NTreeBuilder<T>::build_r( int elevation )
{
    T* node = NULL ; 
    if(elevation > 1)
    {
        T* left = build_r( elevation - 1 );
        T* right = build_r( elevation - 1 );
        node = T::make_operator(m_operator, left, right ) ; 
    }
    else
    {
        T* left =  T::make_node( CSG_ZERO ); 
        T* right =  T::make_node( CSG_ZERO ); 
        node = T::make_operator(m_operator, left, right ) ; 
    }
    return node ; 
}

/**
NTreeBuilder<T>::populate()
----------------------------

Iterate over a canned inorder sequence of the tree 
(tree starts mono-operator with CSG_ZERO leaves).

When reach bileaf level with placeholder in left slot (node->left->is_zero())
pop a node off the back of src, make a copy and refererence 
it from the tree.

Ditto for right slot. 

Initially created new nodes with eg::

     node->left = new T(*back)  

But that caused bbox infinite recursion, as it drops the vtable : see NTreeBuilderTest
Solution was to make_copy instead.

**/

template <typename T>
void NTreeBuilder<T>::populate(std::vector<T*>& src)
{
    std::vector<T*> inorder ; 
    NNodeCollector<T>::Inorder_r( inorder, m_root ) ;  
    LOG(LEVEL) << " inorder.size " << inorder.size() ; 

    for(unsigned i=0 ; i < inorder.size() ; i++)
    {
        T* node = inorder[i]; 

        if(node->is_operator())
        {
            if(node->left->is_zero() && src.size() > 0)
            {
                T* back = src.back() ;
                node->left = back->make_copy();
                src.pop_back();  // popping destroys it, so need the copy 
            }         
            if(node->right->is_zero() && src.size() > 0)
            {
                T* back = src.back() ;
                node->right = back->make_copy();
                src.pop_back(); 
            }         
        }
    }
}


/**
NTreeBuilder<T>::prune()
-------------------------

Pulls leaves partnered with CSG.ZERO up to a higher elevation. 

**/

template <typename T>
void NTreeBuilder<T>::prune()
{
    prune_r(m_root);
}

template <typename T>
void NTreeBuilder<T>::prune_r(T* node)
{
    if(node == NULL) return ; 
    if(node->is_operator())
    {
        prune_r(node->left);
        prune_r(node->right);

        // postorder visit : so both children always visited before their parents 

        if(node->left->is_lrzero()) // left node is an operator which has both its left and right zero 
        {
            node->left = T::make_node(CSG_ZERO) ;  // prune : ie replace operator with CSG_ZERO placeholder  
        }
        else if( node->left->is_rzero() ) // left node is an operator with left non-zero and right zero   
        {
            T* ll = node->left->left ; 
            node->left = ll ;        // moving the lonely primitive up to higher elevation   
        }

        if(node->right->is_lrzero())  // right node is operator with both its left and right zero 
        {
            node->right = T::make_node(CSG_ZERO) ;  // prune
        }
        else if( node->right->is_rzero() ) // right node is operator with its left non-zero and right zero
        {
            T* rl = node->right->left ; 
            node->right = rl ;        // moving the lonely primitive up to higher elevation   
        }
    }
}

/**
rootprune
-------------

See notes/issues/deep-tree-balancing-unexpected-un-ze.rst 

MIXED tree pruning sometimes leaves a hanging lonely placeholder "ze" off the root  

**/

template <typename T>
void NTreeBuilder<T>::rootprune()
{
    T* node = root(); 

    if(!node->is_operator()) return ; 

    if(node->left->is_operator() && node->right->is_zero() )
    {
        if(m_dump) LOG(LEVEL) << "promoting root.left to root " ; 
        setRoot( node->left ); 
    } 
}



#include "NNode.hpp"
#include "No.hpp"
template class NTreeBuilder<nnode> ; 
template class NTreeBuilder<no> ; 

