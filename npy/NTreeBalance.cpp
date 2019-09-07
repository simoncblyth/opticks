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

#include <algorithm>
#include <cassert>

#include "OpticksCSG.h"
#include "OpticksCSGMask.h"
#include "NTreeBalance.hpp"
#include "NTreeBuilder.hpp"
#include "PLOG.hh"

template <typename T>
const plog::Severity NTreeBalance<T>::LEVEL = info ; 


template <typename T>
NTreeBalance<T>::NTreeBalance(T* root_)
    :
    root(root_),
    height0(0)
{
    init(); 
}

template <typename T>
void NTreeBalance<T>::init()
{
    height0 = depth_r(root,0, true);     
    subdepth_r(root,0);     
}


template <typename T>
T* NTreeBalance<T>::create_balanced()
{
    assert( is_positive_form() && " must positivize the tree before balancing "); 

    unsigned op_mask = operators(); 
    unsigned hop_mask = operators(2);  // operators above the bileaf operators


    LOG(LEVEL) << "op_mask " << CSGMaskDesc(op_mask ); 
    LOG(LEVEL) << "hop_mask " << CSGMaskDesc(hop_mask ); 


    OpticksCSG_t op = CSG_MonoOperator(op_mask) ;    // sysrap/OpticksCSGMask.h returns CSG_ZERO when not a mono mask
    OpticksCSG_t hop = CSG_MonoOperator(hop_mask) ;  

    T* balanced = NULL ; 

    if( op == CSG_INTERSECTION || op == CSG_UNION )  // simple mono-operator case : all operators in tree are the same 
    {
        std::vector<T*> prims ; 
        std::vector<T*> otherprim ; 
        subtrees( prims, 0, otherprim );    // subdepth 0 
        LOG(LEVEL) << " CommonTree prims " << prims.size() ; 
        assert( otherprim.size() == 0 );  

        balanced = NTreeBuilder<T>::CommonTree(prims, op ); 
    }
    else if( hop == CSG_INTERSECTION || hop == CSG_UNION ) 
    {
        std::vector<T*> bileafs ; 
        std::vector<T*> otherprim ; 

        subtrees( bileafs, 1, otherprim );  // subdepth 1
        LOG(LEVEL) 
            << " bileafs " << bileafs.size()
            << " otherprim " << otherprim.size() 
            ;

        if( otherprim.size() == 0 )
        {
            balanced = NTreeBuilder<T>::BileafTree(bileafs, hop ); 
        }
        else
        {
            balanced = NTreeBuilder<T>::MixedTree(bileafs, otherprim, hop ); 
        }
        // can also have bileafs + primitives in a mixed tree, like with sFasterner bolts
    }
    else
    {
        LOG(fatal) << "balancing trees of this structure not implemented" ; 
        //assert(0); 
        balanced = root ; 
    }
    return balanced ; 
}


template <typename T>
unsigned NTreeBalance<T>::depth_r(T* node, unsigned depth, bool label )
{
    // cf nnode::maxdepth, this one provides labelling

    if(node == NULL) return depth ; 

    if(label)
    {
        node->depth = depth ; 
    } 

    if(node->left == NULL && node->right == NULL) return depth ; 

    unsigned ldepth = depth_r(node->left,  depth+1, label ); 
    unsigned rdepth = depth_r(node->right, depth+1, label ); 
    return std::max(ldepth, rdepth) ;
}


/** 

NTreeBalance<T>::subdepth_r
------------------------------

Labels the nodes with the subdepth, which is 
the max height of each node treated as a subtree::


               3                    

       1               2            

   0       0       0           1    

                           0       0

**/

template <typename T>
void NTreeBalance<T>::subdepth_r(T* node, unsigned depth_ )
{
     assert(node); 

     node->subdepth = depth_r(node, 0, false) ;

     if(!(node->left == NULL && node->right == NULL)) 
     {
         subdepth_r(node->left,  depth_+1 ); 
         subdepth_r(node->right, depth_+1 ); 
     }
}



/**
NTreeBalance<T>::subtrees
---------------------------

Collect all subtrees of a particular subdepth into *subs*
on first pass.  On second pass collect into *otherprim* 
any primitives that are not already collected  
either directly or as bileaf leaves within subs.

**/

template <typename T>
void NTreeBalance<T>::subtrees(std::vector<T*>& subs, unsigned subdepth, std::vector<T*>& otherprim )
{
    subtrees_r( root, subs, subdepth, otherprim, 0  );
    subtrees_r( root, subs, subdepth, otherprim, 1  );
}

template <typename T>
void NTreeBalance<T>::subtrees_r(T* node, std::vector<T*>& subs, unsigned subdepth, std::vector<T*>& otherprim, unsigned pass ) // static
{
    if( node == NULL ) return ; 
    subtrees_r(node->left , subs, subdepth, otherprim, pass);
    subtrees_r(node->right, subs, subdepth, otherprim, pass);
    // postorder visit (ie from leftmost visiting children before their parents)

    if(pass == 0 ) 
    {
        if(node->subdepth == subdepth) 
        {
            subs.push_back(node) ;
        }
    }
    else if( pass == 1 )   // mop up any other primitives that are not already collected
    {
        if( node->is_primitive() && !is_collected(subs, node) )
        { 
            otherprim.push_back(node); 
        }
    }
}


/**
NTreeBalance<T>::is_collected
--------------------------------

Check if the argument node is already present in the subs list of subtrees 
eitherdirectly or as first child.

NB implicit assumption of maximum depth 1 of the subtrees, ie either leaf or bileaf 

**/

template <typename T>
bool NTreeBalance<T>::is_collected(const std::vector<T*>& subs, const T* node) // static
{
    assert( node->is_primitive() ); 
    for(unsigned i=0 ; i < subs.size() ; i++)
    {
        const T* sub = subs[i] ; 
        if( sub->is_primitive() && sub == node ) return true ; 
        if( sub->is_operator() && sub->left == node ) return true ; 
        if( sub->is_operator() && sub->right == node ) return true ; 
    }
    return false ; 

}



template <typename T>
std::string NTreeBalance<T>::operatorsDesc(unsigned minsubdepth) const 
{
    unsigned ops = operators(minsubdepth);
    return CSGMaskDesc(ops); 
}


template <typename T>
unsigned NTreeBalance<T>::operators(unsigned minsubdepth) const 
{
   unsigned mask = 0 ;  
   NTreeBalance<T>::operators_r(root, mask, minsubdepth);  
   return mask ;  
}

template <typename T>
void NTreeBalance<T>::operators_r(const T* node, unsigned& mask, unsigned minsubdepth) // static
{
    if(node->left && node->right)
    {
        if( node->subdepth >= minsubdepth )
        {
            switch( node->type )
            {
                case CSG_UNION         : mask |= CSGMASK_UNION        ; break ; 
                case CSG_INTERSECTION  : mask |= CSGMASK_INTERSECTION ; break ; 
                case CSG_DIFFERENCE    : mask |= CSGMASK_DIFFERENCE   ; break ; 
                default                : mask |= 0                    ; break ; 
            }
        }
        operators_r( node->left,  mask, minsubdepth ); 
        operators_r( node->right, mask, minsubdepth ); 
    }
}
template <typename T>
bool NTreeBalance<T>::is_positive_form() const 
{
    unsigned ops = operators(); 
    return (ops & CSGMASK_DIFFERENCE) == 0 ; 
}
template <typename T>
bool NTreeBalance<T>::is_mono_form() const // only one type of operator
{
    unsigned ops = operators(); 
    return (ops == CSGMASK_DIFFERENCE) || (ops == CSGMASK_UNION)  || (ops == CSGMASK_INTERSECTION)  ; 
}

#include "No.hpp"
#include "NNode.hpp"

template struct NTreeBalance<no> ; 
template struct NTreeBalance<nnode> ; 


