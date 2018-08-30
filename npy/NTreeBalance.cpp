#include <algorithm>
#include <cassert>

#include "OpticksCSG.h"
#include "OpticksCSGMask.h"
#include "NTreeBalance.hpp"
#include "NTreeBuilder.hpp"
#include "PLOG.hh"




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

    OpticksCSG_t op = CSG_MonoOperator(op_mask) ;  
    OpticksCSG_t hop = CSG_MonoOperator(hop_mask) ;  

    T* balanced = NULL ; 

    if( op == CSG_INTERSECTION || op == CSG_UNION ) 
    {
        std::vector<T*> prims ; 
        subtrees( prims, 0 );    // subdepth 0 
        //LOG(info) << " prims " << prims.size() ; 
        balanced = NTreeBuilder<T>::CommonTree(prims, op ); 
    }
    else if( hop == CSG_INTERSECTION || hop == CSG_UNION ) 
    {
        std::vector<T*> bileafs ; 
        subtrees( bileafs, 1 );  // subdepth 1
        //LOG(info) << " bileafs " << bileafs.size() ; 
        balanced = NTreeBuilder<T>::BileafTree(bileafs, hop ); 
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

Collect all subtrees of a particular subdepth.

**/

template <typename T>
void NTreeBalance<T>::subtrees(std::vector<T*>& subs, unsigned subdepth )
{
    subtrees_r( root, subs, subdepth );      
}

template <typename T>
void NTreeBalance<T>::subtrees_r(T* node, std::vector<T*>& subs, unsigned subdepth ) // static
{
    if( node == NULL ) return ; 
    subtrees_r(node->left , subs, subdepth );
    subtrees_r(node->right, subs, subdepth );
    // postorder visit (ie from leftmost visiting children before their parents)
    if(node->subdepth == subdepth) subs.push_back(node) ;
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


