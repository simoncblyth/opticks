#include <iostream>
#include <sstream>

#include "NNodeCollector.hpp"
#include "NTreeBuilder.hpp"
#include "PLOG.hh"

template <typename T>
T* NTreeBuilder<T>::UnionTree(const std::vector<T*>& prims )  // static
{
    return CommonTree(prims, CSG_UNION ) ; 
}
template <typename T>
T* NTreeBuilder<T>::CommonTree(const std::vector<T*>& prims, OpticksCSG_t operator_ ) // static
{
    NTreeBuilder tb(prims, operator_ );  
    return tb.root() ; 
}
template <typename T>
T* NTreeBuilder<T>::BileafTree(const std::vector<T*>& bileafs, OpticksCSG_t operator_ ) // static
{
    bool bileaf = true ; 
    NTreeBuilder tb(bileafs, operator_, bileaf );  
    return tb.root() ; 
}



/**
Find complete binary tree height sufficient for num_leaves
        
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
        if( tprim >= num_leaves )
        {
           height = h ;
           break ;
        }
    }
    assert(height > -1 ); 
    return height ; 
}

template <typename T>
NTreeBuilder<T>::NTreeBuilder( const std::vector<T*>& subs, OpticksCSG_t operator_, bool bileaf )
    :
    m_subs(subs),
    m_operator(operator_),
    m_optag(CSGTag(operator_)),
    m_bileaf(bileaf),
    m_height(0),
    m_root(NULL),
    m_verbosity(3)
{
    init(); 
} 

template <typename T>
std::string NTreeBuilder<T>::desc() const 
{
    std::stringstream ss ; 
    ss 
       << " num_subs " << m_subs.size() 
       << " height " << m_height 
       << " operator " << CSGName(m_operator) 
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
void NTreeBuilder<T>::init()
{
    for(unsigned i=0 ; i < m_subs.size() ; i++)
    {
        T* sub = m_subs[i] ;
        //if(m_verbosity > 2) std::cout << sub->desc() << std::endl ; 

        if( m_bileaf )
        {
            assert( sub->is_bileaf() );
        }
        else
        { 
            assert( sub->is_primitive() ); 
        } 
    }

    unsigned num_prim = m_bileaf ? 2*m_subs.size() : m_subs.size() ; 

    m_height = FindBinaryTreeHeight(num_prim) ; 

    m_csubs = m_subs ; 

    std::reverse( m_csubs.begin(), m_csubs.end() ); 

    if(m_height == 0)
    {
         assert( num_prim == 1 && m_bileaf == false );
         setRoot( m_subs[0] ); 
    } 
    else
    {
         T* root = build_r( m_bileaf ? m_height - 1 : m_height ) ; // bileaf are 2 levels, so height - 1
         setRoot(root);
         populate(); 
         prune();
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
        node = T::make_operator_ptr(m_operator, left, right ) ; 
    }
    else
    {
        T* left =  new T(T::make_node( CSG_ZERO )); 
        T* right =  new T(T::make_node( CSG_ZERO )); 
        node = T::make_operator_ptr(m_operator, left, right ) ; 
    }
    return node ; 
}


template <typename T>
void NTreeBuilder<T>::populate()
{
    std::vector<T*> inorder ; 
    NNodeCollector<T>::Inorder_r( inorder, m_root ) ;  

    std::vector<T*>& csubs = m_csubs ; 

    for(unsigned i=0 ; i < inorder.size() ; i++)
    {
        T* node = inorder[i]; 

        if(node->is_operator())
        {
            if(node->left->is_zero() && csubs.size() > 0)
            {
               T* back = csubs.back() ;
               //node->left = new T(*back) ; // <-- this caused bbox infinite recursion, as it drops the vtable : see NTreeBuilderTest
               node->left = back->make_copy();

               csubs.pop_back();  // popping destroys it, so need the copy 
            }         
            if(node->right->is_zero() && csubs.size() > 0)
            {
               T* back = csubs.back() ;
               //node->right = new T(*back) ;   // ditto
               node->right = back->make_copy();
               csubs.pop_back(); 
            }         
        }
    }
}

template <typename T>
void NTreeBuilder<T>::prune()
{
    prune_r(m_root);
}

template <typename T>
void NTreeBuilder<T>::prune_r(T* node)
{
    // Pulling leaves partnered with CSG.ZERO up to a higher elevation. 

    if(node == NULL) return ; 
    if(node->is_operator())
    {
        prune_r(node->left);
        prune_r(node->right);

        // postorder visit : so both children always visited before their parents 

        if(node->left->is_lrzero()) // left node is an operator which has both its left and right zero 
        {
            node->left = new T(T::make_node(CSG_ZERO)) ;  // prune : ie replace operator with CSG_ZERO placeholder  
        }
        else if( node->left->is_rzero() ) // left node is an operator with left non-zero and right zero   
        {
            T* ll = node->left->left ; 
            node->left = ll ;        // moving the lonely primitive up to higher elevation   
        }

        if(node->right->is_lrzero())  // right node is operator with both its left and right zero 
        {
            node->right = new T(T::make_node(CSG_ZERO)) ;  // prune
        }
        else if( node->right->is_rzero() ) // right node is operator with its left non-zero and right zero
        {
            T* rl = node->right->left ; 
            node->right = rl ;        // moving the lonely primitive up to higher elevation   
        }
    }
}

#include "NNode.hpp"
#include "No.hpp"
template class NTreeBuilder<nnode> ; 
template class NTreeBuilder<no> ; 

