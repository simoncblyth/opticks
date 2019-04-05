#include <iostream>
#include <sstream>

#include "NNodeCollector.hpp"
#include "NTreeBuilder.hpp"
#include "PLOG.hh"


template <typename T> const plog::Severity NTreeBuilder<T>::LEVEL = error ; 

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
T* NTreeBuilder<T>::UnionTree(const std::vector<T*>& prims )  // static
{
    return CommonTree(prims, CSG_UNION ) ; 
}
template <typename T>
T* NTreeBuilder<T>::CommonTree(const std::vector<T*>& prims, OpticksCSG_t operator_ ) // static
{
    std::vector<T*> otherprim ; 
    NTreeBuilder tb(prims, otherprim, operator_, PRIM  );  
    return tb.root() ; 
}
template <typename T>
T* NTreeBuilder<T>::BileafTree(const std::vector<T*>& bileafs, OpticksCSG_t operator_ ) // static
{
    std::vector<T*> otherprim ; 
    NTreeBuilder tb(bileafs, otherprim, operator_, BILEAF );  
    return tb.root() ; 
}

template <typename T>
T* NTreeBuilder<T>::MixedTree(const std::vector<T*>& bileafs, const std::vector<T*>& otherprim,  OpticksCSG_t operator_ ) // static
{
    NTreeBuilder tb(bileafs, otherprim, operator_, MIXED );  
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
NTreeBuilder<T>::NTreeBuilder( const std::vector<T*>& subs, const std::vector<T*>& otherprim, OpticksCSG_t operator_, NTreeBuilderMode_t mode )
    :
    m_subs(subs),
    m_otherprim(otherprim),
    m_operator(operator_),
    m_optag(CSGTag(operator_)),
    m_mode(mode),
    m_num_prim(0),
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
       << " num_otherprim " << m_otherprim.size() 
       << " num_prim " << m_num_prim
       << " height " << m_height 
       << " mode " << BuilderMode(m_mode)
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

         populate(m_otherprim_copy); 
         populate(m_subs_copy); 

         // see notes/issues/OKX4Test_sFasteners_generalize_tree_balancing.rst 

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

**/

template <typename T>
void NTreeBuilder<T>::populate(std::vector<T*>& src)
{
    std::vector<T*> inorder ; 
    NNodeCollector<T>::Inorder_r( inorder, m_root ) ;  

    for(unsigned i=0 ; i < inorder.size() ; i++)
    {
        T* node = inorder[i]; 

        if(node->is_operator())
        {
            if(node->left->is_zero() && src.size() > 0)
            {
                T* back = src.back() ;
                //node->left = new T(*back) ; // <-- this caused bbox infinite recursion, as it drops the vtable : see NTreeBuilderTest
                node->left = back->make_copy();

                src.pop_back();  // popping destroys it, so need the copy 
            }         
            if(node->right->is_zero() && src.size() > 0)
            {
                T* back = src.back() ;
                //node->right = new T(*back) ;   // ditto
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

#include "NNode.hpp"
#include "No.hpp"
template class NTreeBuilder<nnode> ; 
template class NTreeBuilder<no> ; 

