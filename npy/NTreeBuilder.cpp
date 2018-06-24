#include <iostream>
#include <sstream>

#include "NNodeAnalyse.hpp"
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
    //LOG(info) << tb.desc(); 
    return tb.root() ; 
}

template <typename T>
int NTreeBuilder<T>::FindBinaryTreeHeight(unsigned num_leaves) // static
{
    /**
    Find complete binary tree height sufficient for num_leaves
        
      height: 0, 1, 2, 3,  4,  5,  6,   7,   8,   9,   10, 
      tprim : 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 

    **/

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
NTreeBuilder<T>::NTreeBuilder( const std::vector<T*>& prims, OpticksCSG_t operator_ )
    :
    m_prims(prims),
    m_height(FindBinaryTreeHeight(prims.size())),
    m_operator(operator_),
    m_optag(CSGTag(operator_)),
    m_root(NULL),
    m_ana(NULL),
    m_verbosity(1)
{
    init(); 
} 

template <typename T>
std::string NTreeBuilder<T>::desc() const 
{
    std::stringstream ss ; 
    ss 
       << " num_prims " << m_prims.size() 
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
void NTreeBuilder<T>::analyse()
{
    delete m_ana ; 
    m_ana  = new NNodeAnalyse<T>(m_root) ; 

    if(m_verbosity > 2 )
        LOG(info) << " NNodeAnalyse \n" << m_ana->desc(); 
}


template <typename T>
void NTreeBuilder<T>::init()
{
    unsigned num_prim = m_prims.size() ; 

    m_cprims = m_prims ; 
    std::reverse( m_cprims.begin(), m_cprims.end() ); 

    for(unsigned i=0 ; i < num_prim ; i++)
    {
        T* prim = m_prims[i] ;
        if(m_verbosity > 2)
            std::cout << prim->desc() << std::endl ; 
        assert( prim->is_primitive() ); 
    }

    if(m_height == 0)
    {
         assert( num_prim == 1 );
         setRoot( m_prims[0] ); 
    } 
    else
    {
         T* root = build_r(m_height) ;
         setRoot(root);
         analyse();

         populate(); 
         analyse();

         prune();
         analyse();
    }
}



template <typename T>
T* NTreeBuilder<T>::build_r( int elevation )
{
    /*
    Build complete binary tree with all operators the same
    and CSG_ZERO placeholders for elevation 0
    */

    T* node = NULL ; 
    if(elevation > 1)
    {
        T* left = build_r( elevation - 1 );
        T* right = build_r( elevation - 1 );
        //node = new T(T::make_node( m_operator , left , right )); 
        node = T::make_operator_ptr(m_operator, left, right ) ; 
    }
    else
    {
        T* left =  new T(T::make_node( CSG_ZERO )); 
        T* right =  new T(T::make_node( CSG_ZERO )); 
        //node = new T(T::make_node( m_operator , left , right )); 
        node = T::make_operator_ptr(m_operator, left, right ) ; 
    }
    return node ; 
}


template <typename T>
void NTreeBuilder<T>::populate()
{
    assert(m_ana);
    const std::vector<T*>& inorder = m_ana->nodes->inorder ;     
    std::vector<T*>& cprims = m_cprims ; 

    for(unsigned i=0 ; i < inorder.size() ; i++)
    {
        T* node = inorder[i]; 
        if(node->is_operator())
        {
            if(node->left->is_zero() && cprims.size() > 0)
            {
               T* back = cprims.back() ;
               //node->left = new T(*back) ; // <-- this caused bbox infinite recursion, as it drops the vtable : see NTreeBuilderTest
               node->left = back->make_copy();

               cprims.pop_back();  // popping destroys it, so need the copy 
            }         
            if(node->right->is_zero() && cprims.size() > 0)
            {
               T* back = cprims.back() ;
               //node->right = new T(*back) ;   // ditto
               node->right = back->make_copy();
               cprims.pop_back(); 
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

