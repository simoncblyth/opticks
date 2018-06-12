#include <sstream>

#include "NNode.hpp"
#include "NTreeBuilder.hpp"
#include "PLOG.hh"

nnode* NTreeBuilder::UnionTree(const std::vector<nnode*>& prims )
{
    return CommonTree(prims, CSG_UNION ) ; 
}

nnode* NTreeBuilder::CommonTree(const std::vector<nnode*>& prims, OpticksCSG_t operator_ )
{
    NTreeBuilder tb(prims, operator_ );  
    LOG(info) << tb.desc(); 
    return tb.root() ; 
}

int NTreeBuilder::FindBinaryTreeHeight(unsigned num_leaves)
{
    /**
    Find complete binary tree height sufficient for nprim leaves
        
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



NTreeBuilder::NTreeBuilder( const std::vector<nnode*>& prims, OpticksCSG_t operator_ )
    :
    m_prims(prims),
    m_height(FindBinaryTreeHeight(prims.size())),
    m_operator(operator_),
    m_placeholder(CSG_ZERO),
    m_root(NULL)
{
    init(); 
} 

std::string NTreeBuilder::desc() const 
{
    std::stringstream ss ; 
    ss 
       << " num_prims " << m_prims.size() 
       << " height " << m_height 
       << " operator " << CSGName(m_operator) 
       ; 
    return ss.str(); 
}


nnode* NTreeBuilder::root()
{
   return m_root ; 
}


void NTreeBuilder::init()
{
    unsigned num_prim = m_prims.size() ; 

    m_cprims = m_prims ; 
    std::reverse( m_cprims.begin(), m_cprims.end() ); 

    for(unsigned i=0 ; i < num_prim ; i++)
    {
        nnode* prim = m_prims[i] ;
        prim->dump(); 
        assert( prim->is_primitive() ); 
    }
    for(unsigned i=0 ; i < num_prim ; i++)
    {
        nnode* prim = m_cprims[i] ;
        prim->dump(); 
        assert( prim->is_primitive() ); 
    }


    if(m_height == 0)
    {
         assert( num_prim == 1 );
         m_root = m_prims[0]; 
    } 
    else
    {
         m_root = build(m_height) ;
         //populate(); 
         //prune();
    }
}


nnode* NTreeBuilder::build( int height )
{
    /*
    Build complete binary tree with all operators the same
    and CSG.ZERO placeholders for elevation 0
    */
    nnode* root = build_r( height ) ; 
    return root ; 
}


nnode* NTreeBuilder::build_r( int elevation )
{
    nnode* node = NULL ; 
    if(elevation > 1)
    {
        nnode* left = build_r( elevation - 1 );
        nnode* right = build_r( elevation - 1 );
        node = new nnode(make_node( m_operator , left , right )); 
    }
    else
    {
        nnode* left =  new nnode(make_node( m_placeholder, NULL, NULL )); 
        nnode* right =  new nnode(make_node( m_placeholder, NULL, NULL )); 
        node = new nnode(make_node( m_operator , left , right )); 
    }
    return node ; 
}


void NTreeBuilder::populate()
{
}

void NTreeBuilder::prune()
{
}


