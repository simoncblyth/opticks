#pragma once
/**
sn.h : minimal pointer based transient binary tree node
========================================================

* used from sndtree.h 

Motivation
-----------

In order to duplicate at CSG/CSGNode level the old workflow geometry 
(that goes thru GGeo/NNode) it is necessary to perform binary tree 
manipulations equivalent to those done by npy/NTreeBuilder::UnionTree in order 
to handle shapes such as G4Polycone. 

However the array based *snd/scsg* node approach with integer index addressing 
lacks the capability to easily delete nodes making it unsuitable
for tree manipulations such as pruning and rearrangement that are needed 
in order to flexibly create complete binary trees with any number of leaf nodes.

Hence the *sn* nodes are developed to transiently act as a template 
for binary trees that are subsequently solidified into *snd* trees. 
In this way the initial tree setup is done using the very flexible 
pointer based *sn*. The *sn* tree is then used as a guide for the creation 
of the less flexible (from creation/deletion point of view) *snd* 
tree that gets persisted.  

Future
--------

Hopefully this functionality can be removed once have leaped
to CSG_CONTIGUOUS use as standard, which retains n-ary tree 
all the way to the GPU. 


**/

#include <vector>
#include <sstream>
#include <cassert>

#include "scanvas.h"

struct sn
{
    int t ; 
    sn* l ; 
    sn* r ;     

    static sn* Prim(int type) ; 
    static sn* Zero() ; 

    bool is_primitive() const ; 
    bool is_bileaf() const ; 
    bool is_operator() const ; 
    bool is_zero() const ; 

    bool is_lrzero() const ;  //  l-zero AND  r-zero
    bool is_rzero() const ;   // !l-zero AND  r-zero
    bool is_lzero() const ;   //  l-zero AND !r-zero


    int num_node() const ; 
    int num_node_r(int d) const ; 

    int num_leaf() const ; 
    int num_leaf_r(int d) const ; 

    int max_depth() const ; 
    int max_depth_r(int d) const ; 


    void postorder(std::vector<const sn*>& order ) const ; 
    void postorder_r(std::vector<const sn*>& order, int d ) const ; 

    void inorder(std::vector<const sn*>& order ) const ; 
    void inorder_r(std::vector<const sn*>& order, int d ) const ; 

    void inorder_(std::vector<sn*>& order ) ; 
    void inorder_r_(std::vector<sn*>& order, int d ); 




    std::string desc_order(const std::vector<const sn*>& order ) const ; 
    std::string desc() const ; 
    std::string render() const ; 
    void render_r(scanvas* canvas, std::vector<const sn*>& order, int mode, int d) const ; 

    static sn* Build_r(int elevation, int op); 

    static int BinaryTreeHeight(int num_leaves); 
    static int BinaryTreeHeight_1(int num_leaves); 

    static sn* CommonTree(int num_leaves, int op ); 
    static sn* CommonTree( std::vector<int>& leaftypes,  int op ); 

    static void Populate(sn* root, std::vector<int>& leaftypes  ); 

    void prune(); 
    static void Prune_r(sn* n, int d); 
    static void Check(const sn* n); 

};



inline sn* sn::Prim(int type)   // static
{
    return new sn {type, nullptr, nullptr} ; 
}
inline sn* sn::Zero()   // static
{
    return Prim(0); 
}




inline bool sn::is_primitive() const
{   
    return l == nullptr && r == nullptr ;
}   
inline bool sn::is_bileaf() const 
{   
    return !is_primitive() && l->is_primitive() && r->is_primitive() ;
}   
inline bool sn::is_operator() const 
{   
    return l != nullptr && r != nullptr ;
}
inline bool sn::is_zero() const 
{   
    return t == 0 ;  
}
inline bool sn::is_lrzero() const 
{   
    return is_operator() && l->is_zero() && r->is_zero() ;
}
inline bool sn::is_rzero() const
{
    return is_operator() && !l->is_zero() && r->is_zero() ; 
}
inline bool sn::is_lzero() const 
{   
    return is_operator() && l->is_zero() && !r->is_zero() ;
}







inline int sn::num_node() const
{
    return num_node_r(0);
}
inline int sn::num_node_r(int d) const
{
    int nn = 1 ;   // always at least 1 node,  no exclusion of CSG_ZERO
    nn += l ? l->num_node_r(d+1) : 0 ; 
    nn += r ? r->num_node_r(d+1) : 0 ; 
    return nn ;
}


inline int sn::num_leaf() const
{
    return num_leaf_r(0);
}
inline int sn::num_leaf_r(int d) const
{
    int nl = is_primitive() ? 1 : 0 ; 
    nl += l ? l->num_leaf_r(d+1) : 0 ; 
    nl += r ? r->num_leaf_r(d+1) : 0 ; 
    return nl ;
}







inline int sn::max_depth() const
{
    return max_depth_r(0);
}
inline int sn::max_depth_r(int d) const
{
    return l && r ? std::max( l->max_depth_r(d+1), r->max_depth_r(d+1)) : d ; 
}



inline void sn::postorder(std::vector<const sn*>& order ) const
{
    postorder_r(order, 0);
}
inline void sn::postorder_r(std::vector<const sn*>& order, int d ) const
{
    if(l) l->postorder_r(order, d+1) ; 
    if(r) r->postorder_r(order, d+1) ; 
    order.push_back(this); 
}


inline void sn::inorder(std::vector<const sn*>& order ) const
{
    inorder_r(order, 0);
}
inline void sn::inorder_r(std::vector<const sn*>& order, int d ) const
{
    if(l) l->inorder_r(order, d+1) ; 
    order.push_back(this); 
    if(r) r->inorder_r(order, d+1) ; 
}

inline void sn::inorder_(std::vector<sn*>& order )
{
    inorder_r_(order, 0);
}
inline void sn::inorder_r_(std::vector<sn*>& order, int d )
{
    if(l) l->inorder_r_(order, d+1) ; 
    order.push_back(this); 
    if(r) r->inorder_r_(order, d+1) ; 
}


inline std::string sn::desc_order(const std::vector<const sn*>& order ) const 
{
    std::stringstream ss ;
    ss << "sn::desc_order [" ; 
    for(int i=0 ; i < int(order.size()) ; i++)
    {
        const sn* n = order[i] ; 
        ss << n->t << " " ;  
    }
    ss << "]" ; 
    std::string str = ss.str();
    return str ;
}


inline std::string sn::desc() const
{
    std::stringstream ss ;
    ss << "sn::desc"
       << " num_node " << num_node() 
       << " num_leaf " << num_leaf() 
       << " max_depth " << max_depth() 
       ; 
    std::string str = ss.str();
    return str ;
}


inline std::string sn::render() const
{
    int width = num_node();
    int height = max_depth();

    std::vector<const sn*> in ;
    inorder(in);
    assert( int(in.size()) == width );

    std::vector<const sn*> post ;
    postorder(post);
    assert( int(post.size()) == width );


    int mode = width > 16 ? 0 : 1 ; // compact presentation for large trees

    int xscale = 3 ; 
    int yscale = 2 ; 

    if(mode == 1)
    {
        xscale = 8 ; 
        yscale = 4 ; 
    } 

    scanvas canvas( width+1, height+2, xscale, yscale );
    render_r(&canvas, in, mode,  0);

    std::stringstream ss ;
    ss << std::endl ;
    ss << desc() << std::endl ;  
    ss << "sn::render mode " << mode << std::endl ;
    ss << canvas.c << std::endl ;

    ss << "inorder   " << desc_order(in) << std::endl ; 
    ss << "postorder " << desc_order(post) << std::endl ; 

    std::string str = ss.str();
    return str ;
}

void sn::render_r(scanvas* canvas, std::vector<const sn*>& order, int mode, int d) const
{
    int ordinal = std::distance( order.begin(), std::find(order.begin(), order.end(), this )) ;
    assert( ordinal < int(order.size()) );

    int ix = ordinal ;
    int iy = d ;

    if(mode == 0)  // compact single char presentation 
    {
        char l0 = 'o' ;
        canvas->drawch( ix, iy, 0,0,  l0 );
    }
    else if(mode == 1)  // typecode presentation 
    {
        canvas->draw( ix, iy, 0,0,  t );
    } 


    if(l) l->render_r( canvas, order, mode, d+1 );
    if(r) r->render_r( canvas, order, mode, d+1 );
}



inline sn* sn::Build_r( int elevation, int op )  // static
{
    sn* l = elevation > 1 ? Build_r( elevation - 1 , op ) : sn::Zero() ; 
    sn* r = elevation > 1 ? Build_r( elevation - 1 , op ) : sn::Zero() ; 
    return new sn { op, l, r } ;  
}


/**
sn::BinaryTreeHeight
---------------------

Return complete binary tree height sufficient for num_leaves
        
   height: 0, 1, 2, 3,  4,  5,  6,   7,   8,   9,   10, 
   tprim : 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 


                          1                                  h=0,  1    

            10                        11                     h=1,  2 

      100         101          110            111            h=2,  4 

   1000 1001  1010  1011   1100   1101     1110  1111        h=3,  8
        

**/

inline int sn::BinaryTreeHeight(int q_leaves )
{
    int h = 0 ; 
    while( (1 << h) < q_leaves )  h += 1 ; 
    return h ; 
}

inline int sn::BinaryTreeHeight_1(int q_leaves )
{
    int  height = -1 ;
    for(int h=0 ; h < 10 ; h++ )
    {
        int tprim = 1 << h ;
        if( tprim >= q_leaves )
        {
           height = h ;
           break ;
        }
    }
    return height ; 
}

inline sn* sn::CommonTree( int num_leaves, int op ) // static
{   
    int height = BinaryTreeHeight(num_leaves) ;
    return Build_r( height, op );
}          


inline sn* sn::CommonTree( std::vector<int>& leaftypes, int op ) // static
{   
    int num_leaves = leaftypes.size(); 
    sn* root = nullptr ; 
    if( num_leaves == 1 )
    {
        root = sn::Prim(leaftypes[0]) ; 
    }
    else
    {
        root = CommonTree(num_leaves, op );   
        Populate(root, leaftypes); 
        root->prune(); 
        Check(root);  
    }
    return root ; 
} 

        

inline void sn::Populate(sn* root, std::vector<int>& leaftypes )
{
    int num_leaves = leaftypes.size(); 
    int num_leaves_placed = 0 ; 

    std::vector<sn*> order ; 
    root->inorder_(order) ; 

    int num_nodes = order.size(); 

    for(int i=0 ; i < num_nodes ; i++)
    {
        sn* n = order[i]; 

        if(n->is_operator())
        {
           if(n->l->is_zero() && num_leaves_placed < num_leaves)
            {
                n->l = sn::Prim(leaftypes[num_leaves_placed]) ; 
                num_leaves_placed += 1 ; 
            }    
            if(n->r->is_zero() && num_leaves_placed < num_leaves)
            {
                n->r = sn::Prim(leaftypes[num_leaves_placed]) ;
                num_leaves_placed += 1 ; 
            }    
        } 
    } 

    assert( num_leaves_placed == num_leaves ); 
}



inline void sn::prune()
{   
    Prune_r(this, 0);
}

/**
Based on npy/NTreeBuilder
**/

inline void sn::Prune_r(sn* n, int d)  // static
{   
    if(n == nullptr) return ;
    if(n->is_operator())
    {   
        Prune_r(n->l, d+1);
        Prune_r(n->r, d+1);
        
        // postorder visit : so both children always visited before their parents 
        
        if(n->l->is_lrzero())         // left node is an operator which has both its left and right zero 
        {   
            n->l = sn::Zero() ;       // prune : ie replace operator with CSG_ZERO placeholder  
        }
        else if( n->l->is_rzero() )   // left node is an operator with left non-zero and right zero   
        {   
            n->l = n->l->l ;          // moving the lonely primitive up to higher elevation   
        }
        
        if(n->r->is_lrzero())        // right node is operator with both its left and right zero 
        {   
            n->r = sn::Zero() ;      // prune
        }
        else if( n->r->is_rzero() )  // right node is operator with its left non-zero and right zero
        {   
            n->r = n->r->l ;         // moving the lonely primitive up to higher elevation   
        }
    }
}

inline void sn::Check(const sn* n) // static 
{
    if(n->l->is_operator() && n->r->is_zero() )
    {
        std::cerr << "sn::Check ERROR detected dangling zero (see NTreeBuilder::rootprune) " << std::endl ;  
        assert(0); 
    }
}


