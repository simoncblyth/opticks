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

#include "OpticksCSG.h"
#include "scanvas.h"

struct sn
{
    int t ; 
    int depth ; 
    int subdepth ; 
    bool complement ; 

    sn* l ; 
    sn* r ;     

    static sn* Zero() ; 
    static sn* Prim(int type) ; 
    static sn* Boolean(int type, sn* left, sn* right); 

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

    int maxdepth() const ; 
    int maxdepth_r(int d) const ; 


    void label(); 

    int maxdepth_label() ; 
    int maxdepth_label_r(int d) ; 

    void subdepth_label() ; 
    void subdepth_label_r(int d); 

    unsigned operators(int minsubdepth) const ; 
    static void Operators_r(const sn* n, unsigned& mask, int minsubdepth); 
    bool is_positive_form() const ; 




    void postorder(std::vector<const sn*>& order ) const ; 
    void postorder_r(std::vector<const sn*>& order, int d ) const ; 

    void inorder(std::vector<const sn*>& order ) const ; 
    void inorder_r(std::vector<const sn*>& order, int d ) const ; 

    void inorder_(std::vector<sn*>& order ) ; 
    void inorder_r_(std::vector<sn*>& order, int d ); 




    std::string desc_order(const std::vector<const sn*>& order ) const ; 
    std::string desc() const ; 
    std::string render() const ; 
    std::string render(int mode) const ; 

    static constexpr const char* MODE_MINIMAL = "MINIMAL" ; 
    static constexpr const char* MODE_TYPECODE = "TYPECODE" ; 
    static constexpr const char* MODE_DEPTH = "DEPTH" ; 
    static constexpr const char* MODE_SUBDEPTH = "SUBDEPTH" ; 
    static constexpr const char* MODE_TYPETAG = "TYPETAG" ; 
    static const char* rendermode(int mode); 

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


    void positivize() ; 
    void positivize_r(bool negate, int d) ; 

};


inline sn* sn::Zero()   // static
{
    return Prim(0); 
}
inline sn* sn::Prim(int type)   // static
{
    return new sn {type, 0, 0, false, nullptr, nullptr} ; 
}
inline sn* sn::Boolean(int type, sn* left, sn* right)   // static
{
    return new sn {type, 0, 0, false, left, right} ; 
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


inline int sn::maxdepth() const
{
    return maxdepth_r(0);
}
inline int sn::maxdepth_r(int d) const
{
    return l && r ? std::max( l->maxdepth_r(d+1), r->maxdepth_r(d+1)) : d ; 
}



inline void sn::label()
{
    maxdepth_label(); 
    subdepth_label(); 
}

inline int sn::maxdepth_label() 
{
    return maxdepth_label_r(0);
}
inline int sn::maxdepth_label_r(int d)
{
    depth = d ; 
    return l && r ? std::max( l->maxdepth_label_r(d+1), r->maxdepth_label_r(d+1)) : d ; 
}



/** 
sn::subdepth_label  (based on NTreeBalance::subdepth_r)
------------------------------------------------------------

How far down can you go from each node. 

Labels the nodes with the subdepth, which is 
the max height of each node treated as a subtree::


               3                    

      [1]               2            

   [0]    [0]       0          [1]    

                           [0]     [0]


bileafs are triplets of nodes with subdepths 1,0,0
The above tree has two bileafs and one other leaf. 

**/

inline void sn::subdepth_label()
{
    subdepth_label_r(0); 
}
inline void sn::subdepth_label_r(int d)
{
    subdepth = maxdepth() ;
    if(l && r)
    {
        l->subdepth_label_r(d+1);
        r->subdepth_label_r(d+1);
    }
}



/**
sn::operators (based on NTreeBalance::operators)
----------------------------------------------------

Returns mask of CSG operators in the tree restricted to nodes with subdepth >= *minsubdepth*

**/

inline unsigned sn::operators(int minsubdepth) const 
{
   unsigned mask = 0 ;   
   Operators_r(this, mask, minsubdepth);  
   return mask ;   
}

inline void sn::Operators_r(const sn* n, unsigned& mask, int minsubdepth) // static
{
    if(n->l && n->r )
    {   
        if( n->subdepth >= minsubdepth )
        {   
            switch( n->t )
            {   
                case CSG_UNION         : mask |= CSG::Mask(CSG_UNION)        ; break ; 
                case CSG_INTERSECTION  : mask |= CSG::Mask(CSG_INTERSECTION) ; break ; 
                case CSG_DIFFERENCE    : mask |= CSG::Mask(CSG_DIFFERENCE)   ; break ; 
                default                : mask |= 0                           ; break ; 
            }   
        }   
        Operators_r( n->l ,  mask, minsubdepth );  
        Operators_r( n->r , mask, minsubdepth );  
    }   
}

inline bool sn::is_positive_form() const 
{
    unsigned ops = operators(0);  // minsubdepth:0 ie entire tree 
    return (ops & CSG::Mask(CSG_DIFFERENCE)) == 0 ; 
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
       << " maxdepth " << maxdepth() 
       << " is_positive_form " << ( is_positive_form() ? "Y" : "N" ) 
       ; 
    std::string str = ss.str();
    return str ;
}

inline std::string sn::render() const
{
    std::stringstream ss ;
    for(int mode=0 ; mode < 4 ; mode++) ss << render(mode) << std::endl ; 
    std::string str = ss.str();
    return str ;
}

inline std::string sn::render(int mode) const
{
    int width = num_node();
    int height = maxdepth();

    std::vector<const sn*> in ;
    inorder(in);
    assert( int(in.size()) == width );

    std::vector<const sn*> post ;
    postorder(post);
    assert( int(post.size()) == width );

    int xscale = 3 ; 
    int yscale = 2 ; 

    scanvas canvas( width+1, height+2, xscale, yscale );
    render_r(&canvas, in, mode,  0);

    std::stringstream ss ;
    ss << std::endl ;
    ss << desc() << std::endl ;  
    ss << "sn::render mode " << mode << " " << rendermode(mode) << std::endl ;
    ss << canvas.c << std::endl ;

    if(mode == 0 )
    {
        ss << "inorder   " << desc_order(in) << std::endl ; 
        ss << "postorder " << desc_order(post) << std::endl ; 

        unsigned ops = operators(0); 
        bool pos = is_positive_form() ; 

        ss << " ops = operators(0) " << ops << std::endl ; 
        ss << " CSG::MaskDesc(ops) : " << CSG::MaskDesc(ops) << std::endl ; 
        ss << " is_positive_form() : " << ( pos ? "YES" : "NO" ) << std::endl ;  
    }

    std::string str = ss.str();
    return str ;
}

inline const char* sn::rendermode(int mode) // static
{
    const char* md = nullptr ; 
    switch(mode) 
    {
        case 0: md = MODE_MINIMAL  ; break ; 
        case 1: md = MODE_TYPECODE ; break ; 
        case 2: md = MODE_DEPTH    ; break ; 
        case 3: md = MODE_SUBDEPTH ; break ; 
        case 4: md = MODE_TYPETAG  ; break ; 
    }
    return md ; 
}

inline void sn::render_r(scanvas* canvas, std::vector<const sn*>& order, int mode, int d) const
{
    int ordinal = std::distance( order.begin(), std::find(order.begin(), order.end(), this )) ;
    assert( ordinal < int(order.size()) );

    int ix = ordinal ;
    int iy = d ;
    std::string tag = CSG::Tag(t, complement); 

    switch(mode)
    {
        case 0: canvas->drawch( ix, iy, 0,0, 'o' )         ; break ; 
        case 1: canvas->draw(   ix, iy, 0,0,  t     )      ; break ; 
        case 2: canvas->draw(   ix, iy, 0,0,  depth )      ; break ;   
        case 3: canvas->draw(   ix, iy, 0,0,  subdepth )   ; break ; 
        case 4: canvas->draw(   ix, iy, 0,0,  tag.c_str()) ; break ;    
    } 

    if(l) l->render_r( canvas, order, mode, d+1 );
    if(r) r->render_r( canvas, order, mode, d+1 );
}



inline sn* sn::Build_r( int elevation, int op )  // static
{
    sn* l = elevation > 1 ? Build_r( elevation - 1 , op ) : sn::Zero() ; 
    sn* r = elevation > 1 ? Build_r( elevation - 1 , op ) : sn::Zero() ; 
    return sn::Boolean(op, l, r ) ;  
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




/**
sn::positivize (base on NTreePositive::positivize_r)
--------------------------------------------------------

* https://smartech.gatech.edu/bitstream/handle/1853/3371/99-04.pdf?sequence=1&isAllowed=y

* addition: union
* subtraction: difference
* product: intersect

Tree positivization (which is not the same as normalization) 
eliminates subtraction by propagating negations down the tree using deMorgan rules. 

**/


inline void sn::positivize()
{
    positivize_r(false, 0); 
}
inline void sn::positivize_r(bool negate, int d)
{
    if(l == nullptr && r == nullptr)  // primitive 
    {   
        if(negate) complement = !complement ; 
    }   
    else
    {   
        bool left_negate = false ; 
        bool right_negate = false ; 

        if(t == CSG_INTERSECTION || t == CSG_UNION)
        {   
            if(negate)                             // !( A*B ) ->  !A + !B       !(A + B) ->     !A * !B
            {    
                 t = CSG::DeMorganSwap(t) ;   // UNION->INTERSECTION, INTERSECTION->UNION
                 left_negate = true ; 
                 right_negate = true ; 
            }   
            else
            {                                      //  A * B ->  A * B         A + B ->  A + B
                 left_negate = false ;
                 right_negate = false ;
            }
        }
        else if(t == CSG_DIFFERENCE)
        {
            if(negate)                             //  !(A - B) -> !(A*!B) -> !A + B
            {
                t = CSG_UNION ;
                left_negate = true ;
                right_negate = false  ;
            }
            else
            {
                t = CSG_INTERSECTION ;    //    A - B ->  A * !B
                left_negate = false ;
                right_negate = true ;
            }
        }

        l->positivize_r(left_negate,  d+1);
        r->positivize_r(right_negate, d+1);
    }
}


