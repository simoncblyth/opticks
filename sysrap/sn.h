#pragma once
/**
sn.h : minimal pointer based transient binary tree node
========================================================

* used from sndtree.h 

Usage Example
--------------

::

    #include "sn.h"
    sn::POOL sn::pool = {} ;  // initialize static pool 


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


Could snd/sn be consolidated ?
--------------------------------

There is pressure to add things from snd to sn 
(eg param, bbox, transforms, n-ary "std::vector<sn*> child")
such that *sn* can be a complete representation of the CSG. 
But dont want to duplicate things.

In order to convert active *sn* pointers into indices 
have explictly avoided leaking any *sn* by taking care to delete
appropriately. This means that can use the *sn* ctor/dtor
to add/erase update an std::map of active *sn* pointers
keyed on a creation index.  This map allows the active 
pointers to be converted into a contiguous set of indices 
to facilitate serialization. 

Future
--------

Hopefully this functionality can be removed once have leaped
to CSG_CONTIGUOUS use as standard, which retains n-ary tree 
all the way to the GPU. 


**/

#include <map>
#include <vector>
#include <sstream>
#include <iomanip>
#include <cassert>

#include "OpticksCSG.h"
#include "scanvas.h"
#include "s_pool.h"


struct _sn
{
#ifdef WITH_CHILD
    static constexpr const int NV = 7 ; 
#else
    static constexpr const int NV = 5 ; 
#endif

    int t ; 
    int complement ; // could hold this in sign of t 
    int p ; 

#ifdef WITH_CHILD
    int sibdex ;  // 0-based sibling index 
    int num_child ; 
    int first_child ; 
    int next_sibling ; 
#else
    int l ; 
    int r ; 
#endif

    bool is_root() const ; 
    std::string desc() const ; 
};

bool _sn::is_root() const 
{
    return p == -1 ;  
}

inline std::string _sn::desc() const
{
    std::stringstream ss ; 
    ss << "_sn::desc " 
       << " t " << std::setw(4) << t 
       << " c " << std::setw(1) << complement
       << " p " << std::setw(4) << p 
#ifdef WITH_CHILD
       << " sx " << std::setw(4) << sibdex 
       << " nc " << std::setw(4) << num_child
       << " fc " << std::setw(4) << first_child
       << " xs " << std::setw(4) << next_sibling
#else
       << " l " << std::setw(4) << l 
       << " r " << std::setw(4) << r 
#endif
       ;
    std::string str = ss.str(); 
    return str ; 
}


struct sn
{
    typedef s_pool<sn> POOL ;
    static POOL pool ;

    static std::string Desc(const char* msg=nullptr); 
    static int level(); 

    int  index() const ; 
    bool is_root() const ; 

    int  num_child() const ; 
    sn*  first_child() const ; 
    sn*  last_child() const ; 
    sn*  get_child(int ch) const ;

#ifdef WITH_CHILD
    int  total_siblings() const ;
    int  sibling_index() const ;
    const sn*  get_sibling(int sx) const ; // returns this when sx is sibling_index
    const sn*  next_sibling() const ;      // returns nullptr when this is last 
#endif

    static void Serialize(     _sn& p, const sn* o ); 
    static sn*  Import(  const _sn* p, const std::vector<_sn>& buf ); 
    static sn*  Import_r(const _sn* p, const std::vector<_sn>& buf); 

    //static constexpr const bool LEAK = true ; 
    static constexpr const bool LEAK = false ; 

    int pid ;       // (not persisted)
    int depth ;     // (not persisted)
    int subdepth ;  // (not persisted)


    int t ; 
    int complement ; 
#ifdef WITH_CHILD
    std::vector<sn*> child ; 
#else
    sn* l ; 
    sn* r ;     
#endif

    sn* p ; 

    sn(int type, sn* left, sn* right);
#ifdef WITH_CHILD
    void add_child( sn* ch ); 
#endif

    ~sn(); 

    static sn* Zero() ; 
    static sn* Prim(int type) ; 
    static sn* Create(int type, sn* left, sn* right); 

    void disown_child(sn* ch) ; 
    sn* deepcopy() const ; 
    sn* deepcopy_r(int d) const ; 

    void set_child( int ix, sn* ch, bool copy ); 
    void set_left( sn* left, bool copy ); 
    void set_right( sn* right, bool copy  );

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
    void operators_v(unsigned& mask, int minsubdepth) const ; 
    void operators_r(unsigned& mask, int minsubdepth) const ; 
    bool is_positive_form() const ; 

    void preorder( std::vector<const sn*>& order ) const ; 
    void inorder(  std::vector<const sn*>& order ) const ; 
    void postorder(std::vector<const sn*>& order ) const ; 

    void preorder_r( std::vector<const sn*>& order, int d ) const ; 
    void inorder_r(  std::vector<const sn*>& order, int d ) const ; 
    void postorder_r(std::vector<const sn*>& order, int d ) const ; 

    void inorder_(std::vector<sn*>& order ) ; 
    void inorder_r_(std::vector<sn*>& order, int d ); 

    std::string desc_order(const std::vector<const sn*>& order ) const ; 
    std::string desc() const ; 
    std::string desc_child() const ; 
    std::string desc_r() const ; 
    void desc_r(int d, std::stringstream& ss) const ; 

    std::string render() const ; 
    std::string render(int mode) const ; 

    static constexpr const char* MODE_MINIMAL = "MINIMAL" ; 
    static constexpr const char* MODE_TYPECODE = "TYPECODE" ; 
    static constexpr const char* MODE_DEPTH = "DEPTH" ; 
    static constexpr const char* MODE_SUBDEPTH = "SUBDEPTH" ; 
    static constexpr const char* MODE_TYPETAG = "TYPETAG" ; 
    static constexpr const char* MODE_PID = "PID" ; 
    static const char* rendermode(int mode); 

    void render_r(scanvas* canvas, std::vector<const sn*>& order, int mode, int d) const ; 


    static int BinaryTreeHeight(int num_leaves); 
    static int BinaryTreeHeight_1(int num_leaves); 

    static sn* ZeroTree_r(int elevation, int op); 
    static sn* ZeroTree(int num_leaves, int op ); 

    static sn* CommonTree( std::vector<int>& leaftypes,  int op ); 
    void populate(std::vector<int>& leaftypes ); 
    void prune(); 
    void prune_r(int d) ; 
    bool has_dangle() const ; 

    void positivize() ; 
    void positivize_r(bool negate, int d) ; 
};

inline std::string sn::Desc(const char* msg){ return pool.desc(msg); }
inline int         sn::level() {  return pool.level ; } // static 

inline int         sn::index() const { return pool.index(this); }
inline bool        sn::is_root() const { return p == nullptr ; }


inline int sn::num_child() const
{
#ifdef WITH_CHILD
    return int(child.size()); 
#else
    return l && r ? 2 : 0 ; 
#endif
}

inline sn* sn::first_child() const 
{
#ifdef WITH_CHILD
    return child.size() > 0 ? child[0] : nullptr ; 
#else
    return l ; 
#endif
}
inline sn* sn::last_child() const 
{
#ifdef WITH_CHILD
    return child.size() > 0 ? child[child.size()-1] : nullptr ; 
#else
    return r ; 
#endif
}
inline sn* sn::get_child(int ch) const 
{
#ifdef WITH_CHILD
    return ch > -1 && ch < int(child.size()) ? child[ch] : nullptr ; 
#else
    switch(ch)
    {
        case 0: return l ; break ; 
        case 1: return r ; break ; 
    }
    return nullptr ; 
#endif
}


#ifdef WITH_CHILD
inline int sn::total_siblings() const
{
    return p ? int(p->child.size()) : 1 ;  // root regarded as sole sibling (single child)  
}
inline int sn::sibling_index() const 
{
    int tot_sib = total_siblings() ; 
    int sibdex = p == nullptr ? 0 : std::distance( p->child.begin(), std::find( p->child.begin(), p->child.end(), this )) ; 

    std::cerr << "sn::sibling_index"
              << " tot_sib " << tot_sib 
              << " sibdex " << sibdex
              << std::endl 
              ;

    assert( sibdex < tot_sib ); 
    return sibdex ;  
}

inline const sn* sn::get_sibling(int sx) const     // NB this return self for appropriate sx
{
    assert( sx < total_siblings() ); 
    return p ? p->child[sx] : this ; 
}

inline const sn* sn::next_sibling() const
{
    int next_sib = 1+sibling_index() ; 
    int tot_sib = total_siblings() ; 
    std::cerr << "sn::next_sibling" 
              << " tot_sib " << tot_sib
              << " next_sib " << next_sib 
              << std::endl 
              ;
 
    return next_sib < tot_sib - 1 ? get_sibling(next_sib) : nullptr ;   
}
#endif



inline void sn::Serialize(_sn& n, const sn* x) // static 
{
    n.t = x->t ; 
    n.complement = x->complement ;
    n.p = pool.index(x->p);  

#ifdef WITH_CHILD
    n.sibdex = x->sibling_index(); 
    n.num_child = x->num_child() ; 
    n.first_child = pool.index(x->first_child());  
    n.next_sibling = pool.index(x->next_sibling()); 
#else
    n.l = pool.index(x->l);  
    n.r = pool.index(x->r);  
#endif

}

inline sn* sn::Import( const _sn* p, const std::vector<_sn>& buf ) // static
{
    return p->is_root() ? Import_r(p, buf) : nullptr ; 
}

inline sn* sn::Import_r(const _sn* _n,  const std::vector<_sn>& buf)
{
    if(_n == nullptr) return nullptr ; 
    std::cerr << "sn::Import_r " << _n->desc() << std::endl ; 

#ifdef WITH_CHILD
    sn* n = Create( _n->t, nullptr, nullptr );  
    const _sn* _child = _n->first_child  > -1 ? &buf[_n->first_child] : nullptr  ; 
    while( _child ) 
    {    
        sn* ch = Import_r( _child, buf ); 
        assert(ch); 
        n->child.push_back(ch); 
        _child = _child->next_sibling > -1 ? &buf[_child->next_sibling] : nullptr ;
    }    
#else
    const _sn* _l = _n->l > -1 ? &buf[_n->l] : nullptr ;  
    const _sn* _r = _n->r > -1 ? &buf[_n->r] : nullptr ;  
    sn* l = Import_r( _l, buf ); 
    sn* r = Import_r( _r, buf ); 
    sn* n = Create( _n->t, l, r ); 
#endif
    n->complement = _n->complement ; 

    return n ;  
}  

// ctor

inline sn::sn(int type, sn* left, sn* right)
    :
    pid(pool.add(this)),
    depth(0),
    subdepth(0),
    t(type),
    complement(0),
#ifdef WITH_CHILD
#else
    l(left),
    r(right),
#endif
    p(nullptr)
{
    if(pool.level > 1) std::cerr << "sn::sn pid " << pid << std::endl ; 

#ifdef WITH_CHILD
    if(left && right)
    {
        add_child(left); 
        add_child(right); 
    }
#else
    if(l && r)
    {
        l->p = this ; 
        r->p = this ; 
    }
#endif

}

#ifdef WITH_CHILD
inline void sn::add_child( sn* ch )
{
    ch->p = this ; 
    child.push_back(ch) ; 
}
#endif





// dtor 
inline sn::~sn()   
{
    if(pool.level > 1) std::cerr << "[ sn::~sn pid " << pid << std::endl ; 

#ifdef WITH_CHILD
    for(int i=0 ; i < int(child.size()) ; i++)
    {
        sn* ch = child[i] ; 
        delete ch ;  
    }
#else
    delete l ; 
    delete r ; 
#endif

    pool.remove(this); 

    if(pool.level > 1) std::cerr << "] sn::~sn pid " << pid << std::endl ; 
}



inline sn* sn::Zero()   // static
{
    return Prim(0); 
}
inline sn* sn::Prim(int type)   // static
{
    return new sn(type, nullptr, nullptr) ; 
}
inline sn* sn::Create(int type, sn* left, sn* right)   // static
{
    return new sn(type, left, right) ; 
}




#ifdef WITH_CHILD
inline void sn::disown_child(sn* ch)
{
    typedef std::vector<sn*>::iterator IT ; 
    IT it = std::find(child.begin(), child.end(), ch );
    if(it != child.end() ) child.erase(it) ; 
}
#endif

inline sn* sn::deepcopy() const 
{
    return deepcopy_r(0); 
}

inline sn* sn::deepcopy_r(int d) const 
{
    sn* copy = new sn(*this) ;    
#ifdef WITH_CHILD
    // above copy ctor copies the child vector, but that is a shallow copy  
    // so in the below the shallow copies are disowned and deep copies made and added 
    // to the copy child vector
    for(int i=0 ; i < int(child.size()) ; i++)
    {
        sn* ch = child[i] ; 
        copy->disown_child( ch ) ;          // remove shallow copied child from the vector
        sn* deep_ch = ch->deepcopy_r(d+1) ; 
        copy->child.push_back( deep_ch ); 
    }
#else
    copy->l = l ? l->deepcopy_r(d+1) : nullptr ; 
    copy->r = r ? r->deepcopy_r(d+1) : nullptr ;   
#endif
    copy->p = nullptr ; 

    return copy ;   
}

/**
sn::set_child
---------------

When the new child is from within the tree when pruning 
it is necessary to deepcopy it first. 

**/

inline void sn::set_child( int ix, sn* ch, bool copy )
{
    sn* new_ch = copy ? ch->deepcopy() : ch ; 
    new_ch->p = this ; 

#ifdef WITH_CHILD
    assert( ix < int(child.size()) );   
    sn*& target = child[ix] ;
    if(!LEAK) delete target ;
    target = new_ch ; 
#else
    sn** target = ix == 0 ? &l : &r ;  
    if(!LEAK) delete *target ;
    *target = new_ch ;  
#endif

}

inline void sn::set_left( sn* ch, bool copy )
{
    set_child(0, ch, copy ); 
}
inline void sn::set_right( sn* ch, bool copy )
{
    set_child(1, ch, copy ); 
}





inline bool sn::is_primitive() const
{   
#ifdef WITH_CHILD
    return child.size() == 0 ; 
#else
    return l == nullptr && r == nullptr ;
#endif

}   
inline bool sn::is_bileaf() const 
{   
#ifdef WITH_CHILD
    int num_ch   = int(child.size()) ; 
    int num_prim = 0 ; 
    for(int i=0 ; i < num_ch ; i++) if(child[i]->is_primitive()) num_prim += 1 ; 
    bool all_prim = num_prim == num_ch ; 
    return !is_primitive() && all_prim ;  
#else
    return !is_primitive() && l->is_primitive() && r->is_primitive() ;
#endif
}   
inline bool sn::is_operator() const 
{   
#ifdef WITH_CHILD
    return child.size() == 2 ;  
#else
    return l != nullptr && r != nullptr ;
#endif
}
inline bool sn::is_zero() const 
{   
    return t == 0 ;  
}
inline bool sn::is_lrzero() const 
{   
#ifdef WITH_CHILD
    int num_ch   = int(child.size()) ; 
    int num_zero = 0 ; 
    for(int i=0 ; i < num_ch ; i++) if(child[i]->is_zero()) num_zero += 1 ; 
    bool all_zero = num_zero == num_ch ; 
    return is_operator() && all_zero ;  
#else
    return is_operator() && l->is_zero() && r->is_zero() ;
#endif
}
inline bool sn::is_rzero() const
{
#ifdef WITH_CHILD
    return is_operator() && !child[0]->is_zero() && child[1]->is_zero() ; 
#else
    return is_operator() && !l->is_zero() && r->is_zero() ; 
#endif
}
inline bool sn::is_lzero() const 
{   
#ifdef WITH_CHILD
    return is_operator() && child[0]->is_zero() && !child[1]->is_zero() ;
#else
    return is_operator() && l->is_zero() && !r->is_zero() ;
#endif
}







inline int sn::num_node() const
{
    return num_node_r(0);
}
inline int sn::num_node_r(int d) const
{
    int nn = 1 ;   // always at least 1 node,  no exclusion of CSG_ZERO
#ifdef WITH_CHILD
    for(int i=0 ; i < int(child.size()) ; i++) nn += child[i]->num_node_r(d+1) ; 
#else
    nn += l ? l->num_node_r(d+1) : 0 ; 
    nn += r ? r->num_node_r(d+1) : 0 ; 
#endif
    return nn ;
}


inline int sn::num_leaf() const
{
    return num_leaf_r(0);
}
inline int sn::num_leaf_r(int d) const
{
    int nl = is_primitive() ? 1 : 0 ; 
#ifdef WITH_CHILD
    for(int i=0 ; i < int(child.size()) ; i++) nl += child[i]->num_leaf_r(d+1) ; 
#else
    nl += l ? l->num_leaf_r(d+1) : 0 ; 
    nl += r ? r->num_leaf_r(d+1) : 0 ; 
#endif
    return nl ;
}


inline int sn::maxdepth() const
{
    return maxdepth_r(0);
}
inline int sn::maxdepth_r(int d) const
{
#ifdef WITH_CHILD
    if( child.size() == 0 ) return d ; 
    int mx = 0 ; 
    for(int i=0 ; i < int(child.size()) ; i++) mx = std::max( mx, child[i]->maxdepth_r(d+1) ) ; 
    return mx ; 
#else
    return l && r ? std::max( l->maxdepth_r(d+1), r->maxdepth_r(d+1)) : d ; 
#endif
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
#ifdef WITH_CHILD
    if( child.size() == 0 ) return d ; 
    int mx = 0 ; 
    for(int i=0 ; i < int(child.size()) ; i++) mx = std::max( mx, child[i]->maxdepth_label_r(d+1) ) ; 
    return mx ; 
#else
    return l && r ? std::max( l->maxdepth_label_r(d+1), r->maxdepth_label_r(d+1)) : d ; 
#endif
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
#ifdef WITH_CHILD
    for(int i=0 ; i < int(child.size()) ; i++) child[i]->subdepth_label_r(d+1) ; 
#else
    if(l && r)
    {
        l->subdepth_label_r(d+1);
        r->subdepth_label_r(d+1);
    }
#endif
}



/**
sn::operators (based on NTreeBalance::operators)
----------------------------------------------------

Returns mask of CSG operators in the tree restricted to nodes with subdepth >= *minsubdepth*

**/

inline unsigned sn::operators(int minsubdepth) const 
{
   unsigned mask = 0 ;   
   operators_r(mask, minsubdepth);  
   return mask ;   
}


inline void sn::operators_v(unsigned& mask, int minsubdepth) const 
{
    if( subdepth >= minsubdepth )
    {   
        switch( t )
        {   
            case CSG_UNION         : mask |= CSG::Mask(CSG_UNION)        ; break ; 
            case CSG_INTERSECTION  : mask |= CSG::Mask(CSG_INTERSECTION) ; break ; 
            case CSG_DIFFERENCE    : mask |= CSG::Mask(CSG_DIFFERENCE)   ; break ; 
            default                : mask |= 0                           ; break ; 
        }   
    }   
}

inline void sn::operators_r(unsigned& mask, int minsubdepth) const
{
#ifdef WITH_CHILD
    if(child.size() >= 2) operators_v(mask, minsubdepth) ; 
    for(int i=0 ; i < int(child.size()) ; i++) child[i]->operators_r(mask, minsubdepth ) ; 
#else
    if(l && r )
    {   
        operators_v(mask, minsubdepth );
        l->operators_r( mask, minsubdepth );  
        r->operators_r( mask, minsubdepth );  
    }   
#endif

}

inline bool sn::is_positive_form() const 
{
    unsigned ops = operators(0);  // minsubdepth:0 ie entire tree 
    return (ops & CSG::Mask(CSG_DIFFERENCE)) == 0 ; 
}





inline void sn::preorder(std::vector<const sn*>& order ) const
{
    preorder_r(order, 0);
}
inline void sn::inorder(std::vector<const sn*>& order ) const
{
    inorder_r(order, 0);
}
inline void sn::postorder(std::vector<const sn*>& order ) const
{
    postorder_r(order, 0);
}


inline void sn::preorder_r(std::vector<const sn*>& order, int d ) const
{
    order.push_back(this); 
#ifdef WITH_CHILD
    for(int i=0 ; i < int(child.size()) ; i++) child[i]->preorder_r(order, d+1) ; 
#else
    if(l) l->preorder_r(order, d+1) ; 
    if(r) r->preorder_r(order, d+1) ; 
#endif


}
inline void sn::inorder_r(std::vector<const sn*>& order, int d ) const
{
#ifdef WITH_CHILD
    int nc = int(child.size()) ; 
    if( nc > 0 )
    {
        int split = nc - 1 ; 
        for(int i=0 ; i < split ; i++) child[i]->inorder_r(order, d+1) ; 
        order.push_back(this); 
        for(int i=split ; i < nc ; i++) child[i]->inorder_r(order, d+1) ; 
    }
    else
    {
        order.push_back(this); 
    }
#else
    if(l) l->inorder_r(order, d+1) ; 
    order.push_back(this); 
    if(r) r->inorder_r(order, d+1) ; 
#endif
}
inline void sn::postorder_r(std::vector<const sn*>& order, int d ) const
{
#ifdef WITH_CHILD
    for(int i=0 ; i < int(child.size()) ; i++) child[i]->postorder_r(order, d+1) ; 
#else
    if(l) l->postorder_r(order, d+1) ; 
    if(r) r->postorder_r(order, d+1) ; 
#endif
    order.push_back(this); 
}


inline void sn::inorder_(std::vector<sn*>& order )
{
    inorder_r_(order, 0);
}
inline void sn::inorder_r_(std::vector<sn*>& order, int d )
{
#ifdef WITH_CHILD
    int nc = int(child.size()) ; 
    if( nc > 0 )
    {
        int split = nc - 1 ; 
        for(int i=0 ; i < split ; i++) child[i]->inorder_r_(order, d+1) ; 
        order.push_back(this); 
        for(int i=split ; i < nc ; i++) child[i]->inorder_r_(order, d+1) ; 
    }
    else
    {
        order.push_back(this); 
    }
#else
    if(l) l->inorder_r_(order, d+1) ; 
    order.push_back(this); 
    if(r) r->inorder_r_(order, d+1) ; 
#endif
}


inline std::string sn::desc_order(const std::vector<const sn*>& order ) const 
{
    std::stringstream ss ;
    ss << "sn::desc_order [" ; 
    for(int i=0 ; i < int(order.size()) ; i++)
    {
        const sn* n = order[i] ; 
        ss << n->pid << " " ;  
    }
    ss << "]" ; 
    std::string str = ss.str();
    return str ;
}


inline std::string sn::desc() const
{
    std::stringstream ss ;
    ss << "sn::desc"
       << " pid " << std::setw(4) << pid
       << " idx " << std::setw(4) << index()
       << " t " << std::setw(3) << t 
       << " num_node " << std::setw(3) << num_node() 
       << " num_leaf " << std::setw(3) << num_leaf() 
       << " maxdepth " << std::setw(2) << maxdepth() 
       << " is_positive_form " << ( is_positive_form() ? "Y" : "N" ) 
       ; 
    std::string str = ss.str();
    return str ;
}

inline std::string sn::desc_child() const
{
    std::stringstream ss ;
    ss << "sn::desc_child num " << num_child() << std::endl ; 
    for( int i=0 ; i < num_child() ; i++)
    {
        const sn* ch = get_child(i) ; 
        ss << " i " << std::setw(2) << i << " 0x" << std::hex << uint64_t(ch) << std::dec << std::endl ; 
    }
    std::string str = ss.str();
    return str ;
}

inline std::string sn::desc_r() const
{
    std::stringstream ss ; 
    ss << "sn::desc_r" << std::endl ; 
    desc_r(0, ss); 
    std::string str = ss.str();
    return str ;
}
inline void sn::desc_r(int d, std::stringstream& ss) const
{
    ss << std::setw(3) << d << ": 0x" << std::hex << uint64_t(this) << " " << std::dec << desc()  << std::endl ; 
#ifdef WITH_CHILD
    for(int i=0 ; i < int(child.size()) ; i++) child[i]->desc_r(d+1, ss ) ; 
#else
    if( l && r )
    {
        l->desc_r(d+1, ss); 
        r->desc_r(d+1, ss); 
    }
#endif
}



inline std::string sn::render() const
{
    std::stringstream ss ;
    for(int mode=0 ; mode < 6 ; mode++) ss << render(mode) << std::endl ; 
    std::string str = ss.str();
    return str ;
}



inline std::string sn::render(int mode) const
{
    int nn = num_node(); 

    std::vector<const sn*> pre ;
    preorder(pre);
    assert( int(pre.size()) == nn );

    std::vector<const sn*> in ;
    inorder(in);
    assert( int(in.size()) == nn );

    std::vector<const sn*> post ;
    postorder(post);
    assert( int(post.size()) == nn );


    int width = nn ;
    int height = maxdepth();

    int xscale = 3 ; 
    int yscale = 2 ; 

    scanvas canvas( width+1, height+2, xscale, yscale );
    render_r(&canvas, in, mode,  0);

    std::stringstream ss ;
    ss << std::endl ;
    ss << desc() << std::endl ;  
    ss << "sn::render mode " << mode << " " << rendermode(mode) << std::endl ;
    ss << canvas.c << std::endl ;

    if(mode == 0 || mode == 5)
    {
        ss << "preorder  " << desc_order(pre)  << std::endl ; 
        ss << "inorder   " << desc_order(in)   << std::endl ; 
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
        case 5: md = MODE_PID      ; break ; 
    }
    return md ; 
}

inline void sn::render_r(scanvas* canvas, std::vector<const sn*>& order, int mode, int d) const
{
    int ordinal = std::distance( order.begin(), std::find(order.begin(), order.end(), this )) ;
    assert( ordinal < int(order.size()) );

    int ix = ordinal ;
    int iy = d ;
    std::string tag = CSG::Tag(t, complement == 1); 

    switch(mode)
    {
        case 0: canvas->drawch( ix, iy, 0,0, 'o' )         ; break ; 
        case 1: canvas->draw(   ix, iy, 0,0,  t     )      ; break ; 
        case 2: canvas->draw(   ix, iy, 0,0,  depth )      ; break ;   
        case 3: canvas->draw(   ix, iy, 0,0,  subdepth )   ; break ; 
        case 4: canvas->draw(   ix, iy, 0,0,  tag.c_str()) ; break ;    
        case 5: canvas->draw(   ix, iy, 0,0,  pid )        ; break ;    
    } 

#ifdef WITH_CHILD
    for(int i=0 ; i < int(child.size()) ; i++) child[i]->render_r(canvas, order, mode, d+1) ; 
#else
    if(l) l->render_r( canvas, order, mode, d+1 );
    if(r) r->render_r( canvas, order, mode, d+1 );
#endif
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


/**
sn::ZeroTree_r
---------------

Recursively builds complete binary tree 
with all operator nodes with a common *op* type 
and all leaf nodes are sn::Zero. 

**/

inline sn* sn::ZeroTree_r( int elevation, int op )  // static
{
    sn* l = elevation > 1 ? ZeroTree_r( elevation - 1 , op ) : sn::Zero() ; 
    sn* r = elevation > 1 ? ZeroTree_r( elevation - 1 , op ) : sn::Zero() ; 
    sn* lr = sn::Create(op, l, r ) ; 
    return lr  ;  
}
inline sn* sn::ZeroTree( int num_leaves, int op ) // static
{   
    int height = BinaryTreeHeight(num_leaves) ;
    if(pool.level > 0 ) std::cerr << "[sn::ZeroTree num_leaves " << num_leaves << " height " << height << std::endl; 
    sn* root = ZeroTree_r( height, op );
    if(pool.level > 0) std::cerr << "]sn::ZeroTree " << std::endl ; 
    return root ; 
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
        root = ZeroTree(num_leaves, op );   

        if(pool.level > 0) std::cerr << "sn::CommonTree ZeroTree num_leaves " << num_leaves << std::endl ; 
        if(pool.level > 1) std::cerr << root->render(5) ; 

        root->populate(leaftypes); 

        if(pool.level > 0) std::cerr << "sn::CommonTree populated num_leaves " << num_leaves << std::endl ; 
        if(pool.level > 1) std::cerr << root->render(5) ; 

        root->prune();
 
        if(pool.level > 0) std::cerr << "sn::CommonTree pruned num_leaves " << num_leaves << std::endl ; 
        if(pool.level > 1) std::cerr << root->render(5) ; 
    }
    return root ; 
} 

/**
sn::populate
--------------

Replacing zeros with leaf nodes

**/
        
inline void sn::populate(std::vector<int>& leaftypes )
{
    int num_leaves = leaftypes.size(); 
    int num_leaves_placed = 0 ; 

    std::vector<sn*> order ; 
    inorder_(order) ; 

    int num_nodes = order.size(); 

    if(pool.level > 0) std::cerr 
        << "sn::populate"
        << " num_leaves " << num_leaves
        << " num_nodes " << num_nodes
        << std::endl
        ;

    for(int i=0 ; i < num_nodes ; i++)
    {
        sn* n = order[i]; 
        if(pool.level > 1) std::cerr 
            << "sn::populate " << std::setw(3) << i 
            << " n.desc " << n->desc()
            << std::endl 
            ; 
    }

    for(int i=0 ; i < num_nodes ; i++)
    {
        sn* n = order[i]; 

#ifdef WITH_CHILD
        if(pool.level > 1) std::cerr 
            << "sn::populate"
            << " WITH_CHILD "
            << " i " << i 
            << " n.is_operator " << n->is_operator()
            << " n.child.size " << n->child.size()
            << " num_leaves_placed " << num_leaves_placed
            << std::endl 
            ; 

        if(n->is_operator())
        {
            assert( n->child.size() == 2 ); 
            for(int j=0 ; j < 2 ; j++)
            {
                sn* ch = n->child[j] ; 
                if(pool.level > 1 ) std::cerr << "sn::populate ch.desc " << ch->desc() << std::endl ; 

                if( ch->is_zero() && num_leaves_placed < num_leaves )
                {
                    n->set_child(j, sn::Prim(leaftypes[num_leaves_placed]), false) ;  
                    num_leaves_placed += 1 ; 
                } 
            } 
        }
#else
        if(n->is_operator())
        {
            if(n->l->is_zero() && num_leaves_placed < num_leaves)
            {
                n->set_left( sn::Prim(leaftypes[num_leaves_placed]), false ) ; 
                num_leaves_placed += 1 ; 
            }    
            if(n->r->is_zero() && num_leaves_placed < num_leaves)
            {
                n->set_right(sn::Prim(leaftypes[num_leaves_placed]), false ) ;
                num_leaves_placed += 1 ; 
            }    
        }
#endif
    } 
    assert( num_leaves_placed == num_leaves ); 
}



inline void sn::prune()
{   
    prune_r(0);

    if(has_dangle())
    {
        if(pool.level > -1) std::cerr << "sn::prune ERROR left with dangle " << std::endl ; 
    }

}

/**
Based on npy/NTreeBuilder
**/

inline void sn::prune_r(int d) 
{   
    if(!is_operator()) return ; 

#ifdef WITH_CHILD
    assert( child.size() == 2 ); 

    sn* l = child[0] ; 
    sn* r = child[1] ; 

    l->prune_r(d+1); 
    r->prune_r(d+1); 

    if( l->is_lrzero() )     // left node is an operator which has both its left and right zero
    {
        set_left(sn::Zero(), false) ;       // prune : ie replace operator with CSG_ZERO placeholder  
    }  
    else if( l->is_rzero() )  // left node is an operator with left non-zero and right zero 
    {
        sn* ll = l->child[0] ; 
        set_left( ll, true ); 
    }

    if(r->is_lrzero())        // right node is operator with both its left and right zero 
    {   
        set_right(sn::Zero(), false) ;      // prune
    }
    else if( r->is_rzero() )  // right node is operator with its left non-zero and right zero
    {  
        sn* rl = r->child[0] ; 
        set_right(rl, true) ;         // moving the lonely primitive up to higher elevation   
    }
#else

    l->prune_r(d+1);
    r->prune_r(d+1);
    
    // postorder visit : so both children always visited before their parents 
    
    if(l->is_lrzero())         // left node is an operator which has both its left and right zero 
    {   
        set_left(sn::Zero(), false) ;       // prune : ie replace operator with CSG_ZERO placeholder  
    }
    else if( l->is_rzero() )   // left node is an operator with left non-zero and right zero   
    {  
        set_left(l->l, true) ;          // moving the lonely primitive up to higher elevation   
    }
    
    if(r->is_lrzero())        // right node is operator with both its left and right zero 
    {   
        set_right(sn::Zero(), false) ;      // prune
    }
    else if( r->is_rzero() )  // right node is operator with its left non-zero and right zero
    {   
        set_right(r->l, true) ;         // moving the lonely primitive up to higher elevation   
    }
#endif


}

inline bool sn::has_dangle() const  // see NTreeBuilder::rootprune
{
#ifdef WITH_CHILD
    int num_zero = 0 ; 
    for(int i=0 ; i < int(child.size()) ; i++) if(child[i]->is_zero()) num_zero += 1 ; 
    return num_zero > 0 ;  
#else
    return is_operator() && ( r->is_zero() || l->is_zero()) ; 
#endif
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
    if(is_primitive()) 
    {   
        if(negate) 
        {
            switch(complement)
            {
                case 0: complement = 1 ; break ; 
                case 1: complement = 0 ; break ; 
                default: assert(0)     ; break ; 
            }
        }
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

#ifdef WITH_CHILD
        assert( child.size() == 2 ); 
        sn* l = child[0] ; 
        sn* r = child[1] ; 
#endif
        l->positivize_r(left_negate,  d+1);
        r->positivize_r(right_negate, d+1);
    }
}


