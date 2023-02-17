#pragma once
/**
sn.h : minimal pointer based transient binary tree node
========================================================

* used from sndtree.h 

Usage Example
--------------

::

    #include "sn.h"
    std::map<int, sn*> sn::pool = {} ; 
    int sn::count = 0 ; 
    int sn::level = 0 ; 


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


struct _sn
{
    static constexpr const int NV = 5 ; 

    int t ; 
    int complement ; // could hold this in sign of t 
    int l ; 
    int r ; 
    int p ; 

    std::string desc() const ; 
};

inline std::string _sn::desc() const
{
    std::stringstream ss ; 
    ss << "_sn::desc " 
       << " t " << std::setw(4) << t 
       << " c " << std::setw(1) << complement
       << " l " << std::setw(4) << l 
       << " r " << std::setw(4) << r 
       << " p " << std::setw(4) << p 
       ;
    std::string str = ss.str(); 
    return str ; 
}


struct sn ; 

struct sn_query
{
    const sn* q ; 
    sn_query(const sn* q_) : q(q_) {} ;  
    bool operator()(const std::pair<int, sn*>& p){ return q == p.second ; }  
}; 

struct sn
{
    static std::map<int, sn*> pool ; 
    static int count ; 
    static int level ; 
    static std::string Desc(const char* msg=nullptr); 
    static int Index(const sn* q); 
    int index() const ; 

    static void Serialize( std::vector<_sn>& buf ) ; 
    static void Serialize( _sn& n, const sn* p ); 

    static sn* Import(                 const std::vector<_sn>& buf); 
    static sn* Import_r(const _sn* n,  const std::vector<_sn>& buf); 

    static constexpr const bool LEAK = false ; 

    int pid ;       // (not persisted)
    int depth ;     // (not persisted)
    int subdepth ;  // (not persisted)


    int t ; 
    int complement ; 
    sn* l ; 
    sn* r ;     
    sn* p ; 


    sn(int type, sn* left, sn* right);
    ~sn(); 


    static sn* Zero() ; 
    static sn* Prim(int type) ; 
    static sn* Create(int type, sn* left, sn* right); 

    sn* deepcopy() const ; 
    sn* deepcopy_r(int d) const ; 

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


inline std::string sn::Desc(const char* msg)
{
    if(level > 0) std::cerr << "[sn::Desc "
              << ( msg ? msg : "-" )
              << " LEAK " << ( LEAK ? "YES" : "NO" )
              << " count " << count  
              << " pool.size " << pool.size() 
              << std::endl
              ; 

    std::stringstream ss ; 
    ss << "sn::Desc "
       << ( msg ? msg : "-" )
       << " count " << count 
       << " pool.size " << pool.size() 
       << std::endl
        ; 

    typedef std::map<int, sn*>::const_iterator IT ; 
    for(IT it=pool.begin() ; it != pool.end() ; it++) 
    {
        int key = it->first ; 
        sn* n = it->second ;  
        assert( n->pid == key ); 
        ss << std::setw(3) << key << " : " << n->desc() << std::endl ; 
    }
    std::string str = ss.str(); 

    if(level > 0) std::cerr << "]sn::Desc" << std::endl ; 

    return str ; 
}

/**
sn::Index
-----------

Contiguous index of *q* within all active nodes in creation order.
NB this is different from the *pid* because node deletions will 
cause gaps in the pid values whereas the indices will be contiguous. 

**/

inline int sn::Index(const sn* q)  // static
{
    if( q == nullptr ) return -1 ;     
    sn_query query(q); 
    size_t idx = std::distance( pool.begin(), std::find_if( pool.begin(), pool.end(), query )); 
    return idx < pool.size() ? idx : -1 ;  
}
inline int sn::index() const { return Index(this); }



inline void sn::Serialize( std::vector<_sn>& buf )
{
    int tot_nodes = pool.size(); 
    buf.resize(tot_nodes);  
    typedef std::map<int, sn*>::const_iterator IT ; 

    IT it = pool.begin(); 

    std::cerr << "[ sn::Serialize tot_nodes " << tot_nodes << std::endl ; 
    int idx = 0 ; 
    while( it != pool.end() )
    {
        int key = it->first ; 
        sn* x = it->second ;  

        assert( x->pid == key ); 
        assert( x->index() == idx ); 
        assert( idx < tot_nodes ); 

        _sn& n = buf[idx]; 

        Serialize( n, x ); 

        it++ ; idx++ ;  
    }
    std::cerr << "] sn::Serialize" << std::endl ; 
}


inline void sn::Serialize(_sn& n, const sn* x) // static 
{
    n.t = x->t ; 
    n.complement = x->complement ;
    n.l = Index(x->l);  
    n.r = Index(x->r);  
    n.p = Index(x->p);  
}




/**
sn::Import
-------------

HMM: previous tree imports like NCSG::import_tree_r
used complete binary tree ordering and ran from root down  

Are here trying to bring in from bottom up, which presents
problem for setting the refs.

BUT have random access into the buf so can jump around following 
l/r index : but need to identify the root from the _sn 
to know where to start.  There can be multiple so cannot just pick the last. 

Hence added parent to sn/_sn so can identify the roots. 

**/

inline sn* sn::Import(const std::vector<_sn>& buf )
{
    sn* root = nullptr ; 
    int tot_nodes = buf.size() ;
    std::cerr << "[ sn::Import tot_nodes " << tot_nodes << std::endl ; 
    for(int idx=0 ; idx < tot_nodes ; idx++)
    { 
        const _sn* n = &buf[idx]; 
        if(n->p == -1) 
        {
            root = Import_r( n, buf );  // Import_r from the roots 
            root->label(); 
        }
    }  
    std::cerr << "] sn::Import" << std::endl ; 
    return root ; 
}
inline sn* sn::Import_r(const _sn* _n,  const std::vector<_sn>& buf)
{
    if(_n == nullptr) return nullptr ; 

    std::cerr << "sn::Import_r " << _n->desc() << std::endl ; 

    const _sn* _l = _n->l > -1 ? &buf[_n->l] : nullptr ;  
    const _sn* _r = _n->r > -1 ? &buf[_n->r] : nullptr ;  

    sn* l = Import_r( _l, buf ); 
    sn* r = Import_r( _r, buf ); 
    sn* n = Create( _n->t, l, r ); 

    n->complement = _n->complement ; 

    return n ; 
}  



// ctor
inline sn::sn(int type, sn* left, sn* right)
    :
    pid(count),
    t(type),
    depth(0),
    subdepth(0),
    complement(0),
    l(left),
    r(right),
    p(nullptr)
{
    pool[pid] = this ; 
    if(level > 1) std::cerr << "sn::sn pid " << pid << std::endl ; 

    if(left && right)
    {
        left->p = this ; 
        right->p = this ; 
    }
    count += 1 ;   
}

// dtor 
inline sn::~sn()   
{
    if(level > 1) std::cerr << "[ sn::~sn pid " << pid << std::endl ; 

    delete l ; 
    delete r ; 
    pool.erase(pid); 

    if(level > 1) std::cerr << "] sn::~sn pid " << pid << std::endl ; 
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



inline sn* sn::deepcopy() const 
{
    return deepcopy_r(0); 
}
inline sn* sn::deepcopy_r(int d) const 
{
    sn* c = new sn(*this) ;    
    c->l = l ? l->deepcopy_r(d+1) : nullptr ; 
    c->r = r ? r->deepcopy_r(d+1) : nullptr ;   
    c->p = nullptr ; 

    return c ;   
}






/**
sn::set_left
-------------

As the new left will be from within the old left when pruning 
need to deepcopy it first. 

**/

inline void sn::set_left( sn* left, bool copy )
{
    sn* new_l = copy ? left->deepcopy() : left ; 
    new_l->p = this ; 

    if(!LEAK) delete l ; 
    l = new_l ;
}

inline void sn::set_right( sn* right, bool copy )
{
    sn* new_r = copy ? right->deepcopy() : right ; 
    new_r->p = this ; 

    if(!LEAK) delete r ; 
    r = new_r ; 
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
   operators_r(mask, minsubdepth);  
   return mask ;   
}

inline void sn::operators_r(unsigned& mask, int minsubdepth) const
{
    if(l && r )
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
        l->operators_r( mask, minsubdepth );  
        r->operators_r( mask, minsubdepth );  
    }   
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
    if(l) l->preorder_r(order, d+1) ; 
    if(r) r->preorder_r(order, d+1) ; 
}
inline void sn::inorder_r(std::vector<const sn*>& order, int d ) const
{
    if(l) l->inorder_r(order, d+1) ; 
    order.push_back(this); 
    if(r) r->inorder_r(order, d+1) ; 
}
inline void sn::postorder_r(std::vector<const sn*>& order, int d ) const
{
    if(l) l->postorder_r(order, d+1) ; 
    if(r) r->postorder_r(order, d+1) ; 
    order.push_back(this); 
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

    if(l) l->render_r( canvas, order, mode, d+1 );
    if(r) r->render_r( canvas, order, mode, d+1 );
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
    if(level > 0 ) std::cerr << "[sn::ZeroTree num_leaves " << num_leaves << " height " << height << std::endl; 
    sn* root = ZeroTree_r( height, op );
    if(level > 0) std::cerr << "]sn::ZeroTree " << std::endl ; 
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

        if(level > 0) std::cerr << "sn::CommonTree ZeroTree num_leaves " << num_leaves << std::endl ; 
        if(level > 1) std::cerr << root->render(5) ; 

        root->populate(leaftypes); 

        if(level > 0) std::cerr << "sn::CommonTree populated num_leaves " << num_leaves << std::endl ; 
        if(level > 1) std::cerr << root->render(5) ; 

        root->prune();
 
        if(level > 0) std::cerr << "sn::CommonTree pruned num_leaves " << num_leaves << std::endl ; 
        if(level > 1) std::cerr << root->render(5) ; 
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

    for(int i=0 ; i < num_nodes ; i++)
    {
        sn* n = order[i]; 

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
    } 
    assert( num_leaves_placed == num_leaves ); 
}



inline void sn::prune()
{   
    prune_r(0);

    if(has_dangle())
    {
        if(level > -1) std::cerr << "sn::prune ERROR left with dangle " << std::endl ; 
    }

}

/**
Based on npy/NTreeBuilder
**/

inline void sn::prune_r(int d) 
{   
    if(is_operator())
    {   
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
    }
}

inline bool sn::has_dangle() const  // see NTreeBuilder::rootprune
{
    return is_operator() && ( r->is_zero() || l->is_zero()) ; 
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

        l->positivize_r(left_negate,  d+1);
        r->positivize_r(right_negate, d+1);
    }
}


