#pragma once
/**
sn.h : minimal pointer based transient binary tree node
========================================================

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

#include "ssys.h"
#include "OpticksCSG.h"
#include "scanvas.h"
#include "s_pa.h"
#include "s_bb.h"
#include "s_tv.h"
#include "s_pool.h"
#include "NPFold.h"

struct _sn
{
    int type ;         // 0
    int complement ;   // 1 
    int lvid ;         // 2
    int tv ;           // 3
    int pa ;           // 4
    int bb ;           // 5 
    int parent ;       // 6 
     
#ifdef WITH_CHILD
    int sibdex ;       // 7     0-based sibling index 
    int num_child ;    // 8
    int first_child ;  // 9
    int next_sibling ; // 10  
    static constexpr const int NV = 11 ; 
#else
    int left ;         // 7
    int right ;        // 8
    static constexpr const int NV = 9 ; 
#endif
    std::string desc() const ; 
    bool is_root_importable() const ; 
};

inline std::string _sn::desc() const
{
    std::stringstream ss ; 
    ss << "_sn::desc " 
       << " type " << std::setw(4) << type 
       << " complement " << std::setw(1) << complement
       << " lvid " << std::setw(4) << lvid 
       << " tv " << std::setw(4) << tv
       << " pa " << std::setw(4) << pa
       << " bb " << std::setw(4) << bb
       << " parent " << std::setw(4) << parent 
#ifdef WITH_CHILD
       << " sx " << std::setw(4) << sibdex 
       << " nc " << std::setw(4) << num_child
       << " fc " << std::setw(4) << first_child
       << " ns " << std::setw(4) << next_sibling
#else
       << " left " << std::setw(4) << left 
       << " right " << std::setw(4) << right 
#endif
       << " is_root_importable " << ( is_root_importable() ? "YES" : "NO " ) 
       ;
    std::string str = ss.str(); 
    return str ; 
}

/**
_sn::is_root_importable
------------------------

Only root expected to have parent -1 

**/
inline bool _sn::is_root_importable() const 
{
    return parent == -1 ;  
}

#include "SYSRAP_API_EXPORT.hh"
struct SYSRAP_API sn
{
    // persisted
    int   type ; 
    int   complement ; 
    int   lvid ; 
    s_tv* tv ;    
    s_pa* pa ; 
    s_bb* bb  ;
    sn*   parent ;   // NOT owned by this sn 

#ifdef WITH_CHILD
    std::vector<sn*> child ;   
#else
    sn* left ;          
    sn* right ;        
#endif

    // internals, not persisted  
    int pid ;       
    int depth ;    
    int subdepth ; 


    typedef s_pool<sn,_sn> POOL ;
    static POOL* pool ;  
    static void SetPOOL( POOL* pool_ ); 

    static constexpr const int VERSION = 0 ;
    static constexpr const char* NAME = "sn.npy" ; 
    static constexpr const double zero = 0. ; 

    static std::string Desc(const char* msg=nullptr); 
    static int level(); 

    int  index() const ; 
    bool is_root_importable() const ; 

    int  num_child() const ; 
    sn*  first_child() const ; 
    sn*  last_child() const ; 
    sn*  get_child(int ch) const ;

    int  total_siblings() const ;
    int  child_index( const sn* ch ) ; 
    int  sibling_index() const ;
    const sn*  get_sibling(int sx) const ; // returns this when sx is sibling_index
    const sn*  next_sibling() const ;      // returns nullptr when this is last 

    static void Serialize(     _sn& p, const sn* o ); 
    static sn*  Import(  const _sn* p, const std::vector<_sn>& buf ); 
    static sn*  Import_r(const _sn* p, const std::vector<_sn>& buf, int d ); 

    static constexpr const bool LEAK = false ; 


    sn(int type, sn* left, sn* right);
#ifdef WITH_CHILD
    void add_child( sn* ch ); 
#endif

    ~sn(); 


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

    int checktree() const ; 
    int checktree_r(char code,  int d ) const ; 

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

    static sn* CommonOperatorTypeTree( std::vector<int>& leaftypes,  int op ); 
    static sn* CommonOperatorTree(     std::vector<sn*>& leaves    , int op ); 


    void populate_leaftypes(std::vector<int>& leaftypes ); 
    void populate_leaves(   std::vector<sn*>& leaves ); 


    void prune(); 
    void prune_r(int d) ; 
    bool has_dangle() const ; 

    void positivize() ; 
    void positivize_r(bool negate, int d) ; 

    void set_lvid(int lvid_); 
    void set_lvid_r(int lvid_, int d); 



    void setPA( double x, double y, double z, double w, double z1, double z2 );
    void setBB(  double x0, double y0, double z0, double x1, double y1, double z1 ); 

    void setXF(     const glm::tmat4x4<double>& t ); 
    void setXF(     const glm::tmat4x4<double>& t, const glm::tmat4x4<double>& v ) ; 
    void combineXF( const glm::tmat4x4<double>& t ); 
    void combineXF( const glm::tmat4x4<double>& t, const glm::tmat4x4<double>& v ) ; 



    
    /**
    considered returing (int)nd::index here to match snd.hh 
    but that would not be workable once deleting  
    any sn nodes as the indices would change
    **/

    static sn* Cylinder(double radius, double z1, double z2) ;
    static sn* Cone(double r1, double z1, double r2, double z2);
    static sn* Sphere(double radius); 
    static sn* ZSphere(double radius, double z1, double z2);
    static sn* Box3(double fullside);
    static sn* Box3(double fx, double fy, double fz );
    static sn* Zero(double  x,  double y,  double z,  double w,  double z1, double z2);
    static sn* Zero();
    static sn* Prim(int type) ; 
    static sn* Create(int type, sn* left=nullptr, sn* right=nullptr ); 
    static sn* Boolean( int op, sn* l, sn* r );

    static void ZNudgeEnds(  std::vector<sn*>& prims); 
    static void ZNudgeJoints(std::vector<sn*>& prims); 
    static std::string ZDesc(const std::vector<sn*>& prims); 

    double zmin() const ; 
    double zmax() const ; 


    static sn* Collection( std::vector<sn*>& prims ); 
    static sn* UnionTree(  std::vector<sn*>& prims ); 
    static sn* Contiguous( std::vector<sn*>& prims ); 
    static sn* Compound(   std::vector<sn*>& prims, int type_ ); 


};







inline std::string sn::Desc(const char* msg){ return pool ? pool->desc(msg) : "-" ; }
inline int         sn::level() {  return pool ? pool->level : ssys::getenvint("sn__level",-1) ; } // static 

inline int         sn::index() const { return pool ? pool->index(this) : -1 ; }
inline bool        sn::is_root_importable() const { return parent == nullptr ; }


inline int sn::num_child() const
{
#ifdef WITH_CHILD
    return int(child.size()); 
#else
    return left && right ? 2 : 0 ; 
#endif
}

inline sn* sn::first_child() const 
{
#ifdef WITH_CHILD
    return child.size() > 0 ? child[0] : nullptr ; 
#else
    return left ; 
#endif
}
inline sn* sn::last_child() const 
{
#ifdef WITH_CHILD
    return child.size() > 0 ? child[child.size()-1] : nullptr ; 
#else
    return right ; 
#endif
}
inline sn* sn::get_child(int ch) const 
{
#ifdef WITH_CHILD
    return ch > -1 && ch < int(child.size()) ? child[ch] : nullptr ; 
#else
    switch(ch)
    {
        case 0: return left  ; break ; 
        case 1: return right ; break ; 
    }
    return nullptr ; 
#endif
}


inline int sn::total_siblings() const
{
#ifdef WITH_CHILD
    return parent ? int(parent->child.size()) : 1 ;  // root regarded as sole sibling (single child)  
#else
    if(parent == nullptr) return 1 ; 
    return ( parent->left && parent->right ) ? 2 : -1 ;   
#endif
}

inline int sn::child_index( const sn* ch )
{
#ifdef WITH_CHILD
    size_t idx = std::distance( child.begin(), std::find( child.begin(), child.end(), ch )) ; 
    return idx < child.size() ? idx : -1 ; 
#else
    int idx = -1 ; 
    if(      ch == left )  idx = 0 ; 
    else if( ch == right ) idx = 1 ; 
    return idx ; 
#endif
}

inline int sn::sibling_index() const 
{
    int tot_sib = total_siblings() ; 
    int sibdex = parent == nullptr ? 0 : parent->child_index(this) ; 

    if(level() > 1) std::cerr << "sn::sibling_index"
              << " tot_sib " << tot_sib 
              << " sibdex " << sibdex
              << std::endl 
              ;

    assert( sibdex < tot_sib ); 
    return sibdex ;  
}

inline const sn* sn::get_sibling(int sx) const     // NB this return self for appropriate sx
{
#ifdef WITH_CHILD
    assert( sx < total_siblings() ); 
    return parent ? parent->child[sx] : this ; 
#else
    const sn* sib = nullptr ; 
    switch(sx)
    {
        case 0: sib = parent ? parent->left  : nullptr ; break ; 
        case 1: sib = parent ? parent->right : nullptr ; break ; 
    }
    return sib ; 
#endif
}

inline const sn* sn::next_sibling() const
{
    int next_sib = 1+sibling_index() ; 
    int tot_sib = total_siblings() ; 

    if(level() > 1) std::cerr << "sn::next_sibling" 
              << " tot_sib " << tot_sib
              << " next_sib " << next_sib 
              << std::endl 
              ;
 
    return next_sib < tot_sib  ? get_sibling(next_sib) : nullptr ;   
}





/**
sn::Serialize
--------------

The Serialize operates by converting pointer members into pool indices 
This T::Serialize is invoked from s_pool<T,P>::serialize_ 
for with paired T and P for all pool objects. 

At first glance this looks like the WITH_CHILD vector of child nodes
is restricted to working with two children, but that is not the case 
because the the full vector in the T pool gets represented via next_sibling
links in the P buffer allowing any number of child nodes to be handled. 
This functionality is needed for multiunion.  

**/

inline void sn::Serialize(_sn& n, const sn* x) // static 
{
    assert( pool      && "sn::pool  is required for sn::Serialize" );    
    assert( s_tv::pool && "s_tv::pool is required for sn::Serialize" ); 
    assert( s_pa::pool && "s_pa::pool is required for sn::Serialize" ); 
    assert( s_bb::pool && "s_bb::pool is required for sn::Serialize" ); 

    n.type = x->type ; 
    n.complement = x->complement ;
    n.lvid = x->lvid ;

    n.tv = s_tv::pool->index(x->tv) ;  
    n.pa = s_pa::pool->index(x->pa) ;  
    n.bb = s_bb::pool->index(x->bb) ;  
    n.parent = pool->index(x->parent);  

#ifdef WITH_CHILD
    n.sibdex = x->sibling_index(); 
    n.num_child = x->num_child() ; 
    n.first_child = pool->index(x->first_child());  
    n.next_sibling = pool->index(x->next_sibling()); 
#else
    n.left  = pool->index(x->left);  
    n.right = pool->index(x->right);  
#endif
}

/**
sn::Import
-----------

Used by s_pool<T,P>::import_ in a loop providing 
pointers to every entry in the vector buf.  
However only root_importable _sn nodes with parent -1 
get recursively imported. 

**/

inline sn* sn::Import( const _sn* p, const std::vector<_sn>& buf ) // static
{
    if(level() > 0) std::cerr << "sn::Import" << std::endl ; 
    return p->is_root_importable() ? Import_r(p, buf, 0) : nullptr ; 
}

/**
sn::Import_r
-------------

Note that because all _sn nodes are available in the buf 
issues of ordering of Import are avoided. 

**/

inline sn* sn::Import_r(const _sn* _n,  const std::vector<_sn>& buf, int d)
{
    assert( s_tv::pool && "s_tv::pool is required for sn::Import_r " ); 
    if(level() > 0) std::cerr << "sn::Import_r d " << d << " " << ( _n ? _n->desc() : "(null)" ) << std::endl ; 
    if(_n == nullptr) return nullptr ; 

#ifdef WITH_CHILD
    sn* n = Create( _n->type , nullptr, nullptr );  
    n->complement = _n->complement ; 
    n->lvid = _n->lvid ; 
    n->tv = s_tv::pool->get(_n->tv) ; 
    n->pa = s_pa::pool->get(_n->pa) ; 
    n->bb = s_bb::pool->get(_n->bb) ; 

    const _sn* _child = _n->first_child  > -1 ? &buf[_n->first_child] : nullptr  ; 

    while( _child ) 
    {    
        sn* ch = Import_r( _child, buf, d+1 ); 
        n->add_child(ch);  // push_back and sets *ch->parent* to *n* 
        _child = _child->next_sibling > -1 ? &buf[_child->next_sibling] : nullptr ;
    }    
#else
    const _sn* _l = _n->left  > -1 ? &buf[_n->left]  : nullptr ;  
    const _sn* _r = _n->right > -1 ? &buf[_n->right] : nullptr ;  
    sn* l = Import_r( _l, buf, d+1 ); 
    sn* r = Import_r( _r, buf, d+1 ); 
    sn* n = Create( _n->type, l, r );  // sn::sn ctor sets parent of l and r to n 
    n->complement = _n->complement ; 
    n->lvid = _n->lvid ; 
    n->tv = s_tv::pool->get(_n->tv) ; 
    n->pa = s_pa::pool->get(_n->pa) ; 
    n->bb = s_bb::pool->get(_n->bb) ; 
#endif
    return n ;  
}  




// ctor

inline sn::sn(int type_, sn* left_, sn* right_)
    :
    type(type_),
    complement(0),
    lvid(-1),
    tv(nullptr),
    pa(nullptr),
    bb(nullptr),
    parent(nullptr),
#ifdef WITH_CHILD
#else
    left(left_),
    right(right_),
#endif
    pid(pool ? pool->add(this) : -1),
    depth(0),
    subdepth(0)
{
    if(level() > 1) std::cerr << "sn::sn pid " << pid << std::endl ; 

#ifdef WITH_CHILD
    if(left_ && right_)
    {
        add_child(left_);   // sets parent of left_ to this
        add_child(right_);  // sets parent of right_ to this
    }
#else
    if(left && right)
    {
        left->parent = this ; 
        right->parent = this ; 
    }
#endif
}

#ifdef WITH_CHILD
inline void sn::add_child( sn* ch )
{
    assert(ch); 
    ch->parent = this ; 
    child.push_back(ch) ; 
}
#endif





// dtor 
inline sn::~sn()   
{
    if(level() > 1) std::cerr << "[ sn::~sn pid " << pid << std::endl ; 

    delete tv ; 


#ifdef WITH_CHILD
    for(int i=0 ; i < int(child.size()) ; i++)
    {
        sn* ch = child[i] ; 
        delete ch ;  
    }
#else
    delete left ; 
    delete right ; 
#endif

    if(pool) pool->remove(this); 

    if(level() > 1) std::cerr << "] sn::~sn pid " << pid << std::endl ; 
}







#ifdef WITH_CHILD
/**
sn::disown_child
------------------

Note that the erase calls the dtor which 
will also delete child nodes (recursively)
and removes pool entries. 

**/
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
    copy->left  = left  ? left->deepcopy_r(d+1) : nullptr ; 
    copy->right = right ? right->deepcopy_r(d+1) : nullptr ;   
#endif
    copy->parent = nullptr ; 

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
    new_ch->parent = this ; 

#ifdef WITH_CHILD
    assert( ix < int(child.size()) );   
    sn*& target = child[ix] ;
    if(!LEAK) delete target ;
    target = new_ch ; 
#else
    sn** target = ix == 0 ? &left : &right ;  
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
    return left == nullptr && right == nullptr ;
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
    return !is_primitive() && left->is_primitive() && right->is_primitive() ;
#endif
}   
inline bool sn::is_operator() const 
{   
#ifdef WITH_CHILD
    return child.size() == 2 ;  
#else
    return left != nullptr && right != nullptr ;
#endif
}
inline bool sn::is_zero() const 
{   
    return type == 0 ;  
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
    return is_operator() && left->is_zero() && right->is_zero() ;
#endif
}
inline bool sn::is_rzero() const
{
#ifdef WITH_CHILD
    return is_operator() && !child[0]->is_zero() && child[1]->is_zero() ; 
#else
    return is_operator() && !left->is_zero() && right->is_zero() ; 
#endif
}
inline bool sn::is_lzero() const 
{   
#ifdef WITH_CHILD
    return is_operator() && child[0]->is_zero() && !child[1]->is_zero() ;
#else
    return is_operator() && left->is_zero() && !right->is_zero() ;
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
    nn += left ? left->num_node_r(d+1) : 0 ; 
    nn += right ? right->num_node_r(d+1) : 0 ; 
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
    nl += left ? left->num_leaf_r(d+1) : 0 ; 
    nl += right ? right->num_leaf_r(d+1) : 0 ; 
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
    return left && right ? std::max( left->maxdepth_r(d+1), right->maxdepth_r(d+1)) : d ; 
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

    int nc = num_child();
    if(nc == 0) return d ;  

    int mx = 0 ; 
    for(int i=0 ; i < nc ; i++) 
    {
        sn* ch = get_child(i) ; 
        mx = std::max(mx, ch->maxdepth_label_r(d+1) ) ; 
    }
    return mx ; 
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
The above tree has two bileafs, one other leaf and root. 

**/

inline void sn::subdepth_label()
{
    subdepth_label_r(0); 
}
inline void sn::subdepth_label_r(int d)
{
    subdepth = maxdepth() ;
    for(int i=0 ; i < num_child() ; i++) 
    {
        sn* ch = get_child(i) ; 
        ch->subdepth_label_r(d+1) ; 
    }
}


inline int sn::checktree() const
{
    int chk_D = checktree_r('D', 0);  
    int chk_P = checktree_r('P', 0);  
    int chk = chk_D + chk_P ; 

    if( chk > 0 )  
    {    
        if(level()>0) std::cerr 
            << "sn::checktree"
            << " chk_D " << chk_D
            << " chk_P " << chk_P
            << desc()
            << std::endl
            ;    
    }    
    return chk ; 
}


inline int sn::checktree_r(char code,  int d ) const
{
    int chk = 0 ;

    if( code == 'D' ) // check expected depth
    {
        if(d != depth) chk += 1 ;
    }
    else if( code == 'P' ) // check for non-roots without parent set 
    {
        if( depth > 0 && parent == nullptr ) chk += 1 ;
    }

    for(int i=0 ; i < num_child() ; i++) 
    {
        sn* ch = get_child(i) ; 
        ch->checktree_r(code, d+1) ; 
    }

    return chk ;
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
        switch( type )
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
    if(left && right )
    {   
        operators_v(mask, minsubdepth );
        left->operators_r( mask, minsubdepth );  
        right->operators_r( mask, minsubdepth );  
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
    if(left) left->preorder_r(order, d+1) ; 
    if(right) right->preorder_r(order, d+1) ; 
#endif
}

/**
sn::inorder_r
-------------

**/

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
    if(left) left->inorder_r(order, d+1) ; 
    order.push_back(this); 
    if(right) right->inorder_r(order, d+1) ; 
#endif
}
inline void sn::postorder_r(std::vector<const sn*>& order, int d ) const
{
#ifdef WITH_CHILD
    for(int i=0 ; i < int(child.size()) ; i++) child[i]->postorder_r(order, d+1) ; 
#else
    if(left) left->postorder_r(order, d+1) ; 
    if(right) right->postorder_r(order, d+1) ; 
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
    if(left) left->inorder_r_(order, d+1) ; 
    order.push_back(this); 
    if(right) right->inorder_r_(order, d+1) ; 
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
       << " type " << std::setw(3) << type 
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
    if( left && right )
    {
        left->desc_r(d+1, ss); 
        right->desc_r(d+1, ss); 
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
    std::string tag = CSG::Tag(type, complement == 1); 

    switch(mode)
    {
        case 0: canvas->drawch( ix, iy, 0,0, 'o' )         ; break ; 
        case 1: canvas->draw(   ix, iy, 0,0,  type  )      ; break ; 
        case 2: canvas->draw(   ix, iy, 0,0,  depth )      ; break ;   
        case 3: canvas->draw(   ix, iy, 0,0,  subdepth )   ; break ; 
        case 4: canvas->draw(   ix, iy, 0,0,  tag.c_str()) ; break ;    
        case 5: canvas->draw(   ix, iy, 0,0,  pid )        ; break ;    
    } 

#ifdef WITH_CHILD
    for(int i=0 ; i < int(child.size()) ; i++) child[i]->render_r(canvas, order, mode, d+1) ; 
#else
    if(left)  left->render_r( canvas, order, mode, d+1 );
    if(right) right->render_r( canvas, order, mode, d+1 );
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
    if(level() > 0 ) std::cerr << "[sn::ZeroTree num_leaves " << num_leaves << " height " << height << std::endl; 
    sn* root = ZeroTree_r( height, op );
    if(level() > 0) std::cerr << "]sn::ZeroTree " << std::endl ; 
    return root ; 
}          

/**
sn::CommonOperatorTypeTree (formerly sn::CommonTree)
-------------------------------------------------------

This was implemented while sn was not fully featured.
It was used to provide a "template" tree with types only, 
to be used for form snd trees. 

**/

inline sn* sn::CommonOperatorTypeTree( std::vector<int>& leaftypes, int op ) // static
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

        if(level() > 0) std::cerr << "sn::CommonOperatorTypeTree ZeroTree num_leaves " << num_leaves << std::endl ; 
        if(level() > 1) std::cerr << root->render(5) ; 

        root->populate_leaftypes(leaftypes); 

        if(level() > 0) std::cerr << "sn::CommonOperatorTypeTree populated num_leaves " << num_leaves << std::endl ; 
        if(level() > 1) std::cerr << root->render(5) ; 

        root->prune();
 
        if(level() > 0) std::cerr << "sn::CommonOperatorTypeTree pruned num_leaves " << num_leaves << std::endl ; 
        if(level() > 1) std::cerr << root->render(5) ; 
    }
    return root ; 
} 



inline sn* sn::CommonOperatorTree( std::vector<sn*>& leaves, int op ) // static
{   
    int num_leaves = leaves.size(); 
    sn* root = nullptr ; 
    if( num_leaves == 1 )
    {
        root = leaves[0] ; 
    }
    else
    {
        root = ZeroTree(num_leaves, op );   

        if(level() > 0) std::cerr << "sn::CommonOperatorTree ZeroTree num_leaves " << num_leaves << std::endl ; 
        if(level() > 1) std::cerr << root->render(5) ; 

        root->populate_leaves(leaves); 

        if(level() > 0) std::cerr << "sn::CommonOperatorTree populated num_leaves " << num_leaves << std::endl ; 
        if(level() > 1) std::cerr << root->render(5) ; 

        root->prune();
 
        if(level() > 0) std::cerr << "sn::CommonOperatorTree pruned num_leaves " << num_leaves << std::endl ; 
        if(level() > 1) std::cerr << root->render(5) ; 
    }
    return root ; 
} 















/**
sn::populate_leaftypes
-------------------------

Replacing zeros with leaftype nodes (not fully featured ones).

**/
        
inline void sn::populate_leaftypes(std::vector<int>& leaftypes )
{
    int num_leaves = leaftypes.size(); 
    int num_leaves_placed = 0 ; 

    std::vector<sn*> order ; 
    inorder_(order) ; 

    int num_nodes = order.size(); 

    if(level() > 0) std::cerr 
        << "sn::populate_leaftypes"
        << " num_leaves " << num_leaves
        << " num_nodes " << num_nodes
        << std::endl
        ;

    for(int i=0 ; i < num_nodes ; i++)
    {
        sn* n = order[i]; 
        if(level() > 1) std::cerr 
            << "sn::populate_leaftypes " << std::setw(3) << i 
            << " n.desc " << n->desc()
            << std::endl 
            ; 
    }

    for(int i=0 ; i < num_nodes ; i++)
    {
        sn* n = order[i]; 

#ifdef WITH_CHILD
        if(level() > 1) std::cerr 
            << "sn::populate_leaftypes"
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
                if(level() > 1 ) std::cerr 
                    << "sn::populate_leaftypes"
                    << " ch.desc " << ch->desc() 
                    << std::endl 
                    ; 

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
            if(n->left->is_zero() && num_leaves_placed < num_leaves)
            {
                n->set_left( sn::Prim(leaftypes[num_leaves_placed]), false ) ; 
                num_leaves_placed += 1 ; 
            }    
            if(n->right->is_zero() && num_leaves_placed < num_leaves)
            {
                n->set_right(sn::Prim(leaftypes[num_leaves_placed]), false ) ;
                num_leaves_placed += 1 ; 
            }    
        }
#endif
    } 
    assert( num_leaves_placed == num_leaves ); 
}





/**
sn::populate_leaves
---------------------

Replacing zeros with fully featured leaf nodes. 

**/
        
inline void sn::populate_leaves(std::vector<sn*>& leaves )
{
    int num_leaves = leaves.size(); 
    int num_leaves_placed = 0 ; 

    std::vector<sn*> order ; 
    inorder_(order) ;   // these all all nodes of the tree, not just leaves

    int num_nodes = order.size(); 

    if(level() > 0) std::cerr 
        << "sn::populate_leaves"
        << " num_leaves " << num_leaves
        << " num_nodes " << num_nodes
        << std::endl
        ;

    for(int i=0 ; i < num_nodes ; i++)
    {
        sn* n = order[i]; 
        if(level() > 1) std::cerr 
            << "sn::populate_leaves " << std::setw(3) << i 
            << " n.desc " << n->desc()
            << std::endl 
            ; 
    }

    for(int i=0 ; i < num_nodes ; i++)
    {
        sn* n = order[i]; 

#ifdef WITH_CHILD
        if(level() > 1) std::cerr 
            << "sn::populate_leaves"
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
                if(level() > 1 ) std::cerr 
                    << "sn::populate_leaves"
                    << " ch.desc " << ch->desc() 
                    << std::endl 
                    ; 

                if( ch->is_zero() && num_leaves_placed < num_leaves )
                {
                    n->set_child(j, leaves[num_leaves_placed], false) ;  
                    num_leaves_placed += 1 ; 
                } 
            } 
        }
#else
        if(n->is_operator())
        {
            if(n->left->is_zero() && num_leaves_placed < num_leaves)
            {
                n->set_left( leaves[num_leaves_placed], false ) ; 
                num_leaves_placed += 1 ; 
            }    
            if(n->right->is_zero() && num_leaves_placed < num_leaves)
            {
                n->set_right( leaves[num_leaves_placed], false ) ;
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
        if(level() > -1) std::cerr << "sn::prune ERROR left with dangle " << std::endl ; 
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

    left->prune_r(d+1);
    right->prune_r(d+1);
    
    // postorder visit : so both children always visited before their parents 
    
    if(left->is_lrzero())         // left node is an operator which has both its left and right zero 
    {   
        set_left(sn::Zero(), false) ;       // prune : ie replace operator with CSG_ZERO placeholder  
    }
    else if( left->is_rzero() )   // left node is an operator with left non-zero and right zero   
    {  
        set_left(left->left, true) ;          // moving the lonely primitive up to higher elevation   
    }
    
    if(right->is_lrzero())        // right node is operator with both its left and right zero 
    {   
        set_right(sn::Zero(), false) ;      // prune
    }
    else if( right->is_rzero() )  // right node is operator with its left non-zero and right zero
    {   
        set_right(right->left, true) ;         // moving the lonely primitive up to higher elevation   
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
    return is_operator() && ( right->is_zero() || left->is_zero()) ; 
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

        if(type == CSG_INTERSECTION || type == CSG_UNION)
        {   
            if(negate)                             // !( A*B ) ->  !A + !B       !(A + B) ->     !A * !B
            {    
                type = CSG::DeMorganSwap(type) ;   // UNION->INTERSECTION, INTERSECTION->UNION
                left_negate = true ; 
                right_negate = true ; 
            }   
            else
            {                                      //  A * B ->  A * B         A + B ->  A + B
                left_negate = false ;
                right_negate = false ;
            }
        }
        else if(type == CSG_DIFFERENCE)
        {
            if(negate)                             //  !(A - B) -> !(A*!B) -> !A + B
            {
                type = CSG_UNION ;
                left_negate = true ;
                right_negate = false  ;
            }
            else
            {
                type = CSG_INTERSECTION ;    //    A - B ->  A * !B
                left_negate = false ;
                right_negate = true ;
            }
        }

#ifdef WITH_CHILD
        assert( child.size() == 2 ); 
        sn* left = child[0] ; 
        sn* right = child[1] ; 
#endif
        left->positivize_r(left_negate,  d+1);
        right->positivize_r(right_negate, d+1);
    }
}


inline void sn::set_lvid(int lvid_)
{
    set_lvid_r(lvid_, 0);  

    int chk = checktree();
    if( chk != 0 )
    {
        if(level() > 0 ) std::cerr
           << "sn::set_lvid"
           << " lvid " << lvid_
           << " checktree " << chk
           << std::endl
           ;
    }
    assert( chk == 0 );
}
inline void sn::set_lvid_r(int lvid_, int d)
{
    lvid = lvid_ ; 
    for(int i=0 ; i < num_child() ; i++)
    {
        sn* ch = get_child(i) ;       
        ch->set_lvid_r(lvid_, d+1 ); 
    }
}









inline void sn::setPA( double x0, double y0, double z0, double w0, double x1, double y1 )
{
    if( pa == nullptr ) pa = new s_pa ; 
    pa->x0 = x0 ; 
    pa->y0 = y0 ; 
    pa->z0 = z0 ; 
    pa->w0 = w0 ; 
    pa->x1 = x1 ; 
    pa->y1 = y1 ; 
}

inline void sn::setBB( double x0, double y0, double z0, double x1, double y1, double z1 )
{
    if( bb == nullptr ) bb = new s_bb ; 
    bb->x0 = x0 ; 
    bb->y0 = y0 ; 
    bb->z0 = z0 ; 
    bb->x1 = x1 ; 
    bb->y1 = y1 ; 
    bb->z1 = z1 ; 
}


inline void sn::setXF( const glm::tmat4x4<double>& t )
{
    glm::tmat4x4<double> v = glm::inverse(t) ; 
    setXF(t, v); 
}
inline void sn::combineXF( const glm::tmat4x4<double>& t )
{
    glm::tmat4x4<double> v = glm::inverse(t) ; 
    combineXF(t, v); 
}
inline void sn::setXF( const glm::tmat4x4<double>& t, const glm::tmat4x4<double>& v )
{
    if( tv == nullptr ) tv = new s_tv ; 
    tv->t = t ; 
    tv->v = v ; 
}
inline void sn::combineXF( const glm::tmat4x4<double>& t, const glm::tmat4x4<double>& v )
{
    if( tv == nullptr )
    {
        tv = new s_tv ; 
        tv->t = t ; 
        tv->v = v ; 
    }
    else
    {
        glm::tmat4x4<double> tt = tv->t * t ;   
        glm::tmat4x4<double> vv = v * tv->v ;   
        tv->t = tt ; 
        tv->v = vv ; 
    }
}




inline sn* sn::Cylinder(double radius, double z1, double z2) // static
{   
    assert( z2 > z1 );  
    sn* nd = Create(CSG_CYLINDER); 
    nd->setPA( 0.f, 0.f, 0.f, radius, z1, z2)  ;   
    nd->setBB( -radius, -radius, z1, +radius, +radius, z2 );
    return nd ;
}
inline sn* sn::Cone(double r1, double z1, double r2, double z2)  // static
{   
    assert( z2 > z1 );
    double rmax = fmax(r1, r2) ;
    sn* nd = Create(CSG_CONE) ;
    nd->setPA( r1, z1, r2, z2, 0., 0. ) ;
    nd->setBB( -rmax, -rmax, z1, rmax, rmax, z2 );
    return nd ;
}
inline sn* sn::Sphere(double radius)  // static
{
    assert( radius > zero );
    sn* nd = Create(CSG_SPHERE) ;
    nd->setPA( zero, zero, zero, radius, zero, zero );
    nd->setBB(  -radius, -radius, -radius,  radius, radius, radius  );
    return nd ;
}
inline sn* sn::ZSphere(double radius, double z1, double z2)  // static
{
    assert( radius > zero ); 
    assert( z2 > z1 );   
    sn* nd = Create(CSG_ZSPHERE) ; 
    nd->setPA( zero, zero, zero, radius, z1, z2 );   
    nd->setBB(  -radius, -radius, z1,  radius, radius, z2  );   
    return nd ;
}
inline sn* sn::Box3(double fullside)  // static 
{
    return Box3(fullside, fullside, fullside); 
}
inline sn* sn::Box3(double fx, double fy, double fz )  // static 
{
    assert( fx > 0. );   
    assert( fy > 0. );   
    assert( fz > 0. );   

    sn* nd = Create(CSG_BOX3) ; 
    nd->setPA( fx, fy, fz, 0.f, 0.f, 0.f );   
    nd->setBB( -fx*0.5 , -fy*0.5, -fz*0.5, fx*0.5 , fy*0.5, fz*0.5 );   
    return nd ;
}

inline sn* sn::Zero(double  x,  double y,  double z,  double w,  double z1, double z2) // static 
{
    sn* nd = Create(CSG_ZERO); 
    nd->setPA( x, y, z, w, z1, z2 );  
    return nd ; 
}   
inline sn* sn::Zero() // static
{   
    sn* nd = Create(CSG_ZERO); 
    return nd ; 
}
inline sn* sn::Prim(int type_)   // static
{
    return new sn(type_, nullptr, nullptr) ; 
}
inline sn* sn::Create(int type_, sn* left_, sn* right_)  // static
{
    sn* nd = new sn(type_, left_, right_) ;
    return nd ;
}
inline sn* sn::Boolean(int type_, sn* left_, sn* right_)  // static
{
    sn* nd = Create(type_, left_, right_); 
    return nd ; 
}



/**
sn::ZNudgeEnds
-----------------

CAUTION: changes geometry, only appropriate 
for subtracted consituents eg inners 

**/

inline void sn::ZNudgeEnds(std::vector<sn*>& prims) // static
{
    if(level() > 0) std::cout 
       << std::endl
       << "sn::ZNudgeEnds PLACEHOLDER "
       << std::endl
       << ZDesc(prims)
       << std::endl
       ;

    /*
    for(unsigned i=1 ; i < prims.size() ; i++)
    {
        sn* a = prims[i-1]; 
        sn* b = prims[i]; 
        a->check_z(); 
        b->check_z();
    }
    */
}

inline void sn::ZNudgeJoints(std::vector<sn*>& prims) // static
{
    if(level() > 0) std::cout
       << std::endl
       << "sn::ZNudgeJoints PLACEHOLDER "
       << std::endl
       << ZDesc(prims)
       << std::endl
       ;
}





/**
sn::ZDesc
-----------

   +----+
   |    |
   +----+
   |    |
   +----+
   |    |
   +----+

**/

inline std::string sn::ZDesc(const std::vector<sn*>& prims) // static
{
    std::stringstream ss ;
    ss << "sn::ZDesc" ;
    ss << " prims(" ;
    for(unsigned i=0 ; i < prims.size() ; i++) ss << prims[i]->index() << " " ;
    ss << ") " ;
    ss << std::endl ;

    for(unsigned i=0 ; i < prims.size() ; i++)
    {
        sn* a = prims[i];
        ss << std::setw(3) << a->index()
           << ":"
           << " " << std::setw(10) << a->zmin()
           << " " << std::setw(10) << a->zmax()
           << std::endl
           ;
    }
    std::string str = ss.str();
    return str ;
}


inline double sn::zmin() const
{
    assert( CSG::CanZNudge(type) );
    assert( pa );
    return pa ? pa->zmin() : 0. ;
}

inline double sn::zmax() const
{
    assert( CSG::CanZNudge(type) );
    assert( pa );
    return pa ? pa->zmax() : 0. ;
}





/**
sn::Collection
-----------------

Used for example from U4Polycone::init 

+-------------+-------------------+-------------------+
|  VERSION    |  Impl             |  Notes            |
+=============+===================+===================+ 
|     0       |  sn::UnionTree    | backward looking  | 
+-------------+-------------------+-------------------+
|     1       |  sn::Contiguous   | forward looking   |   
+-------------+-------------------+-------------------+

**/

inline sn* sn::Collection(std::vector<sn*>& prims ) // static
{ 
    sn* n = nullptr ; 
    switch(VERSION)
    {   
        case 0: n = UnionTree(prims)  ; break ; 
        case 1: n = Contiguous(prims) ; break ;
    }   
    return n ; 
}

inline sn* sn::UnionTree(std::vector<sn*>& prims )
{
    sn* n = CommonOperatorTree( prims, CSG_UNION ); 
    return n ; 
}



inline sn* sn::Contiguous(std::vector<sn*>& prims )
{
    sn* n = Compound( prims, CSG_CONTIGUOUS ); 
    return n ; 
}



inline sn* sn::Compound(std::vector<sn*>& prims, int type_ )
{
    assert( type_ == CSG_CONTIGUOUS || type_ == CSG_DISCONTIGUOUS ); 

    int num_prim = prims.size(); 
    assert( num_prim > 0 ); 

    sn* nd = Create( type_ ); 

    for(int i=0 ; i < num_prim ; i++)
    {
        sn* pr = prims[i] ; 
#ifdef WITH_CHILD
        nd->add_child(pr) ; 
#else
        assert(0 && "sn::Compound requires WITH_CHILD " ); 
        assert(num_prim == 2 ); 
        if(i==0) nd->set_left(pr,  false) ; 
        if(i==1) nd->set_right(pr, false) ; 
#endif
    }
    return nd ; 
}

