#pragma once
/**
sn.h : minimal pointer based transient binary tree node
========================================================

Motivation
-----------

In order to duplicate at CSG/CSGNode level the old workflow geometry 
(that goes thru GGeo/NNode) it is necessary to perform binary tree 
manipulations equivalent to those done by npy/NTreeBuilder::UnionTree 
in order to handle shapes such as G4Polycone. 

However the old array based *snd/scsg* node approach with integer index 
addressing lacks the capability to easily delete nodes making it unsuitable
for tree manipulations such as pruning and rearrangement that are needed 
in order to flexibly create complete binary trees with any number of leaf nodes.

Hence the *sn* nodes are developed. Initially sn.h was used as transient 
template for binary trees that are subsequently solidified into *snd* trees. 
But have now moved most snd functionality over to sn. So can directly use 
only sn and eliminate the old WITH_SND. 

sn ctor/dtor register/de-register from s_pool<sn,_sn> 
-------------------------------------------------------

In order to convert active *sn* pointers into indices 
on persisting have explictly avoided leaking any *sn* by 
taking care to always delete appropriately. 
This means that can use the *sn* ctor/dtor to add/erase update 
an std::map of active *sn* pointers keyed on a creation index.  
This map allows the active *sn* pointers to be converted into 
a contiguous set of indices to facilitate serialization. 

Possible Future
-----------------

CSG_CONTIGUOUS could keep n-ary CSG trees all the way to the GPU

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

//#include "s_csg.h" // DONT DO THAT : CIRCULAR 
#include "st.h"      // complete binary tree math 
#include "stra.h"    // glm transform utilities 

#include "NPFold.h"

struct _sn
{
    int  typecode ;     // 0
    int  complement ;   // 1 
    int  lvid ;         // 2
    int  xform ;        // 3
    int  param ;        // 4
    int  aabb ;         // 5 
    int  parent ;       // 6 
    
#ifdef WITH_CHILD
    int  sibdex ;       // 7     0-based sibling index 
    int  num_child ;    // 8
    int  first_child ;  // 9
    int  next_sibling ; // 10  
    int  index ;        // 11 
    int  depth ;        // 12
    char label[16] ;    // 13,14,15,16 
    static constexpr const char* ITEM = "17" ;  
#else
    int  left ;         // 7
    int  right ;        // 8
    int  index ;        // 9
    int  depth ;        // 10
    char label[16] ;    // 11,12,13,14 
    static constexpr const char* ITEM = "15" ;  
#endif

    std::string desc() const ; 
    bool is_root() const ; 
};

inline std::string _sn::desc() const
{
    std::stringstream ss ; 
    ss << "_sn::desc " 
       << " typecode " << std::setw(4) << typecode 
       << " complement " << std::setw(1) << complement
       << " lvid " << std::setw(4) << lvid 
       << " xform " << std::setw(4) << xform
       << " param " << std::setw(4) << param
       << " aabb " << std::setw(4) << aabb
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
       << " is_root " << ( is_root() ? "YES" : "NO " ) 
       ;
    std::string str = ss.str(); 
    return str ; 
}

/**
_sn::is_root
------------

Only root expected to have parent -1 

**/
inline bool _sn::is_root() const 
{
    return parent == -1 ;  
}

#include "SYSRAP_API_EXPORT.hh"
struct SYSRAP_API sn
{
    // persisted
    int   typecode ; 
    int   complement ; 
    int   lvid ; 
    s_tv* xform ;    
    s_pa* param ; 
    s_bb* aabb  ;
    sn*   parent ;   // NOT owned by this sn 

#ifdef WITH_CHILD
    std::vector<sn*> child ;   
#else
    sn* left ;          
    sn* right ;        
#endif
    int depth ;    
    char label[16] ; 

    // internals, not persisted  
    int pid ;       
    int subdepth ; 


    typedef s_pool<sn,_sn> POOL ;
    static POOL* pool ;  
    static constexpr const int VERSION = 0 ;
    static constexpr const char* NAME = "sn.npy" ; 
    static constexpr const double zero = 0. ; 
    static constexpr const double Z_EPSILON = 1e-3 ; 

    static void SetPOOL( POOL* pool_ ); 
    static int level(); 
    static std::string Desc(); 

    template<typename N>
    static std::string Desc(N* n); 

    template<typename N>
    static std::string Desc(const std::vector<N*>& nds); 


    int  idx() const ;  // to match snd.hh
    int  index() const ; 
    bool is_root() const ; 

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


    sn(int typecode, sn* left, sn* right);
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

    void labeltree(); 

    int labeltree_maxdepth() ; 
    int labeltree_maxdepth_r(int d) ; 

    void labeltree_subdepth() ; 
    void labeltree_subdepth_r(int d); 

    int checktree() const ; 
    int checktree_r(char code,  int d ) const ; 

    unsigned operators(int minsubdepth) const ; 
    void operators_v(unsigned& mask, int minsubdepth) const ; 
    void operators_r(unsigned& mask, int minsubdepth) const ; 
    bool is_positive_form() const ; 
    bool is_listnode() const ; 
    std::string tag() const ; 

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
    std::string brief() const ; 
    std::string desc_child() const ; 
    std::string desc_r() const ; 
    void desc_r(int d, std::stringstream& ss) const ; 

    std::string render() const ; 
    std::string render(int mode) const ; 

    enum { MINIMAL, TYPECODE, DEPTH, SUBDEPTH, TYPETAG, PID } ; 

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

    static sn* CommonOperatorTypeTree(   std::vector<int>& leaftypes,  int op ); 

    void populate_leaftypes(std::vector<int>& leaftypes ); 
    void populate_leaves(   std::vector<sn*>& leaves ); 


    void prune(); 
    void prune_r(int d) ; 
    bool has_dangle() const ; 

    void positivize() ; 
    void positivize_r(bool negate, int d) ; 

    void zero_label(); 
    void set_label( const char* label_ ); 
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
    static sn* Prim(int typecode) ; 
    static sn* Create(int typecode, sn* left=nullptr, sn* right=nullptr ); 
    static sn* Boolean( int op, sn* l, sn* r );

    static void ZNudgeExpandEnds(  std::vector<sn*>& prims, bool enable); 
    static void ZNudgeOverlapJoints(std::vector<sn*>& prims, bool enable); 

    bool can_znudge() const ; 
    static bool CanZNudgeAll(std::vector<sn*>& prims); 

    void increase_zmax( double dz ); // expand upwards in +Z direction 
    void decrease_zmin( double dz ); // expand downwards in -Z direction
    double zmin() const ; 
    double zmax() const ; 
    void set_zmin(double zmin_) ; 
    void set_zmax(double zmax_) ; 

    double rperp_at_zmax() const ; 
    double rperp_at_zmin() const ; 

    static std::string ZDesc(const std::vector<sn*>& prims); 

    const double* getParam() const ; 
    const double* getAABB() const ; 
    bool hasAABB() const ;  // not-nullptr and not all zero 


    static sn* Collection( std::vector<sn*>& prims ); 
    static sn* UnionTree(  std::vector<sn*>& prims ); 
    static sn* Contiguous( std::vector<sn*>& prims ); 
    static sn* Compound(   std::vector<sn*>& prims, int typecode_ ); 

    static sn* Buggy_CommonOperatorTree( std::vector<sn*>& leaves    , int op ); 
    static sn* BuildCommonTypeTree_Unbalanced( const std::vector<sn*>& leaves, int typecode ); 

    static void GetLVNodes( std::vector<sn*>& nds, int lvid ); 
    void getLVNodes( std::vector<sn*>& nds ) const ;
    static bool Includes(const std::vector<sn*>& nds, sn* nd ); 

    static sn* Get(int idx); 
    static sn* GetLVRoot( int lvid ); 

    std::string rbrief() const ; 
    void rbrief_r(std::ostream& os, int d) const  ; 

    bool has_type(const std::vector<OpticksCSG_t>& types) const ; 
    template<typename ... Args> 
    void typenodes_(std::vector<const sn*>& nds, Args ... tcs ) const ; 
    void typenodes_r_(std::vector<const sn*>& nds, const std::vector<OpticksCSG_t>& types, int d) const ; 

    int max_binary_depth() const ; 
    int max_binary_depth_r(int d) const ; 

    int getLVBinNode() const ; 
    int getLVSubNode() const ; 
    int getLVNumNode() const ; 

    static void GetLVNodesComplete(std::vector<const sn*>& nds, int lvid); 
    void        getLVNodesComplete(std::vector<const sn*>& nds) const ; 
    static void GetLVNodesComplete_r(std::vector<const sn*>& nds, const sn* nd, int idx); 

    void ancestors(std::vector<const sn*>& nds) const ; 

    void connectedtype_ancestors(std::vector<const sn*>& nds ) const ; 
    static void ConnectedTypeAncestors(const sn* n, std::vector<const sn*>& nds, int q_typecode); 

    void collect_progeny( std::vector<const sn*>& progeny, int exclude_typecode ) const ; 
    static void CollectProgeny_r( const sn* n, std::vector<const sn*>& progeny, int exclude_typecode ); 

    void collect_monogroup( std::vector<const sn*>& monogroup ) const ; 

    static bool AreFromSameMonogroup(const sn* a, const sn* b, int op); 
    static bool AreFromSameUnion(const sn* a, const sn* b); 






    static void NodeTransformProduct(
        int idx, 
        glm::tmat4x4<double>& t, 
        glm::tmat4x4<double>& v, 
        bool reverse, 
        std::ostream* out); 

    static std::string DescNodeTransformProduct(
        int idx, 
        glm::tmat4x4<double>& t, 
        glm::tmat4x4<double>& v, 
        bool reverse ); 

    void getNodeTransformProduct(
        glm::tmat4x4<double>& t, 
        glm::tmat4x4<double>& v, 
        bool reverse, 
        std::ostream* out) const ; 

    std::string desc_getNodeTransformProduct(
        glm::tmat4x4<double>& t, 
        glm::tmat4x4<double>& v,  
        bool reverse) const ; 

    
};  // END

inline void        sn::SetPOOL( POOL* pool_ ){ pool = pool_ ; }
inline int         sn::level() {  return ssys::getenvint("sn__level",-1) ; } // static 
inline std::string sn::Desc(){    return pool ? pool->desc() : "-" ; } // static

template<typename N>
inline std::string sn::Desc(N* n) // static
{
    return n ? n->desc() : "(null)" ;  
}

template<typename N>
inline std::string sn::Desc(const std::vector<N*>& nds) // static
{
    std::stringstream ss ; 
    ss << "sn::Desc nds.size " << nds.size() << std::endl ; 
    for(int i=0 ; i < int(nds.size()) ; i++) ss << Desc(nds[i]) << std::endl ; 
    std::string str = ss.str();
    return str ; 
}





inline int  sn::idx() const { return index() ; } // to match snd.hh 
inline int  sn::index() const { return pool ? pool->index(this) : -1 ; }
inline bool sn::is_root() const { return parent == nullptr ; }

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
    if(level() > 1) std::cerr 
        << "sn::Serialize ["  
        << std::endl 
        ;

    assert( pool      && "sn::pool  is required for sn::Serialize" );    
    assert( s_tv::pool && "s_tv::pool is required for sn::Serialize" ); 
    assert( s_pa::pool && "s_pa::pool is required for sn::Serialize" ); 
    assert( s_bb::pool && "s_bb::pool is required for sn::Serialize" ); 

    n.typecode = x->typecode ; 
    n.complement = x->complement ;
    n.lvid = x->lvid ;

    n.xform = s_tv::pool->index(x->xform) ;  
    n.param = s_pa::pool->index(x->param) ;  
    n.aabb = s_bb::pool->index(x->aabb) ;  
    n.parent = pool->index(x->parent);  

#ifdef WITH_CHILD
    n.sibdex = x->sibling_index();  // 0 for root 
    n.num_child = x->num_child() ; 
    n.first_child = pool->index(x->first_child());  
    n.next_sibling = pool->index(x->next_sibling()); 
#else
    n.left  = pool->index(x->left);  
    n.right = pool->index(x->right);  
#endif

    n.index = pool->index(x) ; 
    n.depth = x->depth ; 

    assert( sizeof(n.label) == sizeof(x->label) ); 
    strncpy( &n.label[0], x->label, sizeof(x->label) );


    if(level() > 1) std::cerr 
        << "sn::Serialize ]"  
        << std::endl 
        << "(sn)x" 
        << std::endl 
        << x->desc()
        << std::endl 
        << "(_sn)n" 
        << std::endl 
        << n.desc()
        << std::endl 
        ;


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
    return p->is_root() ? Import_r(p, buf, 0) : nullptr ; 
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
    sn* n = Create( _n->typecode , nullptr, nullptr );  
    n->complement = _n->complement ; 
    n->lvid = _n->lvid ; 
    n->xform = s_tv::pool->get(_n->xform) ; 
    n->param = s_pa::pool->get(_n->param) ; 
    n->aabb =  s_bb::pool->get(_n->aabb) ; 

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
    sn* n = Create( _n->typecode, l, r );  // sn::sn ctor sets parent of l and r to n 
    n->complement = _n->complement ; 
    n->lvid = _n->lvid ; 
    n->xform = s_tv::pool->get(_n->xform) ; 
    n->param = s_pa::pool->get(_n->param) ; 
    n->aabb = s_bb::pool->get(_n->aabb) ; 
#endif
    return n ;  
}  




// ctor

inline sn::sn(int typecode_, sn* left_, sn* right_)
    :
    typecode(typecode_),
    complement(0),
    lvid(-1),
    xform(nullptr),
    param(nullptr),
    aabb(nullptr),
    parent(nullptr),
#ifdef WITH_CHILD
#else
    left(left_),
    right(right_),
#endif
    depth(0),
    pid(pool ? pool->add(this) : -1),
    subdepth(0)
{
    if(level() > 1) std::cerr << "sn::sn pid " << pid << std::endl ; 
    zero_label(); 

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

    delete xform ; 


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

/**
sn::deepcopy_r
----------------

The default copy ctor copies the child vector, but that is a shallow copy  
just duplicating pointers into the new child vector. 
Hence within the child loop 
so in the below the shallow copies are disowned and deep copies made and added 
to the copy child vector

**/

inline sn* sn::deepcopy_r(int d) const 
{
    sn* copy = new sn(*this) ;    
#ifdef WITH_CHILD
    for(int i=0 ; i < int(child.size()) ; i++)
    {
        sn* ch = child[i] ; 
        copy->disown_child( ch ) ;          // remove shallow copied child from the vector
        sn* deep_ch = ch->deepcopy_r(d+1) ; 
        copy->child.push_back( deep_ch ); 
    }
#else
    // whether nullptr or not the shallow default copy 
    // should have copied left and right 
    assert( copy->left == left ); 
    assert( copy->right == right ); 
    // but thats just a shallow copy so replace here with deep copies
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
    return typecode == 0 ;  
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



inline void sn::labeltree()
{
    labeltree_maxdepth(); 
    labeltree_subdepth(); 
}

inline int sn::labeltree_maxdepth() 
{
    return labeltree_maxdepth_r(0);
}
inline int sn::labeltree_maxdepth_r(int d)
{
    depth = d ; 

    int nc = num_child();
    if(nc == 0) return d ;  

    int mx = 0 ; 
    for(int i=0 ; i < nc ; i++) 
    {
        sn* ch = get_child(i) ; 
        mx = std::max(mx, ch->labeltree_maxdepth_r(d+1) ) ; 
    }
    return mx ; 
}



/** 
sn::labeltree_subdepth  (based on NTreeBalance::subdepth_r)
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

inline void sn::labeltree_subdepth()
{
    labeltree_subdepth_r(0); 
}
inline void sn::labeltree_subdepth_r(int d)
{
    subdepth = maxdepth() ;
    for(int i=0 ; i < num_child() ; i++) 
    {
        sn* ch = get_child(i) ; 
        ch->labeltree_subdepth_r(d+1) ; 
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
        switch( typecode )
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

inline bool        sn::is_listnode() const { return CSG::IsList(typecode); }
inline std::string sn::tag() const {         return CSG::Tag(typecode) ;  }



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
       << " typecode " << std::setw(3) << typecode 
       << " num_node " << std::setw(3) << num_node() 
       << " num_leaf " << std::setw(3) << num_leaf() 
       << " maxdepth " << std::setw(2) << maxdepth() 
       << " is_positive_form " << ( is_positive_form() ? "Y" : "N" ) 
       << " lvid " << std::setw(3) << lvid 
       << " tag " << tag() 
       ; 
    std::string str = ss.str();
    return str ;
}


inline std::string sn::brief() const
{
    std::stringstream ss ;
    ss << "sn::brief"
       << " tc " << std::setw(4) << typecode
       << " cm " << std::setw(2) << complement
       << " lv " << std::setw(3) << lvid 
       << " xf " << std::setw(1) << ( xform ? "Y" : "N" )
       << " pa " << std::setw(1) << ( param ? "Y" : "N" )
       << " bb " << std::setw(1) << ( aabb  ? "Y" : "N" )
       << " pt " << std::setw(1) << ( parent ? "Y" : "N" )
#ifdef WITH_CHILD
       << " nc " << std::setw(2) << child.size() 
#else
       << " l  " << std::setw(1) << ( left  ? "Y" : "N" )   
       << " r  " << std::setw(1) << ( right  ? "Y" : "N" )   
#endif
       << " dp " << std::setw(2) << depth
       << " tg " << tag()
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
        case MINIMAL:  md = MODE_MINIMAL  ; break ; 
        case TYPECODE: md = MODE_TYPECODE ; break ; 
        case DEPTH:    md = MODE_DEPTH    ; break ; 
        case SUBDEPTH: md = MODE_SUBDEPTH ; break ; 
        case TYPETAG:  md = MODE_TYPETAG  ; break ; 
        case PID:      md = MODE_PID      ; break ; 
    }
    return md ; 
}

inline void sn::render_r(scanvas* canvas, std::vector<const sn*>& order, int mode, int d) const
{
    int ordinal = std::distance( order.begin(), std::find(order.begin(), order.end(), this )) ;
    assert( ordinal < int(order.size()) );

    int ix = ordinal ;
    int iy = d ;
    std::string tag = CSG::Tag(typecode, complement == 1); 

    switch(mode)
    {
        case 0: canvas->drawch( ix, iy, 0,0, 'o' )         ; break ; 
        case 1: canvas->draw(   ix, iy, 0,0,  typecode  )      ; break ; 
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
with all operator nodes with a common *op* typecode 
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
------------------------------------------------------------

This was implemented while sn was not fully featured.
It was used to provide a "template" tree with typecodes only, 
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

        if(typecode == CSG_INTERSECTION || typecode == CSG_UNION)
        {   
            if(negate)                             // !( A*B ) ->  !A + !B       !(A + B) ->     !A * !B
            {    
                typecode = CSG::DeMorganSwap(typecode) ;   // UNION->INTERSECTION, INTERSECTION->UNION
                left_negate = true ; 
                right_negate = true ; 
            }   
            else
            {                                      //  A * B ->  A * B         A + B ->  A + B
                left_negate = false ;
                right_negate = false ;
            }
        }
        else if(typecode == CSG_DIFFERENCE)
        {
            if(negate)                             //  !(A - B) -> !(A*!B) -> !A + B
            {
                typecode = CSG_UNION ;
                left_negate = true ;
                right_negate = false  ;
            }
            else
            {
                typecode = CSG_INTERSECTION ;    //    A - B ->  A * !B
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

inline void sn::zero_label()
{
    for(int i=0 ; i < int(sizeof(label)) ; i++) label[i] = '\0' ;    
}

inline void sn::set_label( const char* label_ )
{
    strncpy( &label[0], label_, sizeof(label) );
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
    depth = d ; 

    for(int i=0 ; i < num_child() ; i++)
    {
        sn* ch = get_child(i) ;       
        ch->set_lvid_r(lvid_, d+1 ); 
    }
}









inline void sn::setPA( double x0, double y0, double z0, double w0, double x1, double y1 )
{
    if( param == nullptr ) param = new s_pa ; 
    param->x0 = x0 ; 
    param->y0 = y0 ; 
    param->z0 = z0 ; 
    param->w0 = w0 ; 
    param->x1 = x1 ; 
    param->y1 = y1 ; 
}

inline void sn::setBB( double x0, double y0, double z0, double x1, double y1, double z1 )
{
    if( aabb == nullptr ) aabb = new s_bb ; 
    aabb->x0 = x0 ; 
    aabb->y0 = y0 ; 
    aabb->z0 = z0 ; 
    aabb->x1 = x1 ; 
    aabb->y1 = y1 ; 
    aabb->z1 = z1 ; 
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
    if( xform == nullptr ) xform = new s_tv ; 
    xform->t = t ; 
    xform->v = v ; 
}
inline void sn::combineXF( const glm::tmat4x4<double>& t, const glm::tmat4x4<double>& v )
{
    if( xform == nullptr )
    {
        xform = new s_tv ; 
        xform->t = t ; 
        xform->v = v ; 
    }
    else
    {
        glm::tmat4x4<double> tt = xform->t * t ;   
        glm::tmat4x4<double> vv = v * xform->v ;   
        xform->t = tt ; 
        xform->v = vv ; 
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
inline sn* sn::Prim(int typecode_)   // static
{
    return new sn(typecode_, nullptr, nullptr) ; 
}
inline sn* sn::Create(int typecode_, sn* left_, sn* right_)  // static
{
    sn* nd = new sn(typecode_, left_, right_) ;
    return nd ;
}
inline sn* sn::Boolean(int typecode_, sn* left_, sn* right_)  // static
{
    sn* nd = Create(typecode_, left_, right_); 
    return nd ; 
}



/**
sn::ZNudgeExpandEnds
---------------------

CAUTION: changes geometry, only appropriate 
for subtracted constituents eg inners 

This is used from U4Polycone::init_inner 
and is probably only applicable to the 
very controlled situation of the polycone
with a bunch of cylinders and cones. 

* cf X4Solid::Polycone_Inner_Nudge

**/

inline void sn::ZNudgeExpandEnds(std::vector<sn*>& prims, bool enable) // static
{
    int num_prim = prims.size() ; 

    sn* lower = prims[0] ; 
    sn* upper = prims[prims.size()-1] ; 
    bool can_znudge_ends = lower->can_znudge() && upper->can_znudge() ; 
    assert( can_znudge_ends ); 
  
    double lower_zmin = lower->zmin() ; 
    double upper_zmax = upper->zmax() ; 
    bool z_expect = upper_zmax > lower_zmin  ; 


    if(true || level() > 0) std::cout 
       << std::endl
       << "sn::ZNudgeExpandEnds "
       << " num_prim " << num_prim 
       << " enable " << ( enable ? "YES" : "NO " )
       << " can_znudge_ends " << ( can_znudge_ends ? "YES" : "NO " ) 
       << " lower_zmin " << lower_zmin
       << " upper_zmax " << upper_zmax
       << " z_expect " << ( z_expect ? "YES" : "NO " )
       << std::endl
       << ZDesc(prims)
       << std::endl
       ;

    if(!enable) return ; 
    assert( z_expect ); 

    double dz = 1. ; 
    lower->decrease_zmin(dz);   
    upper->increase_zmax(dz);   
}

/**
sn::ZNudgeOverlapJoints
-------------------------

lower_rperp_at_zmax > upper_rperp_at_zmin::
            
        +-----+
        |     |   
    +---+.....+---+  
    |   +~~~~~+   |       upper->decrease_zmin
    |             | 
    +-------------+

!(lower_rperp_at_zmax > upper_rperp_at_zmin)::
               
    +-------------+
    |             |
    |   +~~~~~+   |    lower->increase_zmax
    +---+-----+---+
        |     |   
        +-----+             


HMM a cone atop a cylinder where the cone at zmin 
starts at the same radius of the cylinder will mean that 
there will be a change to the shape for both 
uncoinciding by the cylinder expanding up and the cone 
expanding down. So there is no way to avoid concidence 
on that joint, without changing geometry::


        +----------------+
       /                  \
      /                    \
     /                      \
    ~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    |                        |
    |                        | 
    |                        | 
    |                        | 
    |                        | 
    +------------------------+

This happens with::

   NNVTMCPPMTsMask_virtual
   HamamatsuR12860sMask_virtual

As they are virtual it doesnt matter for physics in this 
case : but that doesnt stop it being an issue. 

**/

inline void sn::ZNudgeOverlapJoints(std::vector<sn*>& prims, bool enable ) // static
{
    int num_prim = prims.size() ; 
    assert( num_prim > 1 && "one prim has no joints" );  
    double dz = 1. ; 

    bool dump = true || level() > 0 ; 

    if(dump) std::cout
       << std::endl
       << "sn::ZNudgeOverlapJoints "
       << " num_prim " << num_prim 
       << " enable " << ( enable ? "YES" : "NO " )
       << std::endl
       << ZDesc(prims)
       << std::endl
       ;

    for(int i=1 ; i < num_prim ; i++)
    {
        sn* lower = prims[i-1] ; 
        sn* upper = prims[i] ; 
        bool can_znudge_ends = lower->can_znudge() && upper->can_znudge() ; 
        assert( can_znudge_ends ); 
        
        double lower_zmax = lower->zmax(); 
        double upper_zmin = upper->zmin() ; 
        bool z_coincident_joint = std::abs( lower_zmax - upper_zmin ) < Z_EPSILON  ; 

        double upper_rperp_at_zmin = upper->rperp_at_zmin() ; 
        double lower_rperp_at_zmax = lower->rperp_at_zmax() ; 

        if(dump) std::cerr
            << "sn::ZNudgeOverlapJoints"
            << " ("<< i-1 << "," << i << ") "
            << " lower_zmax " << lower_zmax  
            << " upper_zmin " << upper_zmin
            << " z_coincident_joint " << ( z_coincident_joint ? "YES" : "NO " )
            << " enable " << ( enable ? "YES" : "NO " )
            << " upper_rperp_at_zmin " << upper_rperp_at_zmin
            << " lower_rperp_at_zmax " << lower_rperp_at_zmax
            << std::endl 
            ; 

        if(!z_coincident_joint) continue ; 

        if( lower_rperp_at_zmax > upper_rperp_at_zmin )    
        {    
            upper->decrease_zmin( dz ); 
            std::cerr  
                << "sn::ZNudgeOverlapJoints"
                << " lower_rperp_at_zmax > upper_rperp_at_zmin : upper->decrease_zmin( dz ) "
                << "  : expand upper down into bigger lower "
                << std::endl 
                ;  
        }
        else
        {
            lower->increase_zmax( dz ); 
            std::cerr  
                << "sn::ZNudgeOverlapJoints"
                << " !(lower_rperp_at_zmax > upper_rperp_at_zmin) : lower->increase_zmax( dz ) "
                << "  : expand lower up into bigger upper "
                << std::endl 
                ;  
        }
    }
}


/**
sn::can_znudge
----------------

Typecode currently must be one of::

   CSG_CYLINDER 
   CSG_CONE 
   CSG_DISC
   CSG_ZSPHERE

**/

inline bool sn::can_znudge() const 
{
    return param && CSG::CanZNudge(typecode) ; 
}

/**
sn::CanZNudgeAll
-----------------

Returns true when all prim are ZNudge capable 

**/

inline bool sn::CanZNudgeAll(std::vector<sn*>& prims)  // static
{
    int num_prim = prims.size() ; 
    int count = 0 ; 
    for(int i=0 ; i < num_prim ; i++) if(prims[i]->can_znudge()) count += 1 ; 
    return count == num_prim ; 
}



/**
sn::increase_zmax
------------------

Expand upwards in +Z direction::

    +~~~~~~~~+  zmax + dz  (dz > 0.)
    +--------+  zmax
    |        |
    |        |
    +--------+  zmin

**/
inline void sn::increase_zmax( double dz )
{
    assert( dz > 0. ); 
    double _zmax = zmax(); 
    double new_zmax = _zmax + dz ; 

    std::cerr
        << "sn::increase_zmax"
        << " lvid " << lvid 
        << " _zmax " << _zmax 
        << " dz " << dz
        << " new_zmax " << new_zmax 
        << std::endl 
        ;   

    set_zmax(new_zmax); 
}
/**
sn::decrease_zmin
------------------

Expand downwards in -Z direction::

    +--------+  zmax
    |        |
    |        |
    +--------+  zmin
    +~~~~~~~~+  zmin - dz    (dz > 0.)

**/
inline void sn::decrease_zmin( double dz )
{
    assert( dz > 0. ); 
    double _zmin = zmin(); 
    double new_zmin = _zmin - dz ; 

    std::cerr
        << "sn::decrease_zmin"
        << " lvid " << lvid 
        << " _zmin " << _zmin 
        << " dz " << dz
        << " new_zmin " << new_zmin 
        << std::endl 
        ;   

    set_zmin(new_zmin); 
}
inline double sn::zmin() const
{
    assert( can_znudge() );
    double v = 0. ; 
    switch(typecode)
    {
        case CSG_CYLINDER: v = param->value(4) ; break ; 
        case CSG_CONE:     v = param->value(1) ; break ;  
    }
    return v ;
}

inline void sn::set_zmin(double zmin_)
{
    assert( can_znudge() );
    switch(typecode)
    {
        case CSG_CYLINDER: param->set_value(4, zmin_) ; break ; 
        case CSG_CONE:     param->set_value(1, zmin_) ; break ; 
    }
}

inline double sn::zmax() const
{
    assert( can_znudge() );
    double v = 0. ; 
    switch(typecode)
    {
        case CSG_CYLINDER: v = param->value(5) ; break ; 
        case CSG_CONE:     v = param->value(3) ; break ;  
    }
    return v ;
}
inline void sn::set_zmax(double zmax_)
{
    assert( can_znudge() );
    switch(typecode) 
    {
        case CSG_CYLINDER: param->set_value(5, zmax_) ; break ; 
        case CSG_CONE:     param->set_value(3, zmax_) ; break ; 
    } 
}

inline double sn::rperp_at_zmax() const 
{
    assert( can_znudge() );
    double v = 0. ; 
    switch(typecode)
    {
        case CSG_CYLINDER: v = param->value(3) ; break ; 
        case CSG_CONE:     v = param->value(2) ; break ; 
    }
    return v ; 
}

inline double sn::rperp_at_zmin() const 
{
    assert( can_znudge() );
    double v = 0. ; 
    switch(typecode)
    {
        case CSG_CYLINDER: v = param->value(3) ; break ; 
        case CSG_CONE:     v = param->value(0) ; break ; 
    }
    return v ; 
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
    int num_prim = prims.size() ; 
    std::stringstream ss ;
    ss << "sn::ZDesc" ;
    ss << " prims(" ;
    for(int i=0 ; i < num_prim ; i++) ss << prims[i]->index() << " " ;
    ss << ") " ;
    ss << std::endl ;

    for(int i=0 ; i < num_prim ; i++)
    {
        sn* a = prims[i];
        ss << " idx "  << std::setw(3) << a->index()
           << " tag "   << std::setw(3) << a->tag()
           << " zmin " << std::setw(10) << a->zmin()
           << " zmax " << std::setw(10) << a->zmax()
           << " rperp_at_zmin " << std::setw(10) << a->rperp_at_zmin()
           << " rperp_at_zmax " << std::setw(10) << a->rperp_at_zmax()
           << std::endl
           ;
    }
    std::string str = ss.str();
    return str ;
}

inline const double* sn::getParam() const { return param ? param->data() : nullptr ; }
inline const double* sn::getAABB()  const { return aabb ? aabb->data() : nullptr ; }

inline bool sn::hasAABB() const   // not-nullptr and not all zero 
{
    const double* aabb = getAABB();  
    return aabb != nullptr && !s_bb::IsZero(aabb) ; 
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
    //sn* n = Buggy_CommonOperatorTree( prims, CSG_UNION ); 
    sn* n = BuildCommonTypeTree_Unbalanced(prims, CSG_UNION ); 
    return n ; 
}
inline sn* sn::Contiguous(std::vector<sn*>& prims )
{
    sn* n = Compound( prims, CSG_CONTIGUOUS ); 
    return n ; 
}



inline sn* sn::Compound(std::vector<sn*>& prims, int typecode_ )
{
    assert( typecode_ == CSG_CONTIGUOUS || typecode_ == CSG_DISCONTIGUOUS ); 

    int num_prim = prims.size(); 
    assert( num_prim > 0 ); 

    sn* nd = Create( typecode_ ); 

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






/**
sn::Buggy_CommonOperatorTree
-----------------------------

This has issues of inadvertent node deletion when 
there are for example 3 leaves::


        U
      U   2
    0  1

The populate_leaves and/or prune needs to be cleverer 
to make this approach work. 

See sn::BuildCommonTypeTree_Unbalanced below for 
alternative without this bug. 

**/


inline sn* sn::Buggy_CommonOperatorTree( std::vector<sn*>& leaves, int op ) // static
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

        if(level() > 0) std::cerr 
            << "sn::CommonOperatorTree after ZeroTree"
            << " num_leaves " << num_leaves 
            << " level " << level()
            << std::endl
            ; 
        if(level() > 1) std::cerr << root->render(5) ; 

        root->populate_leaves(leaves); 

        if(level() > 0) std::cerr 
            << "sn::CommonOperatorTree after populate_leaves" 
            << " num_leaves " << num_leaves 
            << " level " << level()
            << std::endl 
            ; 
        if(level() > 1) std::cerr << root->render(5) ; 

        root->prune();
 
        if(level() > 0) std::cerr 
            << "sn::CommonOperatorTree after prun"
            << " num_leaves " << num_leaves 
            << " level " << level()
            << std::endl 
            ; 
        if(level() > 1) std::cerr << root->render(5) ; 
    }
    return root ; 
} 







/**
sn::BuildCommonTypeTree_Unbalanced
------------------------------------

Simple unbalanced tree building from leaves that is now used from sn::UnionTree.
Previously used a more complicated approach sn::Buggy_CommonOperatorTree

For development of tree manipulations see::

     sysrap/tests/tree_test.cc 
     sysrap/tests/tree.h

To build unbalanced, after the first single leaf root, 
each additional leaf is accompanied by an operator node
that becomes the new root::

    0


      U
    0   1

          U 
      U     2
    0   1

             U
          U     3
      U     2
    0   1



**/


inline sn* sn::BuildCommonTypeTree_Unbalanced( const std::vector<sn*>& leaves, int typecode )  // static
{
    int num_leaves = leaves.size() ;
    int num_leaves_placed = 0 ;  
    if(num_leaves == 0) return nullptr ; 

    sn* root = leaves[num_leaves_placed] ; 
    num_leaves_placed += 1 ; 

    while( num_leaves_placed < num_leaves )
    {
        root = Create(typecode, root, leaves[num_leaves_placed]); 
        num_leaves_placed += 1 ; 
    } 
    return root ; 
}


/**
sn::GetLVNodes
---------------

Collect all sn with the provided lvid

**/

struct sn_find_lvid
{
    int lvid ; 
    sn_find_lvid(int q_lvid) : lvid(q_lvid) {}   
    bool operator()(const sn* n){ return lvid == n->lvid ; }  
};


inline void sn::GetLVNodes( std::vector<sn*>& nds, int lvid ) // static
{
    sn_find_lvid flv(lvid); 
    pool->find(nds, flv );   
}


/**
sn::getLVNodes
---------------

Collect all sn with the lvid of this node. 
The vector is expected to include this node. 

**/

inline void sn::getLVNodes( std::vector<sn*>& nds ) const 
{
    GetLVNodes(nds, lvid ); 
    assert( Includes(nds, const_cast<sn*>(this) ) ); 
}

inline bool sn::Includes( const std::vector<sn*>& nds, sn* nd ) // static
{
    return std::find(nds.begin(), nds.end(), nd ) != nds.end() ; 
}

inline sn* sn::Get(int idx) // static 
{
    return pool->get(idx) ; 
}


/**
sn::GetLVRoot
---------------

First sn with the lvid and sn::is_root():true in (s_csg)pool 

**/

struct sn_find_lvid_root
{
    int lvid ; 
    sn_find_lvid_root(int q_lvid) : lvid(q_lvid) {}   
    bool operator()(const sn* n){ return lvid == n->lvid && n->is_root() ; }  
};

inline sn* sn::GetLVRoot( int lvid ) // static
{
    std::vector<sn*> nds ; 
    sn_find_lvid_root flvr(lvid); 
    pool->find(nds, flvr );   
    int count = nds.size() ; 
    assert( count == 0 || count == 1 ); 
    return count == 1 ? nds[0] : nullptr ; 
}



inline std::string sn::rbrief() const 
{
    std::stringstream ss ; 
    ss << "sn::rbrief" << std::endl ; 

    rbrief_r(ss, 0) ; 
    std::string str = ss.str(); 
    return str ; 
}

inline void sn::rbrief_r(std::ostream& os, int d) const 
{
    os << std::setw(3) << d << " : " << brief() << std::endl ; 
    for(int i=0 ; i < num_child() ; i++) get_child(i)->rbrief_r(os, d+1) ;
}


/**
sn::has_type
------------

Returns true when this node has typecode present in the types vector.  

**/


inline bool sn::has_type(const std::vector<OpticksCSG_t>& types) const 
{
    return std::find( types.begin(), types.end(), typecode ) != types.end() ; 
}

/**
sn::typenodes_
-----------------

Collect sn with typecode provided in the args. 

**/

template<typename ... Args> 
inline void sn::typenodes_(std::vector<const sn*>& nds, Args ... tcs ) const  
{
    std::vector<OpticksCSG_t> types = {tcs ...};
    typenodes_r_(nds, types, 0 ); 
}

// NB MUST USE SYSRAP_API TO PLANT THE SYMBOLS IN THE LIB (OR MAKE THEM VISIBLE FROM ELSEWHERE) 
template SYSRAP_API void sn::typenodes_(std::vector<const sn*>& nds, OpticksCSG_t ) const  ; 
template SYSRAP_API void sn::typenodes_(std::vector<const sn*>& nds, OpticksCSG_t, OpticksCSG_t ) const ; 
template SYSRAP_API void sn::typenodes_(std::vector<const sn*>& nds, OpticksCSG_t, OpticksCSG_t, OpticksCSG_t ) const  ; 

/**
sn::typenodes_r_
-------------------

Recursive traverse CSG tree collecting snd::index when the snd::typecode is in the types vector. 

**/

inline void sn::typenodes_r_(std::vector<const sn*>& nds, const std::vector<OpticksCSG_t>& types, int d) const 
{
    if(has_type(types)) nds.push_back(this); 
    for(int i=0 ; i < num_child() ; i++) get_child(i)->typenodes_r_(nds, types, d+1 ) ;
}





/**
sn::max_binary_depth
-----------------------

Maximum depth of the binary compliant portion of the n-ary tree, 
ie with listnodes not recursed and where nodes have either 0 or 2 children.  
The listnodes are regarded as leaf node primitives.  

* Despite the *sn* tree being an n-ary tree (able to hold polycone and multiunion compounds)
  it must be traversed as a binary tree by regarding the compound nodes as effectively 
  leaf node "primitives" in order to generate the indices into the complete binary 
  tree serialization in level order 

* hence the recursion is halted at list nodes

**/

inline int sn::max_binary_depth() const 
{
    return max_binary_depth_r(0) ; 
}
inline int sn::max_binary_depth_r(int d) const   
{
    int mx = d ; 
    if( is_listnode() == false )
    {
        int nc = num_child() ; 
        if( nc > 0 ) assert( nc == 2 ) ; 
        for(int i=0 ; i < nc ; i++)  
        {
            sn* ch = get_child(i) ; 
            mx = std::max( mx,  ch->max_binary_depth_r(d + 1) ) ; 
        }
    }
    return mx ; 
}





/**
sn::getLVBinNode
------------------

Returns the number of nodes in a complete binary tree
of height corresponding to the max_binary_depth 
of this node. 

**/

inline int sn::getLVBinNode() const 
{
    int h = max_binary_depth(); 
    return st::complete_binary_tree_nodes( h );  
}

/**
sn::getLVSubNode
-------------------

Sum of children of compound nodes found beneath this node. 
HMM: this assumes compound nodes only contain leaf nodes 

Notice that the compound nodes themselves are regarded as part of
the binary tree. 

**/

inline int sn::getLVSubNode() const 
{
    int constituents = 0 ; 
    std::vector<const sn*> subs ; 
    typenodes_(subs, CSG_CONTIGUOUS, CSG_DISCONTIGUOUS, CSG_OVERLAP );  
    int nsub = subs.size(); 
    for(int i=0 ; i < nsub ; i++)
    {
        const sn* nd = subs[i] ; 
        assert( nd->typecode == CSG_CONTIGUOUS || nd->typecode == CSG_DISCONTIGUOUS ); 
        constituents += nd->num_child() ; 
    } 
    return constituents ; 
}


/**
sn::getLVNumNode
-------------------

Returns total number of nodes that can contain 
a complete binary tree + listnode constituents
serialization of this node.  

**/

inline int sn::getLVNumNode() const 
{
    int bn = getLVBinNode() ; 
    int sn = getLVSubNode() ; 
    return bn + sn ; 
}






/**
sn::GetLVNodesComplete
-------------------------

As the traversal is constrained to the binary tree portion of the n-ary snd tree 
can populate a vector of *snd* pointers in complete binary tree level order indexing
with nullptr left for the zeros.  This is similar to the old NCSG::export_tree_r.

**/

inline void sn::GetLVNodesComplete(std::vector<const sn*>& nds, int lvid) // static 
{
    const sn* root = GetLVRoot(lvid);  // first sn from pool with lvid that is_root
    assert(root); 
    root->getLVNodesComplete(nds);    

    if(level() > 0 && nds.size() > 8 )
    {
        std::cout
            << "sn::GetLVNodesComplete"
            << " lvid " << lvid
            << " level " << level()
            << std::endl
            << root->rbrief()
            << std::endl
            << root->render(SUBDEPTH)
            ;
    }
}

/**
sn::getLVNodesComplete
-------------------------

**/

inline void sn::getLVNodesComplete(std::vector<const sn*>& nds) const 
{
    int bn = getLVBinNode();  
    int sn = getLVSubNode();  
    int numParts = bn + sn ; 
    nds.resize(numParts); 

    assert( sn == 0 ); // CHECKING : AS IMPL LOOKS LIKE ONLY HANDLES BINARY NODES

    GetLVNodesComplete_r( nds, this, 0 ); 
}

/**
sn::GetLVNodesComplete_r
-------------------------
**/

inline void sn::GetLVNodesComplete_r(std::vector<const sn*>& nds, const sn* nd, int idx)  // static
{
    assert( idx < int(nds.size()) ); 
    nds[idx] = nd ; 

    int nc = nd->num_child() ; 

    if( nc > 0 && nd->is_listnode() == false ) // non-list operator node
    {
        assert( nc == 2 ) ;
        for(int i=0 ; i < nc ; i++)
        {
            const sn* child = nd->get_child(i) ;

            int cidx = 2*idx + 1 + i ; // 0-based complete binary tree level order indexing 

            GetLVNodesComplete_r(nds, child, cidx );
        }
    }
}




/**
sn::ancestors (not including this node)
-----------------------------------------

Collect by following parent links then reverse 
the vector to put into root first order. 

**/

inline void sn::ancestors(std::vector<const sn*>& nds) const
{
    const sn* nd = this ; 
    while( nd && nd->parent ) 
    {    
        nds.push_back(nd->parent);
        nd = nd->parent ; 
    }    
    std::reverse( nds.begin(), nds.end() );
}

/**
sn::connectedtype_ancestors
-----------------------------

Follow impl from nnode::collect_connectedtype_ancestors

Notice this is different from selecting all ancestors and then requiring 
a type, because the traversal up the parent links is stopped 
once reaching an node of type different to the parent type.  

**/

inline void sn::connectedtype_ancestors(std::vector<const sn*>& nds ) const 
{
    if(!parent) return ;   // start from parent to avoid collecting self
    ConnectedTypeAncestors( parent, nds, parent->typecode ); 
}
inline void sn::ConnectedTypeAncestors(const sn* n, std::vector<const sn*>& nds, int q_typecode) // static
{
    while(n && n->typecode == q_typecode)
    {    
        nds.push_back(n);
        n = n->parent ; 
    }    
}



/**
sn::collect_progeny
---------------------

Follow impl from nnode::collect_progeny

Progeny excludes self, so start from child

**/

inline void sn::collect_progeny( std::vector<const sn*>& progeny, int exclude_typecode ) const
{   
    for(int i=0 ; i < num_child() ; i++)
    {   
        const sn* ch = get_child(i); 
        CollectProgeny_r(ch, progeny, exclude_typecode );  
    }
}
inline void sn::CollectProgeny_r( const sn* n, std::vector<const sn*>& progeny, int exclude_typecode ) // static
{   
    if(n->typecode != exclude_typecode || exclude_typecode == CSG_ZERO)  
    {   
        if(std::find(progeny.begin(), progeny.end(), n) == progeny.end()) progeny.push_back(n);
    }
    
    for(int i=0 ; i < n->num_child() ; i++)
    {   
        const sn* ch = n->get_child(i); 
        CollectProgeny_r(ch, progeny, exclude_typecode );  
    }
}


/**
sn::collect_monogroup
-----------------------

Follow impl from nnode::collect_monogroup


1. follow parent links collecting ancestors until reach ancestor of another CSG type
   eg on starting with a primitive of CSG_UNION parent finds 
   direct lineage ancestors that are also CSG_UNION

2. for each of those same type ancestors collect
   all progeny but exclude the operator nodes to 
   give just the prims within the same type monogroup 

**/

inline void sn::collect_monogroup( std::vector<const sn*>& monogroup ) const
{
   if(!parent) return ;

   std::vector<const sn*> connectedtype ;
   connectedtype_ancestors(connectedtype);
   int num_connectedtype = connectedtype.size() ; 

   int exclude_typecode = parent->typecode ;  

   for(int i=0 ; i < num_connectedtype ; i++)
   {
       const sn* ca = connectedtype[i];
       ca->collect_progeny( monogroup, exclude_typecode );
   }
}

/**
sn::AreFromSameMonogroup
--------------------------

After nnode::is_same_monogroup

1. if a or b have no parent or either of their parent type is not *op* returns false

2. collect monogroup of a 

3. return true if b is found within the monogroup of a 

**/



inline bool sn::AreFromSameMonogroup(const sn* a, const sn* b, int op)  // static
{
   if(!a->parent || !b->parent || a->parent->typecode != op || b->parent->typecode != op) return false ;

   std::vector<const sn*> monogroup ;
   a->collect_monogroup(monogroup);

   return std::find(monogroup.begin(), monogroup.end(), b ) != monogroup.end() ;
}


inline bool sn::AreFromSameUnion(const sn* a, const sn* b) // static
{
   return AreFromSameMonogroup(a,b, CSG_UNION );
}



/**
sn::NodeTransformProduct
---------------------------

cf nmat4triple::product

1. finds CSG node ancestors of snd idx 


**/



inline void sn::NodeTransformProduct(
    int idx, 
    glm::tmat4x4<double>& t, 
    glm::tmat4x4<double>& v, 
    bool reverse, 
    std::ostream* out)  // static
{
    sn* nd = Get(idx); 
    assert(nd); 
    nd->getNodeTransformProduct(t,v,reverse,out) ; 
}

inline std::string sn::DescNodeTransformProduct(
    int idx, 
    glm::tmat4x4<double>& t, 
    glm::tmat4x4<double>& v, 
    bool reverse ) // static 
{
    std::stringstream ss ; 
    ss << "sn::DescNodeTransformProduct" << std::endl ;
    NodeTransformProduct( idx, t, v, reverse, &ss );     
    std::string str = ss.str(); 
    return str ; 
}

inline void sn::getNodeTransformProduct(
    glm::tmat4x4<double>& t, 
    glm::tmat4x4<double>& v, 
    bool reverse, std::ostream* out) const
{
    std::vector<const sn*> nds ; 
    ancestors(nds);
    nds.push_back(this); 

    int num_nds = nds.size();

    if(out)
    {
        *out 
             << std::endl 
             << "sn::getNodeTransformProduct" 
             << " idx " << idx() 
             << " reverse " << reverse
             << " num_nds " << num_nds 
             << std::endl 
             ;
    }

    glm::tmat4x4<double> tp(1.); 
    glm::tmat4x4<double> vp(1.); 

    for(int i=0 ; i < num_nds ; i++ ) 
    {
        int j  = num_nds - 1 - i ;  
        const sn* ii = nds[reverse ? j : i] ; 
        const sn* jj = nds[reverse ? i : j] ; 

        const s_tv* ixf = ii->xform ; 
        const s_tv* jxf = jj->xform ; 

        if(out)
        {
            *out
                << " i " << i 
                << " j " << j 
                << " ii.idx " << ii->idx() 
                << " jj.idx " << jj->idx()
                << " ixf " << ( ixf ? "Y" : "N" ) 
                << " jxf " << ( jxf ? "Y" : "N" ) 
                << std::endl 
                ; 

           if(ixf) *out << stra<double>::Desc( ixf->t, ixf->v, "(ixf.t)", "(ixf.v)" ) << std::endl ;   
           if(jxf) *out << stra<double>::Desc( jxf->t, jxf->v, "(jxf.t)", "(jxf.v)" ) << std::endl ;   
        }


        if(ixf) tp *= ixf->t ; 
        if(jxf) vp *= jxf->v ;  // // inverse-transform product in opposite order
    }
    memcpy( glm::value_ptr(t), glm::value_ptr(tp), sizeof(glm::tmat4x4<double>) );
    memcpy( glm::value_ptr(v), glm::value_ptr(vp), sizeof(glm::tmat4x4<double>) );

    if(out) *out << stra<double>::Desc( tp, vp , "tp", "vp" ) << std::endl ;
}

inline std::string sn::desc_getNodeTransformProduct(
    glm::tmat4x4<double>& t, 
    glm::tmat4x4<double>& v,  
    bool reverse) const
{
    std::stringstream ss ; 
    ss << "sn::desc_getNodeTransformProduct" << std::endl ;
    getNodeTransformProduct( t, v, reverse, &ss );     
    std::string str = ss.str(); 
    return str ; 
}





