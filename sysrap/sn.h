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
But have now moved all snd functionality over to sn. So can directly use
only sn and the old "WITH-SND" is removed.

sn ctor/dtor register/de-register from s_pool<sn,_sn>
-------------------------------------------------------

In order to convert active *sn* pointers into indices
on persisting have explictly avoided leaking ANY *sn* by
taking care to ALWAYS delete appropriately.
This means that can use the *sn* ctor/dtor to add/erase update
an std::map of active *sn* pointers keyed on a creation index.
This map allows the active *sn* pointers to be converted into
a contiguous set of indices to facilitate serialization.

Possible Future
-----------------

CSG_CONTIGUOUS could keep n-ary CSG trees all the way to the GPU

**/

#include <map>
#include <set>
#include <vector>
#include <sstream>
#include <iomanip>
#include <cassert>
#include <csignal>
#include <cstdint>
#include <functional>
#include <algorithm>

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
#include "sgeomtools.h"

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
    int  note  ;        // 13
    int  coincide ;     // 14
    char label[16] ;    // 15,16,17,18
    static constexpr const char* ITEM = "19" ;
#else
    int  left ;         // 7
    int  right ;        // 8
    int  index ;        // 9
    int  depth ;        // 10
    int  note  ;        // 11
    int  coincide ;     // 12
    char label[16] ;    // 13,14,15,16
    static constexpr const char* ITEM = "17" ;
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
    static int Check_LEAK(const char* msg, int i=-1);

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
    int note ;
    int coincide ;
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
    static constexpr const double UNBOUNDED_DEFAULT_EXTENT = 0. ;

    static void SetPOOL( POOL* pool_ );
    static int level();
    static std::string Desc();

    // templating allows to work with both "sn*" and "const sn*"
    template<typename N> static std::string Desc(N* n);
    template<typename N> static std::string DescPrim(N* n);
    template<typename N> static std::string Desc(const std::vector<N*>& nds, bool reverse=false);
    template<typename N> static std::string DescPrim(const std::vector<N*>& nds, bool reverse=false);

    template<typename N> static std::string Brief(const std::vector<N*>& nds, bool reverse=false);


    static int Index(const sn* n);

    int  idx() const ;  // to match snd.hh
    int  index() const ;
    int  parent_index() const ;

    bool is_root() const ;

    int  num_child() const ;
    sn*  first_child() const ;
    sn*  last_child() const ;
    sn*  get_child(int ch) const ;
    sn*  get_left() const ;
    sn*  get_right() const ;

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

    sn* deepcopy_excluding_leaf(const sn* l) const ;
    sn* deepcopy_excluding_leaf_r(int d, const sn* l) const ;


    sn* copy() const ;

    static void DeepCopy(std::vector<sn*>& p1, const std::vector<sn*>& p0) ;

    void set_child( int ix, sn* ch, bool copy );
    void set_child_leaking_prior( int ix, sn* ch, bool copy );


    void set_left( sn* left, bool copy );
    void set_right( sn* right, bool copy  );

    bool is_primitive() const ;
    bool is_complement() const ;
    bool is_complement_primitive() const ;
    bool is_bileaf() const ;
    bool is_operator() const ;
    bool is_zero() const ;

    bool is_lrzero() const ;  //  l-zero AND  r-zero
    bool is_rzero() const ;   // !l-zero AND  r-zero
    bool is_lzero() const ;   //  l-zero AND !r-zero

    int num_node() const ;
    int num_node_r(int d) const ;

    int num_notsupported_node() const ;
    int num_notsupported_node_r(int d) const ;


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

    void typecodes(std::set<int>& tcs, int minsubdepth=0 ) const ;
    void typecodes_r(std::set<int>& tcs, int minsubdepth ) const ;
    std::string desc_typecodes() const ;
    int  typecodes_count(const std::vector<int>& tcq, int minsubdepth=0 ) const ;
    std::string desc_typecodes_count() const ;


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
    std::string desc_prim() const ;
    std::string desc_prim_all(bool reverse) const ;

    std::string id() const ;
    std::string brief() const ;
    std::string desc_child() const ;
    std::string desc_this() const ;

    std::string desc_r() const ;
    void desc_r(int d, std::stringstream& ss) const ;

    std::string detail_r() const ;
    void detail_r(int d, std::stringstream& ss) const ;
    std::string detail() const ;


    std::string render() const ;
    std::string render_typetag() const ;
    std::string render_parent() const ;
    std::string rdr() const ;
    std::string render(int mode) const ;

    enum { MINIMAL, TYPECODE, DEPTH, SUBDEPTH, TYPETAG, PID, NOTE, PARENT, IDX,  NUM_MODE=9 } ;

    static constexpr const char* MODE_MINIMAL = "MINIMAL" ;
    static constexpr const char* MODE_TYPECODE = "TYPECODE" ;
    static constexpr const char* MODE_DEPTH = "DEPTH" ;
    static constexpr const char* MODE_SUBDEPTH = "SUBDEPTH" ;
    static constexpr const char* MODE_TYPETAG = "TYPETAG" ;
    static constexpr const char* MODE_PID = "PID" ;
    static constexpr const char* MODE_NOTE = "NOTE" ;
    static constexpr const char* MODE_PARENT = "PARENT" ;
    static constexpr const char* MODE_IDX = "IDX" ;

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
    void flip_complement();
    void flip_complement_child();



    void zero_label();
    void set_label( const char* label_ );
    void set_lvid(int lvid_);
    void set_lvid_r(int lvid_, int d);
    int  check_idx(const char* msg) const ;
    int  check_idx_r(int d, const char* msg) const ;


    void setPA( double x, double y, double z, double w, double z1, double z2 );
    const double* getPA_data() const  ;
    void    copyPA_data(double* dst) const ;

    void setBB(  double x0, double y0, double z0, double x1, double y1, double z1 );
    void setBB(  double x0 );

    const double* getBB_data() const ;
    void    copyBB_data(double* dst) const ;

    double  getBB_xmin() const ;
    double  getBB_ymin() const ;
    double  getBB_zmin() const ;

    double  getBB_xmax() const ;
    double  getBB_ymax() const ;
    double  getBB_zmax() const ;

    double  getBB_xavg() const ;
    double  getBB_yavg() const ;
    double  getBB_zavg() const ;

    static double AABB_XMin( const sn* n );
    static double AABB_YMin( const sn* n );
    static double AABB_ZMin( const sn* n );

    static double AABB_XMax( const sn* n );
    static double AABB_YMax( const sn* n );
    static double AABB_ZMax( const sn* n );

    static double AABB_XAvg( const sn* n );
    static double AABB_YAvg( const sn* n );
    static double AABB_ZAvg( const sn* n );





    void setXF(     const glm::tmat4x4<double>& t );
    void setXF(     const glm::tmat4x4<double>& t, const glm::tmat4x4<double>& v ) ;
    void combineXF( const glm::tmat4x4<double>& t );
    void combineXF( const glm::tmat4x4<double>& t, const glm::tmat4x4<double>& v ) ;
    std::string descXF() const ;

    static sn* Cylinder(double radius, double z1, double z2) ;
    static sn* CutCylinder(
        double R,
        double dz,
        double _pz_nrm_x,
        double _pz_nrm_y,
        double _pz_nrm_z,
        double _nz_nrm_x,
        double _nz_nrm_y,
        double _nz_nrm_z
    );

    static void CutCylinderZRange(
        double& zmin,
        double& zmax,
        double R,
        double dz,
        double pz_nrm_x,
        double pz_nrm_y,
        double pz_nrm_z,
        double nz_nrm_x,
        double nz_nrm_y,
        double nz_nrm_z
    );

    static sn* Cone(double r1, double z1, double r2, double z2);
    static sn* Sphere(double radius);
    static sn* ZSphere(double radius, double z1, double z2);
    static sn* Box3(double fullside);
    static sn* Box3(double fx, double fy, double fz );
    static sn* Torus(double rmin, double rmax, double rtor, double startPhi_deg, double deltaPhi_deg );
    static sn* Notsupported();

    static sn* Zero(double  x,  double y,  double z,  double w,  double z1, double z2);
    static sn* Zero();
    static sn* Prim(int typecode) ;
    static sn* Create(int typecode, sn* left=nullptr, sn* right=nullptr );
    static sn* Boolean( int op, sn* l, sn* r );

    static void ZNudgeExpandEnds(   int lvid, std::vector<sn*>& prims, bool enable);
    static void ZNudgeOverlapJoints(int lvid, std::vector<sn*>& prims, bool enable);
    static void ZNudgeOverlapJoint(int lvid, int i, sn* lower, sn* upper, bool enable, std::ostream* out  );

    bool can_znudge() const ;
    static bool CanZNudgeAll(std::vector<sn*>& prims);


    enum {
            NOTE_INCREASE_ZMAX = 0x1 << 0,
            NOTE_DECREASE_ZMIN = 0x1 << 1,
            HINT_LISTNODE_PRIM_DISCONTIGUOUS = 0x1 << 2,
            HINT_LISTNODE_PRIM_CONTIGUOUS    = 0x1 << 3
         };

    static unsigned NameHint(const char* name);
    static constexpr const char* _HINT_LISTNODE_PRIM_DISCONTIGUOUS = "HINT_LISTNODE_PRIM_DISCONTIGUOUS" ;
    static constexpr const char* _HINT_LISTNODE_PRIM_CONTIGUOUS    = "HINT_LISTNODE_PRIM_CONTIGUOUS" ;
    void set_hint_note(unsigned hint);
    void set_hint_listnode_prim_discontiguous();
    void set_hint_listnode_prim_contiguous();
    bool  is_hint_listnode_prim_discontiguous() const ;
    bool  is_hint_listnode_prim_contiguous() const ;

    void increase_zmax( double dz ); // expand upwards in +Z direction
    void decrease_zmin( double dz ); // expand downwards in -Z direction

    void increase_zmax_( double dz );
    void decrease_zmin_( double dz );

    void increase_zmax_cone( double dz );
    void decrease_zmin_cone( double dz );




    double zmin() const ;
    double zmax() const ;
    void set_zmin(double zmin_) ;
    void set_zmax(double zmax_) ;

    double    rperp_at_zmax() const ;
    void set_rperp_at_zmax(double rperp_) const ;

    double   rperp_at_zmin() const ;
    void set_rperp_at_zmin(double rperp_) const ;


    static double Sphere_RPerp_At_Z(double r, double z);
    double rperp_at_zmin_zsphere() const ;
    double rperp_at_zmax_zsphere() const ;




    static std::string ZDesc(const std::vector<sn*>& prims);

    void getParam_(double& p0, double& p1, double& p2, double& p3, double& p4, double& p5 ) const ;
    const double* getParam() const ;
    const double* getAABB() const ;
    bool hasAABB() const ;  // not-nullptr and not all zero


    static sn* Collection( std::vector<sn*>& prims );
    static sn* UnionTree(  std::vector<sn*>& prims );
    static sn* Contiguous( std::vector<sn*>& prims );
    static sn* Discontiguous( std::vector<sn*>& prims );
    static sn* Compound(   std::vector<sn*>& prims, int typecode_ );

    static sn* Buggy_CommonOperatorTree( std::vector<sn*>& leaves    , int op );
    static sn* BuildCommonTypeTree_Unbalanced( const std::vector<sn*>& leaves, int typecode );

    static void GetLVListnodes( std::vector<const sn*>& lns, int lvid );
    static int  GetChildTotal(  const std::vector<const sn*>& nds );

    static void GetLVNodes(  std::vector<sn*>& nds, int lvid );
    static void GetLVNodes_( std::vector<const sn*>& nds, int lvid );
    void getLVNodes( std::vector<sn*>& nds ) const ;
    static bool Includes(const std::vector<sn*>& nds, sn* nd );

    static sn* Get(int idx);
    static void FindLVRootNodes( std::vector<sn*>& nds, int lvid );
    static sn* GetLVRoot( int lvid );

    std::string rbrief() const ;
    void rbrief_r(std::ostream& os, int d) const  ;

    bool has_type(const std::vector<OpticksCSG_t>& types) const ;
    template<typename ... Args>
    void typenodes_(std::vector<const sn*>& nds, Args ... tcs ) const ;
    void typenodes_r_(std::vector<const sn*>& nds, const std::vector<OpticksCSG_t>& types, int d) const ;

    int max_binary_depth() const ;
    int max_binary_depth_r(int d) const ;

    uint64_t getLVBinNode() const ;
    uint64_t getLVSubNode() const ;
    uint64_t getLVNumNode() const ;

    static void GetLVNodesComplete(std::vector<const sn*>& nds, int lvid);
    void        getLVNodesComplete(std::vector<const sn*>& nds) const ;
    static void GetLVNodesComplete_r(std::vector<const sn*>& nds, const sn* nd, int idx);

    static void SelectListNode(std::vector<const sn*>& lns, const std::vector<const sn*>& nds);

    void ancestors(std::vector<const sn*>& nds) const ;

    void connectedtype_ancestors(std::vector<const sn*>& nds ) const ;
    static void ConnectedTypeAncestors(const sn* n, std::vector<const sn*>& nds, int q_typecode);

    void collect_progeny( std::vector<const sn*>& progeny, int exclude_typecode ) const ;
    static void CollectProgeny_r( const sn* n, std::vector<const sn*>& progeny, int exclude_typecode );

    void collect_prim( std::vector<const sn*>& prim ) const ;
    void collect_prim_r( std::vector<const sn*>& prim, int d ) const ;

    bool has_note(int q_note) const ;
    void collect_prim_note(  std::vector<sn*>& prim, int q_note );
    void collect_prim_note_r( std::vector<sn*>& prim, int q_note, int d );

    sn* find_joint_to_candidate_listnode( std::vector<sn*>& prim, int q_note ) ;
    sn* find_joint_to_candidate_listnode_r( int d,  std::vector<sn*>& prim, int q_note );
    bool has_candidate_listnode(int q_note);
    bool has_candidate_listnode_discontiguous() ;
    bool has_candidate_listnode_contiguous() ;

    static sn* CreateSmallerTreeWithListNode(sn* root0, int q_note);
    static sn* CreateSmallerTreeWithListNode_discontiguous(sn* root0);
    static sn* CreateSmallerTreeWithListNode_contiguous(   sn* root0);
    static int TypeFromNote(int q_note);

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

    double radius_sphere() const ;

    void setAABB_LeafFrame() ;
    void setAABB_LeafFrame_All() ;

    void setAABB_TreeFrame_All() ;
    //void setAABB(); // CONSIDER CAREFULLY between setAABB_(Tree/Leaf)Frame_All


    void postconvert(int lvid);

    template<typename N>
    static void OrderPrim( std::vector<N*>& prim, std::function<double(N* p)> fn, bool ascending  );




    static void Transform_Leaf2Tree( glm::tvec3<double>& xyz,  const sn* leaf, std::ostream* out );

    void uncoincide() ;
    void uncoincide_( bool enable, std::ostream* out);
    void uncoincide_zminmax( int i, sn* lower, sn* upper, bool enable, std::ostream* out  ) ;


};  // END



inline int sn::Check_LEAK(const char* msg, int i)  // static
{
    //const char* spacer = "\n\n" ;
    const char* spacer = "" ;

    std::stringstream ss ;
    ss << "sn::Check_LEAK[" << std::setw(3) << i << "] " << std::setw(30) << ( msg ? msg : "-" ) << "  " << sn::pool->brief() << spacer << std::endl ;
    std::string str = ss.str();
    std::cout << str ;

    if(!sn::LEAK)
    {
        assert( sn::pool->size() == 0 );
    }

    if(!s_tv::LEAK)
    {
        assert( s_tv::pool->size() == 0 );
    }

    if(!s_pa::LEAK)
    {
        assert( s_pa::pool->size() == 0 );
    }

    if(!s_bb::LEAK)
    {
        assert( s_bb::pool->size() == 0 );
    }

    return 0 ;
}








inline void        sn::SetPOOL( POOL* pool_ ){ pool = pool_ ; }
inline int         sn::level() {  return ssys::getenvint("sn__level",-1) ; } // static
inline std::string sn::Desc(){    return pool ? pool->desc() : "-" ; } // static

template<typename N>
inline std::string sn::Desc(N* n) // static
{
    return n ? n->desc() : "(null)" ;
}

template<typename N>
inline std::string sn::DescPrim(N* n) // static
{
    return n ? n->desc_prim() : "(null)" ;
}


template<typename N>
inline std::string sn::Desc(const std::vector<N*>& nds, bool reverse) // static
{
    int num_nd = nds.size() ;
    std::stringstream ss ;
    ss << "sn::Desc num_nd " << num_nd << ( reverse ? " DESC ORDER REVERSED " : "-" ) << std::endl ;
    for(int i=0 ; i < num_nd ; i++) ss << Desc(nds[reverse ? num_nd - 1 - i : i]) << std::endl ;
    std::string str = ss.str();
    return str ;
}

template<typename N>
inline std::string sn::DescPrim(const std::vector<N*>& nds, bool reverse) // static
{
    int num_nd = nds.size() ;
    std::stringstream ss ;
    ss << "sn::DescPrim num_nd " << num_nd << ( reverse ? " DESC ORDER REVERSED " : "-" ) << std::endl ;
    for(int i=0 ; i < num_nd ; i++) ss << DescPrim(nds[reverse ? num_nd - 1 - i : i]) << std::endl ;
    std::string str = ss.str();
    return str ;
}


template<typename N>
inline std::string sn::Brief(const std::vector<N*>& nds, bool reverse) // static
{
    int num_nd = nds.size() ;
    std::stringstream ss ;
    ss << "sn::Brief num_nd " << num_nd << ( reverse ? " DESC ORDER REVERSED " : "-" ) << std::endl ;
    for(int i=0 ; i < num_nd ; i++) ss << Desc(nds[reverse ? num_nd - 1 - i : i]) << std::endl ;
    std::string str = ss.str();
    return str ;
}



inline int sn::Index(const sn* n){ return pool ? pool->index(n) : -1 ; } // static
inline int  sn::idx() const { return Index(this); } // to match snd.hh
inline int  sn::index() const { return Index(this); }
inline int  sn::parent_index() const { return parent ? Index(parent) : -2 ; }
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


inline sn* sn::get_left() const
{
    sn* l = nullptr ;
#ifdef WITH_CHILD
    assert( child.size() == 2 );
    l = child[0] ;
#else
    l = left ;
#endif
    return l ;
}

inline sn* sn::get_right() const
{
    sn* r = nullptr ;
#ifdef WITH_CHILD
    assert( child.size() == 2 );
    r = child[1] ;
#else
    r = right ;
#endif
    return r ;
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

    if(level() > 1) std::cout << "sn::sibling_index"
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

    if(level() > 1) std::cout << "sn::next_sibling"
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
    if(level() > 1) std::cout
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
    n.note  = x->note  ;
    n.coincide  = x->coincide  ;

    assert( sizeof(n.label) == sizeof(x->label) );
    strncpy( &n.label[0], x->label, sizeof(n.label) );


    if(level() > 1) std::cout
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
    if(level() > 0) std::cout << "sn::Import" << std::endl ;
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
    if(level() > 0) std::cout << "sn::Import_r d " << d << " " << ( _n ? _n->desc() : "(null)" ) << std::endl ;
    if(_n == nullptr) return nullptr ;

#ifdef WITH_CHILD
    sn* n = Create( _n->typecode , nullptr, nullptr );
    n->complement = _n->complement ;
    n->lvid = _n->lvid ;
    n->xform = s_tv::pool->getbyidx(_n->xform) ;
    n->param = s_pa::pool->getbyidx(_n->param) ;
    n->aabb =  s_bb::pool->getbyidx(_n->aabb) ;

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
    n->xform = s_tv::pool->getbyidx(_n->xform) ;
    n->param = s_pa::pool->getbyidx(_n->param) ;
    n->aabb = s_bb::pool->getbyidx(_n->aabb) ;
#endif
    return n ;
}



/**
sn::sn ctor
-------------

note that sn::pid cannot be relied upon for indexing into the pool
as other ctor/dtor can change the pool while this holds on to the old stale pid

**/

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
    note(0),
    coincide(0),
    pid(pool ? pool->add(this) : -1),
    subdepth(0)
{
    if(level() > 1) std::cout << "[ sn::sn " << id() << "\n" ;
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

    if(level() > 1) std::cout << "] sn::sn " << id() << "\n" ;
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
    if(level() > 1) std::cout << "[ sn::~sn " << id() << "\n" ;

    delete xform ;
    delete param ;
    delete aabb  ;
    // parent is not deleted : as it is regarded as weakly linked (ie not owned by this node)


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

    if(level() > 1) std::cout << "] sn::~sn " << id() << "\n" ;
}







#ifdef WITH_CHILD
/**
sn::disown_child
------------------

* find ch pointer within child vector
* appy std::vector::erase with the iterator

HUH: contrary to prior note the std::vector::erase
simply removes from the vector, it DOES NOT
call the dtor of the erased pointer

Prior note was wrong::

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


/**
sn::deepcopy
-------------

Note that the xform, param and aabb pointers are
now copied by sn::copy which is invoked from sn::deepcopy.
This means that the copied tree is now fully independent
from the source tree. This allows the source tree to be
fully deleted without effecting the copied tree.

**/

inline sn* sn::deepcopy() const
{
    return deepcopy_r(0);
}

/**
sn::deepcopy_r
----------------

The default copy ctor copies the child vector, but that is a shallow copy
just duplicating pointers into the new child vector.
Hence within the child loop below those shallow copies are disowned
(removed from the child vector) and deep copies are made and added
to the child vector of the copy

**/

inline sn* sn::deepcopy_r(int d) const
{
    sn* n = copy();

#ifdef WITH_CHILD
    for(int i=0 ; i < int(child.size()) ; i++)
    {
        sn* ch = child[i] ;
        n->disown_child( ch ) ;          // remove shallow copied child from the vector
        sn* deep_ch = ch->deepcopy_r(d+1) ;
        n->child.push_back( deep_ch );
    }
#else
    // whether nullptr or not the shallow default copy
    // should have copied left and right
    assert( n->left == left );
    assert( n->right == right );
    // but thats just a shallow copy so replace here with deep copies
    n->left  = left  ? left->deepcopy_r(d+1) : nullptr ;
    n->right = right ? right->deepcopy_r(d+1) : nullptr ;
#endif

    return n ;
}



inline sn* sn::deepcopy_excluding_leaf(const sn* l) const
{
    return deepcopy_excluding_leaf_r(0, l );
}

inline sn* sn::deepcopy_excluding_leaf_r(int d, const sn* l) const
{
    if(this == l) return nullptr ;

    sn* n = copy();

#ifdef WITH_CHILD
    for(int i=0 ; i < int(child.size()) ; i++)
    {
        sn* ch = child[i] ;
        n->disown_child( ch ) ;          // remove shallow copied child from the vector
        sn* deep_ch = ch->deepcopy_excluding_leaf_r(d+1, l ) ;
        n->child.push_back( deep_ch );
    }
#else
    // whether nullptr or not the shallow default copy
    // should have copied left and right
    assert( n->left == left );
    assert( n->right == right );
    // but thats just a shallow copy so replace here with deep copies
    n->left  = left  ? left->deepcopy_excluding_leaf_r(d+1) : nullptr ;
    n->right = right ? right->deepcopy_excluding_leaf_r(d+1) : nullptr ;
#endif
    return n ;
}










/**
sn::copy
---------

default copy ctor just shallow copies the pointers and the pid,
so fix those up here to make the copy truly independant of the
original node

**/

inline sn* sn::copy() const
{
    sn* n = new sn(*this) ;

    n->pid = pool ? pool->add(n) : -1 ;
    n->parent = nullptr ;
    n->xform = xform ? xform->copy() : nullptr ;
    n->param = param ? param->copy() : nullptr ;
    n->aabb  = aabb ? aabb->copy() : nullptr ;

    return n ;
}


/**
sn::DeepCopy
-------------

It makes most sense to use this for copying prim nodes

**/

inline void sn::DeepCopy(std::vector<sn*>& p1, const std::vector<sn*>& p0) // static
{
    for(unsigned i=0 ; i < p0.size() ; i++) p1.push_back(p0[i]->deepcopy());
}



/**
sn::set_child
---------------

When *ch* (the new child to be set) is from within the tree and
not a newly created disconnected node, eg when pruning and moving
around nodes it is necessary to deepcopy that preexisting node
using copy:true


copy:false

   * parent of *ch* is set to this
   * reference to child[ix] slot is obtained
   * old occupier of the child[ix] slot is deleted (for LEAK:false the standard)
   * child[ix] slot pointer set to *ch*

copy:true

   * as above but deepcopy the *ch* first


* NB former slot residents are deleted

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


inline void sn::set_child_leaking_prior( int ix, sn* ch, bool copy )
{
    sn* new_ch = copy ? ch->deepcopy() : ch ;
    new_ch->parent = this ;

#ifdef WITH_CHILD
    assert( ix < int(child.size()) );
    sn*& target = child[ix] ;
    target = new_ch ;
#else
    sn** target = ix == 0 ? &left : &right ;
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

inline bool sn::is_complement() const           { return complement == 1 ; }
inline bool sn::is_complement_primitive() const { return is_complement() && is_primitive() ; }


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


inline int sn::num_notsupported_node() const
{
    return num_notsupported_node_r(0);
}

inline int sn::num_notsupported_node_r(int d) const
{
    int nn = typecode == CSG_NOTSUPPORTED ? 1 : 0 ;
#ifdef WITH_CHILD
    for(int i=0 ; i < int(child.size()) ; i++) nn += child[i]->num_notsupported_node_r(d+1) ;
#else
    nn += left ? left->num_notsupported_node_r(d+1) : 0 ;
    nn += right ? right->num_notsupported_node_r(d+1) : 0 ;
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
        if(level()>0) std::cout
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



/**
sn::typecodes
-------------

Collect distinct typecode into the set for nodes with subdepth >= minsubdepth,
minsubdepth=0 corresponds to entire tree.

**/

inline void sn::typecodes(std::set<int>& tcs, int minsubdepth ) const
{
    typecodes_r(tcs, minsubdepth);
}

inline void sn::typecodes_r(std::set<int>& tcs, int minsubdepth ) const
{
    if(subdepth >= minsubdepth) tcs.insert(typecode);
#ifdef WITH_CHILD
    for(int i=0 ; i < int(child.size()) ; i++) child[i]->typecodes_r(tcs, minsubdepth ) ;
#else
    if(left && right )
    {
        left->typecodes_r( tcs, minsubdepth );
        right->typecodes_r( tcs, minsubdepth );
    }
#endif
}


inline std::string sn::desc_typecodes() const
{
    std::stringstream ss ;

    std::set<int> tcs ;
    int minsubdepth = 0;
    typecodes(tcs, minsubdepth );

    ss << "[sn::desc_typecodes\n" ;
    for (int tc : tcs) ss << CSG::Name(tc) << "\n" ;
    ss << "]sn::desc_typecodes\n" ;

    std::string str = ss.str();
    return str ;
}

/**
sn::typecodes_count
---------------------

Returns the number of typecodes in the tcq query vector that
are present in the set of typecodes of the sn node tree.

**/

inline int sn::typecodes_count(const std::vector<int>& tcq, int minsubdepth) const
{
    std::set<int> tcs ;
    typecodes(tcs, minsubdepth );

    int count = 0;
    for (int value : tcq) if (tcs.count(value)) ++count;
    return count;
}

inline std::string sn::desc_typecodes_count() const
{
    std::vector<int> tcq = {CSG_TORUS, CSG_NOTSUPPORTED, CSG_CUTCYLINDER } ;

    int minsubdepth = 0;
    int count = typecodes_count(tcq, minsubdepth );

    std::stringstream ss ;
    ss << "[sn::desc_typecodes_count [" << count << "]\n" ;
    for (int tc : tcq) ss << CSG::Name(tc) << "\n" ;
    ss << "]sn::desc_typecodes_count [" << count << "]\n" ;

    std::string str = ss.str();
    return str ;
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

inline std::string sn::desc_prim() const
{
    bool hint_lnprd = is_hint_listnode_prim_discontiguous();
    bool hint_lnprc = is_hint_listnode_prim_contiguous();

    std::stringstream ss ;
    ss
       << " idx " << std::setw(4) << index()
       << " lvid " << std::setw(3) << lvid
       << " " << tag()
       << ( aabb ? aabb->desc() : "-" )
       << " " << ( hint_lnprd ? "HINT_LNPRD" : "" )
       << " " << ( hint_lnprc ? "HINT_LNPRC" : "" )
       ;

    std::string str = ss.str();
    return str ;
}

inline std::string sn::desc_prim_all(bool reverse) const
{
    std::vector<const sn*> prim ;
    collect_prim(prim);
    bool ascending = true ;
    OrderPrim<const sn>(prim, sn::AABB_ZMin, ascending );

    return DescPrim(prim, reverse);
}


inline std::string sn::id() const
{
    std::stringstream ss ;
    ss << "sn::id"
       << " pid " << pid
       << " idx " << idx()
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
       << " bb.desc " << ( aabb ? aabb->desc() : "-" )
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

inline std::string sn::desc_this() const
{
    std::stringstream ss ;
    ss << " 0x" << std::hex << uint64_t(this) << " " << std::dec ;
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
    ss << std::setw(3) << d << ":" << desc_this() << desc()  << std::endl ;
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


inline std::string sn::detail_r() const
{
    std::stringstream ss ;
    ss << "sn::detail_r" << std::endl ;
    detail_r(0, ss);
    std::string str = ss.str();
    return str ;
}
inline void sn::detail_r(int d, std::stringstream& ss) const
{
    ss << std::setw(3) << d << ":" << desc_this() << detail()  << std::endl ;
    for(int i=0 ; i < int(child.size()) ; i++) child[i]->detail_r(d+1, ss ) ;
}

inline std::string sn::detail() const
{
    std::stringstream ss ;
    ss << descXF() ;
    std::string str = ss.str();
    return str ;
}





inline std::string sn::render() const
{
    std::stringstream ss ;
    for(int mode=0 ; mode < NUM_MODE ; mode++) ss << render(mode) << std::endl ;
    std::string str = ss.str();
    return str ;
}

inline std::string sn::render_typetag() const { return render(TYPETAG);  }
inline std::string sn::render_parent() const {  return render(PARENT);  }

inline std::string sn::rdr() const {            return render_typetag();  }

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

    if(mode == MINIMAL || mode == PID)
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
        case NOTE:     md = MODE_NOTE     ; break ;
        case PARENT:   md = MODE_PARENT   ; break ;
        case IDX:      md = MODE_IDX      ; break ;
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
        case MINIMAL  : canvas->drawch( ix, iy, 0,0, 'o' )            ; break ;
        case TYPECODE : canvas->draw(   ix, iy, 0,0,  typecode  )     ; break ;
        case DEPTH    : canvas->draw(   ix, iy, 0,0,  depth )         ; break ;
        case SUBDEPTH : canvas->draw(   ix, iy, 0,0,  subdepth )      ; break ;
        case TYPETAG  : canvas->draw(   ix, iy, 0,0,  tag.c_str())    ; break ;
        case PID      : canvas->draw(   ix, iy, 0,0,  pid )           ; break ;
        case NOTE     : canvas->draw(   ix, iy, 0,0,  note )          ; break ;
        case PARENT   : canvas->draw(   ix, iy, 0,0,  parent_index()) ; break ;
        case IDX      : canvas->draw(   ix, iy, 0,0,  idx())          ; break ;
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
    if(level() > 0 ) std::cout << "[sn::ZeroTree num_leaves " << num_leaves << " height " << height << std::endl;
    sn* root = ZeroTree_r( height, op );
    if(level() > 0) std::cout << "]sn::ZeroTree " << std::endl ;
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

        if(level() > 0) std::cout << "sn::CommonOperatorTypeTree ZeroTree num_leaves " << num_leaves << std::endl ;
        if(level() > 1) std::cout << root->render(5) ;

        root->populate_leaftypes(leaftypes);

        if(level() > 0) std::cout << "sn::CommonOperatorTypeTree populated num_leaves " << num_leaves << std::endl ;
        if(level() > 1) std::cout << root->render(5) ;

        root->prune();

        if(level() > 0) std::cout << "sn::CommonOperatorTypeTree pruned num_leaves " << num_leaves << std::endl ;
        if(level() > 1) std::cout << root->render(5) ;
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

    if(level() > 0) std::cout
        << "sn::populate_leaftypes"
        << " num_leaves " << num_leaves
        << " num_nodes " << num_nodes
        << std::endl
        ;

    for(int i=0 ; i < num_nodes ; i++)
    {
        sn* n = order[i];
        if(level() > 1) std::cout
            << "sn::populate_leaftypes " << std::setw(3) << i
            << " n.desc " << n->desc()
            << std::endl
            ;
    }

    for(int i=0 ; i < num_nodes ; i++)
    {
        sn* n = order[i];

#ifdef WITH_CHILD
        if(level() > 1) std::cout
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
                if(level() > 1 ) std::cout
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

    if(level() > 0) std::cout
        << "sn::populate_leaves"
        << " num_leaves " << num_leaves
        << " num_nodes " << num_nodes
        << std::endl
        ;

    for(int i=0 ; i < num_nodes ; i++)
    {
        sn* n = order[i];
        if(level() > 1) std::cout
            << "sn::populate_leaves " << std::setw(3) << i
            << " n.desc " << n->desc()
            << std::endl
            ;
    }

    for(int i=0 ; i < num_nodes ; i++)
    {
        sn* n = order[i];

#ifdef WITH_CHILD
        if(level() > 1) std::cout
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
                if(level() > 1 ) std::cout
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
        if(level() > -1) std::cout << "sn::prune ERROR root still has dangle " << std::endl ;
    }

}

/**
sn::prune_r
-------------

* based on npy/NTreeBuilder
* returning to tree integrity relies on actions of set_left/set_right




l->is_lrzero::


      before                   after prune_r

            this                    this
            /   \                  /    \
           /     \                /      \
          l       r              0        r
         / \
        /   \
       0     0


l->is_rzero::


     before                    after prune_r

           this                    this
          /    \                  /     \
         /      \                /       \
        l        r             l=>"ll"    r
       / \
      /   \
     ll    0

**/

inline void sn::prune_r(int d)
{
    if(!is_operator()) return ;

    sn* l = get_left();
    sn* r = get_right();

    l->prune_r(d+1);
    r->prune_r(d+1);

    // postorder visit : so both children always visited before their parents

    if( l->is_lrzero() )     // left node is an operator which has both its left and right zero
    {
        set_left(sn::Zero(), false) ;       // prune : ie replace operator with CSG_ZERO placeholder
    }
    else if( l->is_rzero() )  // left node is an operator with left non-zero and right zero
    {
        sn* ll = l->get_left();
        set_left( ll, true );   // moving the lonely primitive up to higher elevation
    }

    if(r->is_lrzero())        // right node is operator with both its left and right zero
    {
        set_right(sn::Zero(), false) ;      // prune
    }
    else if( r->is_rzero() )  // right node is operator with its left non-zero and right zero
    {
        sn* rl = r->get_left() ;
        set_right(rl, true) ;         // moving the lonely primitive up to higher elevation
    }
}



/**
sn::has_dangle   (non recursive)
---------------------------------

A dangle node is an operator with a one or two placeholder children (aka zeros), eg::

      op           op           op
     /  \         /  \         /  \
    0    V       V    0       0    0

**/

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
eliminates subtraction operators by propagating negations down the tree using deMorgan rules.

Q: What about compound (aka listnodes) ?
A: From the point of view of the binary tree the listnode should be regarded as a primitive,
   so if a negation is signalled all the listed prims should be complemented.
   For example with hole subtraction the subtraction gets converted to intersect and the
   holes are complemented.

**/


inline void sn::positivize()
{
    positivize_r(false, 0);
}
inline void sn::positivize_r(bool negate, int d)
{
    if(is_primitive() || is_listnode())
    {
        if(negate)
        {
            if(is_primitive())
            {
                flip_complement();
            }
            else if(is_listnode())
            {
                flip_complement_child();
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


inline void sn::flip_complement()
{
    switch(complement)
    {
        case 0: complement = 1 ; break ;
        case 1: complement = 0 ; break ;
        default: assert(0)     ; break ;
    }
}
inline void sn::flip_complement_child()
{
    for(int i=0 ; i < num_child() ; i++)
    {
        sn* ch = get_child(i) ;
        ch->flip_complement() ;
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

/**
sn::set_lvid
-------------

Recursively set:

* lvid
* depth
* parent pointers : ordinarily this is not necessary as it is done in the ctor,
  but after deepcopy the parent pointers are scrubbed, so set that here too


**/

inline void sn::set_lvid(int lvid_)
{
    set_lvid_r(lvid_, 0);

    int chk = checktree();
    if( chk != 0 )
    {
        if(level() > 0 ) std::cout
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

        if(ch->parent == nullptr)
        {
            ch->parent = this ; // ordinarily not needed, but deepcopy scrubs parent pointers
        }
        else
        {
            assert( ch->parent == this ) ;
        }
    }
}


/**
sn::check_idx
--------------

Recursive check that all nodes of the tree are
accessible from the pool

**/

inline int sn::check_idx(const char* msg) const
{
    return check_idx_r(0, msg);
}
inline int sn::check_idx_r(int d, const char* msg) const
{
    int idx_ = idx(); // lookup contiguous index of this object within the pool of active nodes
    const sn* chk = Get(idx_);

    bool expect = chk == this ;
    if(!expect) std::cerr
        << "sn::check_idx_r"
        << " ERROR Get(idx()) != this  ? "
        << " idx_ " << idx_
        << " msg " << ( msg ? msg : "-" )
        << "\n"
        << " this.desc " << desc() << "\n"
        << " chk.desc  " << ( chk ? chk->desc() : "-" ) << "\n"
        ;

    int rc = expect ? 0 : 1 ;
    //assert(expect);

    for(int i=0 ; i < num_child() ; i++) rc += get_child(i)->check_idx_r(d+1, msg);
    return rc ;
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


inline const double* sn::getPA_data() const
{
    return param ? param->data() : nullptr ;
}

inline void sn::copyPA_data(double* dst) const
{
    const double* src = getPA_data() ;
    for(int i=0 ; i < 6 ; i++) dst[i] = src[i] ;
}





inline void sn::setBB( double x0, double y0, double z0, double x1, double y1, double z1 )
{
    bool bad_bbox =  x0 >= x1 || y0 >= y1 || z0 >= z1  ;
    if(bad_bbox) std::cerr
          << "sn::setBB BAD BOUNDING BOX "
          << "\n"
          << " x0 " << x0
          << " x1 " << x1
          << "\n"
          << " y0 " << y0
          << " y1 " << y1
          << "\n"
          << " z0 " << z0
          << " z1 " << z1
          << "\n"
          ;

    assert(!bad_bbox);


    if( aabb == nullptr ) aabb = new s_bb ;
    aabb->x0 = x0 ;
    aabb->y0 = y0 ;
    aabb->z0 = z0 ;
    aabb->x1 = x1 ;
    aabb->y1 = y1 ;
    aabb->z1 = z1 ;
}

inline void sn::setBB( double x0 )
{
    if( aabb == nullptr ) aabb = new s_bb ;
    aabb->x0 = -x0 ;
    aabb->y0 = -x0 ;
    aabb->z0 = -x0 ;
    aabb->x1 = +x0 ;
    aabb->y1 = +x0 ;
    aabb->z1 = +x0 ;
}

inline const double* sn::getBB_data() const
{
    return aabb ? aabb->data() : nullptr ;
}

/**
sn::copyBB_data
----------------

Canonical usage is from stree::get_combined_tran_and_aabb
which is canonically called after sn::postconvert has been
run for the CSG tree. The sn::postconvert does::

    positivize
    setAABB_TreeFrame_All
    uncoincide
    setAABB_LeafFrame_All

So when this sn::copyBB_data is called the AABB are
in leaf frame form.

**/

inline void sn::copyBB_data(double* dst) const
{
    const double* src = getBB_data() ;
    for(int i=0 ; i < 6 ; i++) dst[i] = src[i] ;
}



inline double sn::getBB_xmin() const { return aabb ? aabb->x0 : 0. ; }
inline double sn::getBB_ymin() const { return aabb ? aabb->y0 : 0. ; }
inline double sn::getBB_zmin() const { return aabb ? aabb->z0 : 0. ; }

inline double sn::getBB_xmax() const { return aabb ? aabb->x1 : 0. ; }
inline double sn::getBB_ymax() const { return aabb ? aabb->y1 : 0. ; }
inline double sn::getBB_zmax() const { return aabb ? aabb->z1 : 0. ; }

inline double sn::getBB_xavg() const { return ( getBB_xmin() + getBB_xmax() )/2. ; }
inline double sn::getBB_yavg() const { return ( getBB_ymin() + getBB_ymax() )/2. ; }
inline double sn::getBB_zavg() const { return ( getBB_zmin() + getBB_zmax() )/2. ; }


inline double sn::AABB_XMin( const sn* n ){ return n ? n->getBB_xmin() : 0. ; }
inline double sn::AABB_YMin( const sn* n ){ return n ? n->getBB_ymin() : 0. ; }
inline double sn::AABB_ZMin( const sn* n ){ return n ? n->getBB_zmin() : 0. ; }

inline double sn::AABB_XMax( const sn* n ){ return n ? n->getBB_xmax() : 0. ; }
inline double sn::AABB_YMax( const sn* n ){ return n ? n->getBB_ymax() : 0. ; }
inline double sn::AABB_ZMax( const sn* n ){ return n ? n->getBB_zmax() : 0. ; }

inline double sn::AABB_XAvg( const sn* n ){ return n ? n->getBB_xavg() : 0. ; }
inline double sn::AABB_YAvg( const sn* n ){ return n ? n->getBB_yavg() : 0. ; }
inline double sn::AABB_ZAvg( const sn* n ){ return n ? n->getBB_zavg() : 0. ; }


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

inline std::string sn::descXF() const
{
    std::stringstream ss ;
    ss << "sn::descXF (s_tv)xform " << ( xform ? xform->desc() : "-" ) ;
    std::string str = ss.str();
    return str ;
}



inline sn* sn::Cylinder(double radius, double z1, double z2) // static
{
    assert( z2 > z1 );
    sn* nd = Create(CSG_CYLINDER);
    nd->setPA( 0.f, 0.f, 0.f, radius, z1, z2)  ;
    nd->setBB( -radius, -radius, z1, +radius, +radius, z2 );
    return nd ;
}

/**
sn::CutCylinder
----------------

HMM asis s_pa.h restricts to 6 param so:

1. require endnormals to have Y component of zero, bringing down to 4 param
2. symmetric +dz -dz height : 1 param
3. just radius (not inner) : 1 param

**/

inline sn* sn::CutCylinder(
    double R,
    double dz,
    double _pz_nrm_x,
    double _pz_nrm_y,
    double _pz_nrm_z,
    double _nz_nrm_x,
    double _nz_nrm_y,
    double _nz_nrm_z ) // static
{
    assert( _pz_nrm_z > 0. );  // expect outwards normal away from +DZ top edge
    assert( _nz_nrm_z < 0. );  // expect outwards normal away from -DZ bot edge

    assert( _pz_nrm_y == 0. ); // simplifying assumptions for initial impl
    assert( _nz_nrm_y == 0. );

    double pz_nrm = std::sqrt( _pz_nrm_x*_pz_nrm_x + _pz_nrm_y*_pz_nrm_y +  _pz_nrm_z*_pz_nrm_z );
    double pz_nrm_x = _pz_nrm_x/pz_nrm ;
    double pz_nrm_y = _pz_nrm_y/pz_nrm ;
    double pz_nrm_z = _pz_nrm_z/pz_nrm ;

    double nz_nrm = std::sqrt( _nz_nrm_x*_nz_nrm_x + _nz_nrm_y*_nz_nrm_y + _nz_nrm_z*_nz_nrm_z );
    double nz_nrm_x = _nz_nrm_x/nz_nrm ;
    double nz_nrm_y = _nz_nrm_y/nz_nrm ;
    double nz_nrm_z = _nz_nrm_z/nz_nrm ;


    double zmin, zmax ;
    CutCylinderZRange(zmin, zmax, R, dz, pz_nrm_x, pz_nrm_y, pz_nrm_z, nz_nrm_x, nz_nrm_y, nz_nrm_z );


    bool dump = false ;
    if(dump) std::cout
       << "sn::CutCylinder\n"
       << " R " << std::setw(10) << std::fixed << std::setprecision(6) << R << "\n"
       << " dz " << std::setw(10) << std::fixed << std::setprecision(6) << dz << "\n"
       << " _pz_nrm_x " << std::setw(10) << std::fixed << std::setprecision(6) << _pz_nrm_x << "\n"
       << " _pz_nrm_y " << std::setw(10) << std::fixed << std::setprecision(6) << _pz_nrm_y << "\n"
       << " _pz_nrm_z " << std::setw(10) << std::fixed << std::setprecision(6) << _pz_nrm_z << "\n"
       << "\n"
       << " pz_nrm_x " << std::setw(10) << std::fixed << std::setprecision(6) << pz_nrm_x << "\n"
       << " pz_nrm_y " << std::setw(10) << std::fixed << std::setprecision(6) << pz_nrm_y << "\n"
       << " pz_nrm_z " << std::setw(10) << std::fixed << std::setprecision(6) << pz_nrm_z << "\n"
       << "\n"
       << " _nz_nrm_x " << std::setw(10) << std::fixed << std::setprecision(6) << _nz_nrm_x << "\n"
       << " _nz_nrm_y " << std::setw(10) << std::fixed << std::setprecision(6) << _nz_nrm_y << "\n"
       << " _nz_nrm_z " << std::setw(10) << std::fixed << std::setprecision(6) << _nz_nrm_z << "\n"
       << "\n"
       << " nz_nrm_x " << std::setw(10) << std::fixed << std::setprecision(6) << nz_nrm_x << "\n"
       << " nz_nrm_y " << std::setw(10) << std::fixed << std::setprecision(6) << nz_nrm_y << "\n"
       << " nz_nrm_z " << std::setw(10) << std::fixed << std::setprecision(6) << nz_nrm_z << "\n"
       << "\n"
       << " zmax " << std::setw(10) << std::fixed << std::setprecision(6) << zmax << "\n"
       << " zmin " << std::setw(10) << std::fixed << std::setprecision(6) << zmin << "\n"
       << "\n"
       ;

    sn* nd = Create(CSG_CUTCYLINDER);
    nd->setPA( R, dz, pz_nrm_x, pz_nrm_z, nz_nrm_x, nz_nrm_z)  ;
    nd->setBB( -R, -R, zmin, +R, +R, zmax );

    return nd ;
}


/**
sn::CutCylinderZRange
-----------------------

::

     (pz_nrm_x,0,pz_nrm_z) "highNormal"
       \
        \
         \     A              +dz+pdz
          P    | - - - - - -  +dz
     B    |    |
     |    |    |
     |    |    +
     |    C    |
     +    |    |
     |    |    |
     |    |    D
     |    N       - - - - -  -dz
     C     \                 -dz-ndz
            \
             \
           (nz_nrm_x,0,nz_nrm_z) "lowNormal"


          |    |
          0   radius


Vectors PA and PB are perpendicular to the normal and in opposite directions::

    ( pz_nrm_z, 0, -pz_nrm_x)   # PA as pz_nrm_z > 0
    (-pz_nrm_z, 0,  pz_nrm_x)   # PB

Similar triangles "Z"/"X" (equivalent to equating tangents)::

    pdz/R = (-pz_nrm_x)/(pz_nrm_z)
    pdz   = R*std::abs(-pz_nrm_x)/(pz_nrm_z)    # pz_nrm_z>0.  R > 0. => pdz > 0.
    ## do not known the sign of pz_nrm_x : the cut could be either way, but want maximum z so take std::abs to pick extreme

Vectors ND and NC are perpendicular to normal and in opposite directions::

    ( nz_nrm_z, 0, -nz_nrm_x )   ##  NC   as nz_nrm_z < 0.
    (-nz_nrm_z, 0,  nz_nrm_x )   ##  ND

Similar triangles "Z"/"X"::

     ndz/R = (nz_nrm_x/-nz_nrm_z)
     ndz = R*std::abs(nz_nrm_x)/(-nz_nrm_z)      # nz_nrm_z < 0.  R > 0. => ndz > 0.
     ## again do not known the sign of nz_nrm_x : the cut could be either way, but want minimum z so take std::abs to pick extreme

**/

inline void sn::CutCylinderZRange(
    double& zmin,
    double& zmax,
    double R,
    double dz,
    double pz_nrm_x,
    double pz_nrm_y,
    double pz_nrm_z,
    double nz_nrm_x,
    double nz_nrm_y,
    double nz_nrm_z ) // static
{
    double pdz = R*std::abs(-pz_nrm_x)/(pz_nrm_z) ;
    assert( pdz > 0. );
    zmax = dz + pdz ;

    double ndz = R*std::abs(nz_nrm_x)/(-nz_nrm_z) ;
    assert( ndz > 0. );
    zmin = -dz -ndz ;
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
    nd->setPA( fx, fy, fz, 0., 0., 0. );
    nd->setBB( -fx*0.5 , -fy*0.5, -fz*0.5, fx*0.5 , fy*0.5, fz*0.5 );
    return nd ;
}


/**
sn::Torus
----------

BB now accounts for phi range using sgeomtools.h based on G4GeomTools

Square torus for ease of illustration::

    +---------------------------------------+
    |                                       |
    |                                       |
    |                                       |
    |       +-----------------------+       |
    |       |                       |       |
    |       |                       |       |
    |       |                       |       |
    |       |                       |       |
    |   +   |           +           |   +   |
    |       |                       |       |
    |       |                       |       |
    |       |                       |       |
    |       |                       |       |
    |       +-----------------------+       |
    |                                       |
    |                                       |
    |                                       |
    +---------------------------------------+

                        |     rtor      |

                                    |   |   |
                                    rmax rmax

                        |     rtor + rmax   |

                        | rtor-rmax |

**/

inline sn* sn::Torus(double rmin, double rmax, double rtor, double startPhi_deg, double deltaPhi_deg )
{
    double rext = rtor+rmax ;
    double rint = rtor-rmax ;

    double startPhi = startPhi_deg/180.*M_PI ;
    double deltaPhi = deltaPhi_deg/180.*M_PI ;
    double2 pmin ;
    double2 pmax ;
    sgeomtools::DiskExtent(rint, rext, startPhi, deltaPhi, pmin, pmax );

    sn* nd = Create(CSG_TORUS) ;
    nd->setPA( rmin, rmax, rtor, startPhi_deg, deltaPhi_deg, 0.  );
    nd->setBB( pmin.x, pmin.y, -rmax, pmax.x, pmax.y, +rmax );

    return nd ;
}

inline sn* sn::Notsupported() // static
{
    sn* nd = Create(CSG_NOTSUPPORTED);
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

inline void sn::ZNudgeExpandEnds(int lvid, std::vector<sn*>& prims, bool enable) // static
{
    int num_prim = prims.size() ;

    sn* lower = prims[0] ;
    sn* upper = prims[prims.size()-1] ;
    bool can_znudge_ends = lower->can_znudge() && upper->can_znudge() ;
    assert( can_znudge_ends );

    double lower_zmin = lower->zmin() ;
    double upper_zmax = upper->zmax() ;
    bool z_expect = upper_zmax > lower_zmin  ;

    if(level() > 0) std::cout
       << std::endl
       << "sn::ZNudgeExpandEnds "
       << " lvid " << lvid
       << " num_prim " << num_prim
       << " enable " << ( enable ? "YES" : "NO " )
       << " level " << level()
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
**/

inline void sn::ZNudgeOverlapJoints(int lvid, std::vector<sn*>& prims, bool enable ) // static
{
    int num_prim = prims.size() ;
    assert( num_prim > 1 && "one prim has no joints" );

    bool dump = level() > 0 ;

    std::stringstream ss ;
    std::ostream* out = dump ? &ss : nullptr  ;

    if(out) *out
       << std::endl
       << "sn::ZNudgeOverlapJoints "
       << " lvid " << lvid
       << " num_prim " << num_prim
       << " enable " << ( enable ? "YES" : "NO " )
       << " level " << level()
       << std::endl
       << ZDesc(prims)
       << std::endl
       ;

    for(int i=1 ; i < num_prim ; i++)
    {
        sn* lower = prims[i-1] ;
        sn* upper = prims[i] ;
        ZNudgeOverlapJoint(lvid, i, lower, upper, enable, out );
    }

    if(out)
    {
        std::string str = ss.str();
        std::cout << str ;
    }
}

/**
sn::ZNudgeOverlapJoint
-----------------------

This is used from U4Polycone nudging.

It is not so easy to use this with the more general sn::uncoincide nudging
because in this case there are transforms involved, so the zmax/zmin param
may not look coincident but with the transforms they are.



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

As they are virtual it doesnt matter for physics in this case :
but that doesnt stop it being an issue.

**/

inline void sn::ZNudgeOverlapJoint(int lvid, int i, sn* lower, sn* upper, bool enable, std::ostream* out  ) // static
{
    bool can_znudge_ = lower->can_znudge() && upper->can_znudge() ;
    if(!can_znudge_) std::raise(SIGINT) ;
    assert( can_znudge_ );

    double dz = 1. ;
    double lower_zmax = lower->zmax();
    double upper_zmin = upper->zmin() ;
    bool z_coincident_joint = std::abs( lower_zmax - upper_zmin ) < Z_EPSILON  ;

    double upper_rperp_at_zmin = upper->rperp_at_zmin() ;
    double lower_rperp_at_zmax = lower->rperp_at_zmax() ;

    if(out) *out
        << "sn::ZNudgeOverlapJoint"
        << " ("<< i-1 << "," << i << ") "
        << " lower_zmax " << lower_zmax
        << " upper_zmin " << upper_zmin
        << " z_coincident_joint " << ( z_coincident_joint ? "YES" : "NO " )
        << " enable " << ( enable ? "YES" : "NO " )
        << " upper_rperp_at_zmin " << upper_rperp_at_zmin
        << " lower_rperp_at_zmax " << lower_rperp_at_zmax
        << std::endl
        ;

    if(!z_coincident_joint) return ;

    if( lower_rperp_at_zmax > upper_rperp_at_zmin )
    {
        upper->decrease_zmin( dz );
        if(out) *out
            << "sn::ZNudgeOverlapJoint"
            << " lvid " << lvid
            << " lower_rperp_at_zmax > upper_rperp_at_zmin : upper->decrease_zmin( dz ) "
            << "  : expand upper down into bigger lower "
            << std::endl
            ;
    }
    else
    {
        lower->increase_zmax( dz );
        if(out) *out
            << "sn::ZNudgeOverlapJoints"
            << " lvid " << lvid
            << " !(lower_rperp_at_zmax > upper_rperp_at_zmin) : lower->increase_zmax( dz ) "
            << "  : expand lower up into bigger upper "
            << std::endl
            ;
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


inline unsigned sn::NameHint(const char* name) // static
{
    unsigned hint = 0 ;
    if(     strstr(name, _HINT_LISTNODE_PRIM_DISCONTIGUOUS) != nullptr) hint = HINT_LISTNODE_PRIM_DISCONTIGUOUS ;
    else if(strstr(name, _HINT_LISTNODE_PRIM_CONTIGUOUS)    != nullptr) hint = HINT_LISTNODE_PRIM_CONTIGUOUS ;
    return hint ;
}


/**
sn::set_hint_note
------------------

Canonically invoked from U4Solid::init_Tree based on the solid name

**/

inline void sn::set_hint_note(unsigned hint)
{
    if(hint == 0u) return ;
    if(      (hint & HINT_LISTNODE_PRIM_DISCONTIGUOUS) != 0 )  note |= HINT_LISTNODE_PRIM_DISCONTIGUOUS ;
    else if( (hint & HINT_LISTNODE_PRIM_CONTIGUOUS)    != 0 )  note |= HINT_LISTNODE_PRIM_CONTIGUOUS    ;
}

inline void sn::set_hint_listnode_prim_discontiguous(){ set_hint_note( HINT_LISTNODE_PRIM_DISCONTIGUOUS ); }
inline void sn::set_hint_listnode_prim_contiguous(){    set_hint_note( HINT_LISTNODE_PRIM_CONTIGUOUS    ); }

inline bool sn::is_hint_listnode_prim_discontiguous() const { return ( note & HINT_LISTNODE_PRIM_DISCONTIGUOUS ) != 0 ; }
inline bool sn::is_hint_listnode_prim_contiguous() const {    return ( note & HINT_LISTNODE_PRIM_CONTIGUOUS )    != 0 ; }





inline void sn::increase_zmax( double dz )
{
    if(typecode == CSG_CONE)
    {
        increase_zmax_cone( dz );
    }
    else
    {
        increase_zmax_( dz );
    }
    note |= NOTE_INCREASE_ZMAX ;
}

inline void sn::decrease_zmin( double dz )
{
    if(typecode == CSG_CONE)
    {
        decrease_zmin_cone( dz );
    }
    else
    {
        decrease_zmin_( dz );
    }
    note |= NOTE_DECREASE_ZMIN ;
}





/**
sn::increase_zmax_
------------------

Expand upwards in +Z direction::

    +~~~~~~~~+  zmax + dz  (dz > 0.)
    +--------+  zmax
    |        |
    |        |
    +--------+  zmin

**/
inline void sn::increase_zmax_( double dz )
{
    assert( dz > 0. );
    double _zmax = zmax();
    double new_zmax = _zmax + dz ;

    std::cout
        << "sn::increase_zmax_"
        << " lvid " << lvid
        << " _zmax "    << std::fixed << std::setw(7) << std::setprecision(2) << _zmax
        << " dz "       << std::fixed << std::setw(7) << std::setprecision(2) << dz
        << " new_zmax " << std::fixed << std::setw(7) << std::setprecision(2) << new_zmax
        << std::endl
        ;

    set_zmax(new_zmax);
}





/**
sn::decrease_zmin_
--------------------

Expand downwards in -Z direction::

    +--------+  zmax
    |        |
    |        |
    +--------+  zmin
    +~~~~~~~~+  zmin - dz    (dz > 0.)

**/
inline void sn::decrease_zmin_( double dz )
{
    assert( dz > 0. );
    double _zmin = zmin();
    double new_zmin = _zmin - dz ;

    std::cout
        << "sn::decrease_zmin_"
        << " lvid " << lvid
        << " _zmin "    << std::fixed << std::setw(7) << std::setprecision(2) << _zmin
        << " dz "       << std::fixed << std::setw(7) << std::setprecision(2) << dz
        << " new_zmin " << std::fixed << std::setw(7) << std::setprecision(2) << new_zmin
        << std::endl
        ;

    set_zmin(new_zmin);
}


/**
sn::increase_zmax_cone
------------------------

Impl from ncone::decrease_z1

This avoids increase_zmax and decrease_zmin
changing angle of the cone by proportionately changing
rperp_at_zmin and rperp_at_zmax



                   new_rperp_at_zmax : new_r2
                    |
                    | rperp_at_zmax  : r2
                    | |
                    | |
              +~~~~~+ |        new_zmax : new_z2
             /   .   \|
            +----.----+        zmax : z2           z2 > z1  by assertion
           /     .     \
          /      .      \
         /       .       \
        +--------.--------+    zmin : z1
       /         .        |\
      +~~~~~~~~~~.~~~~~~~~|~+  new_zmin : new_z1
                          | |
                          | |
                          rperp_at_zmin : r1
                            |
                           new_rperp_at_zmin : new_r1


Consider increasing zmax by dz (dz > 0) to new_zmax
while keeping the same cone angle::


       new_z2 - z1       z2 - z1
      -------------  =  ----------       ratios of corresponding z and r diffs
       new_r2 - r1       r2 - r1         must be equal for fixed cone angle


      new_r2 - r1     new_z2 - z1
      ----------  =  -------------       collect r and z terms on each side
       r2  - r1        z2 - z1

       new_r2 =   r1 +  (r2 - r1) ( new_z2 - z1 )/(z2 - z1)


Similarly for decreasing zmin by dz (dz > 0) to new_zmin


     new_r1 - r2          r2 - r1
   ---------------  =   ----------
     new_z1 - z2          z2 - z1



     new_r1 = r2 + ( r2 - r1 )*(new_z1 - z2) / (z2 - z1 )


**/



inline void sn::increase_zmax_cone( double dz )
{
    double r1 = param->value(0) ;  // aka rperp_at_zmin
    double z1 = param->value(1) ;  // aka zmin
    double r2 = param->value(2) ;  // aka rperp_at_zmax
    double z2 = param->value(3) ;  // aka zmax

    double new_z2 = z2 + dz ;
    double new_r2 = r1 +  (r2 - r1)*( new_z2 - z1 )/(z2 - z1)  ;

    std::cout
        << "sn::increase_zmax_cone"
        << " lvid " << lvid
        << " z2 " << z2
        << " r2 " << r2
        << " dz " << dz
        << " new_z2 " << new_z2
        << " new_r2 " << new_r2
        << std::endl
        ;

    set_zmax(new_z2);
    set_rperp_at_zmax(new_r2 < 0. ? 0. : new_r2);
}


inline void sn::decrease_zmin_cone( double dz )
{
    double r1 = param->value(0) ;  // aka rperp_at_zmin
    double z1 = param->value(1) ;  // aka zmin
    double r2 = param->value(2) ;  // aka rperp_at_zmax
    double z2 = param->value(3) ;  // aka zmax

    double new_z1 = z1 - dz ;
    double new_r1 = r2 + (r2 - r1)*(new_z1 - z2) /(z2 - z1 ) ;

    std::cout
        << "sn::decrease_zmin_cone"
        << " lvid " << lvid
        << " z1 " << z1
        << " r1 " << r1
        << " dz " << dz
        << " new_z1 " << new_z1
        << " new_r1 " << new_r1
        << std::endl
        ;

    set_zmin(new_z1);
    set_rperp_at_zmin(new_r1 < 0. ? 0. : new_r1);

}







inline double sn::zmin() const
{
    assert( can_znudge() );
    double v = 0. ;
    switch(typecode)
    {
        case CSG_CYLINDER: v = param->value(4) ; break ;
        case CSG_ZSPHERE:  v = param->value(4) ; break ;
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
        case CSG_ZSPHERE:  param->set_value(4, zmin_) ; break ;
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
        case CSG_ZSPHERE:  v = param->value(5) ; break ;
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
        case CSG_ZSPHERE:  param->set_value(5, zmax_) ; break ;
        case CSG_CONE:     param->set_value(3, zmax_) ; break ;
    }
}

inline double sn::rperp_at_zmax() const
{
    assert( can_znudge() );
    double v = 0. ;
    switch(typecode)
    {
        case CSG_CYLINDER: v = param->value(3)         ; break ;
        case CSG_ZSPHERE:  v = rperp_at_zmax_zsphere() ; break ;
        case CSG_CONE:     v = param->value(2)         ; break ;
    }
    return v ;
}

inline void sn::set_rperp_at_zmax(double rperp_) const
{
    assert( can_znudge() );
    switch(typecode)
    {
        case CSG_CYLINDER: param->set_value(3, rperp_) ; break ;
        case CSG_CONE:     param->set_value(2, rperp_) ; break ;
    }
}

inline double sn::rperp_at_zmin() const
{
    assert( can_znudge() );
    double v = 0. ;
    switch(typecode)
    {
        case CSG_CYLINDER: v = param->value(3)         ; break ;
        case CSG_ZSPHERE:  v = rperp_at_zmin_zsphere() ; break ;
        case CSG_CONE:     v = param->value(0)         ; break ;
    }
    return v ;
}





inline void sn::set_rperp_at_zmin(double rperp_) const
{
    assert( can_znudge() );
    switch(typecode)
    {
        case CSG_CYLINDER: param->set_value(3, rperp_) ; break ;
        case CSG_CONE:     param->set_value(0, rperp_) ; break ;
    }
}





/**
sn::Sphere_RPerp_At_Z
-----------------------

                :
                :
                *
                :   *
                :  /:    .
                : r : z
                :/  :       .
      +---------+---+-------+
                  p

      pp + zz = rr   =>   p = sqrt( rr - zz )

**/

inline double sn::Sphere_RPerp_At_Z(double r, double z)  // static
{
    assert( std::abs(z) <= r );
    return sqrt(r*r - z*z ) ;
}

/**
sn::rperp_at_zmin_zsphere sn::rperp_at_zmax_zsphere(
------------------------------------------------------

::

           _  _
        .          .
      /             \
     .               .
     +---------------+


HMM: For ellipsoid need to apply transform to the node local
param before using values at "tree level"


**/
inline double sn::rperp_at_zmin_zsphere() const
{
    double r = radius_sphere() ;
    double z = zmin();
    double p = Sphere_RPerp_At_Z( r, z );
    return p  ;
}
inline double sn::rperp_at_zmax_zsphere() const
{
    double r = radius_sphere() ;
    double z = zmax();
    double p = Sphere_RPerp_At_Z( r, z );
    return p ;
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


inline void sn::getParam_(double& p0, double& p1, double& p2, double& p3, double& p4, double& p5 ) const
{
    const double* d = param ? param->data() : nullptr ;
    p0 = d ? d[0] : 0. ;
    p1 = d ? d[1] : 0. ;
    p2 = d ? d[2] : 0. ;
    p3 = d ? d[3] : 0. ;
    p4 = d ? d[4] : 0. ;
    p5 = d ? d[5] : 0. ;
}
inline const double* sn::getParam() const { return param ? param->data() : nullptr ; }
inline const double* sn::getAABB()  const { return aabb ? aabb->data() : nullptr ; }

inline bool sn::hasAABB() const   // not-nullptr and not all zero
{
    const double* aabb = getAABB();
    return aabb != nullptr && !s_bb::AllZero(aabb) ;
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
inline sn* sn::Discontiguous(std::vector<sn*>& prims )
{
    sn* n = Compound( prims, CSG_DISCONTIGUOUS );
    return n ;
}

/**
sn::Compound
------------

Note there is no subNum/subOffset here, those are needed when
serializing the n-ary sn tree of nodes into CSGNode presumably.

**/

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

        if(level() > 0) std::cout
            << "sn::CommonOperatorTree after ZeroTree"
            << " num_leaves " << num_leaves
            << " level " << level()
            << std::endl
            ;
        if(level() > 1) std::cout << root->render(5) ;

        root->populate_leaves(leaves);

        if(level() > 0) std::cout
            << "sn::CommonOperatorTree after populate_leaves"
            << " num_leaves " << num_leaves
            << " level " << level()
            << std::endl
            ;
        if(level() > 1) std::cout << root->render(5) ;

        root->prune();

        if(level() > 0) std::cout
            << "sn::CommonOperatorTree after prun"
            << " num_leaves " << num_leaves
            << " level " << level()
            << std::endl
            ;
        if(level() > 1) std::cout << root->render(5) ;
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
sn::GetLVListnodes
-------------------

Q: What about repeated globals, will this yield duplicates ?
A: No, as are searching the sn::POOL which should have only one of each CSG node

**/

inline void sn::GetLVListnodes( std::vector<const sn*>& lns, int lvid ) // static
{
    std::vector<const sn*> nds ;
    GetLVNodes_(nds, lvid );

    int num_nd = nds.size();
    for(int i=0 ; i < num_nd ; i++)
    {
        const sn* n = nds[i];
        if(n->is_listnode()) lns.push_back(n) ;
    }
}

inline int sn::GetChildTotal(  const std::vector<const sn*>& nds ) // static
{
    int child_total = 0 ;
    int num_nd = nds.size();
    for(int i=0 ; i < num_nd ; i++)
    {
        const sn* n = nds[i];
        child_total += ( n ? n->num_child() : 0 ) ;
    }
    return child_total ;
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


inline void sn::GetLVNodes_( std::vector<const sn*>& nds, int lvid ) // static
{
    sn_find_lvid flv(lvid);
    pool->find_(nds, flv );
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
    return pool->getbyidx(idx) ;
}


/**
sn::GetLVRoot
---------------

First sn with the lvid and sn::is_root():true in (s_csg.h)pool

**/

struct sn_find_lvid_root
{
    int lvid ;
    sn_find_lvid_root(int q_lvid) : lvid(q_lvid) {}
    bool operator()(const sn* n){ return lvid == n->lvid && n->is_root() ; }
};


inline void sn::FindLVRootNodes( std::vector<sn*>& nds, int lvid ) // static
{
    sn_find_lvid_root flvr(lvid);
    pool->find(nds, flvr );
}


inline sn* sn::GetLVRoot( int lvid ) // static
{
    std::vector<sn*> nds ;
    FindLVRootNodes( nds, lvid );

    int count = nds.size() ;
    bool expect = count == 0 || count == 1 ;

    const char* _DUMP = "sn__GetLVRoot_DUMP" ;
    bool DUMP = ssys::getenvbool(_DUMP) ;
    if(DUMP || !expect) std::cout
        << _DUMP << ":" << ( DUMP ? "YES" : "NO " ) << "\n"
        << "Desc(nds)\n"
        << Desc(nds)
        << "\n"
        << " nds.size " << count
        << " expect " << ( expect ? "YES" : "NO " )
        << "\n"
        ;

    assert(expect);
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

inline uint64_t sn::getLVBinNode() const
{
    int h = max_binary_depth();
    uint64_t n = st::complete_binary_tree_nodes( h );
    if(false) std::cout
        << "sn::getLVBinNode"
        << " h " << h
        << " n " << n
        << "\n"
        ;
    return n ;
}

/**
sn::getLVSubNode
-------------------

Sum of children of compound nodes found beneath this node.
HMM: this assumes compound nodes only contain leaf nodes

Notice that the compound nodes themselves are regarded as part of
the binary tree.

**/

inline uint64_t sn::getLVSubNode() const
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

inline uint64_t sn::getLVNumNode() const
{
    uint64_t bn = getLVBinNode() ;
    uint64_t sn = getLVSubNode() ;
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
    const sn* root = GetLVRoot(lvid);  // first sn from pool with requested lvid that is_root
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

For BoxGridMultiUnion10:30_YX  which is grid of 3x3x3=27 multiunions with 7x7x7=343 prim each
getting bn 1 ns 343

THIS METHOD IS ALL ABOUT THE BINARY TREE

ACCESSING THE LISTED CHILDREN OF THE LISTNODE NEEDS
TO BE SEPARATE : BUT THE LISTNODE "HEAD" ITSELF
IS PART OF THE BINARY TREE SO SHOULD BE INCLUDED
IN THE RETURNED VECTOR

**/

inline void sn::getLVNodesComplete(std::vector<const sn*>& nds) const
{
    uint64_t bn = getLVBinNode();
    uint64_t ns = getLVSubNode();
    uint64_t numParts = bn + ns ;

    //nds.resize(numParts);  // MAYBE THIS SHOULD JUST BE bn ?
    nds.resize(bn);

    if(false) std::cout
        << "sn::getLVNodesComplete"
        << " bn " << bn
        << " ns " << ns
        << " numParts " << numParts
        << "\n"
        ;

    // assert( ns == 0 ); // CHECKING : AS IMPL LOOKS LIKE ONLY HANDLES BINARY NODES
    // WIP: need to detect listnode at CSGImport::importPrim_ level

    GetLVNodesComplete_r( nds, this, 0 );
}



/**
sn::GetLVNodesComplete_r
-------------------------

Serializes tree nodes into complete binary tree vector using
0-based binary tree level order layout of nodes in the vector.

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



inline void sn::SelectListNode(std::vector<const sn*>& lns, const std::vector<const sn*>& nds) // static
{
    for(unsigned i=0 ; i < nds.size() ; i++)
    {
       const sn* nd = nds[i] ;
       if(nd->is_listnode()) lns.push_back(nd);
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
sn::collect_prim
-----------------

Current impl includes listnode subs in the prim vector
as is_primitive judged by num child is true for listnode subs
and false for the listnode itself.

In a binary tree sense the listnode itself is a leaf encompassing
its subs but in an n-ary tree sense the subs are the leaves and
the listnode is their compound container parent.

**/

inline void sn::collect_prim( std::vector<const sn*>& prim ) const
{
    collect_prim_r(prim, 0);
}
inline void sn::collect_prim_r( std::vector<const sn*>& prim, int d ) const
{
    if(is_primitive()) prim.push_back(this);
    for(int i=0 ; i < num_child() ; i++) get_child(i)->collect_prim_r(prim, d+1) ;
}


inline bool sn::has_note( int q_note ) const
{
    return ( note & q_note ) != 0 ;
}

/**
sn::collect_prim_note
----------------------

Collect prim with sn::note that bitwise includes the q_note bit
Due to tail recursion this descends all the way before unwinding
and collecting any prim, so the order is from the bottom first.

**/

inline void sn::collect_prim_note( std::vector<sn*>& prim, int q_note )
{
    collect_prim_note_r(prim, q_note, 0);
}
inline void sn::collect_prim_note_r( std::vector<sn*>& prim, int q_note, int d )
{
    if(is_primitive() && has_note(q_note)) prim.push_back(this);
    for(int i=0 ; i < num_child() ; i++) get_child(i)->collect_prim_note_r(prim, q_note, d+1) ;
}



/**
sn::find_joint_to_candidate_listnode
--------------------------------------

::

    sn::desc pid   18 idx   18 typecode   1 num_node  19 num_leaf  10 maxdepth  9 is_positive_form Y lvid   0 tag un
    sn::render mode 4 TYPETAG
                                                       un

                                                 un       cy

                                           un       cy

                                     un       cy

                               un       cy

                         un       cy

                   un       cy

            [un]      cy

       in       cy

    cy    !cy




* off the top, where d=0, find all the listnode hinted prim (tail recursion, so the order is from bottom)
* recursive traverse, not descending to leaves

  * look for left and right listnode hinted sub-trees
  * where left has no listnode hinted and right has fun : return that joint node


Q: HMM does this require having run postconvert ?


**/


inline sn* sn::find_joint_to_candidate_listnode(std::vector<sn*>& prim, int q_note )
{
    sn* j = find_joint_to_candidate_listnode_r(0, prim, q_note );
    return j ;
}

inline sn* sn::find_joint_to_candidate_listnode_r( int d, std::vector<sn*>& prim, int q_note )
{
    if( d == 0 )
    {
        prim.clear();
        collect_prim_note(prim, q_note);
    }
    if(prim.size()==0) return nullptr ;  // none of the prim are hinted as listnode constituents

    sn* joint = nullptr ;
    if(num_child() != 2) return nullptr ;  // dont traverse leaves or compounds

    // look left and right
    std::vector<sn*> l_prim ;
    std::vector<sn*> r_prim ;
    get_child(0)->collect_prim_note(l_prim, q_note);
    get_child(1)->collect_prim_note(r_prim, q_note);

    if(r_prim.size() == 1 && l_prim.size() == 0 )
    {
        joint = this ;
    }
    else
    {
        for(int i=0 ; i < num_child() ; i++)
        {
            joint = get_child(i)->find_joint_to_candidate_listnode_r(d+1, prim, q_note ) ;
            if(joint) break ;
        }
    }
    return joint ;
}

inline bool sn::has_candidate_listnode(int q_note)
{
    std::vector<sn*> prim ;
    sn* j = find_joint_to_candidate_listnode(prim, q_note);
    return j != nullptr && prim.size() > 0 ;
}

inline bool sn::has_candidate_listnode_discontiguous(){ return has_candidate_listnode( HINT_LISTNODE_PRIM_DISCONTIGUOUS ) ; }
inline bool sn::has_candidate_listnode_contiguous(){    return has_candidate_listnode( HINT_LISTNODE_PRIM_CONTIGUOUS    ) ; }




/**
sn::CreateSmallerTreeWithListNode
-----------------------------------

Example tree, where the eight nodes with + are hinted as listnode prim::


                                                      (un)

                                                (un)      cy+

                                          (un)      cy+

                                    (un)      cy+

                              (un)      cy+

                       (un)       cy+

                  (un)      cy+

            {un}      cy+

       in       cy+

    cy    !cy


After shrinkage::


            {un}

       in       ln[cy+,cy+,cy+,cy+,cy+,cy+,cy+,cy+]

    cy    !cy



Manipulations needed::

1. find the joint node {un} between the extraneous (un) union nodes and
   the hinted prim cy+ to be incorporated into listnode

2. collect the cy+ hinted prim doing a deepcopy that includes xform, param, aabb

3. form the "ln" listnode from the deepcopied cy+ hinted prim

4. set the right node of the deepcopied joint node {un} to the listnode "ln"


Note that the created tree including param, xform, aabb is independent
of the original tree due to the use of deepcopy for everything.
This enables the entire old tree root to be deleted.

Note that deepcopy excludes the parent pointer, but includes xform, param, aabb

NB : this depends a lot on deepcopy

**/

inline sn* sn::CreateSmallerTreeWithListNode(sn* root0, int q_note ) // static
{
    std::cerr << "[sn::CreateSmallerTreeWithListNode\n" ;

    std::vector<sn*> prim0 ;  // populated with the hinted listnode prim
    sn* j0 = root0->find_joint_to_candidate_listnode(prim0, q_note);
    if(j0 == nullptr) return nullptr ;

    std::vector<sn*> prim1 ;
    sn::DeepCopy(prim1, prim0);

    sn* ln = sn::Compound( prim1, TypeFromNote(q_note) );

    sn* j1 = j0->deepcopy();

    j1->set_right( ln, false );  // NB this deletes the extraneous RHS just copied by j0->deepcopy
    //j1->set_child_leaking_prior(1, ln, false);


    // ordering may be critical here as nodes get created and deleted by the above

    std::cerr << "]sn::CreateSmallerTreeWithListNode\n" ;
    return j1 ;
}

inline sn* sn::CreateSmallerTreeWithListNode_discontiguous(sn* root0)
{
    return CreateSmallerTreeWithListNode( root0, HINT_LISTNODE_PRIM_DISCONTIGUOUS ) ;
}
inline sn* sn::CreateSmallerTreeWithListNode_contiguous(sn* root0)
{
    return CreateSmallerTreeWithListNode( root0, HINT_LISTNODE_PRIM_CONTIGUOUS ) ;
}

inline int sn::TypeFromNote(int q_note) // static
{
    int type = CSG_ZERO ;
    switch(q_note)
    {
       case HINT_LISTNODE_PRIM_DISCONTIGUOUS: type = CSG_DISCONTIGUOUS ; break ;
       case HINT_LISTNODE_PRIM_CONTIGUOUS:    type = CSG_CONTIGUOUS    ; break ;
    }
    assert( type != CSG_ZERO ) ;
    return type ;
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



inline double sn::radius_sphere() const
{
    double cx, cy, cz, r, z1, z2 ;
    getParam_(cx, cy, cz, r, z1, z2 );
    assert( cx == 0. && cy == 0. && cz == 0. ) ;
    return r ;
}


/**
sn::setAABB_LeafFrame
----------------------

See sn::postconvert for call context

Gets parameters and uses to setBB depending on typecode.
No transforms are used.

Migrated down from CSGNode::setAABBLocal as nudging
needs this done earlier.

**/

inline void sn::setAABB_LeafFrame()
{
    if(typecode == CSG_SPHERE)
    {
        double cx, cy, cz, r, a, b ;
        getParam_(cx, cy, cz, r, a, b );
        assert( cx == 0. && cy == 0. && cz == 0. );
        assert( a == 0. && b == 0. );
        setBB(  -r, -r, -r,  r, r, r  );
    }
    else if(typecode == CSG_ZSPHERE)
    {
        double cx, cy, cz, r, z1, z2 ;
        getParam_(cx, cy, cz, r, z1, z2 );
        assert( cx == 0. && cy == 0. && cz == 0. ) ;
        assert( z1 == zmin());
        assert( z2 == zmax());
        assert( z2 > z1 );
        setBB(  -r, -r, z1,  r, r, z2  );
    }
    else if( typecode == CSG_CONE )
    {
        double r1, z1, r2, z2, a, b ;
        getParam_(r1, z1, r2, z2, a, b );
        assert( a == 0. && b == 0. );
        double rmax = fmaxf(r1, r2) ;
        setBB( -rmax, -rmax, z1, rmax, rmax, z2 );
    }
    else if( typecode == CSG_BOX3 )
    {
        double fx, fy, fz, a, b, c ;
        getParam_(fx, fy, fz, a, b, c );
        assert( a == 0. && b == 0. && c == 0. );
        setBB( -fx*0.5 , -fy*0.5, -fz*0.5, fx*0.5 , fy*0.5, fz*0.5 );
    }
    else if( typecode == CSG_CYLINDER || typecode == CSG_OLDCYLINDER )
    {
        double px, py, a, radius, z1, z2 ;
        getParam_(px, py, a, radius, z1, z2 ) ;
        assert( px == 0. && py == 0. && a == 0. );
        setBB( px-radius, py-radius, z1, px+radius, py+radius, z2 );
    }
    else if( typecode == CSG_CUTCYLINDER  )
    {
        double R, dz, pz_nrm_x, pz_nrm_z, nz_nrm_x, nz_nrm_z ;
        getParam_( R, dz, pz_nrm_x, pz_nrm_z, nz_nrm_x, nz_nrm_z );

        double pz_nrm_y = 0. ;
        double nz_nrm_y = 0. ;

        double zmin = 0. ;
        double zmax = 0. ;
        CutCylinderZRange(zmin, zmax, R, dz, pz_nrm_x, pz_nrm_y, pz_nrm_z, nz_nrm_x, nz_nrm_y, nz_nrm_z );

        setBB( -R, -R, zmin, +R, +R, zmax );
    }
    else if( typecode == CSG_DISC )
    {
        double px, py, ir, r, z1, z2 ;
        getParam_(px, py, ir, r, z1, z2);
        assert( px == 0. && py == 0. && ir == 0. );
        setBB( px - r , py - r , z1, px + r, py + r, z2 );
    }
    else if( typecode == CSG_HYPERBOLOID )
    {
        double r0, zf, z1, z2, a, b ;
        getParam_(r0, zf, z1, z2, a, b ) ;
        assert( a == 0. && b == 0. );
        assert( z1 < z2 );
        const double rr0 = r0*r0 ;
        const double z1s = z1/zf ;
        const double z2s = z2/zf ;

        const double rr1 = rr0 * ( z1s*z1s + 1. ) ;
        const double rr2 = rr0 * ( z2s*z2s + 1. ) ;
        const double rmx = sqrtf(fmaxf( rr1, rr2 )) ;

        setBB(  -rmx,  -rmx,  z1,  rmx, rmx, z2 );
    }
    else if( typecode == CSG_TORUS )
    {
        double rmin, rmax, rtor, startPhi_deg, deltaPhi_deg, zero ;
        getParam_(rmin, rmax, rtor, startPhi_deg, deltaPhi_deg, zero) ;

        double rext = rtor+rmax ;
        double rint = rtor-rmax ;
        double startPhi = startPhi_deg/180.*M_PI ;
        double deltaPhi = deltaPhi_deg/180.*M_PI ;
        double2 pmin ;
        double2 pmax ;
        sgeomtools::DiskExtent(rint, rext, startPhi, deltaPhi, pmin, pmax );

        setBB( pmin.x, pmin.y, -rmax, pmax.x, pmax.y, +rmax );

    }
    else if( typecode == CSG_UNION || typecode == CSG_INTERSECTION || typecode == CSG_DIFFERENCE )
    {
        setBB( 0. );
    }
    else if( typecode == CSG_CONTIGUOUS || typecode == CSG_DISCONTIGUOUS )
    {
        // cannot define bbox of list header nodes without combining bbox of all the subs
        // so have to defer setting the bbox until all the subs are converted
        setBB( 0. );
    }
    else if( typecode == CSG_NOTSUPPORTED )
    {
        setBB( 0. );
    }
    else if( typecode == CSG_ZERO )
    {
        setBB( UNBOUNDED_DEFAULT_EXTENT );
    }
    else if( typecode == CSG_PHICUT )
    {
        setBB( UNBOUNDED_DEFAULT_EXTENT );
    }
    else if( typecode == CSG_THETACUT )
    {
        setBB( UNBOUNDED_DEFAULT_EXTENT );
    }
    else
    {
        std::cout
            << "sn::setAABB_LeafFrame : FATAL NOT IMPLEMENTED : "
            << " typecode " << typecode
            << " CSG::Name(typecode) " << CSG::Name(typecode)
            << std::endl
            ;
        assert(0);
        setBB( 0. );
    }
}


/**
sn::setAABB_LeafFrame_All
---------------------------

See sn::postconvert for call context

1. collects vector of all prim
2. for each prim call setAABB_LeadFrame

**/

inline void sn::setAABB_LeafFrame_All()
{
    std::vector<const sn*> prim ;
    collect_prim(prim);
    int num_prim = prim.size() ;
    for(int i=0 ; i < num_prim ; i++)
    {
        const sn* p = prim[i] ;
        sn* _p = const_cast<sn*>(p) ;
        _p->setAABB_LeafFrame() ;
    }
}


/*
sn::setAABB_TreeFrame_All formerly sn::setAABB
-----------------------------------------------

See sn::postconvert for call context

1. collect vector of all prim
2. setAABB_LeafFrame for each prim using param and typecode
3. gets transform for the node and inplace applies it to the BB

**/

inline void sn::setAABB_TreeFrame_All()
{
    std::vector<const sn*> prim ;
    collect_prim(prim);
    int num_prim = prim.size() ;

    for(int i=0 ; i < num_prim ; i++)
    {
        const sn* p = prim[i] ;
        sn* _p = const_cast<sn*>(p) ;
        _p->setAABB_LeafFrame() ;

        glm::tmat4x4<double> t(1.) ;
        glm::tmat4x4<double> v(1.) ;

        bool reverse = false ;
        std::ostream* out = nullptr ;
        p->getNodeTransformProduct(t, v, reverse, out );

        const double* pbb = _p->getBB_data() ;
        stra<double>::Transform_AABB_Inplace( const_cast<double*>(pbb),  t );
    }
}



/**
sn::postconvert
-----------------

This is called from U4Solid::init_Tree only from the depth zero solid (ie root node of trees)

Note that uncoincide needs the leaf bbox in tree frame so they can be compared
to look for coincidences and make some parameter nudges.
But subsequent code such as stree::get_combined_tran_and_aabb
expects the bbox to be in leaf frame, hence the bbox are recomputed
from the possibly nudged parameters in leaf frame after the uncoincide.

HOW TO HANDLE AABB FOR LISTNODE ?

* each sub has transform


**/

inline void sn::postconvert(int lvid)
{
    set_lvid(lvid);

    positivize() ;

    setAABB_TreeFrame_All();

    uncoincide();

    setAABB_LeafFrame_All();
}

struct sn_compare
{
    std::function<double(const sn* p)>& fn ;
    bool ascending ;

    sn_compare( std::function<double(const sn* p)>& fn_ , bool ascending_ )
        :
        fn(fn_),
        ascending(ascending_)
    {}
    bool operator()(const sn* a, const sn* b)
    {
        bool cmp = fn(a) < fn(b) ;
        return ascending ? cmp : !cmp ;
    }
};

/**
sn::OrderPrim
---------------

Sorts the vector of prim by comparison of the
results of the argument function on each node.

**/


template<typename N>
inline void sn::OrderPrim( std::vector<N*>& prim, std::function<double(N* p)> fn, bool ascending  ) // static
{
    sn_compare cmp(fn, ascending) ;
    std::sort( prim.begin(), prim.end(), cmp );
}


inline void sn::Transform_Leaf2Tree( glm::tvec3<double>& xyz,  const sn* leaf, std::ostream* out )  // static
{
    glm::tvec4<double> pos0 ;
    pos0.x = xyz.x ;
    pos0.y = xyz.y ;
    pos0.z = xyz.z ;
    pos0.w = 1. ;

    glm::tmat4x4<double> t(1.) ;
    glm::tmat4x4<double> v(1.) ;
    bool reverse = false ;
    leaf->getNodeTransformProduct(t, v, reverse, nullptr );

    glm::tvec4<double> pos = t * pos0 ;

    xyz.x = pos.x ;
    xyz.y = pos.y ;
    xyz.z = pos.z ;

    if(out) *out
        << "sn::Transform_Leaf2Tree"
        << std::endl
        << " pos0 " << stra<double>::Desc(pos0)
        << std::endl
        << " pos  " << stra<double>::Desc(pos)
        << std::endl
        ;
}


inline void sn::uncoincide()
{
    int uncoincide_dump_lvid = ssys::getenvint("sn__uncoincide_dump_lvid", 107) ;
    bool dump = lvid == std::abs(uncoincide_dump_lvid) ;
    bool enable = true ;

    std::stringstream ss ;
    if(dump) ss
        << "sn::uncoincide"
        << " sn__uncoincide_dump_lvid " << uncoincide_dump_lvid
        << " lvid " << lvid
        << std::endl
        ;

    uncoincide_( enable, dump ? &ss : nullptr );

    if( dump )
    {
        std::string str = ss.str() ;
        std::cout << str << std::endl ;
    }
}

/**
sn::uncoincide_
-------------------

Many box box coincidences are currently not counted from can_znudge:false

**/

inline void sn::uncoincide_(bool enable, std::ostream* out)
{
    std::vector<const sn*> prim ;
    collect_prim(prim);
    int num_prim = prim.size() ;

    bool ascending = true ;
    OrderPrim<const sn>(prim, AABB_ZAvg, ascending ) ;

    if(out) *out
        << "sn::uncoincide_"
        << " lvid " << lvid
        << " num_prim " << num_prim
        << std::endl
        ;

    for(int i=1 ; i < num_prim ; i++)
    {
        const sn* lower = prim[i-1] ;
        const sn* upper = prim[i] ;
        sn* _lower = const_cast<sn*>( lower );
        sn* _upper = const_cast<sn*>( upper );
        uncoincide_zminmax( i, _lower, _upper, enable, out );
    }

    if(out) *out
        << "sn::uncoincide_"
        << " lvid " << lvid
        << " num_prim " << num_prim
        << " coincide " << coincide
        << std::endl
        ;
}

/**
sn::uncoincide_zminmax
------------------------

TODO : should be using a transformed rperp as this needs to
operate in the frame of the root of the CSG tree

**/

inline void sn::uncoincide_zminmax( int i, sn* lower, sn* upper, bool enable, std::ostream* out )
{
    bool can_znudge = lower->can_znudge() && upper->can_znudge() ;
    bool same_union = AreFromSameUnion(lower, upper) ;
    double lower_zmax = lower->getBB_zmax() ;
    double upper_zmin = upper->getBB_zmin() ;
    bool z_minmax_coincide =  std::abs( lower_zmax - upper_zmin ) < Z_EPSILON ;
    bool fixable_coincide = z_minmax_coincide && can_znudge && same_union ;

    if(out && z_minmax_coincide) *out
        << "sn::uncoincide_zminmax"
        << " lvid " << lvid
        << " ("<< i-1 << "," << i << ") "
        << " lower_zmax " << lower_zmax
        << " upper_zmin " << upper_zmin
        << " lower_tag " << lower->tag()
        << " upper_tag " << upper->tag()
        << " can_znudge " << ( can_znudge ? "YES" : "NO " )
        << " same_union " << ( same_union ? "YES" : "NO " )
        << " z_minmax_coincide " << ( z_minmax_coincide ? "YES" : "NO " )
        << " fixable_coincide " << ( fixable_coincide ? "YES" : "NO " )
        << " enable " << ( enable ? "YES" : "NO " )
        << std::endl
        ;

    if(!fixable_coincide) return ;

    coincide += 1 ;

    double upper_rperp_at_zmin = upper->rperp_at_zmin() ;
    double lower_rperp_at_zmax = lower->rperp_at_zmax() ;

    // NB these positions must use coordinates/param in corresponding upper/lower leaf frame
    glm::tvec3<double> upper_pos( upper->rperp_at_zmin(), upper->rperp_at_zmin(), upper->zmin() );
    glm::tvec3<double> lower_pos( lower->rperp_at_zmax(), lower->rperp_at_zmax(), lower->zmax() );

    // transforming the leaf frame coordinates into tree frame
    Transform_Leaf2Tree( upper_pos, upper, nullptr );
    Transform_Leaf2Tree( lower_pos, lower, nullptr );

    bool upper_lower_pos_z_equal = std::abs( upper_pos.z - lower_pos.z) < Z_EPSILON ;
    bool rperp_equal = std::abs( lower_pos.x - upper_pos.x ) < Z_EPSILON  ;
    bool upper_rperp_smaller = lower_pos.x > upper_pos.x ;
    bool upper_shape_smaller = upper->typecode == CSG_ZSPHERE && lower->typecode == CSG_CYLINDER ;
    bool upper_decrease_zmin = rperp_equal ?  upper_shape_smaller : upper_rperp_smaller ;

    double dz = 1. ;
    if( upper_decrease_zmin )  upper->decrease_zmin( dz );
    else                       lower->increase_zmax( dz );


    if(out) *out
        << "sn::uncoincide_zminmax"
        << " lvid " << lvid
        << " ("<< i-1 << "," << i << ") "
        << " lower_rperp_at_zmax " << lower_rperp_at_zmax
        << " upper_rperp_at_zmin " << upper_rperp_at_zmin
        << " (leaf frame) "
        << std::endl
        << " upper_pos " << stra<double>::Desc(upper_pos) << " (tree frame) "
        << std::endl
        << " lower_pos " << stra<double>::Desc(lower_pos) << " (tree frame) "
        << std::endl
        << " rperp_equal " << ( rperp_equal ? "YES" : "NO " )
        << " upper_rperp_smaller " << ( upper_rperp_smaller ? "YES" : "NO " )
        << " upper_shape_smaller " << ( upper_shape_smaller ? "YES" : "NO " )
        << " upper_decrease_zmin " << ( upper_decrease_zmin ? "YES" : "NO " )
        << " upper_lower_pos_z_equal " << ( upper_lower_pos_z_equal ? "YES" : "NO " )
        << std::endl
        << ( upper_decrease_zmin  ?
             "  upper->decrease_zmin( dz ) : expand upper down into bigger lower "
             :
             "  lower->increase_zmax( dz ) : expand lower up into bigger upper "
           )
        << " coincide " << coincide
        << std::endl
        ;

   assert( upper_lower_pos_z_equal && "EXPECTED upper_lower_pos_z_equal FOR COINCIDENT " );

}



