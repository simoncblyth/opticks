#pragma once
/**
snd.hh : constituent CSG node in preparation
============================================= 

snd.h intended as minimal first step, holding parameters of 
G4VSolid CSG trees for subsequent use by CSGNode::Make
and providing dependency fire break between G4 and CSG 

* snd.hh instances are one-to-one related to CSG/CSGNode.h

* initially thought snd.hh would be transient with no persisting and no role on GPU. 
  But that is inconsistent with the rest of stree.h and also want to experiment 
  with non-binary intersection in future, so are using snd.hh to test non-binary 
  solid persisting following the same approach as snode.h structural nodes

Usage requires the scsg.hh POOL. That is now done at stree instanciation::

    snd::SetPOOL(new scsg); 

TODO: 

* add polycone ZNudge and tests ?
* how about convexpolyhedron with planes ? just add spl ? 
* how about multiunion ?

**/

#include <vector>
#include <string>
#include <functional>

#include "glm/fwd.hpp"
#include "OpticksCSG.h"
#include "SYSRAP_API_EXPORT.hh"

struct spa ; 
struct sbb ; 
struct sxf ; 
struct scsg ; 
struct NPFold ; 
struct scanvas ; 

struct SYSRAP_API snd
{
    int index ; 
    int depth ;   // optionally set by calling SetLVID 
    int sibdex ;  // 0-based sibling index 
    int parent ; 

    int num_child ; 
    int first_child ; 
    int next_sibling ; 
    int lvid ;

    int typecode ; 
    int param ; 
    int aabb ; 
    int xform ; 

    char label[16] ;   // sizeof 4 int 





    static constexpr const int VERSION = 0 ;
    static constexpr const char* NAME = "snd" ; 
    static constexpr const double zero = 0. ; 
    static scsg* POOL ; 
    static void SetPOOL( scsg* pool ); 
    static int  Level(); 

    static NPFold* Serialize(); 
    static void    Import(const NPFold* fold); 

    static std::string Desc();
    static std::string Brief(int idx);
    static std::string Brief( const std::vector<int>& nodes);
    static std::string Brief_(const std::vector<snd>& nodes);

    static const snd* Get( int idx);
    static       snd* Get_(int idx);

    static int  GetMaxDepth(int idx) ; 
    static int  GetNumNode(int idx) ; 
    static void GetTypes(std::vector<int>& types, const std::vector<int>& idxs ); 

    static int  GetNodeXForm(int idx) ; 
    static void SetNodeXForm(int idx, const glm::tmat4x4<double>& tr );

    void setXF( const glm::tmat4x4<double>& t ); 
    void setXF( const glm::tmat4x4<double>& t, const glm::tmat4x4<double>& v ) ; 

    static const sxf* GetXF(int idx) ; 
    static       sxf* GetXF_(int idx) ; 

    const sxf* getXF() const ; 
    sxf*       getXF_(); 

    static void            NodeTransformProduct(int nidx, glm::tmat4x4<double>& t, glm::tmat4x4<double>& v, bool reverse ) ; 
    static std::string DescNodeTransformProduct(int nidx, glm::tmat4x4<double>& t, glm::tmat4x4<double>& v, bool reverse ) ; 
    void node_transform_product(                          glm::tmat4x4<double>& t, glm::tmat4x4<double>& v, bool reverse ) const ; 



    static void SetLabel(    int idx , const char* label ); 
    static void SetLVID(int idx, int lvid);  // label node tree 

    static void GetLVID( std::vector<snd>& nds, int lvid ); 
    static const snd* GetLVRoot( int lvid );
 
    static int GetLVNumNode( int lvid ); // total nodes 
    static int GetLVBinNode( int lvid ); // binary tree nodes
    static int GetLVSubNode( int lvid ); // compound constituent nodes

    int getLVNumNode() const ;  // total nodes 
    int getLVBinNode() const ;  // binary tree nodes (compound constituents not included) 
    int getLVSubNode() const ;  // sub nodes : ie the compound constituents  


    static void GetLVNodesComplete(std::vector<const snd*>& nds, int lvid); // unbalanced deep trees will have many nullptr 
    void        getLVNodesComplete(std::vector<const snd*>& nds) const ; 
    static void GetLVNodesComplete_r(std::vector<const snd*>& nds, const snd* nd, int idx); 

    static std::string Desc(int idx);
    static std::string DescParam(int idx);
    static std::string DescXForm(int idx);
    static std::string DescAABB( int idx);

    static int Add(const snd& nd); 


    bool is_listnode() const ; 
    std::string tag() const ; 
    std::string brief() const ; 

    std::string rbrief() const ; 
    void rbrief_r(std::ostream& os, int d) const ; 

    std::string desc() const ; 


    static const char* ERROR_NO_POOL_NOTES ; 
    static void CheckPOOL(const char* msg); 

    void setParam( double x,  double y,  double z,  double w,  double z1, double z2 ); 
    void setAABB(  double x0, double y0, double z0, double x1, double y1, double z1 );
    void setLabel( const char* l ); 

    void setLVID(int lvid_ ); 
    void setLVID_r(int lvid_, int d ); 

    int checktree() const ; 
    int checktree_r(char code,  int d ) const ; 

    static void Visit(int idx) ; 

    static void PreorderTraverse(int idx, std::function<void(int)> fn ); 
    void preorder_traverse(   std::function<void(int)> fn ) ; 
    void preorder_traverse_r( std::function<void(int)> fn, int d) ; 

    static void PostorderTraverse(int idx, std::function<void(int)> fn); 
    void postorder_traverse(   std::function<void(int)> fn ) ; 
    void postorder_traverse_r( std::function<void(int)> fn, int d) ; 

    int max_depth() const ; 
    int max_depth_r(int d) const ; 

    int max_binary_depth() const ;   // listnodes not recursed, listnodes regarded as leaf node primitives 
    int max_binary_depth_r(int d) const ; 

    int num_node() const ; 
    int num_node_r(int d) const ; 

    bool is_root() const ; 
    bool is_leaf() const ; 
    bool is_binary_leaf() const ;   // listnodes are regarded as binary leaves
    bool is_sibdex(int q_sibdex) const ; 


    static void Inorder(std::vector<int>& order, int idx ); 
    void inorder(std::vector<int>& nodes ) const ;    // collects absolute snd::index from inorder traversal 
    void inorder_r(std::vector<int>& order, int d ) const ; 

    static void Ancestors(int idx, std::vector<int>& nodes); 
    void ancestors(std::vector<int>& nodes) const; 


    void leafnodes( std::vector<int>& nodes ) const ; 
    void leafnodes_r( std::vector<int>& nodes, int d  ) const ; 

    static int Find(int idx, char l0); 
    int find(char l0) const ; 
    void find_r(std::vector<int>& nodes, char l0, int d) const ; 

    
    template<typename ... Args> 
    void typenodes_(  std::vector<int>& nodes, Args ... tcs ) const ; 
    void typenodes_r_(std::vector<int>& nodes, const std::vector<OpticksCSG_t>& types, int d) const ; 
    bool has_type(const std::vector<OpticksCSG_t>& types) const  ; 

    template<typename ... Args> 
    static std::string DescType(Args ... tcs); 

    void typenodes(std::vector<int>& nodes, int tc ) const ; 
    void typenodes_r(std::vector<int>& nodes, int tc, int d) const ; 

    int get_ordinal( const std::vector<int>& order ) const ; 


    std::string dump() const ; 
    void dump_r( std::ostream& os, int d ) const ; 

    std::string dump2() const ; 
    void dump2_r( std::ostream& os, int d ) const ; 


    static std::string Render(int idx, int mode=-1 ) ; 
    std::string render(int mode=-1) const ; 
    void render_r( scanvas* canvas, const std::vector<int>& order, int mode, int d) const ; 
    void render_v( scanvas* canvas, const std::vector<int>& order, int mode, int d) const ; 


    void check_z() const ; 
    double zmin() const ; 
    double zmax() const ; 
    void decrease_zmin( double dz ); 
    void increase_zmax( double dz ); 
    static std::string ZDesc(const std::vector<int>& prims);
    static void ZNudgeEnds(    const std::vector<int>& prims); 
    static void ZNudgeJoints(  const std::vector<int>& prims); 


    static snd Init(int tc); 
    void init(); 

    static int Boolean( int op, int l, int r ); 
    static int Compound(int type, const std::vector<int>& prims ); 

    private:
    static int UnionTree( const std::vector<int>& prims ); 
    static int Contiguous(const std::vector<int>& prims ); 
    public:
    static int Collection(const std::vector<int>& prims );   // UnionTree OR Contiguous depending on VERSION

    // signatures need to match CSGNode where these param will end up 
    static int Cylinder(double radius, double z1, double z2) ;
    static int Cone(double r1, double z1, double r2, double z2); 
    static int Sphere(double radius); 
    static int ZSphere(double radius, double z1, double z2); 
    static int Box3(double fullside); 
    static int Box3(double fx, double fy, double fz ); 
    static int Zero(double  x,  double y,  double z,  double w,  double z1, double z2); 
    static int Zero(); 
};


