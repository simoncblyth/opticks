#pragma once

class G4VSolid ; 
class G4Ellipsoid ; 
class G4Tubs ; 
class G4Polycone ; 
class G4Torus ; 
class G4DisplacedSolid ; 
class G4Box ; 

#include "G4RotationMatrix.hh" 
#include "G4ThreeVector.hh" 

#include <string>
#include <map>
#include <vector>


/**
X4SolidTree : CSG tree manipulations  
======================================

X4SolidTree was developed initially as j/PMTSim/ZSolid 

Even though X4SolidTree.h is a "private" header it still needs to be 
used across compilation units (eg for tests) hence assume that 
the API_EXPORT is needed 
**/

struct SCanvas ; 

#ifdef PMTSIM_STANDALONE
#include "PMTSIM_API_EXPORT.hh"
struct PMTSIM_API X4SolidTree   
{
#else
#include "X4_API_EXPORT.hh"
struct X4_API X4SolidTree   
{
#endif
    // primary API
    static const bool verbose ; 
    static G4VSolid* ApplyZCutTree( const G4VSolid* original, double zcut ); 
    static void Draw(const G4VSolid* original, const char* msg="X4SolidTree::Draw" ); 

    // members
    const G4VSolid* original ; 
    G4VSolid*       root ;     // DeepClone of original, which is identical to original AND fully independent from it 
    bool            edited ;   // false, until root changed  or tree pruned
  

    // maps populated by instrumentTree
    std::map<const G4VSolid*, const G4VSolid*>* parent_map ; 
    std::map<const G4VSolid*, int>*             in_map ; 
    std::map<const G4VSolid*, int>*             rin_map ; 
    std::map<const G4VSolid*, int>*             pre_map ; 
    std::map<const G4VSolid*, int>*             rpre_map ; 
    std::map<const G4VSolid*, int>*             post_map ; 
    std::map<const G4VSolid*, int>*             rpost_map ; 
    std::map<const G4VSolid*, int>*             zcls_map ; 
    std::map<const G4VSolid*, int>*             depth_map ; 
    std::map<const G4VSolid*, char>*            mkr_map ; 

    unsigned width ; 
    unsigned height ; 
    unsigned extra_width ; 
    unsigned extra_height ; 
    SCanvas* canvas ; 
    std::vector<std::string>* names ; 
    const char* nameprefix ; 

    std::vector<const G4VSolid*>* inorder ; 
    std::vector<const G4VSolid*>* rinorder ; 
    std::vector<const G4VSolid*>* preorder ; 
    std::vector<const G4VSolid*>* rpreorder ; 
    std::vector<const G4VSolid*>* postorder ; 
    std::vector<const G4VSolid*>* rpostorder ; 

    std::vector<G4VSolid*>* crux ; 

    // object methods
    X4SolidTree(const G4VSolid* root ); 


    void init(); 

    void instrumentTree();
    void parent_r(    const G4VSolid* node, int depth); 
    void depth_r(     const G4VSolid* node, int depth);
    void inorder_r(   const G4VSolid* node, int depth);
    void rinorder_r(  const G4VSolid* node, int depth);
    void preorder_r(  const G4VSolid* node, int depth);
    void rpreorder_r( const G4VSolid* node, int depth);
    void postorder_r( const G4VSolid* node, int depth);
    void rpostorder_r(const G4VSolid* node, int depth);

    const G4VSolid* parent( const G4VSolid* node_) const ;
    G4VSolid*       parent_( const G4VSolid* node_) const ;

    int depth( const G4VSolid* node_) const ;
    int in(    const G4VSolid* node_) const ;
    int rin(   const G4VSolid* node_) const ;
    int pre(   const G4VSolid* node_) const ;
    int rpre(  const G4VSolid* node_) const ;
    int post(  const G4VSolid* node_) const ;
    int rpost( const G4VSolid* node_) const ;
    int index( const G4VSolid* n, int mode ) const ; 

    enum { IN, RIN, PRE, RPRE, POST, RPOST } ; 

    static const char* IN_ ; 
    static const char* RIN_ ;
    static const char* PRE_ ; 
    static const char* RPRE_ ;
    static const char* POST_ ;
    static const char* RPOST_ ;
    static const char* OrderName(int mode);



    double getX(const G4VSolid* node ) const ; 
    double getY(const G4VSolid* node ) const ; 
    double getZ(const G4VSolid* node ) const ; 

    void   getTreeTransform( G4RotationMatrix* rot, G4ThreeVector* tla, const G4VSolid* node ) const ; 

    int classifyTree( double zcut ); 
    int classifyTree_r( G4VSolid* node_, int depth, double zcut ); 

    int classifyMask(const G4VSolid* top) const ;
    int classifyMask_r( const G4VSolid* node_, int depth ) const ;

    void apply_cut(double zcut);
    void cutTree_r( const G4VSolid* node_, int depth, double zcut ); 
    void collectNames_inorder_r( const G4VSolid* n, int depth ); 


    int zcls( const G4VSolid* node_ ) const ;   // formerly had move bool arg, now using move:false 
    void set_zcls( const G4VSolid* node_, int zc ); 

    char mkr( const G4VSolid* node_) const ;
    void set_mkr( const G4VSolid* node_, char mk ); 

    bool is_include( const G4VSolid* node_ ) const ; 
    bool is_exclude( const G4VSolid* node_ ) const ; 
    bool is_exclude_include( const G4VSolid* node_ ) const ;
    bool is_include_exclude( const G4VSolid* node_ ) const ;
    bool is_crux( const G4VSolid* node_ ) const ;

    int num_prim() const ;
    int num_prim_r(const G4VSolid* n) const;
    int num_node() const; 
    static int NumNode_r(const G4VSolid* n, int depth );

    int num_node_select(int qcls) const; 
    int num_node_select_r(const G4VSolid* n, int qcls) const ;

    const char* desc() const ; 

    void prune(bool act, int pass); 
    void prune_crux(G4VSolid* x, bool act, int pass);


    void collectNodes( std::vector<const G4VSolid*>& nodes, const G4VSolid* top, int query_zcls  );
    void collectNodes_r( std::vector<const G4VSolid*>& nodes, const G4VSolid* node_, int query_zcls, int depth  );

    void draw(const char* msg="X4SolidTree::draw", int pass=-1); 
    void draw_r( const G4VSolid* n, int mode); 

    void dumpNames(const char* msg="X4SolidTree::dumpNames") const  ; 


    void zdump(const char* msg="X4SolidTree::zdump") const ; 
    void zdump_r( const G4VSolid* node_, int mode ) const ; 



    int maxdepth() const  ;
    static int Maxdepth_r( const G4VSolid* node_, int depth); 

    void dump(const char* msg="X4SolidTree::dump") const ; 

    void dumpUp(const char* msg="X4SolidTree::dumpUp") const ; 
    void dumpUp_r(const G4VSolid* node, int depth) const ; 

    void dumpTree(const char* msg="X4SolidTree::dumpTree" ) const ; 
    void dumpTree_r( const G4VSolid* node, int depth ) const ; 

    // enumerations
    enum { 
        _G4Other, 
        _G4Ellipsoid, 
        _G4Tubs, 
        _G4Polycone, 
        _G4Torus, 
        _G4Box,
        _G4Orb,
        _G4MultiUnion,
        _G4Sphere,
        _G4UnionSolid, 
        _G4SubtractionSolid, 
        _G4IntersectionSolid, 
        _G4DisplacedSolid 
     }; 

    static const char* G4Ellipsoid_ ; 
    static const char* G4Tubs_      ;
    static const char* G4Polycone_  ;
    static const char* G4Torus_     ;
    static const char* G4Box_     ;
    static const char* G4Orb_     ;
    static const char* G4MultiUnion_     ;
    static const char* G4Sphere_     ;
    static const char* G4UnionSolid_        ;
    static const char* G4SubtractionSolid_  ;
    static const char* G4IntersectionSolid_ ;
    static const char* G4DisplacedSolid_    ;

    static const char* DirtyEntityTag_( const G4VSolid* node );
    static const char* EntityTag_( const G4VSolid* solid );

    enum { 
       UNDEFINED=0, 
         INCLUDE=1<<0, 
        STRADDLE=1<<1, 
         EXCLUDE=1<<2, 
         MIXED=INCLUDE|EXCLUDE
     }; 

    static const char* UNDEFINED_ ; 
    static const char* INCLUDE_; 
    static const char* STRADDLE_ ; 
    static const char* EXCLUDE_; 
    static const char* ClassifyName( int zcls ); 
    static const char* ClassifyMaskName( int zcls ); 

    // simple static convenience functions
    static int    ClassifyZCut( double az0, double az1, double zcut ); 

    // basic solid functions
    static int             EntityType(      const G4VSolid* solid) ; 
    static const char*     EntityTypeName(  const G4VSolid* solid) ; 
    static const char*     EntityTag(       const G4VSolid* solid, bool move ) ; // move:true sees thru G4DisplacedSolid 
    static bool            Boolean(         const G4VSolid* solid) ;
    static bool            Displaced(       const G4VSolid* solid) ;
    static std::string     Desc(const G4VSolid* solid); 

    // navigation
    static const G4VSolid* Left(  const G4VSolid* node) ;
    static const G4VSolid* Right( const G4VSolid* node) ;  // NB the G4VSolid returned might be a G4DisplacedSolid wrapping the G4VSolid 
    static const G4VSolid* Moved( const G4VSolid* node) ;
    static const G4VSolid* Moved( G4RotationMatrix* rot, G4ThreeVector* tla, const G4VSolid* node) ;

    static       G4VSolid* Left_(       G4VSolid* node) ;
    static       G4VSolid* Right_(      G4VSolid* node) ;
    static       G4VSolid* Moved_(      G4VSolid* node) ;
    static       G4VSolid* Moved_( G4RotationMatrix* rot, G4ThreeVector* tla, G4VSolid* node) ;

    // local node frame access 
    static void XRange( double& x0, double& x1,  const G4VSolid* solid ) ; 
    static void YRange( double& y0, double& y1,  const G4VSolid* solid ) ; 
    static void ZRange( double& z0, double& z1,  const G4VSolid* solid ) ; 

    static bool CanX(   const G4VSolid* solid ); 
    static bool CanY(   const G4VSolid* solid ); 
    static bool CanZ(   const G4VSolid* solid ); 

    static void GetXRange( const G4Box*       const box      , double& x0, double& x1 );
    static void GetYRange( const G4Box*       const box      , double& y0, double& y1 );
    static void GetZRange( const G4Box*       const box      , double& z0, double& z1 );

    static void GetZRange( const G4Ellipsoid* const ellipsoid, double& z0, double& z1 );
    static void GetZRange( const G4Tubs*      const tubs     , double& z0, double& z1 ); 
    static void GetZRange( const G4Polycone*  const polycone , double& z0, double& z1 ); 
    static void GetZRange( const G4Torus*     const torus,     double& z0, double& z1 ); 

    // tree cloning methods
    static G4VSolid* DeepClone(    const G4VSolid* solid ); 
    static G4VSolid* DeepClone_r(  const G4VSolid* solid, int depth, G4RotationMatrix* rot, G4ThreeVector* tla ); 
    static G4VSolid* BooleanClone( const G4VSolid* solid, int depth, G4RotationMatrix* rot, G4ThreeVector* tla ); 


    static void GetBooleanBytes(char** bytes, int& num_bytes, const G4VSolid* solid );
    static int CompareBytes( char* bytes0, char* bytes1, int num_bytes ); 
    static void PlacementNewDupe( G4VSolid* solid); 
    static void SetLeft(  G4VSolid* node, G4VSolid* left);  
    static void SetRight( G4VSolid* node, G4VSolid* right, G4RotationMatrix* rrot=nullptr, G4ThreeVector* rtla=nullptr ); 


    static void      CheckBooleanClone( const G4VSolid* clone, const G4VSolid* left, const G4VSolid* right ); 
    static G4VSolid* PrimitiveClone( const  G4VSolid* solid ); 
    static G4VSolid* PrimitiveClone( const  G4VSolid* solid, const char* name ); 


    static G4VSolid* PromoteTubsToPolycone( const G4VSolid* solid ); 
    static const bool PROMOTE_TUBS_TO_POLYCONE ; 

    static void ApplyZCut(             G4VSolid* node, double local_zcut); 
    static void ApplyZCut_G4Ellipsoid( G4VSolid* node, double local_zcut);
    static void ApplyZCut_G4Tubs(      G4VSolid* node, double local_zcut);
    static void ApplyZCut_G4Polycone(  G4VSolid* node, double local_zcut);
    static void ApplyZCut_G4Polycone_NotWorking(  G4VSolid* node, double local_zcut);

    static const char* CommonPrefix(const std::vector<std::string>* a); 

}; 

