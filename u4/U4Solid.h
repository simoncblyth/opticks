#pragma once
/**
U4Solid.h : Convert G4VSolid CSG trees into sn.h trees
==========================================================

Canonical usage from U4Tree::initSolid


Priors:

x4/X4Solid
   G4VSolid->NNode 

u4/U4SolidTree
   some overlap with X4Entity, tree mechanics might be better that X4Solid 

npy/NNode
   organic mess that need to leave behind 

npy/NNodeUncoincide npy/NNodeNudger
   some of this rather complex code will 
   need to be moved to an "sndUncoincide"

**/


#include <set>
#include <csignal>

#include "ssys.h"
#include "scuda.h"

#include "sn.h"
#include "stran.h"
#include "stra.h"
#include "OpticksCSG.h"

#include "G4VSolid.hh"

#include "G4Orb.hh"
#include "G4Sphere.hh"
#include "G4Ellipsoid.hh"
#include "G4Box.hh"
#include "G4Tubs.hh"
#include "G4Polycone.hh"
#include "G4Cons.hh"
#include "G4Hype.hh"
#include "G4MultiUnion.hh"
#include "G4Torus.hh"
#include "G4UnionSolid.hh"
#include "G4IntersectionSolid.hh"
#include "G4SubtractionSolid.hh"

#include "G4BooleanSolid.hh"
#include "G4RotationMatrix.hh"
#include <CLHEP/Units/SystemOfUnits.h> 


#include "U4Polycone.h"
#include "U4Transform.h"


enum { 
    _G4Other, 
    _G4Orb,
    _G4Sphere,
    _G4Ellipsoid, 
    _G4Box,
    _G4Tubs, 
    _G4Polycone, 
    _G4Cons,
    _G4Hype,
    _G4MultiUnion,
    _G4Torus, 
    _G4UnionSolid, 
    _G4IntersectionSolid,
    _G4SubtractionSolid,
    _G4DisplacedSolid
 }; 

struct U4Solid
{
    static constexpr const char* G4Orb_               = "Orb" ;
    static constexpr const char* G4Sphere_            = "Sph" ;
    static constexpr const char* G4Ellipsoid_         = "Ell" ; 
    static constexpr const char* G4Box_               = "Box" ;
    static constexpr const char* G4Tubs_              = "Tub" ;
    static constexpr const char* G4Polycone_          = "Pol" ;
    static constexpr const char* G4Cons_              = "Con" ;
    static constexpr const char* G4Hype_              = "Hyp" ;
    static constexpr const char* G4MultiUnion_        = "Mul" ;
    static constexpr const char* G4Torus_             = "Tor" ;
    static constexpr const char* G4UnionSolid_        = "Uni" ;
    static constexpr const char* G4IntersectionSolid_ = "Int" ;
    static constexpr const char* G4SubtractionSolid_  = "Sub" ;
    static constexpr const char* G4DisplacedSolid_    = "Dis" ;

    static constexpr const char* _U4Solid__IsFlaggedLVID = "U4Solid__IsFlaggedLVID" ; 
    static const int   IsFlaggedLVID_ ; 
    static const char* IsFlaggedName_ ; 
    static const char* IsFlaggedType_ ; 

    static bool IsFlaggedLVID(int q_lvid); 
    static bool IsFlaggedName(const char* name); 
    static bool IsFlaggedType(const char* type); 

    static const char*  brief_KEY ; 
    std::string         brief() const ; 
    std::string         desc() const ; 
    static const char*  Name( const G4VSolid* solid) ; 
    static unsigned     Hint( const char* name ); 
    static const char*  EType(const G4VSolid* solid) ; 
    static int          Level(int level_, const char* name, const char* entityType ); 

    static int          Type(const char* entityType ) ;  
    static const char*  Tag( int type ) ;   
    const char* tag() const ; 


    static sn* Convert(const G4VSolid* solid, int lvid, int depth, int level=-1 ) ; 

private:
    U4Solid( const G4VSolid* solid, int lvid, int depth, int level  ); 

    void init(); 
    void init_Constituents(); 
    void init_Tree(); 
    void init_Tree_Shrink(); 

    void init_Orb(); 
    void init_Sphere(); 
    void init_Ellipsoid(); 
    void init_Box(); 
    void init_Tubs(); 
    void init_Polycone(); 
    void init_Cons(); 
    void init_Hype(); 
    void init_MultiUnion(); 
    void init_Torus(); 
    void init_UnionSolid(); 
    void init_IntersectionSolid(); 
    void init_SubtractionSolid(); 

    sn* init_Sphere_(char layer); 
    sn* init_Cons_(char layer); 


    static OpticksCSG_t BooleanOperator(const G4BooleanSolid* solid ); 
    static bool  IsBoolean(   const G4VSolid* solid) ;
    static bool  IsDisplaced( const G4VSolid* solid) ;

    // TODO: clean up these statics, most not needed 
    static const G4VSolid* Left(  const G4VSolid* node) ;
    static const G4VSolid* Right( const G4VSolid* node) ;  // NB the G4VSolid returned might be a G4DisplacedSolid wrapping the G4VSolid 
    static       G4VSolid* Left_(       G4VSolid* node) ;
    static       G4VSolid* Right_(      G4VSolid* node) ;

    static const G4VSolid* Moved(  G4RotationMatrix* rot, G4ThreeVector* tla, const G4VSolid* node) ;
    static       G4VSolid* Moved_( G4RotationMatrix* rot, G4ThreeVector* tla,       G4VSolid* node) ;
    static const G4VSolid* Moved( const G4VSolid* node) ;
    static       G4VSolid* Moved_(      G4VSolid* node) ;

    void init_BooleanSolid(); 
    void init_DisplacedSolid(); 

    // members
    const G4VSolid* solid ; 
    const char*     name ; 
    unsigned        hint ;   
    const char*     entityType ; 
    int             level ; 
    int             type ;

    int             lvid ; 
    int             depth ;   // recursion depth across different G4VSolid

    sn*             root ;  

};

inline const int U4Solid::IsFlaggedLVID_ = ssys::getenvint(_U4Solid__IsFlaggedLVID, -1) ; 
inline bool U4Solid::IsFlaggedLVID(int q_lvid)
{
    return IsFlaggedLVID_ == q_lvid ;  
}



inline const char* U4Solid::IsFlaggedName_ = ssys::getenvvar("U4Solid__IsFlaggedName", "PLACEHOLDER") ; 
inline bool U4Solid::IsFlaggedName(const char* name)
{
    return name && IsFlaggedName_ && strstr(name, IsFlaggedName_ ) ;  
}

inline const char* U4Solid::IsFlaggedType_ = ssys::getenvvar("U4Solid__IsFlaggedType", "G4Dummy") ; 
inline bool U4Solid::IsFlaggedType(const char* type)
{
    return type && IsFlaggedType_ && strstr(type, IsFlaggedType_ ) ;  
}

inline const char* U4Solid::brief_KEY = "lvid/depth/tag/root/level" ; 

inline std::string U4Solid::brief() const 
{
    std::stringstream ss ; 
    ss << std::setw(3) << lvid
       << "/" << depth 
       << "/" << tag()
       << "/"  << std::setw(3) << ( root ? root->index() : -1 )
       << "/" << std::setw(1) << level 
       ;
    std::string str = ss.str(); 
    return str ; 
}

inline std::string U4Solid::desc() const 
{
    std::stringstream ss ; 
    ss 
       << brief()
       << " level " << level
       << " entityType " << std::setw(16) << ( entityType ? entityType : "-" )
       << " name " << ( name ? name : "-" )
       ; 
    std::string str = ss.str(); 
    return str ; 
}

inline const char* U4Solid::Name(const G4VSolid* solid)  // static
{
    G4String _name = solid->GetName() ; // bizarre: G4VSolid::GetName returns by value, not reference
    const char* name = _name.c_str();    
    return strdup(name) ; 
}

inline unsigned U4Solid::Hint(const char* name)  // static
{
    return sn::NameHint(name); 
}


inline const char* U4Solid::EType(const G4VSolid* solid)  // static
{
    G4GeometryType _etype = solid->GetEntityType();  // G4GeometryType typedef for G4String
    return strdup(_etype.c_str()) ; 
}


/**
U4Solid::Level
----------------

IsFlaggedName 
   returns true for solid name strings that are present within the U4Solid__IsFlaggedName envvar value

IsFlaggedType 
   returns true for solid entityType strings that are present within the U4Solid__IsFlaggedType envvar value
   for example::

        export U4Solid__IsFlaggedType=G4MultiUnion



**/


inline int U4Solid::Level(int level_, const char* name, const char* entityType )   // static 
{
    if(level_ > 0 ) return level_ ; 
    if(IsFlaggedName(name) || IsFlaggedType(entityType)) return 1 ; 
    return -1 ; 
}

inline int U4Solid::Type(const char* name)   // static 
{
    int type = _G4Other ; 
    if( strcmp(name, "G4Orb") == 0 )               type = _G4Orb ; 
    if( strcmp(name, "G4Sphere") == 0 )            type = _G4Sphere ; 
    if( strcmp(name, "G4Ellipsoid") == 0 )         type = _G4Ellipsoid ; 
    if( strcmp(name, "G4Box") == 0 )               type = _G4Box ; 
    if( strcmp(name, "G4Tubs") == 0 )              type = _G4Tubs ; 
    if( strcmp(name, "G4Polycone") == 0 )          type = _G4Polycone ; 
    if( strcmp(name, "G4Cons") == 0 )              type = _G4Cons ; 
    if( strcmp(name, "G4Hype") == 0 )              type = _G4Hype ; 
    if( strcmp(name, "G4MultiUnion") == 0 )        type = _G4MultiUnion ; 
    if( strcmp(name, "G4Torus") == 0 )             type = _G4Torus ; 
    if( strcmp(name, "G4UnionSolid") == 0 )        type = _G4UnionSolid ; 
    if( strcmp(name, "G4SubtractionSolid") == 0 )  type = _G4SubtractionSolid ; 
    if( strcmp(name, "G4IntersectionSolid") == 0 ) type = _G4IntersectionSolid ; 
    if( strcmp(name, "G4DisplacedSolid") == 0 )    type = _G4DisplacedSolid ; 
    return type ; 
}

inline const char* U4Solid::Tag(int type)   // static 
{
    const char* tag = nullptr ; 
    switch(type)
    {
        case _G4Orb:               tag = G4Orb_               ; break ; 
        case _G4Sphere:            tag = G4Sphere_            ; break ; 
        case _G4Ellipsoid:         tag = G4Ellipsoid_         ; break ;         
        case _G4Box:               tag = G4Box_               ; break ; 
        case _G4Tubs:              tag = G4Tubs_              ; break ; 
        case _G4Polycone:          tag = G4Polycone_          ; break ; 
        case _G4Cons:              tag = G4Cons_              ; break ; 
        case _G4Hype:              tag = G4Hype_              ; break ; 
        case _G4MultiUnion:        tag = G4MultiUnion_        ; break ; 
        case _G4Torus:             tag = G4Torus_             ; break ; 
        case _G4UnionSolid:        tag = G4UnionSolid_        ; break ; 
        case _G4SubtractionSolid:  tag = G4SubtractionSolid_  ; break ; 
        case _G4IntersectionSolid: tag = G4IntersectionSolid_ ; break ; 
        case _G4DisplacedSolid:    tag = G4DisplacedSolid_    ; break ; 
    }
    return tag ; 
}

inline const char* U4Solid::tag() const { return Tag(type) ; }


/**
U4Solid::Convert
-----------------

Canonically invoked from U4Tree::initSolid

**/


inline sn* U4Solid::Convert(const G4VSolid* solid, int lvid, int depth, int level ) // static
{
    bool flagged_LVID = IsFlaggedLVID(lvid);  
    if(flagged_LVID) std::cout 
        << "U4Solid::Convert"
        << " solid.GetName[" << solid->GetName() << "]"
        << " lvid " << lvid 
        << " " << _U4Solid__IsFlaggedLVID << " : " << ( flagged_LVID ? "YES" : "NO " ) 
        << " depth " << depth
        << " level " << level 
        << "\n"
        ; 


    U4Solid so(solid, lvid, depth, level ); 
    return so.root ; 
}




inline U4Solid::U4Solid(const G4VSolid* solid_, int lvid_, int depth_, int level_ )
    :
    solid(solid_),
    name(Name(solid_)),
    hint(Hint(name)),
    entityType(EType(solid)),
    level(Level(level_,name,entityType)),
    type(Type(entityType)),
    lvid(lvid_),
    depth(depth_),
    root(nullptr)
{
    init() ; 
}

inline void U4Solid::init()
{
    if(level > 0) std::cerr 
        <<  ( depth == 0 ? "\n" : "" )
        << "[U4Solid::init " << brief() 
        << " " << ( depth == 0 ? brief_KEY : "" )
        << " " << ( depth == 0 ? name : "" )
        << std::endl
        ; 

    init_Constituents(); 
    init_Tree() ; 

    if(level > 0) std::cerr << "]U4Solid::init " << brief() << std::endl ; 
}


/**
U4Solid::init_Constituents
--------------------------

Creates sn nodes that capture the G4VSolid type and parameters.
This is recursive for compound nodes like booleans.

**/

inline void U4Solid::init_Constituents()
{
    switch(type)
    {
        case _G4Orb               : init_Orb()                   ; break ; 
        case _G4Sphere            : init_Sphere()                ; break ; 
        case _G4Ellipsoid         : init_Ellipsoid()             ; break ; 
        case _G4Box               : init_Box()                   ; break ; 
        case _G4Tubs              : init_Tubs()                  ; break ; 
        case _G4Polycone          : init_Polycone()              ; break ; 
        case _G4Cons              : init_Cons()                  ; break ; 
        case _G4Hype              : init_Hype()                  ; break ; 
        case _G4MultiUnion        : init_MultiUnion()            ; break ; 
        case _G4Torus             : init_Torus()                 ; break ; 
        case _G4UnionSolid        : init_UnionSolid()            ; break ; 
        case _G4IntersectionSolid : init_IntersectionSolid()     ; break ; 
        case _G4SubtractionSolid  : init_SubtractionSolid()      ; break ; 
        case _G4DisplacedSolid    : init_DisplacedSolid()        ; break ; 
    } 
    assert( root); 
    root->set_hint_note(hint); 
}



/**
U4Solid::init_Tree
------------------

Only invoked from the top solid after init_Constituents so the 
entire tree of G4VSolid has been recursively traversed 
when the sn::postconvert is called

**/

inline void U4Solid::init_Tree()
{
    if( depth != 0 )  return ; 

    init_Tree_Shrink();  

    root->postconvert(lvid); 
}

/**
U4Solid::init_Tree_Shrink
--------------------------

try to shrink tree if prim are listnode hinted 

**/

inline void U4Solid::init_Tree_Shrink()
{
    if( depth != 0 )  return ; 

    sn* root0 = root ;    

    if(root0->has_candidate_listnode_discontiguous())
    {
        root = sn::CreateSmallerTreeWithListNode_discontiguous(root0);
    }
    else if(root0->has_candidate_listnode_contiguous())
    {
        root = sn::CreateSmallerTreeWithListNode_contiguous(root0);
    }

    if(root != root0) 
    {
        std::cerr << "U4Solid::init_Tree_Shrink CHANGED root with sn::CreateSmallerTreeWithListNode_discontiguous/contiguous\n" ; 
        delete root0 ; 
    }
}

















inline void U4Solid::init_Orb()
{
    const G4Orb* orb = dynamic_cast<const G4Orb*>(solid);
    assert(orb);
    double radius = orb->GetRadius()/CLHEP::mm ;
    root = sn::Sphere(radius) ; 
}

inline void U4Solid::init_Sphere()
{
    sn* outer = init_Sphere_('O') ; assert( outer ) ; 
    sn* inner = init_Sphere_('I');
    root = inner == nullptr ? outer : sn::Boolean( CSG_DIFFERENCE, outer, inner ) ; 
}

/**
U4Solid::init_Sphere_
-----------------------

TODO: bring over phicut handling from X4Solid 

::

      .       +z
               :
            ___:___       __ z = r,   theta = 0 
           /   :   \
          /    :    \
      -x + - - : - - + +x 
          \    :    /
           \___:___/      __ z = -r,  theta = pi  
               :
              -z

**/

inline sn* U4Solid::init_Sphere_(char layer)
{
    assert( layer == 'I' || layer == 'O' ); 
    const G4Sphere* sphere = dynamic_cast<const G4Sphere*>(solid);
    assert(sphere);  

    double rmax = sphere->GetOuterRadius()/CLHEP::mm ; 
    double rmin = sphere->GetInnerRadius()/CLHEP::mm ; 
    double radius = layer == 'I' ? rmin : rmax ;
    if(radius == 0.) return nullptr ; 

    double startThetaAngle = sphere->GetStartThetaAngle()/CLHEP::radian ; 
    double deltaThetaAngle = sphere->GetDeltaThetaAngle()/CLHEP::radian ; 

    double rTheta = startThetaAngle ;
    double lTheta = startThetaAngle + deltaThetaAngle ;
    assert( rTheta >= 0. && rTheta <= CLHEP::pi ) ; 
    assert( lTheta >= 0. && lTheta <= CLHEP::pi ) ; 

    double zmin = radius*std::cos(lTheta) ;
    double zmax = radius*std::cos(rTheta) ;
    assert( zmax > zmin ) ;

    bool z_slice = startThetaAngle > 0. || deltaThetaAngle < CLHEP::pi ; 

    double startPhi = sphere->GetStartPhiAngle()/CLHEP::radian ;
    double deltaPhi = sphere->GetDeltaPhiAngle()/CLHEP::radian ;
    bool has_deltaPhi = startPhi != 0. || deltaPhi != 2.*CLHEP::pi  ;

    bool has_deltaPhi_expect = has_deltaPhi == false ;
    assert( has_deltaPhi_expect ); 
    if(!has_deltaPhi_expect) std::raise(SIGINT); 

    return z_slice ? sn::ZSphere( radius, zmin, zmax ) : sn::Sphere(radius ) ; 
}


/**
U4Solid::init_Ellipsoid
--------------------------

Implemented using a sphere with an associated scale transform.
The sphere radius is picked to be the ZSemiAxis with scale factors being::

    ( XSemiAxis/ZSemiAxis, YSemiAxis/ZSemiAxis, ZSemiAxis/ZSemiAxis )
    ( XSemiAxis/ZSemiAxis, YSemiAxis/ZSemiAxis, 1. )
 
This means that for a PMT bulb like shape flattened in Z the scale factors might be::

     ( 1.346, 1.346, 1.00 )

So the underlying sphere is being expanded equally in x and y directions
with z unscaled.   

**/

inline void U4Solid::init_Ellipsoid()
{
    const G4Ellipsoid* ellipsoid = static_cast<const G4Ellipsoid*>(solid);
    assert(ellipsoid);

    double sx = ellipsoid->GetSemiAxisMax(0)/CLHEP::mm ;
    double sy = ellipsoid->GetSemiAxisMax(1)/CLHEP::mm ;
    double sz = ellipsoid->GetSemiAxisMax(2)/CLHEP::mm ;

    glm::tvec3<double> sc( sx/sz, sy/sz, 1. ) ;   // unity scaling in z, so z-coords are unaffected  
    glm::tmat4x4<double> scale(1.); 
    U4Transform::GetScaleTransform(scale, sc.x, sc.y, sc.z ); 


    double zcut1 = ellipsoid->GetZBottomCut()/CLHEP::mm ;
    double zcut2 = ellipsoid->GetZTopCut()/CLHEP::mm ;

    double zmin = zcut1 > -sz ? zcut1 : -sz ; 
    double zmax = zcut2 <  sz ? zcut2 :  sz ; 
    assert( zmax > zmin ) ;  

    bool upper_cut = zmax <  sz ;
    bool lower_cut = zmin > -sz ;
    bool zslice = lower_cut || upper_cut ;

    if(level > 0) std::cerr
        << "U4Solid::init_Ellipsoid"
        << " upper_cut " << upper_cut
        << " lower_cut " << lower_cut
        << " zcut1 " << zcut1
        << " zcut2 " << zcut2
        << " zmin " << zmin
        << " zmax " << zmax
        << " sx " << sx
        << " sy " << sy
        << " sz " << sz
        << " zslice " << ( zslice ? "YES" : "NO" )
        << std::endl 
        ;


    if( upper_cut == false && lower_cut == false )
    {
        root = sn::Sphere(sz) ;
    }
    else if( upper_cut == true && lower_cut == true )
    {
        root = sn::ZSphere(sz, zmin, zmax) ;
    }
    else if ( upper_cut == false && lower_cut == true )   // PMT mask uses this 
    {
        double zmax_safe = zmax + 0.1 ;
        root = sn::ZSphere( sz, zmin, zmax_safe )  ;

    }
    else if ( upper_cut == true && lower_cut == false )
    {
        double zmin_safe = zmin - 0.1 ; 
        root = sn::ZSphere( sz, zmin_safe, zmax )  ;
    }

    // zmin_safe/zmax_safe use safety offset when there is no cut 
    // this avoids rare apex(nadir) hole bug  
    // see notes/issues/unexpected_zsphere_miss_from_inside_for_rays_that_would_be_expected_to_intersect_close_to_apex.rst
    root->setXF(scale); 

}


inline void U4Solid::init_Hype()
{
    assert(0); 
} 


inline void U4Solid::init_MultiUnion()
{
    const G4MultiUnion* const muni = static_cast<const G4MultiUnion*>(solid);
    assert(muni); 

    int hint = CSG::HintCode(name); 
    int type = ( hint == CSG_DISCONTIGUOUS || hint == CSG_CONTIGUOUS ) ? hint : CSG_CONTIGUOUS ; 

    unsigned sub_num = muni->GetNumberOfSolids() ; 
    if(level > 0 ) std::cout 
        << "[U4Solid::init_MultiUnion" 
        << " level " << level 
        << " name " << name 
        << " hint " << hint
        << " CSG::Name(hint) " << CSG::Name(hint)
        << " type " << type
        << " CSG::Name(type) " << CSG::Name(type)
        << " sub_num " << sub_num
        << "\n"
        ;    

    std::vector<sn*> prims ; 
    for( unsigned i=0 ; i < sub_num ; i++)
    {    
        const G4VSolid* sub = muni->GetSolid(i);
        G4String name = sub->GetName() ; 

        glm::tmat4x4<double> xf(1.) ; 
        U4Transform::GetMultiUnionItemTransform( xf, muni, i ); 

        if(level > 0 ) std::cout 
            << "U4Solid::init_MultiUnion" 
            << name 
            << "\n" 
            << stra<double>::Desc(xf) 
            << "\n" 
            ; 

        sn* p = Convert( sub,  lvid, depth+1, level ); 
        p->combineXF(xf); 

        assert( p && p->is_primitive() );   

        prims.push_back(p) ;  
    }    
    root = sn::Compound( prims, type );  

    if(level > 0 ) std::cout 
        << "]U4Solid::init_MultiUnion" 
        << " level " << level 
        << "\n"
        ;    

 

}



/**
U4Solid::init_Torus
--------------------

Not planning to get analytic GPU intersect working anytime soon, 
just need analytic placeholder whilst use triangulated geom. 

**/

inline void U4Solid::init_Torus()
{
    const G4Torus* const torus = static_cast<const G4Torus*>(solid);
    assert(torus); 

    double rmin_mm = torus->GetRmin()/CLHEP::mm ; 
    double rmax_mm = torus->GetRmax()/CLHEP::mm ;
    double rtor_mm = torus->GetRtor()/CLHEP::mm ;
    double startPhi_deg = torus->GetSPhi()/CLHEP::degree ; 
    double deltaPhi_deg = torus->GetDPhi()/CLHEP::degree ; 

    if(level > 0) std::cout 
       << " U4Solid::init_Torus "
       << " rmin_mm " << rmin_mm 
       << " rmax_mm " << rmax_mm 
       << " rtor_mm " << rtor_mm 
       << " startPhi_deg " << startPhi_deg
       << " deltaPhi_deg " << deltaPhi_deg
       << "\n"
       ;

    assert( rmax_mm > rmin_mm );
    assert( rtor_mm > rmax_mm );

    root = sn::Torus(rmin_mm, rmax_mm, rtor_mm, startPhi_deg, deltaPhi_deg ) ; 
}




inline void U4Solid::init_Box()
{
    const G4Box* box = dynamic_cast<const G4Box*>(solid);
    assert(box);

    double fx = 2.0*box->GetXHalfLength()/CLHEP::mm ;
    double fy = 2.0*box->GetYHalfLength()/CLHEP::mm ;
    double fz = 2.0*box->GetZHalfLength()/CLHEP::mm ;

    root = sn::Box3(fx, fy, fz) ; 
}


inline void U4Solid::init_Tubs()
{
    const G4Tubs* tubs = dynamic_cast<const G4Tubs*>(solid);
    assert(tubs); 

    double hz = tubs->GetZHalfLength()/CLHEP::mm ;  
    double rmax = tubs->GetOuterRadius()/CLHEP::mm ; 
    double rmin = tubs->GetInnerRadius()/CLHEP::mm ; 
    bool has_inner = rmin > 0. ; 

    sn* outer = sn::Cylinder(rmax, -hz, hz );


    if(has_inner == false)
    {
        root = outer ; 
    }
    else
    {
        bool   do_nudge_inner = true ; 
        double nudge_inner = 0.01 ;
        double dz = do_nudge_inner ? hz*nudge_inner : 0. ; 

        sn* inner = sn::Cylinder(rmin, -(hz+dz), hz+dz );  
        root = sn::Boolean( CSG_DIFFERENCE, outer, inner ); 
    }

} 


inline void U4Solid::init_Polycone()
{
    const G4Polycone* polycone = dynamic_cast<const G4Polycone*>(solid);
    assert(polycone);
    root = U4Polycone::Convert(polycone, lvid, depth, level ); 

    if(level > 0 ) std::cerr 
        << "U4Solid::init_Polycone"
        << " level " << level 
        << std::endl
        << desc()
        << std::endl
        ; 
}



inline sn* U4Solid::init_Cons_(char layer)
{
    assert( layer == 'I' || layer == 'O' ); 
    const G4Cons* cone = dynamic_cast<const G4Cons*>(solid);
    assert(cone);  

    double rmax1    = cone->GetOuterRadiusMinusZ()/CLHEP::mm ;
    double rmax2    = cone->GetOuterRadiusPlusZ()/CLHEP::mm  ;
    
    double rmin1    = cone->GetInnerRadiusMinusZ()/CLHEP::mm ;
    double rmin2    = cone->GetInnerRadiusPlusZ()/CLHEP::mm  ;
    
    double hz       = cone->GetZHalfLength()/CLHEP::mm   ;

    //double startPhi = cone->GetStartPhiAngle()/CLHEP::degree ;
    //double deltaPhi = cone->GetDeltaPhiAngle()/CLHEP::degree ;
    
    double r1 = layer == 'I' ? rmin1 : rmax1 ;
    double r2 = layer == 'I' ? rmin2 : rmax2 ;
    double z1 = -hz ; 
    double z2 = hz ;

    bool invalid =  r1 == 0. && r2 == 0. ; 

    return invalid ? nullptr : sn::Cone( r1, z1, r2, z2 ) ;  
} 

inline void U4Solid::init_Cons()
{
    sn* outer = init_Cons_('O');  assert( outer ); 
    sn* inner = init_Cons_('I'); 
    root = inner == nullptr ? outer : sn::Boolean( CSG_DIFFERENCE, outer, inner ) ;

    if(level > 0 ) std::cerr 
        << "U4Solid::init_Cons"
        << " level " << level 
        << desc() 
        ; 

}




inline void U4Solid::init_UnionSolid()
{ 
    init_BooleanSolid(); 
} 
inline void U4Solid::init_IntersectionSolid()
{  
    init_BooleanSolid(); 
} 
inline void U4Solid::init_SubtractionSolid()
{
    init_BooleanSolid(); 
} 

inline OpticksCSG_t U4Solid::BooleanOperator(const G4BooleanSolid* solid ) // static
{
    OpticksCSG_t _operator = CSG_ZERO ;
    if      (dynamic_cast<const G4IntersectionSolid*>(solid)) _operator = CSG_INTERSECTION ;
    else if (dynamic_cast<const G4SubtractionSolid*>(solid))  _operator = CSG_DIFFERENCE ;
    else if (dynamic_cast<const G4UnionSolid*>(solid))        _operator = CSG_UNION ;
    assert( _operator != CSG_ZERO ) ;
    return _operator ;
}


inline bool U4Solid::IsBoolean(const G4VSolid* solid) // static
{
    return dynamic_cast<const G4BooleanSolid*>(solid) != nullptr ; 
}
inline bool U4Solid::IsDisplaced(const G4VSolid* solid) // static
{
    return dynamic_cast<const G4DisplacedSolid*>(solid) != nullptr ; 
}
inline const G4VSolid* U4Solid::Left(const G4VSolid* solid ) // static
{
    return IsBoolean(solid) ? solid->GetConstituentSolid(0) : nullptr ; 
}
inline const G4VSolid* U4Solid::Right(const G4VSolid* solid ) // static
{
    return IsBoolean(solid) ? solid->GetConstituentSolid(1) : nullptr ; 
}
inline G4VSolid* U4Solid::Left_(G4VSolid* solid ) // static
{
    return IsBoolean(solid) ? solid->GetConstituentSolid(0) : nullptr ; 
}
inline G4VSolid* U4Solid::Right_(G4VSolid* solid ) // static
{
    return IsBoolean(solid) ? solid->GetConstituentSolid(1) : nullptr ; 
}

/**
U4Solid::Moved         
---------------            
    
When node isa G4DisplacedSolid sets the rotation and translation and returns the constituentMovedSolid
otherwise returns the input node.

cf X4Transform3D::GetDisplacementTransform does the same

**/ 
inline const G4VSolid* U4Solid::Moved( G4RotationMatrix* rot, G4ThreeVector* tla, const G4VSolid* node )  // static
{   
    const G4DisplacedSolid* disp = dynamic_cast<const G4DisplacedSolid*>(node) ;
    if(disp)     
    {
        if(rot) *rot = disp->GetFrameRotation();
        if(tla) *tla = disp->GetObjectTranslation();  
        // HMM: looks a bit fishy being inconsistent here : until look at g4-cls G4DisplacedSolid
    }
    return disp ? disp->GetConstituentMovedSolid() : node  ;
}   
inline G4VSolid* U4Solid::Moved_( G4RotationMatrix* rot, G4ThreeVector* tla, G4VSolid* node )  // static
{   
    G4DisplacedSolid* disp = dynamic_cast<G4DisplacedSolid*>(node) ; 
    if(disp)
    {
        if(rot) *rot = disp->GetFrameRotation();
        if(tla) *tla = disp->GetObjectTranslation();
        // HMM: looks a bit fishy being inconsistent here : until look at g4-cls G4DisplacedSolid
    }
    return disp ? disp->GetConstituentMovedSolid() : node  ;
}

inline const G4VSolid* U4Solid::Moved( const G4VSolid* node )  // static
{
    const G4DisplacedSolid* disp = dynamic_cast<const G4DisplacedSolid*>(node) ;
    return disp ? disp->GetConstituentMovedSolid() : node  ;
} 

inline G4VSolid* U4Solid::Moved_( G4VSolid* node )  // static
{
    G4DisplacedSolid* disp = dynamic_cast<G4DisplacedSolid*>(node) ;
    return disp ? disp->GetConstituentMovedSolid() : node  ;
}


inline void U4Solid::init_BooleanSolid()
{
    const G4BooleanSolid* boo = static_cast<const G4BooleanSolid*>(solid);
    assert(boo);

    OpticksCSG_t op = BooleanOperator(boo);
    G4VSolid*    left  = const_cast<G4VSolid*>(boo->GetConstituentSolid(0));
    G4VSolid*    right = const_cast<G4VSolid*>(boo->GetConstituentSolid(1));

    bool is_left_displaced = dynamic_cast<G4DisplacedSolid*>(left) != nullptr ;
    bool is_right_displaced = dynamic_cast<G4DisplacedSolid*>(right) != nullptr ;

    bool is_left_displaced_expect = is_left_displaced == false ; 
    assert( is_left_displaced_expect && "not expecting left displacement " );
    if(!is_left_displaced_expect) std::raise(SIGINT); 

    sn* l = Convert( left,  lvid, depth+1, level ); 
    sn* r = Convert( right, lvid, depth+1, level ); 

    if(l->xform && level > 0) std::cout
        << "U4Solid::init_BooleanSolid "
        << " observe transform on left node " 
        << " l.xform " << l->xform->desc()
        << std::endl 
        ; 

    if(is_right_displaced) assert( r->xform && "expecting transform on right displaced node " ); 
 
    root = sn::Boolean( op, l, r ); 
}


/**
U4Solid::init_DisplacedSolid
------------------------------

When booleans are created with transforms the right hand side 
gets wrapped into a DisplacedSolid.:: 

    U4Solid::init SUCCEEDED desc: U4Solid::desc solid Y type Y U4Solid::Tag(type) Box root 57
    U4Solid::init SUCCEEDED desc: U4Solid::desc solid Y type Y U4Solid::Tag(type) Box root 58
    U4Solid::init SUCCEEDED desc: U4Solid::desc solid Y type Y U4Solid::Tag(type) Dis root 58
    U4Solid::init SUCCEEDED desc: U4Solid::desc solid Y type Y U4Solid::Tag(type) Uni root 59
    U4Solid::init SUCCEEDED desc: U4Solid::desc solid Y type Y U4Solid::Tag(type) Box root 60
    U4Solid::init SUCCEEDED desc: U4Solid::desc solid Y type Y U4Solid::Tag(type) Dis root 60


* "Dis" root not incremented ?

   * thats because Dis are "internal" nodes for the Geant4 model that just carry the transform
     they dont actually correspond to a separate constituent   

Dis ARE FUNNY NODES THAT JUST ACT TO HOLD THE TRANSFORM 
**/
inline void U4Solid::init_DisplacedSolid()
{
    const G4DisplacedSolid* disp = static_cast<const G4DisplacedSolid*>(solid);
    assert(disp); 

    glm::tmat4x4<double> xf(1.) ; 
    U4Transform::GetDispTransform( xf, disp );     

    const G4VSolid* moved = Moved(solid); 
    assert(moved); 

    bool single_disp = dynamic_cast<const G4DisplacedSolid*>(moved) == nullptr ; 
    assert(single_disp && "only single disp is expected" );
    if(!single_disp) std::raise(SIGINT); 

    root = Convert( moved, lvid, depth+1, level ); 
    root->combineXF(xf); 
} 


