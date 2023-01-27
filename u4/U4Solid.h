#pragma once
/**
U4Solid.h : Convert G4VSolid CSG trees into snd.hh trees
=================================================================

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
#include "scuda.h"
#include "snd.hh"
#include "stran.h"
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

    std::string desc() const ; 
    static int          Type(const G4VSolid* solid ) ;  
    static const char*  Tag( int type ) ;   
    static int          Convert(const G4VSolid* solid, int lvid=-1 ) ; 

    U4Solid( const G4VSolid* solid, int lvid=-1 ); 
    void init(); 
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

    int init_Sphere_(char layer); 
    int init_Cons_(char layer); 



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
    int             lvid ; 
    int             type ;
    int             root ;     
};

inline std::string U4Solid::desc() const 
{
    std::stringstream ss ; 
    ss << "U4Solid::desc" 
       << " solid " << ( solid ? "Y" : "N" )
       << " lvid "   << std::setw(3) << lvid    
       << " type "  << std::setw(3) << type
       << " root "  << std::setw(4) << root
       << " U4Solid::Tag(type) " << U4Solid::Tag(type) 
       ; 
    std::string str = ss.str(); 
    return str ; 
}

inline int U4Solid::Type(const G4VSolid* solid)   // static 
{
    G4GeometryType etype = solid->GetEntityType();  // G4GeometryType typedef for G4String
    const char* name = etype.c_str(); 
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

inline int U4Solid::Convert(const G4VSolid* solid, int lvid ) // static
{
    U4Solid so(solid, lvid); 
    return so.root ; 
}

inline U4Solid::U4Solid(const G4VSolid* solid_, int lvid_ )
    :
    solid(solid_),
    lvid(lvid_),
    type(Type(solid)),
    root(-1)
{
    init() ; 
}

inline void U4Solid::init()
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

    if(root == -1)
    {
        std::cerr << "U4Solid::init FAILED desc: " << desc() << std::endl ; 
        assert(0); 
    }
    else
    {
        std::cerr << "U4Solid::init SUCCEEDED desc: " << desc() << std::endl ; 
        // TODO: snd::SetLVID(root, lvid); 
    }
}


inline void U4Solid::init_Orb()
{
    const G4Orb* orb = dynamic_cast<const G4Orb*>(solid);
    assert(orb);
    double radius = orb->GetRadius()/CLHEP::mm ;
    root = snd::Sphere(radius) ; 
}

inline void U4Solid::init_Sphere()
{
    int outer = init_Sphere_('O');  assert( outer > -1 ); 
    int inner = init_Sphere_('I');
    root = inner == -1 ? outer : snd::Boolean( CSG_DIFFERENCE, outer, inner ) ;
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

inline int U4Solid::init_Sphere_(char layer)
{
    assert( layer == 'I' || layer == 'O' ); 
    const G4Sphere* sphere = dynamic_cast<const G4Sphere*>(solid);
    assert(sphere);  

    double rmax = sphere->GetOuterRadius()/CLHEP::mm ; 
    double rmin = sphere->GetInnerRadius()/CLHEP::mm ; 
    double radius = layer == 'I' ? rmin : rmax ;
    if(radius == 0.) return -1 ; 

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
    assert( has_deltaPhi == false ); 

    return z_slice ? snd::ZSphere( radius, zmin, zmax ) : snd::Sphere(radius ) ; 
}

inline void U4Solid::init_Ellipsoid()
{
}
inline void U4Solid::init_Hype()
{
} 
inline void U4Solid::init_MultiUnion()
{
} 
inline void U4Solid::init_Torus()
{
    assert(0); 
}




inline void U4Solid::init_Box()
{
    const G4Box* box = dynamic_cast<const G4Box*>(solid);
    assert(box);

    double fx = 2.0*box->GetXHalfLength()/CLHEP::mm ;
    double fy = 2.0*box->GetYHalfLength()/CLHEP::mm ;
    double fz = 2.0*box->GetZHalfLength()/CLHEP::mm ;

    root = snd::Box3(fx, fy, fz) ; 
}


inline void U4Solid::init_Tubs()
{
    const G4Tubs* tubs = dynamic_cast<const G4Tubs*>(solid);
    assert(tubs); 

    double hz = tubs->GetZHalfLength()/CLHEP::mm ;  
    double rmax = tubs->GetOuterRadius()/CLHEP::mm ; 
    double rmin = tubs->GetInnerRadius()/CLHEP::mm ; 
    bool has_inner = rmin > 0. ; 

    int outer = snd::Cylinder(rmax, -hz, hz );


    if(has_inner == false)
    {
        root = outer ; 
    }
    else
    {
        bool   do_nudge_inner = true ; 
        double nudge_inner = 0.01 ;
        double dz = do_nudge_inner ? hz*nudge_inner : 0. ; 

        int inner = snd::Cylinder(rmin, -(hz+dz), hz+dz );  

        root = snd::Boolean( CSG_DIFFERENCE, outer, inner ); 
    }

} 


inline void U4Solid::init_Polycone()
{
    const G4Polycone* polycone = dynamic_cast<const G4Polycone*>(solid);
    assert(polycone);
    root = U4Polycone::Convert(polycone); 
}




inline int U4Solid::init_Cons_(char layer)
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

    return invalid ? -1 : snd::Cone( r1, z1, r2, z2 ) ;  
} 

inline void U4Solid::init_Cons()
{
    int outer = init_Cons_('O');  assert( outer > -1 ); 
    int inner = init_Cons_('I');
    root = inner == -1 ? outer : snd::Boolean( CSG_DIFFERENCE, outer, inner ) ;
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
    assert( !is_left_displaced && "not expecting left displacement " );

    int l = Convert( left ); 
    int r = Convert( right ); 

    int l_xf = snd::GetNodeXForm(l);
    int r_xf = snd::GetNodeXForm(r);

    assert( l_xf == -1  && "NOT expecting transform on left node " ); 
    if(is_right_displaced) assert( r_xf > -1  && "expecting transform on right displaced node " ); 

    root = snd::Boolean( op, l, r ); 
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

    int m = Convert( moved ); 
    assert(m > -1); 

    snd* nd = snd::GetNode_(m);  assert(nd);   
    nd->setXForm(xf); 

    root = m ;    // HMM : Dis ARE FUNNY NODES THAT JUST ACT TO HOLD THE TRANSFORM 
} 


