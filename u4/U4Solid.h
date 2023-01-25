#pragma once
/**
U4Solid.h : Convert G4VSolid CSG trees into transient snd<double> trees
========================================================================

Canonical usage from U4Tree::initSolid

Looks like can base U4Solid on X4Solid with fairly minor changes:

1. NNode swapped to snd<double>
2. no equivalent to X4SolidBase consolidate the base
3. header-only U4Entity.h based on X4Entity.{hh,cc}   




X4Solid
    from G4VSolid into NNode 

X4SolidBase
    does little, G4 params, mainly placeholder convert methods that assert

X4Entity
    create headeronly U4Entity.h from this 

NNode
    organic mess that need to leave behind 

U4SolidTree
    some overlap with X4Entity, tree mechanics might be better that X4Solid 

**/


#include "scuda.h"
#include "snd.h"
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

    static int          Type(const G4VSolid* solid ) ;  
    static const char*  Tag( int type ) ;   
    static snd*         Convert(const G4VSolid* solid ) ; 

    U4Solid( const G4VSolid* solid ); 
    void init(); 
    void setRoot(const snd& nd); 

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

    static OpticksCSG_t BooleanOperator(const G4BooleanSolid* solid ); 
    static bool  IsBoolean(   const G4VSolid* solid) ;
    static bool  IsDisplaced( const G4VSolid* solid) ;

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
    std::string desc() const ; 

    // members
    const G4VSolid* solid ; 
    int             type ;
    snd*            root ;     
};

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

inline snd* U4Solid::Convert(const G4VSolid* solid) // static
{
    U4Solid so(solid); 
    return so.root ; 
}

inline U4Solid::U4Solid(const G4VSolid* solid_)
    :
    solid(solid_),
    type(Type(solid)),
    root(nullptr)
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

    if(root == nullptr)
    {
        std::cerr << "U4Solid::init FAILED desc: " << desc() << std::endl ; 
        assert(0); 
    }
}

inline void U4Solid::setRoot(const snd& nd)
{
    root = new snd(nd) ; 
}

inline void U4Solid::init_Orb()
{
    const G4Orb* orb = dynamic_cast<const G4Orb*>(solid);
    assert(orb);
    double radius = orb->GetRadius()/CLHEP::mm ;
    snd nd = snd::Sphere(radius) ; 
    setRoot(nd); 
}

inline void U4Solid::init_Sphere()
{
}

inline void U4Solid::init_Ellipsoid()
{
}


inline void U4Solid::init_Box()
{
    const G4Box* box = dynamic_cast<const G4Box*>(solid);
    assert(box);

    float hx = box->GetXHalfLength()/CLHEP::mm ;
    float hy = box->GetYHalfLength()/CLHEP::mm ;
    float hz = box->GetZHalfLength()/CLHEP::mm ;

    float fx = 2.0*hx ;
    float fy = 2.0*hy ;
    float fz = 2.0*hz ;

    snd nd = snd::Box3(fx, fy, fz) ; 
    setRoot(nd); 
}


inline void U4Solid::init_Tubs()
{
} 
inline void U4Solid::init_Polycone()
{
} 
inline void U4Solid::init_Cons()
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
        // HMM: looks a bit fishy being inconsisent here : until look at g4-cls G4DisplacedSolid
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
        // HMM: looks a bit fishy being inconsisent here : until look at g4-cls G4DisplacedSolid
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

    snd* l = Convert( left ); 
    snd* r = Convert( right ); 

    if(is_right_displaced) 
    {
        assert( r->tr && "expecting transform on right displaced node " ); 
    }

    snd nd = snd::Boolean( op, l, r ); 
    setRoot(nd);
}


/**
U4Solid::init_DisplacedSolid
------------------------------

When booleans are created with transforms the right hand side 
gets wrapped into a DisplacedSolid. 

**/
inline void U4Solid::init_DisplacedSolid()
{
    const G4VSolid* moved = Moved(solid); 
    assert(moved); 

    bool single_disp = dynamic_cast<const G4DisplacedSolid*>(moved) == nullptr ; 
    assert(single_disp && "only single disp is expected" );

    snd* m = Convert( moved ); 
    assert(m); 

    const G4DisplacedSolid* disp = static_cast<const G4DisplacedSolid*>(solid);
    assert(disp); 

    glm::tmat4x4<double> _tr(1.) ; 
    U4Transform::GetDispTransform( _tr, disp );     
    m->tr = Tran<double>::ConvertFromData(glm::value_ptr(_tr) ); 

    bool dump = false ;  
    if(dump) std::cout 
        << "U4Solid::init_DisplacedSolid" << std::endl 
        << " m.tr.desc " << std::endl 
        << m->tr->desc() 
        ; 

    root = m ; 
} 

inline std::string U4Solid::desc() const 
{
    std::stringstream ss ; 
    ss << "U4Solid::desc" 
       << " solid " << ( solid ? "Y" : "N" )
       << " type "  << ( type ? "Y" : "N" )
       << " U4Solid::Tag(type) " << U4Solid::Tag(type) 
       << " root "  << ( root ? "Y" : "N" )
       ; 
    std::string str = ss.str(); 
    return str ; 
}
