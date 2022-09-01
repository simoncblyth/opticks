#pragma once

#include <cassert>
#include <iostream>
#include <iomanip>
#include <glm/glm.hpp>
#include "scuda.h"

#include "x4geomdefs.h"

#include "G4VSolid.hh"
#include "G4MultiUnion.hh"
#include "G4ThreeVector.hh"

struct x4solid
{
    static void GetCenterExtent( float4& ce,             const G4VSolid* solid );  
    static void GetCenterExtent( glm::tvec4<double>& ce, const G4VSolid* solid );  
    static G4double Distance_(const G4VSolid* solid, const G4ThreeVector& pos, const G4ThreeVector& dir, EInside& in ); 
    static G4double DistanceMultiUnionNoVoxels_(
                          const G4MultiUnion* solid, const G4ThreeVector& pos, const G4ThreeVector& dir, EInside& in ); 

    static G4double Distance(const G4VSolid* solid, const G4ThreeVector& pos, const G4ThreeVector& dir, bool dump ); 

}; 

inline void x4solid::GetCenterExtent( glm::tvec4<double>& ce, const G4VSolid* solid )
{
    G4ThreeVector pMin ; 
    G4ThreeVector pMax ; 
    solid->BoundingLimits(pMin, pMax); 
    G4ThreeVector center = ( pMin + pMax )/2. ;   
    G4ThreeVector fulldiag = pMax - pMin ; 
    G4ThreeVector halfdiag = fulldiag/2.  ;   
    G4double extent = std::max( std::max( halfdiag.x(), halfdiag.y() ), halfdiag.z() ) ;   

    ce.x = center.x() ; 
    ce.y = center.y() ; 
    ce.z = center.z() ; 
    ce.w = extent ; 
}

inline void x4solid::GetCenterExtent( float4& ce, const G4VSolid* solid )
{
    glm::tvec4<double> ce_ ; 
    GetCenterExtent(ce_, solid ); 
    ce.x = ce_.x ;  
    ce.y = ce_.y ;  
    ce.z = ce_.z ;  
    ce.w = ce_.w ;  
}


inline G4double x4solid::Distance_(const G4VSolid* solid, const G4ThreeVector& pos, const G4ThreeVector& dir, EInside& in ) // static
{
    in =  solid->Inside(pos) ; 
    G4double t = kInfinity ; 
    switch( in )
    {
        case kInside:  t = solid->DistanceToOut( pos, dir ) ; break ; 
        case kSurface: t = solid->DistanceToOut( pos, dir ) ; break ; 
        case kOutside: t = solid->DistanceToIn(  pos, dir ) ; break ; 
        default:  assert(0) ; 
    }
    return t ; 
}



inline G4double x4solid::DistanceMultiUnionNoVoxels_(const G4MultiUnion* solid, const G4ThreeVector& pos, const G4ThreeVector& dir, EInside& in ) // static
{
    in =  solid->InsideNoVoxels(pos) ; 
    G4double t = kInfinity ; 
    switch( in )
    {
        case kInside:  t = solid->DistanceToOutNoVoxels( pos, dir, nullptr ) ; break ; 
        case kSurface: t = solid->DistanceToOutNoVoxels( pos, dir, nullptr ) ; break ; 
        case kOutside: t = solid->DistanceToInNoVoxels(  pos, dir ) ; break ; 
        default:  assert(0) ; 
    }
    return t ; 
}


inline G4double x4solid::Distance(const G4VSolid* solid, const G4ThreeVector& pos, const G4ThreeVector& dir, bool dump ) // static
{
    EInside in ; 
    const G4MultiUnion* m = dynamic_cast<const G4MultiUnion*>(solid) ; 
    G4double t = m ? DistanceMultiUnionNoVoxels_(m, pos, dir, in ) : Distance_( solid, pos, dir, in  );  

    if(dump && t != kInfinity)
    {
        std::cout 
            << " pos " 
            << "(" 
            << std::fixed << std::setw(10) << std::setprecision(3) << pos.x() << " "
            << std::fixed << std::setw(10) << std::setprecision(3) << pos.y() << " "
            << std::fixed << std::setw(10) << std::setprecision(3) << pos.z() 
            << ")"
            << " dir " 
            << "(" 
            << std::fixed << std::setw(10) << std::setprecision(3) << dir.x() << " "
            << std::fixed << std::setw(10) << std::setprecision(3) << dir.y() << " "
            << std::fixed << std::setw(10) << std::setprecision(3) << dir.z() 
            << ")"
            << " in " << x4geomdefs::EInside_(in ) 
            ;

       if( t == kInfinity)
       {  
            std::cout 
                << " t " << std::setw(10) << "kInfinity" 
                << std::endl 
                ; 
       }
       else
       {
           G4ThreeVector ipos = pos + dir*t ;  
           std::cout 
                << " t " << std::fixed << std::setw(10) << std::setprecision(3) << t 
                << " ipos " 
                << "(" 
                << std::fixed << std::setw(10) << std::setprecision(3) << ipos.x() << " "
                << std::fixed << std::setw(10) << std::setprecision(3) << ipos.y() << " "
                << std::fixed << std::setw(10) << std::setprecision(3) << ipos.z() 
                << ")"
                << std::endl 
                ; 
       }
    }
    return t ; 
}


