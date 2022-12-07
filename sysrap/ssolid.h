#pragma once
/**
ssolid.h
===========

**/

#include <cassert>
#include <iostream>
#include <iomanip>

#include <glm/glm.hpp>
#include "scuda.h"
#include "squad.h"
#include "sgeomdefs.h"

#include "G4VSolid.hh"
#include "G4MultiUnion.hh"
#include "G4ThreeVector.hh"

struct ssolid
{
    static void GetCenterExtent( float4& ce,             const G4VSolid* solid );  
    static void GetCenterExtent( glm::tvec4<double>& ce, const G4VSolid* solid );  
    static G4double Distance_(const G4VSolid* solid, const G4ThreeVector& pos, const G4ThreeVector& dir, EInside& in ); 
    static G4double DistanceMultiUnionNoVoxels_(
                          const G4MultiUnion* solid, const G4ThreeVector& pos, const G4ThreeVector& dir, EInside& in ); 

    static G4double Distance(const G4VSolid* solid, const G4ThreeVector& pos, const G4ThreeVector& dir, bool dump ); 
    static void Simtrace( quad4& p, const G4VSolid* solid, bool dump=false); 
}; 

inline void ssolid::GetCenterExtent( glm::tvec4<double>& ce, const G4VSolid* solid ) // static
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

inline void ssolid::GetCenterExtent( float4& ce, const G4VSolid* solid ) // static
{
    glm::tvec4<double> ce_ ; 
    GetCenterExtent(ce_, solid ); 
    ce.x = ce_.x ;  
    ce.y = ce_.y ;  
    ce.z = ce_.z ;  
    ce.w = ce_.w ;  
}


inline G4double ssolid::Distance_(const G4VSolid* solid, const G4ThreeVector& pos, const G4ThreeVector& dir, EInside& in ) // static
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



inline G4double ssolid::DistanceMultiUnionNoVoxels_(const G4MultiUnion* solid, const G4ThreeVector& pos, const G4ThreeVector& dir, EInside& in ) // static
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


inline G4double ssolid::Distance(const G4VSolid* solid, const G4ThreeVector& pos, const G4ThreeVector& dir, bool dump ) // static
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
            << " in " << sgeomdefs::EInside_(in ) 
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


/**
ssolid::Simtrace
-------------------

Updates quad4& p in simtrace layout with intersect position onto the solid. 

p.q0.f.xyz,w
    surface normal at intersect (TODO) and intersect distance *t*

p.q1.f.xyz, w
    intersect position and "tmin" param

p.q2.f.xyz
    input position 

p.q3.f.xyz
    input direction

**/

inline void ssolid::Simtrace(quad4& p, const G4VSolid* solid, bool dump) // static
{
    G4ThreeVector ori(p.q2.f.x, p.q2.f.y, p.q2.f.z); 
    G4ThreeVector dir(p.q3.f.x, p.q3.f.y, p.q3.f.z); 
 
    G4double t = Distance( solid, ori, dir, dump );  
    //std::cout << "ssolid::Simtrace " << t << std::endl ; 

    if( t == kInfinity ) return ;   // hmm: perhaps set ipos to ori for MISS ? Currently gets left at origin

    G4ThreeVector ipos = ori + dir*t ; 
    float tmin = 0.f ; 

    p.q0.f.x = 0.f ; // TODO surface normal at intersect ? 
    p.q0.f.y = 0.f ; 
    p.q0.f.z = 0.f ; 
    p.q0.f.w = t ; 

    p.q1.f.x = float(ipos.x()) ; 
    p.q1.f.y = float(ipos.y()) ; 
    p.q1.f.z = float(ipos.z()) ; 
    p.q1.f.w = tmin  ; 
}


