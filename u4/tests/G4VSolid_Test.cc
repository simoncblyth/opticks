
// ./G4VSolid_Test.sh 

#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <string>
#include <cstring>
#include <cassert>

#include "G4ThreeVector.hh"
#include "G4Polycone.hh"
#include "G4Ellipsoid.hh"
#include "G4UnionSolid.hh"

#include "sgeomdefs.h"
#include "sfmt.h"
#include "ssolid.h"

#include <CLHEP/Units/SystemOfUnits.h>
using CLHEP::mm ; 
using CLHEP::deg ; 


G4VSolid* MakePolycone( double r, double z0 , double z1 )
{
    G4double phiStart = 0.00*deg ; 
    G4double phiTotal = 360.00*deg ;
    G4int numZPlanes = 2 ; 
    G4double zPlane[] = { z0    , z1 } ;   
    G4double rInner[] = {  0.0  , 0.0   } ;   
    G4double rOuter[] = {  r    , r  } ;    

    G4VSolid* solid_1_2 = new G4Polycone(
                               "_1_2",
                               phiStart,
                               phiTotal,
                               numZPlanes,
                               zPlane,
                               rInner,
                               rOuter
                               );  

    return solid_1_2 ; 
}

G4VSolid* MakeUpperHemiEllipsoid( double P_I_R, double P_I_H )
{
    G4VSolid* solid_I = new G4Ellipsoid(
                    "_I",
                    P_I_R,
                    P_I_R,
                    P_I_H,
                    0, // pzBottomCut -> equator
                    P_I_H // pzTopCut -> top
                    );  
    return solid_I ; 
} 

G4VSolid* MakeLowerHemiEllipsoid( double P_I_R, double P_I_H )
{
    G4VSolid* solid_III = new G4Ellipsoid(
                      "_III",
                      P_I_R,
                      P_I_R,
                      P_I_H,
                      -P_I_H,
                      0);

    return solid_III ; 
}


G4VSolid* MakeInner1()
{
    G4double P_I_R = 254.*mm ; 
    G4double P_I_H = 190.*mm ; 
    return MakeUpperHemiEllipsoid( P_I_R, P_I_H ); 
}


G4VSolid* MakeInner2(const char* style)
{
    bool hemi_only = strcmp(style, "NNVT") == 0 ; 

    G4double P_I_R = 254.*mm ; 
    G4double P_I_H = 190.*mm ; 
    G4double m2_h  = 5.*mm ;  

    G4VSolid* solid_1_2 = MakePolycone( P_I_R, -m2_h, 0. ) ; 
    G4VSolid* solid_III = MakeLowerHemiEllipsoid( P_I_R, P_I_H ); 

    G4VSolid* solid_1_3 = new G4UnionSolid(
                 "_1_3",
                 solid_1_2,
                 solid_III,
                 0,
                 G4ThreeVector(0,0,-m2_h )
                 );

    return hemi_only ? solid_III : solid_1_3 ; 
}



void DumpDist(const char* style, const char* desc)
{
    G4ThreeVector dir(0,0,1) ;  // +Z
    G4VSolid* sol[2] ; 
    sol[0] = MakeInner1() ; 
    sol[1] = MakeInner2(style) ; 

    EInside  inn[2] ; 
    G4double d2i[2] ;
    G4double d2o[2] ;
    G4double dis[2] ;

    double zstep = 5. ; 
    const char* spacer = "    " ; 

    std::cout
        << std::setw(55) << ""
        << spacer
        << std::setw(10) << style 
        << std::setw(20) << desc
        << std::endl 
        ; 

    std::cout 
        << std::setw(15) << "pos"
        << std::setw(10) << "INNER1" 
        << std::setw(10) << "DistToIn"
        << std::setw(10) << "DistToOut"
        << std::setw(10) << "Dist"
        << spacer
        << std::setw(10) << "INNER2" 
        << std::setw(10) << "DistToIn"
        << std::setw(10) << "DistToOut"
        << std::setw(10) << "Dist"
        << spacer
        << std::setw(5) << "trig"
        << std::endl 
        ;

    for(double z=20 ; z >= -300 ; z -= zstep )
    {
        zstep = z < 10 && z > -10 ? 1. : 5. ; 
        G4ThreeVector pos(0,0,z) ; 

        for(int i=0 ; i < 2 ; i++)
        {
            d2i[i] = sol[i]->DistanceToIn(pos, dir);
            d2o[i] = sol[i]->DistanceToOut(pos, dir); 
            dis[i] = ssolid::Distance_( sol[i], pos, dir, inn[i] ); 
        }

        double dist1 = d2i[0] ; 
        double dist2 = d2i[1] ;

        bool trig = false ; 
        if(dist1 == kInfinity)
        {   
            trig = false;
        }   
        else if(dist1>dist2)
        {   
            trig = false;
        }   
        else
        {   
            trig = true;
        }   


        std::cout 
            << std::setw(15) << sfmt::Format(&pos)
            << std::setw(10) << sgeomdefs::EInside_(inn[0]) 
            << sfmt::Format(d2i[0])
            << sfmt::Format(d2o[0]) 
            << sfmt::Format(dis[0])
            << spacer
            << std::setw(10) << sgeomdefs::EInside_(inn[1]) 
            << sfmt::Format(d2i[1])
            << sfmt::Format(d2o[1]) 
            << sfmt::Format(dis[1])
            << spacer
            << std::setw(5) << ( trig ? "YES" : "NO " ) 
            << std::endl 
            ;  
    }


}



int main()
{
    DumpDist("NNVT", " INNER2: Just Lower-Hemi-Ellipsoid"); 
    DumpDist("HAMA", " INNER2: Union of Polycone and Lower-Hemi-Ellipsoid" ); 
    return 0 ; 
}


/**

320 // Approximate nearest distance from the point p to the union of
321 // two solids
322 
323 G4double
324 G4UnionSolid::DistanceToIn( const G4ThreeVector& p) const
325 {
326 #ifdef G4BOOLDEBUG 
327   if( Inside(p) == kInside )
328   { 
329     G4cout << "WARNING - Invalid call in "
330            << "G4UnionSolid::DistanceToIn(p)" << G4endl
331            << "  Point p is inside !" << G4endl;
332     G4cout << "          p = " << p << G4endl;
333     G4cerr << "WARNING - Invalid call in "
334            << "G4UnionSolid::DistanceToIn(p)" << G4endl
335            << "  Point p is inside !" << G4endl;
336     G4cerr << "          p = " << p << G4endl;
337   }
338 #endif
339   G4double distA = fPtrSolidA->DistanceToIn(p) ;
340   G4double distB = fPtrSolidB->DistanceToIn(p) ;
341   G4double safety = std::min(distA,distB) ;
342   if(safety < 0.0) safety = 0.0 ;
343   return safety ;
344 }
345 


// note that fPtrSolidB will be a G4DisplacedSolid when there is a RHS transform 

**/



