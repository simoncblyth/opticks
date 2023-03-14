
// ./G4VSolid_Test.sh 

#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <string>
#include <cassert>

#include "G4ThreeVector.hh"
#include "G4Polycone.hh"
#include "G4Ellipsoid.hh"
#include "G4UnionSolid.hh"

#include "sgeomdefs.h"
#include "ssolid.h"
#include <CLHEP/Units/SystemOfUnits.h>


const G4VSolid* GetSolid()
{
    using CLHEP::mm ; 
    using CLHEP::deg ; 

    G4double P_I_R = 254.*mm ; 
    G4double P_I_H = 190.*mm ; 
    G4double m2_h = 5.*mm ;  

    G4double phiStart = 0.00*deg ; 
    G4double phiTotal = 360.00*deg ;
    G4int numZPlanes = 2 ; 
    G4double zPlane[] = { -m2_h            , 0.0  } ;   
    G4double rInner[] = {  0.0             , 0.0   } ;   
    G4double rOuter[] = {  P_I_R           , P_I_R  } ;    

    G4VSolid* solid_1_2 = new G4Polycone(
                               "_1_2",
                               phiStart,
                               phiTotal,
                               numZPlanes,
                               zPlane,
                               rInner,
                               rOuter
                               );  

    G4VSolid* solid_III = new G4Ellipsoid(
                      "_III",
                      P_I_R,
                      P_I_R,
                      P_I_H,
                      -P_I_H,
                      0);


    G4double boolean_shift = -m2_h ;  
    //G4double boolean_shift = 0. ;  


    G4VSolid* solid_1_3 = new G4UnionSolid(
                 "_1_3",
                 solid_1_2,
                 solid_III,
                 0,
                 G4ThreeVector(0,0,boolean_shift )
                 );

    const G4VSolid* sso2 = solid_1_3 ; 
    return sso2 ; 
}


std::string Format(double v, int w=10)
{
    std::stringstream ss ;    
    if( v == kInfinity ) 
    { 
        ss << std::setw(w) << "kInfinity" ; 
    }
    else
    {
        ss << std::setw(w) << std::fixed << std::setprecision(4) << v ; 
    }
    std::string str = ss.str(); 
    return str ; 
}

std::string Format(const G4ThreeVector* v)
{
    std::stringstream ss ;    
    ss << *v ; 
    std::string str = ss.str(); 
    return str ; 
}

int main()
{
    const G4VSolid* so = GetSolid(); 
    G4ThreeVector dir(0,0,1) ;  // +Z

    for(double z=20 ; z >= -220 ; z-=5. )
    {
        G4ThreeVector pos(0,0,z) ; 
        G4double d2i = so->DistanceToIn(pos, dir);
        G4double d2o = so->DistanceToOut(pos, dir);

        EInside in ; 
        G4double dis = ssolid::Distance_(so, pos, dir, in);
        std::cout 
            << " pos " << std::setw(15) << Format(&pos)
            << " " << std::setw(10) << sgeomdefs::EInside_(in) 
            << " "
            << " d2i " << Format(d2i)
            << " d2o " << Format(d2o) 
            << " dis " << Format(dis) 
            << std::endl 
            ;  
    }
    return 0 ; 
}

/**
Surprising DistanceToIn behaviour with booleans when inside

epsilon:tests blyth$ ./G4VSolid_Test.sh 
 pos        (0,0,20)   kOutside  d2i  kInfinity d2o     0.0000 dis  kInfinity
 pos        (0,0,15)   kOutside  d2i  kInfinity d2o     0.0000 dis  kInfinity
 pos        (0,0,10)   kOutside  d2i  kInfinity d2o     0.0000 dis  kInfinity
 pos         (0,0,5)   kOutside  d2i  kInfinity d2o     0.0000 dis  kInfinity
 pos         (0,0,0)   kSurface  d2i  kInfinity d2o     0.0000 dis     0.0000
 pos        (0,0,-5)    kInside  d2i     0.0000 d2o     5.0000 dis     5.0000
 pos       (0,0,-10)    kInside  d2i     5.0000 d2o    10.0000 dis    10.0000
 pos       (0,0,-15)    kInside  d2i    10.0000 d2o    15.0000 dis    15.0000
 pos       (0,0,-20)    kInside  d2i    15.0000 d2o    20.0000 dis    20.0000
 pos       (0,0,-25)    kInside  d2i    20.0000 d2o    25.0000 dis    25.0000
 pos       (0,0,-30)    kInside  d2i    25.0000 d2o    30.0000 dis    30.0000
 pos       (0,0,-35)    kInside  d2i    30.0000 d2o    35.0000 dis    35.0000
 pos       (0,0,-40)    kInside  d2i    35.0000 d2o    40.0000 dis    40.0000
 pos       (0,0,-45)    kInside  d2i    40.0000 d2o    45.0000 dis    45.0000
 pos       (0,0,-50)    kInside  d2i    45.0000 d2o    50.0000 dis    50.0000
 pos       (0,0,-55)    kInside  d2i    50.0000 d2o    55.0000 dis    55.0000
 pos       (0,0,-60)    kInside  d2i    55.0000 d2o    60.0000 dis    60.0000
 pos       (0,0,-65)    kInside  d2i    60.0000 d2o    65.0000 dis    65.0000
 pos       (0,0,-70)    kInside  d2i    65.0000 d2o    70.0000 dis    70.0000
 pos       (0,0,-75)    kInside  d2i    70.0000 d2o    75.0000 dis    75.0000
 pos       (0,0,-80)    kInside  d2i    75.0000 d2o    80.0000 dis    80.0000
 pos       (0,0,-85)    kInside  d2i    80.0000 d2o    85.0000 dis    85.0000
 pos       (0,0,-90)    kInside  d2i    85.0000 d2o    90.0000 dis    90.0000
 pos       (0,0,-95)    kInside  d2i    90.0000 d2o    95.0000 dis    95.0000
 pos      (0,0,-100)    kInside  d2i    95.0000 d2o   100.0000 dis   100.0000
 pos      (0,0,-105)    kInside  d2i   100.0000 d2o   105.0000 dis   105.0000
 pos      (0,0,-110)    kInside  d2i   105.0000 d2o   110.0000 dis   110.0000
 pos      (0,0,-115)    kInside  d2i   110.0000 d2o   115.0000 dis   115.0000
 pos      (0,0,-120)    kInside  d2i   115.0000 d2o   120.0000 dis   120.0000
 pos      (0,0,-125)    kInside  d2i   120.0000 d2o   125.0000 dis   125.0000
 pos      (0,0,-130)    kInside  d2i   125.0000 d2o   130.0000 dis   130.0000
 pos      (0,0,-135)    kInside  d2i   130.0000 d2o   135.0000 dis   135.0000
 pos      (0,0,-140)    kInside  d2i   135.0000 d2o   140.0000 dis   140.0000
 pos      (0,0,-145)    kInside  d2i   140.0000 d2o   145.0000 dis   145.0000
 pos      (0,0,-150)    kInside  d2i   145.0000 d2o   150.0000 dis   150.0000
 pos      (0,0,-155)    kInside  d2i   150.0000 d2o   155.0000 dis   155.0000
 pos      (0,0,-160)    kInside  d2i   155.0000 d2o   160.0000 dis   160.0000
 pos      (0,0,-165)    kInside  d2i   160.0000 d2o   165.0000 dis   165.0000
 pos      (0,0,-170)    kInside  d2i   165.0000 d2o   170.0000 dis   170.0000
 pos      (0,0,-175)    kInside  d2i   170.0000 d2o   175.0000 dis   175.0000
 pos      (0,0,-180)    kInside  d2i   175.0000 d2o   180.0000 dis   180.0000
 pos      (0,0,-185)    kInside  d2i   180.0000 d2o   185.0000 dis   185.0000
 pos      (0,0,-190)    kInside  d2i   185.0000 d2o   190.0000 dis   190.0000
 pos      (0,0,-195)   kSurface  d2i     0.0000 d2o   195.0000 dis   195.0000
 pos      (0,0,-200)   kOutside  d2i     5.0000 d2o     0.0000 dis     5.0000
 pos      (0,0,-205)   kOutside  d2i    10.0000 d2o     0.0000 dis    10.0000
 pos      (0,0,-210)   kOutside  d2i    15.0000 d2o     0.0000 dis    15.0000
 pos      (0,0,-215)   kOutside  d2i    20.0000 d2o     0.0000 dis    20.0000
 pos      (0,0,-220)   kOutside  d2i    25.0000 d2o     0.0000 dis    25.0000
epsilon:tests blyth$ 



**/

