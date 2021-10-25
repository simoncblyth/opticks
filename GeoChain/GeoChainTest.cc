#include <cassert>
#include "OPTICKS_LOG.hh"
#include "Opticks.hh"
#include "GeoChain.h"

#include "G4SystemOfUnits.hh"
#include "G4Polycone.hh"
#include "G4Sphere.hh"
#include "G4Tubs.hh"
#include "G4SubtractionSolid.hh"
#include "G4ThreeVector.hh"
#include "G4Orb.hh"

using CLHEP::pi ;

const G4VSolid* const make_default(const char* name)
{
    return new G4Orb(name, 100.) ; 
}

const G4VSolid* const make_AdditionAcrylicConstruction(const char* name)
{
    G4VSolid*   solidAddition_down;
    G4VSolid*   solidAddition_up;
    G4VSolid*   uni_acrylic1;

    double ZNodes3[3];
    double RminNodes3[3];
    double RmaxNodes3[3];
    ZNodes3[0] = 5.7*mm; RminNodes3[0] = 0*mm; RmaxNodes3[0] = 450.*mm;
    ZNodes3[1] = 0.0*mm; RminNodes3[1] = 0*mm; RmaxNodes3[1] = 450.*mm;
    ZNodes3[2] = -140.0*mm; RminNodes3[2] = 0*mm; RmaxNodes3[2] = 200.*mm;

    solidAddition_down = new G4Polycone("solidAddition_down",0.0*deg,360.0*deg,3,ZNodes3,RminNodes3,RmaxNodes3);
    solidAddition_up = new G4Sphere("solidAddition_up",0*mm,17820*mm,0.0*deg,360.0*deg,0.0*deg,180.*deg);
    uni_acrylic1 = new G4SubtractionSolid("uni_acrylic1",solidAddition_down,solidAddition_up,0,G4ThreeVector(0*mm,0*mm,+17820.0*mm));

    G4VSolid*   solidAddition_up1;
    G4VSolid*   solidAddition_up2;
    G4VSolid*   uni_acrylic2;
    G4VSolid*   uni_acrylic3;

    G4VSolid*   uni_acrylic2_initial ;


    solidAddition_up1 = new G4Tubs("solidAddition_up1",120*mm,208*mm,15.2*mm,0.0*deg,360.0*deg);
    uni_acrylic2 = new G4SubtractionSolid("uni_acrylic2",uni_acrylic1,solidAddition_up1,0,G4ThreeVector(0.*mm,0.*mm,-20*mm));
    solidAddition_up2 = new G4Tubs("solidAddition_up2",0,14*mm,52.5*mm,0.0*deg,360.0*deg);

    uni_acrylic2_initial = uni_acrylic2 ; 

    for(int i=0;i<8;i++)
    {
    uni_acrylic3 = new G4SubtractionSolid("uni_acrylic3",uni_acrylic2,solidAddition_up2,0,G4ThreeVector(164.*cos(i*pi/4)*mm,164.*sin(i*pi/4)*mm,-87.5));
    uni_acrylic2 = uni_acrylic3;
    }

    //return solidAddition_down ; 
    //return uni_acrylic1 ; 
    return uni_acrylic2_initial ; 
}

const G4VSolid* const make_solid(const char* name)
{
    const G4VSolid* solid = nullptr ; 
    if(strcmp(name, "default") == 0)                     solid = make_default(name);  
    if(strcmp(name,"AdditionAcrylicConstruction") == 0 ) solid = make_AdditionAcrylicConstruction(name); 
    assert(solid); 
    G4cout << *solid << G4endl ; 
    return solid ; 
}

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    const char* name = "AdditionAcrylicConstruction"  ; 
    const G4VSolid* const solid = make_solid(name);   

    const char* argforced = "--allownokey" ; 
    Opticks ok(argc, argv, argforced); 
    ok.configure(); 

    GeoChain chain(&ok); 
    chain.convert(solid);  
    chain.save(name); 

    return 0 ; 
}
