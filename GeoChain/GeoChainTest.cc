#include <cassert>
#include "OPTICKS_LOG.hh"
#include "SSys.hh"
#include "SStr.hh"
#include "Opticks.hh"
#include "GeoChain.h"

#ifdef WITH_PMTSIM
#include "PMTSim.hh"
#endif

#include "G4SystemOfUnits.hh"
#include "G4Polycone.hh"
#include "G4Sphere.hh"
#include "G4Box.hh"
#include "G4Tubs.hh"
#include "G4SubtractionSolid.hh"
#include "G4ThreeVector.hh"
#include "G4Orb.hh"

using CLHEP::pi ;

const G4VSolid* const make_PMTSim(const char* name, std::string& meta)
{
    std::stringstream ss ; 
    ss << "creator:make_PMTSim" << std::endl ; 
    const G4VSolid* solid = nullptr ;  
#ifdef WITH_PMTSIM
    LOG(info) << "PMTSim::GetSolid " ; 
    solid = PMTSim::GetSolid(name) ;   // for zcut include integer in name eg "PMTSim_Z-400" 
#endif
    meta = ss.str();   
    return solid ; 
}

const G4VSolid* const make_default(const char* name)
{
    return new G4Orb(name, 100.) ; 
}

const G4VSolid* const make_AdditionAcrylicConstruction(const char* name)
{
    G4VSolid*   simple             = nullptr ;
    G4VSolid*   solidAddition_down = nullptr ;
    G4VSolid*   solidAddition_up   = nullptr ;
    G4VSolid*   uni_acrylic1       = nullptr ;

    double ZNodes3[3];
    double RminNodes3[3];
    double RmaxNodes3[3];
    ZNodes3[0] = 5.7*mm; RminNodes3[0] = 0*mm; RmaxNodes3[0] = 450.*mm;
    ZNodes3[1] = 0.0*mm; RminNodes3[1] = 0*mm; RmaxNodes3[1] = 450.*mm;
    ZNodes3[2] = -140.0*mm; RminNodes3[2] = 0*mm; RmaxNodes3[2] = 200.*mm;

    bool replace_poly = true ; 
    bool skip_sphere = true ; 

    simple = new G4Tubs("simple", 0*mm, 450*mm, 200*mm, 0.0*deg,360.0*deg);
    solidAddition_down = replace_poly ? simple : new G4Polycone("solidAddition_down",0.0*deg,360.0*deg,3,ZNodes3,RminNodes3,RmaxNodes3);

    solidAddition_up = new G4Sphere("solidAddition_up",0*mm,17820*mm,0.0*deg,360.0*deg,0.0*deg,180.*deg);
    uni_acrylic1 = skip_sphere ? solidAddition_down : new G4SubtractionSolid("uni_acrylic1",solidAddition_down,solidAddition_up,0,G4ThreeVector(0*mm,0*mm,+17820.0*mm));

    G4VSolid*   solidAddition_up1 = nullptr ;
    G4VSolid*   solidAddition_up2 = nullptr ;
    G4VSolid*   uni_acrylic2      = nullptr ;
    G4VSolid*   uni_acrylic3      = nullptr ;

    G4VSolid*   uni_acrylic2_initial = nullptr ;

    //double zshift = -20*mm ; 
    double zshift = 0*mm ; 

    solidAddition_up1 = new G4Tubs("solidAddition_up1",120*mm,208*mm,15.2*mm,0.0*deg,360.0*deg);
    uni_acrylic2 = new G4SubtractionSolid("uni_acrylic2",uni_acrylic1,solidAddition_up1,0,G4ThreeVector(0.*mm,0.*mm,zshift));
    uni_acrylic2_initial = uni_acrylic2 ; 


    solidAddition_up2 = new G4Tubs("solidAddition_up2",0,14*mm,52.5*mm,0.0*deg,360.0*deg);

    for(int i=0;i<8;i++)
    {
    uni_acrylic3 = new G4SubtractionSolid("uni_acrylic3",uni_acrylic2,solidAddition_up2,0,G4ThreeVector(164.*cos(i*pi/4)*mm,164.*sin(i*pi/4)*mm,-87.5));
    uni_acrylic2 = uni_acrylic3;
    }


    LOG(info)
        << " solidAddition_down " << solidAddition_down
        << " solidAddition_up " << solidAddition_up
        << " solidAddition_up1 " << solidAddition_up1
        << " solidAddition_up2 " << solidAddition_up2
        << " uni_acrylic2_initial " << uni_acrylic2_initial
        << " uni_acrylic1 " << uni_acrylic1 
        << " uni_acrylic2 " << uni_acrylic2
        << " uni_acrylic3 " << uni_acrylic3
        ;   

    //return solidAddition_down ;  // union of cylinder and cone
    //return solidAddition_up1 ;     // pipe cylinder 
    //return uni_acrylic1 ; 
    return uni_acrylic2_initial ; 
}

const G4VSolid* const make_BoxMinusTubs0(const char* name)  // is afflicted
{
    double tubs_hz = 15.2*mm ;   
    double zshift = 0*mm ; 
    G4VSolid* box   = new G4Box("box",  250*mm, 250*mm, 100*mm ); 
    G4VSolid* tubs =  new G4Tubs("tubs",120*mm,208*mm,tubs_hz,0.0*deg,360.0*deg);
    G4VSolid* box_minus_tubs = new G4SubtractionSolid(name,box,tubs,0,G4ThreeVector(0.*mm,0.*mm,zshift));  
    return box_minus_tubs ; 
}

const G4VSolid* const make_BoxMinusTubs1(const char* name) 
{
    double tubs_hz = 15.2*mm ;   
    G4VSolid* box   = new G4Box("box",  250*mm, 250*mm, 100*mm ); 
    G4VSolid* tubs =  new G4Tubs("tubs",120*mm,208*mm,tubs_hz,0.0*deg,360.0*deg);
    G4VSolid* box_minus_tubs = new G4SubtractionSolid(name,box,tubs);  
    return box_minus_tubs ; 
}

const G4VSolid* const make_solid(const char* name, std::string& meta)
{
    LOG(info) << "[ " << name ; 
    const G4VSolid* solid = nullptr ; 
    if(strcmp(name, "default") == 0)                     solid = make_default(name);  
    if(strcmp(name,"AdditionAcrylicConstruction") == 0 ) solid = make_AdditionAcrylicConstruction(name); 
    if(strcmp(name,"BoxMinusTubs0") == 0 )               solid = make_BoxMinusTubs0(name); 
    if(strcmp(name,"BoxMinusTubs1") == 0 )               solid = make_BoxMinusTubs1(name); 
    if(SStr::StartsWith(name, "PMTSim"))                 solid = make_PMTSim(name, meta) ;  
    assert(solid); 
    //G4cout << *solid << G4endl ; 

    LOG(info) << "] " << name ; 
    return solid ; 
}

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    const char* name_default = "AdditionAcrylicConstruction"  ; 
    const char* name = SSys::getenvvar("GEOCHAINTEST", name_default ); 

    std::string meta ; 
    const G4VSolid* const solid = make_solid(name, meta);   

    const char* argforced = "--allownokey" ; 
    Opticks ok(argc, argv, argforced); 
    ok.configure(); 
    //for(int lvIdx=-1 ; lvIdx < 10 ; lvIdx+= 1 ) LOG(info) << " lvIdx " << lvIdx << " ok.isX4TubsNudgeSkip(lvIdx) " << ok.isX4TubsNudgeSkip(lvIdx)  ; 

    GeoChain chain(&ok); 
    chain.convert(solid, meta);  
    chain.save(name); 

    return 0 ; 
}
