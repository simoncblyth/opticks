#include <cstring>
#include "G4SystemOfUnits.hh"
#include "G4Polycone.hh"
#include "G4Sphere.hh"
#include "G4Box.hh"
#include "G4Tubs.hh"
#include "G4Cons.hh"
#include "G4SubtractionSolid.hh"
#include "G4ThreeVector.hh"
#include "G4Orb.hh"
#include "G4UnionSolid.hh"
#include "G4Ellipsoid.hh"

using CLHEP::pi ;

#include "SSys.hh"
#include "X4SolidMaker.hh"
#include "PLOG.hh"

const plog::Severity X4SolidMaker::LEVEL = PLOG::EnvLevel("X4SolidMaker", "DEBUG"); 

bool X4SolidMaker::StartsWith( const char* n, const char* q ) // static
{
    return strlen(q) >= strlen(n) && strncmp(q, n, strlen(n)) == 0 ; 
}

bool X4SolidMaker::CanMake(const char* qname) // static 
{
    bool found = false ; 
    std::stringstream ss(NAMES) ;    
    std::string name ; 
    while (!found && std::getline(ss, name)) if(!name.empty() && StartsWith(name.c_str(), qname)) found = true ;
    LOG(LEVEL) << " qname " << qname << " found " << found ; 
    return found ; 
}

const char* X4SolidMaker::NAMES = R"LITERAL(
Orb
SphereWithPhiSegment
SphereWithThetaSegment
AdditionAcrylicConstruction
XJfixtureConstruction
XJanchorConstruction
BoxMinusTubs0
BoxMinusTubs1
BoxMinusOrb
UnionOfHemiEllipsoids
PolyconeWithMultipleRmin
)LITERAL"; 

const G4VSolid* PolyconeWithMultipleRmin(const char* name);

const G4VSolid* X4SolidMaker::Make(const char* qname)  // static
{
    const G4VSolid* solid = nullptr ; 
    if(     StartsWith("Orb",qname))                          solid = X4SolidMaker::Orb(qname); 
    else if(StartsWith("SphereWithPhiSegment",qname))         solid = X4SolidMaker::SphereWithPhiSegment(qname); 
    else if(StartsWith("SphereWithThetaSegment",qname))       solid = X4SolidMaker::SphereWithThetaSegment(qname); 
    else if(StartsWith("AdditionAcrylicConstruction",qname))  solid = X4SolidMaker::AdditionAcrylicConstruction(qname); 
    else if(StartsWith("XJfixtureConstruction", qname))       solid = X4SolidMaker::XJfixtureConstruction(qname); 
    else if(StartsWith("XJanchorConstruction", qname))        solid = X4SolidMaker::XJanchorConstruction(qname) ; 
    else if(StartsWith("BoxMinusTubs0",qname))                solid = X4SolidMaker::BoxMinusTubs0(qname); 
    else if(StartsWith("BoxMinusTubs1",qname))                solid = X4SolidMaker::BoxMinusTubs1(qname); 
    else if(StartsWith("BoxMinusOrb",qname))                  solid = X4SolidMaker::BoxMinusOrb(qname); 
    else if(StartsWith("UnionOfHemiEllipsoids", qname))       solid = X4SolidMaker::UnionOfHemiEllipsoids(qname); 
    else if(StartsWith("PolyconeWithMultipleRmin", qname))    solid = X4SolidMaker::PolyconeWithMultipleRmin(qname) ; 

    assert(solid); 
    return solid ; 
}

const G4VSolid* X4SolidMaker::Orb(const char* name)  // static
{
    return new G4Orb(name, 100.) ; 
}

/**
X4SolidMaker::SphereWithPhiSegment
--------------------------------------

Best way to view phi segment is with XY cross section 

phi_start:0 phi_delta:2 
    full sphere in phi 

phi_start:0 phi_delta:0.5 
    cheese shape : suspect position of the cheese 
    may differ between Opticks and Geant4


**/

const G4VSolid* X4SolidMaker::SphereWithPhiSegment(const char* name)  // static
{
    double phi_start = SSys::getenvfloat("X4SolidMaker_SphereWithPhiSegment_phi_start", 0.f) ;  // units of pi
    double phi_delta = SSys::getenvfloat("X4SolidMaker_SphereWithPhiSegment_phi_delta", 0.5f) ; // units of pi

    LOG(info)
        << " (inputs are scaled by pi) "
        << " phi_start  " << phi_start
        << " phi_delta  " << phi_delta
        << " phi_start+phi_delta  " << phi_start+phi_delta
        << " phi_start*pi " << phi_start*pi
        << " phi_delta*pi " << phi_delta*pi
        << " (phi_start+phi_delta)*pi " << (phi_start+phi_delta)*pi
        ;

    G4String pName = name ; 
    G4double pRmin = 0. ; 
    G4double pRmax = 100. ; 
    G4double pSPhi = phi_start*pi ;    
    G4double pDPhi = phi_delta*pi ; 
    G4double pSTheta = 0. ; 
    G4double pDTheta = pi ;     // pi: full in theta

    return new G4Sphere(pName, pRmin, pRmax, pSPhi, pDPhi, pSTheta, pDTheta ); 
}






const G4VSolid* X4SolidMaker::SphereWithThetaSegment(const char* name)  // static
{
    double theta_start = SSys::getenvfloat("X4SolidMaker_SphereWithThetaSegment_theta_start", 0.f) ;  // units of pi
    double theta_delta = SSys::getenvfloat("X4SolidMaker_SphereWithThetaSegment_theta_delta", 0.5f) ; // units of pi

    LOG(info)
        << " (inputs are scaled by pi) "
        << " theta_start  " << theta_start
        << " theta_delta  " << theta_delta
        << " theta_start+theta_delta  " << theta_start+theta_delta
        << " theta_start*pi " << theta_start*pi
        << " theta_delta*pi " << theta_delta*pi
        << " (theta_start+theta_delta)*pi " << (theta_start+theta_delta)*pi
        ;

    G4String pName = name ; 
    G4double pRmin = 0. ; 
    G4double pRmax = 100. ; 
    G4double pSPhi = 0.*pi ;    
    G4double pDPhi = 2.*pi ; 
    G4double pSTheta = theta_start*pi ; 
    G4double pDTheta = theta_delta*pi ;  // theta_delta 1. : full in theta  

    return new G4Sphere(pName, pRmin, pRmax, pSPhi, pDPhi, pSTheta, pDTheta ); 
}


const G4VSolid* X4SolidMaker::AdditionAcrylicConstruction(const char* name)
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




const G4VSolid* X4SolidMaker::XJfixtureConstruction(const char* name)
{
    G4VSolid* solidXJfixture_down1;
    G4VSolid* solidXJfixture_down2;
    G4VSolid* solidXJfixture_down3;
    G4VSolid* solidXJfixture_down_uni1;
    G4VSolid* solidXJfixture_down_uni2;
    G4VSolid* solidXJfixture_down_uni3;
    G4VSolid* solidXJfixture_down_uni4;
    //G4VSolid* solidXJfixture_down_uni5;

    G4VSolid* solidXJfixture_up1;
    G4VSolid* solidXJfixture_up2;
    G4VSolid* solidXJfixture_up_uni;

    G4VSolid* solidXJfixture;


// fixture part
    solidXJfixture_down1 = new G4Tubs("solidXJfixture_down1", 25.*mm, 45.*mm, 13./2*mm, 0.*deg, 360.*deg);
    solidXJfixture_down2 = new G4Box("solidXJfixture_down2", 10.*mm, 11.5*mm, 13/2.*mm);
    solidXJfixture_down_uni1 = new G4UnionSolid("solidXJfixture_down_uni1", solidXJfixture_down1, solidXJfixture_down2, 0, G4ThreeVector(52.*mm, 0.*mm, 0.*mm));
    solidXJfixture_down_uni2 = new G4UnionSolid("solidXJfixture_down_uni2", solidXJfixture_down_uni1, solidXJfixture_down2, 0, G4ThreeVector(-52.*mm, 0.*mm, 0.*mm));
    solidXJfixture_down3 = new G4Box("solidXJfixture_down3", 15.*mm, 15.*mm, 13/2.*mm);
    solidXJfixture_down_uni3 = new G4UnionSolid("solidXJfixture_down_uni3", solidXJfixture_down_uni2, solidXJfixture_down3, 0, G4ThreeVector(0.*mm, 50.*mm, 0.*mm));
    solidXJfixture_down_uni4 = new G4UnionSolid("solidXJfixture_down_uni4", solidXJfixture_down_uni3, solidXJfixture_down3, 0, G4ThreeVector(0.*mm, -50.*mm, 0.*mm));

// cover part
    solidXJfixture_up1 = new G4Box("solidXJfixture_up1", 15.*mm, 40.*mm, 17/2.*mm);    
    solidXJfixture_up2 = new G4Box("solidXJfixture_up2", 15.*mm, 65*mm, 5.*mm);
    solidXJfixture_up_uni = new G4UnionSolid("solidXJfixture_up_uni", solidXJfixture_up1, solidXJfixture_up2, 0, G4ThreeVector(0.*mm, 0.*mm, 13.5*mm));

// union 
    solidXJfixture = new G4UnionSolid("solidXJfixture", solidXJfixture_down_uni4, solidXJfixture_up_uni, 0, G4ThreeVector(0.*mm, 0.*mm, -25.*mm));

    return solidXJfixture ;
}



const G4VSolid* X4SolidMaker::XJanchorConstruction(const char* name)
{
    G4VSolid* solidXJanchor_up;
    G4VSolid* solidXJanchor_down;
    G4VSolid* solidXJanchor_ball;

    solidXJanchor_up   = new G4Tubs("solidXJanchor_up", 0.*mm, 25.*mm, 13./2*mm, 0.*deg, 360.*deg);
    solidXJanchor_down = new G4Cons("solidXJanchor_down", 0.*mm, 47.*mm, 0.*mm, 73.*mm, 10.*mm, 0.*deg, 360.*deg);   // to subtract the ball
    solidXJanchor_ball = new G4Sphere("solidXJanchor_ball", 0.*mm, 17820.*mm, 0.*deg, 360.*deg, 0.*deg, 180.*deg); 
    G4SubtractionSolid* solidXJanchor_sub = new G4SubtractionSolid("solidXJanchor_sub",solidXJanchor_down, solidXJanchor_ball, 0, G4ThreeVector(0.*mm, 0*mm,  17820.*mm));
    G4UnionSolid* solidXJanchor = new G4UnionSolid("solidXJanchor",solidXJanchor_sub, solidXJanchor_up, 0, G4ThreeVector(0.*mm, 0*mm,-16.5*mm));

    return solidXJanchor ;
} 



const G4VSolid* X4SolidMaker::BoxMinusTubs0(const char* name)  // is afflicted
{
    double tubs_hz = 15.2*mm ;   
    double zshift = 0*mm ; 
    G4VSolid* box   = new G4Box("box",  250*mm, 250*mm, 100*mm ); 
    G4VSolid* tubs =  new G4Tubs("tubs",120*mm,208*mm,tubs_hz,0.0*deg,360.0*deg);
    G4VSolid* box_minus_tubs = new G4SubtractionSolid(name,box,tubs,0,G4ThreeVector(0.*mm,0.*mm,zshift));  
    return box_minus_tubs ; 
}

const G4VSolid* X4SolidMaker::BoxMinusTubs1(const char* name) 
{
    double tubs_hz = 15.2*mm ;   
    G4VSolid* box   = new G4Box("box",  250*mm, 250*mm, 100*mm ); 
    G4VSolid* tubs =  new G4Tubs("tubs",120*mm,208*mm,tubs_hz,0.0*deg,360.0*deg);
    G4VSolid* box_minus_tubs = new G4SubtractionSolid(name,box,tubs);  
    return box_minus_tubs ; 
}

const G4VSolid* X4SolidMaker::BoxMinusOrb(const char* name) 
{
    double radius = SSys::getenvfloat("X4SolidMaker_BoxMinusOrb_radius", 100.f) ; 

    double sx     = SSys::getenvfloat("X4SolidMaker_BoxMinusOrb_sx", 100.f) ; 
    double sy     = SSys::getenvfloat("X4SolidMaker_BoxMinusOrb_sy", 100.f) ; 
    double sz     = SSys::getenvfloat("X4SolidMaker_BoxMinusOrb_sz", 100.f) ; 

    double dx     = SSys::getenvfloat("X4SolidMaker_BoxMinusOrb_dx", 0.f) ;  
    double dy     = SSys::getenvfloat("X4SolidMaker_BoxMinusOrb_dy", 0.f) ;  
    double dz     = SSys::getenvfloat("X4SolidMaker_BoxMinusOrb_dz", 0.f) ;  

    G4VSolid* box = new G4Box("box",  sx, sy, sz ); 
    G4VSolid* orb = new G4Orb("orb",  radius ); 

    G4VSolid* box_minus_orb = new G4SubtractionSolid(name,box,orb,nullptr, G4ThreeVector(dx, dy, dz) );  
    return box_minus_orb ; 
}


const G4VSolid* X4SolidMaker::PolyconeWithMultipleRmin(const char* name)
{ 
    double ZUpper4[4];
    double RminUpper4[4];
    double RmaxUpper4[4];
    ZUpper4[0] = 0*mm;   RminUpper4[0] = 43.*mm; RmaxUpper4[0] = 195.*mm;
    ZUpper4[1] = -15*mm; RminUpper4[1] = 43.*mm; RmaxUpper4[1] = 195.*mm;
    ZUpper4[2] = -15*mm; RminUpper4[2] = 55.5*mm; RmaxUpper4[2] = 70.*mm;
    ZUpper4[3] = -101*mm; RminUpper4[3] = 55.5*mm; RmaxUpper4[3] = 70.*mm;
    
    G4VSolid* base_steel = new G4Polycone("base_steel",0.0*deg,360.0*deg,4,ZUpper4,RminUpper4,RmaxUpper4);
    return base_steel ;
}


const G4VSolid* X4SolidMaker::UnionOfHemiEllipsoids(const char* name )
{
    assert( strstr( name, "UnionOfHemiEllipsoids" ) != nullptr ); 

    std::vector<long> vals ; 
    Extract(vals, name); 
    long iz = vals.size() > 0 ? vals[0] : 0 ; 

    std::cout 
        << "X4SolidMaker::UnionOfHemiEllipsoids"
        << " name " << name 
        << " vals.size " << vals.size()
        << " iz " << iz
        << std::endl 
        ; 


    double rx = 150. ; 
    double ry = 150. ; 
    double rz = 100. ; 

    double z2 = rz ; 
    double z1 = 0. ; 
    double z0 = -rz ; 

    G4VSolid* upper = new G4Ellipsoid("upper", rx, ry, rz, z1, z2 );  
    G4VSolid* lower = new G4Ellipsoid("lower", rx, ry, rz, z0, z1 );  

    G4VSolid* solid = nullptr ; 
    if( iz == 0 )
    {
        solid = new G4UnionSolid(name, upper, lower );   
    }
    else
    {
        double zoffset = double(iz) ; 
        G4ThreeVector tlate(0., 0., zoffset) ; 
        solid = new G4UnionSolid(name, upper, lower, nullptr, tlate );   
    }
    return solid ; 
}


void X4SolidMaker::Extract( std::vector<long>& vals, const char* s )  // static
{
    char* s0 = strdup(s); 
    char* p = s0 ; 
    while (*p) 
    {   
        if( (*p >= '0' && *p <= '9') || *p == '+' || *p == '-') vals.push_back(strtol(p, &p, 10)) ; 
        else p++ ;
    }   
    free(s0); 
}




