#include <cstring>
#include <csignal>
#include "sstr.h"
#include "ssys.h"
#include "s_bb.h"
#include "NP.hh"

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
#include "G4IntersectionSolid.hh"
#include "G4Ellipsoid.hh"
#include "G4MultiUnion.hh"
#include "G4CutTubs.hh"
#include "G4Torus.hh"

using CLHEP::pi ;
using CLHEP::mm ;

#include "U4SolidMaker.hh"
#include "U4SolidTree.hh"
#include "SLOG.hh"

const plog::Severity U4SolidMaker::LEVEL = SLOG::EnvLevel("U4SolidMaker", "DEBUG");

const char* U4SolidMaker::NAMES = R"LITERAL(
JustOrbOrbUnion
JustOrbOrbIntersection
JustOrbOrbDifference
JustOrb
ThreeOrbUnion
SphereIntersectBox
SphereWithPhiSegment
SphereWithPhiCutDEV
GeneralSphereDEV
SphereWithThetaSegment
AdditionAcrylicConstruction
XJfixtureConstruction
AltXJfixtureConstruction
AltXJfixtureConstructionU
XJanchorConstruction
SJReceiverConstruction
BoxMinusTubs0
BoxMinusTubs1
BoxMinusOrb
UnionOfHemiEllipsoids
PolyconeWithMultipleRmin
AnnulusBoxUnion
AnnulusTwoBoxUnion
AnnulusTwoBoxUnionContiguous
AnnulusOtherTwoBoxUnion
AnnulusCrossTwoBoxUnion
AnnulusFourBoxUnion
CylinderFourBoxUnion
BoxFourBoxUnion
BoxCrossTwoBoxUnion
BoxThreeBoxUnion
OrbOrbMultiUnion
OrbGridMultiUnion
BoxGridMultiUnion
BoxFourBoxContiguous
LHCbRichSphMirr
LHCbRichFlatMirr
LocalFastenerAcrylicConstruction
AltLocalFastenerAcrylicConstruction
AnotherLocalFastenerAcrylicConstruction
WaterDistributer
AltWaterDistributer
WaterDistributorPartIIIUnion
OuterReflectorOrbSubtraction
R12860_PMTSolid
)LITERAL";


const char* U4SolidMaker::Name( const char* prefix, unsigned idx ) // static
{
    std::stringstream ss ;
    ss << prefix << idx ;
    std::string s = ss.str();
    return strdup(s.c_str()) ;
}


/**
U4SolidMaker::PrimitiveClone
-----------------------------

Create G4VSolid using copy ctor of appropriate EntityType

**/

G4VSolid* U4SolidMaker::PrimitiveClone( const G4VSolid* src, const char* prefix, unsigned idx) // static
{
    const char* name = Name(prefix, idx );
    G4VSolid* clone = U4SolidTree::PrimitiveClone(src, name);
    return clone ;
}

bool U4SolidMaker::StartsWith( const char* n, const char* q ) // static
{
    return strlen(q) >= strlen(n) && strncmp(q, n, strlen(n)) == 0 ;
}

bool U4SolidMaker::CanMake(const char* qname) // static
{
    bool found = false ;
    std::stringstream ss(NAMES) ;
    std::string name ;
    while (!found && std::getline(ss, name)) if(!name.empty() && StartsWith(name.c_str(), qname)) found = true ;
    LOG(LEVEL) << " qname " << qname << " found " << found ;
    return found ;
}

const G4VSolid* PolyconeWithMultipleRmin(const char* name);


const G4VSolid* U4SolidMaker::Make(const char* qname)  // static
{
    std::string meta ;
    return U4SolidMaker::Make(qname, meta);
}
const G4VSolid* U4SolidMaker::Make(const char* qname, std::string& meta )  // static
{
    if(strcmp(qname, "NAMES") == 0 )
    {
        std::cout << NAMES ;
        return nullptr ;
    }

    const G4VSolid* solid = nullptr ;
    if(     StartsWith("JustOrbOrbUnion",qname))              solid = JustOrbOrbUnion(qname);
    else if(StartsWith("JustOrbOrbIntersection",qname))       solid = JustOrbOrbIntersection(qname);
    else if(StartsWith("JustOrbOrbDifference",qname))         solid = JustOrbOrbDifference(qname);
    else if(StartsWith("JustOrb",qname))                      solid = JustOrb(qname);
    else if(StartsWith("ThreeOrbUnion",qname))                solid = ThreeOrbUnion(qname);
    else if(StartsWith("SphereWithPhiSegment",qname))         solid = SphereWithPhiSegment(qname);
    else if(StartsWith("SphereWithPhiCutDEV",qname))          solid = SphereWithPhiCutDEV(qname);
    else if(StartsWith("GeneralSphereDEV",qname))             solid = GeneralSphereDEV(qname, meta);
    else if(StartsWith("SphereWithThetaSegment",qname))       solid = SphereWithThetaSegment(qname);
    else if(StartsWith("AdditionAcrylicConstruction",qname))  solid = AdditionAcrylicConstruction(qname);
    else if(StartsWith("XJfixtureConstruction", qname))       solid = XJfixtureConstruction(qname);
    else if(StartsWith("AltXJfixtureConstructionU", qname))   solid = AltXJfixtureConstructionU(qname);
    else if(StartsWith("AltXJfixtureConstruction", qname))    solid = AltXJfixtureConstruction(qname);
    else if(StartsWith("XJanchorConstruction", qname))        solid = XJanchorConstruction(qname) ;
    else if(StartsWith("SJReceiverConstruction", qname))      solid = SJReceiverConstruction(qname) ;
    else if(StartsWith("BoxMinusTubs0",qname))                solid = BoxMinusTubs0(qname);
    else if(StartsWith("BoxMinusTubs1",qname))                solid = BoxMinusTubs1(qname);
    else if(StartsWith("BoxMinusOrb",qname))                  solid = BoxMinusOrb(qname);
    else if(StartsWith("UnionOfHemiEllipsoids", qname))       solid = UnionOfHemiEllipsoids(qname);
    else if(StartsWith("PolyconeWithMultipleRmin", qname))    solid = PolyconeWithMultipleRmin(qname) ;
    else if(StartsWith("AnnulusBoxUnion", qname))             solid = AnnulusBoxUnion(qname) ;
    else if(StartsWith("AnnulusTwoBoxUnion", qname))          solid = AnnulusTwoBoxUnion(qname) ;
    else if(StartsWith("AnnulusOtherTwoBoxUnion", qname))     solid = AnnulusOtherTwoBoxUnion(qname) ;
    else if(StartsWith("AnnulusCrossTwoBoxUnion", qname))     solid = AnnulusCrossTwoBoxUnion(qname) ;
    else if(StartsWith("AnnulusFourBoxUnion", qname))         solid = AnnulusFourBoxUnion(qname) ;
    else if(StartsWith("CylinderFourBoxUnion", qname))        solid = CylinderFourBoxUnion(qname) ;
    else if(StartsWith("BoxFourBoxUnion", qname))             solid = BoxFourBoxUnion(qname) ;
    else if(StartsWith("BoxCrossTwoBoxUnion", qname))         solid = BoxCrossTwoBoxUnion(qname) ;
    else if(StartsWith("BoxThreeBoxUnion", qname))            solid = BoxThreeBoxUnion(qname) ;
    else if(StartsWith("OrbOrbMultiUnion", qname))            solid = OrbOrbMultiUnion(qname) ;
    else if(StartsWith("OrbGridMultiUnion", qname))           solid = OrbGridMultiUnion(qname) ;
    else if(StartsWith("BoxGridMultiUnion", qname))           solid = BoxGridMultiUnion(qname) ;
    else if(StartsWith("BoxFourBoxContiguous", qname))        solid = BoxFourBoxContiguous(qname) ;
    else if(StartsWith("LHCbRichSphMirr", qname))             solid = LHCbRichSphMirr(qname) ;
    else if(StartsWith("LHCbRichFlatMirr", qname))            solid = LHCbRichFlatMirr(qname) ;
    else if(StartsWith("SphereIntersectBox", qname))          solid = SphereIntersectBox(qname) ;
    else if(StartsWith("LocalFastenerAcrylicConstruction", qname)) solid = LocalFastenerAcrylicConstruction(qname) ;
    else if(StartsWith("AltLocalFastenerAcrylicConstruction", qname)) solid = AltLocalFastenerAcrylicConstruction(qname) ;
    else if(StartsWith("BltLocalFastenerAcrylicConstruction", qname)) solid = BltLocalFastenerAcrylicConstruction(qname) ;
    else if(StartsWith("WaterDistributer", qname))                    solid = WaterDistributer(qname) ;
    else if(StartsWith("AltWaterDistributer", qname))                 solid = AltWaterDistributer(qname) ;
    else if(StartsWith("WaterDistributorPartIIIUnion", qname))        solid = WaterDistributorPartIIIUnion(qname);
    else if(StartsWith("OuterReflectorOrbSubtraction", qname))        solid = OuterReflectorOrbSubtraction(qname);
    else if(StartsWith("R12860_PMTSolid", qname))                     solid = R12860_PMTSolid(qname);
    LOG(LEVEL) << " qname " << qname << " solid " << solid ;
    LOG_IF(error, solid==nullptr) << " Failed to create solid for qname " << qname << " CHECK U4SolidMaker::Make " ;
    return solid ;
}






const G4VSolid* U4SolidMaker::JustOrb(const char* name)  // static
{
    return new G4Orb(name, 100.) ;
}


const G4VSolid* U4SolidMaker::JustOrbOrbUnion(       const char* name){ return JustOrbOrb_(name, 'U') ; }
const G4VSolid* U4SolidMaker::JustOrbOrbIntersection(const char* name){ return JustOrbOrb_(name, 'I') ; }
const G4VSolid* U4SolidMaker::JustOrbOrbDifference(  const char* name){ return JustOrbOrb_(name, 'D') ; }

const G4VSolid* U4SolidMaker::JustOrbOrb_(const char* name, char op)  // static
{
    G4Orb* l = new G4Orb("l", 100.) ;
    G4Orb* r = new G4Orb("r", 100.) ;
    G4VSolid* oo  = nullptr ;
    switch(op)
    {
         case 'U': oo = new G4UnionSolid(       name, l, r,  0, G4ThreeVector(0.*mm, 50.*mm, 0.*mm ) ) ; break ;
         case 'I': oo = new G4IntersectionSolid(name, l, r,  0, G4ThreeVector(0.*mm, 50.*mm, 0.*mm ) ) ; break ;
         case 'D': oo = new G4SubtractionSolid( name, l, r,  0, G4ThreeVector(0.*mm, 50.*mm, 0.*mm ) ) ; break ;
    }
    return oo ;
}

/**
U4SolidMaker::SphereWithPhiCutDEV
--------------------------------------
**/

const G4VSolid* U4SolidMaker::SphereWithPhiCutDEV(const char* name)  // static
{
    double radius    = ssys::getenvfloat("U4SolidMaker_SphereWithPhiCutDEV_radius", 20.f) ;    // mm
    double phi_start = ssys::getenvfloat("U4SolidMaker_SphereWithPhiCutDEV_phi_start", 0.f) ;  // units of pi
    double phi_delta = ssys::getenvfloat("U4SolidMaker_SphereWithPhiCutDEV_phi_delta", 0.5f) ; // units of pi

    G4String pName = name ;
    G4double pRmin = 0. ;
    G4double pRmax = radius ;
    G4double pSPhi = phi_start*pi ;
    G4double pDPhi = phi_delta*pi ;
    G4double pSTheta = 0. ;
    G4double pDTheta = pi ;     // pi: full in theta

    return new G4Sphere(pName, pRmin, pRmax, pSPhi, pDPhi, pSTheta, pDTheta );
}




const G4VSolid* U4SolidMaker::GeneralSphereDEV(const char* name, std::string& meta )  // static
{
    const char* radiusMode = ssys::getenvvar("U4SolidMaker_GeneralSphereDEV_radiusMode");
    double innerRadius = ssys::getenvfloat("U4SolidMaker_GeneralSphereDEV_innerRadius", 50.f) ;    // mm
    double outerRadius = ssys::getenvfloat("U4SolidMaker_GeneralSphereDEV_outerRadius", 100.f) ;    // mm

    // The two azimuthal angles from the phi cut should be in range 0. to 2. (in units of pi) use XY projection to visualize
    const char* phiMode = ssys::getenvvar("U4SolidMaker_GeneralSphereDEV_phiMode");
    double phiStart    = ssys::getenvfloat("U4SolidMaker_GeneralSphereDEV_phiStart",    0.f) ;
    double phiDelta    = ssys::getenvfloat("U4SolidMaker_GeneralSphereDEV_phiDelta",    2.0f) ;

    // The two polar angles from the theta cut should be in range 0. to 1. (in units of pi) use XZ or YZ projections to visualize
    const char* thetaMode = ssys::getenvvar("U4SolidMaker_GeneralSphereDEV_thetaMode");
    double thetaStart  = ssys::getenvfloat("U4SolidMaker_GeneralSphereDEV_thetaStart",  0.f) ;
    double thetaDelta  = ssys::getenvfloat("U4SolidMaker_GeneralSphereDEV_thetaDelta",  0.5f) ;


    NP::SetMeta<std::string>( meta, "creator", "U4SolidMaker::GeneralSphereDEV" );
    NP::SetMeta<std::string>( meta, "name",    name );

    NP::SetMeta<std::string>( meta, "radiusMode", radiusMode );
    NP::SetMeta<float>( meta, "innerRadius", float(innerRadius) );
    NP::SetMeta<float>( meta, "outerRadius", float(outerRadius) );

    NP::SetMeta<std::string>( meta, "phiMode", phiMode );
    NP::SetMeta<float>( meta, "phiStart", float(phiStart) );
    NP::SetMeta<float>( meta, "phiDelta", float(phiDelta) );

    NP::SetMeta<std::string>( meta, "thetaMode", thetaMode );
    NP::SetMeta<float>( meta, "thetaStart", float(thetaStart) );
    NP::SetMeta<float>( meta, "thetaDelta", float(thetaDelta) );


    G4String pName = name ;
    G4double pRmin = innerRadius*mm ;
    G4double pRmax = outerRadius*mm ;
    G4double pSPhi = phiStart*pi ;
    G4double pDPhi = phiDelta*pi ;
    G4double pSTheta = thetaStart*pi ;
    G4double pDTheta = thetaDelta*pi ;

    return new G4Sphere(pName, pRmin, pRmax, pSPhi, pDPhi, pSTheta, pDTheta );
}






/**
U4SolidMaker::SphereWithPhiSegment
--------------------------------------

Best way to view phi segment is with XY cross section

phi_start:0 phi_delta:2
    full sphere in phi

phi_start:0 phi_delta:0.5
    cheese shape : suspect position of the cheese
    may differ between Opticks and Geant4


**/

const G4VSolid* U4SolidMaker::SphereWithPhiSegment(const char* name)  // static
{
    double phi_start = ssys::getenvfloat("U4SolidMaker_SphereWithPhiSegment_phi_start", 0.f) ;  // units of pi
    double phi_delta = ssys::getenvfloat("U4SolidMaker_SphereWithPhiSegment_phi_delta", 0.5f) ; // units of pi

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






const G4VSolid* U4SolidMaker::SphereWithThetaSegment(const char* name)  // static
{
    double theta_start = ssys::getenvfloat("U4SolidMaker_SphereWithThetaSegment_theta_start", 0.f) ;  // units of pi
    double theta_delta = ssys::getenvfloat("U4SolidMaker_SphereWithThetaSegment_theta_delta", 0.5f) ; // units of pi

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


const G4VSolid* U4SolidMaker::AdditionAcrylicConstruction(const char* name)
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

/**
U4SolidMaker::ThreeOrbUnion : Checking the cloning of a boolean tree with two levels of transforms
----------------------------------------------------------------------------------------------------

ThreeOrbUnion0: left-hand form does NOT combine transforms

* this way the transforms are both associated with leaf nodes

       abc
       /  \
      /   (c)  <- transform on leaf
     ab
    /  \
   a   (b)  <- transform on leaf

* all right hand sides are leaf, so no double level of transforms


ThreeOrbUnion1: right-hand form DOES combine two transforms : and trips assert in U4SolidTree::BooleanClone

      cab
    /    \
   c     (ab)   <- transform on operator node
          / \
         a  (b)  <- transform on leaf

* happens because a boolean combination is on the right hand side of another boolean combination


The righthand edge of 0 is at 120 and 1 is at 130.
They are different because of the combination of the transforms for 1::

    XX=120 GEOM=ThreeOrbUnion0_XZ ./xxs.sh
    XX=130 GEOM=ThreeOrbUnion1_XZ ./xxs.sh

**/

const G4VSolid* U4SolidMaker::ThreeOrbUnion(const char* name)
{
    long mode = sstr::ExtractLong(name, 0);
    LOG(LEVEL) << " mode " << mode ;

    G4Orb* a = new G4Orb("a", 100. );
    G4Orb* b = new G4Orb("b", 100. );
    G4Orb* c = new G4Orb("c", 100. );

    const G4VSolid* solid = nullptr ;
    if( mode == 0 )
    {
        G4UnionSolid* ab = new G4UnionSolid( "ab", a, b, 0, G4ThreeVector( 10., 0., 0. ) );
        G4UnionSolid* ab_c = new G4UnionSolid( "ab_c", ab, c, 0, G4ThreeVector( 20., 0., 0. ) );
        solid = ab_c ;
    }
    else if( mode == 1 )
    {
        G4UnionSolid* ab = new G4UnionSolid( "ab", a, b, 0, G4ThreeVector( 10., 0., 0. ) );
        G4UnionSolid* c_ab = new G4UnionSolid( "c_ab", c, ab, 0, G4ThreeVector( 20., 0., 0. ) );
        solid = c_ab ;
    }
    return solid ;
}

/**
U4SolidMaker::XJfixtureConstruction
-------------------------------------

solidXJfixture             Union
   down_uni4               Union
      down_uni3            Union
         down_uni2         Union
             down_uni1     Union
                down1      Tubs
                down2      Box       52.*mm, 0.*mm, 0.*mm
             down2         Box      -52.*mm, 0.*mm, 0.*mm
         down3             Box       0.*mm,  50.*mm, 0.*mm
      down3                Box       0.*mm, -50.*mm, 0.*mm
   up_uni                  Union
      up1                  Box
      up2                  Box       0.*mm, 0.*mm, 13.5*mm     (Up in Z)



         up2 is raised by 13.5 to form the thinner in z table top of the altar


                Spurious vertical at 35          In Y box is at 50 +- 15
                                                           35    50     65
                                                            :
                                                            :     :     :                                altar frame              fixture frame
                                                            :
             -------------+                             +---+---+-+-----+        - - - - - - - - - - - - 18.5+13  =   31.5             6.5     - - - - - -
             |            |                             |   :   |      13/2=6.5
             +            +                             +   :   + :     :         - - - - - - - - - - -  18.5+6.5 =   25               0.0
             |            |                             |   :   |       :
             +------------+----------------+-----25-----+---20--+-+-----+         - - - - - - - - - -      8.5+10 =  18.5              -6.5       13+10 = 23
             |                                                          5
             +    up2                      +                            +       - - - - - - - - - - - - -   8.5+5  = 13.5              -11.5
             |                                                          5
             +---------+^^^^^^^^^^^^^^^^^^^+^^^^^^^^^^^^^^^^^^+---------+       - - - - - - - - - - - - -             8.5              -16.5    - - - - -
                       |                                      |
                       |                                     17/2=8.5
                       +  up1                                 +                - - - - - - - - - - - - -              0.0              -25.0
                       |                                      |
                       |                                      |
                       +-------------------+-------40---------+            - - - - - - - - - - - - - - - -           -8.5              -33.5

                                           |            |    :   |
                                           0            25   35  45
                                                        |    :   |
                                                        |    :   |
                                                        |    :   outer tubs
                                                        |    :
                                                        |    spurious vertical from box edge (why? it is within the tubs ring)
                                                        |
                                                        inner tubs


               Z
               |
               +-- Y
              /
             X


       Then altar is offset by -25. pushing its top down to 18.5 - 25. = -6.5 in final frame
       which is flush with the lower edge of the celtic cross



**/


/**
U4SolidMaker::AltXJfixtureConstruction
----------------------------------------

Contract this with XJfixtureConstruction : the shape is very nearly the
same but this uses only 3 boxes and 2 tubs rather than 6 boxes and 2 tubs.


                        :      65     :
                        : 11.5        :
        +-----------+---+---+---------+
        |           + x - --+ 13      | 23/2    - - - - -
        |           +---+---+         |            (23-13)/2 = 10/2
        + - - - - - - - | - - - - - - +         - - - - -
        |    u          |             | 23/2
        |               |             |
        +-----+^^^^^^^^^|^^^^^^^^+----+            (23+17)/2  = 40/2
              |  l      |        | 17/2
              + - - - - | - - - -+              ----
              |         |        | 17/2
              +---------+---45---+

       Z
       |
       |
       +---> Y
      /
     X
                                  ulxoi
                      ulxo                    i
             ulx              o
         ul        x
       u    l

**/

const G4VSolid* U4SolidMaker::AltXJfixtureConstruction(const char* name)
{
    return AltXJfixtureConstruction_(name, "");
}
const G4VSolid* U4SolidMaker::AltXJfixtureConstructionU(const char* name)
{
    return AltXJfixtureConstruction_(name, "ulxo");
}

const G4VSolid* U4SolidMaker::AltXJfixtureConstruction_(const char* name, const char* opt)
{
    G4VSolid* u ;
    G4VSolid* l ;
    G4VSolid* x ;
    G4VSolid* o ;
    G4VSolid* i ;

    G4VSolid* ul ;
    G4VSolid* ulx ;
    G4VSolid* ulxo ;
    G4VSolid* ulxoi ;

    // Y is the long (left-right) dimension
    G4double l_uncoincide = 1. ;

    u = new G4Box("u", 15.*mm, 65.*mm,  23/2.*mm);
    l = new G4Box("l", 15.*mm, 40.*mm,  (17+l_uncoincide)/2.*mm);    // increase lbox in half_z by  lbox_uncoincide/2.
    ul = new G4UnionSolid("ul", u, l, 0, G4ThreeVector(0.*mm, 0.*mm, (-40.+l_uncoincide)/2*mm ) ) ;

    G4double zs = 10/2.*mm ;
    x = new G4Box("x", 62.*mm, 11.5*mm, 13/2.*mm);
    ulx = new G4UnionSolid("ulx", ul, x, 0, G4ThreeVector(0.*mm, 0.*mm, zs )) ;

    o = new G4Tubs("o", 0.*mm, 45.*mm, 13./2*mm, 0.*deg, 360.*deg);
    ulxo = new G4UnionSolid("ulxo_CSG_CONTIGUOUS", ulx, o, 0, G4ThreeVector(0.*mm, 0.*mm, zs )) ;
    if( strstr(opt, "ulxo"))  return ulxo ;

    G4double i_uncoincide = 1. ;
    // increase the half_z of subtracted tubs : avoiding upper coincident face
    // and raise by the same to keep low edge sub the same : hmm that leaves a thin cross piece from the base of x
    // perhaps better to subtract more and get rid of that ?

    i = new G4Tubs("i", 0.*mm, 25.*mm, 13./2*mm + i_uncoincide/2.*mm , 0.*deg, 360.*deg);
    ulxoi = new G4SubtractionSolid("ulxoi", ulxo, i, 0, G4ThreeVector(0.*mm, 0.*mm, zs+i_uncoincide/2.*mm  )) ;

    //return ul ;
    //return ulx ;
    //return ulxo ;
    return ulxoi ;
}
const int U4SolidMaker::XJfixtureConstruction_debug_mode = ssys::getenvint("U4SolidMaker__XJfixtureConstruction_debug_mode", 0 ) ;

/**
U4SolidMaker::XJfixtureConstruction
------------------------------------


**/

const G4VSolid* U4SolidMaker::XJfixtureConstruction(const char* name)
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

    // down_uni4 is celtic-cross shape or uniform z half-thickness 13./2. = 6.5 mm  (shifts in x and y,  not z)

// cover part  : Y is the long dimension
    solidXJfixture_up1 = new G4Box("solidXJfixture_up1", 15.*mm, 40.*mm, 17/2.*mm);
    solidXJfixture_up2 = new G4Box("solidXJfixture_up2", 15.*mm, 65*mm, 5.*mm);
    solidXJfixture_up_uni = new G4UnionSolid("solidXJfixture_up_uni", solidXJfixture_up1, solidXJfixture_up2, 0, G4ThreeVector(0.*mm, 0.*mm, 13.5*mm));

    //G4VSolid* new_solidXJfixture_up_uni = Uncoincide_Box_Box_Union( solidXJfixture_up_uni );
    //solidXJfixture_up_uni = new_solidXJfixture_up_uni ;

    solidXJfixture = new G4UnionSolid("solidXJfixture", solidXJfixture_down_uni4, solidXJfixture_up_uni, 0, G4ThreeVector(0.*mm, 0.*mm, -25.*mm));


    // twiddling puts the zero at the altar frame zero
    // so would have to offset the placement

    G4VSolid* solidXJfixture_twiddle = new G4UnionSolid("solidXJfixture_twiddle", solidXJfixture_up_uni, solidXJfixture_down_uni4, 0, G4ThreeVector(0.*mm, 0.*mm, 25.*mm));
    G4VSolid* celtic_cross_sub_altar = new G4SubtractionSolid("solidXJfixture_celtic_cross_sub_altar", solidXJfixture_down_uni4, solidXJfixture_up_uni, 0, G4ThreeVector(0.*mm, 0.*mm, -25.*mm));
    G4VSolid* solidXJfixture_split = new G4UnionSolid("solidXJfixture_split", solidXJfixture_down_uni4, solidXJfixture_up_uni, 0, G4ThreeVector(0.*mm, 0.*mm, -50.*mm));


    G4VSolid* solid = solidXJfixture ;

    int debug_mode = XJfixtureConstruction_debug_mode ;
    if( debug_mode > 0 )
    {
        switch(debug_mode)
        {
           case  0: solid = solidXJfixture           ; break ;
           case  1: solid = solidXJfixture_down1     ; break ;
           case  2: solid = solidXJfixture_down2     ; break ;
           case  3: solid = solidXJfixture_down_uni1 ; break ;
           case  4: solid = solidXJfixture_down_uni2 ; break ;
           case  5: solid = solidXJfixture_down3     ; break ;
           case  6: solid = solidXJfixture_down_uni3 ; break ;
           case  7: solid = solidXJfixture_down_uni4 ; break ;
           case  8: solid = solidXJfixture_up1       ; break ;
           case  9: solid = solidXJfixture_up2       ; break ;
           case 10: solid = solidXJfixture_up_uni    ; break ;
           case 11: solid = celtic_cross_sub_altar   ; break ;
           case 12: solid = solidXJfixture_split     ; break ;
           case 13: solid = solidXJfixture_twiddle   ; break ;
        }
        LOG(info)
            << "U4SolidMaker__XJfixtureConstruction_debug_mode " << debug_mode
            << " solid.GetName " << ( solid ? solid->GetName() : "-" )
            ;
        assert(solid);
    }
    return solid ;
}

const G4VSolid* U4SolidMaker::AnnulusBoxUnion(const char* name)
{
    // do not see spurious intersects
    G4VSolid* down1  = new G4Tubs("down1", 25.*mm, 45.*mm, 13./2*mm, 0.*deg, 360.*deg);
    G4VSolid* down3 = new G4Box("down3", 15.*mm, 15.*mm, 13/2.*mm);
    G4VSolid* uni13 = new G4UnionSolid("uni13", down1, down3, 0, G4ThreeVector(0.*mm, 50.*mm, 0.*mm));
    return uni13 ;
}

/**
U4SolidMaker::AnnulusTwoBoxUnion
-----------------------------------

YZ view looks symmetric as both sides of tubs have box-extensions::

                                               35       50       65
      -65      -45        -25             25    :   45  :        :
       +---------+---------+       +       +---------+--:--------+
       |         |         |               |         |           |
       |         |         |               |         |           |
       |         |         |               |         |           |
       +---------+---------+               +---------+-----------+

    Z
    |
    +-- Y
   /
  X


U4SolidMaker::AnnulusTwoBoxUnion


               tub_bpy_bny  T
                             \
     tub_bpy T                bny
              \
  tub          bpy



For hinting as CSG_CONTIGUOUS or CSG_DISCONTIGUOUS to work
requires setting the inner to zero to avoid the CSG_DIFFERENCE.

**/


const G4VSolid* U4SolidMaker::AnnulusTwoBoxUnion(const char* name)
{
    const char* suffix = nullptr ;
    if(     strstr(name, "Contiguous"))    suffix = "_CSG_CONTIGUOUS" ;
    else if(strstr(name, "Discontiguous")) suffix = "_CSG_DISCONTIGUOUS" ;

    bool toplist = strstr(name, "List") != nullptr ;

    const char* listname = sstr::Concat("tub_bpy_bny", suffix );

    double innerRadius = 0.*mm ;
    double bpy_scale_z = 2. ;
    double bny_scale_z = 4. ;

    LOG(LEVEL)
        << " name " << name
        << " suffix " << suffix
        << " listname " << listname
        << " innerRadius " << innerRadius
        << " bpy_scale_z " << bpy_scale_z
        << " bny_scale_z " << bny_scale_z
        ;

    G4VSolid* tub  = new G4Tubs("tub", 0.*mm, 45.*mm, 13./2*mm, 0.*deg, 360.*deg);

    G4VSolid* bpy = new G4Box("bpy", 15.*mm, 15.*mm, bpy_scale_z*13/2.*mm);
    G4VSolid* tub_bpy = new G4UnionSolid(  "tub_bpy", tub, bpy, 0, G4ThreeVector(0.*mm, 50.*mm, 0.*mm));  // +Y

    G4VSolid* bny = new G4Box("bny", 15.*mm, 15.*mm, bny_scale_z*13/2.*mm);
    G4VSolid* tub_bpy_bny = new G4UnionSolid(listname, tub_bpy, bny, 0, G4ThreeVector(0.*mm, -50.*mm, 0.*mm)); // -Y

    double uncoincide = 1. ;
    G4VSolid* itu  = new G4Tubs("itu", 0.*mm, 25.*mm, (uncoincide+13./2)*mm, 0.*deg, 360.*deg);
    G4VSolid* tub_bpy_bny_itu = new G4SubtractionSolid("tub_bpy_bny_itu", tub_bpy_bny, itu );

    return toplist ? tub_bpy_bny : tub_bpy_bny_itu  ;
}



const G4VSolid* U4SolidMaker::AnnulusOtherTwoBoxUnion(const char* name)
{
    G4VSolid* down1 = new G4Tubs("down1", 25.*mm, 45.*mm, 13./2*mm, 0.*deg, 360.*deg);
    G4VSolid* down2 = new G4Box("down2", 10.*mm, 11.5*mm, 13/2.*mm);
    G4VSolid* down_uni1 = new G4UnionSolid("down_uni1", down1    , down2, 0, G4ThreeVector(52.*mm, 0.*mm, 0.*mm));  // +X
    G4VSolid* down_uni2 = new G4UnionSolid("down_uni2", down_uni1, down2, 0, G4ThreeVector(-52.*mm, 0.*mm, 0.*mm)); // -X
    return down_uni2 ;
}


const G4VSolid* U4SolidMaker::AnnulusCrossTwoBoxUnion(const char* name)
{
    G4VSolid* down1 = new G4Tubs("down1", 25.*mm, 45.*mm, 13./2*mm, 0.*deg, 360.*deg);
    G4VSolid* down2 = new G4Box("down2", 10.*mm, 11.5*mm, 13/2.*mm);
    G4VSolid* down_uni1 = new G4UnionSolid("down_uni1", down1    , down2, 0, G4ThreeVector(52.*mm, 0.*mm, 0.*mm));  // +X

    G4VSolid* down3 = new G4Box("down3", 15.*mm, 15.*mm, 13/2.*mm);
    G4VSolid* down_uni3 = new G4UnionSolid("down_uni3", down_uni1, down3, 0, G4ThreeVector(0.*mm, 50.*mm, 0.*mm));  // +Y

    return down_uni3 ;
}



/**
Not yet managed to see the spurious intersects with a render::

    EYE=0,0,2 UP=0,1,0 CAM=1 TMIN=2 ./cxr_geochain.sh

**/

const G4VSolid* U4SolidMaker::AnnulusFourBoxUnion_(const char* name, G4double inner_radius  )
{
    // spurious intersects appear in XY cross section but not YZ
    G4VSolid* down1 = new G4Tubs("down1", inner_radius, 45.*mm, 13./2*mm, 0.*deg, 360.*deg);
    G4VSolid* down2 = new G4Box("down2", 10.*mm, 11.5*mm, 13/2.*mm);
    G4VSolid* down_uni1 = new G4UnionSolid("down_uni1", down1    , down2, 0, G4ThreeVector(52.*mm, 0.*mm, 0.*mm));  // +X
    G4VSolid* down_uni2 = new G4UnionSolid("down_uni2", down_uni1, down2, 0, G4ThreeVector(-52.*mm, 0.*mm, 0.*mm)); // -X
    G4VSolid* down3 = new G4Box("down3", 15.*mm, 15.*mm, 13/2.*mm);
    G4VSolid* down_uni3 = new G4UnionSolid("down_uni3", down_uni2, down3, 0, G4ThreeVector(0.*mm, 50.*mm, 0.*mm));  // +Y
    G4VSolid* down_uni4 = new G4UnionSolid("down_uni4", down_uni3, down3, 0, G4ThreeVector(0.*mm, -50.*mm, 0.*mm)); // -Y
    return down_uni4 ;
}

const G4VSolid* U4SolidMaker::AnnulusFourBoxUnion(const char* name){  return AnnulusFourBoxUnion_(name, 25.*mm );  }
const G4VSolid* U4SolidMaker::CylinderFourBoxUnion(const char* name){ return AnnulusFourBoxUnion_(name,  0.*mm );  }


/**
U4SolidMaker::BoxFourBoxUnion_
---------------------------------


                          +----------+
                          | B        |
                          |          |
               +----------|----------|-------------+
               |  A       |          |             |
               |          +----------+             |
               |                                   |
          +---------+                        +------------+
          |    |    |                        |     |    C |
          |    |    |                        |     |      |
          |    |    |                        |     |      |
          | D  |    |                        |     |      |
          +---------+                        +------------+
               |                                   |
               |                                   |
               |          +----------+             |
               |          |          |             |
               +----------|----------|-------------+
                          |          |
                          |       D  |
                          +----------+



TODO: switch off balancing and check the impact of pairing order

* eg disjoint unions (like B+C)
  although they work on their own they may be implicated with inner boundary spurious



Notice that the tree grows upwards with a new union root
as each prim is union-ed into the combination.

                       U
                  U       E
             U       D
        U       C
       A  B

:google:`CSG disjoint union`


**/


const G4VSolid* U4SolidMaker::BoxFourBoxUnion_(const char* name, const char* opt )
{
    G4VSolid* cbo = new G4Box("cbo", 45.*mm, 45.*mm, 45.*mm );
    G4VSolid* xbo = new G4Box("xbo", 10.*mm, 11.5*mm, 13/2.*mm);
    G4VSolid* ybo = new G4Box("ybo", 15.*mm, 15.*mm, 13/2.*mm);

    bool px = strstr(opt, "+X");
    bool nx = strstr(opt, "-X");
    bool py = strstr(opt, "+Y");
    bool ny = strstr(opt, "-Y");

    G4VSolid* comb = nullptr ;
    unsigned idx = 0 ;

    if(true)
    {
        comb = PrimitiveClone(cbo,"cbo",idx) ;
        idx += 1 ;
    }
    if(px)
    {
        comb = new G4UnionSolid(Name("cpx",idx), comb, PrimitiveClone(xbo,"bpx",idx+1), 0, G4ThreeVector(52.*mm, 0.*mm, 0.*mm));  // +X
        idx += 2 ;
    }
    if(nx)
    {
        comb = new G4UnionSolid(Name("cnx",idx), comb, PrimitiveClone(xbo,"bnx",idx+1), 0, G4ThreeVector(-52.*mm, 0.*mm, 0.*mm)); // -X
        idx += 2 ;
    }
    if(py)
    {
        comb = new G4UnionSolid(Name("cpy",idx), comb, PrimitiveClone(ybo,"bpy",idx+1), 0, G4ThreeVector(0.*mm, 50.*mm, 0.*mm));  // +Y
        idx += 2 ;
    }
    if(ny)
    {
        comb = new G4UnionSolid(Name("cny",idx), comb, PrimitiveClone(ybo,"bny",idx+1), 0, G4ThreeVector(0.*mm, -50.*mm, 0.*mm)); // -Y
        idx += 2 ;
    }

    return comb ;
}

const G4VSolid* U4SolidMaker::BoxFourBoxUnion(const char* name ){      return BoxFourBoxUnion_(name, "+X,-X,+Y,-Y") ; }
const G4VSolid* U4SolidMaker::BoxCrossTwoBoxUnion(const char* name ){  return BoxFourBoxUnion_(name, "+X,+Y") ; }
const G4VSolid* U4SolidMaker::BoxThreeBoxUnion(const char* name ){     return BoxFourBoxUnion_(name, "+X,+Y,-Y") ; }



const G4VSolid* U4SolidMaker::BoxFourBoxContiguous_(const char* name, const char* opt )
{
    G4VSolid* cbo = new G4Box("cbo", 45.*mm, 45.*mm, 45.*mm );
    G4VSolid* xbo = new G4Box("xbo", 10.*mm, 11.5*mm, 13/2.*mm);
    G4VSolid* ybo = new G4Box("ybo", 15.*mm, 15.*mm, 13/2.*mm);

    bool px = strstr(opt, "+X");
    bool nx = strstr(opt, "-X");
    bool py = strstr(opt, "+Y");
    bool ny = strstr(opt, "-Y");

    G4VSolid* item ;
    unsigned idx = 0 ;
    G4MultiUnion* comb = new G4MultiUnion(name);
    G4RotationMatrix rot(0, 0, 0);

    if(true)
    {
        G4ThreeVector pos(0.*mm, 0.*mm, 0.*mm);  // center
        G4Transform3D tr(rot, pos);
        item = PrimitiveClone(cbo,"cbo",idx) ;
        comb->AddNode(*item, tr);
        idx+=1 ;
    }
    if(px)
    {
        G4ThreeVector pos(52.*mm, 0.*mm, 0.*mm);  // +X
        G4Transform3D tr(rot, pos);
        item = PrimitiveClone(xbo,"bpx",idx) ;
        comb->AddNode(*item, tr);
        idx+=1 ;
    }
    if(nx)
    {
        G4ThreeVector pos(-52.*mm, 0.*mm, 0.*mm);  // -X
        G4Transform3D tr(rot, pos);
        item = PrimitiveClone(xbo,"bnx",idx) ;
        comb->AddNode(*item, tr);
        idx += 1 ;
    }
    if(py)
    {
        G4ThreeVector pos(0.*mm, 50.*mm, 0.*mm);  // +Y
        G4Transform3D tr(rot, pos);
        item = PrimitiveClone(ybo,"bpy",idx) ;
        comb->AddNode(*item, tr);
        idx += 1 ;
    }
    if(ny)
    {
        G4ThreeVector pos(0.*mm, -50.*mm, 0.*mm);  // -Y
        G4Transform3D tr(rot, pos);
        item = PrimitiveClone(ybo,"bny",idx) ;
        comb->AddNode(*item, tr);
        idx += 1 ;
    }
    return comb ;
}


const G4VSolid* U4SolidMaker::BoxFourBoxContiguous(const char* name ){     return BoxFourBoxContiguous_(name, "-X,+X,+Y,-Y") ; }






/**
U4SolidMaker::Uncoincide_Box_Box_Union
----------------------------------------

To avoid coincidence need to expand the smaller box into the larger
without changing the position the lower edge.
Hence increase the half-size in Z from *hz* to *new_hz* and
simultaneously shift upwards by the same amount (*zoff*)
to keep the lower edge at same z position::


       +hz + uncoincide - - - - - - - +~~~~~~~~~+ - -    zoff + new_hz  - - -
                                      |         |
                                      |         |
       +hz  +---------+ - - - - - - - | - - - - | - - - - - - - - - - - - - -
            |         |               |         |
            |         |               |         |
            |         |               |         |
            |         |               |_________|        zoff
            |         |               |         |
        0 --|---------| - - - - - - - - - - - - - - - - - - - - - - - - - - -
            |         |               |         |
            |         |               |         |
            |         |               |         |
            |         |               |         |
            |         |               |         |
       -hz  +---------+ - - - - - - - +---------+ - - -  zoff - new_hz  - - -



Line equations::

      hz + uncoincide = zoff + new_hz

      -hz             = zoff - new_hz

Add them::

     zoff = uncoincide/2

Subtract them::

     new_hz = hz + uncoincide/2





up2 is raised by 13.5 to form the thinner in z table top of the altar


         +---------------------------+     5mm                     - - -  8.5 + 10 = 18.5
         |         up2               |  - - - -   13.5  = 8.5+5
         +-----+---------------+-----+
               |               |   17/2 = 8.5mm
               |   up1         |   - - - -
               |               |
               +---------------+
                                          10 mm thin top of altar,
                                          17 mm thicker bottom of altar




HMM: the sign of the change to the translation depends on
whether the smaller_box (which needs to grow into the larger) is on
the rhs of the combination which has the translation applied to it


**/

G4VSolid* U4SolidMaker::Uncoincide_Box_Box_Union( const G4VSolid* bbu  )  // static
{
    LOG(info) << " bbu.GetName " << bbu->GetName() ;

    const G4BooleanSolid* bs = dynamic_cast<const G4BooleanSolid*>(bbu) ;

    const G4VSolid* a = bs->GetConstituentSolid(0) ;
    const G4VSolid* _b = bs->GetConstituentSolid(1) ;
    const G4DisplacedSolid* _b_disp = dynamic_cast<const G4DisplacedSolid*>(_b) ;
    G4ThreeVector b_tla = _b_disp->GetObjectTranslation();
    const G4VSolid* b = _b_disp->GetConstituentMovedSolid()  ;

    LOG(info) << " a.GetName " << a->GetName() ;
    LOG(info) << " _b.GetName " << _b->GetName() ;
    LOG(info) << " _b_disp.GetName " << _b_disp->GetName() ;
    LOG(info) << " b.GetName " << b->GetName() ;
    LOG(info) << " b_tla " << Desc(&b_tla) ;

    const G4Box* a_box = dynamic_cast<const G4Box*>(a);
    const G4Box* b_box = dynamic_cast<const G4Box*>(b);
    LOG(info) << " a_box " << Desc(a_box) ;
    LOG(info) << " b_box " << Desc(b_box) ;

    G4ThreeVector new_tla(b_tla);

    std::string new_name = bs->GetName()  ;

    G4Box* new_a = new G4Box(a->GetName(), a_box->GetXHalfLength(), a_box->GetYHalfLength(), a_box->GetZHalfLength() );
    G4Box* new_b = new G4Box(b->GetName(), b_box->GetXHalfLength(), b_box->GetYHalfLength(), b_box->GetZHalfLength() );

    int shift_axis = OneAxis(&b_tla);
    LOG(info) << " shift_axis " << shift_axis ;



    enum { A, B, UNKNOWN } ;
    int smaller = UNKNOWN ;

    for(int axis=0 ; axis < 3 ; axis++)
    {
        if(axis == shift_axis) continue ;   //
        double ah = HalfLength(a_box, axis);
        double bh = HalfLength(b_box, axis);
        if(ah == bh) continue ;    // ignore equal axes
        smaller = ah < bh ? A : B ;
    }

    LOG(info) << " smaller " << smaller ;


    double uncoincide = 1.*mm ;
    if(smaller != UNKNOWN )
    {
        G4Box* smaller_box = smaller == A ? new_a : new_b ;
        LOG(info) << " smaller_box.GetName " << smaller_box->GetName() ;

        ChangeBoxHalfLength( smaller_box, shift_axis, uncoincide/2. );

        ChangeThreeVector(   &new_tla ,   shift_axis, uncoincide/2. ) ;
    }
    else
    {
        LOG(fatal) << " failed to uncoincide " ;
    }

    G4UnionSolid* new_bbu = new G4UnionSolid( new_name, new_a, new_b, 0, new_tla );
    return new_bbu ;
}


double U4SolidMaker::HalfLength( const G4Box* box, int axis )
{
    double value = 0. ;
    switch(axis)
    {
        case X: value = box->GetXHalfLength() ; break ;
        case Y: value = box->GetYHalfLength() ; break ;
        case Z: value = box->GetZHalfLength() ; break ;
    }
    return value ;
}

void U4SolidMaker::ChangeThreeVector( G4ThreeVector* v, int axis, double delta )
{
    if( v == nullptr ) return ;
    switch(axis)
    {
        case X: v->setX(v->x() + delta)  ; break ;
        case Y: v->setY(v->y() + delta)  ; break ;
        case Z: v->setZ(v->z() + delta)  ; break ;
    }
}
void U4SolidMaker::ChangeBoxHalfLength( G4Box* box, int axis, double delta )
{
    switch(axis)
    {
        case X: box->SetXHalfLength(box->GetXHalfLength() + delta)  ; break ;
        case Y: box->SetYHalfLength(box->GetYHalfLength() + delta)  ; break ;
        case Z: box->SetZHalfLength(box->GetZHalfLength() + delta)  ; break ;
    }
}






int U4SolidMaker::OneAxis( const G4ThreeVector* v )
{
    double x = v ? v->x() : 0. ;
    double y = v ? v->y() : 0. ;
    double z = v ? v->z() : 0. ;
    int axis = ERR ;
    if( x != 0. && y == 0. && z == 0. ) axis = X ;
    if( x == 0. && y != 0. && z == 0. ) axis = Y ;
    if( x == 0. && y == 0. && z != 0. ) axis = Z ;
    return axis ;
}



std::string U4SolidMaker::Desc( const G4Box* box )
{
    std::stringstream ss ;
    ss
       << "("
       << std::fixed << std::setw(10) << std::setprecision(3) << box->GetXHalfLength() << " "
       << std::fixed << std::setw(10) << std::setprecision(3) << box->GetYHalfLength() << " "
       << std::fixed << std::setw(10) << std::setprecision(3) << box->GetZHalfLength()
       << ") "
       << box->GetName()
       ;
    std::string s = ss.str();
    return s ;
}

std::string U4SolidMaker::Desc( const G4ThreeVector* v )
{
    std::stringstream ss ;
    ss
       << "("
       << std::fixed << std::setw(10) << std::setprecision(3) << (v ? v->x() : 0. ) << " "
       << std::fixed << std::setw(10) << std::setprecision(3) << (v ? v->y() : 0. )  << " "
       << std::fixed << std::setw(10) << std::setprecision(3) << (v ? v->z() : 0. )
       << ") "
       ;
    std::string s = ss.str();
    return s ;
}




/**
U4SolidMaker::XJanchorConstruction
-----------------------------------

Observed spurious Geant4 intersects on the line between the Tubs and the Cons::

    solidXJanchor          G4UnionSolid

        sub                G4SubtractionSolid      (subtract big sphere from cone)
              down         G4Cons
              ball         G4Sphere

        up                 G4Tubs




        +-------------------------+--------------------------+         - - - - - -
         \                        .                         /                             10.0
          .                       +                        .           - - - - - -                   - - - - - -
           \                      .                       /                               10.0            |
            +---------+^^^^^^^^^^^.^^^^^^^^^^^^+---------+             - - - - - -                       16.5
                      |           .            |                                    13/2 = 6.5            |
                      +           .            +                       - - - - - -                   - - - - - -
                      |           .            |                                    13/2 = 6.5
                      +-----------.------------+         |   |         - - - - - -
                                  0           25        47  73


     FIX :
         increase tubs hz by uncoincide/2
         shift upwards uncoincide/2 (-> low edge stays same) by shifting down less

**/

const int U4SolidMaker::XJanchorConstruction_debug_mode = ssys::getenvint("U4SolidMaker__XJanchorConstruction_debug_mode", 0 ) ;

const G4VSolid* U4SolidMaker::XJanchorConstruction(const char* name)
{
    bool do_uncoincide = false ;
    bool do_noball = false ;

    switch(XJanchorConstruction_debug_mode)
    {
        case 0: do_uncoincide = false ;  do_noball = false ; break ;
        case 1: do_uncoincide = true  ;  do_noball = false ; break ;
        case 2: do_uncoincide = false ;  do_noball = true  ; break ;
        case 3: do_uncoincide = true  ;  do_noball = true  ; break ;
    }

    double uncoincide = do_uncoincide ? 1. : 0. ;

    LOG(info)
        << " U4SolidMaker__XJanchorConstruction_debug_mode " << XJanchorConstruction_debug_mode
        << " do_uncoincide " << do_uncoincide
        << " uncoincide " << uncoincide
        << " do_noball " << do_noball
        ;


    G4VSolid* solidXJanchor_up;
    G4VSolid* solidXJanchor_down;
    G4VSolid* solidXJanchor_ball;

    solidXJanchor_up   = new G4Tubs("solidXJanchor_up", 0.*mm, 25.*mm, (13.+uncoincide)/2*mm, 0.*deg, 360.*deg);
    solidXJanchor_down = new G4Cons("solidXJanchor_down", 0.*mm, 47.*mm, 0.*mm, 73.*mm, 10.*mm, 0.*deg, 360.*deg);   // to subtract the ball
    solidXJanchor_ball = new G4Sphere("solidXJanchor_ball", 0.*mm, 17820.*mm, 0.*deg, 360.*deg, 0.*deg, 180.*deg);

    G4SubtractionSolid* solidXJanchor_sub_ = new G4SubtractionSolid("solidXJanchor_sub",solidXJanchor_down, solidXJanchor_ball, 0, G4ThreeVector(0.*mm, 0*mm,  17820.*mm));
    G4VSolid* solidXJanchor_sub = do_noball ? solidXJanchor_down : (G4VSolid*)solidXJanchor_sub_ ;

    G4UnionSolid* solidXJanchor = new G4UnionSolid("solidXJanchor",solidXJanchor_sub, solidXJanchor_up, 0, G4ThreeVector(0.*mm, 0*mm,(-16.5 + uncoincide/2)*mm));

    return solidXJanchor ;
}




const G4VSolid* U4SolidMaker::SJReceiverConstruction(const char* name)
{
#ifdef DIRTY
    G4VSolid* solidSJReceiver_up ;
    G4VSolid* solidSJReceiver_down ;
    G4VSolid* solidSJReceiver_box ;
    G4VSolid* solidSJReceiver_ball ;

    //solidSJReceiver_up   = new G4Cons("solidSJReceiver_up", 0.*mm, 31.7*mm, 0.*mm, 25*mm, 13./2*mm, 0.*deg, 360.0*deg);
    solidSJReceiver_up = new G4Tubs("solidXJanchor_up", 0.*mm, 25.*mm, 13./2*mm, 0.*deg, 360.*deg);
    solidSJReceiver_down = new G4Cons("solidSJReceiver_down", 0.*mm, 73.*mm, 0.*mm, 47.*mm, 10.*mm, 0.*deg, 360.*deg);   // to subtract the ball
    //solidSJReceiver_down = new G4Cons("solidSJReceiver_down", 0.*mm, 47.*mm, 0.*mm, 60.*mm, 5.*mm, 0.*deg, 360.*deg); // original size
    solidSJReceiver_box = new G4Box("solidSJReceiver_box", 17780.*mm, 17780.*mm, 17780.*mm);
    solidSJReceiver_ball = new G4Sphere("solidSJReceiver_ball", 0.*mm, 17700.*mm, 0.*deg, 360.*deg, 0.*deg, 180.*deg);
    G4SubtractionSolid* solidSphere_sub = new G4SubtractionSolid("solidSphere_sub", solidSJReceiver_box, solidSJReceiver_ball);
    G4SubtractionSolid* solidSJReceiver_sub = new G4SubtractionSolid("solidSJReceiver_sub",solidSJReceiver_down, solidSJReceiver_ball, 0, G4ThreeVector(0.*mm, 0*mm,  17699.938*mm));
    G4UnionSolid* solidSJReceiver = new G4UnionSolid("solidSJReceiver",solidSJReceiver_sub, solidSJReceiver_up, 0, G4ThreeVector(0.*mm, 0*mm,16.5*mm));
#else
    G4VSolid* solidSJReceiver_up ;
    G4VSolid* solidSJReceiver_down ;
    G4VSolid* solidSJReceiver_ball ;

    solidSJReceiver_up = new G4Tubs("solidXJanchor_up", 0.*mm, 25.*mm, 13./2*mm, 0.*deg, 360.*deg);
    solidSJReceiver_down = new G4Cons("solidSJReceiver_down", 0.*mm, 73.*mm, 0.*mm, 47.*mm, 10.*mm, 0.*deg, 360.*deg);   // to subtract the ball
    solidSJReceiver_ball = new G4Sphere("solidSJReceiver_ball", 0.*mm, 17700.*mm, 0.*deg, 360.*deg, 0.*deg, 180.*deg);
    G4SubtractionSolid* solidSJReceiver_sub = new G4SubtractionSolid("solidSJReceiver_sub",solidSJReceiver_down, solidSJReceiver_ball, 0, G4ThreeVector(0.*mm, 0*mm,  17699.938*mm));
    G4UnionSolid* solidSJReceiver = new G4UnionSolid("solidSJReceiver",solidSJReceiver_sub, solidSJReceiver_up, 0, G4ThreeVector(0.*mm, 0*mm,16.5*mm));
#endif
    return solidSJReceiver ;
}




const G4VSolid* U4SolidMaker::BoxMinusTubs0(const char* name)  // is afflicted
{
    double tubs_hz = 15.2*mm ;
    double zshift = 0*mm ;
    G4VSolid* box   = new G4Box("box",  250*mm, 250*mm, 100*mm );
    G4VSolid* tubs =  new G4Tubs("tubs",120*mm,208*mm,tubs_hz,0.0*deg,360.0*deg);
    G4VSolid* box_minus_tubs = new G4SubtractionSolid(name,box,tubs,0,G4ThreeVector(0.*mm,0.*mm,zshift));
    return box_minus_tubs ;
}

const G4VSolid* U4SolidMaker::BoxMinusTubs1(const char* name)
{
    double tubs_hz = 15.2*mm ;
    G4VSolid* box   = new G4Box("box",  250*mm, 250*mm, 100*mm );
    G4VSolid* tubs =  new G4Tubs("tubs",120*mm,208*mm,tubs_hz,0.0*deg,360.0*deg);
    G4VSolid* box_minus_tubs = new G4SubtractionSolid(name,box,tubs);
    return box_minus_tubs ;
}

const G4VSolid* U4SolidMaker::BoxMinusOrb(const char* name)
{
    double radius = ssys::getenvfloat("U4SolidMaker_BoxMinusOrb_radius", 130.f) ;

    double sx     = ssys::getenvfloat("U4SolidMaker_BoxMinusOrb_sx", 100.f) ;
    double sy     = ssys::getenvfloat("U4SolidMaker_BoxMinusOrb_sy", 100.f) ;
    double sz     = ssys::getenvfloat("U4SolidMaker_BoxMinusOrb_sz", 100.f) ;

    double dx     = ssys::getenvfloat("U4SolidMaker_BoxMinusOrb_dx", 0.f) ;
    double dy     = ssys::getenvfloat("U4SolidMaker_BoxMinusOrb_dy", 0.f) ;
    double dz     = ssys::getenvfloat("U4SolidMaker_BoxMinusOrb_dz", 0.f) ;

    G4VSolid* box = new G4Box("box",  sx, sy, sz );
    G4VSolid* orb = new G4Orb("orb",  radius );

    G4VSolid* box_minus_orb = new G4SubtractionSolid(name,box,orb,nullptr, G4ThreeVector(dx, dy, dz) );
    return box_minus_orb ;
}


const G4VSolid* U4SolidMaker::PolyconeWithMultipleRmin(const char* name)
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


const G4VSolid* U4SolidMaker::UnionOfHemiEllipsoids(const char* name )
{
    assert( strstr( name, "UnionOfHemiEllipsoids" ) != nullptr );

    std::vector<long> vals ;
    sstr::Extract(vals, name);
    long iz = vals.size() > 0 ? vals[0] : 0 ;

    std::cout
        << "U4SolidMaker::UnionOfHemiEllipsoids"
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


/**
U4SolidMaker::OrbOrbMultiUnion
-------------------------------

OrbOrbMultiUnion0
    single Orb within MultiUnion
OrbOrbMultiUnion1
    three Orb at (-100,0,100) x positions within MultiUnion
OrbOrbMultiUnion2
    five Orb at (-200,-100,0,100,200) x positions within MultiUnion
OrbOrbMultiUnion{N}
    2N+1 Orb at {-N, -N+1, ..., 0, 1, .... N}*100 x positions within MultiUnion

**/

const G4VSolid* U4SolidMaker::OrbOrbMultiUnion(const char* name )
{
    long num = sstr::ExtractLong(name, 1) ;
    G4MultiUnion* comb = new G4MultiUnion(name);

    G4RotationMatrix rot(0, 0, 0);
    G4Orb* orb = new G4Orb("Orb", 50.) ;

    for(long i=-num ; i <= num ; i++)
    {
       G4ThreeVector tlate( double(i)*100.*mm, 0.*mm, 0.*mm);
       G4Transform3D tr(rot, tlate) ;
       comb->AddNode( *orb, tr );
    }

    return comb ;
}





/**
U4SolidMaker::GridMultiUnion_
---------------------------------

(nx, ny, nz) of (3,3,3) yields a 7x7x7 grid from eg::

    nx : -3, -2, -1, 0, +1, +2, +3

**/


const G4VSolid* U4SolidMaker::GridMultiUnion_(const char* name, G4VSolid* item, double gridspace, int nx, int ny, int nz )
{
    G4MultiUnion* grid = new G4MultiUnion(name);

    for(int i=-nx ; i <= nx ; i++ )
    for(int j=-ny ; j <= ny ; j++ )
    for(int k=-nz ; k <= nz ; k++ )
    {
        G4ThreeVector pos(double(i)*gridspace*mm, double(j)*gridspace*mm, double(k)*gridspace*mm );
        LOG(info) << pos ;

        G4RotationMatrix rot(0, 0, 0);
        G4Transform3D tr(rot, pos);
        grid->AddNode(*item, tr);

    }
    //grid->Voxelize();
    return grid ;
}

/**
U4SolidMaker::OrbGridMultiUnion
----------------------------------

                    :       :
                    :       :
          +---+   +-:-+   +-:-+
          |   |   | : |   | : |
          +---+   +-:-+   +-:-+
                    :       :
          +---+   +-:-+   +-:-+
          |   |   | 0 |   | : |
          +---+   +-:-+   +-:-+
                    :       :
          +---+   +-:-+   +-:-+
          |   |   | : |   | : |
          +---+   +-:-+   +-:-+
                    :       :
                  : : :     :
            -radius:+radius
                    :


* when there is no overlap  radius < gridscale , only the middle Orb gets intersects

**/

const G4VSolid* U4SolidMaker::OrbGridMultiUnion(const char* name)
{
    std::vector<long> vals ;
    sstr::Extract(vals, name);

    assert( vals.size() == 2 );

    double radius(vals[0]) ;
    double gridscale(vals[1]) ;

    LOG(info)
        << " name " << name
        << " radius " << radius
        << " gridscale " << gridscale
        ;

    G4VSolid* item = new G4Orb("orb", radius*mm );

    int nx = 3 ;
    int ny = 3 ;
    int nz = 3 ;

    return GridMultiUnion_(name, item, gridscale, nx, ny, nz );
}


/**
U4SolidMaker::BoxGridMultiUnion
--------------------------------

eg BoxGridMultiUnion10:30_YX

halfside   10 mm
gridscale  30

**/


const G4VSolid* U4SolidMaker::BoxGridMultiUnion(const char* name)
{
    std::vector<long> vals ;
    sstr::Extract(vals, name);

    assert( vals.size() == 2 );

    double halfside(vals[0]) ;
    double gridscale(vals[1]) ;

    LOG(info)
        << " name " << name
        << " halfside " << halfside
        << " gridscale " << gridscale
        ;

    G4VSolid* item = new G4Box("box", halfside*mm, halfside*mm, halfside*mm );

    int nx = 3 ;
    int ny = 3 ;
    int nz = 3 ;

    return GridMultiUnion_(name, item, gridscale, nx, ny, nz );
}








/**
U4SolidMaker::LHCbRichSphMirr
--------------------------------


                               Z
                               |

              |+--------------|+
               |               |
               |               |
               |               |
               |               |
               |               |  1500/2
               |               |
               |               |
               |               |
               + - - - + - - - +
               |               |
               |               |
               |               |
               |               |  1500/2
               |               |
               |               |
               |               |
               |               |
              |+--------------|+
               :               :
             3650.0           3651.5


                   ------->  X





**/


const G4VSolid* U4SolidMaker::LHCbRichSphMirr(const char* qname)  // static
{
    long mode = sstr::ExtractLong(qname, 0);
    LOG(LEVEL) << " qname " << qname << " mode " << mode ;

   // /Users/blyth/liyu/Rich_Simplified/include/SphMirrorGeometryParameters.hh
    //Parameters for simplified rich1 sph mirror.
    const G4double InnerRadius = 3650.0 *  CLHEP::mm;
    const G4double Thickness   = 1.5 * CLHEP::mm;
    const G4double OuterRadius =  InnerRadius + Thickness ;

    const G4double SegmentSizeX = 1500.0 * CLHEP::mm;  // SCB : misnammed should be Z not X
    const G4double SegmentSizeZ = SegmentSizeX  ;
    const G4double SegmentSizeY = 650.0  * CLHEP::mm;


    // To account for spherical geometry with straight edges, the angluar sizes are increased by a factor and then boolen subtraction made.
    const G4double DeltaExtendedFactor = 1.2;
    const G4double DeltaTheta =  2 * asin(SegmentSizeX/(2.0*InnerRadius)) * DeltaExtendedFactor *  CLHEP::rad;
    const G4double DeltaPhi =  2 * asin(SegmentSizeY/(2.0*InnerRadius) )* DeltaExtendedFactor * CLHEP::rad;
    const G4double ThetaStart = (0.5 * CLHEP::pi * CLHEP::rad)  - (0.5 * DeltaTheta) ;
    const G4double PhiStart   =  -0.5 * DeltaPhi ;
    const G4double SubLargeSize = 5000 *  CLHEP::mm;
    const G4double SubHalfSizeX = SubLargeSize;
    const G4double SubHalfSizeY = SubLargeSize;
    const G4double SubHalfSizeZ = SubLargeSize;
    const G4double SubPosX = 0.0 *  CLHEP::mm;
    const G4double SubPosY = 0.0 *  CLHEP::mm;
    const G4double SubUpPosY = (0.5 * SegmentSizeY) + SubHalfSizeY;
    const G4double SubDownPosY = -1.0 * SubUpPosY ;
    const G4double SubPosZ = 0.0 * CLHEP::mm;
    const G4double SubRightPosZ =(0.5 * SegmentSizeX) +  SubHalfSizeZ;
    const G4double SubLeftPosZ = -1.0* SubRightPosZ;



   // /Users/blyth/liyu/Rich_Simplified/src/RichTbLHCbR1or.cc

    G4Sphere* SphFullSph = new G4Sphere ("SphFullSphDEV",InnerRadius,
                       OuterRadius,PhiStart,
                                           DeltaPhi, ThetaStart,
                                           DeltaTheta);
    G4Box * SphSubBox = new G4Box("SphSubBox",SubHalfSizeX,
                    SubHalfSizeY, SubHalfSizeZ);
    G4RotationMatrix SubBoxRot;
    G4ThreeVector  SubBoxTPos ( SubPosX,SubUpPosY,SubPosZ);
    G4ThreeVector  SubBoxBPos ( SubPosX,SubDownPosY,SubPosZ);
    G4ThreeVector  SubBoxLPos ( SubPosX,SubPosY,SubLeftPosZ);
    G4ThreeVector  SubBoxRPos ( SubPosX,SubPosY,SubRightPosZ);
    G4Transform3D  SubBoxTTrans(  SubBoxRot, SubBoxTPos);
    G4Transform3D  SubBoxBTrans(  SubBoxRot, SubBoxBPos);
    G4Transform3D  SubBoxLTrans(  SubBoxRot, SubBoxLPos);
    G4Transform3D  SubBoxRTrans(  SubBoxRot, SubBoxRPos);
    G4SubtractionSolid* TSph = new G4SubtractionSolid("SphTSph",SphFullSph,
                                            SphSubBox , SubBoxTTrans);
   G4SubtractionSolid* BSph = new G4SubtractionSolid("SphBSphBox",TSph,
                                            SphSubBox , SubBoxBTrans);
   G4SubtractionSolid* LSph = new G4SubtractionSolid("SphLSphBox",BSph,
                                            SphSubBox , SubBoxLTrans);
   G4SubtractionSolid* RSph = new G4SubtractionSolid("SphRSphBox",LSph,
                                            SphSubBox , SubBoxRTrans);


    const G4VSolid* simple = nullptr ;
    if( mode == 1 )
    {
        G4double MiddleRadius = (InnerRadius + OuterRadius)/2. ;

        G4Orb* inner = new G4Orb("inner", InnerRadius );
        G4Orb* outer = new G4Orb("outer", OuterRadius );
        G4SubtractionSolid* shell = new G4SubtractionSolid("shell", outer, inner );

        G4double sag_max = SagittaMax( InnerRadius, OuterRadius, SegmentSizeY, SegmentSizeZ );
        G4double SagMax = 80.*mm ;
        bool sag_max_expect = SagMax > sag_max ;
        assert( sag_max_expect );
        if(!sag_max_expect) std::raise(SIGINT);
        G4double FullDepthX = Thickness + SagMax ;

        G4Box* cutbox = new G4Box("cutbox", FullDepthX/2. , SegmentSizeY/2. , SegmentSizeZ/2. );
        simple = new G4IntersectionSolid("simple_CSG_EXBB", shell, cutbox, 0, G4ThreeVector(MiddleRadius, 0., 0.) ) ;
    }


    const G4VSolid* solid = nullptr ;
    switch(mode)
    {
       case 0: solid = RSph   ; break ;
       case 1: solid = simple ; break ;
    }
    assert(solid);
    return solid  ;
}
/**
U4SolidMaker::Sagitta
---------------------



                  .
            .     .        .
        .         s           .
      /           .            \
   A +------------+-------------+ B
          l/2     |C    l/2
       .          |           .
                 r-s
          .       |      .
        r         |         r
              .   |   .
                  |
                  +
                  O

s: sagitta or depth of the arc

l: full length of chord across base of arc

r: radius


Pythagoras for triangle AOC::

     (r-s)^2 + (l/2)^2 = r^2

       r - s = sqrt( r^2 - (l/2)^2 )

           s = r - sqrt( r^2 - (l/2)^2 )

**/


G4double U4SolidMaker::Sagitta( G4double radius, G4double chord )
{
    G4double sagitta = radius - sqrt( radius*radius - chord*chord/4. ) ;
    return sagitta ;
}

G4double U4SolidMaker::SagittaMax( G4double radius, G4double sy, G4double sz )
{
    G4double sag_sy = Sagitta( radius, sy );
    G4double sag_sz = Sagitta( radius, sz );
    G4double sag_max = std::max( sag_sy, sag_sz );
    return sag_max ;
}

G4double U4SolidMaker::SagittaMax( G4double InnerRadius, G4double OuterRadius,  G4double sy, G4double sz )
{
    G4double sag_inner = SagittaMax( InnerRadius, sy, sz );
    G4double sag_outer = SagittaMax( OuterRadius, sy, sz );
    G4double sag_max = std::max( sag_inner, sag_outer );

    LOG(LEVEL)
        << " InnerRadius " << InnerRadius
        << " OuterRadius " << OuterRadius
        << " sag_max " << sag_max
        << std::endl
        << " sag_inner : SaggitaMax(" << InnerRadius << "," <<  sy << "," << sz << ")" << sag_inner
        << std::endl
        << " sag_outer : SaggitaMax(" << OuterRadius << "," <<  sy << "," << sz << ")" << sag_outer
        << std::endl
        ;

    return sag_max ;
}


/**
U4SolidMaker::LHCbRichFlatMirr
---------------------------------


                  Z
                  |
                  |    Y
                  |   /
                  |  /
                  | /
                  |/                                                                                               |
                  +-------> X - - - - - - - - - - - - 5,000,000 mm  - - - - - - - - - - - - - - - - - - - - - - - -+
                 .                                                                                                /
                .
               .
             -Y


      Rectangle is formed close to X axis, deltaPhi (in XY plane) is symmetric around phi=0.




                               Z
                               |

              |+--------------|+
               |               |
               |               |
               |               |
               |               |
               |               |  1480/2
               |               |
               |               |
               |               |
               + - - - + - - - +
               |               |
               |               |
               |               |
               |               |  1480/2
               |               |
               |               |
               |               |
               |               |
              |+--------------|+

            5,000,000     5,000,006

                   ------->  X


   sagitta

         l^2        ( 1480/2 )^2
         ---   =     -------------   = 0.01369
         8r            8*5,000,000


   intersect box
       center    :  ( 5000003, 0 , 0 )
       halfsides :  (       4, 880/2,   1480/2 )


                    4 in X is more than enough to contain the sagitta

**/

const G4VSolid* U4SolidMaker::LHCbRichFlatMirr(const char* qname)  // static
{
    long mode = sstr::ExtractLong(qname, 0);
    LOG(LEVEL) << " qname " << qname << " mode " << mode ;


    // /Users/blyth/Rich_Simplified/Rich_Simplified/include/RichTbR1FlatMirrorGeometryParameters.hh

    //  :.,+30s/RichTbR1FlatMirr//g

    //Parameters for simplified rich1 flat mirror.
    const G4double InnerRadius = 5000.0 *  CLHEP::m;
    const G4double Thickness   = 6.0 * CLHEP::mm;
    const G4double OuterRadius =  InnerRadius + Thickness ;
    const G4double MiddleRadius = (InnerRadius + OuterRadius)/2. ;


    const G4double SegmentSizeX = 1480.0 * CLHEP::mm;  // <-- should be Z (not X)
    const G4double SegmentSizeZ = SegmentSizeX  ;

    const G4double SegmentSizeY = 880.0 * CLHEP::mm;

   /**
   From xxs.sh looking at XZ, XY, YZ planes

         X range :  5M -> 5M + 6 mm
         Y range : -440 -> 440 mm
         Z range : -740 -> 740 mm
   **/


   /*
    const G4double BotInLHCbPosY  = 337.90 * CLHEP::mm;
    const G4double BotInLHCbPosZ  = 1323.31 * CLHEP::mm;
    const G4double VertTilt = 0.25656 * CLHEP:: rad;
    const G4double RotX = -1.0 * VertTilt;
    const G4double RotY = (0.5 * CLHEP::pi * CLHEP::rad);
   */

    const G4double DeltaTheta = asin(SegmentSizeX/InnerRadius) * CLHEP::rad;  // NO NEED FOR THE asin AS EXTREMELY SMALL ANGLES
    const G4double DeltaPhi   = asin(SegmentSizeY/InnerRadius) * CLHEP::rad;
    const G4double StartTheta = (0.5 * CLHEP::pi * CLHEP::rad)  - (0.5*  DeltaTheta);
    const G4double StartPhi   = -0.5 * DeltaPhi;

    LOG(LEVEL)
        << " DeltaTheta " << DeltaTheta
        << " SegmentSizeX " << SegmentSizeX
        << " InnerRadius " << InnerRadius
        << " SegmentSizeX/InnerRadius " << SegmentSizeX/InnerRadius
        << " asin(SegmentSizeX/InnerRadius) " << asin(SegmentSizeX/InnerRadius)
        << " CLHEP::rad " << CLHEP::rad
        << " StartTheta " << StartTheta
        ;


    LOG(LEVEL)
        << " DeltaPhi " << DeltaPhi
        << " SegmentSizeY " << SegmentSizeY
        << " InnerRadius " << InnerRadius
        << " SegmentSizeY/InnerRadius " << SegmentSizeY/InnerRadius
        << " asin(SegmentSizeY/InnerRadius) " << asin(SegmentSizeY/InnerRadius)
        << " CLHEP::rad " << CLHEP::rad
        << " StartPhi " << StartPhi
        ;


    /*
    const G4double InLHCbPosX = 0.0* CLHEP::mm;
    const G4double InLHCbPosY = BotInLHCbPosY +
                                                 0.5 * SegmentSizeY * cos (VertTilt) +
                                                 OuterRadius * sin ( VertTilt );
     */


    // /Users/blyth/Rich_Simplified/Rich_Simplified/src/RichTbLHCbR1FlatMirror.cc

    G4Sphere* RichTbR1FlatFull = new G4Sphere ("RichTbR1FlatFullDEV",InnerRadius,OuterRadius,StartPhi,DeltaPhi, StartTheta,DeltaTheta);



    G4VSolid* solid = nullptr ;
    if( mode == 0 )
    {
        solid = RichTbR1FlatFull ;
    }
    else
    {
        LOG(LEVEL) << " constructing the flat mirror using intersection box rather than attemping to describe with miniscule thetacut and phicut angles"  ;


        G4double sag_max = SagittaMax( InnerRadius, OuterRadius, SegmentSizeY, SegmentSizeZ );
        G4double SagMax = 1.*mm ;
        assert( SagMax > sag_max );
        G4double FullDepthX = Thickness + SagMax ;


        LOG(LEVEL)
            << " sag_max " << sag_max
            << " SagMax " << SagMax
            << " FullDepthX  " << FullDepthX
            ;

        G4Orb* outer = new G4Orb("outer",OuterRadius);
        G4Orb* inner = new G4Orb("inner",InnerRadius);
        G4SubtractionSolid* shell = new G4SubtractionSolid("shell", outer, inner );
        G4Box* box = new G4Box("box", FullDepthX/2. , SegmentSizeY/2. , SegmentSizeZ/2. );

        std::string rootname = qname ;
        rootname += "_CSG_EXBB" ;
        solid = new G4IntersectionSolid( rootname, shell, box, 0, G4ThreeVector( MiddleRadius, 0., 0. ));
    }

    return solid ;
}




const G4VSolid* U4SolidMaker::SphereIntersectBox(const char* qname)  // static
{
    G4double phiStart = 0. ;
    G4double phiDelta = 2. ;
    G4double thetaStart = 0. ;
    G4double thetaDelta = 1.  ;
    G4Sphere* sph = new G4Sphere("sph", 95.*mm, 105.*mm,  phiStart*CLHEP::pi, phiDelta*CLHEP::pi, thetaStart*CLHEP::pi, thetaDelta*CLHEP::pi );
    G4Box* box = new G4Box("box", 20.*mm, 20.*mm, 20.*mm );
    G4IntersectionSolid* sph_box = new G4IntersectionSolid("sph_box_CSG_EXBB", sph, box, 0, G4ThreeVector(0.*mm, 0.*mm, 100.*mm ) );
    return sph_box ;
}

/**
U4SolidMaker::LocalFastenerAcrylicConstruction
------------------------------------------------

The tree is grown upwards::


                                uni1
                              /     \
                          uni1       screw
                        /      \
                    uni1        screw
                  /     \
               uni1      screw
              /    \
       IonRing     screw



**/



const G4VSolid* U4SolidMaker::LocalFastenerAcrylicConstruction(const char* name) // static
{
    const char* PREFIX = "LocalFastenerAcrylicConstruction" ;
    assert( sstr::StartsWith(name,PREFIX ));
    int num_column = strlen(name) > strlen(PREFIX) ? std::atoi( name + strlen(PREFIX) ) : 8 ;

    LOG(info)
        << " name " <<  ( name ? name : "-" )
        << " num_column " << num_column
        ;


    const char* screw_name = "screw_HINT_LISTNODE_PRIM_CONTIGUOUS" ;
    //const char* screw_name = "screw_HINT_LISTNODE_PRIM_DISCONTIGUOUS" ;

    G4VSolid* uni_Addition(nullptr);
    {
        G4Tubs *IonRing = new G4Tubs("IonRing",123*mm,206.2*mm,7*mm,0.0*deg,360.0*deg);
        G4Tubs* screw = new G4Tubs(screw_name,0,13*mm,50.*mm,0.0*deg,360.0*deg);
        uni_Addition = IonRing;
        for(int i=0;i<num_column;i++)
        {
            G4ThreeVector tlate(164.*cos(i*pi/4)*mm, 164.*sin(i*pi/4)*mm,-65.0*mm);
            G4UnionSolid* uni1 = new G4UnionSolid("uni1",uni_Addition, screw, 0, tlate);
            uni_Addition = uni1;
        }
    }
    return uni_Addition ;
}


/**
U4SolidMaker::AltLocalFastenerAcrylicConstruction
--------------------------------------------------

         uni1
        /     \
     IonRing    muni


**/

const G4VSolid* U4SolidMaker::AltLocalFastenerAcrylicConstruction(const char* name) // static
{
    [[maybe_unused]] const char* PREFIX = "AltLocalFastenerAcrylicConstruction" ;
    assert( sstr::StartsWith(name,PREFIX ));
    long num_column = sstr::ExtractLong(name, 1) ;

    LOG(info)
        << " name " <<  ( name ? name : "-" )
        << " num_column " << num_column
        ;

    assert( num_column > 0 );

    G4Tubs* IonRing = new G4Tubs("IonRing",123*mm,206.2*mm,7*mm,0.0*deg,360.0*deg);

    G4MultiUnion* muni = new G4MultiUnion(name);
    G4Tubs* screw = new G4Tubs("screw",0,13*mm,50.*mm,0.0*deg,360.0*deg);

    G4RotationMatrix rot(0, 0, 0);
    for(long i=0;i<num_column;i++)
    {
       G4ThreeVector tlate(164.*cos(i*pi/4)*mm, 164.*sin(i*pi/4)*mm,-65.0*mm);
       G4Transform3D tr(rot, tlate) ;
       muni->AddNode( *screw, tr );
    }

    G4UnionSolid* uni1 = new G4UnionSolid(name, IonRing, muni, 0, G4ThreeVector(0.,0.,0.));
    return uni1 ;
}


/**
U4SolidMaker::BltLocalFastenerAcrylicConstruction
--------------------------------------------------

Was trying to form a boolean tree  with the nodes destined
to become constituents of the listnode within a separate G4UnionSolid tree.
But it is not so easy to split off the screws like that, as they all need translation
including the 0th.

The reason for trying to do that is because the inner radius on the IonRing tubs
means have to keep that separate.

**/

const G4VSolid* U4SolidMaker::BltLocalFastenerAcrylicConstruction(const char* name) // static
{
    [[maybe_unused]] const char* PREFIX = "BltLocalFastenerAcrylicConstruction" ;
    assert( sstr::StartsWith(name,PREFIX ));
    long num_column = sstr::ExtractLong(name, 1) ;

    LOG(info)
        << " name " <<  ( name ? name : "-" )
        << " num_column " << num_column
        ;

    assert( num_column > 0 );

    G4Tubs* IonRing = new G4Tubs("IonRing",123*mm,206.2*mm,7*mm,0.0*deg,360.0*deg);

    G4Tubs* screw = new G4Tubs("screw",0,13*mm,50.*mm,0.0*deg,360.0*deg);

    G4ThreeVector tlate[num_column] = {} ;
    for(long i=0;i<num_column;i++) tlate[i] = G4ThreeVector(164.*cos(i*pi/4)*mm, 164.*sin(i*pi/4)*mm,-65.0*mm);

    G4VSolid* muni = screw ;
    for(long i=1 ; i < num_column ; i++) muni = new G4UnionSolid( name, muni, screw, 0, tlate[i] ) ;

    G4UnionSolid* uni1 = new G4UnionSolid(name, IonRing, muni, 0, tlate[0] );
    return uni1 ;
}



const G4VSolid* U4SolidMaker::WaterDistributer(const char* name_) // static
{
    G4MultiUnion* multiUnion = new G4MultiUnion(name_);

    double TubeR_type_I = 38 * mm;
    double TubeR_type_II = 70 * mm;
    double holeR = TubeR_type_I ;



    G4double distance = 120 *cm;
    G4double TubeR_I = TubeR_type_I;
    G4double TubeR_II = TubeR_type_II;
    G4double TubeR_III = TubeR_type_I;

    G4double holeRadius = holeR;

    G4String name = multiUnion->GetName() ;
    G4String name_I = name + "_I" ;
    G4String name_II = name + "_II" ;
    G4String name_III = name + "_III" ;

    G4VSolid* solid_I = WaterDistributerHelper(name_I.c_str(),distance, TubeR_I);
    G4VSolid* solid_II = WaterDistributerHelper(name_II.c_str(),distance, TubeR_II);
    G4VSolid* solid_III = WaterDistributerHelper(name_III.c_str(),distance, TubeR_III);

    G4ThreeVector positionI(0, 23.5*cm, 0);
    G4ThreeVector positionII(0, 0, 0);
    G4ThreeVector positionIII(0, - 23.5*cm, 0);

    G4RotationMatrix* rotation = new G4RotationMatrix();

    G4Transform3D trI = G4Transform3D(*rotation, positionI);
    G4Transform3D trII = G4Transform3D(*rotation, positionII);
    G4Transform3D trIII = G4Transform3D(*rotation, positionIII);

    multiUnion->AddNode(*solid_II, trII);
    multiUnion->AddNode(*solid_I, trI);
    multiUnion->AddNode(*solid_III, trIII);

    for (int i = 0; i < 8; i++) {
        // Create circular hole geometry
        G4Tubs* hole = new G4Tubs("Hole_" + std::to_string(i), 0, holeRadius, (23.5) * cm, 0, 360 * deg);

        G4double theta = i * 45 * deg;
        G4RotationMatrix* rotation = new G4RotationMatrix();
        rotation->rotateX(90 * deg);

        G4ThreeVector position(-distance * cos(theta), 0, distance * sin(theta));
        G4Transform3D tr = G4Transform3D(*rotation, position);

        multiUnion->AddNode(*hole, tr);
    }

    multiUnion->Voxelize();

    return multiUnion ;
}













/**
U4SolidMaker::WaterDistributerHelper
-------------------------------------


::

               _________
              / _______ \
             / /       \ \
            / /         \ \
            | |         | |
            | |         | |
            | |         | |
            \ \         / /
             \ \_______/ /
              \_________/

**/




G4VSolid* U4SolidMaker::WaterDistributerHelper(const char* name, double distance, double TubeR) // static
{
    G4double rMax = TubeR;
    G4double rMin = 0;
    G4double dz = distance * tan(22.5 * deg);
    G4double sPhi = 0.0;
    G4double dPhi = 2 * M_PI;
    //G4double holeRadius = 33 *mm;

    G4MultiUnion* multiUnion = new G4MultiUnion(name);

    // Rotating and combining eight cylinders
    G4double rotationAngle = 22.5 * deg;
    for (int i = 0; i < 8; i++) {
        // Calculate the normal vector of the cutting plane
        G4double theta_i = rotationAngle;
        G4double theta_j = rotationAngle;
        G4ThreeVector pLowNorm(sin(theta_i), 0, -cos(theta_j));
        G4ThreeVector pHighNorm(sin(theta_i), 0, cos(theta_j));

        G4CutTubs* cutTube = new G4CutTubs(
            "CutTube_" + std::to_string(i), rMin, rMax, dz, sPhi, dPhi, pLowNorm, pHighNorm);

        //G4Tubs* hole = new G4Tubs("Hole_" + std::to_string(i), 0, holeRadius, TubeR + 1 * cm, 0, 360 * deg);

        G4double theta = i * 45 * deg;
        G4RotationMatrix* rotation = new G4RotationMatrix();
        rotation->rotateY(45 * i * deg);

        G4RotationMatrix* rotation_hole = new G4RotationMatrix();
        rotation_hole->rotateX(90 * deg);

        G4ThreeVector position(-distance * cos(theta), 0, distance * sin(theta));
        G4Transform3D tr = G4Transform3D(*rotation, position);


        //G4SubtractionSolid* cutTubeWithHole = new G4SubtractionSolid(
        //    "CutTubeWithHole_" + std::to_string(i), cutTube, hole, rotation_hole, G4ThreeVector(0, 0, 0));


        multiUnion->AddNode(*cutTube, tr);
    }

    // Complete the construction of MultiUnion (must call!).
    multiUnion->Voxelize();

    return multiUnion ;
}




/**
U4SolidMaker::AltWaterDistributer
----------------------------------

One multiunion of everything works and is simpler
but polygonization taking forever so unusable.

**/

const G4VSolid* U4SolidMaker::AltWaterDistributer(const char* name_) // static
{
    G4MultiUnion* multiUnion = new G4MultiUnion(name_);

    double TubeR_type_I = 38 * mm;
    double TubeR_type_II = 70 * mm;
    double holeR = TubeR_type_I ;

    G4double distance = 120 *cm;
    G4double TubeR_I = TubeR_type_I;
    G4double TubeR_II = TubeR_type_II;
    G4double TubeR_III = TubeR_type_I;

    G4double holeRadius = holeR;

    G4double y_positionI = 23.5*cm ;
    G4double y_positionII = 0 ;
    G4double y_positionIII = -23.5*cm ;

    bool do_I = true ;
    bool do_II = false ;
    bool do_III = false ;

    if(do_I)   AltWaterDistributerHelper(multiUnion, distance, TubeR_I,   y_positionI   );
    if(do_II)  AltWaterDistributerHelper(multiUnion, distance, TubeR_II,  y_positionII  );
    if(do_III) AltWaterDistributerHelper(multiUnion, distance, TubeR_III, y_positionIII );

    for (int i = 0; i < 8; i++) {
        // Create circular hole geometry : MISNAMED THEY ARE CROSS PIECES IN Y-DIRECTION
        G4Tubs* hole = new G4Tubs("Hole_" + std::to_string(i), 0, holeRadius, (23.5) * cm, 0, 360 * deg);

        G4double theta = i * 45 * deg;
        G4RotationMatrix* rotation = new G4RotationMatrix();
        rotation->rotateX(90 * deg);  // G4Tubs starts on Z-axis, rotate around X by 90 puts on Y axis

        G4ThreeVector position(-distance * cos(theta), 0, distance * sin(theta));
        G4Transform3D tr = G4Transform3D(*rotation, position);

        multiUnion->AddNode(*hole, tr);
    }

    multiUnion->Voxelize();

    return multiUnion ;
}


void U4SolidMaker::AltWaterDistributerHelper(G4MultiUnion* multiUnion, double distance, double TubeR, double yoffset ) // static
{
    G4double rMax = TubeR;
    G4double rMin = 0;
    G4double dz = distance * tan(22.5 * deg);
    G4double sPhi = 0.0;
    G4double dPhi = 2 * M_PI;
    //G4double holeRadius = 33 *mm;

    // Rotating and combining eight cylinders
    G4double rotationAngle = 22.5 * deg;
    for (int i = 0; i < 8; i++) {
        // Calculate the normal vector of the cutting plane
        G4double theta_i = rotationAngle;
        G4double theta_j = rotationAngle;
        G4ThreeVector pLowNorm(sin(theta_i), 0, -cos(theta_j));
        G4ThreeVector pHighNorm(sin(theta_i), 0, cos(theta_j));

        G4CutTubs* cutTube = new G4CutTubs(
            "CutTube_" + std::to_string(i), rMin, rMax, dz, sPhi, dPhi, pLowNorm, pHighNorm);

        //G4Tubs* hole = new G4Tubs("Hole_" + std::to_string(i), 0, holeRadius, TubeR + 1 * cm, 0, 360 * deg);

        G4double theta = i * 45 * deg;
        G4RotationMatrix* rotation = new G4RotationMatrix();
        rotation->rotateY(45 * i * deg);

        G4RotationMatrix* rotation_hole = new G4RotationMatrix();
        rotation_hole->rotateX(90 * deg);

        G4ThreeVector position(-distance * cos(theta), yoffset, distance * sin(theta));
        G4Transform3D tr = G4Transform3D(*rotation, position);

        //G4SubtractionSolid* cutTubeWithHole = new G4SubtractionSolid(
        //    "CutTubeWithHole_" + std::to_string(i), cutTube, hole, rotation_hole, G4ThreeVector(0, 0, 0));


        multiUnion->AddNode(*cutTube, tr);
    }
}


/**
U4SolidMaker::Snake
--------------------

WaterdistributorpipeMaker::MakeWaterDistributorPartIIIUnion

**/


const G4VSolid* U4SolidMaker::WaterDistributorPartIIIUnion(const char* name_) // static
{
    //double m_lowerchimney_rotangle = 0. ;

    // Create individual solid components first
    double flange_r_inner = 66.5*mm;
    double flange_r_outer = 125.*mm;
    int numZplanes_flange = 4;
    G4double zPlane_flange[] = {-50 *mm, -25 *mm , -25 *mm, 0 *mm};
    G4double rInner_flange[] = {0 , 0, 0, 0};
    G4double rOuter_flange[] = {flange_r_inner, flange_r_inner, flange_r_outer , flange_r_outer};
    auto solidFlange = new G4Polycone("sBotChimneySSEnclosure_Flange",
                                    0, 360*deg,
                                    numZplanes_flange, zPlane_flange, rInner_flange, rOuter_flange);

    double dz1 = 529/2 * mm;
    double theta1 = 27*deg;
    G4ThreeVector pLowNorm1( sin(theta1), 0, -cos(theta1) );
    G4ThreeVector pHighNorm1( -sin(theta1), 0, cos(theta1) );
    auto cutTube1 = new G4CutTubs("CutTube1", 0, 66.5 * cos(theta1), dz1, 0, 360*deg, pLowNorm1, pHighNorm1);

    double dz2 = 1125.25/2 * mm;
    double theta2 = -16*deg;
    G4ThreeVector pLowNorm2( sin(theta2), 0, -cos(theta2) );
    G4ThreeVector pHighNorm2( -sin(theta2), 0, cos(theta2) );
    auto cutTube2 = new G4CutTubs("CutTube2", 0, 66.5 * cos(theta2), dz2, 0, 360*deg, pLowNorm2, pHighNorm2);

    double r_tube = 66.5 * mm;
    double r_main1 = 283 * mm;
    double phi_start1 = 180 * deg;
    double phi_delta1 = 135 * deg;
    auto torus1 = new G4Torus("torus1", 0, r_tube, r_main1, phi_start1, phi_delta1);

    double r_main2 = 352 * mm;
    double phi_start2 = 45 * deg;
    double phi_delta2 = 45 * deg;
    auto torus2 = new G4Torus("torus2", 0, r_tube, r_main2, phi_start2, phi_delta2);

    // Component definition arrays
    G4VSolid* solids[] = {solidFlange, cutTube1, cutTube2, torus1, torus2};

    // Rotation angles around X and Y axes
    double rotX_angles[] = {0*deg, 0*deg, 0*deg, -90*deg, 90*deg};
    double rotY_angles[] = {0*deg, 27*deg, -16*deg, 180*deg, 0*deg};

    // Pre-calculate offsets for CutTubes
    double offset1 = dz1 * cos(27*deg);
    double offset2 = dz2 * cos(-16*deg);

    // Position vectors
    G4ThreeVector positions[] = {
        G4ThreeVector(0, 0, -19120),                                                          // Flange
        G4ThreeVector(-120, 0, -19120 - 50*mm - offset1),                                    // CutTube1
        G4ThreeVector(-240 + dz2 * sin(16*deg), 0, -19120 - 50*mm - offset1 * 2 - offset2), // CutTube2
        G4ThreeVector(70 - r_main1, 0, -20724),                                             // Torus1
        G4ThreeVector(70 - 449 - 283, 0, -21181)                                            // Torus2
    };

    // Component names for debugging
    //const char* component_names[] = {"Flange", "CutTube1", "CutTube2", "Torus1", "Torus2"};

    // Create MultiUnion and add all components in a loop
    G4MultiUnion* unionSolid = new G4MultiUnion("WaterDistributorPartIIIUnion");

    for (int i = 0; i < 5; ++i) {
        // Create rotation matrix for this component
        G4RotationMatrix rotation;
        if (rotX_angles[i] != 0) {
            rotation.rotateX(rotX_angles[i]);
        }
        if (rotY_angles[i] != 0) {
            rotation.rotateY(rotY_angles[i]);
        }

        // Create transform and add node
        G4Transform3D transform(rotation, positions[i]);
        unionSolid->AddNode(*solids[i], transform);
    }

    unionSolid->Voxelize();

    // Apply final rotation for the entire assembly
    //G4RotationMatrix finalRotation;
    //finalRotation.rotateZ(-m_lowerchimney_rotangle);

    return unionSolid ;
}



/**
U4SolidMaker::MakeLowerWaterDistributorCurvedCutTubes
-------------------------------------------------------

After WaterdistributorpipeMaker::MakeLowerWaterDistributorCurvedCutTubes


2025-12-17 14:14:52.868 INFO  [2884720] [U4SolidMaker::OuterReflectorOrbSubtraction@2690] m_radOuterReflector 20476
2025-12-17 14:14:52.868 INFO  [2884720] [U4SolidMaker::MakeLowerWaterDistributorCurvedCutTubes@2653]
                                                    offset1    235.226
                                                    offset2    540.830
                                        offset1*2 + offset2   1011.281
                                      offset1*2 + offset2*2   1552.111
 trans_cutTube1 [  -120.000,     0.000,-19405.226]
 trans_cutTube2 [   -84.920,     0.000,-20181.281]



::


    In [4]: z = np.array( [-20476, -20181, -19405, -19170] )

    In [5]: z - z[0]
    Out[5]: array([   0,  295, 1071, 1306])

    In [6]: z - z[-1]
    Out[6]: array([-1306, -1011,  -235,     0])


    20476 - 1552 = 18924

    20476 - 235 - 541 = 19700



Start centered::


                +-----+  -
               /     /
              /     /    offset1
             /     /
            /  +  /     -                   z = 0
           /     /
          /     /       offset1
         /     /
        +-----+         -


Offset by  -19120 - 50*mm = -19170 puts middle at -19170::


                +-----+  -
               /     /
              /     /    offset1
             /     /
            /  +  /     - -  - - - - - -    z = -19170
           /     /
          /     /       offset1
         /     /
        +-----+         -


Then offset more, -19170 - offset1 puts top at -19170::



                            +-----+  -   - - - - - - - - - -   z = -19170
                           /     /
                          /     /    offset1
                         /     /
                        /  +  /     - -  - - - - - -    z = -19170 - offset1              = -19170 - 235.22              = -19405.22
                       /     /
                      /     /       offset1
                     /     /
                    +-----+         -                   z =  -19170 - 2*offset1          = -19170 - 2*235.22             = -19640.44
                     \     \
                      \     \
                       \     \
                        \     \
                         \     \
                          \  +  \                       z = -19170 - 2*offset1 - offset2 = -19170 - 2*235.22 - 540.830  =  -20181.27
                           \     \
                            \     \
                             \     \                                                                                -------- -20476 --
                              \     \
                               \     \
                                \     \
                                 +-----+                z = -19170 - 2*offset2 - 2*offset2 = -19170 - 2*235.22 - 2*540.830 = -20722.1


**/



std::vector<PlacedSolid> U4SolidMaker::MakeLowerWaterDistributorCurvedCutTubes(double m_lowerchimney_rotangle) {
    std::vector<PlacedSolid> cutTubes;

    double UNCOINCIDE_MM = ssys::getenvdouble(U4SolidMaker__MakeLowerWaterDistributorCurvedCutTubes_UNCOINCIDE_MM, 0 );

    double rMin = 0;
    //double rMax = 50*mm;
    double sPhi = 0;
    double dPhi = 360*deg;


    // First section parameters
    double dz1 = 529/2 * mm;
    double theta1 = 27*deg;
    G4ThreeVector pLowNorm1( sin(theta1), 0, -cos(theta1) );
    G4ThreeVector pHighNorm1( -sin(theta1), 0, cos(theta1) );
    auto cutTube1 = new G4CutTubs("CutTube1", rMin, 66.5 * cos(theta1), dz1, sPhi, dPhi, pLowNorm1, pHighNorm1);

    // Second section parameters
    double dz2 = 1125.25/2 * mm;
    double theta2 = -16*deg;
    G4ThreeVector pLowNorm2( sin(theta2), 0, -cos(theta2) );
    G4ThreeVector pHighNorm2( -sin(theta2), 0, cos(theta2) );
    auto cutTube2 = new G4CutTubs("CutTube2", rMin, 66.5 * cos(theta2), dz2, sPhi, dPhi, pLowNorm2, pHighNorm2);

    double offset1 = dz1 * cos(27*deg);
    double offset2 = dz2 * cos(-16*deg);

    double cos_rot = cos(m_lowerchimney_rotangle);
    double sin_rot = sin(m_lowerchimney_rotangle);
    double sin_theta2 = sin(16*deg);

    // Rotation and translation for cutTube1
    G4RotationMatrix rot_cutTube1;
    rot_cutTube1.rotateY(27*deg);
    rot_cutTube1.rotateZ(-m_lowerchimney_rotangle);

    G4ThreeVector trans_cutTube1(-120 * cos_rot, 120 * sin_rot, -19120 - 50*mm - offset1);

    cutTubes.push_back({
        cutTube1,
        trans_cutTube1,
        rot_cutTube1,
        0,
        "lCutTube1",
        "pCutTube1"
    });

    // Rotation and translation for cutTube2
    G4RotationMatrix rot_cutTube2;
    rot_cutTube2.rotateY(-16*deg);
    rot_cutTube2.rotateZ(-m_lowerchimney_rotangle);

    G4ThreeVector trans_cutTube2(
        2 * (-120 * cos_rot) + dz2 * sin_theta2 * cos_rot,
        2 * (120 * sin_rot) - dz2 * sin_theta2 * sin_rot,
        -19120 - 50*mm - offset1 * 2 - offset2 + UNCOINCIDE_MM*mm
    );


    cutTubes.push_back({
        cutTube2,
        trans_cutTube2,
        rot_cutTube2,
        1,
        "lCutTube2",
        "pCutTube2"
    });


    std::array<double,3> tr1 = {trans_cutTube1.x(), trans_cutTube1.y(), trans_cutTube1.z() };
    std::array<double,3> tr2 = {trans_cutTube2.x(), trans_cutTube2.y(), trans_cutTube2.z() };

    LOG(info)
       << "\n"
       << "[" << U4SolidMaker__MakeLowerWaterDistributorCurvedCutTubes_UNCOINCIDE_MM << "]"
       << "\n"
       << std::setw(60) << " UNCOINCIDE_MM " << std::fixed << std::setw(10) << std::setprecision(3) << UNCOINCIDE_MM
       << "\n"
       << std::setw(60) << " offset1 " << std::fixed << std::setw(10) << std::setprecision(3) << offset1
       << "\n"
       << std::setw(60) << " offset2 "  << std::fixed << std::setw(10) << std::setprecision(3) << offset2
       << "\n"
       << std::setw(60) <<  " offset1*2 + offset2 " << std::fixed << std::setw(10) << std::setprecision(3) << (offset1*2.+offset2)
       << "\n"
       << std::setw(60) <<  " offset1*2 + offset2*2 " << std::fixed << std::setw(10) << std::setprecision(3) << (offset1*2.+offset2*2.)
       << "\n"
       << " trans_cutTube1 " << s_bb::Desc_<double,3>(tr1.data())
       << "\n"
       << " trans_cutTube2 " << s_bb::Desc_<double,3>(tr2.data())
       ;

    return cutTubes;
}


/**
U4SolidMaker::OuterReflectorOrbSubtraction
--------------------------------------------

Checking::

    DetSim1Construction::helper_subtract_water_distributor

**/

const G4VSolid* U4SolidMaker::OuterReflectorOrbSubtraction(const char* name_) // static
{
    G4String name = name_ ;

    double m_radOuterWater = 20.50*m - 26*mm; // due to back of veto pmt mask is -25.05 mm (See R12860OnlyFrontMaskManager).
    double m_thicknessReflector = 2*mm;
    double m_radOuterReflector = m_radOuterWater + m_thicknessReflector;


    double radOuterReflector = ssys::getenvfloat(U4SolidMaker__OuterReflectorOrbSubtraction_radOuterReflector, m_radOuterReflector );
    G4VSolid* solid_output = new G4Orb(name_,  radOuterReflector );


    double m_lowerchimney_rotangle = 0. ;

  // ==== New: Subtract two inclined cutTubes of lower water distributor ====
    auto cutTubes = MakeLowerWaterDistributorCurvedCutTubes(m_lowerchimney_rotangle);

    for (size_t i = 0; i < cutTubes.size(); ++i) {
        const auto& tube = cutTubes[i];
        G4Transform3D transform(tube.rot, tube.pos);

        solid_output = new G4SubtractionSolid(
            name + "_cutTube" + std::to_string(i+1),
            solid_output,
            tube.solid,
            transform
        );
    }

    G4String solid_output_name = solid_output->GetName();

    LOG(info)
         << "\n"
         << "m_radOuterReflector " << m_radOuterReflector     // 20476 mm
         << "\n"
         << " [" << U4SolidMaker__OuterReflectorOrbSubtraction_radOuterReflector << "]"
         << "\n"
         << " radOuterReflector " << radOuterReflector
         << "\n"
         << " solid_output_name " << solid_output_name
         ;

    return solid_output ;
}















struct R12860_PMTSolid_Maker
{
    R12860_PMTSolid_Maker(double R1, double R1z, double R2, double R3, double H, double H3);

    G4VSolid* GetSolid(G4String solidname, double thickness=0.0, char X='\0' );

    double m_R1;
    double m_R1z;
    double m_R1p;
    double m_R2;
    double m_R3;
    double m_theta;
    double m_hypotenuse ;
    double m_H;
    double m_H_1_2;

    double m_H3;

    double m_delta_torlerance;

};




R12860_PMTSolid_Maker::R12860_PMTSolid_Maker(
    double R1,
    double R1z,
    double R2,
    double R3,
    double H,
    double H3
    )
    :
    m_R1(R1),    // 101.0
    m_R1z(R1z),  //  75.5
    m_R2(R2),    //  23.0   radius of torus circle
    m_R3(R3),    //  42.5
    m_H(H),      // 220.0
    m_H3(H3)     //  62.0
{
    m_H_1_2 = m_H - m_R1z - m_H3;              //    220 - 75.5 - 62 =  82.5
    m_theta = atan((m_R2+m_R3)/(m_H_1_2));     //    23+42.5 = 65.5   65.6/82.5 = 0.7951 atan(0.7951)
    m_hypotenuse = sqrt(pow(m_H_1_2,2)+pow(m_R2+m_R3,2)) ;
    m_R1p = m_hypotenuse - m_R2;  // hypot distance from ellipse center to torus circle

    m_delta_torlerance = 1E-2*mm;


}


/**
                  ........+.........                          +                +
            ..''''        |         ''''..
        ..''              |               ''..
      .'                  |                   '.            m_R1z
    .'                   r1zt                   '.
   '                      |                       '
  '                       |                        '
 '                        +----------r1t----------- +         +
  '                      /:\                       '
   '                    / : \                     '
    '.                 /  :  \                  .'
      '.              /   :   \               .'             m_H_1_2           m_H
        ''..         /    :    \          ..''
            ''''.   /     :     \    .''''
                  ./      :      \  .   .
                  /               \
                 /                 \
                /                   \
               +    +     +     +    +
                    |           |
                    |           |
                    |           |
                    |           |
                    |           |
                    |           |
                    |           |
                    +-----------+
     Z
     |
     +--X


 symbol : pmttube_solid_tube(scarf)
 name   : R12860_PMTSolidU_pmt_solid_2_Tube
 r4t    : 51.1993
 h2t/2  : 9.00616
 symbol  : pmttube_solid_torus(scarfSub)
 name    : R12860_PMTSolidU_pmt_solid_2_Torus
 r2t     : 22.999
 r2t+r3t : 65.5
 symbol  : pmttube_solid_part2
 name    : R12860_PMTSolidU_pmt_solid_part2
 -h2t/2  : -9.00616(neck-torus-dz)
 symbol  : pmttube_solid_1_2(bulb+neck)
 name    : R12860_PMTSolidU_pmt_solid_1_2
 neck_dz : -73.4938
 total_torus_dz : -82.5
 symbol  : pmttube_solid_1_2_3((bulb+neck)+endtube)
 name    : R12860_PMTSolidU_pmt_solid
 endtube_dz : -113.501


POINT=-65.5,0,-82.5,65.5,0,-82.5 BBOX=-51.19,0,-82.4999,51.19,0,-64.48764,-51.19,0,-64.48764,51.19,0,-82.4999   KLUDGE=1 cxt_min.sh pdb


**/




G4VSolid* R12860_PMTSolid_Maker::GetSolid(G4String solidname, double thickness, char X)
{
    // Calculate Parameter first
    double r1t = m_R1 + thickness;
    double r1zt = m_R1z + thickness;
    double r2t = m_R2 - thickness;
    double r3t = m_R3 + thickness;
    double r1zp = (m_R1p+thickness);
    double r4t = r1zp * sin(m_theta);

    double h1t = r1zp * cos(m_theta);   // above necktop height (up to ellipse equator)
    double h2t = r2t * cos(m_theta);    // below necktop height (down to torus plane line)

    double h3t = m_H3 + thickness;


    double scarf_radius = r4t+m_delta_torlerance ;
    double scarf_halfheight = h2t/2+m_delta_torlerance ;

    double endtube_radius = r3t+m_delta_torlerance ;
    double endtube_halfheight = h3t/2+m_delta_torlerance ;


    std::cout
        << " thickness           : " << std::setw(10) << thickness          << "\n"
        << " m_delta_torlerance    : " << std::setw(10) << m_delta_torlerance   << "\n"
        << " m_R1                : " << std::setw(10) << m_R1               << " (input ellipse width)"                  << "\n"
        << " r1t                 : " << std::setw(10) << r1t                << " (m_R1 + thickness)"                     << "\n"
        << " m_R1z               : " << std::setw(10) << m_R1z              << " (input ellipse height)"                 << "\n"
        << " r1zt                : " << std::setw(10) << r1zt               << " (m_R1z + thickness)"                    << "\n"
        << " m_R2                : " << std::setw(10) << m_R2               << " (input torus circle radius)"            << "\n"
        << " m_R3                : " << std::setw(10) << m_R3               << " (input endtube radius)"                 << "\n"
        << " m_H                 : " << std::setw(10) << m_H                << " (input total height)"                   << "\n"
        << " m_H3                : " << std::setw(10) << m_H3               << " (input endtube height)"                 << "\n"
        << " m_H_1_2             : " << std::setw(10) << m_H_1_2            << " (m_H - m_R1z - m_H3)"                   << "\n"
        << " m_theta             : " << std::setw(10) << m_theta            << " (ellipse ^ torus centers)"              << "\n"
        << " m_hypotenuse        : " << std::setw(10) << m_hypotenuse       << " (dist from ellipse to torus center)"    << "\n"
        << " m_R1p               : " << std::setw(10) << m_R1p              << " (m_hypotenuse - m_R2)"                  << "\n"
        << " r1zp                : " << std::setw(10) << r1zp               << " (m_R1p + thickness)"                    << "\n"
        << " r4t                 : " << std::setw(10) << r4t                << " (r1zp*sin(m_theta))"                    << "\n"
        << " scarf_radius        : " << std::setw(10) << scarf_radius       << " (r4t+m_delta_torlerance)"                 << "\n"
        << " r2t                 : " << std::setw(10) << r2t                << " (m_R2 - thickness)"                     << "\n"
        << " h1t                 : " << std::setw(10) << h1t                << " (r1zp*cos(m_theta)) above neck height " << "\n"
        << " h2t                 : " << std::setw(10) << h2t                << " (r2t*cos(m_theta))  neck h"             << "\n"
        << " scarf_halfheight    : " << std::setw(10) << scarf_halfheight   << " (h2t/2+m_delta_torlerance)"               << "\n"
        << " endtube_radius      : " << std::setw(10) << endtube_radius     << " (r3t+m_delta_torlerance)"                 << "\n"
        << " h3t                 : " << std::setw(10) << h3t                << " (m_H3 + thickness)"                     << "\n"
        << " endtube_halfheight  : " << std::setw(10) << endtube_halfheight << " (h3t/2+m_delta_torlerance)"               << "\n"
        ;



    // Show variables
    G4cout << "r1t: " << r1t/mm << " mm" << G4endl;
    G4cout << "r4t: " << r4t/mm << " mm" << G4endl;
    G4cout << "r2t: " << r2t/mm << " mm" << G4endl;
    G4cout << "r3t: " << r3t/mm << " mm" << G4endl;
    G4cout << "h1t: " << h1t/mm << " mm" << G4endl;
    G4cout << "h2t: " << h2t/mm << " mm" << G4endl;
    G4cout << "h3t: " << h3t/mm << " mm" << G4endl;

    // Construct the PMT Solid
    // * PART 1
    // G4Sphere* pmttube_solid_sphere = new G4Sphere(
    //                                         solidname+"_1_Sphere",
    //                                         0*mm, // R min
    //                                         r1t, // R max
    //                                         0*deg, // Start Phi
    //                                         360*deg, // Delta Phi
    //                                         0*deg, // Start Theta
    //                                         180*deg  // Delta Theta
    //                                         );
    G4Ellipsoid* pmttube_solid_sphere = new G4Ellipsoid(
                                            solidname+"_1_Ellipsoid",
                                            r1t, // pxSemiAxis
                                            r1t, // pySemiAxis
                                            r1zt // pzSemiAxis
                                            );
    // * PART 2  : neck


    G4Tubs* pmttube_solid_tube = new G4Tubs(
                                    solidname+"_2_Tube",
                                    0*mm,  /* inner */
                                    scarf_radius, /* pmt_r */
                                    scarf_halfheight, /* part 2 h */
                                    0*deg,
                                    360*deg);

    std::cout
         << " symbol : pmttube_solid_tube(scarf)" << "\n"
         << " name   : " << pmttube_solid_tube->GetName() << "\n"
         << " r4t    : " << r4t << "\n"
         << " h2t/2  : " << h2t/2 << "\n"
         ;

    G4Torus* pmttube_solid_torus = new G4Torus(
                                        solidname+"_2_Torus",
                                        0*mm,  // R min
                                        r2t+m_delta_torlerance, // R max
                                        (r2t+r3t), // Swept Radius
                                        -0.01*deg,
                                        360.01*deg);

    std::cout
         << " symbol  : pmttube_solid_torus(scarfSub) " << "\n"
         << " name    : " << pmttube_solid_torus->GetName() << "\n"
         << " r2t     : " << r2t << "\n"
         << " r2t+r3t : " << r2t+r3t << "\n"
         ;





    [[maybe_unused]] G4VSolid* pmttube_solid_part2 = nullptr ;

    G4VSolid* neck = nullptr ;

    if( X == 'U' )
    {
        pmttube_solid_part2 = new G4UnionSolid(
                                            solidname+"_part2",
                                            pmttube_solid_tube,
                                            pmttube_solid_torus,
                                            0,
                                            G4ThreeVector(0,0,-h2t/2)
                                            );

        neck = pmttube_solid_part2 ;   // debug union : allows to see the position of the torus relative to the rest
    }
    else if( X == 'K' )
    {
        neck = pmttube_solid_tube ;  // KLUDGE : DONT SUBTRACT TORUS
    }
    else if( X == 'P' )
    {
        G4double phiStart = 0.00*deg ;
        G4double phiTotal = 360.00*deg ;
        G4int numZPlanes = 2 ;
        G4double zPlane[] = { -scarf_halfheight, scarf_halfheight  } ;
        G4double rInner[] = {  0.0             , 0.0   } ;
        G4double rOuter[] = {  endtube_radius  , scarf_radius } ;

        G4Polycone* polycone_neck = new G4Polycone(
                                   solidname+"_part2",
                                   phiStart,
                                   phiTotal,
                                   numZPlanes,
                                   zPlane,
                                   rInner,
                                   rOuter
                                   );

        neck = polycone_neck ;
    }
    else
    {
        pmttube_solid_part2 = new G4SubtractionSolid(
                                            solidname+"_part2",
                                            pmttube_solid_tube,
                                            pmttube_solid_torus,
                                            0,
                                            G4ThreeVector(0,0,-h2t/2)
                                            );
        neck = pmttube_solid_part2 ;   // original monstrosity
    }

    std::cout
         << " symbol  : pmttube_solid_part2 " << "\n"
         << " name    : " << ( pmttube_solid_part2 ? pmttube_solid_part2->GetName() : "-" ) << "\n"
         << " -h2t/2  : " << -h2t/2 << "(neck-torus-dz)\n"
         ;


    // * PART 3  : EndTube



    G4Tubs* pmttube_solid_end_tube = new G4Tubs(
                                    solidname+"_3_EndTube",
                                    0*mm,  /* inner */
                                    endtube_radius, //21*cm/2, /* pmt_r */
                                    endtube_halfheight, //30*cm/2, /* pmt_h */
                                    0*deg,
                                    360*deg);



    // * PART 1 + 2   : bulb + neck
    G4UnionSolid* pmttube_solid_1_2 = new G4UnionSolid(
                                            solidname+"_1_2",
                                            pmttube_solid_sphere,
                                            neck,
                                            0,
                                            G4ThreeVector(0, 0, -(h1t+h2t/2))
                                            );

     std::cout
          << " symbol  : pmttube_solid_1_2(bulb+neck) \n"
          << " name    : " << pmttube_solid_1_2->GetName() << "\n"
          << " neck_dz : " << -(h1t+h2t/2) << "\n"
          << " total_torus_dz : " << -h2t/2 + -(h1t+h2t/2) << "\n"
          ;

    // * PART 1+2 + 3
    [[maybe_unused]] G4UnionSolid* pmttube_solid_1_2_3 = new G4UnionSolid(
                                            solidname,
                                            pmttube_solid_1_2,
                                            pmttube_solid_end_tube,
                                            0,
                                            G4ThreeVector(0,0,
                                                -(m_H_1_2+h3t*0.50))
                                            );

     std::cout
          << " symbol  : pmttube_solid_1_2_3((bulb+neck)+endtube) \n"
          << " name    : " << pmttube_solid_1_2_3->GetName() << "\n"
          << " endtube_dz : " << -(m_H_1_2+h3t*0.50) << "\n"
          ;


    return pmttube_solid_1_2_3;
}



const G4VSolid* U4SolidMaker::R12860_PMTSolid(const char* name_) // static
{
    const char* PREFIX = "R12860_PMTSolid" ;
    assert( sstr::StartsWith(name_,PREFIX ));
    const char* suffix = name_ + strlen(PREFIX);
    G4String solidname = name_ ;

    // Ham8inchPMTManager::init_variables

    double m_pmt_r, m_pmt_h, m_z_equator ;

    m_pmt_r = 101.*mm;
    m_pmt_h = 220.*mm;
    m_z_equator = 75.5*mm; // From top to equator

    R12860_PMTSolid_Maker* m_pmtsolid_maker = new R12860_PMTSolid_Maker(
                            m_pmt_r,       // m_R1     101.0
                            m_z_equator,   // m_R1z     75.5
                            23*mm,         // m_R2      23.0
                            42.5*mm,       // m_R3      42.5
                            m_pmt_h,       // m_H       220.0
                            62.*mm         // m_H3      62.0
                            );




    double thickness = 1E-3*mm ;

    bool with_suffix = suffix && ( suffix[0] == 'U' || suffix[0] == 'K' || suffix[0] == 'P' ) ;

    return m_pmtsolid_maker->GetSolid(solidname + "_pmt_solid", thickness, ( with_suffix ? suffix[0] : '\0' ) );
}

