#include <iostream>

#include "G4SystemOfUnits.hh"
#include "G4Sphere.hh"

#include "BStr.hh"

#include "NNode.hpp"
#include "NSphere.hpp"
#include "NZSphere.hpp"
#include "NTreeAnalyse.hpp"
#include "NNodeCollector.hpp"

#include "nmat4triple.hpp"

using CLHEP::pi ; 

#include "convertSphereTest.hh"

nnode* convertSphereTest::MakePhiMask( float radius, float deltaPhi, float centerPhi )
{
    nnode* phiMask = nullptr ; 

    double safetyRadius = 1.01 * radius;
      
    nnode* hemisphereBase = make_zsphere( 0.f, 0.f, 0.f, safetyRadius, 0.f, safetyRadius) ;
    // SCB: this is not the base its the upper half from z=0 to z=safetyRad

    float phiShift = deltaPhi - 180.f;
    //determines angle to rotate second hemisphere to make subtractive wedge
      
    if( phiShift == 180.f )
    {
        phiMask = hemisphereBase;
    } 
    else  
    {
        nnode* hemisphereCutter = make_zsphere( 0.f, 0.f, 0.f, safetyRadius, 0, safetyRadius) ;

        const nmat4triple* rotHemisphere = nmat4triple::make_rotate(1.0, 0.f , 0.f, phiShift);
        //rotation matrix based around X for phiShift degrees

        //ROTATE hemisphereCutter TRANSFORM HERE
        //matrix applied to hemisphere transform 

        nnode* wedge = nnode::make_operator(CSG_INTERSECTION, hemisphereBase, hemisphereCutter);
        //takes intersection of two hemispheres to create wedge of angle

        bool is_reflex = deltaPhi > 180.f && deltaPhi < 360.f ;

        float centreCorrect = is_reflex ? -90.f - 0.5 * deltaPhi : -90.f + 0.5 * deltaPhi ; 

        // if not rotate opposite way
        //ANGLES HERE MAY BE OVERCOMPENSATING, SHOULD CHECK DURING TESTING

        const nmat4triple* rotCentre = nmat4triple::make_rotate(1.0, 0.f , 0.f, centreCorrect);
       //creates rotation matrix to orient wedge correctly
  
       //APPLY rotCentre MATRIX TO wedge->transform
       //orients wedge to correspond with centre at 0, shifting to centre if deltaPhi <180, and opposite if >180

       //if less than 180 degrees, wedge will suffice

       nnode* full = make_sphere( 0.f, 0.f, 0.f, safetyRadius );   
       nnode* reflexMask =  nnode::make_operator(CSG_DIFFERENCE, full, wedge); 

       phiMask = is_reflex ? reflexMask : wedge ; 

       //if greater than 180, phi segment given as the subtraction of wedge from a full sphere
   }
   return phiMask ; 
}

nnode* convertSphereTest::convertSphereLucas()  //
{ 
    const G4Sphere* const solid = static_cast<const G4Sphere*>(m_solid);

    float rmin = solid->GetInnerRadius()/mm ; 
    float radius = solid->GetOuterRadius()/mm ; 

    nnode* cn = make_sphere( 0.f, 0.f, 0.f, radius );
    cn->label = BStr::concat(m_name, "_nsphere", NULL ) ; 
    
    bool has_inner = 0.f < rmin && rmin < radius ; 
    nnode* inner = has_inner ? make_sphere( 0.f, 0.f, 0.f, rmin) : NULL ;  

    nnode* ret = has_inner ? nnode::make_operator(CSG_DIFFERENCE, cn, inner) : cn ; 
    if(has_inner) ret->label = BStr::concat(m_name, "_ndifference", NULL ) ;
    
    float deltaTheta = solid->GetDeltaThetaAngle()/degree ;
    float startTheta = solid->GetStartThetaAngle()/degree ;
    bool cutTheta = deltaTheta < 180.f ;
     
    float deltaPhi = solid->GetDeltaPhiAngle()/degree ; 
    float startPhi = solid->GetStartPhiAngle()/degree ;

    float centerPhi = 0.5 * deltaPhi + startPhi ;
    // SCB: different quantities -> different names :  startPhi -> centerPhi
     
    const nmat4triple* starterAdjust = nmat4triple::make_rotate(1.0, 0.f , 0.f, startPhi);

    //INSERT ROTATE phiMask TO ADJUST FOR startPhi HERE
    //rotates phiMask to align correctly with G4
    const nmat4triple* phiAlign = nmat4triple::make_rotate(0.f, 0.f , 1.0, 90); 
    //NEED TO CHECK AXES ARE CORRECT HERE

    //INSERT ROTATE phiMask TO ALIGN WITH PHI HERE
    //rotates phiMask to align correctly with axis

    nnode* phiMask = deltaPhi < 360.f ? MakePhiMask(radius, deltaPhi, centerPhi) : nullptr  ;

    nnode* globalMask = NULL; //initialises resultant holder for theta-phi mask



    if(cutTheta)
    {
        float rTheta = startTheta ;
        float lTheta = startTheta + deltaTheta ;

        double zmin = radius*std::cos(lTheta*CLHEP::pi/180.) ;
        double zmax = radius*std::cos(rTheta*CLHEP::pi/180.) ;

        // WARNING SHAPE DOES NOT MATCH THE G4Sphere THETA SEGMENT CONES 

        nnode* thetaMask = make_zsphere( 0.f, 0.f, 0.f, radius, zmin, zmax ) ;
        thetaMask->label = BStr::concat(m_name, "_nzsphere", NULL) ; 
        
        //set globalMask as intersection of theta and phi masks 
        globalMask = phiMask!=NULL ? nnode::make_operator(CSG_INTERSECTION, thetaMask, phiMask) : thetaMask ;

      } else {
         //no trimming needed for theta=180 so globalMask is just phiMask
        globalMask = phiMask;

        //if deltaPhi = 360, phi mask will still be null, so maintains full sphere
    }
    
    nnode* result = globalMask != NULL ? nnode::make_operator(CSG_INTERSECTION, ret, globalMask) : ret ; 
    return result ; 
}


int main(int argc, char** argv)
{

    double theta_start = 0. ; 
    double theta_delta = 0.25 ; 
    double phi_start = 0. ; 
    double phi_delta = 0.25 ; 

    G4String pName = "sp" ; 
    G4double pRmin = 50. ; 
    G4double pRmax = 100. ; 
    G4double pSPhi = phi_start*pi ;    
    G4double pDPhi = phi_delta*pi ; 
    G4double pSTheta = theta_start*pi ; 
    G4double pDTheta = theta_delta*pi ;  

    G4VSolid* solid = new G4Sphere(pName, pRmin, pRmax, pSPhi, pDPhi, pSTheta, pDTheta ); 
    const G4String name = solid->GetName(); 

    convertSphereTest t ; 
    t.m_solid = solid ;  
    t.m_name = strdup(name.c_str()); 

    nnode* n = t.convertSphereLucas(); 
    assert(n); 

    NTreeAnalyse<nnode> ana(n); 
    ana.nodes->dump() ; 

    std::cout << ana.desc() << std::endl ;  

    return 0 ; 
}

