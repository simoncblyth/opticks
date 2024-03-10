/**
U4Mesh_test.cc
===============

::
 
   ~/o/u4/tests/U4Mesh_test.sh 

**/

#include <cmath>
#include <cstdlib>

#include "G4Torus.hh"
#include "G4Orb.hh"
#include "G4Box.hh"
#include "G4Tet.hh"

#include "U4Mesh.h"


int main()
{
    char* SOLID = getenv("SOLID"); 
    if(!SOLID) return 0 ; 

    //double extent = 100. ; 
    double extent = 1. ; 

    G4VSolid* solid = nullptr ; 
    if(     strcmp(SOLID,"Orb")==0)
    {
        solid = new G4Orb("Orb", extent) ; 
    }
    else if(strcmp(SOLID,"Box")==0)
    {
        solid = new G4Box("Box", extent, extent, extent ) ; 
    } 
    else if(strcmp(SOLID,"Torus")==0) 
    {
        double rmin = 0. ; 
        double rmax = extent/2. ; 
        double rtor = extent ;
        double sphi = 0. ; 
        double dphi = 2.*M_PI ; 
        solid = new G4Torus("Torus", rmin, rmax, rtor, sphi, dphi) ; 
    }
    else if(strcmp(SOLID,"Tet")==0) 
    {
        double e = extent ; 
        G4ThreeVector p1(e,  e,  e); 
        G4ThreeVector p2(e, -e, -e); 
        G4ThreeVector p3(-e, e, -e); 
        G4ThreeVector p4(-e, -e, e); 
        G4bool degenerate = false ; 
        solid = new G4Tet("Tet", p1, p2, p3, p4, &degenerate ) ; 
        assert( !degenerate ); 
    }


    if(!solid) return 0 ; 
    
    U4Mesh::Save(solid, "$FOLD/$SOLID") ;

    return 0 ; 
};
