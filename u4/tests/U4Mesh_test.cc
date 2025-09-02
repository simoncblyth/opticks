/**
U4Mesh_test.cc
===============

::

   ~/o/u4/tests/U4Mesh_test.sh
   ~/o/u4/tests/U4Mesh_test.sh build_all
   ~/o/u4/tests/U4Mesh_test.sh info

**/

#include <cmath>
#include <cstdlib>

#include "G4Polyhedron.hh"
#include "G4Torus.hh"
#include "G4Orb.hh"
#include "G4Box.hh"
#include "G4Tet.hh"
#include "G4Cons.hh"
#include "G4Tubs.hh"

#include "U4Mesh.h"


int main()
{
    char* SOLID = getenv("SOLID");
    if(!SOLID) return 0 ;

    double extent = 100. ;
    //double extent = 1. ;

    //G4Polyhedron::SetNumberOfRotationSteps(24);

    G4VSolid* solid = nullptr ;
    if(     strcmp(SOLID,"Orb")==0)
    {
        solid = new G4Orb(SOLID, extent) ;
    }
    else if(strcmp(SOLID,"Box")==0)
    {
        solid = new G4Box(SOLID, extent, extent, extent ) ;
    }
    else if(strcmp(SOLID,"Cons")==0)
    {
        double rmin1 = 0. ;
        double rmax1 = extent*0.5 ;
        double rmin2 = 0. ;
        double rmax2 = extent ;
        double dz = extent ;
        double sphi = 0. ;
        double dphi = 2.*M_PI ;

        solid = new G4Cons(SOLID,rmin1, rmax1, rmin2, rmax2, dz, sphi, dphi );
    }
    else if(strcmp(SOLID,"Tubs")==0)
    {
        double rmin = 0. ;
        double rmax = extent ;
        double dz = extent ;
        double sphi = 0. ;
        double dphi = 2.*M_PI ;

        solid = new G4Tubs(SOLID,rmin, rmax, dz, sphi, dphi );
    }
    else if(strcmp(SOLID,"Torus")==0)
    {
        double rmin = 0. ;
        double rmax = extent/2. ;
        double rtor = extent ;
        double sphi = 0. ;
        double dphi = 2.*M_PI ;
        solid = new G4Torus(SOLID, rmin, rmax, rtor, sphi, dphi) ;
    }
    else if(strcmp(SOLID,"Tet")==0)
    {
        double e = extent ;
        G4ThreeVector p1(e,  e,  e);
        G4ThreeVector p2(e, -e, -e);
        G4ThreeVector p3(-e, e, -e);
        G4ThreeVector p4(-e, -e, e);
        G4bool degenerate = false ;
        solid = new G4Tet(SOLID, p1, p2, p3, p4, &degenerate ) ;
        assert( !degenerate );
    }


    if(!solid) return 0 ;

    U4Mesh::Save(solid, "$FOLD/$SOLID") ;

    return 0 ;
};
