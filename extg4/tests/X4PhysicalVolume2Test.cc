/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

// x4-;x4-c;om-;TEST=X4PhysicalVolume2Test om-t

#include "G4Trd.hh"
#include "G4Cons.hh"
#include "G4Orb.hh"
#include "G4Box.hh"
#include "G4SubtractionSolid.hh"
#include "G4UnionSolid.hh"

#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"

#include "OpNoviceDetectorConstruction.hh"
#include "LXe_Materials.hh"

#include "X4Solid.hh"
#include "X4PhysicalVolume.hh"
#include "X4MaterialTable.hh"

#include "GMaterialLib.hh"
#include "GGeo.hh"
#include "GVolume.hh"
#include "GParts.hh"

#include "NCSG.hpp"
#include "NNode.hpp"
#include "NBBox.hpp"

#include "Opticks.hh"
#include "SDirect.hh"
#include "OPTICKS_LOG.hh"


/**

check_transforms
------------------

Checking the glTF in GLTFSceneKitSample.  

* two Orbs appear horizontally left-right with the smaller to the right, 
  x-axis to right.

* making the world thinner in Z, shows +Z is out the screen

        Y 
        |
        |
        | 
        +-----> X
       /
      /
     Z

**/


G4VPhysicalVolume* check_placement(const LXe_Materials& lm)
{
    G4LogicalVolume* mo_0 = NULL ;   
    G4VSolid* so_0 = new G4Box("World",200.,200.,200.) ; 
    G4LogicalVolume* lv_0 = new G4LogicalVolume(so_0,lm.fAir,"World",0,0,0);
    G4VPhysicalVolume* pv_0 = new G4PVPlacement(0,G4ThreeVector(),lv_0, "World",mo_0,false,0);


    G4LogicalVolume* mo_i = lv_0 ;   
    G4double pRmin1 = 0 ; 
    G4double pRmax1 = 5 ; 
    G4double pRmin2 = 0 ; 
    G4double pRmax2 = 0 ;
    G4double pDz = 5 ; 
    G4double pSPhi = 0 ;  
    G4double pDPhi = 2.*CLHEP::pi ;
    G4VSolid* so_Cone = new G4Cons( "cone", pRmin1, pRmax1, pRmin2, pRmax2, pDz, pSPhi, pDPhi  );
    G4LogicalVolume* lv_Cone = new G4LogicalVolume(so_Cone,lm.fAir,"lv_Cone",0,0,0);

    G4VSolid* so_Box = new G4Box("Box",10.,3.,1.) ; 
    G4LogicalVolume* lv_Box = new G4LogicalVolume(so_Box,lm.fGlass,"lv_Box",0,0,0);


    G4ThreeVector y_axis(0,1,0);   
    //G4RotationMatrix* rot_into_xy = new G4RotationMatrix(y_axis, -90.f*CLHEP::pi/180.) ; 
    // rotate -90 degrees about y axis, points old +z axis (cone pointing direction) 
    // onto -x direction (to the left)
    //G4Transform3D* txf = new G4Transform3D(*rot_into_xy, G4ThreeVector(0,0,0) ); 
    G4ThreeVector z_axis(0,0,1);   


    // ring of cones, initially pointing in z 
    float radius = 90. ;   
    unsigned nplace = 16 ; 
    for( unsigned i = 0 ; i < nplace  ; i++)
    {
        float phi = float(i)/float(nplace)*2.0*CLHEP::pi ; 
        float sphi = std::sin(phi) ; 
        float cphi = std::cos(phi) ; 

        float z = 0 ; 
        float x = radius*cphi ; 
        float y = radius*sphi ; 

        //G4ThreeVector axis(sphi,cphi,0);  // tangent around the ring axis

        G4RotationMatrix* rot_orient = new G4RotationMatrix(z_axis, phi  ) ; 
        //G4RotationMatrix rot = (*rot_orient)*(*rot_into_xy) ;

        G4LogicalVolume* lv_i = i % 2 == 0 ? lv_Cone : lv_Box ; 
        G4VPhysicalVolume* pv_i = new G4PVPlacement(rot_orient,G4ThreeVector(x,y,z),lv_i, "ConePV",mo_i,false,0);
        assert( pv_i ) ; 
    }
    G4VPhysicalVolume* top = pv_0 ; 
    return top ;  
}


G4VPhysicalVolume* check_transforms(const LXe_Materials& lm)
{
    G4LogicalVolume* mo_0 = NULL ;   
    G4VSolid* so_0 = new G4Box("World",1.,1.,0.1) ; 
    G4LogicalVolume* lv_0 = new G4LogicalVolume(so_0,lm.fAir,"World",0,0,0);
    G4VPhysicalVolume* pv_0 = new G4PVPlacement(0,G4ThreeVector(),lv_0, "World",mo_0,false,0);

    G4LogicalVolume* mo_1 = lv_0 ;   
    G4VSolid* so_1 = new G4Orb("Orb",0.3) ; 
    G4LogicalVolume* lv_1 = new G4LogicalVolume(so_1,lm.fAir,"Orb",0,0,0);
    G4VPhysicalVolume* pv_1 = new G4PVPlacement(0,G4ThreeVector(0.3,0.1,0),lv_1, "Orb",mo_1,false,0);
    assert( pv_1 );

    G4LogicalVolume* mo_2 = lv_0 ;   
    G4VSolid* so_2 = new G4Orb("Orb",0.4) ; 
    G4LogicalVolume* lv_2 = new G4LogicalVolume(so_2,lm.fAir,"Orb",0,0,0);
    G4VPhysicalVolume* pv_2 = new G4PVPlacement(0,G4ThreeVector(-0.3,0,0),lv_2, "Orb",mo_2,false,0);
    assert( pv_2 );

    G4VPhysicalVolume* top = pv_0 ; 
    return top ;  
}

G4VPhysicalVolume* check_subtraction(const LXe_Materials& lm)
{
    G4LogicalVolume* mo_0 = NULL ;   

    G4VSolid* box = new G4Box("Box",1.,1.,0.1) ; 
    G4VSolid* orb = new G4Orb("Orb",0.3) ; 

    G4VSolid* so_0 = new G4SubtractionSolid("box-orb", box, orb ); 
    G4LogicalVolume* lv_0 = new G4LogicalVolume(so_0,lm.fAir,"World",0,0,0);
    G4VPhysicalVolume* pv_0 = new G4PVPlacement(0,G4ThreeVector(),lv_0, "World",mo_0,false,0);
    G4VPhysicalVolume* top = pv_0 ; 
    return top ;  
}

G4VPhysicalVolume* check_subtraction_displaced(const LXe_Materials& lm)
{
    // from glTF perspective this yields just World and single subtraction solid

    G4LogicalVolume* mo_0 = NULL ;   

    G4VSolid* box = new G4Box("Box",1.,1.,0.1) ; 
    G4VSolid* orb = new G4Orb("Orb",0.3) ; 

    G4RotationMatrix* rotMatrix = NULL ; 
    G4ThreeVector transVector(0.5,0,0);

    G4VSolid* so_0 = new G4SubtractionSolid("box-orb", box, orb, rotMatrix, transVector ); 
    G4LogicalVolume* lv_0 = new G4LogicalVolume(so_0,lm.fAir,"World",0,0,0);
    G4VPhysicalVolume* pv_0 = new G4PVPlacement(0,G4ThreeVector(),lv_0, "World",mo_0,false,0);
    G4VPhysicalVolume* top = pv_0 ; 
    return top ;  
}

G4VPhysicalVolume* check_union(const LXe_Materials& lm)
{
    G4LogicalVolume* mo_0 = NULL ;   

    G4VSolid* box = new G4Box("Box",1.,1.,0.1) ; 
    G4VSolid* orb = new G4Orb("Orb",0.3) ; 

    G4VSolid* so_0 = new G4UnionSolid("box+orb", box, orb ); 
    G4LogicalVolume* lv_0 = new G4LogicalVolume(so_0,lm.fAir,"World",0,0,0);
    G4VPhysicalVolume* pv_0 = new G4PVPlacement(0,G4ThreeVector(),lv_0, "World",mo_0,false,0);
    G4VPhysicalVolume* top = pv_0 ; 
    return top ;  
}


G4VPhysicalVolume* check_convexpolyhedron(const LXe_Materials& lm)
{
    G4LogicalVolume* mo_0 = NULL ;   

    G4VSolid* so_0 = X4Solid::MakeTrapezoidCube("trapezoid_cube", 100.f );
    G4LogicalVolume* lv_0 = new G4LogicalVolume(so_0,lm.fAir,"World",0,0,0);
    G4VPhysicalVolume* pv_0 = new G4PVPlacement(0,G4ThreeVector(),lv_0, "World",mo_0,false,0);
    G4VPhysicalVolume* top = pv_0 ; 
    return top ;  
}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    LXe_Materials lm ; 

    //G4VPhysicalVolume* top = check_transforms(lm);
    //G4VPhysicalVolume* top = check_subtraction(lm);
    //G4VPhysicalVolume* top = check_union(lm);
    //G4VPhysicalVolume* top = check_subtraction_displaced(lm);
    G4VPhysicalVolume* top = check_convexpolyhedron(lm);
    //G4VPhysicalVolume* top = check_placement(lm);

    GGeo* ggeo = X4PhysicalVolume::Convert(top) ;   
    assert(ggeo);  
    
    LOG(info) << " ggeo.getNumVolumes " << ggeo->getNumVolumes() ; 

    const GVolume* vol = ggeo->getVolume(0); 
    assert(vol); 
    LOG(info) << " vol.getNumChildren " << vol->getNumChildren() ; 




    GParts* parts = vol->getParts();

    if(parts == NULL )
    {
        LOG(fatal) << "THIS TEST NEEDS UPDATING FOR DEFERRED GParts CREATION" ; 
        return 0 ;  
    }


    parts->setVerbosity(5); 


    const NCSG* csg = parts->getCSG();
    LOG(info) << " csg.smry " << csg->smry() ;

    nnode* root = csg->getRoot();
    root->update_gtransforms();
    root->dump(); 

    LOG(info) << " root->planes.size() " << root->planes.size() ;


    // primBuffer creation expects a combine, not a single one
    GParts* pts_c = GParts::Combine( parts ); 
    pts_c->close();
    pts_c->save("$TMP/X4/X4PhysicalVolume2Test");

    //Opticks* ok = Opticks::GetOpticks();
    //ok->Summary();

    return 0 ; 
}

