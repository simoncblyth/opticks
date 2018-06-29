#include <iostream>

#include "G4Hype.hh"
#include "G4Ellipsoid.hh"
#include "G4Torus.hh"
#include "G4Cons.hh"
#include "G4Trd.hh"
#include "G4Tubs.hh"
#include "G4Sphere.hh"
#include "G4Orb.hh"
#include "G4Box.hh"
#include "G4VisExtent.hh"
#include "G4SubtractionSolid.hh"
#include "G4UnionSolid.hh"


#include "X4Solid.hh"
#include "X4Mesh.hh"
#include "NNode.hpp"
#include "BStr.hh"
#include "NCSG.hpp"
#include "GParts.hh"
#include "GMaterialLib.hh"
#include "GSurfaceLib.hh"
#include "GBndLib.hh"
#include "Opticks.hh"
#include "SSys.hh"

#include "OPTICKS_LOG.hh"



void test_solid(G4VSolid* so)
{
    G4VisExtent vx = so->GetExtent() ; 
    std::cout << vx << std::endl ; 

    X4Solid* xs = new X4Solid(so) ; 
    LOG(info) << xs->desc() ; 
    nnode* root = xs->root(); 
    assert( root ) ; 
    root->update_gtransforms();
    root->dump();

    NCSG* csg = NCSG::FromNode( root, NULL ); 

    Opticks* ok = new Opticks(0,0);
    GMaterialLib* mlib = new GMaterialLib(ok); 
    GSurfaceLib* slib = new GSurfaceLib(ok); 
    GBndLib* blib = new GBndLib(ok, mlib, slib);
    blib->closeConstituents();

    GParts* pts = GParts::make( csg, "Air///Water", 1 );  
    pts->setBndLib(blib); 

    GParts* cpts = GParts::combine(pts); 

    const char* path = "/tmp/X4SolidTest/GParts" ;
    cpts->save(path);
    SSys::run(BStr::concat("prim.py ", path, NULL )); 


    // X4Mesh* xm = new X4Mesh(so) ; 
    // xm->save(BStr::concat("/tmp/X4SolidTest/",so->GetName().c_str(),".gltf")); 

}

void test_G4Sphere()
{
    G4Sphere* a = X4Solid::MakeSphere("sphere_without_inner", 100.f, 0.f );  test_solid(a) ; 
    G4Sphere* b = X4Solid::MakeSphere("sphere_with_inner", 100.f, 50.f );    test_solid(b) ; 
    G4Sphere* c = X4Solid::MakeZSphere("zsphere_without_inner", 100.f, 0.f, 0.f, 90.f ); test_solid(c) ; 
    G4Sphere* d = X4Solid::MakeZSphere("zsphere_with_inner", 100.f, 50.f, 0.f, 90.f );   test_solid(d) ; 
}

void test_G4Orb()
{
    G4Orb* a = X4Solid::MakeOrb("orb", 100.f );  test_solid(a) ; 
}

void test_G4Box()
{
    G4Box* a = new G4Box("box", 100., 200., 300. );  test_solid(a) ; 
}

void test_G4Tubs()
{
    G4Tubs* a = X4Solid::MakeTubs("tubs"                    ,  0.f, 100.f, 50.f ) ;  test_solid(a) ; 
    G4Tubs* b = X4Solid::MakeTubs("tubs_with_inner"        ,  50.f, 100.f, 50.f ) ;  test_solid(b) ; 
    G4Tubs* c = X4Solid::MakeTubs("tubs_with_segment"      ,   0.f, 100.f, 50.f, 0.f, 90.f ) ;  test_solid(c) ; 
    G4Tubs* d = X4Solid::MakeTubs("tubs_with_segment_inner",  50.f, 100.f, 50.f, 0.f, 90.f ) ;  test_solid(d) ; 
}

void test_G4Trd()
{
    G4Trd* a = X4Solid::MakeTrapezoidCube("trapezoid_cube", 100.f );   test_solid(a) ; 
}

void test_G4Cons()
{
    // MakeCone(const char* name, float z, float rmax1, float rmax2, float rmin1=0.f, float rmin2=0.f, float startphi=0.f, float deltaphi=360.f );

    G4Cons* a = X4Solid::MakeCone("cone",                 200.f, 400.f, 100.f  );                test_solid(a) ; 
    G4Cons* b = X4Solid::MakeCone("cone_with_inner",      200.f, 400.f, 100.f, 350.f, 50.f  );   test_solid(b) ; 
    G4Cons* c = X4Solid::MakeCone("cone_with_inner_segm", 200.f, 400.f, 100.f, 350.f, 50.f, 0.f, 90.f  );   test_solid(c) ; 
}


void test_G4Torus()
{
    G4Torus* a = X4Solid::MakeTorus("torus", 100.f , 10.f ) ;   test_solid(a) ; 
}

void test_G4Ellipsoid()
{
    G4Ellipsoid* a = X4Solid::MakeEllipsoid("ellipsoid", 100.f , 150.f , 50.f ) ;   test_solid(a) ; 
}

void test_G4Hype()
{
       //  float rmin , float rmax, float inst, float outst, float hz 
    G4Hype* a = X4Solid::MakeHyperboloid("hyperboloid", 100.f , 200.f , 45.f, 45.f,  100.f ) ;   test_solid(a) ; 
}

 
void test_intersectWithPhiSegment()
{
    G4String name = "intersectWithPhiSegment" ; 

    G4double rmin = 0. ; 
    G4double rmax = 100 ; 

    G4double startPhi = 0. ; 
    G4double deltaPhi = 180. ;
    //G4double deltaPhi = 90. ;
    //G4double deltaPhi = 360. ;

    G4double startTheta = 0. ; 
    //G4double startTheta = 30. ; 
    //G4double deltaTheta = 30. ;
    //G4double deltaTheta = 180. ;
    G4double deltaTheta = 180. ;


    G4Sphere* sp = X4Solid::MakeZSphere(name, 
             rmin, 
             rmax, 
             startPhi, 
             deltaPhi,
             startTheta, 
             deltaTheta
         );
  
    test_solid(sp); 
}


void test_union_of_two_differences()
{
    G4RotationMatrix* rotMatrix = NULL ; 
    G4ThreeVector right(0.5,0,0);
    G4ThreeVector left(-0.5,0,0);

    G4VSolid* box1 = new G4Box("box1",1.,1.,1.) ;
    G4VSolid* orb1 = new G4Orb("orb1",1.) ;
    G4VSolid* sub1 = new G4SubtractionSolid("sub1", box1, orb1, rotMatrix, right ); 

    G4VSolid* box2 = new G4Box("box2",1.,1.,1.) ;
    G4VSolid* orb2 = new G4Orb("orb2",1.) ;
    G4VSolid* sub2 = new G4SubtractionSolid("sub2", box2, orb2, rotMatrix, left  ); 

    G4VSolid* uni1 = new G4UnionSolid("uni1", sub1, sub2 ); 

    test_solid(uni1);
}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    //test_G4Sphere();
    //test_G4Orb();
    //test_G4Box();
    //test_G4Tubs();
    //test_G4Trd();
    //test_G4Cons();
    //test_G4Torus();
    //test_G4Ellipsoid();
    //test_G4Hype();
    //test_intersectWithPhiSegment();
    test_union_of_two_differences();

    return 0 ; 
}



