
#include "NGLMExt.hpp"
#include "NZSphere.hpp"
#include "NSphere.hpp"
#include "NCylinder.hpp"
#include "NTorus.hpp"
#include "NCone.hpp"

#include "NSolid.hpp"

// X4Solid::convertEllipsoid
nnode* NSolid::createEllipsoid( const char* name, float ax, float by, float cz, float zcut1, float zcut2 )
{
    float z1 = zcut1 != 0.f && zcut1 > -cz ? zcut1 : -cz ;
    float z2 = zcut2 != 0.f && zcut2 <  cz ? zcut2 :  cz ;
    assert( z2 > z1 ) ;

    glm::vec3 scale( ax/cz, by/cz, 1.f) ;


    /*
    // for subsequent setting of zcuts its easier to always use zsphere
    // otherwise would need to promote sphere to zsphere
  
    bool zslice = z1 > -cz || z2 < cz ; 
    nnode* n = zslice ? 
                          (nnode*)make_zsphere( 0.f, 0.f, 0.f, cz, z1, z2 ) 
                       :
                          (nnode*)make_sphere( 0.f, 0.f, 0.f, cz ) 
                       ;
    */

    nnode* n = (nnode*)make_zsphere( 0.f, 0.f, 0.f, cz, z1, z2 ) ;


    n->label = strdup(name) ; 
    n->transform = nmat4triple::make_scale( scale );

    return n ; 
}

// X4Solid::convertTubs_cylinder
nnode* NSolid::createTubs( const char* name , float rmin, float rmax, float hz )
{
    assert( rmin == 0.f ); 
    ncylinder* n = make_cylinder( rmax, -hz, hz ); 
    n->label = strdup(name) ; 
    return (nnode*)n ; 
}

nnode* NSolid::createTorus( const char* name,  float rmin, float rmax, float rtor )
{
    assert( rmin == 0.f );
    ntorus* n = make_torus( rtor, rmax ); 
    n->label = strdup(name) ; 
    return n ; 
}

nnode* NSolid::createSubtractionSolid( const char* name, nnode* left , nnode* right, void* rot, glm::vec3 tlate )
{
    assert( rot == NULL ); 
    ndifference* n = ndifference::make_difference( left, right ); 
    n->label = strdup(name) ;  
    right->transform = nmat4triple::make_translate( tlate ); 
    left->parent = n ;  
    right->parent = n ;  
    return n ; 
} 

nnode* NSolid::createUnionSolid( const char* name, nnode* left , nnode* right, void* rot, glm::vec3 tlate )
{
    assert( rot == NULL ); 
    nunion* n = nunion::make_union( left, right ); 
    n->label = strdup(name) ;  
    right->transform = nmat4triple::make_translate( tlate ); 
    left->parent = n ;  
    right->parent = n ;  

    return n ; 
} 
nnode* NSolid::createIntersectionSolid( const char* name, nnode* left , nnode* right, void* rot, glm::vec3 tlate )
{
    assert( rot == NULL ); 
    nintersection* n = nintersection::make_intersection( left, right ); 
    n->label = strdup(name) ;  
    right->transform = nmat4triple::make_translate( tlate ); 
    left->parent = n ;  
    right->parent = n ;  

    return n ; 
} 


nnode* NSolid::create(int lv)
{
    nnode* n = NULL ; 
    switch(lv)   
    {
        case 18:     n = create_x018()               ; break ; 
        case 18001:  n = create_x018_f()             ; break ; 
        case 18002:  n = create_x018_c()             ; break ; 
        case 19:  n = create_x019()                 ; break ; 
        case 20:  n = create_x020()                 ; break ; 
        case 21:  n = create_x021()                 ; break ; 
        default:  assert(0)                         ; break ; 
    }
    return n ; 
}




nnode* NSolid::create_x021()
{
    // geocache-tcd
    // /home/blyth/local/opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/528f4cefdac670fffe846377973af10a/1/g4codegen/tests/x021.cc

    nnode* c = createEllipsoid("PMT_20inch_pmt_solid_1_Ellipsoid0x4c3bc00", 254.001000, 254.001000, 184.001000, -184.001000, 184.001000) ; // 2
    nnode* f = createTubs("PMT_20inch_pmt_solid_2_Tube0x4c3bc90", 0.000000, 77.976532, 21.496235 ) ; // 3
    nnode* h = createTorus("PMT_20inch_pmt_solid_2_Torus0x4c84bd0", 0.000000, 47.009000, 97.000000 ) ; // 3
    
    glm::vec3 A(0.000000,0.000000,-21.486235);
    nnode* e = createSubtractionSolid("PMT_20inch_pmt_solid_part20x4c84c70", f, h, NULL, A) ; // 2
    
    glm::vec3 B(0.000000,0.000000,-197.513765);
    nnode* b = createUnionSolid("PMT_20inch_pmt_solid_1_20x4c84f90", c, e, NULL, B) ; // 1
    nnode* j = createTubs("PMT_20inch_pmt_solid_3_EndTube0x4c84e60", 0.000000, 50.011000, 60.010500 ) ; // 1
    
    glm::vec3 C(0.000000,0.000000,-279.000500);
    nnode* a = createUnionSolid( x021_label, b, j, NULL, C) ; // 0

    return a ; 
}
const char* NSolid::x021_label = "PMT_20inch_pmt_solid0x4c81b40" ; 

nnode* NSolid::create_x020()
{
    // geocache-tcd
    // /home/blyth/local/opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/528f4cefdac670fffe846377973af10a/1/g4codegen/tests/x020.cc

    nnode* c = createEllipsoid("PMT_20inch_body_solid_1_Ellipsoid0x4c81db0", 254.000000, 254.000000, 184.000000, -184.000000, 184.000000) ; 
    nnode* f = createTubs("PMT_20inch_body_solid_2_Tube0x4c81e90", 0.000000, 77.976127, 21.496692 ) ; 
    nnode* h = createTorus("PMT_20inch_body_solid_2_Torus0x4c81fc0", 0.000000, 47.010000, 97.000000 ) ; 
    
    glm::vec3 A(0.000000,0.000000,-21.486692);
    nnode* e = createSubtractionSolid("PMT_20inch_body_solid_part20x4c820b0", f, h, NULL, A) ; 
    
    glm::vec3 B(0.000000,0.000000,-197.513308);
    nnode* b = createUnionSolid("PMT_20inch_body_solid_1_20x4c90cd0", c, e, NULL, B) ; 
    nnode* j = createTubs("PMT_20inch_body_solid_3_EndTube0x4c90ba0", 0.000000, 50.010000, 60.010000 ) ; 
    
    glm::vec3 C(0.000000,0.000000,-279.000000);
    nnode* a = createUnionSolid( x020_label, b, j, NULL, C) ; 
    return a ; 
}
const char* NSolid::x020_label = "PMT_20inch_body_solid0x4c90e50" ; 


nnode* NSolid::create_x019()
{
    // geocache-tcd
    // /home/blyth/local/opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/528f4cefdac670fffe846377973af10a/1/g4codegen/tests/x019.cc

    nnode* d = createEllipsoid("PMT_20inch_inner_solid_1_Ellipsoid0x4c91130", 249.000000, 249.000000, 179.000000, -179.000000, 179.000000) ; // 3
    nnode* g = createTubs("PMT_20inch_inner_solid_2_Tube0x4c91210", 0.000000, 75.951247, 23.782510 ) ; // 4
    nnode* i = createTorus("PMT_20inch_inner_solid_2_Torus0x4c91340", 0.000000, 52.010000, 97.000000 ) ; // 4
    
    glm::vec3 A(0.000000,0.000000,-23.772510);
    nnode* f = createSubtractionSolid("PMT_20inch_inner_solid_part20x4cb2d80", g, i, NULL, A) ; // 3
    
    glm::vec3 B(0.000000,0.000000,-195.227490);
    nnode* c = createUnionSolid("PMT_20inch_inner_solid_1_20x4cb30f0", d, f, NULL, B) ; // 2
    nnode* k = createTubs("PMT_20inch_inner_solid_3_EndTube0x4cb2fc0", 0.000000, 45.010000, 57.510000 ) ; // 2
    
    glm::vec3 C(0.000000,0.000000,-276.500000);
    nnode* b = createUnionSolid("PMT_20inch_inner_solid0x4cb32e0", c, k, NULL, C) ; // 1
    nnode* m = createTubs("Inner_Separator0x4cb3530", 0.000000, 254.000000, 92.000000 ) ; // 1
    
    glm::vec3 D(0.000000,0.000000,92.000000);
    nnode* a = createSubtractionSolid( x019_label , b, m, NULL, D) ; // 0
    return a ; 
}
const char* NSolid::x019_label = "PMT_20inch_inner2_solid0x4cb3870" ; 

nnode* NSolid::create_x018()
{
    // geocache-tcd
    // /home/blyth/local/opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/528f4cefdac670fffe846377973af10a/1/g4codegen/tests/x018.cc

    nnode* d = createEllipsoid("PMT_20inch_inner_solid_1_Ellipsoid0x4c91130", 249.000000, 249.000000, 179.000000, -179.000000, 179.000000) ; // 3
    nnode* g = createTubs("PMT_20inch_inner_solid_2_Tube0x4c91210", 0.000000, 75.951247, 23.782510 ) ; // 4
    nnode* i = createTorus("PMT_20inch_inner_solid_2_Torus0x4c91340", 0.000000, 52.010000, 97.000000 ) ; // 4
    
    glm::vec3 A(0.000000,0.000000,-23.772510);
    nnode* f = createSubtractionSolid("PMT_20inch_inner_solid_part20x4cb2d80", g, i, NULL, A) ; // 3
    
    glm::vec3 B(0.000000,0.000000,-195.227490);
    nnode* c = createUnionSolid("PMT_20inch_inner_solid_1_20x4cb30f0", d, f, NULL, B) ; // 2
    nnode* k = createTubs("PMT_20inch_inner_solid_3_EndTube0x4cb2fc0", 0.000000, 45.010000, 57.510000 ) ; // 2
    
    glm::vec3 C(0.000000,0.000000,-276.500000);
    nnode* b = createUnionSolid("PMT_20inch_inner_solid0x4cb32e0", c, k, NULL, C) ; // 1
    nnode* m = createTubs("Inner_Separator0x4cb3530", 0.000000, 254.000000, 92.000000 ) ; // 1
    
    glm::vec3 D(0.000000,0.000000,92.000000);
    nnode* a = createIntersectionSolid( x018_label, b, m, NULL, D) ; // 0
 
    return a ; 
}
const char* NSolid::x018_label = "PMT_20inch_inner1_solid0x4cb3610" ; 




nnode* NSolid::create_x018_f()
{
    nnode* g = createTubs("PMT_20inch_inner_solid_2_Tube0x4c91210", 0.000000, 75.951247, 23.782510 ) ; // 4
    nnode* i = createTorus("PMT_20inch_inner_solid_2_Torus0x4c91340", 0.000000, 52.010000, 97.000000 ) ; // 4
    
    glm::vec3 A(0.000000,0.000000,-23.772510);
    nnode* f = createSubtractionSolid("PMT_20inch_inner_solid_part20x4cb2d80", g, i, NULL, A) ; // 3

    return f ;  
}

nnode* NSolid::create_x018_c()
{
    nnode* d = createEllipsoid("PMT_20inch_inner_solid_1_Ellipsoid0x4c91130", 249.000000, 249.000000, 179.000000, -179.000000, 179.000000) ; // 3
    nnode* f = create_x018_f() ;
    glm::vec3 B(0.000000,0.000000,-195.227490);
    nnode* c = createUnionSolid("PMT_20inch_inner_solid_1_20x4cb30f0", d, f, NULL, B) ; // 2
    return c ;  
}






/*

ana/x018.py

class x018(X):
    """
    G4VSolid* make_solid()
    { 
        G4ThreeVector A(0.000000,0.000000,-23.772510);
        G4ThreeVector B(0.000000,0.000000,-195.227490);
        G4ThreeVector C(0.000000,0.000000,-276.500000);
        G4ThreeVector D(0.000000,0.000000,92.000000);


        G4VSolid* d = new G4Ellipsoid("PMT_20inch_inner_solid_1_Ellipsoid0x4c91130", 249.000000, 249.000000, 179.000000, -179.000000, 179.000000) ; // 3
        G4VSolid* g = new G4Tubs("PMT_20inch_inner_solid_2_Tube0x4c91210", 0.000000, 75.951247, 23.782510, 0.000000, CLHEP::twopi) ; // 4
        G4VSolid* i = new G4Torus("PMT_20inch_inner_solid_2_Torus0x4c91340", 0.000000, 52.010000, 97.000000, -0.000175, CLHEP::twopi) ; // 4
        
        G4VSolid* f = new G4SubtractionSolid("PMT_20inch_inner_solid_part20x4cb2d80", g, i, NULL, A) ; // 3
        G4VSolid* c = new G4UnionSolid("PMT_20inch_inner_solid_1_20x4cb30f0", d, f, NULL, B) ; // 2

        G4VSolid* k = new G4Tubs("PMT_20inch_inner_solid_3_EndTube0x4cb2fc0", 0.000000, 45.010000, 57.510000, 0.000000, CLHEP::twopi) ; // 2
        
        G4VSolid* b = new G4UnionSolid("PMT_20inch_inner_solid0x4cb32e0", c, k, NULL, C) ; // 1
        G4VSolid* m = new G4Tubs("Inner_Separator0x4cb3530", 0.000000, 254.000000, 92.000000, 0.000000, CLHEP::twopi) ; // 1
        
        G4VSolid* a = new G4IntersectionSolid("PMT_20inch_inner1_solid0x4cb3610", b, m, NULL, D) ; // 0
        return a ; 
    } 


                           a                                            Intersection
                                                                       /            \
                      b             m(D)                          Union             m:Tubs
                                                                  /    \
                c          k(C)                             Union       Tubs
                                                           /     \
            d     f(B)                            Ellipsoid   Subtraction    
                                                              /          \
                g(B)  i(B+A)                                Tubs         Torus



    * intersection with m:Tubs blows away the rest of the tree leaving 
      just the top half of the ellipsoid 

    * x019 is almost identical to this just with "Intersection" -> "Subtraction"
      so for that root.left becomes root


    """
    def __init__(self, mode=0):

        d = Ellipsoid("d", [249.000, 179.000 ] )
        g = Tubs("g", [75.951247,23.782510] )
        i = Torus("i", [ 52.010000, 97.000000] )

        A = np.array( [0, -23.772510] )
        f = SubtractionSolid("f", [g,i,A] )
        B = np.array( [0, -195.227490] )
        c = UnionSolid("c", [d,f,B] )

        k = Tubs("k", [45.010000, 57.510000] )

        C = np.array( [0, -276.500000] )
        b = UnionSolid("b", [c,k,C] )
        m = Tubs("m", [254.000000, 92.000000] )

        D = np.array( [0, 92.000000] )


*/

