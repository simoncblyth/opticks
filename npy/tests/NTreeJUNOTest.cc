// TEST=NTreeJUNOTest om-t

#include "NTreeJUNO.hpp"

#include "NGLMExt.hpp"
#include "NZSphere.hpp"
#include "NSphere.hpp"
#include "NCylinder.hpp"
#include "NTorus.hpp"
#include "NCone.hpp"

#include "NTreeAnalyse.hpp"

#include "OPTICKS_LOG.hh"


// X4Solid::convertEllipsoid
nnode* create_ellipsoid( const char* name, float ax, float by, float cz, float zcut1, float zcut2 )
{
    float z1 = zcut1 != 0.f && zcut1 > -cz ? zcut1 : -cz ;
    float z2 = zcut2 != 0.f && zcut2 <  cz ? zcut2 :  cz ;
    assert( z2 > z1 ) ;

    glm::vec3 scale( ax/cz, by/cz, 1.f) ;

    bool zslice = z1 > -cz || z2 < cz ; 

    nnode* n = zslice ? 
                          (nnode*)make_zsphere( 0.f, 0.f, 0.f, cz, z1, z2 ) 
                       :
                          (nnode*)make_sphere( 0.f, 0.f, 0.f, cz ) 
                       ;

    n->label = strdup(name) ; 
    n->transform = nmat4triple::make_scale( scale );

    return n ; 
}

// X4Solid::convertTubs_cylinder
nnode* create_tubs( const char* name , float rmin, float rmax, float hz )
{
    assert( rmin == 0.f ); 
    ncylinder* n = make_cylinder( rmax, -hz, hz ); 
    n->label = strdup(name) ; 
    return (nnode*)n ; 
}

nnode* create_torus( const char* name,  float rminor, float rmajor )
{
    ntorus* n = make_torus( rmajor, rminor ); 
    n->label = strdup(name) ; 
    return n ; 
}

nnode* create_difference( const char* name, nnode* left , nnode* right, glm::vec3 tlate )
{
    ndifference* n = ndifference::make_difference( left, right ); 
    n->label = strdup(name) ;  
    right->transform = nmat4triple::make_translate( tlate ); 
    left->parent = n ;  
    right->parent = n ;  
    return n ; 
} 

nnode* create_union( const char* name, nnode* left , nnode* right, glm::vec3 tlate )
{
    nunion* n = nunion::make_union( left, right ); 
    n->label = strdup(name) ;  
    right->transform = nmat4triple::make_translate( tlate ); 
    left->parent = n ;  
    right->parent = n ;  

    return n ; 
} 
nnode* create_intersection( const char* name, nnode* left , nnode* right, glm::vec3 tlate )
{
    nintersection* n = nintersection::make_intersection( left, right ); 
    n->label = strdup(name) ;  
    right->transform = nmat4triple::make_translate( tlate ); 
    left->parent = n ;  
    right->parent = n ;  

    return n ; 
} 


nnode* make_x018()
{
    nnode* d = create_ellipsoid("d", 249.000000, 249.000000, 179.000000, -179.000000, 179.000000 ) ;        
    nnode* g = create_tubs("g", 0.000000, 75.951247, 23.782510 );
    nnode* i = create_torus("i", 52.010000, 97.000000 ); 

    glm::vec3 A(0.000000,0.000000,-23.772510);
    nnode* f = create_difference("f",  g, i, A ); 

    glm::vec3 B(0.000000,0.000000,-195.227490);
    nnode* c = create_union("c", d, f, B ) ;  

    nnode* k = create_tubs("k", 0.000000, 45.010000, 57.510000 );

    glm::vec3 C(0.000000,0.000000,-276.500000);
    nnode* b = create_union("b", c, k, C );      

    nnode* m = create_tubs("m", 0.000000, 254.000000, 92.000000 );
        
    glm::vec3 D(0.000000,0.000000,92.000000);
    nnode* a = create_intersection("a", b, m, D ); 


    //nnode* a2 = a->make_deepcopy(); 
    //LOG(info) << NTreeAnalyse<nnode>::Desc(a2);

    return a ; 
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    nnode* a0 = make_x018(); 
    nnode* a = make_x018(); 
    a->label = "x019" ;    // mis-labelled as x018 too simple

    LOG(info) << NTreeAnalyse<nnode>::Desc(a);

    NTreeJUNO tj(a) ; 
    tj.rationalize(); 

    LOG(info) << NTreeAnalyse<nnode>::Desc(tj.root);


    return 0 ; 
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
