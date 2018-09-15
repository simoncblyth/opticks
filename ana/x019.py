
class X019(object):
    """
    G4VSolid* make_solid()
    { 
        G4VSolid* d = new G4Ellipsoid("PMT_20inch_inner_solid_1_Ellipsoid0x4c91130", 249.000000, 249.000000, 179.000000, -179.000000, 179.000000) ; // 3
        G4VSolid* g = new G4Tubs("PMT_20inch_inner_solid_2_Tube0x4c91210", 0.000000, 75.951247, 23.782510, 0.000000, CLHEP::twopi) ; // 4
        G4VSolid* i = new G4Torus("PMT_20inch_inner_solid_2_Torus0x4c91340", 0.000000, 52.010000, 97.000000, -0.000175, CLHEP::twopi) ; // 4
        
        G4ThreeVector A(0.000000,0.000000,-23.772510);
        G4VSolid* f = new G4SubtractionSolid("PMT_20inch_inner_solid_part20x4cb2d80", g, i, NULL, A) ; // 3
        
        G4ThreeVector B(0.000000,0.000000,-195.227490);
        G4VSolid* c = new G4UnionSolid("PMT_20inch_inner_solid_1_20x4cb30f0", d, f, NULL, B) ; // 2
        G4VSolid* k = new G4Tubs("PMT_20inch_inner_solid_3_EndTube0x4cb2fc0", 0.000000, 45.010000, 57.510000, 0.000000, CLHEP::twopi) ; // 2
        
        G4ThreeVector C(0.000000,0.000000,-276.500000);
        G4VSolid* b = new G4UnionSolid("PMT_20inch_inner_solid0x4cb32e0", c, k, NULL, C) ; // 1
        G4VSolid* m = new G4Tubs("Inner_Separator0x4cb3530", 0.000000, 254.000000, 92.000000, 0.000000, CLHEP::twopi) ; // 1
        
        G4ThreeVector D(0.000000,0.000000,92.000000);
        G4VSolid* a = new G4SubtractionSolid("PMT_20inch_inner2_solid0x4cb3870", b, m, NULL, D) ; // 0
        return a ; 
    } 




                           a

                      b             m(D)
                                   
                c          k(C)
              bulb+neck   endtube    
 
            d     f(B) 
         bulb     neck

                g(B)  i(B+A)
               tubs   torus


    """
    def __init__(self):
         
        A = np.array( [0, -23.772510], dtype=np.float32 )
        B = np.array( [0, -195.227490], dtype=np.float32 )
        C = np.array( [0, -276.500000], dtype=np.float32 )
        D = np.array( [0, 92.000000], dtype=np.float32 )

        d = Ellipsoid( [249.000, 179.000 ], fill=False )
        g = Tubs(    [75.951247,23.782510], fill=False )
        i = Torus( [ 52.010000, 97.000000], fill=False )
        k = Tubs( [45.010000, 57.510000], fill=True )
        m = Tubs( [254.000000, 92.000000], fill=False )

        i.xy += A+B 
        g.xy += B 
        k.xy += C 
        m.xy += D 

        self.sz = 400
        self.prims = [d,g,i,k,m]


