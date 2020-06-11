#!/usr/bin/env python
#
# Copyright (c) 2019 Opticks Team. All Rights Reserved.
#
# This file is part of Opticks
# (see https://bitbucket.org/simoncblyth/opticks).
#
# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License.  
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
# See the License for the specific language governing permissions and 
# limitations under the License.
#

"""
Aiming for this to be generated, so keep it simple
"""
import numpy as np
from opticks.ana.shape import X, SEllipsoid, STubs, STorus, SCons, SSubtractionSolid, SUnionSolid, SIntersectionSolid


class x019(X):
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
        d = SEllipsoid( "d", [249.000, 179.000 ] )
        g = STubs(      "g", [75.951247,23.782510] )
        i = STorus(     "i", [ 52.010000, 97.000000] )

        A = np.array( [0, -23.772510] )
        f = SSubtractionSolid( "f", [g,i, A ] )

        B = np.array( [0, -195.227490] )
        c = SUnionSolid( "c",  [d, f, B] )

        k = STubs(      "k", [45.010000, 57.510000] )

        C = np.array( [0, -276.500000] )
        b = SUnionSolid( "b", [c, k, C] ) 
        m = STubs(      "m", [254.000000, 92.000000] )

        D = np.array( [0, 92.000000] )
        a = SSubtractionSolid( "a", [b, m, D ] )

        self.root = a        


if __name__ == '__main__':
    x = x019()
    print(repr(x))

