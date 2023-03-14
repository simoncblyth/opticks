// ./G4Orb_Test.sh 

#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <string>
#include <cassert>

#include "G4ThreeVector.hh"
#include "G4Orb.hh"
#include "sgeomdefs.h"
#include "ssolid.h"


std::string Label( const char* soname, char p_l, char d_l, const char* meth )
{
    std::stringstream ss ;    
    ss << soname << "." << meth << "(" << p_l << ", " << d_l << ") " ; 
    std::string str = ss.str(); 
    return str ; 
}

std::string Format(double v, int w=10)
{
    std::stringstream ss ;    
    if( v == kInfinity ) 
    { 
        ss << std::setw(w) << "kInfinity" ; 
    }
    else
    {
        ss << std::setw(w) << std::fixed << std::setprecision(4) << v ; 
    }
    std::string str = ss.str(); 
    return str ; 
}

std::string Format(const G4ThreeVector* v)
{
    std::stringstream ss ;    
    ss << *v ; 
    std::string str = ss.str(); 
    return str ; 
}

int main()
{
    G4Orb solid("Orb", 100.) ; 

    std::cout << solid << std::endl ;  

    G4String soname_ = solid.GetName() ; 
    const char* soname = soname_.c_str(); 

    G4ThreeVector A(0.,0., 200.); 
    G4ThreeVector B(0.,0., 150.); 
    G4ThreeVector C(0.,0., 100.); 
    G4ThreeVector D(0.,0.,  50.); 
    G4ThreeVector E(0.,0.,   0.); 
    G4ThreeVector F(0.,0., -50.); 
    G4ThreeVector G(0.,0.,-100.); 
    G4ThreeVector H(0.,0.,-150.); 
    G4ThreeVector I(0.,0.,-200.); 

    G4ThreeVector Z(0.,0.,1.); 
  
    std::vector<G4ThreeVector*> PP = {&A, &B, &C, &D, &E, &F, &G, &H, &I } ; 
    std::vector<char> PP_L         = {'A','B','C','D','E','F','G','H','I'}; 

    std::vector<G4ThreeVector*> DD = {&Z, &Z, &Z, &Z, &Z, &Z, &Z, &Z, &Z } ; 
    std::vector<char> DD_L         = {'Z','Z','Z','Z','Z','Z','Z','Z','Z' }; 

    for(int i=0 ; i < int(PP.size()) ; i++)
    {
        G4ThreeVector* p = PP[i] ;  
        G4ThreeVector* d = DD[i] ; 
        const char p_l = PP_L[i] ; 
        const char d_l = DD_L[i] ; 

        EInside in ; 
        G4double dis = ssolid::Distance_(&solid, *p, *d, in );   
        G4double d2o = solid.DistanceToOut( *p, *d ) ; 
        G4double d2i = solid.DistanceToIn(  *p, *d ) ; 

        std::cout 
             << p_l << " "
             << std::setw(10) << Format(p)
             << std::setw(30) << Label( soname, p_l, d_l, "DistanceToOut")  
             << " : " << Format(d2o)
             << " | "
             << std::setw(30) << Label( soname, p_l, d_l, "DistanceToIn")  
             << " : "
             << " : " << Format(d2i)
             << std::setw(30) << Label( soname, p_l, d_l, "Distance_")  
             << " : " << Format(dis)
             << " " << sgeomdefs::EInside_(in) 
             << std::endl
             ;
    }

    return 0 ; 
}


/**


                        200 A                                                +                                        +
                                                                             0                                       kInfinity
                                                                        
                        150 B                                            +                                        +
                                                                         0                                       kInfinity
                           
               +------- 100-C-----------+    +   +   +   +   +   +   +   +   +   +                            +
               |            |           |                            0                                       kInfinity
               |            |           |  
               |         50 D           |                        +                                        +  
               |            |           |                       50                                        0
               |            |           |  
               +----------0-E-----------+                    +                                        +
               |            |           |                   100                                       0 
               |            |           |                                                            
               |        -50 F           |                +                                        +
               |            |           |               150                                       0 
               |            |           |  
               +------ -100 G-----------+             +                           +   +   +   +   +   +   +   +   +   +   +
                                                     200                                      0

                       -150 H                     +                                       +
                                                 250                                     50
                      
                       -200 I                 +                                       +
                                             300                                     100

                                                       DistanceToOut( A->I, Z )                  DistanceToIn( A->I, Z)


DistanceToOut
   distance to "far" side (HMM how about booleans with holes?)

DistanceToIn
   distance to "near" side, 0 whilst inside, kInfinity at-or-beyond farside    
 
Distance
   distance to closest side  


-----------------------------------------------------------
    *** Dump for solid - Orb ***
    ===================================================
 Solid type: G4Orb
 Parameters: 
    outer radius: 100 mm 
-----------------------------------------------------------

A  (0,0,200)      Orb.DistanceToOut(A, Z)  :     0.0000 |        Orb.DistanceToIn(A, Z)  :  :  kInfinity          Orb.Distance_(A, Z)  :  kInfinity kOutside
B  (0,0,150)      Orb.DistanceToOut(B, Z)  :     0.0000 |        Orb.DistanceToIn(B, Z)  :  :  kInfinity          Orb.Distance_(B, Z)  :  kInfinity kOutside
C  (0,0,100)      Orb.DistanceToOut(C, Z)  :     0.0000 |        Orb.DistanceToIn(C, Z)  :  :  kInfinity          Orb.Distance_(C, Z)  :     0.0000 kSurface
D   (0,0,50)      Orb.DistanceToOut(D, Z)  :    50.0000 |        Orb.DistanceToIn(D, Z)  :  :     0.0000          Orb.Distance_(D, Z)  :    50.0000 kInside
E    (0,0,0)      Orb.DistanceToOut(E, Z)  :   100.0000 |        Orb.DistanceToIn(E, Z)  :  :     0.0000          Orb.Distance_(E, Z)  :   100.0000 kInside
F  (0,0,-50)      Orb.DistanceToOut(F, Z)  :   150.0000 |        Orb.DistanceToIn(F, Z)  :  :     0.0000          Orb.Distance_(F, Z)  :   150.0000 kInside
G (0,0,-100)      Orb.DistanceToOut(G, Z)  :   200.0000 |        Orb.DistanceToIn(G, Z)  :  :     0.0000          Orb.Distance_(G, Z)  :   200.0000 kSurface
H (0,0,-150)      Orb.DistanceToOut(H, Z)  :   250.0000 |        Orb.DistanceToIn(H, Z)  :  :    50.0000          Orb.Distance_(H, Z)  :    50.0000 kOutside
I (0,0,-200)      Orb.DistanceToOut(I, Z)  :   300.0000 |        Orb.DistanceToIn(I, Z)  :  :   100.0000          Orb.Distance_(I, Z)  :   100.0000 kOutside

**/


