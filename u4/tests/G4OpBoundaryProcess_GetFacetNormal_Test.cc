// ./G4OpBoundaryProcess_GetFacetNormal_Test.sh 

#include <iostream>
#include "G4ThreeVector.hh"
#include "Randomize.hh"
#include "NPX.h"

/**
GetFacetNormal
----------------

Only Momentum_FacetNormal < 0. escapes the while loop
ie FacetNormal must end up against the Momentum.

Presumably this is assuming the initial Momentum is against the Normal. 
It would make more sense to constrain FacetNormal to 
staying with the same orientation as the initial one
not just assuming a particular orientation.  


Smearing with polish 0.999 results in something effectively indistingishable from specular, 
so it just consumes randoms and does little else. 

  : t.FacetNormal                                      :            (1000, 3) : 0:03:32.992163 

 min_stamp : 2023-03-03 17:52:04.431359 
 max_stamp : 2023-03-03 17:52:04.431359 
 dif_stamp : 0:00:00 
 age_stamp : 0:03:32.992163 
[[-0.000008198991 -0.000774490262  0.999999700049]
 [-0.000053926407  0.000675428761  0.999999770444]
 [-0.000123759488 -0.000047115051  0.999999991232]
 [-0.000732900199 -0.000029525774  0.999999730993]
 [-0.000347653389 -0.000362541626  0.99999987385 ]
 [-0.000390777914  0.000540491843  0.999999777581]
 [-0.000222860429 -0.000332362899  0.999999919934]
 [-0.000550520639 -0.000345155409  0.999999788897]
 [-0.000083752166 -0.000284001987  0.999999956164]
 [-0.000172539198 -0.000299258882  0.999999940337]
 [-0.000378540935  0.000299374342  0.999999883541]
 [-0.000554646197  0.000462350078  0.9999997393  ]
 [-0.000523868695 -0.000421828994  0.999999773811]
 [-0.00046533652   0.0006765364    0.99999966288 ]
 [-0.000070471159  0.000012100613  0.999999997444]
 [-0.000041047925  0.0000830588    0.999999995708]
 [-0.000027495856  0.000620312374  0.999999807228]
 [-0.000708336559 -0.000352951962  0.999999686842]
 [-0.00005016816   0.000631667703  0.99999979924 ]
 [-0.000074296972  0.000283778564  0.999999956975]

**/


G4ThreeVector GetFacetNormal(G4ThreeVector& smear, const G4ThreeVector& Momentum, const G4ThreeVector& Normal, G4double polish )
{
    G4ThreeVector FacetNormal;
    G4double Momentum_FacetNormal = 0. ; 

    if (polish < 1.0) 
    {
        do { 
            do 
            { 
                smear.setX(2.*G4UniformRand()-1.0);
                smear.setY(2.*G4UniformRand()-1.0);
                smear.setZ(2.*G4UniformRand()-1.0);
            } 
            while (smear.mag()>1.0);     // random vector within unit sphere (not normalized)

            smear = (1.-polish) * smear;   // scale it down (greatly for polish 0.999) 

            FacetNormal = Normal + smear;  // perturb the Normal by the smear 

            Momentum_FacetNormal = Momentum * FacetNormal ;  

        } 
        while (Momentum_FacetNormal >= 0.0);  

        FacetNormal = FacetNormal.unit();
    }    
    else 
    {
        FacetNormal = Normal;
    }    
    return FacetNormal ; 
}


struct FacetNormal_Smear
{
    G4ThreeVector FacetNormal ; 
    G4ThreeVector Smear ; 
};

int main()
{
    G4double      polish = 0.999 ; 

    G4ThreeVector Momentum(1,-1,0);  
    G4ThreeVector Normal(0,0,1) ;    
    G4ThreeVector Meta(polish, 0, 0); 

    Momentum = Momentum.unit(); 
    Normal = Normal.unit() ; 

    std::vector<G4ThreeVector> mm(3) ;
    mm[0] = Momentum ; 
    mm[1] = Normal ;    
    mm[2] = Meta ;    

    const int N=1000 ; 
    std::vector<FacetNormal_Smear> nn(N) ;   

    for(int i=0 ; i < int(nn.size()) ; i++ ) 
    {
        FacetNormal_Smear& fns = nn[i] ;  
        fns.FacetNormal = GetFacetNormal(fns.Smear, Momentum, Normal, polish ) ; 
    }

    NP* a = NPX::ArrayFromVec<double,FacetNormal_Smear>(nn,2,3) ; 
    a->save("$FOLD/FacetNormal.npy" );  

    NP* b = NPX::ArrayFromVec<double,G4ThreeVector>(mm) ; 
    b->save("$FOLD/Meta.npy" );  

    return 0 ; 
}


