/**
S4OpBoundaryProcessTest.cc
============================

Note that the test momentum here is directly 
against the normal... changing that leads to 
very different distribs of normals. 

**/

#include "NP.hh"
#include "S4OpBoundaryProcess.h"

int main()
{
    G4ThreeVector Momentum(0., 0., -1. ); 
    G4ThreeVector Normal(  0., 0.,  1. );

    G4double sigma_alpha = 0.1 ; 
    G4double polish = 0.8 ; 

    const int N = 1000 ; 

    int ni = N ; 
    int nj = 2 ; 
    int nk = 3 ; 

    NP* a = NP::Make<double>( ni, nj, nk ); 
    double* aa = a->values<double>(); 

    a->set_meta<double>("polish", polish ); 
    a->set_meta<double>("sigma_alpha", sigma_alpha ); 
    a->names.push_back("SmearNormal_SigmaAlpha"); 
    a->names.push_back("SmearNormal_Polish"); 


    for(int i=0 ; i < ni ; i++) for(int j=0 ; j < nj ; j++)
    {
        G4ThreeVector FacetNormal( 0., 0., 0.) ; 
        switch(j)
        {
            case 0: FacetNormal = S4OpBoundaryProcess::SmearNormal_SigmaAlpha( Momentum, Normal, sigma_alpha );  break ; 
            case 1: FacetNormal = S4OpBoundaryProcess::SmearNormal_Polish(     Momentum, Normal, polish      );  break ; 
        }

        int idx = i*nj*nk+j*nk ; 
        aa[idx+0] = FacetNormal.x() ; 
        aa[idx+1] = FacetNormal.y() ; 
        aa[idx+2] = FacetNormal.z() ; 

        if(i < 4)  std::cout 
            << " i " << i 
            << " j " << j 
            << "FacetNormal = np.array([" 
            << std::fixed << std::setw(10) << std::setprecision(5) << FacetNormal.x() << ","
            << std::fixed << std::setw(10) << std::setprecision(5) << FacetNormal.y() << ","
            << std::fixed << std::setw(10) << std::setprecision(5) << FacetNormal.z() << "])"
            << std::endl
            ; 
    }
    a->save("$FOLD/SmearNormal.npy"); 

    return 0 ; 
}
