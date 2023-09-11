/**
S4OpBoundaryProcessTest.cc
============================

Note that the test momentum here is directly 
against the normal... changing that leads to 
very different distribs of normals. 

Added use of S4Random to use precooked randoms 
to allow alignment

**/

#include "ssys.h"
#include "NP.hh"
#include "S4OpBoundaryProcess.h"
#include "S4Random.h"


struct Chk
{
    enum { SIGMA_ALPHA, POLISH } ; 
    static constexpr const char* SIGMA_ALPHA_ = "SigmaAlpha.npy" ; 
    static constexpr const char* POLISH_ = "Polish.npy" ; 
    static const char* Name(int chk); 

    static void SmearNormal(); 

}; 

const char* Chk::Name(int chk)
{
    const char* n = nullptr ; 
    switch(chk)
    {
        case SIGMA_ALPHA: n = SIGMA_ALPHA_ ; break ; 
        case POLISH:      n = POLISH_ ; break ; 
    }
    return n ;
}


void Chk::SmearNormal()
{
    int chk = SIGMA_ALPHA ; 
    const char* chkname = Name(chk); 

    S4Random r ; 

    G4ThreeVector Momentum(0., 0., -1. ); 
    G4ThreeVector Normal(  0., 0.,  1. );

    G4double sigma_alpha = 0.1 ; 
    G4double polish = 0.8 ; 

    const int N = ssys::getenvint("NUM", 1000) ; 
    int ni = N ; 
    int nj = 3 ; 

    NP* a = NP::Make<double>( ni, nj ); 
    double* aa = a->values<double>(); 

    if(chk == SIGMA_ALPHA)
    {
        a->set_meta<double>("sigma_alpha", sigma_alpha ); 
        a->names.push_back(chkname); 
    }
    else if( chk == POLISH )
    {
        a->set_meta<double>("polish", polish ); 
        a->names.push_back(chkname); 
    }

    for(int i=0 ; i < ni ; i++) 
    {
        r.setSequenceIndex(i);   // use precooked random streams : so can align 

        G4ThreeVector FacetNormal( 0., 0., 0.) ; 
        switch(chk)
        {
            case SIGMA_ALPHA: FacetNormal = S4OpBoundaryProcess::SmearNormal_SigmaAlpha( Momentum, Normal, sigma_alpha );  break ; 
            case POLISH:      FacetNormal = S4OpBoundaryProcess::SmearNormal_Polish(     Momentum, Normal, polish      );  break ; 
        }

        aa[i*3+0] = FacetNormal.x() ; 
        aa[i*3+1] = FacetNormal.y() ; 
        aa[i*3+2] = FacetNormal.z() ; 

        if(i < 4)  std::cout 
            << " i " << i 
            << " FacetNormal = np.array([" 
            << std::fixed << std::setw(10) << std::setprecision(5) << FacetNormal.x() << ","
            << std::fixed << std::setw(10) << std::setprecision(5) << FacetNormal.y() << ","
            << std::fixed << std::setw(10) << std::setprecision(5) << FacetNormal.z() << "])"
            << std::endl
            ; 
    }
    a->save("$FOLD", chkname); 
}

int main()
{
    Chk::SmearNormal(); 
    return 0 ; 
}
