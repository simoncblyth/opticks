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

    Chk(); 

    const char* CHECK ; 
    NP*         a  ; 

    void save(); 
    void SmearNormal_SigmaAlpha(); 
    void SmearNormal_Polish(); 
    void run(); 

    static NP* SmearNormal(int chk, double value); 

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

Chk::Chk()
    :
    CHECK(ssys::getenvvar("CHECK","SmearNormal_SigmaAlpha")),
    a(nullptr)
{
}


void Chk::save()
{
    std::string arr = CHECK ; 
    arr += ".npy" ; 
    a->save("$FOLD", arr.c_str() ); 
}

void Chk::SmearNormal_SigmaAlpha()
{
    int chk = SIGMA_ALPHA ; 
    double sigma_alpha = 0.1 ; 
    a = SmearNormal( chk, sigma_alpha ); 
    a->set_meta<double>("value", sigma_alpha ); 
    a->set_meta<std::string>("valuename", "sigma_alpha" ); 
    const char* name = Name(chk); 
    a->names.push_back(name); 

    save(); 
}
void Chk::SmearNormal_Polish()
{
    int chk = POLISH ; 
    double polish = 0.8 ; 
    a = SmearNormal(chk, polish ); 
    a->set_meta<double>("value", polish ); 
    a->set_meta<std::string>("valuename", "polish" ); 

    const char* name = Name(chk); 
    a->names.push_back(name); 

    save(); 
}


NP* Chk::SmearNormal(int chk, double value)
{
    S4Random r ; 

    G4ThreeVector Momentum(0., 0., -1. ); 
    G4ThreeVector Normal(  0., 0.,  1. );

    const int N = ssys::getenvint("NUM", 1000) ; 
    int ni = N ; 
    int nj = 3 ; 

    NP* a = NP::Make<double>( ni, nj ); 
    double* aa = a->values<double>(); 

    for(int i=0 ; i < ni ; i++) 
    {
        r.setSequenceIndex(i);   // use precooked random streams : so can align 

        G4ThreeVector FacetNormal( 0., 0., 0.) ; 
        switch(chk)
        {
            case SIGMA_ALPHA: FacetNormal = S4OpBoundaryProcess::SmearNormal_SigmaAlpha( Momentum, Normal, value );  break ; 
            case POLISH:      FacetNormal = S4OpBoundaryProcess::SmearNormal_Polish(     Momentum, Normal, value );  break ; 
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
    return a ; 
}

void Chk::run()
{
    if(     strcmp(CHECK, "SmearNormal_SigmaAlpha")==0) SmearNormal_SigmaAlpha() ; 
    else if(strcmp(CHECK, "SmearNormal_Polish")==0) SmearNormal_Polish() ;
    else
    {
        std::cerr 
            << "Chk::run" 
            << " CHECK " << ( CHECK ? CHECK : "-" ) << " IS UNHANDLED " 
            << std::endl 
            ;
    }
}


int main()
{
    Chk chk ; 
    chk.run(); 

    return 0 ; 
}
