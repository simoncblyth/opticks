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
#include "spath.h"
#include "NP.hh"
#include "S4OpBoundaryProcess.h"
#include "S4Random.h"

struct Chk
{
    Chk(); 
    std::string desc() const ; 

    const char* FOLD ; 
    const char* CHECK ; 
    NP*         a  ; 

    void run(); 
    void SmearNormal(int chk, double value); 

}; 


Chk::Chk()
    :
    FOLD(ssys::getenvvar("FOLD")),
    CHECK(FOLD ? spath::Basename(FOLD) : nullptr),
    a(nullptr)
{
}

std::string Chk::desc() const
{
    std::stringstream ss ; 
    ss << "Chk::desc" 
       << std::endl 
       << " FOLD " << ( FOLD ? FOLD : "-" ) 
       << std::endl 
       << " CHECK " << ( CHECK ? CHECK : "-" ) 
       << std::endl 
       ;
    std::string str = ss.str(); 
    return str ; 
}


void Chk::SmearNormal(int chk, double value)
{
    S4Random r ; 

    G4ThreeVector Momentum(0., 0., -1. ); 
    G4ThreeVector Normal(  0., 0.,  1. );

    const int N = ssys::getenvint("NUM", 1000) ; 
    int ni = N ; 
    int nj = 4 ; 

    NP* a = NP::Make<double>( ni, nj ); 
    double* aa = a->values<double>(); 

    a->set_meta<double>("value", value ); 
    a->set_meta<std::string>("valuename", chk == 0 ? "sigma_alpha" : "polish" ); 
    a->set_meta<std::string>("source", "S4OpBoundaryProcessTest.sh") ; 

    for(int i=0 ; i < ni ; i++) 
    {
        r.setSequenceIndex(i);   // use precooked random streams : so can align 

        G4ThreeVector FacetNormal( 0., 0., 0.) ; 
        switch(chk)
        {
            case 0: FacetNormal = S4OpBoundaryProcess::SmearNormal_SigmaAlpha( Momentum, Normal, value );  break ; 
            case 1: FacetNormal = S4OpBoundaryProcess::SmearNormal_Polish(     Momentum, Normal, value );  break ; 
        }

        aa[i*4+0] = FacetNormal.x() ; 
        aa[i*4+1] = FacetNormal.y() ; 
        aa[i*4+2] = FacetNormal.z() ; 
        aa[i*4+3] = 0.  ; 

        if(i < 4)  std::cout 
            << " i " << i 
            << " FacetNormal = np.array([" 
            << std::fixed << std::setw(10) << std::setprecision(5) << FacetNormal.x() << ","
            << std::fixed << std::setw(10) << std::setprecision(5) << FacetNormal.y() << ","
            << std::fixed << std::setw(10) << std::setprecision(5) << FacetNormal.z() << "])"
            << std::endl
            ; 
    }
    a->save("$FOLD/q.npy" ); 
}

void Chk::run()
{
    std::cout << desc() << std::endl ; 

    if(     strcmp(CHECK, "smear_normal_sigma_alpha")==0) SmearNormal(0, 0.1) ; 
    else if(strcmp(CHECK, "smear_normal_polish")==0)      SmearNormal(1, 0.8) ;
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
