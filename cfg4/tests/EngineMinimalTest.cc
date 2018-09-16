#include "Randomize.hh"

struct MyEngine : public CLHEP::MixMaxRng 
{
    double flat(){ return .42 ; }
};

int main(int argc, char** argv)
{
    if(argc > 1) CLHEP::HepRandom::setTheEngine(new MyEngine()); 

    for( unsigned i=0 ; i < 10 ; i++)
    {
        double u = G4UniformRand(); 
        std::cout << u << std::endl ; 
    }

    return 0 ; 
}

