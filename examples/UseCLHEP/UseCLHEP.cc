#include <CLHEP/Random/Randomize.h>

#define Z4UniformRand() CLHEP::HepRandom::getTheEngine()->flat()


struct MyEngine : public CLHEP::MixMaxRng 
{
    double flat(){ return .42 ; }
};

int main(int argc, char** )
{
    if(argc > 1) CLHEP::HepRandom::setTheEngine(new MyEngine()); 

    for( unsigned i=0 ; i < 10 ; i++)
    {
        double u = Z4UniformRand(); 
        std::cout << u << std::endl ; 
    }

    return 0 ; 
}

