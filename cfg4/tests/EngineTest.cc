
#include "OPTICKS_LOG.hh"
#include "Randomize.hh"


struct MyEngine : public CLHEP::MixMaxRng 
{
    double flat(){ return .42 ; }
};


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    CLHEP::HepRandom::setTheEngine(new MyEngine()); 

    LOG(info) 
        << " using " << CLHEP::HepRandom::getTheEngine()->name()
        ;

    for( unsigned i=0 ; i < 1000000 ; i++)
    {
        double u = G4UniformRand(); 

        if( i % 1000 == 0)
        std::cout 
            << std::setw(8) << i 
            << " : " 
            << std::setw(10) << std::fixed << u
            << std::endl
            ;
    }

    return 0 ; 
}

