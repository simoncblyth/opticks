#include <iostream>
#include "PLOG.hh"

#include "Randomize.hh"
#include "CLHEP/Random/NonRandomEngine.h"


int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    LOG(info) << argv[0] ; 


    unsigned N = 10 ;    // needs to provide all that are consumed
    std::vector<double> seq ; 
    for(unsigned i=0 ; i < N ; i++ ) seq.push_back( double(i)/double(N) );  


    long custom_seed = 9876 ; 
    //CLHEP::HepJamesRandom* custom_engine = new CLHEP::HepJamesRandom();
    //CLHEP::MTwistEngine*   custom_engine = new CLHEP::MTwistEngine();

    CLHEP::NonRandomEngine*   custom_engine = new CLHEP::NonRandomEngine();
    custom_engine->setRandomSequence( seq.data(), seq.size() ) ; 

    CLHEP::HepRandom::setTheEngine( custom_engine );  
    CLHEP::HepRandom::setTheSeed( custom_seed );    // does nothing for NonRandom

    CLHEP::HepRandomEngine* engine = CLHEP::HepRandom::getTheEngine() ;


    long seed = engine->getSeed() ; 
    LOG(info) << " seed " << seed 
              << " name " << engine->name() 
            ; 

    for(int i=0 ; i < 10 ; i++)
    {
        double u = engine->flat() ;   // equivalent to the standardly used: G4UniformRand() 
        std::cout << u << std::endl ; 
    }
    return 0 ; 
}

/*

simon:Random blyth$ G4UniformRandTest
2017-12-02 12:14:34.949 INFO  [1051012] [main@11] G4UniformRandTest
2017-12-02 12:14:34.950 INFO  [1051012] [main@21]  seed 9876 name HepJamesRandom
0.286072
0.366332
0.942989
0.278981
0.18341
0.186724
0.265324
0.452413
0.552432
0.223035

*/


/*
   g4-;g4-cls Randomize
   g4-;g4-cls Random
   g4-;g4-cls RandomEngine
   g4-;g4-cls JamesRandom


simon:Random blyth$ grep public\ HepRandomEngine *.*
DualRand.h:class DualRand: public HepRandomEngine {
JamesRandom.h:class HepJamesRandom: public HepRandomEngine {
MTwistEngine.h:class MTwistEngine : public HepRandomEngine {
MixMaxRng.h:class MixMaxRng: public HepRandomEngine {
NonRandomEngine.h:class NonRandomEngine : public HepRandomEngine {
RanecuEngine.h:class RanecuEngine : public HepRandomEngine {
Ranlux64Engine.h:class Ranlux64Engine : public HepRandomEngine {
RanluxEngine.h:class RanluxEngine : public HepRandomEngine {
RanshiEngine.h:class RanshiEngine: public HepRandomEngine {
simon:Random blyth$ 


https://arxiv.org/pdf/1307.5869.pdf

http://docs.nvidia.com/cuda/curand/host-api-overview.html#host-api-overview


*/


