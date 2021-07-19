#include <iostream>
#include <iomanip>

#include <CLHEP/Vector/ThreeVector.h>
typedef CLHEP::Hep3Vector Z4ThreeVector;

#include <CLHEP/Units/SystemOfUnits.h>
using CLHEP::radian;
using CLHEP::degree;

#include <CLHEP/Random/Randomize.h>
#define Z4UniformRand() CLHEP::HepRandom::getTheEngine()->flat()


struct MyEngine : public CLHEP::MixMaxRng 
{
    double flat(){ return .42 ; }
};

void test_setTheEngine(bool flip)
{
    if(flip) CLHEP::HepRandom::setTheEngine(new MyEngine()); 
    for( unsigned i=0 ; i < 10 ; i++)
    {
        double u = Z4UniformRand(); 
        std::cout << u << std::endl ; 
    }
}

void test_setSeed(long seed)
{
    CLHEP::HepRandomEngine* engine = CLHEP::HepRandom::getTheEngine(); 
    int dummy = 0 ; 
    engine->setSeed(seed, dummy); 
    std::cout << "test_setSeed " << seed << std::endl ; 
    for( unsigned i=0 ; i < 100 ; i++)
    {
        double u = Z4UniformRand(); 
        if( i % 10 == 0 ) std::cout << std::endl ; 
        std::cout << " " << std::fixed << std::setw(10) << std::setprecision(5) << u ; 
    }
    std::cout << std::endl ; 
}


void test_setTheta()
{
    Z4ThreeVector v(0.,0.,100.); 

    for(double t=0. ; t < 90.*degree ; t+=1.*degree )
    {
         double t0=t ; 
         v.setTheta(t0) ; 
         double t1=v.theta(); 
         double dt=t1 - t0 ; 
      
         std::cout
             << " t0 " << std::setw(10) << std::fixed << std::setprecision(8) << t0
             << " t1 " << std::setw(10) << std::fixed << std::setprecision(8) << t1 
             << " dt " << std::setw(10) << std::fixed << std::setprecision(8) << dt 
             << " dt*1e9 " << std::setw(10) << std::fixed << std::setprecision(8) << dt*1e9
             << std::endl 
             ;
    }      
}

int main(int argc, char** )
{
    //test_setTheEngine(argc > 1 ); 
    //test_setTheta(); 

    test_setSeed(1); 
    test_setSeed(2); 
    test_setSeed(1); 
    test_setSeed(-1); 


    return 0 ; 
}

