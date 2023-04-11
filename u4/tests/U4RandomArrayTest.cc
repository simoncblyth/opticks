#include <iostream>
#include <iomanip>

#include "S4RandomArray.h"


int main(int argc, char** argv)
{
    S4RandomArray arr ; 

    for(int i=0 ; i < 10 ; i++)
    {
        double u = G4UniformRand() ; 
        std::cout << " u " << std::setw(10) << std::setprecision(5) << u << std::endl ;
    }

    double chk[10] ; 
    CLHEP::HepRandom::getTheEngine()->flatArray(10, &chk[0] ); 
    for(int i=0 ; i < 10 ; i++ ) std::cout << " flatArray chk " << std::setw(10) << std::setprecision(5) << chk[i] << std::endl ; 


    NP* a = arr.serialize(); 
    const char* path = "$TMP/S4RandomArrayTest/a.npy" ; 
    std::cout << " saving " << a->sstr() << " to " << path << std::endl ; 
    a->save(path); 

    return 0 ; 
} 
/**

  0.13049
   0.61775
   0.99595
    0.4959
   0.11292
   0.28987
   0.47304
   0.83762
   0.35936
   0.92694

**/
