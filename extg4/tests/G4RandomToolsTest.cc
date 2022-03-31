#include <cstdlib>
#include <iostream>
#include "G4RandomTools.hh"
#include "NP.hh"

int main()
{
    int num = U::GetEnvInt("NUM", 100000 ); 
    G4ThreeVector normal(0., 0., 1. ); 
    NP* a = NP::Make<float>(num, 3); 
    float* aa = a->values<float>(); 
    for(int i=0 ; i < num ; i++)
    {
        G4ThreeVector dir = G4LambertianRand(normal) ; 
        aa[3*i+0] = dir.x(); 
        aa[3*i+1] = dir.y();  
        aa[3*i+2] = dir.z(); 
    }

    const char* path = getenv("NPY_PATH"); 
    std::cout << " save to NPY_PATH " << path << " a.desc " << a->desc() << std::endl ; 
    a->save(path);
    return 0 ; 
}


