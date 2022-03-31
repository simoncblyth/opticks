#include <cstdlib>
#include <iostream>

#include "G4RandomDirection.hh"
#include "NP.hh"

int main()
{
    int num = U::GetEnvInt("NUM", 100) ; 
    NP* a = NP::Make<float>( num, 3 ); 
    std::cout << " a.desc " << a->desc() << std::endl ; 

    float* aa = a->values<float>(); 
    for(int i=0 ; i < num ; i++)
    {
        G4ThreeVector dir = G4RandomDirection() ; 
        aa[3*i+0] = dir.x();      
        aa[3*i+1] = dir.y();      
        aa[3*i+2] = dir.z();      
    }
    const char* path = getenv("NPY_PATH") ; 
    if( path )
    {
        std::cout << " save to NPY_PATH " << path << std::endl ; 
        a->save(path) ; 
    }
    return 0 ; 
}

