#include <iostream>
#include "G4ThreeVector.hh"

int main(int argc, char** argv)
{
    G4ThreeVector a(1., 2., 3.); 

    std::cout << " sizeof(a) " << sizeof(a) << std::endl ; 
    std::cout << " sizeof(double)*3 " << sizeof(double)*3  << std::endl; 

    const double* aa = (const double*)&a ; 
    for(int i=0 ; i < 3 ; i++ ) std::cout << aa[i] << std::endl ;   

    return 0 ; 
}
