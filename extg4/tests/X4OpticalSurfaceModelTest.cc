// name=X4OpticalSurfaceModelTest ; gcc $name.cc -std=c++11 -lstdc++ -I.. -I/usr/local/opticks_externals/g4_1042/include/Geant4 -o /tmp/$name && /tmp/$name

#include <iostream>
#include <iomanip>
#include "X4OpticalSurfaceModel.hh"

int main()
{  
    for(unsigned i=0 ; i < 6 ; i++)
    {
        const char* name = X4OpticalSurfaceModel::Name(i); 
        int finish = name ? X4OpticalSurfaceModel::Model(name) : -1 ; 

        std::cout 
           << std::setw(3) << i 
           << " X4OpticalSurfaceModel::Name " 
           << std::setw(30) << ( name ? name : "-" )
           << " X4OpticalSurfaceModel::Model " 
           << std::setw(4) << finish
           << std::endl 
           ;  
    }
    return 0 ; 
}
