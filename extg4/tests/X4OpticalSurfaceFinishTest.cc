// name=X4OpticalSurfaceFinishTest ; gcc $name.cc -g -std=c++11 -lstdc++ -I.. -I/usr/local/opticks_externals/g4_1042/include/Geant4 -o /tmp/$name && lldb__ /tmp/$name

#include <iostream>
#include <iomanip>
#include "X4OpticalSurfaceFinish.hh"

int main()
{  
    for(unsigned i=0 ; i < 10 ; i++)
    {
        const char* name = X4OpticalSurfaceFinish::Name(i); 
        int finish = name ? X4OpticalSurfaceFinish::Finish(name) : -1 ; 
        std::cout 
           << std::setw(3) << i 
           << " X4OpticalSurfaceFinish::Name " 
           << std::setw(30) << ( name ? name : "-" )
           << " X4OpticalSurfaceFinish::Finish " 
           << std::setw(4) << finish
           << std::endl 
           ;  
    }
    return 0 ; 
}
