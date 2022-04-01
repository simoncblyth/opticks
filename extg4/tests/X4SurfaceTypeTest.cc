// name=X4SurfaceTypeTest ; gcc $name.cc -g -std=c++11 -lstdc++ -I.. -I/usr/local/opticks_externals/g4_1042/include/Geant4 -o /tmp/$name && /tmp/$name

#include <cassert>
#include <iostream>
#include <iomanip>
#include <vector>
#include <map>

#include "X4SurfaceType.hh"

int main()
{  
    typedef std::pair<const char*, G4SurfaceType> KV ; 
    std::vector<KV> kvs = { 
        {"dielectric_metal",      dielectric_metal },
        {"dielectric_dielectric", dielectric_dielectric }
      }; 

    for(unsigned i=0 ; i < kvs.size() ; i++)
    {
        const KV& kv = kvs[i] ; 
        const char* name = kv.first ; 
        G4SurfaceType type = kv.second ; 

        std::cout 
            << std::setw(3) << i 
            << " name " << std::setw(25) << name  
            << " type " << std::setw(5) << type 
            << std::endl 
            ;


        G4SurfaceType type1 = X4SurfaceType::Type(name); 
        assert( type == type1 ); 

        const char* name1 = X4SurfaceType::Name(type); 
        assert( strcmp(name, name1) == 0 ); 
    }

    return 0 ; 
}
