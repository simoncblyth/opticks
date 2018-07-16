
#include "X4PhysicalVolume.hh"
#include "X4Sample.hh"

#include "OPTICKS_LOG.hh"
#include "Opticks.hh"

class GGeo ; 


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    char c = argc > 1 ? *argv[1] : 'o' ;  

    G4VPhysicalVolume* top = X4Sample::Sample(c) ; 

    GGeo* ggeo = X4PhysicalVolume::Convert(top) ;   
    assert(ggeo);  

    Opticks* ok = Opticks::GetInstance();
    ok->Summary();

    return 0 ; 
}


